import os
import shutil
import random
import pickle
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

########################################
# 1. Split the dataset into train/val/test
########################################

# Path to the original dataset folder (each subfolder = one identity)
original_data_dir = "VGG2_Dataset"  # e.g. "/data/VGGFace2"
# Create output directories for splits:
split_root = "/path/to/VGGFace2_split"  # New folder where train/val/test will reside
train_dir = os.path.join(split_root, "train")
val_dir   = os.path.join(split_root, "val")
test_dir  = os.path.join(split_root, "test")

# Create directories if not exist
for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

# Get list of identity folders from the original dataset directory
identities = [folder for folder in os.listdir(original_data_dir) 
              if os.path.isdir(os.path.join(original_data_dir, folder))]
print(f"Found {len(identities)} identities.")

# Split identities into train (70%), temp (30%) then split temp equally into val and test (15% each)
train_ids, temp_ids = train_test_split(identities, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

def move_identity_folders(ids_list, src_dir, dst_dir):
    for identity in ids_list:
        src = os.path.join(src_dir, identity)
        dst = os.path.join(dst_dir, identity)
        # Use copytree if you want to copy instead of move:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

# Copy folders to corresponding splits (you can use move if you prefer to remove from original)
move_identity_folders(train_ids, original_data_dir, train_dir)
move_identity_folders(val_ids, original_data_dir, val_dir)
move_identity_folders(test_ids, original_data_dir, test_dir)

########################################
# 2. Define Data Transforms & DataLoaders
########################################

IMG_HEIGHT = 224
IMG_WIDTH  = 224
BATCH_SIZE = 64

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])
    ]),
}

# Define the dataset directories for splits:
dataset_dirs = {
    'train': train_dir,
    'val': val_dir,
    'test': test_dir
}

image_datasets = {x: datasets.ImageFolder(dataset_dirs[x], data_transforms[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print("Number of classes:", num_classes)
print("Class names:", class_names)

########################################
# 3. Define the ArcFace Layer (ArcMarginProduct)
########################################

class ArcMarginProduct(nn.Module):
    """
    ArcMarginProduct layer: adds an additive angular margin to the target logit.
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # Normalize the input features and weight
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        # Compute phi = cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # Convert labels to one-hot
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # Combine: use phi for the true class, cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output

########################################
# 4. Define the Face Recognition Model using ResNet50 Backbone
########################################

class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size=512, num_classes=num_classes):
        super(FaceRecognitionModel, self).__init__()
        # Load a pretrained ResNet50 backbone
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        # Replace final fc layer with an embedding layer
        self.backbone.fc = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        # ArcFace classification layer
        self.arcface = ArcMarginProduct(embedding_size, num_classes, s=64.0, m=0.50, easy_margin=False)
    
    def forward(self, x, label):
        # Get embeddings from backbone
        x = self.backbone(x)  # shape: [batch, embedding_size]
        x = self.bn(x)
        # Pass embeddings and labels to ArcFace layer
        logits = self.arcface(x, label)
        return logits

model = FaceRecognitionModel(embedding_size=512, num_classes=num_classes)
model = model.to(device)
print(model)

########################################
# 5. Training Setup
########################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
# (Optional) Learning rate scheduler:
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=20):
    since = time.time()
    best_acc = 0.0
    best_model_wts = model.state_dict()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*20)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    # For ArcFace, model expects (input, label)
                    outputs = model(inputs, labels)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        # Optionally: gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # deep copy the model if validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # Save the best model weights
                torch.save(best_model_wts, "best_face_recognition_model.pth")
        
        print()
    
    time_elapsed = time.time() - since
    print(f"Training complete in {int(time_elapsed//60)}m {int(time_elapsed % 60)}s")
    print(f"Best Val Acc: {best_acc:.4f}")
    
    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model

# Train the model
num_epochs = 20
model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=num_epochs)

########################################
# 6. Evaluate on Train, Validation, and Test Sets
########################################

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # For inference, pass dummy labels since ArcFace layer is used during training.
            dummy_labels = labels  # We can use the actual labels here to compute accuracy.
            outputs = model(inputs, dummy_labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

train_acc = evaluate_model(model, dataloaders['train'])
val_acc   = evaluate_model(model, dataloaders['val'])
test_acc  = evaluate_model(model, dataloaders['test'])

print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Validation Accuracy: {val_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

########################################
# 7. Save Class Names Mapping
########################################

# Save the list of class names (order corresponds to ImageFolder)
with open("class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

########################################
# 8. Inference Example on a Single Image
########################################

from PIL import Image

def predict_image(model, img_path, transform, device):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)
    # Use a dummy label for ArcFace layer (its value doesn't affect inference when model is in eval mode)
    dummy_label = torch.zeros((img.size(0),), dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(img, dummy_label)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_idx = torch.topk(probs, 1)
    return top_idx.item(), top_prob.item()

# Define an inference transform (same as validation)
inference_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Example: Predict a test image from the test split
test_image_path = "VGG2_Dataset\n000006\0003_01.jpg"  # update this path
pred_idx, pred_prob = predict_image(model, test_image_path, inference_transform, device)
predicted_label = class_names[pred_idx]
print(f"Predicted: {predicted_label} with probability {pred_prob:.3f}")
