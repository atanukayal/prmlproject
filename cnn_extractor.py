import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def load_pretrained_model():
    """
    Load a pretrained ResNet-18 model and remove the classification head.
    
    Returns:
    - model: Modified ResNet-18 model without the classification layer.
    """
    model = models.resnet18(pretrained=True)  # Load ResNet-18
    model = nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    model.eval()  # Set model to evaluation mode
    return model

# Define preprocessing pipeline for images
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert OpenCV image (NumPy array) to PIL
    transforms.Resize((224, 224)),  # Resize to match CNN input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def extract_cnn_features(image, model, device='cpu'):
    """
    Extract deep features using a CNN model.

    Parameters:
    - image: Input image in BGR format (OpenCV format).
    - model: Pretrained CNN model.
    - device: 'cuda' or 'cpu' for computation.

    Returns:
    - feature_vector: Extracted features as a NumPy array.
    """
    if image is None:
        return None

    image = transform(image).unsqueeze(0)  # Apply transformations and add batch dimension
    image = image.to(device)

    with torch.no_grad():  # Disable gradient computation
        features = model(image)  # Extract features
        features = features.view(features.size(0), -1)  # Flatten to 1D vector

    return features.cpu().numpy().flatten()  # Convert to NumPy array
