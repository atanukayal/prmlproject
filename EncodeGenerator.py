import os
import cv2
import numpy as np
import pandas as pd
import datetime
from skimage.feature import local_binary_pattern, hog
from skimage import color
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from hog_extractor import extract_hog_features
from lbp_extractor import extract_lbp_features
from cnn_extractor import load_pretrained_model, extract_cnn_features
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

# Set device and load your CNN model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_pretrained_model().to(device)  # Load model to device

# Initialize Firebase with proper configuration.
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-identification-5dd89-default-rtdb.firebaseio.com/",
    'storageBucket': "face-identification-5dd89.firebasestorage.app"  # Correct bucket name format
})

# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
bucket = storage.bucket()  # Get bucket instance

for path in pathList:
    img = cv2.imread(os.path.join(folderPath, path))
    if img is None:
        print(f"Warning: Could not load {path}")
        continue
    imgList.append(img)
    studentIds.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)
    # Optionally print or log the upload
    print(f"Uploaded {fileName} to Firebase Storage.")

print(studentIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        # Convert image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Extract features using your custom extractors
        hog_feat = extract_hog_features(img_rgb)
        lbp_feat = extract_lbp_features(img_rgb)
        cnn_feat = extract_cnn_features(img_rgb, model, device)
        # Concatenate features (assuming they are 1D arrays or flattened accordingly)
        combined = np.concatenate([cnn_feat, hog_feat, lbp_feat])
        encodeList.append(combined)
    return encodeList

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

with open("EncodeFile.p", 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)
print("File Saved")
