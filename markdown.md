# Face Recognition System Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation and Setup](#installation-and-setup)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Extraction](#feature-extraction)
6. [Feature Processing](#feature-processing)
7. [Machine Learning Models](#machine-learning-models)
8. [Performance Comparison](#performance-comparison)
9. [Firebase Integration](#firebase-integration)
10. [Usage Examples](#usage-examples)
11. [Google Cloud Run Deployment](#-deploying-to-google-cloud-run-using-docker--artifact-registry)
12. [Usage Examples](#usage-examples)
13. [Future Enhancements](#future-enhancements)
14. [Contributing](#contributing)
15. [License](#license)
16. [Acknowledgments](#acknowledgments)

---

## Introduction

The Face Recognition System implements a face identification system for automated attendance tracking. It uses multiple feature extraction techniques (LBP, HOG, CNN) and machine learning models to accurately identify individuals from facial images.

Key Features:

- Multiple feature extraction methods
- Various machine learning models
- Real-time face recognition
- Firebase integration for attendance tracking
- Comprehensive documentation

---

## Project Structure

```
project/
├── requirements.txt        # Package dependencies
├── preprocessing.ipynb     # Data preprocessing and feature extraction
├── main.py                 # Main application script
├── AddDatatoDatabase.py    # Firebase database integration
├── cnn_extractor.py        # CNN feature extraction module
├── hog_extractor.py        # HOG feature extraction module
├── lbp_extractor.py        # LBP feature extraction module
Backend Integration
|──AddDatatoDatabase.py
├──face_identification.py
├──EncodeGenerator.py
├── main.py
script
├── saved_models/           # Trained models and encoders
│   ├── label_encoder.pkl   # Trained label encoder
│   ├── scalar.pkl          # Feature scaler
│   ├── pca_model.pkl       # Trained PCA model
│   ├── svm_model.pkl       # SVM classifier
│   └── ann_model.pth       # ANN model
├── VGG2_Dataset/           # Dataset directory
│   ├── n000001/            # Person 1 images
│   ├── n000002/            # Person 2 images
│   └── ...
├── Images/                 # Project images for documentation
├── Test-Images/            # Test images for evaluation
├── Dockerfile              # Docker configuration for deployment
├── EncodeFile.py
├── Dockerfile      # Script for deploying to Google Cloud Run
└── Resources/              # Additional resources
```

---

## Installation and Setup

### Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- Scikit-learn
- Firebase Admin SDK
- dlib
- requirements.txt

### Installation Steps

1. Clone the repository:

   bash
   git clone https://github.com/atanukayal/prmlproject.git

2. Install dependencies:

   bash
   pip install -r requirements.txt

3. Download the VGGFace2 dataset or use your custom dataset.

4. (Optional) For Firebase integration:
   - Set up a Firebase project
   - Download the serviceAccountKey.json
   - Place it in the project root directory

---

## Data Preprocessing

### Face Detection and Cropping

    def detect_and_crop_face(image,     cascade_path='haarcascade_frontalface_default.xml'):

    # Convert image to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, faces

    # Choose the largest detected face
    (x, y, w, h) = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    face_roi = image[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (128, 128))
    return face_resized, faces

### Dataset Loading

    def load_dataset_with_face_detection(dataset_path):
    cropped_images = []
    labels = []

    # Iterate over each person's folder
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if os.path.isdir(person_path):
            # Process only .jpg files
            for file in os.listdir(person_path):
                if file.lower().endswith('.jpg'):
                    img_path = os.path.join(person_path, file)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    # Apply face detection and crop the face
                    face, _ = detect_and_crop_face(image)
                    if face is not None:
                        cropped_images.append(face)
                        labels.append(person)

    return cropped_images, labels

---

## Feature Extraction

### 1. Histogram of Oriented Gradients (HOG)

    def extract_hog_features(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Extract HOG features
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

### 2. Local Binary Patterns (LBP)

    def extract_lbp_features(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Extract LBP features
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 59))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

### 3. Convolutional Neural Network (CNN) Features

    def extract_cnn_features(image, model, device):
    # Preprocess the image
    image_tensor = transform_image(image).to(device)

    # Extract features from the second-to-last layer
    with torch.no_grad():
        features = model(image_tensor)

    # Convert to numpy array
    features = features.cpu().numpy().flatten()
    return features

### Combined Feature Extraction

    def extract_all_features(images):
    all_features = []
    for img in images:
        # Extract features from each image
        hog_feat = extract_hog_features(img)
        lbp_feat = extract_lbp_features(img)
        cnn_feat = extract_cnn_features(img, model, device)

        # Concatenate all features into one vector
        combined = np.concatenate([lbp_feat, hog_feat, cnn_feat])
        all_features.append(combined)
    return np.array(all_features)

---

## Feature Processing

### Normalization

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

### Dimensionality Reduction with PCA

    from sklearn.decomposition import PCA

# Fit PCA with 90% variance retention

pca = PCA(n*components=0.90)
X_pca = pca.fit_transform(X_normalized)
print("Number of selected components:", pca.n_components*)

---

## Machine Learning Models

Machine learning models play a crucial role in the face recognition system. Each model has its strengths and weaknesses, and their performance depends on the features extracted and the dataset used. Below is a detailed explanation of the models used:

### 1. Support Vector Machine (SVM)

SVM is a supervised learning algorithm that works well for classification tasks. It uses a hyperplane to separate data points into different classes. In this project, the SVM with an RBF kernel achieved the highest accuracy (~77.68%) when using combined features (LBP, HOG, and CNN).

- _Advantages_: Effective in high-dimensional spaces, robust to overfitting.
- _Disadvantages_: Computationally expensive for large datasets.

### 2. K-Nearest Neighbors (KNN)

KNN is a simple, instance-based learning algorithm that classifies data points based on the majority class of their nearest neighbors. It achieved moderate accuracy (~57.56%) in this project.

- _Advantages_: Simple to implement, no training phase.
- _Disadvantages_: Sensitive to noisy data, computationally expensive during prediction.

### 3. Random Forest

Random Forest is an ensemble learning method that uses multiple decision trees to improve classification accuracy. It achieved an accuracy of ~27.38% in this project.

- _Advantages_: Handles large datasets well, reduces overfitting.
- _Disadvantages_: Less effective for high-dimensional feature spaces.

### 4. Artificial Neural Network (ANN)

ANN is a deep learning model inspired by the human brain. It consists of multiple layers of neurons that learn complex patterns in data. The ANN in this project achieved an accuracy of ~62%.

- _Advantages_: Capable of learning complex patterns, adaptable to various tasks.
- _Disadvantages_: Requires large datasets and computational resources.

---

## Workflow

The face recognition system follows a structured workflow to ensure accurate and efficient identification. Below is a step-by-step explanation of the workflow:

### 1. Data Preprocessing

- _Face Detection and Cropping_: Detect faces in images using Haar cascades and crop them to a fixed size (128x128 pixels).
- _Dataset Loading_: Load images from the dataset directory, apply face detection, and store cropped faces along with their labels.

### 2. Feature Extraction

- _HOG Features_: Extract edge and texture information using Histogram of Oriented Gradients.
- _LBP Features_: Capture local texture patterns using Local Binary Patterns.
- _CNN Features_: Use a pre-trained CNN model to extract high-level features from images.
- _Combined Features_: Concatenate HOG, LBP, and CNN features into a single feature vector for each image.

### 3. Feature Processing

- _Normalization_: Scale features to have zero mean and unit variance using StandardScaler.
- _Dimensionality Reduction_: Reduce feature dimensions using PCA while retaining 90% of the variance.

### 4. Model Training

- Train multiple machine learning models (SVM, KNN, Random Forest, ANN) using the processed features and their corresponding labels.
- Evaluate each model's performance using metrics like accuracy.

### 5. Model Evaluation

- Compare the performance of different models to identify the best-performing one (SVM in this case).
- Analyze feature importance to understand the contribution of each feature type.

### 6. Real-Time Face Recognition

- _Camera Integration_: Capture frames from a webcam.
- _Face Detection_: Detect faces in real-time and crop them.
- _Feature Extraction_: Extract features from the detected face using the trained feature extraction pipeline.
- _Prediction_: Use the trained SVM model to predict the identity of the person.
- _Display Results_: Draw bounding boxes around detected faces and display the predicted name on the screen.

This workflow ensures a systematic approach to face recognition, from data preparation to real-time application.

---

## Performance Comparison

The SVM with RBF kernel achieved the highest accuracy of ~77.68% when using a combination of LBP, HOG, and CNN features.

| Model               | Accuracy |
| ------------------- | -------- |
| Random Forest       | 27.38%   |
| KNN                 | 57.56%   |
| SVM                 | 77.68%   |
| Logistic Regression | 65.00%   |
| XGBoost             | 58.00%   |
| ANN                 | 62.00%   |

### Feature Importance Analysis

- CNN features contributed most to model accuracy
- The combination of HOG+CNN+LBP features provided the best performance (75.8%)
- Face landmarks alone performed poorly (12.12%)

---

## Firebase Integration

The AddDatatoDatabase.py script handles the integration with Firebase Realtime Database for attendance tracking.

python

# Example usage

import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase app

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
'databaseURL': 'https://your-firebase-project.firebaseio.com'
})

# Add student data

def add_student_data(student_id, name, major, year):
ref = db.reference('Students')
ref.child(student_id).set({
'name': name,
'major': major,
'year': year,
'attendance': False, # Add more fields as needed
})

---

## Usage Examples

### Face Detection and Feature Extraction

python

# Load image

image = cv2.imread("path/to/image.jpg")

# Detect and crop face

face, \_ = detect_and_crop_face(image)

# Extract features

if face is not None:
hog_features = extract_hog_features(face)
lbp_features = extract_lbp_features(face)
cnn_features = extract_cnn_features(face, model, device)

    # Combine features
    combined = np.concatenate([lbp_features, hog_features, cnn_features])

### Using Trained Models for Prediction

python

# Load saved models

scaler = joblib.load("saved_models/scalar.pkl")
pca = joblib.load("saved_models/pca_model.pkl")
svm = joblib.load("saved_models/svm_model.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")

# Preprocess input features

scaled_features = scaler.transform(combined_features.reshape(1, -1))
pca_features = pca.transform(scaled_features)

# Predict using SVM

predicted_label = svm.predict(pca_features)[0]
person_name = label_encoder.inverse_transform([predicted_label])[0]

### Real-time Face Recognition

# Start camera

cap = cv2.VideoCapture(0)

while True:
ret, frame = cap.read()
if not ret:
break

    # Detect faces
    face, faces = detect_and_crop_face(frame)

    if face is not None:
        # Process and predict
        # (feature extraction, preprocessing, model prediction)

        # Draw rectangle and name
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

---

---

## Deploying to Google Cloud Run using Docker & Artifact Registry

This guide walks you through the steps to containerize your application using Docker, push it to Artifact Registry, and deploy it to _Cloud Run_ on Google Cloud Platform (GCP).

---

### 📦 Prerequisites

- Google Cloud CLI installed: [Install gcloud](https://cloud.google.com/sdk/docs/install)
- Docker installed: [Install Docker](https://docs.docker.com/get-docker/)
- A GCP project created
- Billing enabled
- Artifact Registry API enabled:
  bash
  gcloud services enable artifactregistry.googleapis.com
  ⚙ Step 1: Configure Your GCP Project
  bash

```
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region asia-south2
Replace YOUR_PROJECT_ID with your actual project ID.
```

🗂 Step 2: Create an Artifact Registry Repository
bash

```
gcloud artifacts repositories create face-id-app \
  --repository-format=docker \
  --location=asia-south2 \
  --description="Docker repo for Face ID app"
```

🐳 Step 3: Build Your Docker Image
From the root directory of your project (where your Dockerfile is located):

bash

```
docker build -t asia-south2-docker.pkg.dev/YOUR_PROJECT_ID/face-id-app/face-identification-app .
```

🔐 Step 4: Authenticate Docker to Push to Artifact Registry
bash

```
gcloud auth configure-docker asia-south2-docker.pkg.dev
```

⬆ Step 5: Push Docker Image to Artifact Registry
bash

```
docker push asia-south2-docker.pkg.dev/YOUR_PROJECT_ID/face-id-app/face-identification-app
```

☁ Step 6: Deploy to Cloud Run
bash

```
gcloud run deploy face-id-app \
  --image=asia-south2-docker.pkg.dev/YOUR_PROJECT_ID/face-id-app/face-identification-app \
  --platform=managed \
  --memory=1024Mi \
  --port=8080 \
  --allow-unauthenticated
```

📝 Make sure your app is configured to listen on 0.0.0.0:8080 (the default port for Cloud Run).

🌐 Access Your App
After deployment, GCP will provide you with a public URL. You can access your app at:

bash

```
https://face-id-app-<hash>-<region>.run.app
```

🔁 Updating Your App
To update your app:

Rebuild the Docker image with changes

Push it again:

bash

```
docker push asia-south2-docker.pkg.dev/YOUR_PROJECT_ID/face-id-app/face-identification-app
Redeploy via gcloud run deploy command.
```

🧹 Optional: Delete Your Artifact Registry Repo (if needed)
bash

```
gcloud artifacts repositories delete face-id-app --location=asia-south2
```

---

## Future Enhancements

1. **Real-Time Face Recognition**: Improve the webcam integration for more efficient real-time processing.

2. **Mobile App Integration**: Develop a mobile application that communicates with the Firebase database.

3. **Advanced Models**: Implement state-of-the-art deep learning models like:

   - ResNet
   - EfficientNet
   - FaceNet
   - ArcFace

4. **Advanced Face Detection**: Implement more robust face detection algorithms like MTCNN or RetinaFace.

5. **Data Augmentation**: Enhance model robustness through data augmentation techniques.

6. **Multi-face Recognition**: Support simultaneous identification of multiple faces in a single frame.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch:
   bash
   git checkout -b feature-name
3. Commit your changes:
   bash
   git commit -m "Add feature-name"
4. Push to the branch:
   bash
   git push origin feature-name

5. Open a pull request.

---

## License

This project is under CSL2050 course.

---

## Acknowledgments

- _Dataset_: VGGFace2 Dataset.
- _Libraries_: Scikit-learn, TensorFlow, PyTorch, OpenCV, Firebase Admin SDK.
