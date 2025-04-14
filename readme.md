# Face Identification Project (Attendance System)

Identify a face image by classifying to one of K classes. Extracting the features using the LBP, HoG, and CNN Features. This repo is a solution to the project problem given in the CSL2050 course.

---

## Demo

Below are example screenshots showing how the project works:

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/atanukayal/prmlproject/blob/main/Test-Images/Interface_UI_1%20(1).png">
        <img src="https://github.com/atanukayal/prmlproject/raw/main/Test-Images/Interface_UI_1%20(1).png" alt="Interface UI 1" height="300" width="500"/>
      </a>
    </td>
    <td align="center" style="vertical-align: middle; font-size: 32px;">
      ➡
    </td>
    <td align="center">
      <a href="https://github.com/atanukayal/prmlproject/blob/main/Test-Images/Interface_UI_1%20(2).png">
        <img src="https://github.com/atanukayal/prmlproject/raw/main/Test-Images/Interface_UI_1%20(2).png" alt="Interface UI 2" height="300" width="500"/>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Before</b></td>
    <td></td>
    <td align="center"><b>After</b></td>
  </tr>
</table>

### Spotlight Video-

```
https://youtu.be/Iq36qJS63Eg
```

`And the detailed documentation is in markdown.md `

---

## Features

- _Face Detection_: Detects faces in images using Haar cascades.
- _Feature Extraction_: Extracts robust features using HOG, LBP, and CNN (VGG16).
- _Dimensionality Reduction_: Reduces feature dimensions using PCA while retaining 90% variance.
- _Classification_: Classifies faces into predefined classes using SVM and ANN models.
- _Database Integration_: Integrates with Firebase Realtime Database for attendance management.
- _Scalability_: Supports adding new students and updating the database dynamically.

---

## Dataset Details

The dataset used in this project is the _VGG2_Dataset_.

### Download Instructions

python
import kagglehub

path = kagglehub.dataset_download("hearfool/vggface2")

print("Path to dataset files:", path)

---

## Installation

1. Clone this repository:
   bash
   git clone https://github.com/atanukayal/prmlproject.git
2. Install all the dependencies:
   bash
   pip install -r requirements.txt
3. Download the custom-made dataset:
   bash
   link -
4. Extract and paste it into the project root directory.

5. Preprocessing Steps:
   - _Face Detection and Cropping_:
     - Uses Haar cascades (haarcascade_frontalface_default.xml) to detect faces and crop them to a fixed size (128x128 pixels).
   - _Feature Extraction_:
     - Extracts features using:
       - HOG (Histogram of Oriented Gradients).
       - LBP (Local Binary Patterns).
       - CNN (Convolutional Neural Network) features from a pre-trained VGG16 model.
   - _Feature Scaling_:
     - Standardizes features using StandardScaler.
   - _Dimensionality Reduction_:
     - Applies PCA to reduce feature dimensions while retaining 90% variance.
   - _Dataset Splitting_:
     - Splits the dataset into training and testing sets using train_test_split.
   - _Label Encoding_:
     - Encodes string labels into numerical values using LabelEncoder.
   - _Saving Preprocessed Data_:
     - Saves features, labels, and models (e.g., PCA, scaler) using pickle or joblib.

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

## Steps to Test it Out

1. Extract face in the image using extract_face.py:
   - Edit test_data_path = "to_your_test_path".
2. Run the preprocessing.ipynb notebook for feature extraction (HOG, CNN, etc.).
3. Test your image:
   bash
   python main.py

---

## AddDatatoDatabase.py

This script is responsible for adding student data to the Firebase Realtime Database. It initializes the Firebase app using the serviceAccountKey.json file and uploads predefined student records to the Students reference in the database.

### Key Features:

- Initializes Firebase using the service account key.
- Adds student data, including name, major, year, attendance, and other details, to the database.
- Ensures data is structured under unique keys (e.g., n000002, n000016).

### Usage:

1. Ensure the serviceAccountKey.json file is present in the project directory.
2. Update the data dictionary in the script with the desired student records.
3. Run the script:
   bash
   python AddDatatoDatabase.py
4. Verify that the data has been added to the Firebase Realtime Database.

### Dependencies:

- firebase-admin: Ensure this library is installed by running:
  bash
  pip install firebase-admin

### Notes:

- The databaseURL in the script should match your Firebase project's Realtime Database URL.
- Modify the data dictionary as needed to include additional student records.
- The usage of docker and deployment is written in detail on markdown.md

---

## Performance Metrics

- _Accuracy_: Achieved high accuracy on the test dataset using SVM and ANN models.
- _Precision & Recall_: Evaluated using confusion matrices for each class.
- _Scalability_: Supports large datasets with efficient preprocessing and feature extraction.

---

## Future Enhancements

- _Real-Time Face Recognition_: Integrate webcam support for real-time attendance tracking.
- _Mobile App Integration_: Develop a mobile app to interact with the Firebase database.
- _Advanced Models_: Experiment with state-of-the-art deep learning models like ResNet or EfficientNet.
- _Cloud Deployment_: Deploy the system on cloud platforms like AWS or Google Cloud for scalability.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
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
- _Libraries_: Scikit-learn, TensorFlow, OpenCV, Firebase Admin SDK.
