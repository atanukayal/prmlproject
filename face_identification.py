# face_identification.py

import os
import pickle
import numpy as np
import cv2
import joblib
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime
from hog_extractor import extract_hog_features
from lbp_extractor import extract_lbp_features
from cnn_extractor import load_pretrained_model, extract_cnn_features
import base64

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://face-identification-5dd89-default-rtdb.firebaseio.com/",
        'storageBucket': "face-identification-5dd89.firebasestorage.app"
    })
bucket = storage.bucket()

# Load models and other resources (update paths to use forward slashes for cross‑platform compatibility)
pca = joblib.load("saved_models/pca_model.pkl")
scaler = joblib.load("saved_models/scalar.pkl")
svm_model = joblib.load("saved_models/svm_model.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")

device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
cnn_model = load_pretrained_model().to(device)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load encoded file (if needed)
with open("EncodeFile.p", "rb") as file:
    encodeData = pickle.load(file)
encodeListKnown, studentIds = encodeData

def process_frame(img):
    """
    Process an input image to detect a face, recognize it, and update attendance.
    Returns a dictionary with the result.
    """
    result_dict = {}

    # Convert the image to grayscale and detect faces.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        result_dict["message"] = "Face not detected"
        return result_dict

    # Process the first detected face.
    (x, y, w, h) = faces[0]
    faceROI = img[y:y+h, x:x+w]
    if faceROI.size == 0:
        result_dict["message"] = "Face region extraction failed"
        return result_dict

    faceROI_resized = cv2.resize(faceROI, (128, 128))
    faceROI_rgb = cv2.cvtColor(faceROI_resized, cv2.COLOR_BGR2RGB)

    # Extract features
    hog_feat = extract_hog_features(faceROI_rgb).flatten()
    lbp_feat = extract_lbp_features(faceROI_rgb).flatten()
    cnn_feat = extract_cnn_features(faceROI_rgb, cnn_model, device).flatten()

    combinedFeatures = np.concatenate([hog_feat, lbp_feat, cnn_feat])
    scaled_features = scaler.transform(combinedFeatures.reshape(1, -1))
    pca_features = pca.transform(scaled_features)
    
    predicted_label = svm_model.predict(pca_features)[0]
    predictedId = label_encoder.inverse_transform([predicted_label])[0]
    result_dict["predictedId"] = predictedId

    # Fetch student info from Firebase
    studentInfo = db.reference(f'Students/{predictedId}').get()
    if not studentInfo:
        result_dict["message"] = "Face not found in database"
        return result_dict
    result_dict["studentInfo"] = studentInfo

    # Fetch student image from Firebase Storage
    blob = bucket.get_blob(f'Images/{predictedId}.jpg')
    if blob is not None:
        array = np.frombuffer(blob.download_as_string(), np.uint8)
        imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)
        # Encode image to base64 for web display.
        retval, buffer = cv2.imencode('.jpg', imgStudent)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        result_dict["imgStudent"] = jpg_as_text
    else:
        result_dict["imgStudent"] = None

    # Update attendance if the last attendance time was more than 100 sec ago.
    try:
        datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
    except Exception:
        datetimeObject = datetime.now()

    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
    if secondsElapsed > 100:
        ref = db.reference(f'Students/{predictedId}')
        studentInfo['total_attendance'] += 1
        ref.child('total_attendance').set(studentInfo['total_attendance'])
        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        result_dict["attendance_update"] = "Attendance updated"
    else:
        result_dict["attendance_update"] = "Attendance already marked recently"

    result_dict["message"] = "Face recognized successfully"
    return result_dict
