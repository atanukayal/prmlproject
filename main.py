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
from torchvision import transforms
import cvzone

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-identification-5dd89-default-rtdb.firebaseio.com/",
    'storageBucket': "face-identification-5dd89.firebasestorage.app"
})
bucket = storage.bucket()

# Load models
pca = joblib.load("saved_models/pca_model.pkl")
scaler = joblib.load("saved_models/scalar.pkl")
svm_model = joblib.load("saved_models/svm_model.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")

# Load CNN model
device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
cnn_model = load_pretrained_model().to(device)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Setup video capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Load UI assets
imgBackground = cv2.imread('Resources/background.png')
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Load encoded file
print("Loading Encode File ...")
with open("EncodeFile.p", "rb") as file:
    encodeData = pickle.load(file)
encodeListKnown, studentIds = encodeData
print("Encode File Loaded")

# UI State Variables
modeType = 0
counter = 0
predictedId = -1
imgStudent = []

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to read from camera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            top, right, bottom, left = y, x + w, y + h, x
            faceROI = img[top:bottom, left:right]
            if faceROI.size == 0:
                continue
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

            bbox = (55 + left, 162 + top, right - left, bottom - top)
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

            if counter == 0:
                cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)
                counter = 1
                modeType = 1
        predictedId = predictedId.strip() 
        # Firebase fetch and update
        #Images\n000026.jpg
        if counter != 0:
            if counter == 1:
                studentInfo = db.reference(f'Students/{predictedId}').get()
                print("Student Info:", studentInfo)

                blob = bucket.get_blob(f'Images/{predictedId}.jpg')
                #blob = bucket.blob('Images/n000043.jpg')
                print(f"Trying to fetch image for ID: '{predictedId}'")

                if blob is not None:
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)
                else:
                    print(f"[WARNING] Image not found for ID: {predictedId}")
                    modeType = 3  # or any fallback mode you have
                    counter = 0
                    continue  # Skip the rest and go to the next frame

                
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)

                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print("Time elapsed (sec):", secondsElapsed)
                if secondsElapsed > -1:
                    ref = db.reference(f'Students/{predictedId}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:
                if 10 < counter < 20:
                    modeType = 2
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter <= 10:
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(predictedId), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414 - w) // 2
                    cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                    if imgStudent.shape[:2] != (216, 216):
                        imgStudent = cv2.resize(imgStudent, (216, 216))
                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent


                counter += 1
                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
