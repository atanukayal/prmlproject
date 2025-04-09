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
from time import sleep

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-identification-5dd89-default-rtdb.firebaseio.com/",
    'storageBucket': "face-identification-5dd89.firebasestorage.app"
})
bucket = storage.bucket()

# Load models
pca = joblib.load("saved_models\\pca_model.pkl")
scaler = joblib.load("saved_models\\scalar.pkl")
svm_model = joblib.load("saved_models\\svm_model.pkl")
label_encoder = joblib.load("saved_models\\label_encoder.pkl")

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
modeType = 0           # 0: default live feed; 1: processing/loading; 2: showing attendance; 3: fallback (if image missing)
capture_mode = False   # Flag to trigger capture processing
attendance_marked = False  # True when attendance has been marked and is being displayed
attendance_display_start = None  # Time when attendance was marked

studentInfo = None     # Student info dictionary
predictedId = None
imgStudent = None

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to read from camera.")
        break

    # Always update the background with the live video feed and default UI mode
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    # During attendance display mode, check if 10 seconds have passed; if so, reset the state
    if attendance_marked:
        elapsed = (datetime.now() - attendance_display_start).total_seconds()
        if elapsed >= 10:
            # Reset UI to allow capture again
            attendance_marked = False
            modeType = 0
            studentInfo = None
            imgStudent = None

    # If we are NOT currently displaying attendance info (locked state), allow capture
    if not attendance_marked:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            capture_mode = True

    # Only process the frame for capture when capture_mode is True and not already in attendance display
    if capture_mode and not attendance_marked:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Process only if at least one face is detected
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

                # Draw a bounding box on the UI for visualization
                bbox = (55 + left, 162 + top, right - left, bottom - top)
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                # Display loading feedback before processing Firebase operations
                cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                #cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)

                # Fetch student info from Firebase and update attendance
                studentInfo = db.reference(f'Students/{predictedId}').get()
                print("Student Info:", studentInfo)

                blob = bucket.get_blob(f'Images/{predictedId}.jpg')
                print(f"Trying to fetch image for ID: '{predictedId}'")
                if blob is not None:
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)
                else:
                    print(f"[WARNING] Image not found for ID: {predictedId}")
                    modeType = 1  # Fallback mode if image not found
                    capture_mode = False
                    break  # Skip further processing

                # Update attendance if valid
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print("Time elapsed (sec):", secondsElapsed)
                if secondsElapsed > 100:
                    ref = db.reference(f'Students/{predictedId}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    capture_mode = False
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                    break

                # Update the UI with student information (attendance marked)
                modeType = 2  # Change to attendance display mode
                if imgStudent is not None:
                    if imgStudent.shape[:2] != (216, 216):
                        imgStudent = cv2.resize(imgStudent, (216, 216))
                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
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
                
                cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)  # Ensure it renders

# 2. Hold the display for 3 seconds
                sleep(3)
                # Set the attendance display flag and record the start time
                attendance_marked = True
                attendance_display_start = datetime.now()
                  # Set mode to attendance display
                sleep(5)
                # Reset capture trigger so new capture will wait until the attendance display is over
                capture_mode = False
                break  # Process only one face per capture

        else:
            # Inform the user if no face was detected and reset the capture flag
            cvzone.putTextRect(imgBackground, "No face detected", (275, 400), scale=1)
            capture_mode = False

    # Display the updated UI window
    cv2.imshow("Face Attendance", imgBackground)
    # Check for exit condition on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
