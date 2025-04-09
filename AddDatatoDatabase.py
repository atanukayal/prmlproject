import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Load the service account key and initialize Firebase with configuration once.
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-identification-5dd89-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "n000002": {
        "name": "Sehar_Dev",
        "major": "CSE",
        "starting_year": 2023,
        "total_attendance": 7,
        "standing": "A+",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000016": {
        "name": "Japneet Singh",
        "major": "Economics",
        "starting_year": 2021,
        "total_attendance": 12,
        "standing": "B",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000026": {
        "name": "Elon Musk",
        "major": "Physics",
        "starting_year": 2020,
        "total_attendance": 7,
        "standing": "G",
        "year": 2,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000012": {
        "name": "Vishal Dhaniya",
        "major": "AI&DS",
        "starting_year": 2023,
        "total_attendance": 10,
        "standing": "A+",
        "year": 4,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000043": {
        "name": "Sydney Sweeney",
        "major": "Economics",
        "starting_year": 2021,
        "total_attendance": 12,
        "standing": "B",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    }
}

for key, value in data.items():
    ref.child(key).set(value)
