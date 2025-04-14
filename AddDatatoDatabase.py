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
        "name": "n000002",
        "major": "CSE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A+",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000003": {
        "name": "n000003",
        "major": "AI&DS",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A+",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000004": {
        "name": "n000004",
        "major": "CSE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "B",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000005": {
        "name": "n000005",
        "major": "AI&DS",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A-",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000006": {
        "name": "n000006",
        "major": "EE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "B",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000007": {
        "name": "n000007",
        "major": "EE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000008": {
        "name": "n000008",
        "major": "AI&DS",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A+",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000010": {
        "name": "n000010",
        "major": "EE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "B-",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000011": {
        "name": "n000011",
        "major": "AI&DS",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A+",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000012": {
        "name": "n000012",
        "major": "CSE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "C",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000017": {
        "name": "n000017",
        "major": "AI&DS",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A",
        "year": 4,
        "last_attendance_time": "2025-04-08 00:54:34"
    },
    "n000022": {
        "name": "n000022",
        "major": "CSE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "B",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000023": {
        "name": "n000023",
        "major": "EE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000026": {
        "name": "n000026",
        "major": "AI&DS",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000035": {
        "name": "n000035",
        "major": "CSE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "B",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000043": {
        "name": "n000043",
        "major": "EE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "C",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000045": {
        "name": "n000045",
        "major": "AI&DS",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A-",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000050": {
        "name": "n000050",
        "major": "CSE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "A+",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    },
    "n000053": {
        "name": "n000053",
        "major": "CSE",
        "starting_year": 2023,
        "total_attendance": 0,
        "standing": "C",
        "year": 1,
        "last_attendance_time": "2022-12-11 00:54:34"
    }
}

for key, value in data.items():
    ref.child(key).set(value)
    