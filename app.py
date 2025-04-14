from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from face_identification import process_frame
from datetime import datetime, timedelta

app = Flask(__name__)

# Global list to store attendance events.
# Each record now stores 'predictedId', 'timestamp', and 'studentInfo'.
attendance_records = []

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process_image():
    # Check if the post request has the file.
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    # Convert file content to an OpenCV image.
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image file.'}), 400

    try:
        result = process_frame(img)
        
        # If a face is recognized and a predictedId is returned,
        # record this event along with studentInfo and current timestamp.
        if 'predictedId' in result and result['predictedId']:
            attendance_records.append({
                'predictedId': result['predictedId'],
                'timestamp': datetime.now(),
                'studentInfo': result.get('studentInfo', {})  # Expecting details like name
            })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Updated endpoint: Attendance Stats for the last 24 hours.
@app.route('/attendance-stats', methods=['GET'])
def attendance_stats():
    # Calculate the starting point (24 hours ago)
    now = datetime.now()
    start_time = now - timedelta(hours=24)
    
    # Filter attendance_records that fall within the last 24 hours.
    recent_records = [record for record in attendance_records if record['timestamp'] >= start_time]
    
    # Create a dictionary to hold unique student names.
    unique_students = {}
    for record in recent_records:
        student_info = record.get('studentInfo', {})
        # Use 'name' if available, otherwise fallback to 'predictedId'
        label = student_info.get('name', record['predictedId'])
        unique_students[label] = True

    names = list(unique_students.keys())
    total_present = len(names)
    
    stats = {
        'totalPresent': total_present,
        'names': names,
        'lastUpdated': now.isoformat()
    }
    return jsonify(stats)

if __name__ == '__main__':
    # For local testing, you can still use Flask’s debug server.
    app.run(host='0.0.0.0', port=8080, debug=True)
