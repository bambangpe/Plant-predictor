#pip install flask ultralytics opencv-python numpy
#---Diagram
#-app.py
#-templates/index.html
#-static/uploads/   (optional for uploaded videos)

from flask import Flask, render_template, Response, request, redirect, url_for
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv8 model (use yolov8n or yolov8s for speed, yolov8m/x for accuracy)
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt

# COCO class ID for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Speed estimation parameters (calibrate these for your camera!)
PIXELS_PER_METER = 20  # Adjust based on your camera height & angle
FPS = 30               # Assume 30 FPS (will be measured dynamically)

# Store previous positions for speed calculation
prev_positions = {}
prev_time = time.time()

def estimate_speed(prev_pos, curr_pos, delta_time):
    if prev_pos is None or curr_pos is None:
        return 0
    pixel_distance = np.linalg.norm(np.array(prev_pos) - np.array(curr_pos))
    meters = pixel_distance / PIXELS_PER_METER
    speed_mps = meters / delta_time
    speed_kph = speed_mps * 3.6
    return round(speed_kph, 1)

def generate_frames(source=0):
    global prev_positions, prev_time

    cap = cv2.VideoCapture(source)
    
    # Get actual FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        # YOLOv8 inference
        results = model(frame, stream=True, classes=VEHICLE_CLASSES, conf=0.5)
        
        current_positions = {}

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = box.conf[0]

                # Center point of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)

                # Track by center point (simple tracking - can be improved with SORT/DeepSORT)
                track_id = None
                min_dist = float('inf')
                for pid, pos in prev_positions.items():
                    dist = np.linalg.norm(np.array(pos) - np.array(center))
                    if dist < 50 and dist < min_dist:  # 50px threshold
                        min_dist = dist
                        track_id = pid

                if track_id is None:
                    track_id = len(prev_positions)

                # Estimate speed
                prev_pos = prev_positions.get(track_id, None)
                speed = estimate_speed(prev_pos, center, delta_time if delta_time > 0 else 0.033)

                current_positions[track_id] = center

                # Draw bounding box and info
                label = f"{model.names[cls]} {speed} km/h"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        prev_positions = current_positions

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    source = request.args.get('source', default=0, type=int)
    return Response(generate_frames(source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('video_from_file', filename=file.filename))
    return redirect('/')

@app.route('/video_file')
def video_from_file():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

