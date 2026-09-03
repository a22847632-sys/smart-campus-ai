import os
import sys
import time
import cv2
from flask import Flask, render_template, Response, jsonify, request

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import BASE_DIR
from modules.tracking import ObjectTracker
from modules.parking import ParkingSlotManager
from modules.crowd import CrowdQueueManager
from modules.emergency import EmergencyDetector
from database.db import DatabaseManager
from utils.sample_generator import create_sample_campus_video

# Initialize Flask App with static/template paths
app = Flask(__name__,
            template_folder=os.path.join("dashboard", "templates"),
            static_folder=os.path.join("dashboard", "static"))

# Global AI Engine & DB instances
db_manager = DatabaseManager()
object_tracker = None
parking_manager = None
crowd_manager = None
emergency_detector = None

def init_ai_pipeline():
    global object_tracker, parking_manager, crowd_manager, emergency_detector
    print("[SYSTEM] Initializing Smart Campus AI Modules...")
    object_tracker = ObjectTracker(model_path="yolov8n.pt")
    parking_manager = ParkingSlotManager()
    crowd_manager = CrowdQueueManager()
    emergency_detector = EmergencyDetector(camera_location="Main Campus Entrance & Courtyard")
    print("[SYSTEM] All AI Modules Loaded Successfully!")

# Initialize AI Pipeline
init_ai_pipeline()

def generate_video_stream(source=None):
    """
    Video Streaming Generator function for Flask MJPEG endpoint.
    Executes full pipeline: Detection -> Tracking -> Parking -> Crowd -> Emergency -> DB -> Stream.
    Supports video file paths, environment variable VIDEO_SOURCE, and webcam index (0).
    """
    if source is None or source == "":
        source = os.environ.get("VIDEO_SOURCE", os.path.join("data", "videos", "real_campus.mp4"))
        if not os.path.exists(source) and not (isinstance(source, str) and str(source).isdigit()):
            source = os.path.join("data", "videos", "sample_campus.mp4")
            if not os.path.exists(source):
                create_sample_campus_video(source, duration_sec=10, fps=30)

    # Parse numeric string as integer for OpenCV webcam device index
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source: {source}")
        return

    frame_count = 0
    print(f"[STREAM] Started video processing loop for source: {source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video file feeds continuously for live demonstration
            if isinstance(source, str) and os.path.exists(source):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print(f"[STREAM] Stream ended or frame read failed for source: {source}")
                break

        frame_count += 1

        # 1. Tracking Engine
        annotated_frame, tracked_objs, active_counts, unique_counts, fps = object_tracker.process_frame(frame, draw_annotations=True)

        # 2. Smart Parking Analysis
        slot_results, parking_stats = parking_manager.update_occupancy(tracked_objs)
        annotated_frame = parking_manager.draw_parking_overlay(annotated_frame, slot_results, parking_stats)

        # 3. Crowd & Queue Analysis
        crowd_analytics, person_details = crowd_manager.process_crowd(tracked_objs)
        annotated_frame = crowd_manager.draw_crowd_overlay(annotated_frame, crowd_analytics)

        # 4. Emergency Event Detection
        alerts, fallen_persons = emergency_detector.detect_emergencies(tracked_objs, frame_count)
        annotated_frame = emergency_detector.draw_emergency_overlay(annotated_frame, alerts, fallen_persons)

        # 5. Database Logging (Every 15 frames / ~0.5 sec)
        if frame_count % 15 == 0:
            db_manager.log_parking_stats("Cam-01", parking_stats)
            db_manager.log_crowd_stats("Cam-01", crowd_analytics)
            for alert in alerts:
                db_manager.log_emergency_alert(alert)

        # Encode Frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Micro sleep to regulate stream rate (~25-30 FPS)
        time.sleep(0.03)

    cap.release()

@app.route('/')
def index():
    """Renders main dashboard interface."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """MJPEG Video Stream route supporting optional source query parameter."""
    source_param = request.args.get('source', None)
    return Response(generate_video_stream(source=source_param), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def api_telemetry():
    """API Endpoint returning aggregated campus metrics."""
    summary = db_manager.get_dashboard_summary()
    return jsonify(summary)

@app.route('/api/alerts')
def api_alerts():
    """API Endpoint returning active emergency alerts."""
    alerts = db_manager.get_active_alerts()
    return jsonify(alerts)

@app.route('/api/parking')
def api_parking():
    """API Endpoint returning latest parking telemetry."""
    parking = db_manager.get_latest_parking_stats()
    return jsonify(parking or {})

@app.route('/api/crowd')
def api_crowd():
    """API Endpoint returning latest crowd telemetry."""
    crowd = db_manager.get_latest_crowd_stats()
    return jsonify(crowd or {})

if __name__ == "__main__":
    print("=" * 60)
    print("AI-POWERED SMART CAMPUS MANAGEMENT SYSTEM SERVER")
    print("Running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
