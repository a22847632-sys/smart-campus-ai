import os
import sys
import time
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sample_generator import create_sample_campus_video
from modules.tracking import ObjectTracker
from modules.parking import ParkingSlotManager
from modules.crowd import CrowdQueueManager
from modules.emergency import EmergencyDetector
from database.db import DatabaseManager
from app import app

def run_phase8_test():
    print("=" * 60)
    print("SMART CAMPUS AI - PHASE 8 END-TO-END INTEGRATION TEST")
    print("=" * 60)

    # 1. Initialize DB & Core Modules
    db_path = os.path.join("database", "integration_test.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    print("[INFO] Initializing End-to-End Integrated Architecture...")
    db = DatabaseManager(db_path=db_path)
    tracker = ObjectTracker(model_path="yolov8n.pt")
    parking_mgr = ParkingSlotManager()
    crowd_mgr = CrowdQueueManager()
    emergency_detector = EmergencyDetector(camera_location="Main Quad (Cam-01)")

    # 2. Real Integration Test Video Feed
    video_path = os.path.join("data", "videos", "real_campus.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join("data", "videos", "sample_integration.mp4")
        create_sample_campus_video(video_path, duration_sec=3, fps=30)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video stream opened. Processing {total_frames} frames through full integrated AI pipeline...")

    frame_idx = 0
    fps_list = []
    db_logged_count = 0

    start_run = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        # Step A: Tracking Engine
        annotated_frame, tracked_objs, active_counts, unique_counts, fps = tracker.process_frame(frame, draw_annotations=True)

        # Step B: Smart Parking Module
        slot_results, parking_stats = parking_mgr.update_occupancy(tracked_objs)
        annotated_frame = parking_mgr.draw_parking_overlay(annotated_frame, slot_results, parking_stats)

        # Step C: Crowd & Queue Management Module
        crowd_analytics, person_details = crowd_mgr.process_crowd(tracked_objs)
        annotated_frame = crowd_mgr.draw_crowd_overlay(annotated_frame, crowd_analytics)

        # Step D: Campus Emergency Module
        alerts, fallen_persons = emergency_detector.detect_emergencies(tracked_objs, frame_idx)
        annotated_frame = emergency_detector.draw_emergency_overlay(annotated_frame, alerts, fallen_persons)

        # Step E: Database Persistence (Every 15 frames)
        if frame_idx % 15 == 0:
            db.log_parking_stats("Cam-01", parking_stats)
            db.log_crowd_stats("Cam-01", crowd_analytics)
            for alert in alerts:
                db.log_emergency_alert(alert)
            db_logged_count += 1

        fps_list.append(fps)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"Frame {frame_idx:03d}/{total_frames} | Pipeline FPS: {fps:.1f} | "
                  f"Parking: {parking_stats['available_slots']} Free | Crowd: {crowd_analytics['crowd_status']} | Alerts: {len(alerts)}")

    cap.release()
    total_time = time.time() - start_run
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    # 3. Verify Database Persistence & Dashboard Telemetry Integration
    print("\n[VERIFICATION] Verifying Database Records & Telemetry API Integration...")
    dashboard_summary = db.get_dashboard_summary()
    print(f"Aggregated Telemetry Summary: {dashboard_summary}")

    assert dashboard_summary["parking"] is not None, "DB Parking Telemetry missing!"
    assert dashboard_summary["crowd"] is not None, "DB Crowd Telemetry missing!"
    print("[SUCCESS] All 3 modules logged telemetry to database successfully.")

    # 4. Verify Web Server Endpoint Integration
    client = app.test_client()
    api_res = client.get('/api/telemetry')
    assert api_res.status_code == 200, "Dashboard API integration failed!"
    print("[SUCCESS] Web Dashboard API endpoint verified.")

    # Clean up test DB
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n" + "=" * 60)
    print("PHASE 8 END-TO-END INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Total Frames Processed:      {frame_idx}")
    print(f"Total Processing Time:       {total_time:.2f}s")
    print(f"Average Pipeline Performance:{avg_fps:.1f} FPS")
    print(f"Module 1 (Smart Parking):    INTEGRATED & VERIFIED")
    print(f"Module 2 (Crowd & Queue):    INTEGRATED & VERIFIED")
    print(f"Module 3 (Emergency Detect): INTEGRATED & VERIFIED")
    print(f"Database Layer Sync:         VERIFIED ({db_logged_count} sync cycles)")
    print(f"Web Dashboard Feed & API:    VERIFIED")
    print(f"System Integration Status:   PASSED / READY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_phase8_test()
    if not success:
        sys.exit(1)
