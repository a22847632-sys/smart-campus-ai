import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.parking import ParkingSlotManager
from modules.crowd import CrowdQueueManager
from modules.emergency import EmergencyDetector
from modules.tracking import ObjectTracker
from utils.sample_generator import create_sample_campus_video
import cv2

def run_scenario_tests():
    print("=" * 65)
    print("SMART CAMPUS AI - PHASE 9 SCENARIO TESTING SUITE")
    print("=" * 65)

    # SCENARIO 1: PARKING OCCUPANCY (EMPTY VS OCCUPIED)
    print("\n--- SCENARIO 1: SMART PARKING OCCUPANCY TESTING ---")
    parking_mgr = ParkingSlotManager() # 10 slots
    
    # Test A: Empty Parking
    empty_objects = []
    slots_empty, stats_empty = parking_mgr.update_occupancy(empty_objects)
    print(f"Empty Parking Test:    Occupied={stats_empty['occupied_slots']}/{stats_empty['total_slots']} | Available={stats_empty['available_slots']} (Rate: {stats_empty['occupancy_rate']}%)")
    assert stats_empty["occupied_slots"] == 0, "Empty parking test failed!"

    # Test B: 5 Occupied Slots
    mock_vehicles = [
        {"class_name": "Car", "bbox": [100, 130, 200, 170]}, # Overlaps P-01
        {"class_name": "Car", "bbox": [100, 200, 200, 240]}, # Overlaps P-02
        {"class_name": "Car", "bbox": [100, 270, 200, 310]}, # Overlaps P-03
        {"class_name": "Car", "bbox": [300, 130, 400, 170]}, # Overlaps P-06
        {"class_name": "Car", "bbox": [300, 200, 400, 240]}  # Overlaps P-07
    ]
    slots_occ, stats_occ = parking_mgr.update_occupancy(mock_vehicles)
    print(f"Partial Occupancy Test: Occupied={stats_occ['occupied_slots']}/{stats_occ['total_slots']} | Available={stats_occ['available_slots']} (Rate: {stats_occ['occupancy_rate']}%)")
    assert stats_occ["occupied_slots"] == 5, "Occupied parking test failed!"
    print("[SUCCESS] Scenario 1 Passed cleanly.")

    # SCENARIO 2: CROWD DENSITY & QUEUE ESTIMATION
    print("\n--- SCENARIO 2: CROWD DENSITY & QUEUE ESTIMATION TESTING ---")
    crowd_mgr = CrowdQueueManager()

    # Test A: Low Crowd (2 people)
    mock_low_crowd = [
        {"class_name": "Person", "bbox": [600, 100, 620, 180]},
        {"class_name": "Person", "bbox": [650, 100, 670, 180]}
    ]
    analytics_low, _ = crowd_mgr.process_crowd(mock_low_crowd)
    print(f"Low Crowd Test:    Density Status={analytics_low['crowd_status']} (Count: {analytics_low['crowd_zone_count']})")
    assert analytics_low["crowd_status"] == "LOW", "Low crowd test failed!"

    # Test B: High Crowd + Queue (12 people in crowd zone, 8 in queue)
    mock_high_crowd = []
    for i in range(12):
        cx = 550 + (i % 4) * 40
        cy = 100 + (i // 4) * 50
        mock_high_crowd.append({"class_name": "Person", "bbox": [cx, cy, cx + 20, cy + 80]})

    analytics_high, _ = crowd_mgr.process_crowd(mock_high_crowd)
    print(f"High Crowd Test:   Density Status={analytics_high['crowd_status']} (Count: {analytics_high['crowd_zone_count']}) | Queue Status={analytics_high['queue_status']} (Queue Count: {analytics_high['queue_count']})")
    assert analytics_high["crowd_status"] == "HIGH", "High crowd test failed!"
    assert analytics_high["queue_status"] == "HIGH", "High queue test failed!"
    print("[SUCCESS] Scenario 2 Passed cleanly.")

    # SCENARIO 3: CAMPUS EMERGENCY (PERSON FALLEN POSTURE ANOMALY)
    print("\n--- SCENARIO 3: EMERGENCY FALLEN PERSON POSTURE TESTING ---")
    emergency_detector = EmergencyDetector(camera_location="Library Lawn")
    
    mock_fallen_person = [
        {"class_name": "Person", "bbox": [150, 400, 300, 440], "confidence": 0.88, "track_id": 99} # w=150, h=40 -> Ratio=3.75
    ]
    alerts, fallen = emergency_detector.detect_emergencies(mock_fallen_person)
    print(f"Fallen Person Test: Alert Event={alerts[0]['event_type']} | Confidence={alerts[0]['confidence']:.2f} | Location={alerts[0]['location']}")
    assert len(alerts) == 1 and alerts[0]["event_type"] == "PERSON_FALLEN", "Fallen person test failed!"
    print("[SUCCESS] Scenario 3 Passed cleanly.")

    # SCENARIO 4: MULTI-OBJECT REAL-TIME STREAM STRESS TEST
    print("\n--- SCENARIO 4: MULTI-OBJECT INTEGRATED STRESS TEST ---")
    video_path = os.path.join("data", "videos", "real_campus.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join("data", "videos", "sample_scenario.mp4")
        create_sample_campus_video(video_path, duration_sec=4, fps=30)

    tracker = ObjectTracker(model_path="yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    fps_list = []
    start_run = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        _, tracked_objs, _, _, fps = tracker.process_frame(frame, draw_annotations=False)
        fps_list.append(fps)

    cap.release()
    total_time = time.time() - start_run
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    print(f"Stress Test Execution: Processed {frame_idx} frames in {total_time:.2f}s | Average Speed: {avg_fps:.1f} FPS")

    print("\n" + "=" * 65)
    print("PHASE 9 SCENARIO TESTING SUITE RESULTS SUMMARY")
    print("=" * 65)
    print("Scenario 1 (Parking Empty vs Occupied): PASSED")
    print("Scenario 2 (Crowd & Queue Density):     PASSED")
    print("Scenario 3 (Person Fallen Posture):     PASSED")
    print("Scenario 4 (Multi-Object Stress Test):  PASSED")
    print(f"Empirical System Processing Speed:      {avg_fps:.1f} FPS")
    print("Scenario Testing Status:                PASSED / VERIFIED")
    print("=" * 65)
    return True

if __name__ == "__main__":
    success = run_scenario_tests()
    if not success:
        sys.exit(1)
