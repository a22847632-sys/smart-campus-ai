import os
import sys
import time
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sample_generator import create_sample_campus_video
from modules.tracking import ObjectTracker
from modules.parking import ParkingSlotManager
import cv2

def run_phase3_test():
    print("=" * 60)
    print("SMART CAMPUS AI - PHASE 3 SMART PARKING MODULE TEST")
    print("=" * 60)

    # Step 1: Use real parking video feed
    video_path = os.path.join("data", "videos", "real_parking.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join("data", "videos", "sample_parking.mp4")
        create_sample_campus_video(video_path, duration_sec=4, fps=30)

    # Step 2: Initialize tracker & parking manager
    print("[INFO] Initializing ObjectTracker & ParkingSlotManager...")
    tracker = ObjectTracker(model_path="yolov8n.pt")
    parking_mgr = ParkingSlotManager() # Uses default 10-slot layout

    # Test saving & loading configuration file
    config_path = os.path.join("data", "parking_config.json")
    parking_mgr.save_config(config_path)
    if not os.path.exists(config_path):
        print(f"[ERROR] Failed to save config to {config_path}")
        return False
    print(f"[INFO] Verified JSON configuration save -> {config_path}")

    # Re-load from JSON config file
    parking_mgr = ParkingSlotManager(config_file_path=config_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video opened. Processing {total_frames} frames for Smart Parking detection...")

    frame_idx = 0
    fps_list = []
    last_stats = None

    start_run = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Track objects
        annotated_frame, tracked_objs, active_counts, unique_counts, fps = tracker.process_frame(frame, draw_annotations=False)
        
        # Process Parking occupancy
        slot_results, stats = parking_mgr.update_occupancy(tracked_objs)
        annotated_parking = parking_mgr.draw_parking_overlay(annotated_frame, slot_results, stats)
        
        fps_list.append(fps)
        last_stats = stats

        if frame_idx % 20 == 0 or frame_idx == total_frames:
            occupied_slots_str = ", ".join([s["slot_id"] for s in slot_results if s["occupied"]])
            if not occupied_slots_str:
                occupied_slots_str = "None (All vacant)"
            print(f"Frame {frame_idx:03d}/{total_frames} | Total: {stats['total_slots']} | Occupied: {stats['occupied_slots']} | Available: {stats['available_slots']} | Occupied Slots: [{occupied_slots_str}]")

    cap.release()
    total_time = time.time() - start_run
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    print("\n" + "=" * 60)
    print("PHASE 3 SMART PARKING TEST RESULTS")
    print("=" * 60)
    print(f"Total Frames Processed:   {frame_idx}")
    print(f"Total Processing Time:    {total_time:.2f}s")
    print(f"Average Pipeline Speed:   {avg_fps:.1f} FPS")
    print(f"Configured Slots Count:   {last_stats['total_slots']}")
    print(f"Occupied Slots (Final):   {last_stats['occupied_slots']}")
    print(f"Available Slots (Final):  {last_stats['available_slots']}")
    print(f"Occupancy Rate (Final):   {last_stats['occupancy_rate']:.1f}%")
    print(f"Smart Parking Engine:     PASSED / READY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_phase3_test()
    if not success:
        sys.exit(1)
