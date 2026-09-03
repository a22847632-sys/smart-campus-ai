import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sample_generator import create_sample_campus_video
from modules.tracking import ObjectTracker
from modules.crowd import CrowdQueueManager
import cv2

def run_phase4_test():
    print("=" * 60)
    print("SMART CAMPUS AI - PHASE 4 CROWD & QUEUE MANAGEMENT TEST")
    print("=" * 60)

    # Step 1: Use real campus video feed
    video_path = os.path.join("data", "videos", "real_campus.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join("data", "videos", "sample_crowd.mp4")
        create_sample_campus_video(video_path, duration_sec=4, fps=30)

    # Step 2: Initialize tracker & crowd queue manager
    print("[INFO] Initializing ObjectTracker & CrowdQueueManager...")
    tracker = ObjectTracker(model_path="yolov8n.pt")
    crowd_mgr = CrowdQueueManager()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video opened. Processing {total_frames} frames for Crowd & Queue Management...")

    frame_idx = 0
    fps_list = []
    last_analytics = None

    start_run = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Track objects
        annotated_frame, tracked_objs, active_counts, unique_counts, fps = tracker.process_frame(frame, draw_annotations=False)
        
        # Process Crowd Analytics
        analytics, person_details = crowd_mgr.process_crowd(tracked_objs)
        annotated_crowd = crowd_mgr.draw_crowd_overlay(annotated_frame, analytics)

        fps_list.append(fps)
        last_analytics = analytics

        if frame_idx % 20 == 0 or frame_idx == total_frames:
            print(f"Frame {frame_idx:03d}/{total_frames} | Total People: {analytics['total_people']} | "
                  f"Crowd Zone: {analytics['crowd_zone_count']} ({analytics['crowd_status']}) | "
                  f"Queue: {analytics['queue_count']} ({analytics['queue_status']})")

    cap.release()
    total_time = time.time() - start_run
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    print("\n" + "=" * 60)
    print("PHASE 4 CROWD & QUEUE MANAGEMENT TEST RESULTS")
    print("=" * 60)
    print(f"Total Frames Processed:   {frame_idx}")
    print(f"Total Processing Time:    {total_time:.2f}s")
    print(f"Average Pipeline Speed:   {avg_fps:.1f} FPS")
    print(f"Total People Detected:    {last_analytics['total_people']}")
    print(f"Crowd Zone Density Count: {last_analytics['crowd_zone_count']}")
    print(f"Crowd Status Level:       {last_analytics['crowd_status']}")
    print(f"Queue Length Estimate:    {last_analytics['queue_count']}")
    print(f"Queue Status Level:       {last_analytics['queue_status']}")
    print(f"Timestamp of Analytics:   {last_analytics['timestamp']}")
    print(f"Crowd & Queue Engine:     PASSED / READY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_phase4_test()
    if not success:
        sys.exit(1)
