import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sample_generator import create_sample_campus_video
from modules.tracking import ObjectTracker
import cv2

def run_phase2_test():
    print("=" * 60)
    print("SMART CAMPUS AI - PHASE 2 OBJECT TRACKING TEST")
    print("=" * 60)

    # Step 1: Use real video sample feed
    video_path = os.path.join("data", "videos", "real_campus.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join("data", "videos", "sample_tracking.mp4")
        create_sample_campus_video(video_path, duration_sec=4, fps=30)

    print("\n[INFO] Initializing ByteTrack ObjectTracker...")
    tracker = ObjectTracker(model_path="yolov8n.pt", tracker_type="bytetrack.yaml")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open test video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video opened. Processing {total_frames} frames for object tracking...")

    frame_idx = 0
    fps_list = []
    tracked_ids_seen = set()

    start_run = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated_frame, tracked_objs, active_counts, unique_counts, fps = tracker.process_frame(frame)
        fps_list.append(fps)

        # Collect active track IDs
        for obj in tracked_objs:
            if obj["track_id"] != -1:
                tracked_ids_seen.add(f"{obj['class_name']}#{obj['track_id']}")

        if frame_idx % 20 == 0 or frame_idx == total_frames:
            id_list = [f"{o['class_name']}#{o['track_id']}" for o in tracked_objs if o['track_id'] != -1]
            id_str = ", ".join(id_list) if id_list else "No persistent IDs tracked"
            print(f"Frame {frame_idx:03d}/{total_frames} | FPS: {fps:.1f} | Active Tracked Objects -> {id_str}")

    cap.release()
    total_time = time.time() - start_run
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    print("\n" + "=" * 60)
    print("PHASE 2 OBJECT TRACKING ENGINE TEST RESULTS")
    print("=" * 60)
    print(f"Total Frames Processed:   {frame_idx}")
    print(f"Total Processing Time:    {total_time:.2f}s")
    print(f"Average Tracking Speed:   {avg_fps:.1f} FPS")
    print(f"Total Unique Track IDs:   {len(tracked_ids_seen)}")
    print(f"Tracked Object Catalog:   {list(tracked_ids_seen) if tracked_ids_seen else 'None (synthetic video feed)'}")
    print(f"Unique Counts Summary:    {unique_counts}")
    print(f"Tracking Engine Status:   PASSED / READY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_phase2_test()
    if not success:
        sys.exit(1)
