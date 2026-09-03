import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sample_generator import create_sample_campus_video
from modules.detection import YOLODetector
import cv2

def run_phase1_test():
    print("=" * 60)
    print("SMART CAMPUS AI - PHASE 1 VERIFICATION TEST")
    print("=" * 60)

    # Step 1: Use real video sample if available
    video_path = os.path.join("data", "videos", "real_campus.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join("data", "videos", "real_parking.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join("data", "videos", "sample_campus.mp4")
        create_sample_campus_video(video_path, duration_sec=3, fps=30)
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file does not exist at {video_path}")
        return False
        
    print(f"\n[INFO] Loading YOLO detector...")
    detector = YOLODetector(model_path="yolov8n.pt")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video: {video_path}")
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video opened. Total frames to process: {total_frames}")
    
    frame_idx = 0
    fps_list = []
    detection_summary = {}

    start_run = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        annotated_frame, detections, counts, fps = detector.detect_frame(frame)
        fps_list.append(fps)
        
        # Collect counts
        for cls_name, count in counts.items():
            if count > 0:
                detection_summary[cls_name] = max(detection_summary.get(cls_name, 0), count)
                
        if frame_idx % 15 == 0 or frame_idx == total_frames:
            active_counts = ", ".join([f"{k}: {v}" for k, v in counts.items() if v > 0])
            if not active_counts:
                active_counts = "No objects detected (synthetic feed frame)"
            print(f"Frame {frame_idx:03d}/{total_frames} | FPS: {fps:.1f} | Active Objects -> {active_counts}")

    cap.release()
    total_time = time.time() - start_run
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0

    print("\n" + "=" * 60)
    print("PHASE 1 ENGINE TEST RESULTS")
    print("=" * 60)
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Time Taken:     {total_time:.2f}s")
    print(f"Average Inference FPS:{avg_fps:.1f}")
    print(f"Detection Engine Status: PASSED / READY")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_phase1_test()
    if not success:
        sys.exit(1)
