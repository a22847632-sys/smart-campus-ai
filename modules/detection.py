import time
import cv2
import numpy as np
from ultralytics import YOLO
from config.config import DEFAULT_MODEL_PATH, CONFIDENCE_THRESHOLD, TARGET_CLASSES, CLASS_COLORS, DEFAULT_COLOR

class YOLODetector:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD, target_classes=TARGET_CLASSES):
        """
        Initialize the YOLO Object Detection Engine.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        self.model = YOLO(self.model_path)
        
        # FPS calculation variables
        self.prev_time = time.time()
        self.fps = 0.0

    def detect_frame(self, frame, draw_annotations=True):
        """
        Processes a single BGR image frame and performs object detection.
        Returns:
            annotated_frame: Frame with drawn bounding boxes, labels, FPS, and counts
            detections: List of dicts with bounding box, class, confidence
            counts: Dict containing counts of detected target classes
            fps: Current frames per second
        """
        start_time = time.time()
        
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        counts = {name: 0 for name in self.target_classes.values()}
        
        annotated_frame = frame.copy() if draw_annotations else frame
        
        # Parse detections
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
                # Filter only target classes if specified
                if cls_id in self.target_classes:
                    class_name = self.target_classes[cls_id]
                    counts[class_name] = counts.get(class_name, 0) + 1
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    detections.append({
                        "class_id": int(cls_id),
                        "class_name": class_name,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })
                    
                    if draw_annotations:
                        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Label text
                        label = f"{class_name} {conf:.2f}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Calculate FPS
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        self.prev_time = curr_time

        if draw_annotations:
            # Draw overlay header with FPS and Counts
            self._draw_stats_overlay(annotated_frame, counts, self.fps)

        return annotated_frame, detections, counts, self.fps

    def _draw_stats_overlay(self, frame, counts, fps):
        """
        Renders a stats panel on the top-left corner of the frame.
        """
        # Semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 40 + len(counts) * 22), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Header: FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Object Counts list
        y_offset = 55
        for class_name, count in counts.items():
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            text = f"{class_name}: {count}"
            cv2.putText(frame, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_offset += 22


def test_detector_source(source=0, display=False, max_frames=100):
    """
    Test helper function to run detection on a given video source or synthetic feed.
    """
    print(f"[YOLODetector] Opening input source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return False
        
    detector = YOLODetector()
    frame_count = 0
    
    print("[YOLODetector] Starting detection loop...")
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame, detections, counts, fps = detector.detect_frame(frame)
        frame_count += 1
        
        if frame_count % 10 == 0 or frame_count == 1:
            counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items() if v > 0])
            if not counts_str:
                counts_str = "No target objects detected"
            print(f"Frame {frame_count:03d} | FPS: {fps:.1f} | {counts_str}")
            
        if display:
            cv2.imshow("Smart Campus AI - Phase 1 Detection Engine", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    if display:
        cv2.destroyAllWindows()
    print(f"[YOLODetector] Test completed successfully after {frame_count} frames.")
    return True
