import time
import cv2
import numpy as np
from ultralytics import YOLO
from config.config import DEFAULT_MODEL_PATH, CONFIDENCE_THRESHOLD, TARGET_CLASSES, CLASS_COLORS, DEFAULT_COLOR

class ObjectTracker:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD, target_classes=TARGET_CLASSES, tracker_type="bytetrack.yaml"):
        """
        Initialize the YOLO Object Tracking Engine with persistent ID tracking.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes
        self.tracker_type = tracker_type
        self.model = YOLO(self.model_path)
        
        # Track history: {track_id: [(cx, cy), ...]}
        self.track_history = {}
        self.max_history_len = 30
        
        # Unique IDs seen per class
        self.unique_ids = {name: set() for name in self.target_classes.values()}
        self.unique_ids["Vehicle"] = set()  # Aggregate vehicles (Car, Motorcycle, Bus, Truck)
        
        # Performance metrics
        self.prev_time = time.time()
        self.fps = 0.0

    def process_frame(self, frame, draw_annotations=True):
        """
        Processes a single frame, runs tracking inference, updates persistent IDs, and draws overlays.
        Returns:
            annotated_frame: Frame with bounding boxes, persistent IDs, and trajectory trails
            tracked_objects: List of dicts containing track_id, class_name, bbox, centroid
            active_counts: Count of currently active objects per class
            unique_counts: Count of cumulative unique objects tracked per class
            fps: Current frames per second
        """
        # Run YOLO tracking with persist=True to keep tracks across calls
        results = self.model.track(
            frame, 
            conf=self.conf_threshold, 
            persist=True, 
            tracker=self.tracker_type, 
            verbose=False
        )[0]
        
        tracked_objects = []
        active_counts = {name: 0 for name in self.target_classes.values()}
        annotated_frame = frame.copy() if draw_annotations else frame

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Extract track IDs (may be None if object just appeared before tracking confirmation)
            track_ids = results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else [None] * len(boxes)

            for bbox, conf, cls_id, track_id in zip(boxes, confidences, class_ids, track_ids):
                if cls_id in self.target_classes:
                    class_name = self.target_classes[cls_id]
                    x1, y1, x2, y2 = map(int, bbox)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Fallback track ID if tracker did not assign one
                    tid = int(track_id) if track_id is not None else -1

                    # Register unique ID
                    if tid != -1:
                        self.unique_ids[class_name].add(tid)
                        if class_name in ["Car", "Motorcycle", "Bus", "Truck", "Bicycle"]:
                            self.unique_ids["Vehicle"].add(tid)

                        # Update motion trajectory
                        if tid not in self.track_history:
                            self.track_history[tid] = []
                        self.track_history[tid].append((cx, cy))
                        if len(self.track_history[tid]) > self.max_history_len:
                            self.track_history[tid].pop(0)

                    active_counts[class_name] = active_counts.get(class_name, 0) + 1
                    
                    obj_info = {
                        "track_id": tid,
                        "class_id": int(cls_id),
                        "class_name": class_name,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "centroid": (cx, cy)
                    }
                    tracked_objects.append(obj_info)

                    if draw_annotations:
                        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
                        # Bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Label text with Track ID
                        id_str = f"#{tid}" if tid != -1 else "New"
                        label = f"{class_name} {id_str} ({conf:.2f})"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + w + 4, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 2, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                        # Draw trajectory tail
                        if tid in self.track_history and len(self.track_history[tid]) > 1:
                            pts = np.array(self.track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)

        # FPS Calculation
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        self.prev_time = curr_time

        # Prepare summary of unique counts
        unique_counts_summary = {k: len(v) for k, v in self.unique_ids.items()}

        if draw_annotations:
            self._draw_tracking_overlay(annotated_frame, active_counts, unique_counts_summary, self.fps)

        return annotated_frame, tracked_objects, active_counts, unique_counts_summary, self.fps

    def _draw_tracking_overlay(self, frame, active_counts, unique_counts, fps):
        """
        Renders HUD tracking stats on top of the video frame.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 140), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title & FPS
        cv2.putText(frame, f"TRACKING ENGINE | FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.line(frame, (20, 38), (310, 38), (100, 100, 100), 1)

        # Active vs Unique Statistics
        cv2.putText(frame, f"Active Persons: {active_counts.get('Person', 0)}  (Total Unique: {unique_counts.get('Person', 0)})",
                    (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLASS_COLORS.get("Person", (255, 120, 0)), 1, cv2.LINE_AA)
                    
        active_veh = sum(active_counts.get(cls, 0) for cls in ["Car", "Motorcycle", "Bus", "Truck"])
        cv2.putText(frame, f"Active Vehicles: {active_veh}  (Total Unique: {unique_counts.get('Vehicle', 0)})",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLASS_COLORS.get("Car", (0, 215, 255)), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Car: {active_counts.get('Car', 0)} | Moto: {active_counts.get('Motorcycle', 0)}",
                    (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"ByteTrack Persistent ID Active", (20, 124),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 128), 1, cv2.LINE_AA)
