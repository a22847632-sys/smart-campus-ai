import time
import cv2
import numpy as np
from datetime import datetime

class EmergencyDetector:
    def __init__(self, camera_location="Cam-01 (Main Courtyard)", surge_threshold=4, window_frames=30):
        """
        Initialize Campus Emergency Detection Engine.
        """
        self.camera_location = camera_location
        self.surge_threshold = surge_threshold  # Count increase triggering crowd surge
        self.window_frames = window_frames
        
        # History of person counts for sudden crowd surge detection
        self.person_count_history = []
        
        # Active alerts list
        self.active_alerts = []

    def detect_emergencies(self, tracked_objects, current_frame_idx=0):
        """
        Evaluates frame objects for emergency scenarios:
        1. Person Lying / Fallen
        2. Sudden Crowd Formation
        Returns:
            alerts: List of active emergency alert dicts
            fallen_persons: List of detected fallen person objects
        """
        persons = [obj for obj in tracked_objects if obj.get("class_name") == "Person"]
        curr_count = len(persons)
        
        alerts = []
        fallen_persons = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # -------------------------------------------------------------
        # 1. PERSON FALLEN / LYING ON GROUND DETECTION
        # -------------------------------------------------------------
        for p in persons:
            bbox = p["bbox"]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            if h > 0:
                aspect_ratio = w / float(h)
                # If width significantly exceeds height (horizontal stance)
                if aspect_ratio > 1.15 and w > 20:
                    conf = min(0.95, float(p.get("confidence", 0.8)) + 0.15)
                    fallen_persons.append(p)
                    
                    alert = {
                        "event_type": "PERSON_FALLEN",
                        "location": self.camera_location,
                        "timestamp": now_str,
                        "confidence": conf,
                        "status": "ACTIVE",
                        "details": f"Person #{p.get('track_id', 'Unidentified')} lying horizontally (aspect ratio: {aspect_ratio:.2f})"
                    }
                    alerts.append(alert)

        # -------------------------------------------------------------
        # 2. SUDDEN CROWD FORMATION DETECTION
        # -------------------------------------------------------------
        self.person_count_history.append(curr_count)
        if len(self.person_count_history) > self.window_frames:
            self.person_count_history.pop(0)

        if len(self.person_count_history) >= 10:
            min_count_in_window = min(self.person_count_history[:-3]) if len(self.person_count_history) > 3 else self.person_count_history[0]
            surge = curr_count - min_count_in_window
            
            if surge >= self.surge_threshold and curr_count >= 5:
                alerts.append({
                    "event_type": "SUDDEN_CROWD_FORMATION",
                    "location": self.camera_location,
                    "timestamp": now_str,
                    "confidence": 0.88,
                    "status": "ACTIVE",
                    "details": f"Rapid surge of +{surge} people detected within recent frames (Total: {curr_count})"
                })

        self.active_alerts = alerts
        return alerts, fallen_persons

    def draw_emergency_overlay(self, frame, alerts, fallen_persons):
        """
        Renders visual alert banners and highlights fallen persons on the frame.
        """
        annotated = frame.copy()

        # Highlight Fallen Persons in bright Red with flashing outline
        for p in fallen_persons:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            label = f"EMERGENCY: FALLEN PERSON #{p.get('track_id', '')}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w + 10, y1), (0, 0, 255), -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Top Emergency HUD Banner if any alerts active
        if alerts:
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 45), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            event_names = ", ".join(set([a["event_type"] for a in alerts]))
            alert_text = f"CRITICAL ALERT: [{event_names}] DETECTED AT {self.camera_location}"
            cv2.putText(annotated, alert_text, (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated
