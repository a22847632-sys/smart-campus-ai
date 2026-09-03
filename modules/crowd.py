import cv2
import numpy as np
from datetime import datetime

class CrowdQueueManager:
    def __init__(self, crowd_zone=None, queue_zone=None, thresholds=None):
        """
        Initialize Crowd and Queue Analytics Manager.
        """
        # Default ROIs for 960x540 camera view if not provided
        self.crowd_zone = crowd_zone or [(500, 40), (940, 40), (940, 500), (500, 500)]
        self.queue_zone = queue_zone or [(520, 80), (720, 80), (720, 460), (520, 460)]
        
        # Density & Queue Thresholds
        self.thresholds = thresholds or {
            "crowd_medium": 4,
            "crowd_high": 10,
            "queue_medium": 3,
            "queue_high": 7
        }

    def process_crowd(self, tracked_objects):
        """
        Analyzes person detections, estimates crowd density in zones, and estimates queue length.
        Returns:
            analytics: Dict containing timestamped crowd & queue metrics
            person_details: List of person locations and zone assignments
        """
        persons = [obj for obj in tracked_objects if obj.get("class_name") == "Person"]
        total_people = len(persons)
        
        crowd_poly = np.array(self.crowd_zone, dtype=np.int32)
        queue_poly = np.array(self.queue_zone, dtype=np.int32)

        crowd_zone_count = 0
        queue_count = 0
        person_details = []

        for p in persons:
            bbox = p["bbox"]
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            
            in_crowd = cv2.pointPolygonTest(crowd_poly, (float(cx), float(cy)), False) >= 0
            in_queue = cv2.pointPolygonTest(queue_poly, (float(cx), float(cy)), False) >= 0

            if in_crowd:
                crowd_zone_count += 1
            if in_queue:
                queue_count += 1

            person_details.append({
                "track_id": p.get("track_id", -1),
                "centroid": (cx, cy),
                "in_crowd_zone": in_crowd,
                "in_queue_zone": in_queue
            })

        # Determine Crowd Status
        if crowd_zone_count >= self.thresholds["crowd_high"]:
            crowd_status = "HIGH"
        elif crowd_zone_count >= self.thresholds["crowd_medium"]:
            crowd_status = "MEDIUM"
        else:
            crowd_status = "LOW"

        # Determine Queue Status
        if queue_count >= self.thresholds["queue_high"]:
            queue_status = "HIGH"
        elif queue_count >= self.thresholds["queue_medium"]:
            queue_status = "MEDIUM"
        else:
            queue_status = "LOW"

        analytics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_people": total_people,
            "crowd_zone_count": crowd_zone_count,
            "queue_count": queue_count,
            "crowd_status": crowd_status,
            "queue_status": queue_status
        }

        return analytics, person_details

    def draw_crowd_overlay(self, frame, analytics, draw_hud=True):
        """
        Renders crowd/queue ROI boundaries and analytics HUD on frame.
        """
        annotated = frame.copy()
        overlay = frame.copy()

        # Status Colors (BGR)
        status_colors = {
            "LOW": (0, 220, 0),      # Green
            "MEDIUM": (0, 215, 255), # Yellow
            "HIGH": (0, 0, 255)      # Red
        }

        crowd_color = status_colors.get(analytics["crowd_status"], (255, 255, 255))
        queue_color = status_colors.get(analytics["queue_status"], (255, 255, 255))

        # Render Crowd Zone ROI
        pts_crowd = np.array(self.crowd_zone, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts_crowd], True, crowd_color, 2)
        cv2.putText(annotated, f"CROWD ZONE [{analytics['crowd_status']}]", (self.crowd_zone[0][0] + 10, self.crowd_zone[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, crowd_color, 2, cv2.LINE_AA)

        # Render Queue Zone ROI
        pts_queue = np.array(self.queue_zone, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts_queue], True, (255, 0, 255), 2)
        cv2.putText(annotated, f"QUEUE ZONE ({analytics['queue_count']} in line)", (self.queue_zone[0][0] + 10, self.queue_zone[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)

        if draw_hud:
            self._draw_crowd_hud(annotated, analytics, crowd_color, queue_color)

        return annotated

    def _draw_crowd_hud(self, frame, analytics, crowd_color, queue_color):
        """Renders HUD panel for crowd and queue metrics."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 150), (320, 270), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "CROWD & QUEUE MONITOR", (20, 172),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2, cv2.LINE_AA)
        
        cv2.line(frame, (20, 180), (310, 180), (100, 100, 100), 1)

        cv2.putText(frame, f"Total People Detected: {analytics['total_people']}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Crowd Zone Density:   {analytics['crowd_status']}", (20, 222),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, crowd_color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"Queue Length:         {analytics['queue_count']} ({analytics['queue_status']})", (20, 244),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, queue_color, 1, cv2.LINE_AA)
                    
        cv2.putText(frame, f"Last Updated: {analytics['timestamp'].split()[-1]}", (20, 264),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
