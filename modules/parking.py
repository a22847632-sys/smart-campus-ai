import cv2
import numpy as np
import json
import os

class ParkingSlotManager:
    def __init__(self, slots_config=None, config_file_path=None):
        """
        Initialize the Smart Parking Manager.
        Can load slots from a list of dicts or a JSON configuration file.
        """
        self.slots = []
        
        if slots_config:
            self.slots = slots_config
        elif config_file_path and os.path.exists(config_file_path):
            self.load_config(config_file_path)
        else:
            # Fallback: Generate a default 10-slot parking layout for testing
            self.slots = self._generate_default_parking_slots()

    def _generate_default_parking_slots(self, width=960, height=540):
        """
        Generates default parking slot polygon regions tailored for 960x540 camera feeds.
        """
        slots = []
        # Left Aisle: 5 slots
        for i in range(5):
            y1 = 110 + i * 70
            y2 = y1 + 60
            slot_poly = [(60, y1), (240, y1), (240, y2), (60, y2)]
            slots.append({
                "slot_id": f"P-{i+1:02d}",
                "polygon": slot_poly,
                "type": "car"
            })
            
        # Right Aisle: 5 slots
        for i in range(5):
            y1 = 110 + i * 70
            y2 = y1 + 60
            slot_poly = [(260, y1), (440, y1), (440, y2), (260, y2)]
            slots.append({
                "slot_id": f"P-{i+6:02d}",
                "polygon": slot_poly,
                "type": "car"
            })
            
        return slots

    def load_config(self, file_path):
        """Loads parking slot configurations from a JSON file."""
        with open(file_path, "r") as f:
            self.slots = json.load(f)
        print(f"[ParkingSlotManager] Loaded {len(self.slots)} parking slots from {file_path}")

    def save_config(self, file_path):
        """Saves current parking slot configurations to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.slots, f, indent=4)
        print(f"[ParkingSlotManager] Saved parking slots configuration to {file_path}")

    def update_occupancy(self, detected_objects):
        """
        Evaluates occupancy of each slot based on detected/tracked vehicles.
        Vehicle target classes: Car, Motorcycle, Bus, Truck, Bicycle.
        Returns:
            slot_results: List of dicts detailing status of each slot
            stats: Dict with total, occupied, available counts and occupancy %
        """
        vehicle_classes = {"Car", "Motorcycle", "Bus", "Truck", "Bicycle"}
        vehicles = [obj for obj in detected_objects if obj.get("class_name") in vehicle_classes]
        
        slot_results = []
        occupied_count = 0

        for slot in self.slots:
            slot_id = slot["slot_id"]
            poly_pts = np.array(slot["polygon"], dtype=np.int32)
            
            is_occupied = False
            occupant_info = None

            for veh in vehicles:
                bbox = veh["bbox"]
                cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                bc_x, bc_y = (bbox[0] + bbox[2]) // 2, float(bbox[3]) # Tire contact point on ground
                
                # Check if vehicle centroid or tire contact point lies inside parking slot polygon
                inside_c = cv2.pointPolygonTest(poly_pts, (float(cx), float(cy)), False)
                inside_bc = cv2.pointPolygonTest(poly_pts, (float(bc_x), float(bc_y)), False)
                
                if inside_c >= 0 or inside_bc >= 0:
                    is_occupied = True
                    occupant_info = {
                        "track_id": veh.get("track_id", -1),
                        "class_name": veh.get("class_name", "Vehicle"),
                        "confidence": veh.get("confidence", 0.0)
                    }
                    break

            if is_occupied:
                occupied_count += 1

            slot_results.append({
                "slot_id": slot_id,
                "occupied": is_occupied,
                "polygon": slot["polygon"],
                "occupant": occupant_info
            })

        total_slots = len(self.slots)
        available_count = total_slots - occupied_count
        occupancy_rate = (occupied_count / total_slots * 100.0) if total_slots > 0 else 0.0

        stats = {
            "total_slots": total_slots,
            "occupied_slots": occupied_count,
            "available_slots": available_count,
            "occupancy_rate": occupancy_rate
        }

        return slot_results, stats

    def draw_parking_overlay(self, frame, slot_results, stats, draw_hud=True):
        """
        Renders colored parking polygons (Red = Occupied, Green = Vacant) and HUD summary on the frame.
        """
        annotated = frame.copy()
        overlay = frame.copy()

        # Render Slot Polygons
        for slot in slot_results:
            pts = np.array(slot["polygon"], dtype=np.int32).reshape((-1, 1, 2))
            is_occ = slot["occupied"]
            
            # Colors: Green for vacant, Red for occupied
            color = (0, 0, 220) if is_occ else (0, 200, 0)
            
            # Semi-transparent polygon fill
            cv2.fillPoly(overlay, [pts], color)
            
            # Solid polygon boundary
            cv2.polylines(annotated, [pts], True, color, 2)
            
            # Slot ID text label
            cx = int(np.mean(pts[:, 0, 0]))
            cy = int(np.mean(pts[:, 0, 1]))
            text = slot["slot_id"]
            if is_occ and slot["occupant"] and slot["occupant"]["track_id"] != -1:
                text += f" (#{slot['occupant']['track_id']})"
                
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(annotated, text, (cx - w // 2, cy + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend semi-transparent slot fill with original frame
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

        # Render Parking HUD Header
        if draw_hud:
            self._draw_parking_hud(annotated, stats)

        return annotated

    def _draw_parking_hud(self, frame, stats):
        """Renders parking summary stats overlay."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (640, 10), (950, 100), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "PARKING MANAGEMENT", (650, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)
        
        cv2.line(frame, (650, 36), (940, 36), (100, 100, 100), 1)

        cv2.putText(frame, f"Total Slots:    {stats['total_slots']}", (650, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
        cv2.putText(frame, f"Occupied:       {stats['occupied_slots']}", (650, 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                    
        cv2.putText(frame, f"Available:      {stats['available_slots']} ({100 - stats['occupancy_rate']:.1f}% Free)", (650, 91),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
