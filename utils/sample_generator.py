import cv2
import numpy as np
import os
from config.config import VIDEOS_DIR

def create_sample_campus_video(output_path=None, duration_sec=5, fps=30):
    """
    Generates a synthetic campus video containing moving shapes representing
    vehicles (cars, motorcycles) and pedestrians (people) for testing CV pipelines.
    """
    if output_path is None:
        output_path = os.path.join(VIDEOS_DIR, "sample_campus.mp4")
        
    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    print(f"[SampleGenerator] Generating synthetic video ({total_frames} frames) -> {output_path}")
    
    for i in range(total_frames):
        # Background: Campus courtyard / asphalt / parking layout
        frame = np.ones((height, width, 3), dtype=np.uint8) * 60
        
        # Draw parking lines (gray/white lines)
        cv2.rectangle(frame, (50, 100), (450, 480), (80, 80, 80), -1)
        for p in range(120, 480, 70):
            cv2.line(frame, (50, p), (450, p), (200, 200, 200), 2)
        cv2.line(frame, (250, 100), (250, 480), (200, 200, 200), 3)

        # Draw road / walkway
        cv2.rectangle(frame, (500, 0), (960, 540), (450, 45, 45), -1) # Pedestrian walkway
        
        # Vehicle 1: Moving car down parking aisle
        car1_y = int(120 + (i * 3) % 300)
        cv2.rectangle(frame, (120, car1_y), (200, car1_y + 50), (30, 140, 240), -1) # Car body
        cv2.putText(frame, "CAR", (130, car1_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Vehicle 2: Static parked car
        cv2.rectangle(frame, (280, 200), (360, 250), (40, 200, 40), -1)
        
        # Pedestrian 1: Walking across the courtyard
        p1_x = int(520 + (i * 4) % 400)
        p1_y = int(250 + np.sin(i * 0.1) * 20)
        cv2.circle(frame, (p1_x, p1_y), 15, (220, 180, 50), -1) # Head
        cv2.rectangle(frame, (p1_x - 10, p1_y + 15), (p1_x + 10, p1_y + 45), (200, 100, 50), -1) # Body

        # Pedestrian 2: Another person walking
        p2_x = int(880 - (i * 3) % 350)
        p2_y = int(150 + (i * 2) % 300)
        cv2.circle(frame, (p2_x, p2_y), 14, (200, 200, 0), -1)
        cv2.rectangle(frame, (p2_x - 8, p2_y + 14), (p2_x + 8, p2_y + 40), (180, 50, 180), -1)

        out.write(frame)

    out.release()
    print(f"[SampleGenerator] Video created successfully at: {output_path}")
    return output_path

if __name__ == "__main__":
    create_sample_campus_video()
