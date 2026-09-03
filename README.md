# AI-Powered Smart Campus Management System 🚀

An end-to-end Computer Vision & Deep Learning solution for smart campus automation, real-time vehicle & pedestrian monitoring, smart parking slot management, crowd density analysis, and campus emergency event detection.

---

## 📌 Project Overview

This system processes live camera feeds (webcam or video streams) through a multi-stage Computer Vision pipeline:
`Camera Input → YOLOv8 Object Detection → ByteTrack Object Tracking → Smart Modules (Parking / Crowd / Emergency) → SQLite Database → Flask REST API & Web Dashboard`.

---

## 🛠️ 1. Installation & Requirements

### System Requirements:
- **Python**: Version 3.8 to 3.13
- **OS**: Windows / Linux / macOS

### Required Packages:
Install all necessary Python libraries using `pip`:
```bash
pip install -r requirements.txt
```

*(Requirements include: `ultralytics`, `opencv-python`, `torch`, `flask`, `numpy`, `pillow`)*

---

## 🚀 2. How to Start the Application

Open your terminal or command prompt in the project root folder and run:
```bash
python app.py
```

After starting, open your browser and visit:
👉 **`http://127.0.0.1:5000`**

---

## 🎥 3. Video & Webcam Input Selection

### A. Run Default Campus Video (Included)
By default, the server runs on `data/videos/real_campus.mp4`:
```text
http://127.0.0.1:5000
```

### B. Use Live Webcam Feed (Index 0)
Open this URL in your browser:
```text
http://127.0.0.1:5000/video_feed?source=0
```
*(Or set environment variable `$env:VIDEO_SOURCE="0"` before running `python app.py`)*

### C. Use Custom Video File
Pass any video file path via the URL parameter:
```text
http://127.0.0.1:5000/video_feed?source=data/videos/real_parking.mp4
```

---

## 📹 4. Available Demo Videos
The project includes pre-configured sample clips inside `data/videos/`:
- `data/videos/real_campus.mp4` — Real campus pedestrians, cars, bicycles.
- `data/videos/real_parking.mp4` — Real vehicle traffic & parking lot feed.

---

## 🧩 5. Core AI Modules Explanation

### 🚗 Module 1 — Smart Parking Management
- **How it works**: Uses polygon region-of-interest (ROI) matching from `data/parking_config.json`. Calculates vehicle centroids and ground tire contact points to test if a vehicle occupies a slot.
- **Metrics**: Total slots, Occupied count, Available slots, Occupancy Rate %.
- **Visuals**: Green polygons = Vacant slots, Red polygons = Occupied slots.

### 👥 Module 2 — Crowd & Queue Management
- **How it works**: Tracks detected `Person` objects in designated zone polygons.
- **Density Classification**:
  - `LOW`: < 4 people
  - `MEDIUM`: 4 to 9 people
  - `HIGH`: ≥ 10 people
- **Queue Estimation**: Counts people standing within the queue region boundary.

### 🚨 Module 3 — Campus Emergency Detection
- **Fallen Person Detection**: Evaluates 2D bounding box posture aspect ratio (`width / height > 1.15`). Highlights fallen person in bright red outline.
- **Sudden Crowd Formation**: Monitors frame-over-frame rate of count increase to alert on sudden crowd surges.
- **Alert Payload**: Contains `event_type`, `location`, `timestamp`, `confidence`, and `status`.

---

## 🎓 6. College Presentation / Demo Procedure

1. **Start Server**: `python app.py`
2. **Open Dashboard**: Go to `http://127.0.0.1:5000`
3. **Key Presentation Points**:
   - **Real-Time YOLO Bounding Boxes**: Show live bounding boxes with class names (`Person`, `Car`, `Bicycle`) and confidence values.
   - **ByteTrack Persistent Tracking**: Show persistent IDs (`Person #1`, `Car #6`) and trajectory motion trails.
   - **Smart Parking Overlay**: Show live parking slot status (Green/Red) and real-time total/available slot counts.
   - **Crowd Density Status**: Show `LOW/MEDIUM/HIGH` density status and queue counter.
   - **Database Integration**: Telemetry data logs to SQLite database (`database/smart_campus.db`) every 15 frames.

---

## ⚠️ 7. Troubleshooting & Known Limitations

- **Low FPS on CPU**: Inference runs at ~15-20 FPS on CPU. If a CUDA GPU is available, PyTorch will automatically accelerate execution.
- **Camera Perspective**: Parking slot polygons (`data/parking_config.json`) are configured for standard 16:9 camera views. For a new camera position, ROI polygons can be reconfigured in `parking_config.json`.
- **Fallen Person Heuristics**: Relies on 2D bounding box aspect ratio rules.
