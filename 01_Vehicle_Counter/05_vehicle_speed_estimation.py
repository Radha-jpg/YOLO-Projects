"""
Project 05 - Vehicle Speed Estimation

This project estimates vehicle speed
using YOLOv8 and ByteTrack.

Concepts:
- Speed Estimation
- Pixel Distance
- Frame Time
- km/h Conversion
"""

import cv2
import math
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("british_highway_traffic.mp4")

if not cap.isOpened():
    print("Could not open video.")
    exit()

# Approximate calibration
# 20 pixels ≈ 1 meter
PIXELS_PER_METER = 20

previous_positions = {}
previous_times = {}
vehicle_speeds = {}

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    # Run detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    if results[0].boxes.id is not None:
        for box, track_id, cls in zip(
            results[0].boxes.xyxy,
            results[0].boxes.id,
            results[0].boxes.cls
        ):
            cls = int(cls)

            # Skip non-vehicle objects
            if cls not in VEHICLE_CLASSES:
                continue

            vehicle_name = VEHICLE_CLASSES[cls]
            track_id = int(track_id)

            x1, y1, x2, y2 = map(int, box)

            # Vehicle center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if track_id in previous_positions:
                px, py = previous_positions[track_id]

                distance_pixels = math.hypot(cx - px, cy - py)
                distance_meters = distance_pixels / PIXELS_PER_METER

                dt = current_time - previous_times[track_id]

                if dt > 0:
                    speed = (distance_meters / dt) * 3.6
                    vehicle_speeds[track_id] = speed

            previous_positions[track_id] = (cx, cy)
            previous_times[track_id] = current_time

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            cv2.putText(
                frame,
                vehicle_name,
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"{vehicle_speeds.get(track_id, 0):.1f} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    cv2.imshow("Vehicle Speed Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()