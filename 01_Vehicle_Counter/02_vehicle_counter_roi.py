"""
Project 02 - Vehicle Counter using ROI

This project counts vehicles only after they enter
a user-defined Region of Interest (ROI).

Concepts:
- YOLOv8 Detection
- ByteTrack Tracking
- Polygon ROI
- Point Polygon Test
"""

import cv2
import numpy as np
from ultralytics import YOLO
# Load YOLO model
model = YOLO("yolov8n.pt")
# Open video
cap = cv2.VideoCapture("british_highway_traffic.mp4")
if not cap.isOpened():
    print("Could not open video.")
    exit()
# Region of Interest
roi = np.array([
    [250, 250],
    [1050, 250],
    [1150, 650],
    [150, 650]
], np.int32)
counted_ids = set()
vehicle_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Draw the ROI
    cv2.polylines(frame, [roi], True, (0, 0, 255), 3)
    # Run detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )
    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Vehicle center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            # Check if the center is inside the ROI
            inside = cv2.pointPolygonTest(roi, (cx, cy), False)
            if inside >= 0:
                cv2.putText(
                    frame,
                    "Inside ROI",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                # Count each vehicle only once
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    vehicle_count += 1
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )
    cv2.imshow("Vehicle Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()