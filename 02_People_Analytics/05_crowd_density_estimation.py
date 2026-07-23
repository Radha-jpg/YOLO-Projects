"""
Project 05 - Crowd Density Estimation

This project estimates crowd density
using YOLOv8 person detection.

Concepts:
- Person Detection
- Crowd Counting
- Density Estimation
"""

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

PERSON_CLASS = 0

# Open video
cap = cv2.VideoCapture("crowd.mp4")

if not cap.isOpened():
    print("Could not open video.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run detection
    results = model(frame)
    people_count = 0
    for box, cls, conf in zip(
        results[0].boxes.xyxy,
        results[0].boxes.cls,
        results[0].boxes.conf
    ):
        # Detect only people
        if int(cls) != PERSON_CLASS:
            continue

        people_count += 1

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Person {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    # Estimate crowd density
    if people_count < 5:
        density = "LOW"
        color = (0, 255, 0)

    elif people_count < 15:
        density = "MEDIUM"
        color = (0, 255, 255)

    else:
        density = "HIGH"
        color = (0, 0, 255)

    cv2.putText(
        frame,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Density: {density}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.imshow("Crowd Density Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()