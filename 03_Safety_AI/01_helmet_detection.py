"""
Project 01 - Helmet Detection

This project detects helmets using
a custom YOLOv8 model.

Concepts:
- Custom YOLO Model
- Helmet Detection
- Confidence Score
- Real-Time Monitoring
"""

import cv2
from ultralytics import YOLO

# Load custom model
model = YOLO("best.pt")

# Open video
cap = cv2.VideoCapture("helmet.mp4")

if not cap.isOpened():
    print("Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()