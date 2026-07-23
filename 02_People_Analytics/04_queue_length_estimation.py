"""
Project 04 - Queue Length Estimation

This project estimates queue length by
counting people inside a queue area.

Concepts:
- Person Detection
- Polygon ROI
- Queue Length Estimation
- Real-Time Counting
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

PERSON_CLASS = 0

# Open video
cap = cv2.VideoCapture("people_queue.mp4")

if not cap.isOpened():
    print("Could not open video.")
    exit()

# Maximum display width
MAX_WIDTH = 1280

# Queue area (adjust these after resizing if needed)
roi = np.array([
    [200, 420],
    [1050, 420],
    [1150, 700],
    [120, 700]
], np.int32)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame while keeping aspect ratio
    h, w = frame.shape[:2]

    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        frame = cv2.resize(
            frame,
            (int(w * scale), int(h * scale))
        )

    # Draw queue area
    cv2.polylines(frame, [roi], True, (0, 0, 255), 3)

    queue_length = 0

    # Run detection + tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    if results[0].boxes.id is not None:
        for box, cls in zip(
            results[0].boxes.xyxy,
            results[0].boxes.cls
        ):
            if int(cls) != PERSON_CLASS:
                continue

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Person center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Check if person is inside the queue area
            if cv2.pointPolygonTest(roi, (cx, cy), False) >= 0:
                queue_length += 1

                cv2.putText(
                    frame,
                    "In Queue",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

    cv2.putText(
        frame,
        f"Queue Length: {queue_length}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Queue Length Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()