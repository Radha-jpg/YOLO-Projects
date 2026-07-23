"""
Project 02 - People Counter

This project detects, tracks and counts
people crossing a counting line.

Concepts:
- Person Detection
- ByteTrack Tracking
- Line Crossing Counter
"""

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

PERSON_CLASS = 0

# Open video
cap = cv2.VideoCapture("people.mp4")

if not cap.isOpened():
    print("Could not open video.")
    exit()

LINE_Y = 350

counted_ids = set()
people_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Draw counting line
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 3)

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
            # Detect only people
            if int(cls) != PERSON_CLASS:
                continue

            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Person center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            cv2.putText(
                frame,
                f"Person ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Count once when crossing the line
            if abs(cy - LINE_Y) < 8:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    people_count += 1

    cv2.putText(
        frame,
        f"People: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("People Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()