"""
Project 04 - Vehicle Class Counter

This project detects, tracks and counts
different types of vehicles separately.

Concepts:
- Vehicle Class Filtering
- Individual Class Counting
- YOLO Class Names
"""

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Open video
cap = cv2.VideoCapture("british_highway_traffic.mp4")

if not cap.isOpened():
    print("Could not open video.")
    exit()

LINE_Y = 400

counted_ids = set()

vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

while True:
    ret, frame = cap.read()

    if not ret:
        break

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
            cls = int(cls)

            # Skip non-vehicle objects
            if cls not in VEHICLE_CLASSES:
                continue

            vehicle_name = VEHICLE_CLASSES[cls]
            track_id = int(track_id)

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            cv2.putText(
                frame,
                f"{vehicle_name} ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # Count once when crossing the line
            if abs(cy - LINE_Y) < 8:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    vehicle_counts[vehicle_name] += 1

    # Display statistics
    y = 40
    total = sum(vehicle_counts.values())

    cv2.putText(
        frame,
        f"Total: {total}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    y += 35

    for vehicle, count in vehicle_counts.items():
        cv2.putText(
            frame,
            f"{vehicle}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 30

    cv2.imshow("Vehicle Class Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()