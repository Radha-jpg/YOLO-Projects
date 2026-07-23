"""
Project 03 - Vehicle Direction Counter

This project counts vehicles moving IN and OUT
using YOLOv8 and ByteTrack.

Concepts:
- Direction Detection
- Previous Position Tracking
- IN/OUT Counting
"""

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("british_highway_traffic.mp4")
if not cap.isOpened():
    print("Could not open video.")
    exit()
# Counting line
LINE_Y = 400
previous_positions = {}
counted_in = set()
counted_out = set()

in_count = 0
out_count = 0

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
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Vehicle center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            # Check movement direction
            if track_id in previous_positions:
                previous_y = previous_positions[track_id]
                # Moving down (IN)
                if previous_y < LINE_Y <= cy:
                    if track_id not in counted_in:
                        counted_in.add(track_id)
                        in_count += 1
                # Moving up (OUT)
                elif previous_y > LINE_Y >= cy:
                    if track_id not in counted_out:
                        counted_out.add(track_id)
                        out_count += 1
            # Save current position
            previous_positions[track_id] = cy
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    # Display counts
    cv2.putText(
        frame,
        f"IN: {in_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"OUT: {out_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    cv2.imshow("Vehicle Direction Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()