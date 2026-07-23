"""
Project: 01 - Vehicle Counter using YOLOv8 + ByteTrack

Description:
------------
This project detects, tracks, and counts vehicles crossing a
horizontal counting line using YOLOv8 and ByteTrack.

Features:
---------
Vehicle Detection
Object Tracking
Unique ID Assignment
Vehicle Counting
Real-time Display
"""
import cv2
from ultralytics import YOLO
# Load YOLOv8 Model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("british_highway_traffic.mp4")
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()
# Counting Line
LINE_Y = 400
# Store counted vehicle IDs
counted_ids = set()
# Total vehicle count
vehicle_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Perform Detection + Tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )
    cv2.line(
        frame,
        (0, LINE_Y),
        (frame.shape[1], LINE_Y),
        (0, 0, 255),
        3,
    )
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id
        classes = results[0].boxes.cls
        for box, track_id, cls in zip(boxes, ids, classes):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            # Draw Bounding Box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            # Compute Center Point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # Draw Center
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (255, 0, 0),
                -1,
            )
            # Display Tracking ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            # Count Vehicles Crossing the Line
            if abs(cy - LINE_Y) < 8:

                if track_id not in counted_ids:

                    counted_ids.add(track_id)
                    vehicle_count += 1
    # Display Total Count
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
    cv2.imshow("Vehicle Counter", frame)
    # Press Q to Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()