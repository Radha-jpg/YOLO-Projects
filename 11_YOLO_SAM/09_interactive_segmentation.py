"""
Project 09 - Interactive Object Segmentation

This project detects objects using YOLO.
Click on any detected object to generate
its segmentation mask using Segment
Anything (SAM).

Concepts:
- YOLO Detection
- Segment Anything (SAM)
- Mouse Interaction
- Interactive Segmentation
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Configuration
IMAGE_PATH = "images/street.jfif"
OUTPUT_PATH = "outputs/interactive_segmentation.jpg"

YOLO_MODEL = "yolov8n.pt"

SAM_MODEL = "vit_b"
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"

os.makedirs("outputs", exist_ok=True)

# Load models
print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading SAM...")
sam = sam_model_registry[SAM_MODEL](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)

# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(rgb)

# Run YOLO once
print("Running YOLO...")

results = yolo(image)

detections = []

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    for box, cls, score in zip(boxes, classes, scores):
        detections.append(
            {
                "box": box.astype(int),
                "class": yolo.names[int(cls)],
                "score": float(score)
            }
        )

display = image.copy()

for detection in detections:
    x1, y1, x2, y2 = detection["box"]

    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        detection["class"],
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


def mouse_callback(event, x, y, flags, param):
    global display

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    nearest = None
    minimum_distance = float("inf")

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        distance = np.hypot(cx - x, cy - y)

        if distance < minimum_distance:
            minimum_distance = distance
            nearest = detection

    if nearest is None:
        return

    x1, y1, x2, y2 = nearest["box"]

    masks, _, _ = predictor.predict(
        box=np.array([x1, y1, x2, y2]),
        multimask_output=False
    )

    mask = masks[0]

    display = image.copy()

    overlay = display.copy()

    color = np.random.randint(0, 255, 3)

    overlay[mask] = color

    display = cv2.addWeighted(
        overlay,
        0.45,
        display,
        0.55,
        0
    )

    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        color.tolist(),
        2
    )

    cv2.putText(
        display,
        nearest["class"],
        (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color.tolist(),
        2
    )

    cv2.imwrite(OUTPUT_PATH, display)

    print(f"Saved: {OUTPUT_PATH}")


cv2.namedWindow("Interactive Segmentation")
cv2.setMouseCallback("Interactive Segmentation", mouse_callback)

print("\nClick on an object to segment it.")
print("Press ESC to exit.")

while True:
    cv2.imshow("Interactive Segmentation", display)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()