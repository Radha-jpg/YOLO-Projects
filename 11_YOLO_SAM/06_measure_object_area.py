"""
Project 06 - Object Area Measurement

This project detects objects using YOLO
and measures their segmented area using
Segment Anything (SAM).

Concepts:
- YOLO Detection
- Segment Anything (SAM)
- Object Measurement
- Segmentation Overlay
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Configuration
IMAGE_PATH = "images/fruits.jfif"
OUTPUT_PATH = "outputs/measured_area.jpg"

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

# Run YOLO detection
print("Running YOLO...")

results = yolo(image)

output = image.copy()

# Keep colors consistent
np.random.seed(42)

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box.astype(int)

        masks, _, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False
        )

        mask = masks[0]

        # Calculate segmented area
        area_pixels = int(mask.sum())

        color = np.random.randint(0, 255, 3)

        overlay = output.copy()
        overlay[mask] = color

        output = cv2.addWeighted(
            overlay,
            0.4,
            output,
            0.6,
            0
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color.tolist(),
            2
        )

        cv2.putText(
            output,
            f"{yolo.names[int(cls)]} | {area_pixels} px²",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color.tolist(),
            2
        )

        print(f"{yolo.names[int(cls)]:<15} Area: {area_pixels} px²")

cv2.imwrite(OUTPUT_PATH, output)

print(f"\nSaved: {OUTPUT_PATH}")

cv2.imshow("Object Area Measurement", output)
cv2.waitKey(0)
cv2.destroyAllWindows()