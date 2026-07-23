"""
Project 11 - Object Cutout Generator

This project detects objects using YOLO
and extracts each object as a transparent
PNG using Segment Anything (SAM).

Concepts:
- YOLO Detection
- Segment Anything (SAM)
- Object Extraction
- Transparent PNG
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Configuration
IMAGE_PATH = "busy_street.jfif"
OUTPUT_DIR = "outputs/cutouts"

YOLO_MODEL = "yolov8n.pt"

SAM_MODEL = "vit_b"
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading SAM...")
sam = sam_model_registry[SAM_MODEL](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)

# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError("Image not found.")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(rgb)

# Detect objects
print("Running YOLO detection...")

results = yolo(image)

object_count = {}

print("\nExtracting objects...\n")

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        class_name = yolo.names[int(cls)]

        object_count[class_name] = object_count.get(class_name, 0) + 1
        index = object_count[class_name]

        x1, y1, x2, y2 = box.astype(int)

        masks, _, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False
        )

        mask = masks[0]

        # Create transparent image
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask.astype(np.uint8) * 255

        # Crop detected object
        cutout = rgba[y1:y2, x1:x2]

        filename = f"{class_name}_{index}.png"

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, filename),
            cutout
        )

        print(f"Saved {filename}")

print("\nAll objects extracted successfully!")