"""
Project 07 - Background Replacement

This project detects people using YOLO
and replaces the background using
Segment Anything (SAM).

Concepts:
- YOLO Detection
- Segment Anything (SAM)
- Background Replacement
- Image Compositing
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Configuration
IMAGE_PATH = "person2.jfif"
BACKGROUND_PATH = "beach.avif"
OUTPUT_PATH = "outputs/replaced_background.jpg"

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

# Load images
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

background = cv2.imread(BACKGROUND_PATH)

if background is None:
    raise FileNotFoundError(f"Background not found: {BACKGROUND_PATH}")

background = cv2.resize(
    background,
    (image.shape[1], image.shape[0])
)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(rgb)

# Run YOLO detection
print("Running YOLO...")

results = yolo(image)

combined_mask = np.zeros(image.shape[:2], dtype=bool)

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        # Segment only people
        if int(cls) != 0:
            continue

        masks, _, _ = predictor.predict(
            box=np.array(box),
            multimask_output=False
        )

        combined_mask |= masks[0]

# Replace background
output = background.copy()
output[combined_mask] = image[combined_mask]

cv2.imwrite(OUTPUT_PATH, output)

print(f"\nSaved: {OUTPUT_PATH}")

cv2.imshow("Original", image)
cv2.imshow("Background Replaced", output)

cv2.waitKey(0)
cv2.destroyAllWindows()