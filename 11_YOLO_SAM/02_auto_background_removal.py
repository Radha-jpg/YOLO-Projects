"""
Project 10 - Person Background Removal

This project uses YOLO to detect people
and Segment Anything (SAM) to create
high-quality segmentation masks for
background removal.

Concepts:
- YOLO Detection
- Segment Anything (SAM)
- Background Removal
- Transparent PNG
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Configuration
IMAGE_PATH = "person2.jfif"
OUTPUT_DIR = "outputs"

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
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(rgb)

# Detect people
print("Running YOLO detection...")

results = yolo(image)

person_boxes = []

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if int(cls) == 0:
            person_boxes.append(box.astype(int))

print(f"People detected: {len(person_boxes)}")

if not person_boxes:
    print("No people detected.")
    exit()

# Generate segmentation masks
combined_mask = np.zeros(image.shape[:2], dtype=bool)

print("Generating SAM masks...")

for box in person_boxes:
    masks, _, _ = predictor.predict(
        box=box,
        multimask_output=False
    )

    combined_mask |= masks[0]

binary_mask = (combined_mask.astype(np.uint8)) * 255

# Save mask
cv2.imwrite(
    os.path.join(OUTPUT_DIR, "mask.png"),
    binary_mask
)

# Save transparent PNG
rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = binary_mask

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "transparent.png"),
    rgba
)

# White background
white_background = np.full_like(image, 255)
white_background[combined_mask] = image[combined_mask]

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "white_background.png"),
    white_background
)

# Preview
overlay = image.copy()

overlay[combined_mask] = (
    0.4 * overlay[combined_mask] +
    0.6 * np.array([0, 255, 0])
).astype(np.uint8)

cv2.imshow("Original", image)
cv2.imshow("Mask", binary_mask)
cv2.imshow("Overlay", overlay)
cv2.imshow("White Background", white_background)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nSaved Files")
print("----------------")
print("mask.png")
print("transparent.png")
print("white_background.png")