"""
Project 02 - Person Segmentation

This project segments people from an
image using YOLOv8 Segmentation.

Concepts:
- Person Segmentation
- Binary Mask
- Background Removal
- Person Extraction
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Load segmentation model
model = YOLO("yolov8n-seg.pt")

PERSON_CLASS = 0

# Load image
image = cv2.imread("people.jfif")

if image is None:
    print("Could not load image.")
    exit()

# Run segmentation
results = model(image)

final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

for result in results:
    if result.masks is None:
        continue

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for mask, cls in zip(masks, classes):
        # Segment only people
        if int(cls) != PERSON_CLASS:
            continue

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        final_mask = cv2.bitwise_or(final_mask, binary_mask)

# Extract people
person = cv2.bitwise_and(image, image, mask=final_mask)

cv2.imshow("Original", image)
cv2.imshow("Mask", final_mask)
cv2.imshow("Person", person)

cv2.waitKey(0)
cv2.destroyAllWindows()