"""
Project 03 - Background Removal

This project removes the background
from an image using YOLOv8 Segmentation.

Concepts:
- Person Segmentation
- Background Removal
- Background Replacement
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Load segmentation model
model = YOLO("yolov8n-seg.pt")

PERSON_CLASS = 0

# Load image
image = cv2.imread("person2.jfif")
image = cv2.resize(image,(500,400))
if image is None:
    print("Could not load image.")
    exit()

# Run segmentation
results = model(image)

mask = np.zeros(image.shape[:2], dtype=np.uint8)

for result in results:
    if result.masks is None:
        continue

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for seg_mask, cls in zip(masks, classes):
        # Segment only people
        if int(cls) != PERSON_CLASS:
            continue

        seg_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]))
        binary_mask = (seg_mask > 0.5).astype(np.uint8) * 255

        mask = cv2.bitwise_or(mask, binary_mask)

# Extract foreground
foreground = cv2.bitwise_and(image, image, mask=mask)

# Create white background
background = np.full_like(image, 255)

inverse_mask = cv2.bitwise_not(mask)

background = cv2.bitwise_and(
    background,
    background,
    mask=inverse_mask
)

# Combine foreground and background
final = cv2.add(foreground, background)

cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Background Removed", final)

cv2.waitKey(0)
cv2.destroyAllWindows()