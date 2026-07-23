"""
Project 01 - Image Segmentation

This project performs image segmentation
using a YOLOv8 segmentation model.

Concepts:
- YOLOv8 Segmentation
- Segmentation Masks
- Mask Overlay
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Load segmentation model
model = YOLO("yolov8n-seg.pt")

# Load image
image = cv2.imread("street2.jfif")
# image = cv2.resize(image , (400,400))
if image is None:
    print("Could not load image.")
    exit()

# Run segmentation
results = model(image)

for result in results:
    if result.masks is None:
        continue

    for mask in result.masks.data.cpu().numpy():
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)

        image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

cv2.imshow("Image Segmentation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()