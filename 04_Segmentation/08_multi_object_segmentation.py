"""
Project 08 - Multi Object Segmentation

This project segments every detected
object using a unique color.

Concepts:
- Instance Segmentation
- Random Colors
- Mask Overlay
- Class Labels
"""

import cv2
import numpy as np
import random
from ultralytics import YOLO

# Load segmentation model
model = YOLO("yolov8n-seg.pt")

# Load image
image = cv2.imread("fruits_veg.jfif")

if image is None:
    print("Could not load image.")
    exit()

# Run segmentation
results = model(image)

output = image.copy()

for result in results:
    if result.masks is None:
        continue

    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for mask, box, cls, conf in zip(masks, boxes, classes, confidences):
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        binary_mask = mask > 0.5

        # Generate a random color
        color = np.array([
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ], dtype=np.uint8)

        # Create colored mask
        colored_mask = np.zeros_like(output)
        colored_mask[binary_mask] = color

        output = cv2.addWeighted(
            output,
            1,
            colored_mask,
            0.5,
            0
        )

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color.tolist(),
            2
        )

        cv2.putText(
            output,
            f"{model.names[int(cls)]} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color.tolist(),
            2
        )

cv2.imshow("Original", image)
cv2.imshow("Multi Object Segmentation", output)

cv2.waitKey(0)
cv2.destroyAllWindows()