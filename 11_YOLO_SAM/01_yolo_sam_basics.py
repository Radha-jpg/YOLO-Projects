"""
Project 09 - YOLO + Segment Anything

This project uses YOLO to detect objects
and Segment Anything (SAM) to generate
high-quality segmentation masks.

Concepts:
- YOLO Object Detection
- Segment Anything (SAM)
- Instance Segmentation
- Mask Overlay
"""

import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Load YOLO model
yolo = YOLO("yolov8n.pt")

# Load SAM model
sam = sam_model_registry["vit_b"](
    checkpoint="models/sam_vit_b_01ec64.pth"
)

predictor = SamPredictor(sam)

# Load image
image = cv2.imread("person.jpg")

if image is None:
    print("Could not load image.")
    exit()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(rgb)

# Run YOLO detection
results = yolo(image)

output = image.copy()

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

        color = np.random.randint(0, 255, 3)

        overlay = output.copy()
        overlay[mask] = color

        output = cv2.addWeighted(
            overlay,
            0.45,
            output,
            0.55,
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
            f"{yolo.names[int(cls)]} {score:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color.tolist(),
            2
        )

cv2.imshow("YOLO + SAM", output)
cv2.waitKey(0)
cv2.destroyAllWindows()