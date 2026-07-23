"""
Project 08 - Mask Dataset Generator

This project detects objects using YOLO,
segments them with Segment Anything (SAM),
and saves each mask as both PNG and
NumPy formats.

Concepts:
- YOLO Detection
- Segment Anything (SAM)
- Mask Generation
- Dataset Creation
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Configuration
IMAGE_PATH = "images/street.jfif"

MASK_FOLDER = "outputs/masks"
NPY_FOLDER = "outputs/npy"
OVERLAY_PATH = "outputs/overlay.jpg"

YOLO_MODEL = "yolov8n.pt"

SAM_MODEL = "vit_b"
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"

os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(NPY_FOLDER, exist_ok=True)

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

output = image.copy()

np.random.seed(42)
object_count = {}

# Run YOLO detection
print("Running YOLO...")

results = yolo(image)

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        class_name = yolo.names[int(cls)]

        object_count[class_name] = object_count.get(class_name, 0) + 1
        index = object_count[class_name]

        masks, _, _ = predictor.predict(
            box=np.array(box),
            multimask_output=False
        )

        mask = masks[0]

        # Save PNG mask
        mask_image = (mask.astype(np.uint8) * 255)

        png_file = f"{class_name}_{index}.png"
        cv2.imwrite(
            os.path.join(MASK_FOLDER, png_file),
            mask_image
        )

        # Save NumPy mask
        npy_file = f"{class_name}_{index}.npy"
        np.save(
            os.path.join(NPY_FOLDER, npy_file),
            mask
        )

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

        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color.tolist(),
            2
        )

        cv2.putText(
            output,
            class_name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color.tolist(),
            2
        )

        print(f"Saved {png_file}")

cv2.imwrite(OVERLAY_PATH, output)

print("\nFinished!")
print(f"Overlay saved to: {OVERLAY_PATH}")