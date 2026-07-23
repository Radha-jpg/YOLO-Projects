"""
Project 13 - Auto Segmentation Annotation

This project detects objects using YOLO
and generates segmentation masks using
Segment Anything (SAM). It saves the
masks and annotation data for each image.

Concepts:
- YOLO Detection
- Segment Anything (SAM)
- Auto Annotation
- Dataset Generation
"""

import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Configuration
IMAGE_FOLDER = "images"
MASK_FOLDER = "masks"
ANNOTATION_FOLDER = "annotations"

YOLO_MODEL = "yolov8n.pt"

SAM_MODEL = "vit_b"
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"

os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)

# Load models
print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading SAM...")
sam = sam_model_registry[SAM_MODEL](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)

# Read all images
image_files = [
    file for file in os.listdir(IMAGE_FOLDER)
    if file.lower().endswith((".jpg", ".jpeg", ".png","jfif"))
]

print(f"Found {len(image_files)} images.\n")

for image_name in image_files:

    image_path = os.path.join(IMAGE_FOLDER, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {image_name}")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    print(f"Processing {image_name}")

    results = yolo(image)

    annotations = []

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):

            masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False
            )

            mask = masks[0]
            mask_image = mask.astype(np.uint8) * 255

            mask_name = f"{os.path.splitext(image_name)[0]}_{idx}.png"

            cv2.imwrite(
                os.path.join(MASK_FOLDER, mask_name),
                mask_image
            )

            contours, _ = cv2.findContours(
                mask_image,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            polygons = []

            for contour in contours:

                contour = contour.squeeze()

                if len(contour.shape) != 2:
                    continue

                polygon = contour.flatten().tolist()

                if len(polygon) >= 6:
                    polygons.append(polygon)

            annotations.append({
                "class": yolo.names[int(cls)],
                "confidence": float(score),
                "bbox": box.tolist(),
                "mask": mask_name,
                "segmentation": polygons
            })

    json_path = os.path.join(
        ANNOTATION_FOLDER,
        os.path.splitext(image_name)[0] + ".json"
    )

    with open(json_path, "w") as file:
        json.dump(annotations, file, indent=4)

    print(f"Saved {json_path}")

print("\nDataset generation complete!")