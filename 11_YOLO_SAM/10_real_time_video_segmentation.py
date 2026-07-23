"""
Project 10 - Optimized Video Object Segmentation
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

VIDEO_PATH = "traffic2.mp4"
OUTPUT_PATH = "outputs/segmented_video.mp4"

YOLO_MODEL = "yolov8n.pt"

SAM_MODEL = "vit_b"
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"

CONFIDENCE = 0.30

FRAME_WIDTH = 960

MAX_OBJECTS = 5

ALLOWED_CLASSES = {
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
}

os.makedirs("outputs", exist_ok=True)

print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading SAM...")
sam = sam_model_registry[SAM_MODEL](checkpoint=SAM_CHECKPOINT)
predictor = SamPredictor(sam)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Unable to read video")

scale = FRAME_WIDTH / frame.shape[1]
height = int(frame.shape[0] * scale)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (FRAME_WIDTH, height),
)

np.random.seed(42)

frame_count = 0

print("Processing...\n")

# ------------------------------------------------
# Main Loop
# ------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    frame = cv2.resize(frame, (FRAME_WIDTH, height))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    predictor.set_image(rgb)

    results = yolo(frame, verbose=False)

    output = frame.copy()

    detections = []

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for box, cls, score in zip(boxes, classes, scores):

            if score < CONFIDENCE:
                continue

            class_name = yolo.names[int(cls)]

            if class_name not in ALLOWED_CLASSES:
                continue

            x1, y1, x2, y2 = box.astype(int)

            area = (x2 - x1) * (y2 - y1)

            detections.append(
                (area, box, cls, score)
            )

    detections.sort(reverse=True, key=lambda x: x[0])

    detections = detections[:MAX_OBJECTS]

    for _, box, cls, score in detections:

        x1, y1, x2, y2 = box.astype(int)

        masks, _, _ = predictor.predict(
            box=np.array(box),
            multimask_output=False,
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
            0,
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color.tolist(),
            2,
        )

        label = f"{yolo.names[int(cls)]} {score:.2f}"

        cv2.putText(
            output,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color.tolist(),
            2,
        )

    writer.write(output)

    cv2.imshow("YOLO + SAM", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()

cv2.destroyAllWindows()

print(f"\nFrames: {frame_count}")
print(f"Saved: {OUTPUT_PATH}")