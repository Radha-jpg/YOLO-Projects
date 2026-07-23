from ultralytics import YOLO
import cv2
#YOLOv8 pretrained on COCO dataset
model = YOLO("yolov8n.pt")
image = "park.jpg"
results = model(image)
results[0].show()