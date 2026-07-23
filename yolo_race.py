import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if not ret:
        break
    results = model(frame) #run yolo on each frame
    annotated_frame = results[0].plot() #draw boxes
    cv2.imshow("YOLO v8 Detection",annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()