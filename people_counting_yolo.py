import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
image = "people.jpg"
results = model(image)
annotated_frame = results[0].plot() #draw boxes

#count people
people_count= 0
for box in results[0].boxes: # list of bounding box detected
    clas_id = int(box.cls[0]) #class id (0 for person)
    if model.names[clas_id] == "person":
        people_count+=1

cv2.putText(annotated_frame,
            f"People Count:{people_count}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),2)
cv2.imshow("People counting",annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
