from ultralytics import YOLO
import cv2
import cvzone
import math
import json


# Path to the JSON file
file_path = "yolo_classes.json"

# Open and read the JSON file
try:
    with open(file_path, "r") as file:
        data = json.load(file)  # Parse the JSON file into a Python dictionary or list
        print("Contents of yolo_classes.json:")
        #print(data)
        class_mapping = data.get("class", {})  # Retrieve the "class" dictionary from the JSON file

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from '{file_path}'.")

# #For Webcam
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 720)


#For video
# cap = cv2.VideoCapture("Video/cars.mp4")

model =YOLO("Yolo-Weights/yolo11n.pt")

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to read from camera.")
        break

    result = model(img, stream = True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            #Bounding box
            #using normal rectangle
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1
            print(x1, y1, w, h)
            cvzone.cornerRect(img, (x1,y1,w,h))
            #Confident
            conf = math.ceil(box.conf[0]*100) / 100

            #Class name
            cls = int(box.cls[0])
            # Look up the class name from the JSON data using the class ID
            # Use .get() to handle missing keys gracefully
            class_name = class_mapping.get(str(cls), "Unknown")

            cvzone.putTextRect(img, f'{class_name}{conf}', (max(0, x1), max(40, y1 - 20)))

    cv2.imshow("Image", img)
    cv2.waitKey(1)