from ultralytics import YOLO
import cv2



model = YOLO('Yolo-Weights/yolo11n.pt')
result = model("Images/1.jpeg",show = True)
cv2.waitKey(0)