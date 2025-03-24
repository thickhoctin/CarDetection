from ultralytics import YOLO
import cv2
import math
import json
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# Path to the JSON file
file_path = "yolo_classes.json"

# Open and read the JSON file
try:
    with open(file_path, "r") as file:
        data = json.load(file)
        class_mapping = data.get("class", {})
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    # Create a simple class mapping if file not found
    class_mapping = {5: "bus", 1: "bicycle", 2: "car", 3: "motorcycle", 7: "truck"}
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from '{file_path}'.")
    class_mapping = {5: "bus", 1: "bicycle", 2: "car", 3: "motorcycle", 7: "truck"}


# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available. Using CPU.")

# Initialize YOLO model - use a valid model file
try:
    model = YOLO("Yolo-Weights/yolo11n.pt")
except Exception as e:
    print(f"Error loading yolo11l.pt: {e}")
    print("Falling back to YOLOv8n")
    model = YOLO("yolov8n.pt")  # Fall back to standard model

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, nn_budget=70, nms_max_overlap=0.5,
                   embedder="mobilenet", half=True)

# For video capture
input_video = "Video/cars.mp4"
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video file. Check the path.")
    exit()

# Frame rate for velocity calculation
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1 / fps  # Time interval between frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


#Create output video path
output_dir = "Output"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "tracked_" + os.path.basename(input_video))


#Init video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))



# Helper function to calculate bounding box center
def calculate_center(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


# Store previous centers for velocity calculation
previous_centers = {}

#Progress
frame_count = 0
while True:
    success, img = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break

    # Update progress
    frame_count += 1
    if frame_count % 30 == 0:  # Show progress every 30 frames
        progress = (frame_count / total_frames) * 100
        print(f"Processing: {progress:.1f}% ({frame_count}/{total_frames} frames)")

    # Run detection
    results = model(img, stream=True)
    detections = []

    for r in results:
        for box in r.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = float(box.conf[0])

            # Class name
            cls = int(box.cls[0])

            # Filter by confidence and class if needed
            if conf > 0.3:  # Increased confidence threshold for better results
                # Format for DeepSORT: DeepSORT expects detections in format [[x1,y1,w,h], confidence, feature]
                # We'll set feature to None and let DeepSORT compute it
                detections.append(([x1, y1, w, h], conf, cls))

    # Update tracks with properly formatted detections
    tracks = tracker.update_tracks(detections, frame=img)

    # Process and display tracks
    for track in tracks:
        if not track.is_confirmed():
            continue

        # Get track info - DeepSORT returns ltrb (left, top, right, bottom)
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Returns left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)  # These are already in the correct format

        # Calculate the bounding box center
        current_center = calculate_center(x1, y1, x2, y2)

        # Get class name if available
        cls_id = None
        if track.det_class is not None:
            cls_id = int(track.det_class)
        class_name = class_mapping.get(str(cls_id), "unknown") if cls_id is not None else "unknown"

        # Velocity calculation
        velocity_kmh = 0
        if track_id in previous_centers:
            # Calculate distance traveled
            previous_center = previous_centers[track_id]
            distance = math.sqrt((current_center[0] - previous_center[0]) ** 2 +
                                 (current_center[1] - previous_center[1]) ** 2)

            # Calculate velocity (with more realistic scaling)
            # Adjust the scaling factor based on your specific scenario
            velocity = distance / frame_time  # Pixels/second
            velocity_kmh = (velocity * 0.027) * 3.6  # Convert to km/h (adjust scaling factor as needed)

        # Save the current center for the next calculation
        previous_centers[track_id] = current_center

        # Display bounding box and information
        color = (0, 255, 0) if velocity_kmh < 50 else (0, 165, 255) if velocity_kmh < 80 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Display information
        info_text = f'ID: {track_id} | {class_name} | {velocity_kmh:.1f} km/h'
        cv2.putText(img, info_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    #Write the frame
    out.write(img)

    # Display the image
    cv2.imshow("YOLO + DeepSORT", img)

    # Break the loop on 'q' press
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()