!pip install ultralytics opencv-python roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="INyfgiPno2EDwsc7Gw9X")
project = rf.workspace("testworkspace-qz0ne").project("fire_detection-nhru2")
dataset = project.version(1).download("yolov8")
yolo detect train model=yolov8n.pt data=Fire-Smoke-Detection/data.yaml epochs=50 imgsz=640
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, etc.
model.train(data="/content/fire_detection-1/data.yaml", epochs=50, imgsz=640)

from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("/content/runs/detect/train2/weights/best.pt")  # Update if needed

# Load video instead of webcam
video_path = "fire_video.mp4"  # Replace with your video file name
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw predictions
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Fire & Smoke Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Load trained model
model = YOLO("/content/runs/detect/train2/weights/best.pt")

# Load video
video_path = "/content/vecteezy_fire-torch-burning_11996735.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw predictions
    annotated_frame = results[0].plot()

    # Convert BGR (OpenCV) to RGB (matplotlib)
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display frame using matplotlib
    plt.imshow(annotated_frame_rgb)
    plt.axis('off')
    display(plt.gcf())
    clear_output(wait=True)
    plt.pause(0.001)

cap.release()
plt.close()


