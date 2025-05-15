TITLE: Fire Detection Using YOLOv8 Required Libraries: ultralytics For loading and running YOLOv8 model (detection). opencv-python For handling video input and image processing (cv2). matplotlib To display annotated frames (since GUI is unsupported in notebooks). IPython.display For live frame updates inside notebook using display() and clear_output(). Installation (Run this first): pip install ultralytics opencv-python matplotlib Fire Detection Code (Video Upload + Display in Notebook): from ultralytics import YOLO import cv2 import matplotlib.pyplot as plt from IPython.display import display, clear_output

Load trained model
model = YOLO("/content/runs/detect/train2/weights/best.pt")

Load video
video_path = "/content/vecteezy_fire-torch-burning_11996735.mp4" cap = cv2.VideoCapture(video_path)

while True: ret, frame = cap.read() if not ret: break

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
cap.release() plt.close() WORK: YOLOv8 loads a custom-trained model (on fire/smoke data). A video is processed frame-by-frame. Fire/smoke regions are detected and visualized. In notebooks, output is shown using matplotlib since OpenCV GUI doesn’t work. FUTURE ENHANCEMENT: Add Alert System Play an alarm or send email/SMS when fire is detected. Real-Time Deployment Use a webcam with Raspberry Pi or Jetson Nano to deploy on hardware. Save Output Video Save annotated frames into a new .mp4 video file. Upload to Cloud / Web Dashboard Stream results to a cloud dashboard or IoT platform for remote monitoring. Smoke vs. Fire Classification Train YOLO model with different classes (fire, smoke) to differentiate. Flame Area Estimation Add code to calculate the area or intensity of fire based on bounding box size.

CONCLUSION: This project demonstrates a simple yet powerful application of YOLOv8 for fire detection using video data. With just a few lines of Python code and a custom-trained model, we can detect and localize fire in real-time or recorded footage. This has critical applications in public safety, surveillance, disaster response, and smart city systems.

By extending this base, we can build complete AI-powered fire detection systems with alerting, tracking, and cloud integration.
