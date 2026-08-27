from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "town.mp4."
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Object Detection Started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    # Run detection
    results = model(frame)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Resize to fit screen
    scale_percent = 50
    
    resized_frame = cv2.resize(annotated_frame, (800, 500))

    # Show the frame
    cv2.imshow("Object Detection - YOLOv8", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection finished.")