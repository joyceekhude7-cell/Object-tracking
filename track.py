from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO("yolov8n.pt")

# Open video
video_path = "town.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Clean Object Tracking Started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Run tracking
    results = model.track(frame, persist=True, verbose=False)

    # Make a copy of the frame to draw on
    annotated_frame = frame.copy()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = results[0].names

        for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
            x1, y1, x2, y2 = box
            label = f"ID:{track_id} {class_names[cls_id]} {conf:.2f}"

            # Draw clean rectangle
            color = (0, 255, 0)  # Green color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + text_width, y1), color, -1)

            # Put clean text
            cv2.putText(annotated_frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Resize for better view
    resized_frame = cv2.resize(annotated_frame, (1000, 600))

    cv2.imshow("Clean Object Tracking", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Tracking finished.")