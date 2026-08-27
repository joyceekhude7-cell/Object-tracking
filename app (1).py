import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.set_page_config(page_title="Object Detection & Tracking", layout="wide")

st.title("Object Detection and Tracking")
st.markdown("### CodeAlpha Internship Project")
st.write("Upload a video or use the default video to detect and track objects using YOLOv8.")

# Load model (only once)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Video selection
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

use_default = st.checkbox("Use default video (town.mp4)", value=True)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
elif use_default and os.path.exists("town.mp4"):
    video_path = "town.mp4"
else:
    st.warning("Please upload a video or make sure town.mp4 is in the folder.")
    st.stop()

# Start button
if st.button("Start Detection & Tracking"):
    stframe = st.empty()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Could not open video.")
        st.stop()

    st.info("Processing... Press Stop in terminal or close the browser tab to end.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking
        results = model.track(frame, persist=True, verbose=False)

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

                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Convert BGR to RGB for Streamlit
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Resize for better display
        annotated_frame = cv2.resize(annotated_frame, (900, 550))

        stframe.image(annotated_frame, channels="RGB")

    cap.release()
    st.success("Processing finished!")