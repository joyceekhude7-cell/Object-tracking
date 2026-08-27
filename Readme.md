
# Object Detection and Tracking

This project was developed as part of my internship at **CodeAlpha**.

It performs Object Detection and Tracking on a video using **YOLOv8** and also has a clean web interface built with **Streamlit**.

## Features
- Object Detection using YOLOv8
- Object Tracking with unique IDs
- Clean bounding boxes and labels
- Works with video files
- Web interface using Streamlit

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy

## How to Run

### Method 1: OpenCV Version
1. Install the required libraries:
```bash
pip install -r requirements.txt

2. Run the tracking script:
```bash
python track.py

3. Press `q` to quit.

### Method 2: Streamlit Web Application.
1. Install the required libraries:
```bash
pip install -r requirements.txt

2. Run this command:
```bash
streamlit run app.py

3. A link will appear in the terminal (example: http://localhost:8501)
4. Press `Ctrl` and click on the link to open it in browser.

## Project Structure
- `track.py` → OpenCV version with tracking
- `app.py` → Streamlit web version
- `detect.py` → Basic detection version
- `town.mp4` → Sample video
- `requirements.txt` → Required libraries
- `README.md` → Project information

## Author Name
Joycee Khude. 
CodeAlpha Intern 
