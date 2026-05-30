``markdown

# Courtlytics

## Introduction
Courtlytics is a comprehensive tennis analysis application that analyzes tennis players in videos to measure their speed, ball shot speed, and number of shots. This project detects players and the tennis ball using YOLO and utilizes CNNs to extract court keypoints. 

## Features
- Real-time player and ball detection
- Tennis court keypoint extraction
- Player performance analytics
- Shot speed measurement
- Live video processing with Streamlit web interface
- Detailed statistics dashboard

## Output Videos
The analysis produces annotated videos with player tracking, court visualization, and real-time statistics overlay.

## Models Used
* YOLO v8 for player detection
* Fine-tuned YOLO for tennis ball detection
* CNN for court keypoint extraction

**Pre-trained Models:**
* YOLOV5 model: https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing
* Tennis court keypoint model: https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing

## Training
* Tennis ball detector: 	raining/tennis_ball_detector_training.ipynb
* Tennis court keypoints: 	raining/tennis_court_keypoints_training.ipynb

## Requirements
* Python 3.8+
* ultralytics
* pytorch
* pandas
* numpy
* opencv-python
* streamlit

## Installation

1. Clone the repository:
\\\ash
git clone https://github.com/jashkarnsingh1005/Courtlytics.git
cd Courtlytics
\\\

2. Install dependencies:
\\\ash
pip install -r requirements.txt
\\\

3. Download pre-trained models and place them in the models/ directory

## Usage

### Web Interface
\\\ash
streamlit run app.py
\\\

The application will open at http://localhost:8501

## How It Works

1. **Upload Video**: Select a tennis match video (MP4, AVI, MOV, MKV)
2. **Start Processing**: Click "Start Processing" to analyze the video
3. **Live Analysis**: Watch real-time video processing with player detection and statistics
4. **View Results**: See the annotated output video and detailed performance analytics

## Project Structure
- \pp.py\ - Streamlit web interface
- \main.py\ - Main analysis pipeline
- \yolo_inference.py\ - YOLO model inference utilities
- \court_line_detector/\ - Court keypoint detection module
- \mini_court/\ - Mini court visualization
- \	rackers/\ - Player and ball tracking
- \utils/\ - Utility functions for analysis
- \	raining/\ - Model training notebooks

## License
This project is open source and available for educational and research purposes.
