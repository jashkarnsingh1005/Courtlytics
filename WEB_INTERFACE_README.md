# Running the Web Interface

The Courtlytics system now has a Streamlit web interface that allows you to:
- Upload tennis videos
- Start/Stop video processing
- View live video processing in real-time at the same speed as the input video
- See live player statistics as the video plays

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Open a terminal/PowerShell in the project directory

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. A web browser will automatically open at `http://localhost:8501`

## How to Use

1. **Upload Video**: In the sidebar on the left, click "Upload a tennis video" and select your video file (supports mp4, avi, mov, mkv)

2. **Start Processing**: Click the "Start Processing" button to begin analysis

3. **Stop Processing**: Click "Stop Processing" to stop the video at any time

4. **Live View**: The video will process frame-by-frame in real-time at the same speed as your input video, showing:
   - Player bounding boxes
   - Ball detection and tracking
   - Court keypoints
   - Mini court visualization
   - Live player statistics (shots, shot speeds)

5. **Output**: The processed video is automatically saved to `output_videos/output_video.avi`

## Notes

- The first time you run it, it will take longer as models are being loaded
- Processing time depends on video length and your hardware
- Player detections and ball detections can be read from stubs if they exist (will speed up processing)
- The original `main.py` script remains unchanged for batch processing if needed
