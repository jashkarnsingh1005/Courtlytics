import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from utils import (measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import pandas as pd
from copy import deepcopy
import time
import pickle
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Courtlytics", layout="wide")



st.title("🎾 Courtlytics")

# Initialize session state
if 'video_uploaded' not in st.session_state:
    st.session_state.video_uploaded = False
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False
if 'stop_processing' not in st.session_state:
    st.session_state.stop_processing = False
if 'temp_video_path' not in st.session_state:
    st.session_state.temp_video_path = None
if 'player_stats_df' not in st.session_state:
    st.session_state.player_stats_df = None
if 'output_video_ready' not in st.session_state:
    st.session_state.output_video_ready = False
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = 'All Players'
if 'analysis_available' not in st.session_state:
    st.session_state.analysis_available = False

# Function to display player analysis
def display_player_analysis(df):
    """Display comprehensive player analysis with charts and statistics"""
    st.divider()
    st.markdown("## 📊 Player Analysis Dashboard")
    
    # Filter by player
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.write("**Filter by Player:**")
    with col2:
        player_filter = st.selectbox("", ["All Players", "Player 1", "Player 2"], key="player_select")
        st.session_state.selected_player = player_filter
    
    st.divider()
    
    # Calculate summary statistics - use final row values, not means across all rows
    final_row = df.iloc[-1]
    
    p1_shots = int(final_row['player_1_number_of_shots'])
    # For average shot speed, calculate from totals
    if p1_shots > 0:
        p1_avg_speed = final_row['player_1_total_shot_speed'] / p1_shots
    else:
        p1_avg_speed = 0
    p1_max_speed = df['player_1_last_shot_speed'].max()
    if p1_shots > 0:
        p1_avg_player_speed = final_row['player_1_total_player_speed'] / p1_shots
    else:
        p1_avg_player_speed = 0
    
    p2_shots = int(final_row['player_2_number_of_shots'])
    # For average shot speed, calculate from totals
    if p2_shots > 0:
        p2_avg_speed = final_row['player_2_total_shot_speed'] / p2_shots
    else:
        p2_avg_speed = 0
    p2_max_speed = df['player_2_last_shot_speed'].max()
    if p2_shots > 0:
        p2_avg_player_speed = final_row['player_2_total_player_speed'] / p2_shots
    else:
        p2_avg_player_speed = 0
    
    # Show summaries based on filter
    if player_filter == "All Players":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎾 Player 1 Summary")
            col1_1, col1_2, col1_3, col1_4 = st.columns(4)
            with col1_1:
                st.metric("🎾 Total Shots", p1_shots, delta=None)
            with col1_2:
                st.metric("⚡ Avg Shot Speed", f"{p1_avg_speed:.1f}" if p1_avg_speed > 0 else "N/A", delta="km/h")
            with col1_3:
                st.metric("🔥 Max Speed", f"{p1_max_speed:.1f}" if p1_max_speed > 0 else "N/A", delta="km/h")
            with col1_4:
                st.metric("📍 Avg Move Speed", f"{p1_avg_player_speed:.1f}" if p1_avg_player_speed > 0 else "N/A", delta="km/h")
        
        with col2:
            st.markdown("### 🎾 Player 2 Summary")
            col2_1, col2_2, col2_3, col2_4 = st.columns(4)
            with col2_1:
                st.metric("🎾 Total Shots", p2_shots, delta=None)
            with col2_2:
                st.metric("⚡ Avg Shot Speed", f"{p2_avg_speed:.1f}" if p2_avg_speed > 0 else "N/A", delta="km/h")
            with col2_3:
                st.metric("🔥 Max Speed", f"{p2_max_speed:.1f}" if p2_max_speed > 0 else "N/A", delta="km/h")
            with col2_4:
                st.metric("📍 Avg Move Speed", f"{p2_avg_player_speed:.1f}" if p2_avg_player_speed > 0 else "N/A", delta="km/h")
    
    elif player_filter == "Player 1":
        st.markdown("### 🎾 Player 1 Summary")
        col1_1, col1_2, col1_3, col1_4 = st.columns(4)
        with col1_1:
            st.metric("🎾 Total Shots", p1_shots, delta=None)
        with col1_2:
            st.metric("⚡ Avg Shot Speed", f"{p1_avg_speed:.1f}" if p1_avg_speed > 0 else "N/A", delta="km/h")
        with col1_3:
            st.metric("🔥 Max Speed", f"{p1_max_speed:.1f}" if p1_max_speed > 0 else "N/A", delta="km/h")
        with col1_4:
            st.metric("📍 Avg Move Speed", f"{p1_avg_player_speed:.1f}" if p1_avg_player_speed > 0 else "N/A", delta="km/h")
    
    elif player_filter == "Player 2":
        st.markdown("### 🎾 Player 2 Summary")
        col2_1, col2_2, col2_3, col2_4 = st.columns(4)
        with col2_1:
            st.metric("🎾 Total Shots", p2_shots, delta=None)
        with col2_2:
            st.metric("⚡ Avg Shot Speed", f"{p2_avg_speed:.1f}" if p2_avg_speed > 0 else "N/A", delta="km/h")
        with col2_3:
            st.metric("🔥 Max Speed", f"{p2_max_speed:.1f}" if p2_max_speed > 0 else "N/A", delta="km/h")
        with col2_4:
            st.metric("📍 Avg Move Speed", f"{p2_avg_player_speed:.1f}" if p2_avg_player_speed > 0 else "N/A", delta="km/h")
    
    st.divider()
    
    # Charts
    if player_filter == "All Players" or player_filter == "Player 1":
        st.markdown("### 📈 Player 1 Performance Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df['frame_num'], df['player_1_last_shot_speed'], label='Shot Speed', color='#1f77b4', linewidth=2.5)
            ax1.fill_between(df['frame_num'], df['player_1_last_shot_speed'], alpha=0.3, color='#1f77b4')
            ax1.set_xlabel('Frame Number', fontsize=11)
            ax1.set_ylabel('Shot Speed (km/h)', fontsize=11)
            ax1.set_title('Shot Speed Over Time', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
        
        with col2:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(df['frame_num'], df['player_1_number_of_shots'], label='Total Shots', color='#ff7f0e', linewidth=2.5)
            ax2.fill_between(df['frame_num'], df['player_1_number_of_shots'], alpha=0.3, color='#ff7f0e')
            ax2.set_xlabel('Frame Number', fontsize=11)
            ax2.set_ylabel('Number of Shots', fontsize=11)
            ax2.set_title('Cumulative Shots Over Time', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
    
    if player_filter == "All Players" or player_filter == "Player 2":
        st.markdown("### 📈 Player 2 Performance Charts")
        col3, col4 = st.columns(2)
        
        with col3:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(df['frame_num'], df['player_2_last_shot_speed'], label='Shot Speed', color='#2ca02c', linewidth=2.5)
            ax3.fill_between(df['frame_num'], df['player_2_last_shot_speed'], alpha=0.3, color='#2ca02c')
            ax3.set_xlabel('Frame Number', fontsize=11)
            ax3.set_ylabel('Shot Speed (km/h)', fontsize=11)
            ax3.set_title('Shot Speed Over Time', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left')
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
        
        with col4:
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            ax4.plot(df['frame_num'], df['player_2_number_of_shots'], label='Total Shots', color='#d62728', linewidth=2.5)
            ax4.fill_between(df['frame_num'], df['player_2_number_of_shots'], alpha=0.3, color='#d62728')
            ax4.set_xlabel('Frame Number', fontsize=11)
            ax4.set_ylabel('Number of Shots', fontsize=11)
            ax4.set_title('Cumulative Shots Over Time', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='upper left')
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
    
    st.divider()
    
    # Comparison chart - only show if both players selected
    if player_filter == "All Players":
        st.markdown("### ⚖️ Head to Head Comparison")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            st.markdown("#### 🎾 Shots Comparison")
            fig_shots, ax_shots = plt.subplots(figsize=(8, 5))
            players = ['Player 1', 'Player 2']
            shots = [p1_shots, p2_shots]
            colors_shots = ['#1f77b4', '#2ca02c']
            bars = ax_shots.bar(players, shots, color=colors_shots, alpha=0.85, edgecolor='black', linewidth=2)
            ax_shots.set_ylabel('Number of Shots', fontsize=11)
            ax_shots.set_title('Total Shots Comparison', fontsize=13, fontweight='bold')
            ax_shots.grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for bar, shot in zip(bars, shots):
                height = bar.get_height()
                ax_shots.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(shot)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_shots, use_container_width=True)
        
        with col_comp2:
            st.markdown("#### ⚡ Average Shot Speed Comparison")
            fig_speed, ax_speed = plt.subplots(figsize=(8, 5))
            players = ['Player 1', 'Player 2']
            avg_speeds = [
                p1_avg_speed if not np.isnan(p1_avg_speed) else 0,
                p2_avg_speed if not np.isnan(p2_avg_speed) else 0
            ]
            colors_speed = ['#1f77b4', '#2ca02c']
            bars = ax_speed.bar(players, avg_speeds, color=colors_speed, alpha=0.85, edgecolor='black', linewidth=2)
            ax_speed.set_ylabel('Average Speed (km/h)', fontsize=11)
            ax_speed.set_title('Average Shot Speed Comparison', fontsize=13, fontweight='bold')
            ax_speed.grid(True, alpha=0.3, axis='y')
            # Add value labels on bars
            for bar, speed in zip(bars, avg_speeds):
                height = bar.get_height()
                ax_speed.text(bar.get_x() + bar.get_width()/2., height,
                            f'{speed:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig_speed, use_container_width=True)
        
        st.divider()
    
    # Detailed stats table
    st.markdown("### 📋 Detailed Statistics Table")
    
    # Create display dataframe with all data
    display_df = df[['frame_num', 'player_1_number_of_shots', 'player_1_last_shot_speed', 
                    'player_2_number_of_shots', 'player_2_last_shot_speed']].copy()
    display_df.columns = ['Frame', 'P1 Total Shots', 'P1 Shot Speed (km/h)', 
                          'P2 Total Shots', 'P2 Shot Speed (km/h)']
    
    # Create two separate tables for Player 1 and Player 2 shots
    if player_filter == "All Players":
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown("**Player 1 Shots**")
            p1_shots_df = display_df[display_df['P1 Shot Speed (km/h)'] > 0][['Frame', 'P1 Total Shots', 'P1 Shot Speed (km/h)']].copy()
            if len(p1_shots_df) > 0:
                st.dataframe(p1_shots_df.round(2), use_container_width=True, hide_index=True)
            else:
                st.info("No Player 1 shots recorded")
        
        with col_p2:
            st.markdown("**Player 2 Shots**")
            p2_shots_df = display_df[display_df['P2 Shot Speed (km/h)'] > 0][['Frame', 'P2 Total Shots', 'P2 Shot Speed (km/h)']].copy()
            if len(p2_shots_df) > 0:
                st.dataframe(p2_shots_df.round(2), use_container_width=True, hide_index=True)
            else:
                st.info("No Player 2 shots recorded")
    
    elif player_filter == "Player 1":
        st.markdown("**Player 1 Shots**")
        p1_shots_df = display_df[display_df['P1 Shot Speed (km/h)'] > 0][['Frame', 'P1 Total Shots', 'P1 Shot Speed (km/h)']].copy()
        if len(p1_shots_df) > 0:
            st.dataframe(p1_shots_df.round(2), use_container_width=True, hide_index=True)
        else:
            st.info("No Player 1 shots recorded")
    
    elif player_filter == "Player 2":
        st.markdown("**Player 2 Shots**")
        p2_shots_df = display_df[display_df['P2 Shot Speed (km/h)'] > 0][['Frame', 'P2 Total Shots', 'P2 Shot Speed (km/h)']].copy()
        if len(p2_shots_df) > 0:
            st.dataframe(p2_shots_df.round(2), use_container_width=True, hide_index=True)
        else:
            st.info("No Player 2 shots recorded")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("📹 Upload a tennis video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        st.session_state.video_uploaded = True
        st.success("✓ Video uploaded successfully!")
    
    st.divider()
    
    # Buttons in full width stack layout
    if st.button("▶️ Start Processing", disabled=not st.session_state.video_uploaded, use_container_width=True, key="start_btn"):
        st.session_state.processing_started = True
        st.session_state.stop_processing = False
    
    if st.button("⏹️ Stop Processing", use_container_width=True, key="stop_btn"):
        st.session_state.stop_processing = True
        st.session_state.processing_started = False
        st.session_state.analysis_available = True
        st.rerun()
    
    if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
        # Delete output video if it exists
        output_video_path = "output_videos/output_video.avi"
        if os.path.exists(output_video_path):
            try:
                os.remove(output_video_path)
                st.success("✓ Output video deleted")
            except Exception as e:
                st.warning(f"Could not delete video: {str(e)}")
        
        # Delete temp video file if it exists
        if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
            try:
                os.remove(st.session_state.temp_video_path)
            except:
                pass
        
        # Reset session state
        st.session_state.video_uploaded = False
        st.session_state.processing_started = False
        st.session_state.stop_processing = False
        st.session_state.temp_video_path = None
        st.session_state.player_stats_df = None
        st.session_state.output_video_ready = False
        st.session_state.analysis_available = False
        st.session_state.selected_player = 'All Players'
        
        st.rerun()

# Main content area
if st.session_state.video_uploaded and uploaded_file:
    # Save uploaded file temporarily
    if st.session_state.temp_video_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.temp_video_path = tmp_file.name
    
    temp_video_path = st.session_state.temp_video_path
    
    # If analysis is available, hide video preview and only show analysis
    if not st.session_state.analysis_available:
        # Display uploaded video preview
        st.markdown("### 📹 Uploaded Video Preview")
        col_preview, col_info = st.columns([3, 1])
        
        with col_preview:
            # Display playable video
            with open(temp_video_path, "rb") as f:
                video_data = f.read()
                st.video(video_data)
            
            # Get video info
            cap = cv2.VideoCapture(temp_video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 24
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        
        with col_info:
            st.markdown("#### 📊 Video Information")
            st.metric("🎬 Frames", f"{total_frames}")
            st.metric("⚡ FPS", fps)
            st.metric("📐 Resolution", f"{frame_width}×{frame_height}")
            duration_sec = total_frames / fps
            st.metric("⏱️ Duration", f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}")
    
    if st.session_state.processing_started and not st.session_state.stop_processing:
        st.divider()
        st.markdown("### 🔄 Processing Status")
        
        # Create containers for messages
        status_container = st.container()
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(temp_video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 24
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Display initialization message
            with status_container:
                st.info("📋 Initializing video and models...")
                time.sleep(0.5)
                st.success(f"✓ Video loaded: {total_frames} frames at {fps} FPS ({frame_width}x{frame_height})")
            
            # Initialize models
            with status_container:
                st.info("🤖 Loading models...")
                player_tracker = PlayerTracker(model_path='yolov8x')
                ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
                court_model_path = "models/keypoints_model.pth"
                court_line_detector = CourtLineDetector(court_model_path)
                st.success("✓ Models loaded successfully!")
            
            # Read all frames into memory
            with status_container:
                st.info("📹 Reading video frames...")
            video_frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(frame)
            cap.release()
            
            if len(video_frames) == 0:
                raise ValueError("Could not read video frames")
            
            with status_container:
                st.success(f"✓ Successfully loaded {len(video_frames)} frames")
            
            # Create placeholders for display
            video_placeholder = st.empty()
            frame_info = st.empty()
            stats_placeholder = st.empty()
            
            # Detect Players and Ball
            progress_bar.progress(10)
            progress_text.text("Stage 1/5: Detecting players and ball...")
            with status_container:
                st.info("Stage 1/5: Detecting players and ball...")
            try:
                player_detections = player_tracker.detect_frames(video_frames,
                                                                 read_from_stub=True,
                                                                 stub_path="tracker_stubs/player_detections.pkl"
                                                                 )
                ball_detections = ball_tracker.detect_frames(video_frames,
                                                            read_from_stub=True,
                                                            stub_path="tracker_stubs/ball_detections.pkl"
                                                            )
                ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
                with status_container:
                    st.success("✓ Detection complete!")
            except Exception as e:
                with status_container:
                    st.warning(f"Detection failed: {str(e)}")
                player_detections = {}
                ball_detections = {}
            
            # Court Line Detection
            progress_bar.progress(25)
            progress_text.text("Stage 2/5: Detecting court keypoints...")
            with status_container:
                st.info("Stage 2/5: Detecting court keypoints...")
            try:
                court_keypoints = court_line_detector.predict(video_frames[0])
                with status_container:
                    st.success("✓ Court keypoints detected!")
            except Exception as e:
                with status_container:
                    st.warning(f"Court detection failed: {str(e)}")
                court_keypoints = None
            
            # Choose and filter players
            if player_detections:
                player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
            
            # MiniCourt setup
            mini_court = MiniCourt(video_frames[0])
            
            # Detect ball shots
            ball_shot_frames = []
            if ball_detections:
                ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
            
            # Convert positions to mini court coordinates
            progress_bar.progress(40)
            progress_text.text("Stage 3/5: Converting coordinates...")
            with status_container:
                st.info("Stage 3/5: Converting coordinates...")
            if player_detections and ball_detections:
                player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
                    player_detections, ball_detections, court_keypoints)
            else:
                player_mini_court_detections = {}
                ball_mini_court_detections = {}
            with status_container:
                st.success("✓ Coordinates converted!")
            
            # Calculate player stats
            progress_bar.progress(50)
            progress_text.text("Stage 4/5: Calculating player statistics...")
            with status_container:
                st.info("Stage 4/5: Calculating player statistics...")
            player_stats_data = [{
                'frame_num': 0,
                'player_1_number_of_shots': 0,
                'player_1_total_shot_speed': 0,
                'player_1_last_shot_speed': 0,
                'player_1_total_player_speed': 0,
                'player_1_last_player_speed': 0,
                'player_2_number_of_shots': 0,
                'player_2_total_shot_speed': 0,
                'player_2_last_shot_speed': 0,
                'player_2_total_player_speed': 0,
                'player_2_last_player_speed': 0,
            }]
            
            if len(ball_shot_frames) > 1:
                for ball_shot_ind in range(len(ball_shot_frames) - 1):
                    if st.session_state.stop_processing:
                        break
                    
                    start_frame = ball_shot_frames[ball_shot_ind]
                    end_frame = ball_shot_frames[ball_shot_ind + 1]
                    ball_shot_time_in_seconds = (end_frame - start_frame) / fps
                    
                    if ball_shot_time_in_seconds == 0:
                        continue
                    
                    # Distance covered by ball
                    distance_covered_by_ball_pixels = measure_distance(
                        ball_mini_court_detections[start_frame][1],
                        ball_mini_court_detections[end_frame][1])
                    distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
                        distance_covered_by_ball_pixels,
                        constants.DOUBLE_LINE_WIDTH,
                        mini_court.get_width_of_mini_court())
                    
                    # Speed of ball shot
                    speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
                    
                    # Player who shot the ball
                    player_positions = player_mini_court_detections[start_frame]
                    if player_positions:
                        player_shot_ball = min(player_positions.keys(),
                                              key=lambda player_id: measure_distance(
                                                  player_positions[player_id],
                                                  ball_mini_court_detections[start_frame][1]))
                        
                        # Opponent speed
                        opponent_player_id = 1 if player_shot_ball == 2 else 2
                        distance_covered_by_opponent_pixels = measure_distance(
                            player_mini_court_detections[start_frame][opponent_player_id],
                            player_mini_court_detections[end_frame][opponent_player_id])
                        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
                            distance_covered_by_opponent_pixels,
                            constants.DOUBLE_LINE_WIDTH,
                            mini_court.get_width_of_mini_court())
                        
                        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
                        
                        current_player_stats = deepcopy(player_stats_data[-1])
                        current_player_stats['frame_num'] = start_frame
                        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
                        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
                        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
                        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
                        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent
                        
                        player_stats_data.append(current_player_stats)
            
            player_stats_data_df = pd.DataFrame(player_stats_data)
            frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
            player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
            player_stats_data_df = player_stats_data_df.ffill()
            
            try:
                player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
                player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
            except:
                player_stats_data_df['player_1_average_shot_speed'] = 0
                player_stats_data_df['player_2_average_shot_speed'] = 0
            
            # Calculate average player speeds (opponent speed during shot intervals)
            try:
                # Avoid division by zero
                player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']
                player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']
                # Replace inf with 0
                player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_average_player_speed'].replace([np.inf, -np.inf], 0)
                player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_average_player_speed'].replace([np.inf, -np.inf], 0)
            except:
                player_stats_data_df['player_1_average_player_speed'] = 0
                player_stats_data_df['player_2_average_player_speed'] = 0
            
            # Store stats for later use
            st.session_state.player_stats_df = player_stats_data_df
            
            with status_container:
                st.success("✓ Statistics calculated!")
            
            # Draw output frames
            progress_bar.progress(65)
            progress_text.text("Stage 5/5: Generating output video...")
            with status_container:
                st.info("Stage 5/5: Generating output video...")
            
            if player_detections:
                video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
            if ball_detections:
                video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
            if court_keypoints is not None:
                video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)
            
            video_frames = mini_court.draw_mini_court(video_frames)
            if player_mini_court_detections:
                video_frames = mini_court.draw_points_on_mini_court(video_frames, player_mini_court_detections)
            if ball_mini_court_detections:
                video_frames = mini_court.draw_points_on_mini_court(video_frames, ball_mini_court_detections, color=(0, 255, 255))
            
            video_frames = draw_player_stats(video_frames, player_stats_data_df)
            
            # Add frame numbers
            for i, frame in enumerate(video_frames):
                cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save video while streaming
            st.markdown("### 🎬 Live Video Processing")
            st.markdown("*Processing frames in real-time and saving output video...*")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter("output_videos/output_video.avi", fourcc, fps, (frame_width, frame_height))
            
            frame_delay = 1 / fps
            
            for frame_idx, frame in enumerate(video_frames):
                if st.session_state.stop_processing:
                    with status_container:
                        st.warning("⏹️ Processing stopped by user")
                    st.session_state.output_video_ready = False
                    st.session_state.analysis_available = True
                    break
                
                # Save frame to output video
                out.write(frame)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_column_width=True)
                
                # Update progress
                current_progress = (65 + (frame_idx / len(video_frames)) * 35) / 100
                progress_bar.progress(current_progress)
                
                # Update frame info
                frame_info.text(f"Frame: {frame_idx + 1}/{len(video_frames)}")
                
                # Update stats display
                if frame_idx < len(player_stats_data_df):
                    stats_row = player_stats_data_df.iloc[frame_idx]
                    col1, col2 = stats_placeholder.columns(2)
                    with col1:
                        st.write("**Player 1 Stats:**")
                        st.write(f"Shots: {int(stats_row['player_1_number_of_shots'])}")
                        if stats_row['player_1_last_shot_speed'] > 0:
                            st.write(f"Last Shot Speed: {stats_row['player_1_last_shot_speed']:.1f} km/h")
                        if not np.isnan(stats_row['player_1_average_shot_speed']) and stats_row['player_1_average_shot_speed'] > 0:
                            st.write(f"Avg Shot Speed: {stats_row['player_1_average_shot_speed']:.1f} km/h")
                    with col2:
                        st.write("**Player 2 Stats:**")
                        st.write(f"Shots: {int(stats_row['player_2_number_of_shots'])}")
                        if stats_row['player_2_last_shot_speed'] > 0:
                            st.write(f"Last Shot Speed: {stats_row['player_2_last_shot_speed']:.1f} km/h")
                        if not np.isnan(stats_row['player_2_average_shot_speed']) and stats_row['player_2_average_shot_speed'] > 0:
                            st.write(f"Avg Shot Speed: {stats_row['player_2_average_shot_speed']:.1f} km/h")
                
                # Delay to match video speed
                time.sleep(frame_delay)
            
            out.release()
            
            # Success message
            progress_bar.progress(100)
            progress_text.text("✓ Processing complete!")
            with status_container:
                st.success("✓ Detection complete!")
                st.success("✓ Court keypoints detected!")
                st.success("✓ Coordinates converted!")
                st.success("✓ Statistics calculated!")
            
            st.session_state.output_video_ready = True
            st.session_state.analysis_available = True
            st.divider()
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.session_state.processing_started = False
        
        finally:
            # Clean up temp file
            if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
                try:
                    os.remove(st.session_state.temp_video_path)
                except:
                    pass
    
    # Display output video and analysis
    if (st.session_state.output_video_ready or st.session_state.analysis_available) and st.session_state.player_stats_df is not None:
        # Only show output video if processing was completed (not stopped early)
        if st.session_state.output_video_ready and os.path.exists("output_videos/output_video.avi"):
            st.divider()
            st.markdown("## ✅ Analysis Results")
            st.markdown("### 📹 Output Video with Analysis")
            with open("output_videos/output_video.avi", "rb") as f:
                video_data = f.read()
                st.video(video_data)
            st.caption("💾 Video saved to: `output_videos/output_video.avi`")
        
        # Display player analysis if data is available
        if st.session_state.player_stats_df is not None:
            display_player_analysis(st.session_state.player_stats_df)

    elif not st.session_state.processing_started and st.session_state.video_uploaded:
        st.info("▶️ Click 'Start Processing' in the sidebar to begin analyzing the video.")

else:
    st.info("👈 Upload a video file to get started!")
