import gradio as gr
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import soundfile as sf
import matplotlib.animation as animation
from pathlib import Path
import mediapipe as mp
import cv2
import subprocess
import torch
from model import LandmarkPredictor
from sklearn.preprocessing import StandardScaler
import torch.serialization
import sklearn

# Add StandardScaler to safe globals for torch.load
torch.serialization.add_safe_globals([sklearn.preprocessing._data.StandardScaler])

# MediaPipe blendshape names
BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
]

def load_landmarks(landmark_path):
    """Load landmarks from a .npy file"""
    landmarks = np.load(landmark_path)
    print(f"Landmark data shape: {landmarks.shape}")
    
    # Handle different possible data structures
    if len(landmarks.shape) == 1:
        # If 1D array, try to reshape it
        total_points = len(landmarks)
        if total_points % 3 == 0:  # Must be divisible by 3 for x,y,z coordinates
            num_points = total_points // 3
            landmarks = landmarks.reshape(1, num_points, 3)
        else:
            raise ValueError(f"Invalid landmark data shape: {landmarks.shape}")
    elif len(landmarks.shape) == 2:
        # If 2D array, assume it's (num_frames, num_points*3)
        # Reshape to (num_frames, num_points, 3)
        num_frames, num_points_3 = landmarks.shape
        if num_points_3 % 3 != 0:
            raise ValueError(f"Invalid landmark data shape: {landmarks.shape}")
        num_points = num_points_3 // 3
        landmarks = landmarks.reshape(num_frames, num_points, 3)
    elif len(landmarks.shape) == 3:
        # If 3D array, ensure it's (num_frames, num_points, 3)
        if landmarks.shape[2] != 3:
            raise ValueError(f"Invalid landmark data shape: {landmarks.shape}")
    else:
        raise ValueError(f"Invalid landmark data shape: {landmarks.shape}")
    
    print(f"Reshaped landmark data shape: {landmarks.shape}")
    return landmarks

def load_blendshapes(blendshape_path):
    """Load blendshapes from a .npy file"""
    if os.path.exists(blendshape_path):
        blendshapes = np.load(blendshape_path)
        print(f"Blendshape data shape: {blendshapes.shape}")
        return blendshapes
    return None

def load_mfcc(mfcc_path):
    """Load MFCC features from a .npy file"""
    return np.load(mfcc_path)

def find_audio_file(folder_path, video_id):
    """Find the corresponding audio file for a video ID"""
    # Try the exact path structure
    audio_path = folder_path / "audio" / f"{video_id}.wav"
    if audio_path.exists():
        return str(audio_path)
    
    # If not found, try the parent directory
    parent_dir = folder_path.parent
    audio_path = parent_dir / "audio" / f"{video_id}.wav"
    if audio_path.exists():
        return str(audio_path)
    
    # If still not found, try the root data directory
    root_dir = Path("/home/mango/Desktop/wav2face/data/processed_video")
    audio_path = root_dir / video_id / "audio" / f"{video_id}.wav"
    if audio_path.exists():
        return str(audio_path)
    
    return None

def create_frame(landmarks, width=640, height=480):
    """Create a single frame of the animation"""
    # Create a blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw landmarks
    for i, (x, y, z) in enumerate(landmarks):
        # Convert normalized coordinates to pixel coordinates
        px = int(x * 640)
        py = int(y * 480)
        # Draw a circle for each landmark
        cv2.circle(image, (px, py), 1, (0, 255, 0), -1)
    
    # Draw connections between landmarks
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    
    # Get the face mesh connections
    connections = mp_face_mesh.FACEMESH_TESSELATION
    
    # Draw the connections
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        # Get the coordinates of the connected points
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        
        # Convert to pixel coordinates
        start_px = int(start_point[0] * 640)
        start_py = int(start_point[1] * 480)
        end_px = int(end_point[0] * 640)
        end_py = int(end_point[1] * 480)
        
        # Draw the line
        cv2.line(image, (start_px, start_py), (end_px, end_py), (0, 255, 0), 1)
    
    return image

def process_folder(folder_path):
    """Process a folder containing audio and landmark files"""
    folder_path = Path(folder_path)
    
    # Find MFCC and landmark files
    mfcc_files = list(folder_path.glob("**/audio_chunk/*_mfcc.npy"))
    landmark_files = list(folder_path.glob("**/landmarks/*_landmarks.npy"))
    
    if not mfcc_files or not landmark_files:
        raise ValueError("No MFCC or landmark files found in the specified folder")
    
    # Get the first matching pair
    mfcc_path = str(mfcc_files[0])
    landmark_path = str(landmark_files[0])
    
    print(f"Found MFCC file: {mfcc_path}")
    print(f"Found landmark file: {landmark_path}")
    
    # Extract video ID from the file path
    # Example: vid_1706_mfcc.npy -> vid_1706
    video_id = '_'.join(Path(mfcc_path).stem.split('_')[:-1])
    
    # Find corresponding audio file
    audio_path = find_audio_file(folder_path, video_id)
    if not audio_path:
        raise ValueError(f"No audio file found for video ID: {video_id}")
    
    print(f"Found audio file: {audio_path}")
    
    # Load landmarks and MFCC features
    landmarks_seq = load_landmarks(landmark_path)
    mfcc_features = load_mfcc(mfcc_path)
    
    print(f"Landmarks sequence shape: {landmarks_seq.shape}")
    print(f"MFCC features shape: {mfcc_features.shape}")
    
    # Calculate frame rate based on audio duration
    y, sr = librosa.load(audio_path)
    audio_duration = len(y) / sr
    fps = len(landmarks_seq) / audio_duration
    
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Number of landmark frames: {len(landmarks_seq)}")
    print(f"FPS: {fps:.2f}")
    
    # Create temporary files for video and final output
    temp_video = tempfile.mktemp(suffix=".mp4")
    final_video = tempfile.mktemp(suffix=".mp4")
    
    # Create video without audio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (640, 480))
    
    for frame in range(len(landmarks_seq)):
        image = create_frame(landmarks_seq[frame])
        out.write(image)
    
    out.release()
    
    # Combine video and audio using ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_video,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        final_video
    ]
    
    subprocess.run(cmd, check=True)
    
    # Clean up temporary video file
    os.remove(temp_video)
    
    return final_video

def process_audio_with_tcn(audio_file):
    """Process audio file using TCN model to generate landmarks"""
    # Load and process audio
    y, sr = librosa.load(audio_file)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.T  # Transpose to get (time, features)
    
    # Initialize TCN model
    predictor = LandmarkPredictor("model.pth")
    
    # Generate landmarks for each frame
    landmarks_seq = []
    for frame in range(len(mfcc)):
        landmarks = predictor.predict(mfcc[frame])
        landmarks_seq.append(landmarks)
    
    landmarks_seq = np.array(landmarks_seq)
    
    # Calculate frame rate
    fps = len(landmarks_seq) / (len(y) / sr)
    
    # Create temporary files for video and final output
    temp_video = tempfile.mktemp(suffix=".mp4")
    final_video = tempfile.mktemp(suffix=".mp4")
    
    # Create video without audio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (640, 480))
    
    for frame in range(len(landmarks_seq)):
        image = create_frame(landmarks_seq[frame])
        out.write(image)
    
    out.release()
    
    # Combine video and audio using ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_video,
        '-i', audio_file,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        final_video
    ]
    
    subprocess.run(cmd, check=True)
    
    # Clean up temporary video file
    os.remove(temp_video)
    
    return final_video

# Create two separate interfaces
with gr.Blocks() as demo:
    gr.Markdown("# Facial Landmark Viewer")
    
    # with gr.Tab("View Existing Landmarks"):
    #     folder_input = gr.Textbox(
    #         label="Enter folder path containing audio and landmark files",
    #         placeholder="/vid_1706",
    #         value="/vid_1903"
    #     )
    #     view_output = gr.Video(label="Facial Landmark Animation with Audio")
    #     view_button = gr.Button("View Landmarks")
    #     view_button.click(process_folder, inputs=folder_input, outputs=view_output)
    
    with gr.Tab("Generate Landmarks from Audio"):
        audio_input = gr.Audio(type="filepath", label="Upload Audio File", placeholder="udio.wav", value="audio.wav")
        generate_output = gr.Video(label="Generated Facial Landmark Animation")
        generate_button = gr.Button("Generate Landmarks")
        generate_button.click(process_audio_with_tcn, inputs=audio_input, outputs=generate_output)

demo.launch()
