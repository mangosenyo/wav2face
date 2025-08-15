import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import os
import sklearn.preprocessing._data
from torch.cuda.amp import autocast, GradScaler
import logging
from pathlib import Path
import time
import cv2
from scipy.signal import savgol_filter

import argparse

# Add this at the beginning of your script (right after imports)
def parse_args():
    parser = argparse.ArgumentParser(description='Train Audio2Landmark TCN model')
    parser.add_argument('-n', '--model_name', type=str, default='default_model',
                       help='Name for the trained model')
    parser.add_argument('-d', '--data_path', type=str, 
                       default='/home/mango/Desktop/wav2face/data/processed_video',
                       help='Path to training data')
    return parser.parse_args()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Configuration
CONFIG = {
    "audio_dim": 40,           # MFCC features dimension
    "landmark_dim": 478*3,     # 478 landmarks xyz coordinates
    "channels": [64, 128, 256], # TCN channel widths
    "kernel_size": 5,
    "dropout": 0.2,
    "batch_size": 32,
    "lr": 3e-4,
    "epochs": 50,
    "device": "cuda",
    "val_split": 0.1,         # Validation split ratio
    "checkpoint_dir": "checkpoints",
    "early_stopping_patience": 10,
    "smoothing_window": 5,    # Window size for temporal smoothing
    "head_stabilization": True, # Whether to stabilize head movements
    "lip_weight": 2.0,        # Weight for lip region in loss
    "eye_weight": 1.5,        # Weight for eye region in loss
    "face_weight": 1.0        # Weight for rest of face
}

# Define facial regions (MediaPipe indices)
LIP_LANDMARKS = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 
    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 
    317, 318, 321, 324, 375, 402, 405, 409, 415
]
EYE_LANDMARKS = [
    33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 
    246, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398
]

# TCN Block (Causal Conv + Dilations)
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=(kernel_size - 1) * dilation,  # Causal padding
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x[..., :-self.conv.padding[0]]  # Ensure causality
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if self.residual is not None:
            residual = self.residual(residual)
        return x + residual

# Full TCN Model with temporal context
class Audio2LandmarkTCN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = CONFIG["audio_dim"]
        
        # Build TCN blocks with increasing dilation
        for i, out_channels in enumerate(CONFIG["channels"]):
            dilation = 2 ** i  # Exponential dilation
            layers += [
                TCNBlock(
                    in_channels, 
                    out_channels, 
                    CONFIG["kernel_size"], 
                    dilation
                )
            ]
            in_channels = out_channels
        
        self.tcn = nn.Sequential(*layers)
        
        # Temporal context module
        self.temporal_conv = nn.Conv1d(in_channels, in_channels, 
                                      kernel_size=CONFIG["smoothing_window"], 
                                      padding=CONFIG["smoothing_window"]//2)
        
        # Output layers
        self.output = nn.Linear(in_channels, CONFIG["landmark_dim"])
        
    def forward(self, x):
        # x shape: (batch, audio_features, timesteps)
        x = self.tcn(x)
        
        # Apply temporal smoothing
        x = self.temporal_conv(x)
        x = F.relu(x)
        
        # Global average pooling
        x = x.mean(dim=-1)
        return self.output(x)

# Dataset with weighted regions
class LandmarkDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.video_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        self.audio_scaler = StandardScaler()
        self.landmark_scaler = StandardScaler()
        
        # Build frame index mapping
        self.frame_index = []
        self._fit_scalers()
        self._build_index()
        
        # Create weights for different facial regions
        self.weights = np.ones(478*3) * CONFIG["face_weight"]
        
        # Apply weights to lip region
        for idx in LIP_LANDMARKS:
            self.weights[idx*3:(idx+1)*3] = CONFIG["lip_weight"]
            
        # Apply weights to eye region
        for idx in EYE_LANDMARKS:
            self.weights[idx*3:(idx+1)*3] = CONFIG["eye_weight"]
        
    def _fit_scalers(self):
        all_audio = []
        all_landmarks = []
        
        for video_dir in self.video_dirs:
            audio_path = os.path.join(self.data_root, video_dir, "audio_chunk", f"{video_dir}_mfcc.npy")
            landmark_path = os.path.join(self.data_root, video_dir, "landmarks", f"{video_dir}_landmarks.npy")
            
            if os.path.exists(audio_path) and os.path.exists(landmark_path):
                audio = np.load(audio_path)  # shape: (N, 40)
                landmarks = np.load(landmark_path)  # shape: (N, 478*3)
                
                min_frames = min(len(audio), len(landmarks))
                all_audio.append(audio[:min_frames])
                all_landmarks.append(landmarks[:min_frames])
        
        if all_audio:
            self.audio_scaler.fit(np.vstack(all_audio))
            self.landmark_scaler.fit(np.vstack(all_landmarks))
    
    def _build_index(self):
        self.frame_index = []
        for video_idx, video_dir in enumerate(self.video_dirs):
            audio_path = os.path.join(self.data_root, video_dir, "audio_chunk", f"{video_dir}_mfcc.npy")
            landmark_path = os.path.join(self.data_root, video_dir, "landmarks", f"{video_dir}_landmarks.npy")
            if os.path.exists(audio_path) and os.path.exists(landmark_path):
                audio = np.load(audio_path)
                landmarks = np.load(landmark_path)
                min_len = min(len(audio), len(landmarks))
                for frame_idx in range(min_len):
                    self.frame_index.append((video_idx, frame_idx))
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, idx):
        video_idx, frame_idx = self.frame_index[idx]
        video_dir = self.video_dirs[video_idx]
        
        audio_path = os.path.join(self.data_root, video_dir, "audio_chunk", f"{video_dir}_mfcc.npy")
        landmark_path = os.path.join(self.data_root, video_dir, "landmarks", f"{video_dir}_landmarks.npy")
        
        audio = np.load(audio_path)
        landmarks = np.load(landmark_path)
        
        # Get the specific frame
        audio_frame = self.audio_scaler.transform(audio[frame_idx].reshape(1, -1)).flatten()
        landmark_frame = self.landmark_scaler.transform(landmarks[frame_idx].reshape(1, -1)).flatten()
        
        return (
            torch.FloatTensor(audio_frame), 
            torch.FloatTensor(landmark_frame),
            torch.FloatTensor(self.weights)
        )

# Weighted MSE Loss
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target, weights):
        return (weights * (input - target) ** 2).mean()

# Training with weighted loss
def train(dataset_path):
    # Create checkpoint directory
    Path(CONFIG["checkpoint_dir"]).mkdir(exist_ok=True)
    
    # Initialize
    dataset = LandmarkDataset(dataset_path)
    
    # Split into train and validation sets
    val_size = int(len(dataset) * CONFIG["val_split"])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    model = Audio2LandmarkTCN().to(CONFIG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = WeightedMSELoss()
    scaler = GradScaler()
    
    # Training metrics
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    # Training loop
    for epoch in range(CONFIG["epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        
        for audio, landmarks, weights in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            audio = audio.to(CONFIG["device"], non_blocking=True)
            landmarks = landmarks.to(CONFIG["device"], non_blocking=True)
            weights = weights.to(CONFIG["device"], non_blocking=True)
            audio = audio.unsqueeze(-1)
            
            optimizer.zero_grad()
            
            with autocast():
                pred = model(audio)
                loss = criterion(pred, landmarks, weights)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for audio, landmarks, weights in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                audio = audio.to(CONFIG["device"], non_blocking=True)
                landmarks = landmarks.to(CONFIG["device"], non_blocking=True)
                weights = weights.to(CONFIG["device"], non_blocking=True)
                audio = audio.unsqueeze(-1)
                
                with autocast():
                    pred = model(audio)
                    loss = criterion(pred, landmarks, weights)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Log metrics
        logging.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "audio_scaler": dataset.audio_scaler,
                "landmark_scaler": dataset.landmark_scaler,
                "config": CONFIG,
                "epoch": epoch,
                "val_loss": best_val_loss
            }, os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save({
        "model": model.state_dict(),
        "audio_scaler": dataset.audio_scaler,
        "landmark_scaler": dataset.landmark_scaler,
        "config": CONFIG,
        "epoch": epoch,
        "val_loss": avg_val_loss
    }, os.path.join(CONFIG["checkpoint_dir"], "final_model.pth"))
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time/60:.2f} minutes")

# Enhanced Landmark Predictor with smoothing and stabilization
class LandmarkPredictor:
    def __init__(self, model_path):
        # Add necessary safe globals
        torch.serialization.add_safe_globals([
            sklearn.preprocessing._data.StandardScaler,
            np.core.multiarray.scalar
        ])
        
        # Load checkpoint with weights_only=False since we need the scalers
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = Audio2LandmarkTCN()
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.audio_scaler = checkpoint["audio_scaler"]
        self.landmark_scaler = checkpoint["landmark_scaler"]
        self.config = checkpoint.get("config", CONFIG)
        
        # Buffer for temporal smoothing
        self.prediction_buffer = []
        self.buffer_size = self.config["smoothing_window"]
        
        # Reference landmarks for stabilization
        self.reference_landmarks = None
        
        # Key facial landmarks for stabilization (MediaPipe indices)
        self.stabilization_landmarks = [
            1,    # Nose tip
            4,     # Nose bridge
            152,   # Chin
            33,    # Left eye
            263,   # Right eye
            61,    # Left mouth corner
            291,   # Right mouth corner
            199    # Center of forehead
        ]
        
    def _get_stabilization_transform(self, landmarks):
        """Calculate transform to stabilize head position"""
        landmarks = landmarks.reshape(478, 3)
        
        # If we don't have reference landmarks yet, use current frame
        if self.reference_landmarks is None:
            self.reference_landmarks = landmarks[self.stabilization_landmarks]
            return np.eye(3), np.zeros(3)  # Identity transform
        
        # Get current stabilization landmarks
        current_points = landmarks[self.stabilization_landmarks]
        
        # Calculate rigid transformation (rotation + translation)
        # Using Kabsch algorithm to find optimal rotation
        H = (current_points - current_points.mean(0)).T @ (self.reference_landmarks - self.reference_landmarks.mean(0))
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
            
        # Calculate translation
        t = self.reference_landmarks.mean(0) - R @ current_points.mean(0)
        
        return R, t
    
    def _apply_stabilization(self, landmarks, R, t):
        """Apply stabilization transform to landmarks"""
        landmarks = landmarks.reshape(478, 3)
        stabilized = (R @ landmarks.T).T + t
        return stabilized.reshape(-1)
    
    def _smooth_predictions(self, landmarks):
        """Apply temporal smoothing to landmarks"""
        self.prediction_buffer.append(landmarks)
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
            
        if len(self.prediction_buffer) == 1:
            return landmarks
            
        # Apply weighted moving average
        weights = np.linspace(0.1, 1.0, len(self.prediction_buffer))
        weights /= weights.sum()
        
        smoothed = np.zeros_like(landmarks)
        for i, frame in enumerate(self.prediction_buffer):
            smoothed += weights[i] * frame
            
        return smoothed
    
    def predict(self, mfcc_features, stabilize=True, smooth=True):
        """Input: (40,) numpy array of MFCC features
        Output: (478, 3) landmark coordinates
        Parameters:
            stabilize: if True, stabilize head position (default: True)
            smooth: if True, apply temporal smoothing (default: True)
        """
        with torch.no_grad():
            # Scale and reshape
            audio = self.audio_scaler.transform(mfcc_features.reshape(1, -1))
            audio = torch.FloatTensor(audio).unsqueeze(-1)  # Add timestep dim
            
            # Predict
            pred = self.model(audio)
            landmarks = self.landmark_scaler.inverse_transform(pred.numpy())
            
            # Reshape to (478, 3)
            landmarks = landmarks.reshape(478, 3)
            
            # Apply stabilization if requested
            if stabilize:
                R, t = self._get_stabilization_transform(landmarks)
                landmarks = self._apply_stabilization(landmarks, R, t)
                
            # Apply temporal smoothing if requested
            if smooth:
                landmarks = self._smooth_predictions(landmarks)
            
            return landmarks.reshape(478, 3)

if __name__ == "__main__":
    dataset_path = ""
    train(dataset_path)