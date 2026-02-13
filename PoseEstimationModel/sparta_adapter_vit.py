import json
import os
import torch
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import cv2
import sys

# Import SPARTA components
# Assuming current directory is .../SPARTA/PoseEstimationModel
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from models import SPARTA_H
from utils.train_utils import CostumLoss

# --- Configuration ---
SEG_LEN = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SPARTA COCO-18 order
# [Nose, Neck, R-Sho, R-Elb, R-Wri, L-Sho, L-Elb, L-Wri, R-Hip, R-Kne, R-Ank, L-Hip, L-Kne, L-Ank, R-Eye, L-Eye, R-Ear, L-Ear]
SPARTA_ORDER = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def process_pose_data(json_path, ctd_path, ftd_path, video_path=None, output_csv="anomaly_results.csv"):
    # 0. Get Resolution (Not strictly needed for Z-score but good for metadata)
    width, height = 1280, 720
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

    # 1. Load Data
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        pose_data = json.load(f)
    
    df = pd.DataFrame(pose_data)
    print(f"Loaded {len(df)} pose records.")

    # 2. Group by Person ID
    person_groups = df.groupby('person_id')

    # 3. Load Models
    print(f"Loading SPARTA models on {DEVICE}...")
    model = SPARTA_H(
        d_model=72, # 18 kps, 2 channels (x,y), 2 (absolute + relative)
        num_heads=12,
        d_feedforward=64,
        num_layers=4,
        max_len=1000,
        device=DEVICE,
        dropout=0.0
    )
    
    checkpoint_c = torch.load(ctd_path, map_location=DEVICE, weights_only=False)
    checkpoint_f = torch.load(ftd_path, map_location=DEVICE, weights_only=False)
    model.CTD.load_state_dict(checkpoint_c['state_dict'])
    model.FTD.load_state_dict(checkpoint_f['state_dict'])
    model.to(DEVICE)
    model.eval()
    
    loss_func = CostumLoss('mse')
    results = []

    # 4. Iterate through each person
    total_persons = len(person_groups)
    print(f"Analyzing {total_persons} persons...")

    for person_id, group in person_groups:
        group = group.sort_values('frame_id')
        frame_ids = group['frame_id'].values
        
        # --- Optimized Data Extraction ---
        # Get all x, y coordinates as a numpy array in one go
        cols_x = [f"{name}_x" for name in COCO_KEYPOINT_NAMES]
        cols_y = [f"{name}_y" for name in COCO_KEYPOINT_NAMES]
        kps_x = group[cols_x].values # [Frames, 17]
        kps_y = group[cols_y].values # [Frames, 17]
        
        kps_17 = np.stack([kps_x, kps_y], axis=-1) # [Frames, 17, 2]
        
        # Calculate Neck
        neck = 0.5 * (kps_17[:, 5, :] + kps_17[:, 6, :])
        kps_18 = np.concatenate([kps_17, neck[:, np.newaxis, :]], axis=1) # [Frames, 18, 2]
        
        # Reorder to SPARTA format
        kps_sparta = kps_18[:, SPARTA_ORDER, :] # [Frames, 18, 2]
        
        # --- Z-Score Normalization (as found in utils/data_utils.py) ---
        # 1. Pixel to [0, 1] relative to video resolution (used in the codebase)
        kps_norm = kps_sparta.copy()
        kps_norm[..., 0] /= width
        kps_norm[..., 1] /= height
        
        # 2. Subtract Mean and Divide by Std of Y
        # This makes the skeleton size invariant to distance
        # Note: We calculate mean/std PER FRAME to keep it consistent with real-time inference
        for f in range(len(kps_norm)):
            frame_kps = kps_norm[f] # [18, 2]
            mean_vals = frame_kps.mean(axis=0) # [2]
            std_val = frame_kps[:, 1].std() # Std of Y coordinates
            
            if std_val > 0.0001:
                kps_norm[f] = (frame_kps - mean_vals) / std_val
            else:
                kps_norm[f] = frame_kps - mean_vals
        
        # 5. Sliding Window Analysis
        person_results = []
        for i in range(len(kps_norm) - SEG_LEN + 1):
            window = kps_norm[i:i+SEG_LEN] # [SEG_LEN, 18, 2]
            
            # Convert to Tensor
            data_tensor = torch.from_numpy(window).float().unsqueeze(0) # [1, SEG_LEN, 18, 2]
            # Reshape to [1, 2, SEG_LEN, 18]
            data_tensor = data_tensor.permute(0, 3, 1, 2)
            
            # Calculate Relative Movement
            rel = torch.zeros_like(data_tensor)
            rel[:, :, 1:, :] = data_tensor[:, :, 1:, :] - data_tensor[:, :, 0:1, :]
            data_tensor = torch.cat([data_tensor, rel], dim=1) # [1, 4, SEG_LEN, 18]
            
            data_tensor = data_tensor.to(DEVICE)
            
            # Layout: [B, L, C*K]
            # This matches 't' config and what I saw in Tokenizers.py
            input_seq = data_tensor.permute(0, 2, 1, 3).contiguous().view(1, SEG_LEN, -1)
            
            with torch.no_grad():
                recon = model.CTD(input_seq, input_seq)
                loss = loss_func.calculate(input_seq, recon)
                anomaly_score = loss.mean().item()
            
            person_results.append({
                "person_id": person_id,
                "frame_id": frame_ids[i + SEG_LEN - 1],
                "anomaly_score": anomaly_score
            })
        results.extend(person_results)

    # 6. Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"Analysis complete. Results saved to {output_csv}")
    
    # Check max score for debugging
    if not res_df.empty:
        max_score = res_df['anomaly_score'].max()
        avg_score = res_df['anomaly_score'].mean()
        print(f"Statistics: Max Score={max_score:.4f}, Avg Score={avg_score:.4f}")

if __name__ == "__main__":
    JSON_FILE = "fighting_vit_poses.json"
    CTD_MODEL = "../Trained_Models/SHT/CTD.pth.tar"
    FTD_MODEL = "../Trained_Models/SHT/FTD.pth.tar"
    VIDEO_FILE = r"C:\Users\Shahzaib\OneDrive\Desktop\PoseEstimationModel\SPARTA\PoseEstimationModel\Surveillance video shows fight between families in murder case.mp4"
    
    process_pose_data(JSON_FILE, CTD_MODEL, FTD_MODEL, video_path=VIDEO_FILE, output_csv="anomaly_fighting_results.csv")
