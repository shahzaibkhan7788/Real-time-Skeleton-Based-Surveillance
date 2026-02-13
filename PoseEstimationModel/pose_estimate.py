import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
from pathlib import Path
import tqdm

class PoseEstimator:
    def __init__(self, model_name='yolo26x-pose.pt', conf_threshold=0.25):
        """
        Initialize pose estimator with YOLOv26 Pose model
        
        Args:
            model_name: YOLOv26 Pose model to use
            conf_threshold: Confidence threshold for pose detection
        """
        print("Loading YOLOv26 Pose model...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        
        # COCO Pose keypoint names (17 points)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # COCO-Pose 17 keypoint connections (zero-indexed for easy array access)
        # Pairs: [p1_index, p2_index] based on self.keypoint_names order (0-16)
        self.SKELETON_0_INDEXED = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], # Legs and Hips
            [5, 11], [6, 12], [5, 6],                      # Torso/Hips
            [5, 7], [6, 8], [7, 9], [8, 10],               # Arms
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],          # Head/Face
            [3, 5], [4, 6]                                 # Ears/Shoulders
        ]
        
    def estimate_pose_for_tracks(self, tracks_csv_path, video_path, output_dir, video_name):
        """
        Estimate poses for all tracked persons in a video and save the results (CSV/JSON).
        (Function logic unchanged from your original code)
        """
        print(f"Processing poses for: {video_name}")
        
        # Load tracking data
        try:
            tracks_df = pd.read_csv(tracks_csv_path)
        except FileNotFoundError:
            print(f"Tracking file not found: {tracks_csv_path}")
            return
        
        if tracks_df.empty:
            print(f"No tracking data found for {video_name}")
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Group tracks by frame for efficient processing
        frames_data = {}
        for _, row in tracks_df.iterrows():
            frame_id = int(row['frame_id'])
            person_id = int(row['person_id'])
            
            if frame_id not in frames_data:
                frames_data[frame_id] = []
                
            frames_data[frame_id].append({
                'person_id': person_id,
                'bbox': [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']],
                'confidence': row['confidence']
            })
        
        # Store pose results
        pose_results = []
        frame_count = 0
        
        # Process frames with progress bar
        pbar = tqdm.tqdm(total=total_frames, desc=f"Estimating {video_name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frames_data:
                person_data = frames_data[frame_count]
                
                for person in person_data:
                    person_id = person['person_id']
                    bbox = person['bbox']
                    
                    # Extract ROI
                    x1, y1, x2, y2 = map(int, bbox)
                    w = x2 - x1
                    h = y2 - y1
                    x1_exp = max(0, int(x1 - 0.1 * w))
                    y1_exp = max(0, int(y1 - 0.1 * h))
                    x2_exp = min(width, int(x2 + 0.1 * w))
                    y2_exp = min(height, int(y2 + 0.1 * h))
                    
                    roi = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                    
                    if roi.size == 0:
                        continue
                    
                    # Run pose estimation on the ROI
                    try:
                        results = self.model(roi, 
                                           conf=self.conf_threshold,
                                           verbose=False)
                        
                        if len(results) > 0 and results[0].keypoints is not None:
                            keypoints = results[0].keypoints.xy.cpu().numpy()
                            confidences = results[0].keypoints.conf.cpu().numpy()
                            
                            if len(keypoints) > 0:
                                # Convert ROI coordinates back to original image coordinates
                                kp_original = keypoints[0].copy()
                                kp_original[:, 0] += x1_exp
                                kp_original[:, 1] += y1_exp
                                
                                confs = confidences[0]
                                
                                # Store pose data
                                pose_data = {
                                    'video_name': video_name,
                                    'frame_id': frame_count,
                                    'person_id': person_id,
                                    'bbox_x1': x1,
                                    'bbox_y1': y1,
                                    'bbox_x2': x2,
                                    'bbox_y2': y2,
                                    'pose_confidence': np.mean(confs)
                                }
                                
                                # Add each keypoint
                                for i, (kp_name, (x, y), conf) in enumerate(zip(self.keypoint_names, kp_original, confs)):
                                    pose_data[f'{kp_name}_x'] = float(x)
                                    pose_data[f'{kp_name}_y'] = float(y)
                                    pose_data[f'{kp_name}_conf'] = float(conf)
                                
                                pose_results.append(pose_data)
                                
                    except Exception as e:
                        print(f"Error processing frame {frame_count}, person {person_id}: {e}")
                        continue
            
            frame_count += 1
            pbar.update(1)
            
            if frame_count >= total_frames:
                break
        
        pbar.close()
        cap.release()
        
        # Save pose results
        if pose_results:
            pose_df = pd.DataFrame(pose_results)
            output_path = os.path.join(output_dir, f"{video_name}_poses.csv")
            pose_df.to_csv(output_path, index=False)
            
            json_output_path = os.path.join(output_dir, f"{video_name}_poses.json")
            pose_dict = pose_df.to_dict('records')
            with open(json_output_path, 'w') as f:
                json.dump(pose_dict, f, indent=2)
            
            print(f"Saved pose data to {output_path}")
        else:
            print(f"No poses detected in {video_name}")
        
        return pose_results

    def draw_and_save_video(self, pose_csv_path, video_path, output_video_path, min_conf=0.5):
        """
        Draws pose skeletons on the video frames and saves a new video.
        
        Args:
            pose_csv_path: Path to the generated pose results CSV.
            video_path: Path to the original input video.
            output_video_path: Path to save the new annotated video.
            min_conf: Minimum keypoint confidence to draw a keypoint/limb.
        """
        print(f"\nStarting visualization for: {Path(video_path).name}")
        
        try:
            pose_df = pd.read_csv(pose_csv_path)
        except FileNotFoundError:
            print(f"Pose data file not found: {pose_csv_path}. Cannot create video.")
            return
            
        if pose_df.empty:
            print(f"No pose data found in {pose_csv_path}. Cannot create video.")
            return
            
        # Group pose data by frame
        frames_poses = pose_df.groupby('frame_id')
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set up VideoWriter
        # Use 'XVID' for AVI compatibility or 'mp4v' for MP4 (might require FFmpeg).
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        try:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        except Exception as e:
            print(f"Error initializing VideoWriter: {e}")
            return

        frame_count = 0
        pbar = tqdm.tqdm(total=total_frames, desc=f"Visualizing {Path(video_path).name}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frames_poses.groups:
                frame_pose_df = frames_poses.get_group(frame_count)
                
                for _, row in frame_pose_df.iterrows():
                    keypoints = []
                    confidences = []
                    
                    # 1. Extract keypoint coordinates and confidences
                    for kp_name in self.keypoint_names:
                        x = row.get(f'{kp_name}_x')
                        y = row.get(f'{kp_name}_y')
                        conf = row.get(f'{kp_name}_conf')
                        
                        # Store as tuple (x, y) if valid, otherwise None
                        if pd.notna(x) and pd.notna(y):
                            keypoints.append((int(x), int(y)))
                            confidences.append(conf if pd.notna(conf) else 0.0)
                        else:
                            keypoints.append(None) 
                            confidences.append(0.0)

                    # 2. Draw Skeleton Connections
                    for p1_idx, p2_idx in self.SKELETON_0_INDEXED:
                        pt1 = keypoints[p1_idx]
                        pt2 = keypoints[p2_idx]
                        
                        conf1 = confidences[p1_idx]
                        conf2 = confidences[p2_idx]
                        
                        # Draw line if both points are valid and have sufficient confidence
                        if pt1 and pt2 and conf1 >= min_conf and conf2 >= min_conf:
                            cv2.line(frame, pt1, pt2, (255, 0, 0), 2, cv2.LINE_AA) # Blue limb
                            
                    # 3. Draw Keypoints (Joints)
                    for pt, conf in zip(keypoints, confidences):
                        if pt and conf >= min_conf:
                            cv2.circle(frame, pt, 5, (0, 255, 255), -1, cv2.LINE_AA) # Yellow joint
                            
                    # 4. Draw Bounding Box and ID (for reference)
                    x1, y1, x2, y2 = int(row['bbox_x1']), int(row['bbox_y1']), int(row['bbox_x2']), int(row['bbox_y2'])
                    person_id = int(row['person_id'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, f"ID:{person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


            # Write the frame to the output video
            out.write(frame)
            
            frame_count += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Successfully saved annotated video to: {output_video_path}")

def main():
    """Main function to estimate poses and visualize for all tracked videos"""
    
    # Configuration
    BASE_DIR = "/home/waleed64/Desktop/abnormal_activity_project"
    RAW_VIDEOS_DIR = os.path.join(BASE_DIR, "data", "raw_videos")
    TRACKING_DIR = os.path.join(BASE_DIR, "data", "processed", "bounding_boxes")
    POSE_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "raw_poses")
    VISUAL_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "annotated_videos") # New directory
    
    # Create output directories
    os.makedirs(POSE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(
        model_name='yolo26x-pose.pt',
        conf_threshold=0.25
    )
    
    # --- Centralized processing loop ---
    def process_videos_in_dir(videos_dir, data_type):
        print(f"\nProcessing {data_type} videos from: {videos_dir}")
        if os.path.exists(videos_dir):
            for video_folder in sorted(os.listdir(videos_dir)):
                video_folder_path = os.path.join(videos_dir, video_folder)
                if os.path.isdir(video_folder_path):
                    video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.avi')]
                    if video_files:
                        video_name = video_folder
                        video_path = os.path.join(video_folder_path, video_files[0])
                        tracking_file = os.path.join(TRACKING_DIR, f"{video_name}_tracks.csv")
                        pose_csv_path = os.path.join(POSE_OUTPUT_DIR, f"{video_name}_poses.csv")
                        output_video_path = os.path.join(VISUAL_OUTPUT_DIR, f"{video_name}_pose_annotated.avi")
                        
                        # --- STEP 1: Estimate Poses (If tracking file exists and pose data is missing) ---
                        if os.path.exists(tracking_file) and not os.path.exists(pose_csv_path):
                            pose_estimator.estimate_pose_for_tracks(
                                tracking_file, video_path, POSE_OUTPUT_DIR, video_name
                            )
                        elif not os.path.exists(tracking_file):
                            continue # Skip if no tracking data to process

                        # --- STEP 2: Visualize and Save Video (If pose data exists) ---
                        if os.path.exists(pose_csv_path):
                            if not os.path.exists(output_video_path):
                                pose_estimator.draw_and_save_video(
                                    pose_csv_path, 
                                    video_path, 
                                    output_video_path,
                                    min_conf=0.5 # Only draw keypoints/limbs with > 50% confidence
                                )
                            else:
                                 print(f"Annotated video already exists: {output_video_path}")
                        else:
                            print(f"Pose CSV not found for visualization: {pose_csv_path}")


    # Process training videos
    train_videos_dir = os.path.join(RAW_VIDEOS_DIR, "train", "Train")
    process_videos_in_dir(train_videos_dir, "training")
    
    # Process test videos
    test_videos_dir = os.path.join(RAW_VIDEOS_DIR, "test", "Test")
    process_videos_in_dir(test_videos_dir, "test")

def validate_pose_data():
    """Function to validate and analyze the generated pose data"""
    BASE_DIR = "/home/waleed64/Desktop/abnormal_activity_project"
    POSE_DIR = os.path.join(BASE_DIR, "data", "processed", "raw_poses")
    
    if os.path.exists(POSE_DIR):
        pose_files = [f for f in os.listdir(POSE_DIR) if f.endswith('.csv')]
        print(f"Found {len(pose_files)} pose files")
        
        for pose_file in pose_files[:5]:  # Check first 5 files
            pose_path = os.path.join(POSE_DIR, pose_file)
            df = pd.read_csv(pose_path)
            print(f"\n{pose_file}:")
            print(f"  - Total pose detections: {len(df)}")
            print(f"  - Unique persons: {df['person_id'].nunique()}")
            print(f"  - Frame range: {df['frame_id'].min()} to {df['frame_id'].max()}")
            print(f"  - Average pose confidence: {df['pose_confidence'].mean():.3f}")

if __name__ == "__main__":
    main()
    
    # Validate the results
    print("\n" + "="*50)
    print("VALIDATING POSE DATA")
    print("="*50)
    validate_pose_data()
