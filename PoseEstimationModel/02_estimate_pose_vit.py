import os
import cv2
import numpy as np
import pandas as pd
import torch
from mmpose.apis import inference_topdown, init_model
from pathlib import Path
import json
import tqdm
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent  # project root

class ViTPoseEstimator:
    def __init__(self, pose_config=None, pose_checkpoint=None, device='cuda:0'):
        self.device = device
        print("Initializing ViTPose model (Detection skipped - using CSV inputs)...")
        self.pose_model = self._init_pose_model(pose_config, pose_checkpoint)

        # COCO 17 keypoints
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        self.keypoint_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (3, 5), (4, 6)
        ]

        self.colors = [(int(i*15%255), int(i*35%255), int(i*55%255)) for i in range(17)]
        print("ViTPose model initialized successfully!")

    def _init_pose_model(self, pose_config, pose_checkpoint):
        if pose_config is None:
            candidates = [
                Path(__file__).parent / 'td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192.py',
                PROJECT_ROOT / 'td-hm_ViTPose-large_8xb64-210e_coco-256x192.py',
            ]
            pose_config = str(next((p for p in candidates if p.exists()), candidates[0]))
        if pose_checkpoint is None:
            candidates = [
                Path(__file__).parent / 'd-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth',
                PROJECT_ROOT / 'td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth',
            ]
            pose_checkpoint = str(next((p for p in candidates if p.exists()), candidates[0]))
        return init_model(pose_config, pose_checkpoint, device=self.device)

    

    def estimate_pose(self, image, bbox):
        try:
            bbox_array = np.array(bbox, dtype=np.float32).reshape(1, 4)
            pose_results = inference_topdown(self.pose_model, image, bbox_array, bbox_format='xyxy')
            if not pose_results:
                return None

            data_sample = pose_results[0]
            kps_data = data_sample.pred_instances.keypoints[0]
            scores_data = data_sample.pred_instances.keypoint_scores[0]

            # Convert to numpy if needed
            if torch.is_tensor(kps_data):
                keypoints = kps_data.cpu().numpy()
            else:
                keypoints = np.array(kps_data)

            if torch.is_tensor(scores_data):
                scores = scores_data.cpu().numpy()
            else:
                scores = np.array(scores_data)

            return {'keypoints': keypoints, 'scores': scores, 'bbox': bbox, 'mean_score': float(scores.mean())}

        except Exception as e:
            print(f"Pose estimation error: {e}")
            return None


    def process_video(self, video_path, csv_path, output_dir, video_name, conf_threshold=0.3):
        # Load CSV
        df = pd.read_csv(csv_path)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("Failed to open video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_video_path = os.path.join(output_dir, f"{video_name}_vitpose_vis.avi")
        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

        frames_data = {}
        has_bbox_prefixed = all(c in df.columns for c in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
        has_bbox_short = all(c in df.columns for c in ['x1', 'y1', 'x2', 'y2'])
        if not has_bbox_prefixed and not has_bbox_short:
            raise ValueError("CSV must include either bbox_x1/bbox_y1/bbox_x2/bbox_y2 or x1/y1/x2/y2 columns.")

        for _, row in df.iterrows():
            fid = int(row['frame_id'])
            if has_bbox_prefixed:
                bbox = [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']]
            else:
                bbox = [row['x1'], row['y1'], row['x2'], row['y2']]
            person_id = int(row['person_id']) if 'person_id' in df.columns else 0
            frames_data.setdefault(fid, []).append({
                'person_id': person_id,
                'bbox': bbox,
                'confidence': row['confidence']
            })

        pose_results = []
        frame_id = 0
        pbar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="ViTPose")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vis_frame = frame.copy()
            fid = frame_id

            if fid in frames_data:
                for person in frames_data[fid]:
                    pose = self.estimate_pose(frame, person['bbox'])
                    if pose and pose['mean_score'] >= conf_threshold:
                        entry = {
                            'video_name': video_name,
                            'frame_id': fid,
                            'person_id': person['person_id'],
                            'pose_confidence': pose['mean_score']
                        }
                        for name, (x, y), s in zip(self.keypoint_names, pose['keypoints'], pose['scores']):
                            entry[f"{name}_x"] = float(x)
                            entry[f"{name}_y"] = float(y)
                            entry[f"{name}_conf"] = float(s)
                        pose_results.append(entry)
                        self._visualize_pose(vis_frame, pose)

            writer.write(vis_frame)
            frame_id += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()

        # Save JSON
        json_path = os.path.join(output_dir, f"{video_name}_vit_poses.json")
        with open(json_path, "w") as f:
            json.dump(pose_results, f, indent=2)

        print(f"\n✅ Saved JSON: {json_path}")
        print(f"✅ Saved video: {out_video_path}")

    def _visualize_pose(self, img, pose):
        kps, scores = pose['keypoints'], pose['scores']
        for i, (x, y) in enumerate(kps):
            if scores[i] > 0.3:
                cv2.circle(img, (int(x), int(y)), 4, self.colors[i], -1)
        for a, b in self.keypoint_connections:
            if scores[a] > 0.3 and scores[b] > 0.3:
                pt1 = tuple(map(int, kps[a]))
                pt2 = tuple(map(int, kps[b]))
                cv2.line(img, pt1, pt2, (0, 255, 255), 2)

def main():
    SCRIPT_DIR = Path(__file__).parent
    print("This is the script Directory",SCRIPT_DIR)
    video_path = SCRIPT_DIR / "Surveillance video shows fight between families in murder case.mp4"
    csv_path = SCRIPT_DIR / "fighting_detections.csv"

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    estimator = ViTPoseEstimator(device="cuda:0")
    estimator.process_video(
        video_path=str(video_path),
        csv_path=str(csv_path),
        output_dir=str(SCRIPT_DIR),
        video_name="fighting",
        conf_threshold=0.3
    )

if __name__ == "__main__":
    main()
