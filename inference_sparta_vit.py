import argparse
import os
import cv2
import torch
import numpy as np

# Models
from ultralytics import YOLO
try:
    from mmpose.apis import inference_topdown, init_model
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    print("Warning: mmpose not found. ViTPose functionality will be disabled unless installed.")

from models import SPARTA_H
from utils.train_utils import CostumLoss

# --- Keypoint Mapping ---
def keypoints17_to_coco18(kps):
    """
    Converts 17 COCO keypoints to 18 points by adding a neck point.
    Ordering: [Nose, Neck, R-Sho, R-Elb, R-Wri, L-Sho, L-Elb, L-Wri, R-Hip, R-Kne, R-Ank, L-Hip, L-Kne, L-Ank, R-Eye, L-Eye, R-Ear, L-Ear]
    (Based on SPARTA's expected input order)
    """
    kp_np = np.array(kps) # [N, 17, 2 or 3]
    
    # Calculate neck as average of shoulders (indices 5 and 6 in COCO 17)
    neck = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    
    # Expand to 18 points
    # Standard COCO 17: 0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear, 5:l_sho, 6:r_sho, 7:l_elb, 8:r_elb, 9:l_wri, 10:r_wri, 11:l_hip, 12:r_hip, 13:l_knee, 14:r_knee, 15:l_ank, 16:r_ank
    kp18 = np.concatenate([kp_np, neck[..., None, :]], axis=-2) # Index 17 is neck
    
    # SPARTA COCO-18 order used for keypoint remapping:
    # [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int_)
    
    return kp18[..., opp_order, :]

def get_args():
    parser = argparse.ArgumentParser(description="SPARTA + YOLOv26 + ViTPose Inference")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--yolo_model", type=str, default="yolo26x-pose.pt", help="YOLO model path (e.g. yolo26x-pose.pt)")
    parser.add_argument("--vit_config", type=str, default=None, help="MMPose ViTPose config")
    parser.add_argument("--vit_ckpt", type=str, default=None, help="MMPose ViTPose checkpoint")
    parser.add_argument("--ctd_path", type=str, default="Trained_Models/CHAD/CTD.pth.tar", help="Path to CTD model checkpoint")
    parser.add_argument("--ftd_path", type=str, default="Trained_Models/CHAD/FTD.pth.tar", help="Path to FTD model checkpoint")
    parser.add_argument("--save_output", type=str, default="output_vit.mp4", help="Path to save output video")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seg_len", type=int, default=12, help="Segment length for SPARTA (default 12)")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--score_mode", type=str, default="both", choices=["ctd", "ftd", "both"], help="Scoring branch to use")
    parser.add_argument("--anomaly_threshold", type=float, default=None, help="Fixed anomaly threshold. If omitted, adaptive threshold is used")
    parser.add_argument("--calib_frames", type=int, default=150, help="Number of initial frames used for adaptive threshold calibration")
    parser.add_argument("--calib_percentile", type=float, default=97.5, help="Percentile used to derive adaptive threshold from calibration scores")
    parser.add_argument("--smooth_alpha", type=float, default=0.25, help="EMA smoothing factor for per-track anomaly scores")
    parser.add_argument("--min_pose_conf", type=float, default=0.15, help="Minimum mean pose confidence to score a person")
    parser.add_argument("--max_buffer_mult", type=int, default=4, help="Track buffer length multiplier w.r.t seg_len")
    return parser.parse_args()

class MockArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x):
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def std(self):
        if self.n < 2:
            return 1.0
        return float(np.sqrt(max(self.m2 / (self.n - 1), 1e-8)))

    def zscore(self, x):
        x = float(x)
        if self.n < 20:
            return x
        return (x - self.mean) / self.std()


def normalize_window_training_style(seq_18, width, height, eps=1e-6):
    """Approximate training-time normalization used in utils/data_utils.py."""
    seq_norm = np.array(seq_18, dtype=np.float32, copy=True)  # [T, 18, 2]
    seq_norm[..., 0] /= float(max(width, 1))
    seq_norm[..., 1] /= float(max(height, 1))
    mean_xy = seq_norm.reshape(-1, 2).mean(axis=0)
    std_y = float(seq_norm[..., 1].std())
    if std_y < eps:
        std_y = eps
    seq_norm = (seq_norm - mean_xy) / std_y
    return seq_norm


def build_sparta_tensor(seq_norm, relative=True):
    data_tensor = torch.from_numpy(seq_norm).float()  # [T, 18, 2]
    data_tensor = data_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 2, T, 18]
    if relative:
        rel = torch.zeros_like(data_tensor)
        rel[:, :, 1:, :] = data_tensor[:, :, 1:, :] - data_tensor[:, :, 0:1, :]
        data_tensor = torch.cat([data_tensor, rel], dim=1)  # [1, 4, T, 18]
    return data_tensor


def score_ctd(sparta_model, loss_func, data_tensor, seg_len):
    input_seq = data_tensor.permute(0, 2, 1, 3).contiguous().view(1, seg_len, -1)
    ctd_out = sparta_model.CTD(input_seq, input_seq)
    ctd_loss = loss_func.calculate(input_seq, ctd_out)
    return float(ctd_loss.mean().item())


def score_ftd(sparta_model, loss_func, data_tensor, seg_len):
    input_data = data_tensor[:, :, :seg_len, :]
    target_data = data_tensor[:, :, seg_len:(2 * seg_len), :]
    input_seq = input_data.permute(0, 2, 1, 3).contiguous().view(1, seg_len, -1)
    target_seq = target_data.permute(0, 2, 1, 3).contiguous().view(1, seg_len, -1)
    pred = sparta_model.FTD(input_seq, target_seq)
    ftd_loss = loss_func.calculate(target_seq, pred)
    return float(ftd_loss.mean().item())

def main():
    args = get_args()
    
    print(f"Starting Integrated Inference Pipeline")
    print(f"Video: {args.video_path}")
    print(f"Device: {args.device}")

    # 1. Load YOLOv26
    print(f"Loading YOLO model from {args.yolo_model}...")
    yolo_model = YOLO(args.yolo_model)

    # 2. Load ViTPose
    vit_pose_model = None
    if MMPOSE_AVAILABLE:
        if args.vit_config and args.vit_ckpt:
            print(f"Loading ViTPose model...")
            vit_pose_model = init_model(args.vit_config, args.vit_ckpt, device=args.device)
        else:
            print("Warning: ViTPose config or checkpoint not provided. Falling back to YOLO Pose if possible or skipping ViT.")
    
    # 3. Load SPARTA_H
    print("Loading SPARTA models...")
    sparta_args = MockArgs(
        num_heads=12, num_layers=4, latent_dim=64, token_config='pst',
        loss='mse', a=1.0, b=1.0, c=1.0, d=1.0,
        num_kp=18, seg_len=args.seg_len, relative=True, traj=False,
        norm_scale=0, prop_norm_scale=1, headless=False, device=args.device, dropout=0.0
    )
    
    input_dim = sparta_args.num_kp * 2
    expand_ratio = 2 if sparta_args.relative else 1
    
    sparta_model = SPARTA_H(
        d_model=input_dim * expand_ratio,
        num_heads=sparta_args.num_heads,
        d_feedforward=sparta_args.latent_dim,
        num_layers=sparta_args.num_layers,
        max_len=1000,
        device=args.device,
        dropout=sparta_args.dropout
    )
    
    try:
        checkpoint_c = torch.load(args.ctd_path, map_location=args.device, weights_only=False)
        checkpoint_f = torch.load(args.ftd_path, map_location=args.device, weights_only=False)
        sparta_model.CTD.load_state_dict(checkpoint_c['state_dict'])
        sparta_model.FTD.load_state_dict(checkpoint_f['state_dict'])
        print("SPARTA weights loaded successfully.")
    except Exception as e:
        print(f"Error loading SPARTA weights: {e}")
        return

    sparta_model.to(args.device)
    sparta_model.eval()
    loss_func = CostumLoss(sparta_args.loss, a=sparta_args.a, b=sparta_args.b, c=sparta_args.c, d=sparta_args.d)

    # 4. Video Processing
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save_output, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames...")
    
    # Per-person pose buffers: track_id -> list[[17, 2], ...]
    pose_buffers = {}
    track_score_ema = {}
    ctd_stats = RunningStats()
    ftd_stats = RunningStats()
    calib_scores = []
    learned_threshold = None

    if args.score_mode == "ctd":
        required_frames = args.seg_len
    elif args.score_mode == "ftd":
        required_frames = 2 * args.seg_len
    else:
        required_frames = 2 * args.seg_len

    max_buffer_len = max(required_frames, args.seg_len * max(2, args.max_buffer_mult))
    print(f"Scoring mode: {args.score_mode}")
    print(f"Required frames per track: {required_frames}")
    if args.anomaly_threshold is None:
        print(f"Adaptive threshold enabled: calib_frames={args.calib_frames}, percentile={args.calib_percentile}")
    else:
        print(f"Fixed threshold: {args.anomaly_threshold}")

    frame_idx = 0
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- STEP A: YOLOv26 Person Detection + Tracking ---
            results = yolo_model.track(frame, persist=True, classes=[0], conf=args.conf_thres, verbose=False)

            frame_max_anomaly = None
            connection_pairs = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
                (5, 11), (6, 12), (5, 6),
                (5, 7), (6, 8), (7, 9), (8, 10),
                (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                (3, 5), (4, 6)
            ]
            default_color = (0, 255, 0)

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = None
                if getattr(results[0].boxes, "id", None) is not None:
                    track_ids_raw = results[0].boxes.id.cpu().numpy()
                    fallback_ids = np.arange(len(boxes), dtype=np.float32)
                    track_ids = np.where(np.isfinite(track_ids_raw), track_ids_raw, fallback_ids).astype(int)
                else:
                    # Fallback when tracker IDs are unavailable
                    track_ids = np.arange(len(boxes), dtype=int)

                yolo_kpts = None
                yolo_kpt_confs = None
                if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
                    yolo_kpts = results[0].keypoints.xy.cpu().numpy()
                    if getattr(results[0].keypoints, "conf", None) is not None:
                        yolo_kpt_confs = results[0].keypoints.conf.cpu().numpy()

                for det_idx, bx in enumerate(boxes):
                    track_id = int(track_ids[det_idx])
                    current_pose = None
                    pose_conf = 1.0

                    # --- STEP B: Keypoint Extraction per person ---
                    if vit_pose_model and MMPOSE_AVAILABLE:
                        pose_results = inference_topdown(vit_pose_model, frame, bx[None, :], bbox_format='xyxy')
                        if len(pose_results) > 0:
                            pred = pose_results[0].pred_instances
                            current_pose = np.array(pred.keypoints[0], dtype=np.float32)
                            if hasattr(pred, "keypoint_scores"):
                                pose_conf = float(np.mean(pred.keypoint_scores[0]))
                    elif yolo_kpts is not None and det_idx < len(yolo_kpts):
                        current_pose = np.array(yolo_kpts[det_idx], dtype=np.float32)
                        if yolo_kpt_confs is not None and det_idx < len(yolo_kpt_confs):
                            pose_conf = float(np.mean(yolo_kpt_confs[det_idx]))

                    if current_pose is None or current_pose.shape[0] != 17:
                        continue
                    if pose_conf < args.min_pose_conf:
                        continue

                    person_buffer = pose_buffers.setdefault(track_id, [])
                    person_buffer.append(current_pose)
                    if len(person_buffer) > max_buffer_len:
                        pose_buffers[track_id] = person_buffer[-max_buffer_len:]
                        person_buffer = pose_buffers[track_id]

                    # --- STEP C: SPARTA anomaly score per person ---
                    anomaly_score = None
                    ctd_score = None
                    ftd_score = None
                    color = default_color
                    if args.score_mode in ("ctd", "both") and len(person_buffer) >= args.seg_len:
                        seq_17 = np.array(person_buffer[-args.seg_len:])  # [L, 17, 2]
                        seq_18 = keypoints17_to_coco18(seq_17)  # [L, 18, 2]
                        seq_norm = normalize_window_training_style(seq_18, width, height)
                        data_tensor = build_sparta_tensor(seq_norm, relative=sparta_args.relative).to(args.device)
                        ctd_score = score_ctd(sparta_model, loss_func, data_tensor, args.seg_len)
                        ctd_stats.update(ctd_score)

                    if args.score_mode in ("ftd", "both") and len(person_buffer) >= 2 * args.seg_len:
                        seq_17_2x = np.array(person_buffer[-(2 * args.seg_len):])  # [2L, 17, 2]
                        seq_18_2x = keypoints17_to_coco18(seq_17_2x)  # [2L, 18, 2]
                        seq_norm_2x = normalize_window_training_style(seq_18_2x, width, height)
                        data_tensor_2x = build_sparta_tensor(seq_norm_2x, relative=sparta_args.relative).to(args.device)
                        ftd_score = score_ftd(sparta_model, loss_func, data_tensor_2x, args.seg_len)
                        ftd_stats.update(ftd_score)

                    if args.score_mode == "ctd":
                        anomaly_score = ctd_score
                    elif args.score_mode == "ftd":
                        anomaly_score = ftd_score
                    else:
                        if ctd_score is not None and ftd_score is not None:
                            # Z-score fusion can go negative for below-mean behavior; clamp to keep anomaly score >= 0.
                            fused_score = 0.5 * (ctd_stats.zscore(ctd_score) + ftd_stats.zscore(ftd_score))
                            anomaly_score = max(0.0, fused_score)

                    if anomaly_score is not None:
                        anomaly_score = max(0.0, float(anomaly_score))
                        prev = track_score_ema.get(track_id, anomaly_score)
                        anomaly_score = args.smooth_alpha * anomaly_score + (1.0 - args.smooth_alpha) * prev
                        anomaly_score = max(0.0, float(anomaly_score))
                        track_score_ema[track_id] = anomaly_score
                        frame_max_anomaly = anomaly_score if frame_max_anomaly is None else max(frame_max_anomaly, anomaly_score)
                        if args.anomaly_threshold is None and frame_idx < args.calib_frames:
                            calib_scores.append(anomaly_score)

                    min_scores_for_threshold = max(20, args.seg_len)
                    dynamic_threshold = None
                    if args.anomaly_threshold is None and len(calib_scores) >= min_scores_for_threshold:
                        dynamic_threshold = float(np.percentile(np.array(calib_scores), args.calib_percentile))

                    if learned_threshold is None and args.anomaly_threshold is None and frame_idx >= args.calib_frames and dynamic_threshold is not None:
                        learned_threshold = dynamic_threshold
                        print(f"[calibration] Learned threshold: {learned_threshold:.4f}")

                    threshold = args.anomaly_threshold if args.anomaly_threshold is not None else (
                        learned_threshold if learned_threshold is not None else dynamic_threshold
                    )
                    if anomaly_score is not None and threshold is not None and anomaly_score > threshold:
                        color = (0, 0, 255)

                    # --- Visualization: smaller keypoints, thinner connectors, ID only ---
                    for x, y in current_pose:
                        cv2.circle(frame, (int(x), int(y)), 2, color, -1)

                    for s, e in connection_pairs:
                        p1 = tuple(map(int, current_pose[s]))
                        p2 = tuple(map(int, current_pose[e]))
                        cv2.line(frame, p1, p2, color, 1)

                    x1, y1 = int(bx[0]), int(bx[1])
                    cv2.putText(
                        frame,
                        f"ID:{track_id}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )

            min_scores_for_threshold = max(20, args.seg_len)
            dynamic_threshold = None
            if args.anomaly_threshold is None and len(calib_scores) >= min_scores_for_threshold:
                dynamic_threshold = float(np.percentile(np.array(calib_scores), args.calib_percentile))
            threshold = args.anomaly_threshold if args.anomaly_threshold is not None else (
                learned_threshold if learned_threshold is not None else dynamic_threshold
            )
            current_score = frame_max_anomaly if frame_max_anomaly is not None else 0.0
            # No frame-level numeric status; visual color on skeleton indicates anomaly.
            
            out.write(frame)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Progress: {frame_idx}/{total_frames}")

    cap.release()
    out.release()
    print(f"Done! Result saved to {args.save_output}")

if __name__ == "__main__":
    main()
