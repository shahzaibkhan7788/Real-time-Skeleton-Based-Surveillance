# SPARTA: Human-Centric Video Anomaly Detection

This repository contains an updated SPARTA pipeline for pose-based anomaly detection on surveillance videos.

It includes:
- The SPARTA model code (`models.py`, `main.py`, `utils/`)
- Pretrained checkpoints for multiple datasets (`Trained_Models/`)
- End-to-end single-video inference with multi-person tracking (`inference_sparta_vit.py`)
- Optional anomaly graph generation (`inference_with_graph.py`)
- Pose extraction utilities (`PoseEstimationModel/`)

## 1) What The Project Does

SPARTA uses human pose sequences instead of raw RGB pixels to detect anomalies.

Core model idea:
- A shared encoder
- Two decoders:
1. `CTD` (Current Target Decoder): reconstructs current pose sequence
2. `FTD` (Future Target Decoder): predicts future pose sequence

At inference time, anomaly score is based on reconstruction/prediction error:
- `score_mode=ctd`: only CTD
- `score_mode=ftd`: only FTD
- `score_mode=both`: fused CTD + FTD score

## 2) Repository Layout

Key files and folders:

```text
.
├── inference_sparta_vit.py               # Main multi-person video inference
├── inference_with_graph.py               # Inference + anomaly score plot
├── main.py                               # Training/evaluation entry point
├── args.py                               # Training/eval arguments
├── models.py                             # SPARTA model definitions
├── dataset.py                            # Dataset loading
├── utils/
│   ├── train_utils.py
│   ├── eval.py
│   ├── tokenizer.py
│   ├── pose_utils.py
│   └── data_utils.py
├── PoseEstimationModel/
│   ├── 01_detect_and_track.py
│   ├── 02_estimate_pose_vit.py
│   ├── pose_estimate.py
│   └── ...
└── Trained_Models/
    ├── SHT/
    ├── CHAD/
    └── NWPUC/
```

## 3) Model Weights (Google Drive)

Paste your Drive link here:

- `Weights Link`: `<PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE>`

Expected local checkpoint layout:

```text
Trained_Models/
  SHT/
    CTD.pth.tar
    FTD.pth.tar
  CHAD/
    CTD.pth.tar
    FTD.pth.tar
  NWPUC/
    CTD.pth.tar
    FTD.pth.tar
```

You can select which dataset weights to use by passing:
- `--ctd_path`
- `--ftd_path`

## 4) Environment Setup

If you already created `sparta-venv`, use:

```bash
CONDA_NO_PLUGINS=true conda run -n sparta-venv env PYTHONNOUSERSITE=1 \
python -m pip install -r requirements.txt
```

Optional dependencies:
- `tensorboard` (needed by `utils/train_utils.py` imports)
- `matplotlib` (for `inference_with_graph.py`)
- `mmpose` + `mmcv` (only if you want ViTPose instead of YOLO-pose fallback)

## 5) Quick Start: Run On One Video

Recommended command (adaptive threshold, fused CTD+FTD):

```bash
CONDA_NO_PLUGINS=true conda run -n sparta-venv env PYTHONNOUSERSITE=1 \
python inference_sparta_vit.py \
  --video_path "/absolute/path/to/video.mp4" \
  --yolo_model "yolo26x-pose.pt" \
  --ctd_path "Trained_Models/CHAD/CTD.pth.tar" \
  --ftd_path "Trained_Models/CHAD/FTD.pth.tar" \
  --score_mode both \
  --calib_frames 200 \
  --calib_percentile 98.5 \
  --smooth_alpha 0.2 \
  --min_pose_conf 0.2 \
  --device cuda \
  --save_output "out.mp4"
```

Output:
- Annotated video with person IDs, skeletons, and anomaly scores.

## 6) How Inference Works

`inference_sparta_vit.py` pipeline:

1. Detect + track people with YOLOv26 pose.
2. Extract per-person keypoints (ViTPose if configured, otherwise YOLO keypoints).
3. Keep a per-track temporal pose buffer.
4. Convert COCO-17 keypoints to SPARTA COCO-18 order.
5. Normalize pose window with training-style normalization.
6. Compute score from CTD/FTD/Both.
7. Apply score smoothing.
8. Apply threshold logic:
   - Fixed threshold if `--anomaly_threshold` is provided.
   - Adaptive percentile threshold otherwise.

## 7) Score Modes

- `ctd`
  - Uses reconstruction branch only
  - Needs `seg_len` frames per person track
- `ftd`
  - Uses future prediction branch only
  - Needs `2 * seg_len` frames per person track
- `both`
  - Uses both branches
  - Fuses scores
  - Score is clamped to be non-negative

## 8) Thresholding Guidance

Important: pretrained checkpoints do not store a ready-to-use inference threshold.

What is stored:
- model weights (`state_dict`)
- optimizer state
- training args

What is not stored:
- per-dataset final threshold like `eer_th` or inference cutoff

In this codebase, thresholds are computed during evaluation from ground-truth masks (`utils/eval.py`), not embedded in weights.

Recommended strategy for deployment:
1. Use adaptive threshold mode for quick testing.
2. For production, compute dataset/camera-specific threshold on a validation set and pass it with `--anomaly_threshold`.

## 9) Common Problem: Too Many Red Alerts

If normal actions (like sitting) appear anomalous:

1. Increase calibration percentile:
   - Example: `--calib_percentile 99` or `99.5`
2. Increase smoothing:
   - Example: `--smooth_alpha 0.15` to stabilize noise
3. Raise minimum pose confidence:
   - Example: `--min_pose_conf 0.25`
4. Ensure you use matching dataset weights for your domain:
   - CHAD-like scenes -> CHAD weights
   - Campus scenes -> SHT/NWPUC depending on domain similarity

## 10) Head/Shoulder Skeleton Connectivity

Head to shoulder links are included in updated skeleton drawing:
- `(3, 5)` (left ear to left shoulder)
- `(4, 6)` (right ear to right shoulder)

Implemented in:
- `inference_sparta_vit.py`
- `PoseEstimationModel/02_estimate_pose_vit.py`
- `PoseEstimationModel/pose_estimate.py`

## 11) Optional: Generate Anomaly Graph

Run:

```bash
CONDA_NO_PLUGINS=true conda run -n sparta-venv env PYTHONNOUSERSITE=1 \
python inference_with_graph.py \
  --video_path "/absolute/path/to/video.mp4" \
  --yolo_model "yolo26x-pose.pt" \
  --ctd_path "Trained_Models/CHAD/CTD.pth.tar" \
  --ftd_path "Trained_Models/CHAD/FTD.pth.tar" \
  --score_mode both \
  --device cuda
```

Outputs:
- annotated video
- graph image (frame index vs anomaly score)

## 12) Legacy Pose Pipeline Scripts

If you want to run the explicit staged pipeline:

1. `PoseEstimationModel/01_detect_and_track.py`
   - Person detection + CSV generation
2. `PoseEstimationModel/02_estimate_pose_vit.py`
   - ViTPose estimation from CSV bboxes
3. `PoseEstimationModel/sparta_adapter_vit.py`
   - SPARTA scoring over exported pose JSON
4. `PoseEstimationModel/visualize_sparta_results.py`
   - Score visualization video

## 13) Training and Evaluation

Train CTD:

```bash
python main.py --dataset ShanghaiTech --branch SPARTA_C \
  --mask_root <mask_dir> --vid_res <W,H> --seg_len 12 --seg_stride 12 \
  --num_kp 18 --model_num_heads 12 --model_num_layers 4 --relative \
  --model_loss mse --token_config pst --batch_size 512 --model_latent_dim 64
```

Train FTD:

```bash
python main.py --dataset ShanghaiTech --branch SPARTA_F \
  --mask_root <mask_dir> --vid_res <W,H> --seg_len 12 --seg_stride 12 \
  --num_kp 18 --model_num_heads 12 --model_num_layers 4 --relative \
  --model_loss mse --token_config pst --batch_size 512 --model_latent_dim 64 \
  --recon_encoder_path <trained_CTD_path>
```

Evaluate hybrid:

```bash
python main.py --dataset ShanghaiTech --branch SPARTA_H \
  --model_ckpt_C <CTD_path> --model_ckpt_F <FTD_path> \
  --mask_root <mask_dir> --vid_res <W,H> --seg_len 12 --seg_stride 12 \
  --num_kp 18 --model_num_heads 12 --model_num_layers 4 --relative \
  --model_loss mse --token_config pst --batch_size 512 --model_latent_dim 64
```

## 14) Citation

```bibtex
@misc{noghre2025humancentricvideoanomalydetection,
  title={Human-Centric Video Anomaly Detection Through Spatio-Temporal Pose Tokenization and Transformer},
  author={Ghazal Alinezhad Noghre and Armin Danesh Pazho and Hamed Tabkhi},
  year={2025},
  eprint={2408.15185},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2408.15185}
}
```
