import streamlit as st
import yaml
import tempfile
import re
import random
import subprocess
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

# Local pipeline imports
from PoseEstimationModel.pose_estimation import Config, PosePipeline

# --------- Constants & helpers ---------

BASE_DIR = Path(__file__).parent
POSE_DIR = BASE_DIR / "PoseEstimationModel"
BASE_CONFIG_PATH = POSE_DIR / "config.yaml"
UPLOAD_DIR = POSE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

def load_base_cfg() -> Dict:
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

BASE_CFG = load_base_cfg()
POSE_WEIGHTS = BASE_CFG["models"]["pose"]["weights"]
DET_WEIGHTS = BASE_CFG["models"]["detection"].get("weights", {})

def variants_for(family: str) -> Tuple[str, ...]:
    block = POSE_WEIGHTS.get(family, {})
    return tuple(block.keys())

POSE_VARIANTS = {
    "vitpose": variants_for("vitpose"),
    "rtm": variants_for("rtm"),
    "yolo-pose": tuple(POSE_WEIGHTS.get("yolo-pose", {}).keys()),
}

DET_VARIANTS = tuple(DET_WEIGHTS.get("yolo", {}).keys() or ("x", "l", "m", "s", "n"))

PRESETS = {
    "Real-time (fastest)": {
        "pose": ("yolo-pose", "n"),
        "det": "n",
        "blurb": "Uses YOLO-Pose nano; best for real-time with minimal latency.",
    },
    "Balanced": {
        "pose": ("rtm", "s"),
        "det": "s",
        "blurb": "RTMPose-small gives a good speed/accuracy trade-off.",
    },
    "High accuracy": {
        "pose": ("vitpose", "large"),
        "det": "x",
        "blurb": "ViTPose-large maximizes keypoint quality (heavier).",
    },
    "Custom": None,
}

def save_upload(upload) -> Path:
    target = UPLOAD_DIR / upload.name
    with open(target, "wb") as f:
        f.write(upload.getbuffer())
    return target

def get_expected_name(stem: str, prefix: str) -> str:
    """Replicates the 01_xxxx logic from pose_estimation.py"""
    digits = re.findall(r'\d+', stem)
    if digits:
        full_digit_str = "".join(digits)
        suffix = full_digit_str[-4:] if len(full_digit_str) >= 4 else full_digit_str.zfill(4)
    else:
        suffix = "0000" # Fallback if no digits found
    return f"{prefix}{suffix}"

def build_run_config(
    base_cfg: Dict,
    pose_family: str,
    pose_variant: str,
    det_variant: str,
    device: str,
    video_path: Path,
    save_video: bool,
) -> Dict:
    cfg = deepcopy(base_cfg)
    cfg["models"]["pose"]["name"] = pose_family
    cfg["models"]["pose"]["variant"] = pose_variant
    cfg["models"]["detection"]["name"] = "yolo"
    cfg["models"]["detection"]["variant"] = det_variant
    
    if device != "auto":
        cfg["models"]["pose"]["device"] = device
        cfg["models"]["detection"]["device"] = device

    cfg["models"]["pose"]["save_video"] = bool(save_video)
    cfg["paths"]["input_video"] = str(video_path)
    cfg["paths"]["static_prefix"] = "01_"
    cfg["paths"]["pose_json_suffix"] = ".json"
    return cfg


def build_sparta_config(base_cfg: Dict, sparta_branch: str, ckpt_c: str, ckpt_f: str, pose_json_dir: Path, device: str) -> Dict:
    ckpt_block = base_cfg.get("models", {}).get("sparta", {}).get("checkpoints", {})
    th_c = ckpt_block.get("eer_threshold_c")
    th_f = ckpt_block.get("eer_threshold_f")
    sparta_cfg = {
        "mode": "test",
        "no_metrics": True,
        "save_results": True,
        "save_results_dir": str(Path(base_cfg["paths"].get("sparta_output_dir", "evaluation_results_sparta")).resolve()),
        "mask_root": None,
        "pose_path_test": str(pose_json_dir),
        "vid_path_test": None,
        "branch": sparta_branch,
        "relative": base_cfg.get("models", {}).get("sparta", {}).get("relative", True),
        "token_config": base_cfg.get("models", {}).get("sparta", {}).get("token_config", "t"),
        "num_kp": base_cfg.get("models", {}).get("sparta", {}).get("num_kp", 18),
        "seg_len": base_cfg.get("models", {}).get("sparta", {}).get("seg_len", 24),
        "model_num_heads": base_cfg.get("models", {}).get("sparta", {}).get("model_num_heads", 2),
        "model_latent_dim": base_cfg.get("models", {}).get("sparta", {}).get("model_latent_dim", 64),
        "dropout": base_cfg.get("models", {}).get("sparta", {}).get("dropout", 0.3),
        "batch_size": base_cfg.get("batch_size", 256) if isinstance(base_cfg.get("batch_size", 256), int) else 256,
        "device": device if device != "auto" else base_cfg.get("models", {}).get("pose", {}).get("device", "cpu"),
        "dataset": base_cfg.get("dataset", "corridor"),
    }
    if sparta_branch == "SPARTA_H":
        sparta_cfg["model_ckpt_C"] = ckpt_c
        sparta_cfg["model_ckpt_F"] = ckpt_f
        sparta_cfg["eer_threshold_c"] = th_c
        sparta_cfg["eer_threshold_f"] = th_f
    else:
        sparta_cfg["model_ckpt_dir"] = ckpt_c  # for C or F we treat ckpt_c as main ckpt
        if sparta_branch == "SPARTA_C":
            sparta_cfg["eer_threshold_c"] = th_c
        else:
            sparta_cfg["eer_threshold_f"] = th_f
    return sparta_cfg

def main():
    st.set_page_config(page_title="Pose & SPARTA Lab", page_icon="🕺", layout="wide")
    st.markdown(
        """
        <style>
        .big-button button {width:100%; border-radius:12px; height:3rem; font-weight:700;}
        .metric-card {padding:12px 16px; border-radius:12px; background:#0c111c0d; border:1px solid #e5e7eb;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Pose & SPARTA Lab")
    st.caption("Upload a video → extract poses → run SPARTA anomaly scoring (C/F/H) → download scores.")
    
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Settings")
        preset = st.radio("Select Profile", list(PRESETS.keys()), horizontal=True)
        preset_info = PRESETS.get(preset)

        if preset_info:
            pose_family, pose_variant = preset_info["pose"]
            det_variant = preset_info["det"]
            st.info(preset_info["blurb"])
        else:
            pose_family = st.selectbox("Pose Model", list(POSE_VARIANTS.keys()))
            pose_variant = st.selectbox("Variant", POSE_VARIANTS.get(pose_family, ["large"]))
            det_variant = st.selectbox("Detection (YOLO26)", DET_VARIANTS)

        device = st.selectbox("Compute Device", ["cuda:0", "cpu", "auto"])
        save_video = st.toggle("Generate Visualization Video", value=True)
        st.divider()
        st.markdown("#### SPARTA Settings")
        sparta_branch = st.selectbox("SPARTA variant", ["SPARTA_C", "SPARTA_F", "SPARTA_H"])
        ckpt_defaults = BASE_CFG.get("models", {}).get("sparta", {}).get("checkpoints", {})
        if sparta_branch == "SPARTA_H":
            ckpt_c = st.text_input("Checkpoint SPARTA_C", ckpt_defaults.get("sparta_h_c", ""))
            ckpt_f = st.text_input("Checkpoint SPARTA_F", ckpt_defaults.get("sparta_h_f", ""))
        else:
            default_ckpt = ckpt_defaults.get("sparta_c" if sparta_branch == "SPARTA_C" else "sparta_f", "")
            ckpt_c = st.text_input("Checkpoint path", default_ckpt)
            ckpt_f = ""

    with col_right:
        st.subheader("Video Upload")
        upload = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
        video_path = None
        if upload:
            video_path = save_upload(upload)
            st.success(f"File saved: {video_path.name}")
            st.video(upload)

    st.divider()
    
    if st.button("🚀 Start Processing", use_container_width=True, type="primary"):
        if not video_path:
            st.error("Please upload a video first!")
            return

        cfg = build_run_config(BASE_CFG, pose_family, pose_variant, det_variant, device, video_path, save_video)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=POSE_DIR, delete=False) as tmp:
            yaml.safe_dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            with st.spinner("AI is analyzing video... This may take a minute."):
                pipeline = PosePipeline(tmp_path)
                final_json = pipeline.run()
            if final_json is None:
                st.error("No persons detected; SPARTA inference skipped.")
                return
            
            # --- Show Results after successful run ---
            st.subheader("Results")
            res_prefix = cfg["paths"]["static_prefix"]
            expected_stem = get_expected_name(video_path.stem, res_prefix)
            
            out_dir = Path(cfg["paths"]["pose_output_dir"])
            pose_vis_dir = out_dir / "pose_vis"
            vis_file = pose_vis_dir / f"{expected_stem}_vis.avi"

            tab_pose, tab_sparta = st.tabs(["🎥 Pose Output", "⚡ SPARTA Scores"])

            with tab_pose:
                if save_video:
                    if vis_file.exists():
                        st.success(f"Video generated: {vis_file.name}")
                        with open(vis_file, "rb") as f:
                            st.download_button("📥 Download Result Video", f, file_name=vis_file.name)
                    else:
                        st.warning(f"Video was requested but not found at {vis_file}")
                st.info(f"Pose JSON saved to: {final_json}")

            with tab_sparta:
                st.write("Runs SPARTA on the extracted pose JSON; no masks required.")
                pose_json_dir = Path(final_json).parent
                sparta_cfg = build_sparta_config(BASE_CFG, sparta_branch, ckpt_c, ckpt_f, pose_json_dir, device)
                Path(sparta_cfg["save_results_dir"]).mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=BASE_DIR, delete=False) as tmp_sparta:
                    yaml.safe_dump(sparta_cfg, tmp_sparta)
                    tmp_sparta_path = tmp_sparta.name
                with st.spinner("Running SPARTA inference..."):
                    subprocess.run(["python", "main.py", "--config", tmp_sparta_path], cwd=BASE_DIR, check=True)
                scores_files = sorted(Path(sparta_cfg["save_results_dir"]).glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                if scores_files:
                    latest = scores_files[0]
                    df = pd.read_csv(latest)
                    score_col = "score" if "score" in df.columns else df.columns[-1]
                    st.line_chart(df[score_col], height=250)
                    with open(latest, "rb") as f:
                        st.download_button("📥 Download scores CSV", f, file_name=latest.name)
                else:
                    st.warning("No SPARTA score file was produced.")

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
        finally:
            if 'tmp_path' in locals():
                Path(tmp_path).unlink(missing_ok=True)
            if 'tmp_sparta_path' in locals():
                Path(tmp_sparta_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
