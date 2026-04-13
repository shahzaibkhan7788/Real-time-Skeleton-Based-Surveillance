import streamlit as st
import yaml
import tempfile
import re
import random
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

def main():
    st.set_page_config(page_title="Pose Estimation UI", page_icon="🕺", layout="wide")
    st.title("Pose Estimation Dashboard")
    
    col_left, col_right = st.columns(2) # FIXED: Added '2' here

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

    with col_right:
        st.subheader("Video Upload")
        upload = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
        video_path = None
        if upload:
            video_path = save_upload(upload)
            st.success(f"File saved: {video_path.name}")
            st.video(upload)

    st.divider()
    
    if st.button("🚀 Start Processing", use_container_width=True):
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
                pipeline.run()
            st.balloons()
            
            # --- Show Results after successful run ---
            st.subheader("Results")
            res_prefix = cfg["paths"]["static_prefix"]
            expected_stem = get_expected_name(video_path.stem, res_prefix)
            
            out_dir = Path(cfg["paths"]["pose_output_dir"])
            pose_vis_dir = out_dir / "pose_vis"
            vis_file = pose_vis_dir / f"{expected_stem}_vis.avi"

            if save_video:
                if vis_file.exists():
                    st.success(f"Video generated: {vis_file.name}")
                    with open(vis_file, "rb") as f:
                        st.download_button("📥 Download Result Video", f, file_name=vis_file.name)
                else:
                    st.warning(f"Video was requested but not found at {vis_file}")

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
        finally:
            if 'tmp_path' in locals():
                Path(tmp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
