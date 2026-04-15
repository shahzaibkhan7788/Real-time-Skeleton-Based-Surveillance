import streamlit as st
import yaml
import tempfile
import re
import random
import subprocess
import pandas as pd
import cv2
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
    friendly_branch_map = {
        "Reconstruction Model": "SPARTA_C",
        "Future trajecory prediction model": "SPARTA_F",
        "Hybrid": "SPARTA_H",
    }
    sparta_branch = friendly_branch_map.get(sparta_branch, sparta_branch)
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
    st.set_page_config(page_title="Human Surveillance Application", page_icon="🕺", layout="wide")
    st.markdown(
        """
        <style>
        :root{
          --brand:#3b82f6;
          --brand-2:#60a5fa;
          --brand-dark:#1d4ed8;
          --ink:#0f172a;
          --muted:#475569;
          --border:rgba(148,163,184,0.55);
        }

        /* Sharper, cleaner overall look */
        [data-testid="stAppViewContainer"]{
          background:
            radial-gradient(900px 420px at 12% 0%, rgba(59,130,246,0.22), transparent 62%),
            radial-gradient(760px 460px at 92% 12%, rgba(96,165,250,0.16), transparent 56%),
            linear-gradient(180deg, #ffffff 0%, #f8fafc 68%, #f1f5f9 100%);
        }
        [data-testid="stHeader"]{ background: transparent; }
	        section.main > div.block-container{
	          padding-top: 2.1rem;
	          padding-bottom: 2.4rem;
	        }
	        h1, h2, h3{ color: var(--ink); letter-spacing: -0.01em; }
	        [data-testid="stCaptionContainer"]{ color: rgba(71,85,105,0.95); }

	        /* Two-panel layout as clean cards */
	        div[data-testid="stHorizontalBlock"]{
	          gap: 1.25rem;
	        }
	        div[data-testid="column"]{
	          background: rgba(255,255,255,0.86);
	          border: 1px solid rgba(148,163,184,0.42);
	          border-radius: 16px;
	          padding: 0.95rem 0.95rem 0.55rem;
	          box-shadow: 0 14px 28px rgba(15,23,42,0.06);
	        }

	        /* Primary button -> light blue (not red) */
	        button[kind="primary"], div[data-testid="baseButton-primary"] button, button[data-testid="baseButton-primary"]{
	          background: linear-gradient(90deg, var(--brand-dark) 0%, var(--brand) 45%, var(--brand-2) 100%) !important;
	          border: 1px solid rgba(29,78,216,0.85) !important;
	          color: #ffffff !important;
	          border-radius: 14px;
	          height: 3.05rem;
	          font-weight: 750;
	          box-shadow: 0 12px 22px rgba(29,78,216,0.20) !important;
	        }
	        button[kind="primary"]:hover, div[data-testid="baseButton-primary"] button:hover, button[data-testid="baseButton-primary"]:hover{
	          filter: saturate(1.08) brightness(1.02);
	          box-shadow: 0 16px 28px rgba(29,78,216,0.26);
	          transform: translateY(-1px);
	        }
	        button[kind="primary"]:active, div[data-testid="baseButton-primary"] button:active, button[data-testid="baseButton-primary"]:active{
	          transform: translateY(0px);
	        }

	        .big-button button {width:100%; border-radius:12px; height:3rem; font-weight:700;}
	        .metric-card {padding:12px 16px; border-radius:12px; background:#0c111c0d; border:1px solid #e5e7eb;}

	        /* Sharper inputs (select, text, number) */
	        div[data-baseweb="select"] > div{
	          border-radius: 12px;
	          border: 1px solid var(--border);
	          background: rgba(255,255,255,0.92);
	        }
	        div[data-baseweb="select"] > div:focus-within{
	          border-color: var(--brand);
	          box-shadow: 0 0 0 3px rgba(59,130,246,0.14);
	        }
	        div[data-testid="stTextInput"] input,
	        div[data-testid="stNumberInput"] input{
	          border-radius: 12px !important;
	          border: 1px solid var(--border) !important;
	          background: rgba(255,255,255,0.92) !important;
	        }
	        div[data-testid="stTextInput"] input:focus,
	        div[data-testid="stNumberInput"] input:focus{
	          border-color: var(--brand) !important;
	          box-shadow: 0 0 0 3px rgba(59,130,246,0.14) !important;
	        }

	        /* Tabs accent */
	        div[data-testid="stTabs"] button[aria-selected="true"]{
	          color: var(--brand-dark) !important;
	        }

	        /* --- Video uploader: clean light-blue dropzone --- */
	        :root{
	          --upload-accent:var(--brand-2);
          --upload-accent-strong:var(--brand);
          --upload-bg:rgba(79, 141, 247, 0.06);
          --upload-border:rgba(79, 141, 247, 0.35);
          --upload-border-strong:rgba(79, 141, 247, 0.65);
        }

        div[data-testid="stFileUploader"], section[data-testid="stFileUploader"]{
          border:1px solid var(--upload-border);
          border-radius:14px;
          padding:0.75rem 0.85rem 0.9rem;
          background:linear-gradient(180deg, var(--upload-bg), rgba(255,255,255,0.0));
        }

        div[data-testid="stFileUploader"] label, section[data-testid="stFileUploader"] label{
          font-weight:600;
        }

        div[data-testid="stFileUploaderDropzone"], section[data-testid="stFileUploaderDropzone"]{
          border:2px dashed var(--upload-border-strong);
          border-radius:12px;
          background:rgba(255,255,255,0.88);
        }

        div[data-testid="stFileUploaderDropzone"]:hover, section[data-testid="stFileUploaderDropzone"]:hover{
          border-color:var(--upload-accent-strong);
          box-shadow:0 0 0 3px rgba(59,130,246,0.12);
        }

        div[data-testid="stFileUploaderDropzone"] svg, section[data-testid="stFileUploaderDropzone"] svg{
          color:var(--upload-accent-strong);
        }

        div[data-testid="stFileUploaderDropzone"] button, section[data-testid="stFileUploaderDropzone"] button{
          border:1px solid var(--upload-border-strong);
          color:var(--upload-accent-strong);
          background:rgba(255,255,255,0.95);
        }

        div[data-testid="stFileUploaderDropzone"] button:hover,
        section[data-testid="stFileUploaderDropzone"] button:hover{
          border-color:var(--upload-accent-strong);
          background:rgba(59,130,246,0.08);
        }

        div[data-testid="stFileUploader"] small, section[data-testid="stFileUploader"] small{
          color:rgba(15, 23, 42, 0.72);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Human Surveillance Application")
    st.caption("Upload a video → Detection stage will execute --> key points extraction stage execute → anomaly scoring model will execute")
    
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Pose Estimation Model Selection")
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
        st.markdown("#### Spatio Temporal Pose Model selection setting")
        sparta_branch_display = st.selectbox("Scoring Variant", ["Reconstruction Model", "Future trajecory prediction model", "Hybrid"])
        sparta_branch_map = {
            "Reconstruction Model": "SPARTA_C",
            "Future trajecory prediction model": "SPARTA_F",
            "Hybrid": "SPARTA_H",
        }
        sparta_branch = sparta_branch_map.get(sparta_branch_display, sparta_branch_display)
        ckpt_defaults = BASE_CFG.get("models", {}).get("sparta", {}).get("checkpoints", {})
        if sparta_branch == "SPARTA_H":
            ckpt_c = st.text_input("Checkpoint (C)", ckpt_defaults.get("sparta_h_c", ""))
            ckpt_f = st.text_input("Checkpoint (F)", ckpt_defaults.get("sparta_h_f", ""))
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
    if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
        if not video_path:
            st.error("Please upload a video first!")
        else:
            # 1. Setup Config
            run_cfg = build_run_config(BASE_CFG, pose_family, pose_variant, det_variant, device, video_path, save_video)
            
            # Using a temporary file to avoid config conflicts
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=POSE_DIR, delete=False) as tmp:
                yaml.safe_dump(run_cfg, tmp)
                tmp_path = tmp.name

            try:
                # --- STEP 1: UNIFIED LIVE PROCESSING (Detection + Tracking + Pose) ---
                st.markdown("### 🎥 Live Analysis Preview (Original vs Processed)")
                
                # Create two columns for side-by-side display
                col_original, col_processed = st.columns(2)
                
                with col_original:
                    st.write("**📹 Original Video**")
                    original_video_placeholder = st.empty()
                    original_video_placeholder.video(upload)
                
                with col_processed:
                    st.write("**🤖 Processed Output**")
                    processed_frame_placeholder = st.empty()
                
                progress_bar = st.progress(0)
                status_text = st.empty()

                pipeline = PosePipeline(tmp_path)
                final_json_path = None
                
                try:
                    # Iterate through the unified pipeline generator
                    for frame, frame_id, total in pipeline.run_live():
                        # Convert BGR (OpenCV) to RGB (Streamlit)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Update progress
                        progress = (frame_id + 1) / total if total > 0 else 0
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"Processing Frame: {frame_id + 1} / {total}")
                    
                    # After generator completes, get the JSON path
                    res_prefix = run_cfg["paths"]["static_prefix"]
                    expected_json_name = get_expected_name(video_path.stem, res_prefix) + run_cfg["paths"]["pose_json_suffix"]
                    final_json_path = Path(run_cfg["paths"]["pose_output_dir"]) / expected_json_name
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")

                # --- STEP 2: ANOMALY SCORING (SPARTA) ---
                st.divider()
                st.subheader("⚡ Anomaly Intelligence")
                
                if final_json_path and final_json_path.exists():
                    sparta_cfg = build_sparta_config(BASE_CFG, sparta_branch, ckpt_c, ckpt_f, final_json_path.parent, device)
                    
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=BASE_DIR, delete=False) as tmp_sparta:
                        yaml.safe_dump(sparta_cfg, tmp_sparta)
                        tmp_sparta_path = tmp_sparta.name

                    with st.spinner("Calculating Anomaly Scores..."):
                        subprocess.run(["python", "main.py", "--config", tmp_sparta_path], cwd=BASE_DIR, check=True)

                    # Show the Chart
                    scores_dir = Path(sparta_cfg["save_results_dir"])
                    scores_files = sorted(scores_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                    
                    if scores_files:
                        df = pd.read_csv(scores_files[0])
                        score_col = "score" if "score" in df.columns else df.columns[-1]
                        st.line_chart(df[score_col], height=300)
                        st.success(f"✅ Anomaly scores saved to: {scores_files[0]}")
                else:
                    st.warning("⚠️ Pose JSON not found; skipping anomaly scoring.")
                    
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                import traceback
                st.error(f"Details: {traceback.format_exc()}")
            finally:
                if 'tmp_path' in locals(): 
                    Path(tmp_path).unlink(missing_ok=True)
                if 'tmp_sparta_path' in locals(): Path(tmp_sparta_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
