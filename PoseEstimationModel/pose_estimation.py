from __future__ import annotations

import json
import os
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import random
import re

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", "/tmp")

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes


# ---------- Config helpers ----------

@dataclass
class ModelPaths:
    config: Optional[Path] = None
    checkpoint: Optional[Path] = None
    weights: Optional[Path] = None


class Config:
    """YAML wrapper with path resolution and dynamic attribute access."""

    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent
        self.path = Path(config_path) if config_path else self.base_dir / "config.yaml"
        with open(self.path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        
        # Initialize mutable flags
        self.pose_save_video = self.cfg.get("models", {}).get("pose", {}).get("save_video", False)
        self.det_save_video = self.cfg.get("models", {}).get("detection", {}).get("save_video", False)

    def resolve(self, p: str | Path | None) -> Optional[Path]:
        if p is None: return None
        p = Path(p)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def detection_weight(self) -> ModelPaths:
        det = self.cfg["models"]["detection"]
        name = det.get("name", "")
        variant = det.get("variant") or "n"
        weights = det.get("weights", {})
        family = "yolo26" if name.startswith("yolo26") else "yolo"

        available = weights.get(family, {})
        w = available.get(variant)
        if w is None and available:
            # fallback to first available weight
            w = next(iter(available.values()))
        if w is None:
            # final fallback: default lightweight model
            w = "yolov8n.pt"

        resolved = self.resolve(w)
        # If resolved is None or doesn't exist, try any existing file from available
        if (resolved is None or not Path(resolved).exists()) and available:
            for candidate in available.values():
                cand_path = self.resolve(candidate)
                if cand_path and cand_path.exists():
                    resolved = cand_path
                    break
        if resolved is None or not Path(resolved).exists():
            raise FileNotFoundError(
                f"Detection weights not found for family='{family}' variant='{variant}'. "
                f"Checked: {resolved}. Available variants: {list(available.keys())}"
            )
        return ModelPaths(weights=resolved)

    def pose_paths(self) -> ModelPaths:
        pose = self.cfg["models"]["pose"]
        name = pose.get("name", "").lower()
        variant = pose.get("variant", "large")
        weights = pose.get("weights", {})
        
        if "vit" in name:
            block = weights.get("vitpose", {}).get(variant, {})
            cfg_path = self.resolve(block.get("config"))
            ckpt_path = self.resolve(block.get("checkpoint"))
            if cfg_path is None or ckpt_path is None or not cfg_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError(
                    f"VitPose weights/config not found for variant='{variant}'. "
                    f"cfg={cfg_path}, ckpt={ckpt_path}"
                )
            return ModelPaths(config=cfg_path, checkpoint=ckpt_path)
        if "rtm" in name:
            block = weights.get("rtm", {}).get(variant, {})
            cfg_path = self.resolve(block.get("config"))
            ckpt_path = self.resolve(block.get("checkpoint"))
            if cfg_path is None or ckpt_path is None or not cfg_path.exists() or not ckpt_path.exists():
                raise FileNotFoundError(
                    f"RTMPose weights/config not found for variant='{variant}'. "
                    f"cfg={cfg_path}, ckpt={ckpt_path}"
                )
            return ModelPaths(config=cfg_path, checkpoint=ckpt_path)
        
        block = weights.get("yolo-pose", {})
        w = block.get(variant) or block.get(name) or name
        resolved = self.resolve(w)
        if (resolved is None or not resolved.exists()) and block:
            # fallback to any existing file in block
            for candidate in block.values():
                cand_path = self.resolve(candidate)
                if cand_path and cand_path.exists():
                    resolved = cand_path
                    break
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(
                f"YOLO-Pose weights not found for variant='{variant}'. Expected at: {resolved}. "
                f"Available variants: {list(block.keys())}"
            )
        return ModelPaths(weights=resolved)

    @property
    def det_cfg(self) -> Dict[str, Any]: return self.cfg["models"]["detection"]
    
    @property
    def pose_cfg(self) -> Dict[str, Any]: return self.cfg["models"]["pose"]

    def resolved_paths(self) -> Dict[str, Path]:
        raw_paths = self.cfg.get("paths", {})
        paths = {"input_video": self.resolve(raw_paths["input_video"])}
        pose_root = self.resolve(raw_paths.get("pose_output_dir") or "../pose_outputs")
        paths["pose_output_dir"] = pose_root
        paths["pose_video_dir"] = pose_root / "pose_vis"
        return paths

    @property
    def pose_json_suffix(self) -> str:
        return self.cfg.get("paths", {}).get("pose_json_suffix", ".json")

    @property
    def static_prefix(self) -> str:
        return self.cfg.get("paths", {}).get("static_prefix", "01_")

    def human_centric_filename(self, video_stem: str) -> str:
        """Generate human-centric filename: 01_ + 4 digits (from video stem if possible)."""
        digits = "".join(re.findall(r"\d+", video_stem))
        if digits:
            suffix = digits[-4:].zfill(4)
        else:
            # deterministic fallback so UI can predict the filename
            suffix = "0000"
        return f"{self.static_prefix}{suffix}{self.pose_json_suffix}"


# ---------- Detection ----------

class PersonDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        det_params = cfg.det_cfg
        mp = cfg.detection_weight()
        
        self.model = YOLO(str(mp.weights))
        self.device = det_params.get("device", "cpu")
        self.model.to(self.device)

        self.conf = float(det_params.get("confidence_threshold", 0.3))
        self.iou = float(det_params.get("iou_threshold", 0.45))
        self.classes = det_params.get("classes") 
        
        trk_cfg = det_params.get("tracking", {})
        high_thresh = float(trk_cfg.get("track_high_thresh", self.conf))
        
        self.bt_args = Namespace(
            track_high_thresh=high_thresh,
            track_low_thresh=float(trk_cfg.get("track_low_thresh", 0.1)),
            new_track_thresh=float(trk_cfg.get("new_track_thresh", high_thresh)),
            match_thresh=float(trk_cfg.get("match_thresh", 0.8)),
            track_buffer=int(trk_cfg.get("track_buffer", 30)),
            min_box_area=float(trk_cfg.get("min_box_area", 10)),
            mot20=bool(trk_cfg.get("mot20", False)),
            fuse_score=bool(trk_cfg.get("fuse_score", False)),
            gmc=False,
        )

    def run(self, video_path: Path) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise RuntimeError(f"Cannot open: {video_path}")

        tracker = BYTETracker(self.bt_args, frame_rate=cap.get(cv2.CAP_PROP_FPS))
        records = []
        frame_id = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc="Detection & Tracking") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break

                kwargs = {"conf": self.conf, "iou": self.iou, "verbose": False, "device": self.device}
                if self.classes: kwargs["classes"] = self.classes

                # model() returns a list, we index to get the Results object
                results_list = self.model(frame, **kwargs)
                results = results_list[0]  # Results object

                if results.boxes is not None:
                    boxes = results.boxes
                    # Ensure tracker receives CPU tensors
                    if hasattr(boxes, "cpu"):
                        boxes = boxes.cpu()
                else:
                    boxes = Boxes(torch.zeros((0, 6)), frame.shape[:2])

                # BYTETracker expects a Boxes-like object with .conf/.cls attributes
                tracks = tracker.update(boxes, frame)

                for t in tracks:  # t: [x1, y1, x2, y2, track_id, score, cls, det_idx]
                    if len(t) >= 5:
                        x1, y1, x2, y2 = t[:4]
                        track_id = t[4]
                        score = t[5] if len(t) > 5 else None
                        cls = t[6] if len(t) > 6 else None
                        records.append({
                            "frame_id": frame_id,
                            "person_id": int(track_id),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "score": float(score) if score is not None else None,
                            "cls": int(cls) if cls is not None and not np.isnan(cls) else None,
                        })
                
                frame_id += 1
                pbar.update(1)

        cap.release()
        return records


# ---------- Pose Estimator Base ----------

class PoseEstimatorBase:
    """Abstract interface for pose estimators."""
    def process(self, video_path: Path, detections: List[Dict[str, Any]], output_dir: Path, json_suffix: str, output_name: Optional[str] = None) -> Path:
        raise NotImplementedError


# ---------- MMPose Concrete Estimator ----------

class MMPoseTopDownEstimator(PoseEstimatorBase):
    def __init__(self, cfg: Config, model_paths: ModelPaths):
        from mmpose.apis import inference_topdown, init_model
        self.cfg = cfg
        self.model = init_model(str(model_paths.config), str(model_paths.checkpoint), device=cfg.pose_cfg.get("device", "cpu"))
        self.inference_topdown = inference_topdown
        self.conf_threshold = float(cfg.pose_cfg.get("confidence_threshold", 0.25))
        self.expected_kp = None  # determined on first valid frame

    def _estimate(self, frame, bbox):
        results = self.inference_topdown(self.model, frame, [bbox])
        if not results:
            return None

        try:
            sample = results[0]  # we pass one bbox at a time
            inst = sample.pred_instances
            kpts = inst.keypoints  # shape: (num, K, 2)
            scores = inst.keypoint_scores  # shape: (num, K)
            if torch.is_tensor(kpts):
                kpts = kpts.cpu()
            if torch.is_tensor(scores):
                scores = scores.cpu()
            kpts_np = kpts[0] if len(kpts.shape) == 3 else kpts
            scores_np = scores[0] if len(scores.shape) == 2 else scores
            triplets = [[float(x), float(y), float(s)] for (x, y), s in zip(kpts_np, scores_np)]
            return {"keypoints": triplets, "mean": float(np.mean(scores_np))}
        except Exception:
            return None

    def process(
        self,
        video_path: Path,
        detections: List[Dict[str, Any]],
        output_dir: Path,
        json_suffix: str,
        output_name: Optional[str] = None,
        output_video_path: Optional[Path] = None,
    ) -> Path:
        cap = cv2.VideoCapture(str(video_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        sparta_json = defaultdict(dict)
        frame_map = defaultdict(list)
        for d in detections: frame_map[d["frame_id"]].append(d)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        if output_video_path and self.cfg.pose_save_video:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*self.cfg.pose_cfg.get("fourcc", "XVID")), fps, (w, h))

        frame_id = 0
        with tqdm(total=total, desc="Pose Estimation") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret: break
                for p in frame_map.get(frame_id, []):
                    pose = self._estimate(frame, p["bbox"])
                    if pose and pose["mean"] >= self.conf_threshold:
                        # Enforce consistent keypoint count
                        kp_count = len(pose["keypoints"])
                        if self.expected_kp is None:
                            self.expected_kp = kp_count
                        if kp_count != self.expected_kp:
                            print(f"[WARN] Skipping frame {frame_id} person {p['person_id']}: "
                                  f"kpt count {kp_count} != expected {self.expected_kp}")
                            continue
                        flat = [c for trip in pose["keypoints"] for c in trip]
                        sparta_json[str(p["person_id"])][str(frame_id)] = {
                            "keypoints": flat,
                            "scores": float(pose.get("mean", 0.0))
                        }
                        if writer is not None:
                            for (x, y, s) in pose["keypoints"]:
                                if s >= self.cfg.pose_cfg.get("min_keypoint_confidence", 0.3):
                                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                if writer is not None:
                    writer.write(frame)
                frame_id += 1
                pbar.update(1)

        cap.release()
        if writer is not None:
            writer.release()
        fname = output_name or f"{video_path.stem}{json_suffix}"
        out = output_dir / fname
        with open(out, "w") as f: 
            json.dump(sparta_json, f, indent=2)
        return out


# ---------- YOLO Pose Estimator ----------

class YoloPoseEstimator(PoseEstimatorBase):
    def __init__(self, cfg: Config, model_paths: ModelPaths):
        self.cfg = cfg
        self.model = YOLO(str(model_paths.weights))
        self.conf = float(cfg.pose_cfg.get("confidence_threshold", 0.25))
        self.min_kpt_conf = float(cfg.pose_cfg.get("min_keypoint_confidence", 0.3))
        self.expected_kp = None  # determined on first valid frame

    def process(
        self,
        video_path: Path,
        detections: List[Dict[str, Any]],
        output_dir: Path,
        json_suffix: str,
        output_name: Optional[str] = None,
        output_video_path: Optional[Path] = None,
    ) -> Path:
        cap = cv2.VideoCapture(str(video_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        sparta_json = defaultdict(dict)
        frame_map = defaultdict(list)
        for d in detections:
            frame_map[d["frame_id"]].append(d)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        if output_video_path and self.cfg.pose_save_video:
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*self.cfg.pose_cfg.get("fourcc", "XVID")), fps, (w, h))

        frame_id = 0
        with tqdm(total=total, desc="YOLO Pose") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                persons = frame_map.get(frame_id, [])
                for p in persons:
                    x1, y1, x2, y2 = map(int, p["bbox"])
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    res = self.model(roi, conf=self.conf, verbose=False)
                    if (
                        len(res)
                        and res[0].keypoints is not None
                        and res[0].keypoints.xy is not None
                        and len(res[0].keypoints.xy) > 0
                    ):
                        kps = res[0].keypoints.xy[0].cpu().numpy()
                        confs = res[0].keypoints.conf[0].cpu().numpy()
                        if self.expected_kp is None:
                            self.expected_kp = kps.shape[0]
                        if kps.shape[0] != self.expected_kp:
                            print(f"[WARN] Skipping frame {frame_id} person {p['person_id']}: "
                                  f"kpt count {kps.shape[0]} != expected {self.expected_kp}")
                            continue
                        kps[:, 0] += x1
                        kps[:, 1] += y1
                        triplets = [[float(x), float(y), float(c)] for (x, y), c in zip(kps, confs)]
                        flat = [c for trip in triplets for c in trip]
                        sparta_json[str(p["person_id"])][str(frame_id)] = {
                            "keypoints": flat,
                            "scores": float(confs.mean()) if len(confs) else 0.0
                        }
                        if writer is not None:
                            for (x, y), c in zip(kps, confs):
                                if c >= self.min_kpt_conf:
                                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                if writer is not None:
                    writer.write(frame)
                frame_id += 1
                pbar.update(1)

        cap.release()
        if writer is not None:
            writer.release()
        fname = output_name or f"{video_path.stem}{json_suffix}"
        out = output_dir / fname
        with open(out, "w") as f:
            json.dump(sparta_json, f, indent=2)
        return out


# ---------- Pipeline ----------

class PosePipeline:
    """Thin wrapper so Streamlit UI can call a single entry-point."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.paths = self.config.resolved_paths()
        # Ensure output dir exists
        self.paths["pose_output_dir"].mkdir(parents=True, exist_ok=True)
        self.paths["pose_video_dir"].mkdir(parents=True, exist_ok=True)

    def run(self) -> Optional[Path]:
        p = self.paths

        # Step 1: detection + tracking
        detector = PersonDetector(self.config)
        dets = detector.run(p["input_video"])
        if not dets:
            return None

        # Step 2: pose estimation
        pose_name = self.config.pose_cfg.get("name", "").lower()
        pose_paths = self.config.pose_paths()
        if "yolo" in pose_name:
            estimator = YoloPoseEstimator(self.config, pose_paths)
        else:
            estimator = MMPoseTopDownEstimator(self.config, pose_paths)
        out_name = self.config.human_centric_filename(Path(p["input_video"]).stem)
        pose_video_path = p["pose_video_dir"] / f"{Path(out_name).stem}_vis.avi"
        final_json = estimator.process(
            p["input_video"],
            dets,
            p["pose_output_dir"],
            self.config.pose_json_suffix,
            output_name=out_name,
            output_video_path=pose_video_path if self.config.pose_save_video else None,
        )
        return final_json


# ---------- Main ----------

if __name__ == "__main__":
    config = Config("config.yaml")
    p = config.resolved_paths()

    print("\n[STEP 1] Running person detection...")
    detector = PersonDetector(config)
    dets = detector.run(p["input_video"])

    if dets:
        print("\n[STEP 2] Running pose estimation...")
        paths_pose = config.pose_paths()
        estimator = MMPoseTopDownEstimator(config, paths_pose)
        
        # Disable video saving manually if needed
        config.pose_save_video = False 

        hc_name = config.human_centric_filename(Path(p["input_video"]).stem)
        final_json = estimator.process(p["input_video"], dets, p["pose_output_dir"], config.pose_json_suffix, output_name=hc_name)
        print(f"\n[SUCCESS] Saved to: {final_json}")
    else:
        print("\n[SKIP] No persons found in detection phase.")
