import cv2
import pandas as pd
from ultralytics import YOLO
import os
from pathlib import Path

# ---------------- CONFIG ---------------- #
SCRIPT_DIR = Path(__file__).parent
VIDEO_PATH = str(SCRIPT_DIR / "Surveillancevideo.mp4")
MODEL_PATH = str(SCRIPT_DIR / "yolo26x.pt")

CSV_OUTPUT = "fighting_detections.csv"
VIDEO_OUTPUT = "fighting_yolo_visualized.avi"

CONF_THRES = 0.3
IOU_THRES = 0.50
# ---------------------------------------- #

print("Loading YOLOv26x model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

records = []
frame_id = 0

print("Starting video processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        frame,
        conf=CONF_THRES,
        iou=IOU_THRES,
        classes=[0],  # person
        verbose=False
    )

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for person_id, ((x1, y1, x2, y2), conf) in enumerate(zip(boxes, confs)):
            # Save CSV data
            records.append({
                "frame_id": frame_id,
                "person_id": person_id,
                "bbox_x1": float(x1),
                "bbox_y1": float(y1),
                "bbox_x2": float(x2),
                "bbox_y2": float(y2),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(conf)
            })

            # Draw box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"{conf:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    video_writer.write(frame)
    frame_id += 1

    if frame_id % 50 == 0:
        print(f"Processed {frame_id} frames")

cap.release()
video_writer.release()

# Save CSV
df = pd.DataFrame(records)
df.to_csv(CSV_OUTPUT, index=False)

print("\n✅ DONE")
print(f"CSV saved at: {CSV_OUTPUT}")
print(f"Video saved at: {VIDEO_OUTPUT}")
