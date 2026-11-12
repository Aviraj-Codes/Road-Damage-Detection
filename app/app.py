"""Streamlit interface for the RoadSight road damage detector."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="RoadSight", page_icon="🛣️", layout="wide")
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best.pt"
RESULTS_DIR = BASE_DIR / "results"
REPORT_COLUMNS = [
    "frame", "time_seconds", "class_id", "class_name",
    "confidence", "bbox_area_fraction",
]


@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def get_model() -> YOLO:
    if not MODEL_PATH.exists():
        st.error(
            f"Trained weights not found at `{MODEL_PATH}`. Train the model first; "
            "then copy `runs/detect/road_damage/weights/best.pt` to `models/best.pt`."
        )
        st.stop()
    return load_model(str(MODEL_PATH))


def collect_rows(result, frame_number: int, fps: float = 0.0) -> list[dict]:
    height, width = result.orig_img.shape[:2]
    rows = []
    if result.boxes is None:
        return rows
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        class_id = int(box.cls[0].cpu())
        rows.append({
            "frame": frame_number,
            "time_seconds": round((frame_number - 1) / fps, 3) if fps else 0,
            "class_id": class_id,
            "class_name": result.names[class_id],
            "confidence": round(float(box.conf[0].cpu()), 4),
            "bbox_area_fraction": round(
                max(0, x2 - x1) * max(0, y2 - y1) / (width * height), 6
            ),
        })
    return rows


def show_report(rows: list[dict], name: str) -> None:
    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    st.subheader("Detection report")
    if report.empty:
        st.info("No damage detections at the selected confidence threshold.")
    else:
        st.dataframe(report, use_container_width=True, hide_index=True)
        st.download_button(
            "Download detections as CSV",
            report.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(name).stem}_detections.csv",
            mime="text/csv",
        )
    st.caption(
        "Bounding-box area is a rough image-space indicator, not a measurement of "
        "physical damage or repair severity."
    )


st.markdown(
    "<div style='text-align:center;padding:16px 0 6px'>"
    "<h1 style='font-size:3rem;margin-bottom:0'>RoadSight</h1>"
    "<h3 style='font-weight:400;margin-top:4px'>Road Damage Detection and Classification</h3>"
    "<p>YOLO-based detection for road images and video</p></div>",
    unsafe_allow_html=True,
)
st.divider()

with st.sidebar:
    st.header("Configuration")
    confidence = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
    st.write("Model: YOLOv8")
    st.write("Classes: D00, D10, D20, D40")
    st.divider()
    st.info(
        "Academic prototype for research and demonstration. Predictions need "
        "human review and are not an official road-condition assessment."
    )

model = get_model()
image_tab, video_tab, performance_tab = st.tabs(
    ["Analyze an image", "Analyze a video", "Model performance"]
)

with image_tab:
    uploaded_image = st.file_uploader(
        "Upload a road image", type=["jpg", "jpeg", "png", "bmp"], key="image"
    )
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        result = model.predict(image, conf=confidence, verbose=False)[0]
        annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
        left, right = st.columns(2)
        left.image(image, caption="Uploaded image", use_container_width=True)
        right.image(annotated_rgb, caption="Detected road damage", use_container_width=True)
        rows = collect_rows(result, frame_number=1)
        if rows:
            st.metric("Damage instances detected", len(rows))
        show_report(rows, uploaded_image.name)

with video_tab:
    uploaded_video = st.file_uploader(
        "Upload a road video", type=["mp4", "avi", "mov", "mkv"], key="video"
    )
    if uploaded_video:
        st.video(uploaded_video)
        if st.button("Analyze video", type="primary"):
            rows: list[dict] = []
            progress = st.progress(0, text="Preparing video…")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                input_path = temp_dir / f"input{Path(uploaded_video.name).suffix}"
                output_path = temp_dir / "annotated.mp4"
                input_path.write_bytes(uploaded_video.getvalue())
                capture = cv2.VideoCapture(str(input_path))
                fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                writer = None
                frame_number = 0
                try:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        frame_number += 1
                        result = model.predict(frame, conf=confidence, verbose=False)[0]
                        annotated = result.plot()
                        if writer is None:
                            h, w = annotated.shape[:2]
                            writer = cv2.VideoWriter(
                                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                fps, (w, h),
                            )
                        writer.write(annotated)
                        rows.extend(collect_rows(result, frame_number, fps))
                        if total:
                            progress.progress(
                                min(frame_number / total, 1.0),
                                text=f"Analyzing frame {frame_number} of {total}",
                            )
                finally:
                    capture.release()
                    if writer is not None:
                        writer.release()
                progress.empty()
                if frame_number == 0:
                    st.error("The uploaded file could not be read as a video.")
                elif output_path.exists():
                    st.video(output_path.read_bytes())
                    show_report(rows, uploaded_video.name)

with performance_tab:
    metrics_path = RESULTS_DIR / "performance_comparison.csv"
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
    else:
        st.info("Evaluation results will appear here after running `evaluate.py`.")
    st.caption("Review per-class metrics and the test split protocol in the project README.")

st.divider()
st.caption(
    "RoadSight · YOLO object detection · Prototype output supports inspection; "
    "it does not replace an engineer's assessment."
)
