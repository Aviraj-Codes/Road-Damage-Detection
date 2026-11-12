"""Run detection on an image or video and export annotated media plus CSV."""

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_file():
        parser.error(f"Source does not exist: {source}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    is_video = source.suffix.lower() in video_exts
    annotated_path = output_dir / f"{source.stem}_annotated{source.suffix if is_video else '.jpg'}"
    csv_path = output_dir / f"{source.stem}_detections.csv"

    cap = cv2.VideoCapture(str(source)) if is_video else None
    fps = cap.get(cv2.CAP_PROP_FPS) if cap else 0.0
    writer = None
    frame_number = 0
    rows = []

    try:
        results = model.predict(source=str(source), conf=args.conf, stream=True, verbose=False)
        for result in results:
            frame_number += 1
            frame = result.orig_img
            height, width = frame.shape[:2]
            annotated = result.plot()
            if is_video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(annotated_path), fourcc, fps or 25.0, (width, height))
                writer.write(annotated)
            else:
                cv2.imwrite(str(annotated_path), annotated)

            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                class_id = int(box.cls[0].cpu())
                rows.append({
                    "frame": frame_number,
                    "time_seconds": round((frame_number - 1) / fps, 3) if is_video and fps else 0,
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": round(float(box.conf[0].cpu()), 4),
                    "bbox_area_fraction": round(max(0, x2 - x1) * max(0, y2 - y1) / (width * height), 6),
                })
    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()

    columns = ["frame", "time_seconds", "class_id", "class_name", "confidence", "bbox_area_fraction"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=columns)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(f"Annotated media: {annotated_path}")
    print(f"Detection report: {csv_path}")


if __name__ == "__main__":
    main()
