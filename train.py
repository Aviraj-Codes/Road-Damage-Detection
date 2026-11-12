"""Train a YOLOv8 road damage detector."""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/data.yaml", help="Dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model/checkpoint")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="For example: 0, cpu, or omit for auto")
    parser.add_argument("--name", default="road_damage")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name=args.name,
        patience=15,
        plots=True,
    )


if __name__ == "__main__":
    main()
