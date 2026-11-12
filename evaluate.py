"""Evaluate a trained detector and save its summary metrics as CSV."""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", default="results/performance_comparison.csv")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, plots=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": Path(args.weights).name,
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
