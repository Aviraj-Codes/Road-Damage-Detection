# RoadSight
### Road Damage Detection and Classification

**A YOLO-based computer vision system for locating and classifying road defects in images and video**

RoadSight is an academic deep-learning project that detects common road damage such as cracks and potholes. It includes a repeatable training and evaluation workflow and an interactive Streamlit application for image and video inference.

---

## Features

- **Object detection:** locate multiple defects in one frame with bounding boxes.
- **Damage classification:** predict the defect category for each detected region.
- **Image and video inference:** analyze an uploaded road photo or video in the Streamlit app.
- **Explainable output:** display annotated media and a table with class, confidence, frame/time, and box area fraction.
- **Evaluation:** export precision, recall, mAP@0.5, and mAP@0.5:0.95 to CSV.
- **Generalization study:** compare results across held-out locations or lighting conditions.

## Research question

**How well does a road damage detector generalize across locations and lighting conditions?** Train on selected regions, hold out a separate region for evaluation, and compare per-class performance. A model-size comparison (YOLOv8n versus YOLOv8s) can add a speed/accuracy trade-off experiment.

## Classes

This starter follows the four common RDD2022 labels. Ensure the downloaded/exported dataset uses this exact class order.

| ID | Label | Damage type |
|---:|---|---|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

## Dataset

**Dataset:** RDD2022, the multi-national road damage dataset released for the Crowdsensing-based Road Damage Detection Challenge.

**Source:** [Official CRDDC2022 data page](https://crddc2022.sekilab.global/data/). The original challenge training set is annotated, while the official challenge test set is distributed without annotations. To calculate metrics, create a labeled held-out test split from the annotated training data (or use another labeled export); do not try to evaluate against the unlabeled challenge test images. The original annotations may require conversion. A YOLO-format export from Roboflow can be used as a convenient starting point; verify label IDs, split contents, and dataset terms before training or redistribution.

Expected layout:

```text
dataset/
  data.yaml
  train/images/   train/labels/
  val/images/     val/labels/
  test/images/    test/labels/
```

Copy `dataset/data.example.yaml` to `dataset/data.yaml` and update it if your export uses different paths. The template leaves out `path`, so Ultralytics treats the folder containing `data.yaml` (`dataset/`) as the dataset root. Labels must use YOLO detection format: `class_id x_center y_center width height`, with normalized coordinates. Keep image/label basenames paired. For the generalization experiment, split by country, source, or capture route where possible. Do not randomly distribute near-duplicate frames across train and test.

## Methodology

### 1. Training

The project fine-tunes a pretrained YOLOv8 checkpoint for four-class object detection. The starter command uses the small `yolov8n.pt` model at 640-pixel image size. Train a larger `yolov8s.pt` model under the same data and settings for a controlled comparison.

### 2. Inference

The Streamlit interface loads `models/best.pt`, accepts an image or video, and shows detections with confidence scores. It also exports detection rows to CSV. Each row includes the frame index and time for video inputs.

The CSV includes `bbox_area_fraction`: the bounding-box area divided by frame area. This is only a rough image-space signal; perspective, camera distance, and box tightness make it unsuitable as a physical damage-size or severity measurement. A future extension could use instance segmentation and calibrated camera geometry.

### 3. Evaluation

Use precision, recall, mAP@0.5, and mAP@0.5:0.95, with per-class results and error examples. Keep the test set untouched during model selection. For a stronger research comparison, report results separately for held-out locations and lighting subsets, plus inference speed and model size.

## Setup

Python 3.10 or 3.11 is recommended. GPU training requires a compatible CUDA-enabled PyTorch installation; CPU training works but may be slow.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 60 --imgsz 640
```

Training outputs are saved under `runs/detect/road_damage/`. Copy `runs/detect/road_damage/weights/best.pt` to `models/best.pt` for the app and evaluator. Change `--name` if you want to preserve separate experiment checkpoints.

### Evaluate

```bash
python evaluate.py --data dataset/data.yaml --weights models/best.pt --split test
```

The summary is written to `results/performance_comparison.csv`; detailed Ultralytics plots are saved under `runs/detect/`.

### Launch Streamlit

```bash
streamlit run app/app.py
```

The app supports image and video upload. Video analysis can take time for long or high-resolution clips.

### Command-line prediction

```bash
python predict.py --source path/to/road.jpg --weights models/best.pt --output-dir outputs
python predict.py --source path/to/road_video.mp4 --weights models/best.pt --output-dir outputs
```

## Repository structure

```text
RoadSight/
├── app/
│   └── app.py
├── dataset/
│   └── data.example.yaml
├── models/
│   └── README.md              # add best.pt after training
├── notebooks/
│   └── README.md              # experiment notes and analysis
├── results/
│   └── README.md              # generated evaluation artifacts
├── test_samples/
│   └── README.md              # add demo images/clips
├── evaluate.py
├── predict.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

| Area | Tools |
|---|---|
| Deep learning and detection | PyTorch, Ultralytics YOLOv8 |
| Image/video processing | OpenCV, Pillow |
| Interactive application | Streamlit |
| Data and reports | Pandas, CSV |
| Dataset annotation | YOLO format, optionally Roboflow |

## Limitations and responsible use

This is an experimental prototype. Results can vary with road materials, weather, camera angle, resolution, and regions absent from training. Bounding boxes are not pixel-accurate damage masks, and visual box area is not a calibrated measure of severity. Do not use predictions as an official road-condition assessment or to make maintenance decisions without qualified human inspection.

## References

- [CRDDC2022 and RDD2022 data](https://crddc2022.sekilab.global/data/)
- [Ultralytics object detection dataset format](https://docs.ultralytics.com/datasets/detect/)
- [Ultralytics model training](https://docs.ultralytics.com/modes/train/)
