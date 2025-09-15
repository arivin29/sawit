# Palm Fruit Ripeness – YOLOv8 + Roboflow (PyTorch)

python3 -m venv .venv && source .venv/bin/activate

This repo scaffolds a scalable pipeline to detect and/or classify palm fruit ripeness using Ultralytics YOLOv8 and datasets from Roboflow.

- Task modes: detection or classification (select via `.env`)
- Config-by-environment with sane defaults
- Single CLI for dataset download, training, and inference
- GPU-ready when available (Vest.ai or local CUDA)

## Quickstart

1) Python environment (recommended):

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

Usage snippets

- Train on existing dataset folder:
  - `python -m sawit_ai train --dataset palm-fruit-ripeness-classification-2`
  - Best weights copied to `artifacts/latest/best.pt`.

- Predict on a folder of images and get a CSV summary:
  - `python -m sawit_ai predict --source Coba/test1/image --weights artifacts/latest/best.pt`
  - Optional custom output dir: `--out runs/my-predict`
  - See `predictions.csv` under the output dir.

- Export weights for deployment (e.g., to ONNX for TensorRT):
  - `python -m sawit_ai export --weights artifacts/latest/best.pt --fmt onnx --opset 12`
  - Omit `--weights` to auto-pick `artifacts/latest/best.pt`.

- Build a full-train dataset (merge train/val/test, 5% stratified val):
  - `python scripts/prepare_full_train.py --src palm-fruit-ripeness-classification-2 --dst palm-fruit-ripeness-classification-2-fulltrain --val-ratio 0.05`

Notes

- If you see a PyTorch pin_memory thread ConnectionResetError after training completes, it is typically benign and happens during interpreter shutdown on some environments.
```

2) Configure `.env`:

- A `.env` file has been created for you using values you provided.
- If needed, copy `.env.example` → `.env` and adjust.
  - ROBOFLOW_VERSION accepts a number (e.g., 1) or 'latest'.

3) Commands (after editable install):

- Download dataset from Roboflow
```
# Either of these works now:
python -m sawit_ai download
# or
sawit-ai download
```

- Train
```
# Classification (TASK=classify; YOLO_MODEL=yolov8s-cls.pt)
python -m sawit_ai train

# Detection (TASK=detect; YOLO_MODEL=yolov8s.pt)
# Set ROBOFLOW_FORMAT=yolov8 and ensure the dataset is detection-type
python -m sawit_ai train
```

- Predict on images/videos
```
python -m sawit_ai predict --source path/to/images_or_video
```

## Project Layout

```
.
├── src/sawit_ai/
│   ├── cli.py                  # Typer CLI (download/train/predict)
│   ├── __main__.py             # Entrypoint: python -m sawit_ai
│   ├── utils/
│   │   ├── env.py              # .env loader → Config
│   │   ├── paths.py            # Path helpers, directory creation
│   │   ├── yolo.py             # YOLO helpers (latest weights, device)
│   │   └── logging.py          # Rich logging
│   ├── data/roboflow.py        # Roboflow dataset downloader
│   ├── train/
│   │   ├── detect.py           # YOLOv8 detection training wrapper
│   │   └── classify.py         # YOLOv8 classification training wrapper
│   └── predict/
│       ├── detect.py           # Inference for detection
│       └── classify.py         # Inference for classification
├── .env                        # All settings live here (gitignored)
├── .env.example
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes

- For detection tasks, ensure your Roboflow dataset is an object-detection type and set `ROBOFLOW_FORMAT=yolov8`.
- For classification tasks, set `ROBOFLOW_FORMAT=folder` so YOLOv8-cls can read the train/val/test folders.
- After training, the best weights are under `runs/<task>/exp*/weights/best.pt`. The CLI tries to find the newest automatically, or you can set `WEIGHTS` in `.env`.

## Next Steps

- Add experiment tracking (e.g., Weights & Biases) by setting env vars and passing `project=`/`name=` in training.
- Extend to segmentation if your dataset supports it.
- Package this with a `pyproject.toml` or Dockerfile if you want reproducible deployments.
