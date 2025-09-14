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
