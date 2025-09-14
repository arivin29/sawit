from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from ..data.roboflow import ensure_dataset_downloaded
from ..utils.env import load_env, read_train_config


def _resolve_dataset_path(task: str, dataset_root: Path) -> tuple[str, Optional[Path]]:
    """Return data argument for YOLO and optional data.yaml path.

    - For classify: return dataset_root (contains train/valid/test folders)
    - For detect: prefer dataset_root / 'data.yaml' if present, else dataset_root
    """
    if task == "classify":
        return str(dataset_root), None
    else:
        data_yaml = dataset_root / "data.yaml"
        return (str(data_yaml) if data_yaml.exists() else str(dataset_root)), (data_yaml if data_yaml.exists() else None)


def train() -> None:
    load_env()
    tc = read_train_config()

    dataset_root = ensure_dataset_downloaded(overwrite=False)
    data_arg, data_yaml = _resolve_dataset_path(tc.task, dataset_root)

    # Select model
    if tc.task == "classify":
        model_name = f"yolov8{tc.model_size}-cls.pt"
    else:
        model_name = f"yolov8{tc.model_size}.pt"

    model = YOLO(model_name)

    # Build training args
    common_args = dict(
        data=data_arg,
        epochs=tc.epochs,
        imgsz=tc.img_size,
        batch=tc.batch,
        patience=tc.patience,
        lr0=tc.lr0,
        device=tc.device,
        project=tc.runs_dir,
        name=tc.experiment_name,
        verbose=True,
    )

    # Toggle trackers
    if tc.wandb_enabled:
        os.environ.setdefault("WANDB_PROJECT", tc.wandb_project or "mode-sawit")
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # Train
    results = model.train(**common_args)

    # Validate (explicit)
    _ = model.val(data=data_arg, imgsz=tc.img_size, device=tc.device, verbose=True)

    # Print final paths
    print("\n==> Training complete")
    print(f"Runs dir: {tc.runs_dir}")
    # Try to resolve last run best weights
    try:
        last_run_dir = Path(results.save_dir)
        best_pt = last_run_dir / "weights" / "best.pt"
        if best_pt.exists():
            print(f"Best weights: {best_pt}")
    except Exception:
        pass


if __name__ == "__main__":
    train()

