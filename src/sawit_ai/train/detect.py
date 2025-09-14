from pathlib import Path
from typing import Optional
import shutil

from ultralytics import YOLO

from ..utils.env import Config
from ..utils.logging import info, success
from ..utils.paths import ensure_dir, path_from_root


def _find_data_yaml(dataset_dir: Path) -> Optional[Path]:
    # Typical Roboflow YOLOv8 detection dataset includes a data.yaml
    cand = dataset_dir / "data.yaml"
    if cand.exists():
        return cand
    # Search deep just in case
    for p in dataset_dir.rglob("data.yaml"):
        return p
    return None


def train_detection(cfg: Config, dataset_dir: Path) -> Path:
    data_yaml = _find_data_yaml(dataset_dir)
    if not data_yaml:
        raise FileNotFoundError(f"data.yaml not found under {dataset_dir}")

    device = cfg.device

    info(
        f"Training detection: model={cfg.yolo_model}, data={data_yaml}, imgsz={cfg.img_size}, epochs={cfg.epochs}, batch={cfg.batch}, device={device}"
    )
    model = YOLO(cfg.yolo_model)
    results = model.train(
        data=str(data_yaml),
        imgsz=cfg.img_size,
        epochs=cfg.epochs,
        batch=cfg.batch,
        patience=cfg.patience,
        device=device,
        workers=cfg.num_workers,
        seed=cfg.seed,
        project=cfg.runs_dir,
        name=cfg.experiment_name,
    )

    # Copy best.pt to artifacts
    weights_dir = Path(results.save_dir) / "weights"
    best = weights_dir / "best.pt"
    out_dir = ensure_dir(path_from_root(cfg.artifacts_dir, "latest"))
    if best.exists():
        shutil.copy2(best, out_dir / "best.pt")
        success(f"Best weights copied to {out_dir / 'best.pt'}")

    return best

