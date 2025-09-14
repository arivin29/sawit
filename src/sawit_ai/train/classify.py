from pathlib import Path
import shutil

from ultralytics import YOLO

from ..utils.env import Config
from ..utils.logging import info, success
from ..utils.paths import ensure_dir, path_from_root


def _validate_cls_structure(dataset_dir: Path):
    # YOLOv8-cls expects dataset_dir/{train,val(,test)}/{class}/image.jpg
    for split in ("train", "valid", "val"):
        if (dataset_dir / split).exists():
            return  # ok
    raise FileNotFoundError(
        f"Classification dataset folder with train/val not found under {dataset_dir}"
    )


def train_classification(cfg: Config, dataset_dir: Path) -> Path:
    _validate_cls_structure(dataset_dir)
    device = cfg.device

    info(
        f"Training classification: model={cfg.yolo_model}, data={dataset_dir}, imgsz={cfg.img_size}, epochs={cfg.epochs}, batch={cfg.batch}, device={device}"
    )
    model = YOLO(cfg.yolo_model)
    results = model.train(
        data=str(dataset_dir),
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

    weights_dir = Path(results.save_dir) / "weights"
    best = weights_dir / "best.pt"
    out_dir = ensure_dir(path_from_root(cfg.artifacts_dir, "latest"))
    if best.exists():
        shutil.copy2(best, out_dir / "best.pt")
        success(f"Best weights copied to {out_dir / 'best.pt'}")
    return best

