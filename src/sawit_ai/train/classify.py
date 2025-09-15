from pathlib import Path
import shutil
from typing import Dict, List, Set

from ultralytics import YOLO

from ..utils.env import Config
from ..utils.logging import info, success
from ..utils.paths import ensure_dir, path_from_root


def _classes_in_split(split_dir: Path) -> Set[str]:
    classes: Set[str] = set()
    if not split_dir.exists():
        return classes
    for p in split_dir.iterdir():
        if p.is_dir():
            # Count as a class folder if it contains at least 1 image
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                if next(p.glob(ext), None) is not None:
                    classes.add(p.name)
                    break
    return classes


def _validate_cls_structure(dataset_dir: Path):
    # YOLOv8-cls expects dataset_dir/{train,val(,test)}/{class}/image.jpg
    splits_present = [s for s in ("train", "valid", "val", "test") if (dataset_dir / s).exists()]
    if not any(s in splits_present for s in ("train",)):
        raise FileNotFoundError(
            f"Classification dataset folder with train/val not found under {dataset_dir}"
        )

    # Ensure class folders are consistent across splits to avoid label mismatch on GPU
    split_classes: Dict[str, Set[str]] = {s: _classes_in_split(dataset_dir / s) for s in splits_present}
    # Compare all to train if present, else to the first split
    ref_split = "train" if "train" in split_classes else splits_present[0]
    ref_classes = split_classes.get(ref_split, set())
    inconsistent: List[str] = []
    for s, cls in split_classes.items():
        if cls != ref_classes:
            missing = sorted(ref_classes - cls)
            extra = sorted(cls - ref_classes)
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if extra:
                parts.append(f"extra={extra}")
            inconsistent.append(f"{s} ({', '.join(parts)})")
    if inconsistent:
        detail = ", ".join(inconsistent)
        raise ValueError(
            "Inconsistent class folders across splits. Ensure train/valid/test have the same class names. "
            f"Reference split='{ref_split}' classes={sorted(ref_classes)}; issues: {detail}"
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
