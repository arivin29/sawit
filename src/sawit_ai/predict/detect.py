from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO

from ..utils.env import Config
from ..utils.logging import info, success, warn
from ..utils.yolo import latest_weight_file, resolve_default_weights


def predict_detection(
    cfg: Config,
    source: Union[str, Path],
    weights: Optional[Union[str, Path]] = None,
    save: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
):
    resolved = resolve_default_weights(cfg.runs_dir, cfg.artifacts_dir, task="detect", explicit=str(weights or cfg.weights) if (weights or cfg.weights) else None)
    weights_path = Path(resolved or "").resolve()
    if not weights_path.exists():
        warn("Weights not found. Using base model for inference.")
        model = YOLO(cfg.yolo_model)
    else:
        info(f"Loading weights: {weights_path}")
        model = YOLO(str(weights_path))

    project = cfg.runs_dir
    name = f"{cfg.experiment_name}-predict"
    if out_dir:
        out_dir = Path(out_dir)
        project = str(out_dir.parent if out_dir.parent != Path('.') else Path.cwd())
        name = out_dir.name

    results = model.predict(
        source=str(source),
        imgsz=cfg.img_size,
        conf=cfg.conf_thres,
        iou=cfg.iou_thres,
        max_det=cfg.max_det,
        classes=None if not cfg.classes else [int(c) for c in cfg.classes.split(",")],
        half=cfg.half,
        device=cfg.device,
        save=save,
        project=project,
        name=name,
    )
    success("Detection inference completed.")
    return results
