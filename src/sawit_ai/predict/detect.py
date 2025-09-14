from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO

from ..utils.env import Config
from ..utils.logging import info, success, warn
from ..utils.yolo import latest_weight_file


def predict_detection(
    cfg: Config,
    source: Union[str, Path],
    weights: Optional[Union[str, Path]] = None,
    save: bool = True,
):
    weights_path = Path(weights or cfg.weights or (latest_weight_file(cfg.runs_dir, task="detect") or "")).resolve()
    if not weights_path.exists():
        warn("Weights not found. Using base model for inference.")
        model = YOLO(cfg.yolo_model)
    else:
        info(f"Loading weights: {weights_path}")
        model = YOLO(str(weights_path))

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
        project=cfg.runs_dir,
        name=f"{cfg.experiment_name}-predict",
    )
    success("Detection inference completed.")
    return results

