from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO

from ..utils.env import Config
from ..utils.logging import info, success, warn
from ..utils.yolo import latest_weight_file


def predict_classification(
    cfg: Config,
    source: Union[str, Path],
    weights: Optional[Union[str, Path]] = None,
    save: bool = True,
):
    weights_path = Path(weights or cfg.weights or (latest_weight_file(cfg.runs_dir, task="classify") or "")).resolve()
    if not weights_path.exists():
        warn("Weights not found. Using base classification model for inference.")
        model = YOLO(cfg.yolo_model)
    else:
        info(f"Loading weights: {weights_path}")
        model = YOLO(str(weights_path))

    results = model.predict(
        source=str(source),
        imgsz=cfg.img_size,
        device=cfg.device,
        save=save,
        project=cfg.runs_dir,
        name=f"{cfg.experiment_name}-predict",
    )
    success("Classification inference completed.")
    return results

