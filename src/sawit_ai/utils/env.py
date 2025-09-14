import os
from dataclasses import dataclass
from typing import Optional, Union

from dotenv import load_dotenv

# Load .env on import
load_dotenv(override=True)


def _to_bool(val: str, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env(name: str, default=None, cast=None, required: bool = False):
    v = os.getenv(name, default)
    if v is None and required:
        raise ValueError(f"Missing required env var: {name}")
    if cast and v is not None:
        if cast is bool:
            return _to_bool(v)
        try:
            return cast(v)
        except Exception as e:
            raise ValueError(f"Invalid value for {name}: {v} ({e})")
    return v


@dataclass
class Config:
    # Task & model
    task: str
    yolo_model: str

    # Training
    img_size: int
    epochs: int
    batch: int
    patience: int
    num_workers: int
    seed: int

    # Runtime & paths
    device: str
    runs_dir: str
    data_dir: str
    artifacts_dir: str
    experiment_name: str

    # Roboflow
    rf_api_key: str
    rf_workspace: str
    rf_project: str
    # Accept int like 1 or string like "latest"
    rf_version: Union[int, str]
    rf_format: str

    # Inference
    conf_thres: float
    iou_thres: float
    max_det: int
    classes: Optional[str]
    half: bool

    # Optional override
    weights: Optional[str]


def load_config() -> Config:
    return Config(
        task=_get_env("TASK", "classify", str),
        yolo_model=_get_env("YOLO_MODEL", "yolov8s-cls.pt", str),
        img_size=_get_env("IMG_SIZE", 640, int),
        epochs=_get_env("EPOCHS", 50, int),
        batch=_get_env("BATCH", 16, int),
        patience=_get_env("PATIENCE", 20, int),
        num_workers=_get_env("NUM_WORKERS", 8, int),
        seed=_get_env("SEED", 42, int),
        device=str(_get_env("DEVICE", "0")),
        runs_dir=_get_env("RUNS_DIR", "runs", str),
        data_dir=_get_env("DATA_DIR", "data", str),
        artifacts_dir=_get_env("ARTIFACTS_DIR", "artifacts", str),
        experiment_name=_get_env("EXPERIMENT_NAME", "sawit-yolo", str),
        rf_api_key=_get_env("ROBOFLOW_API_KEY", required=True),
        rf_workspace=_get_env("ROBOFLOW_WORKSPACE", required=True),
        rf_project=_get_env("ROBOFLOW_PROJECT", required=True),
        # keep as raw string to allow values like 'latest'; coerce to int later if needed
        rf_version=_get_env("ROBOFLOW_VERSION", "1", str),
        rf_format=_get_env("ROBOFLOW_FORMAT", "folder", str),
        conf_thres=_get_env("CONF_THRES", 0.25, float),
        iou_thres=_get_env("IOU_THRES", 0.45, float),
        max_det=_get_env("MAX_DET", 300, int),
        classes=_get_env("CLASSES", None, str),
        half=_get_env("HALF", False, bool),
        weights=_get_env("WEIGHTS", None, str),
    )
