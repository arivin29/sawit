from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv


def load_env(env_path: Optional[str | os.PathLike] = None) -> None:
    """Load environment variables from a .env file if present."""
    if env_path:
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        # Walk up to repo root to find .env
        cwd = Path.cwd()
        candidates = [cwd / ".env", cwd.parent / ".env", cwd.parent.parent / ".env"]
        for p in candidates:
            if p.exists():
                load_dotenv(dotenv_path=p, override=False)
                break


TaskType = Literal["classify", "detect"]


@dataclass
class RoboflowConfig:
    api_key: str
    workspace: str
    project: str
    version: str | int


@dataclass
class TrainConfig:
    task: TaskType = "classify"
    model_size: Literal["n", "s", "m", "l", "x"] = "n"
    epochs: int = 50
    batch: int = 16
    img_size: int = 640
    patience: int = 20
    lr0: float = 0.01
    device: str = "0"  # "cpu", "0", or "0,1"
    experiment_name: str = "baseline_v1"
    runs_dir: str = "runs"
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None


def getenv_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_roboflow_config() -> RoboflowConfig:
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "").strip()
    project = os.getenv("ROBOFLOW_PROJECT", "").strip()
    version = os.getenv("ROBOFLOW_VERSION", "latest").strip()
    if not api_key or not workspace or not project:
        raise RuntimeError(
            "Missing ROBOFLOW_API_KEY/WORKSPACE/PROJECT in environment (.env)."
        )
    return RoboflowConfig(api_key=api_key, workspace=workspace, project=project, version=version)


def read_train_config() -> TrainConfig:
    task = os.getenv("TASK", "classify").strip().lower()
    model_size = os.getenv("MODEL_SIZE", "n").strip()
    epochs = int(os.getenv("EPOCHS", "50"))
    batch = int(os.getenv("BATCH", "16"))
    img_size = int(os.getenv("IMG_SIZE", "640"))
    patience = int(os.getenv("PATIENCE", "20"))
    lr0 = float(os.getenv("LR0", "0.01"))
    device = os.getenv("DEVICE", "0").strip()
    experiment_name = os.getenv("EXPERIMENT_NAME", "baseline_v1").strip()
    runs_dir = os.getenv("RUNS_DIR", "runs").strip()
    wandb_enabled = getenv_bool("WANDB_ENABLED", False)
    wandb_project = os.getenv("WANDB_PROJECT", None)

    if task not in {"classify", "detect"}:
        raise ValueError("TASK must be 'classify' or 'detect'")
    if model_size not in {"n", "s", "m", "l", "x"}:
        raise ValueError("MODEL_SIZE must be one of n,s,m,l,x")

    return TrainConfig(
        task=task, model_size=model_size, epochs=epochs, batch=batch, img_size=img_size,
        patience=patience, lr0=lr0, device=device, experiment_name=experiment_name,
        runs_dir=runs_dir, wandb_enabled=wandb_enabled, wandb_project=wandb_project,
    )

