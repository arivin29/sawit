from pathlib import Path
from typing import Optional
import typer

from .utils.env import load_config
from .utils.paths import ensure_dir, path_from_root
from .utils.logging import info, success
from .data.roboflow import download_dataset
from .train.detect import train_detection
from .train.classify import train_classification
from .predict.detect import predict_detection
from .predict.classify import predict_classification


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def download():
    """Download dataset from Roboflow using .env settings."""
    cfg = load_config()
    ensure_dir(path_from_root(cfg.data_dir))
    download_dataset(cfg)


@app.command()
def train(dataset: Optional[str] = typer.Option(None, help="Dataset dir override")):
    """Train YOLOv8 model according to TASK (detect/classify)."""
    cfg = load_config()
    data_dir = Path(dataset) if dataset else None

    if data_dir is None:
        info("No dataset override provided. Ensuring dataset is downloaded...")
        data_dir = download_dataset(cfg)
    else:
        data_dir = Path(data_dir).resolve()

    if cfg.task.lower() == "detect":
        best = train_detection(cfg, data_dir)
    elif cfg.task.lower() == "classify":
        best = train_classification(cfg, data_dir)
    else:
        raise typer.BadParameter("TASK must be one of: detect, classify")

    success(f"Training complete. Best weights: {best}")


@app.command()
def predict(
    source: str = typer.Option(..., help="Path to image/video/folder/glob"),
    weights: Optional[str] = typer.Option(None, help="Path to .pt weights to use"),
    nosave: bool = typer.Option(False, help="Do not save output visuals"),
):
    """Run inference on images/videos using latest or provided weights."""
    cfg = load_config()
    if cfg.task.lower() == "detect":
        predict_detection(cfg, source=source, weights=weights, save=not nosave)
    elif cfg.task.lower() == "classify":
        predict_classification(cfg, source=source, weights=weights, save=not nosave)
    else:
        raise typer.BadParameter("TASK must be one of: detect, classify")


def main():
    app()

