from pathlib import Path
from typing import Optional
import typer

from .utils.env import load_config
from .utils.paths import ensure_dir, path_from_root
from .utils.logging import info, success, warn
from .data.roboflow import download_dataset
from .train.detect import train_detection
from .train.classify import train_classification
from .predict.detect import predict_detection
from .predict.classify import predict_classification


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def download(
    force: bool = typer.Option(False, help="Ignore existing dataset and download fresh"),
    out: Optional[str] = typer.Option(None, help="Optional subfolder under data/ to download into"),
    legacy_fallback: bool = typer.Option(True, help="If targeted download yields nothing, retry without location (legacy)"),
):
    """Download dataset from Roboflow using .env settings."""
    cfg = load_config()
    ensure_dir(path_from_root(cfg.data_dir))
    dest = download_dataset(cfg, overwrite=force, out=out, fallback_no_location=legacy_fallback)
    success(f"Dataset ready at: {dest}")


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
    out: Optional[str] = typer.Option(None, help="Optional output dir for predictions"),
):
    """Run inference on images/videos using latest or provided weights."""
    cfg = load_config()
    if cfg.task.lower() == "detect":
        predict_detection(cfg, source=source, weights=weights, save=not nosave, out_dir=out)
    elif cfg.task.lower() == "classify":
        predict_classification(cfg, source=source, weights=weights, save=not nosave, out_dir=out)
    else:
        raise typer.BadParameter("TASK must be one of: detect, classify")


@app.command()
def export(
    weights: Optional[str] = typer.Option(None, help="Path to .pt weights to export (defaults to artifacts/latest/best.pt)"),
    fmt: str = typer.Option("onnx", help="Export format: onnx, torchscript, openvino, engine (TensorRT), coreml, tf, tflite"),
    opset: int = typer.Option(12, help="ONNX opset when format=onnx"),
    half: bool = typer.Option(False, help="FP16 where supported"),
):
    """Export trained weights to deployment formats (e.g., ONNX for TensorRT)."""
    cfg = load_config()
    from ultralytics import YOLO
    from .utils.yolo import resolve_default_weights

    resolved = resolve_default_weights(cfg.runs_dir, cfg.artifacts_dir, task=cfg.task.lower(), explicit=weights)
    if not resolved or not resolved.exists():
        raise typer.BadParameter("Weights not found. Provide --weights or train a model first.")
    info(f"Exporting weights: {resolved}")

    model = YOLO(str(resolved))
    kwargs = {}
    if fmt == "onnx":
        kwargs.update({"opset": opset})
    if half:
        kwargs.update({"half": True})
    try:
        out = model.export(format=fmt, **kwargs)
        success(f"Exported: {out}")
    except Exception as e:
        warn(f"Export failed: {e}")
        raise


def main():
    app()
