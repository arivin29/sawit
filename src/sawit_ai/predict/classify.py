from pathlib import Path
from typing import Optional, Union, List
import csv

from ultralytics import YOLO

from ..utils.env import Config
from ..utils.logging import info, success, warn
from ..utils.yolo import latest_weight_file, resolve_default_weights


def _write_predictions_csv(results, out_dir: Path) -> Optional[Path]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "predictions.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "top1_label", "top1_conf", "top5_labels", "top5_confs"])
            for r in results:
                p = Path(getattr(r, "path", ""))
                names = getattr(r, "names", {})
                probs = getattr(r, "probs", None)
                if probs is None:
                    continue
                # Resolve top-1
                try:
                    top1_idx = int(getattr(probs, "top1"))
                    top1_conf = float(getattr(probs, "top1conf"))
                except Exception:
                    # Fallback for tensor-like
                    data = getattr(probs, "data", None)
                    if data is None:
                        continue
                    top1_idx = int(data.argmax().item())
                    top1_conf = float(data.max().item())
                # Resolve top-5
                try:
                    top5_idx: List[int] = [int(i) for i in list(getattr(probs, "top5"))]
                    top5_conf: List[float] = [float(c) for c in list(getattr(probs, "top5conf"))]
                except Exception:
                    data = getattr(probs, "data", None)
                    if data is not None:
                        vals, idxs = data.topk(5)
                        top5_idx = [int(i) for i in idxs.tolist()]
                        top5_conf = [float(v) for v in vals.tolist()]
                    else:
                        top5_idx, top5_conf = [top1_idx], [top1_conf]

                def idx_to_name(i: int) -> str:
                    try:
                        return names.get(i, str(i)) if isinstance(names, dict) else str(i)
                    except Exception:
                        return str(i)

                top1_label = idx_to_name(top1_idx)
                top5_labels = [idx_to_name(i) for i in top5_idx]
                w.writerow([
                    str(p), top1_label, f"{top1_conf:.4f}",
                    ";".join(top5_labels), ";".join(f"{c:.4f}" for c in top5_conf),
                ])
        return csv_path
    except Exception:
        return None


def predict_classification(
    cfg: Config,
    source: Union[str, Path],
    weights: Optional[Union[str, Path]] = None,
    save: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
):
    resolved = resolve_default_weights(cfg.runs_dir, cfg.artifacts_dir, task="classify", explicit=str(weights or cfg.weights) if (weights or cfg.weights) else None)
    weights_path = Path(resolved or "").resolve()
    if not weights_path.exists():
        warn("Weights not found. Using base classification model for inference.")
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
        device=cfg.device,
        save=save,
        project=project,
        name=name,
    )
    # Try to write a CSV summary for convenience
    try:
        # Ultralytics provides `save_dir` attribute on each result
        save_dir = Path(getattr(results[0], "save_dir", Path(project) / name))
        csv_path = _write_predictions_csv(results, Path(save_dir))
        if csv_path:
            success(f"Classification inference completed. CSV: {csv_path}")
        else:
            success("Classification inference completed.")
    except Exception:
        success("Classification inference completed.")
    return results
