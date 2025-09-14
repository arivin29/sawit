from pathlib import Path
from typing import Optional
import os


def resolve_device(device_env: str) -> str:
    # Accept values like "0", "0,1", "cpu", "cuda:0"
    d = str(device_env).strip()
    if d.lower() in {"cpu", "cuda", "cuda:0"}:
        return d
    # Convert simple numbers to cuda indexes
    if d.replace(",", "").isdigit():
        return d  # Ultralytics accepts "0" or "0,1"
    return "0"


def latest_weight_file(runs_dir: str, task: str = "detect") -> Optional[Path]:
    # YOLOv8 defaults: runs/detect/exp*/weights/best.pt or runs/classify/exp*/weights/best.pt
    root = Path(runs_dir)
    task_dir = root / task
    if not task_dir.exists():
        return None
    candidates = list(task_dir.glob("exp*/weights/best.pt")) + list(task_dir.glob("**/weights/best.pt"))
    if not candidates:
        return None
    # Pick the most recently modified best.pt
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

