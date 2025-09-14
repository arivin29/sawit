from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from ..utils.env import load_env


def find_latest_best(runs_dir: str = "runs") -> Optional[Path]:
    base = Path(runs_dir)
    if not base.exists():
        return None
    candidates = list(base.rglob("best.pt"))
    if not candidates:
        return None
    # Pick most recent by modified time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main(argv: Optional[list[str]] = None) -> None:
    load_env()
    parser = argparse.ArgumentParser(description="YOLOv8 inference helper")
    parser.add_argument("--source", required=True, help="Image/video path or directory")
    parser.add_argument("--weights", default=None, help="Path to a weights .pt file (defaults to latest best.pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="'cpu', '0', '0,1', etc.")
    args = parser.parse_args(argv)

    weights_path = Path(args.weights) if args.weights else find_latest_best()
    if not weights_path or not weights_path.exists():
        raise SystemExit("Could not find weights. Provide --weights or train a model first.")

    model = YOLO(str(weights_path))
    result = model.predict(source=args.source, imgsz=args.imgsz, device=args.device)
    # Print a small summary
    print(f"Saved predictions to: {result[0].save_dir if result else 'runs/predict'}")


if __name__ == "__main__":
    main()

