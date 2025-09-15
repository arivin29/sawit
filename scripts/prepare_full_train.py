#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(d: Path) -> List[Path]:
    return [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def ensure_empty_dir(d: Path) -> None:
    if d.exists():
        # Keep safe: do not delete; user can choose a new dst if exists
        raise FileExistsError(f"Destination already exists: {d}")
    d.mkdir(parents=True, exist_ok=False)


def gather_all_by_class(src: Path) -> Dict[str, List[Path]]:
    by_class: Dict[str, List[Path]] = {}
    splits = [s for s in ("train", "valid", "val", "test") if (src / s).exists()]
    if not splits:
        raise FileNotFoundError(f"No split folders found under {src} (expected train/valid[/val]/test)")
    for split in splits:
        for cls_dir in (src / split).iterdir():
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            by_class.setdefault(cls, []).extend(list_images(cls_dir))
    return by_class


def stratified_split(by_class: Dict[str, List[Path]], val_ratio: float, seed: int) -> tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    train_sel: Dict[str, List[Path]] = {}
    val_sel: Dict[str, List[Path]] = {}
    rng = random.Random(seed)
    for cls, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_val = max(1 if n > 0 else 0, int(round(n * val_ratio)))
        # Cap to n-1 if would consume all
        if n_val >= n and n > 1:
            n_val = n - 1
        val_sel[cls] = items[:n_val]
        train_sel[cls] = items[n_val:]
    return train_sel, val_sel


def copy_to_layout(selection: Dict[str, List[Path]], root: Path, split: str) -> None:
    for cls, files in selection.items():
        out_dir = root / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for src_f in files:
            dst_f = out_dir / src_f.name
            # De-duplicate file names if collisions occur
            if dst_f.exists():
                stem = dst_f.stem
                suf = dst_f.suffix
                i = 1
                while True:
                    cand = out_dir / f"{stem}_{i}{suf}"
                    if not cand.exists():
                        dst_f = cand
                        break
                    i += 1
            shutil.copy2(src_f, dst_f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a full-train classification dataset with small stratified val holdout.")
    ap.add_argument("--src", default="palm-fruit-ripeness-classification-2", help="Source dataset root with train/valid[/test]")
    ap.add_argument("--dst", default="palm-fruit-ripeness-classification-2-fulltrain", help="Destination dataset root")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio per class (0-1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")
    if not (0.0 <= args.val_ratio <= 0.5):
        raise ValueError("--val-ratio must be between 0.0 and 0.5")

    ensure_empty_dir(dst)
    by_class = gather_all_by_class(src)

    # Basic sanity: ensure source has consistent class sets across splits by folder names
    classes = sorted(by_class.keys())
    if not classes:
        raise RuntimeError("No classes found in source dataset.")

    train_sel, val_sel = stratified_split(by_class, args.val_ratio, args.seed)

    copy_to_layout(train_sel, dst, "train")
    copy_to_layout(val_sel, dst, "valid")

    # Summary
    def count_map(sel: Dict[str, List[Path]]) -> Dict[str, int]:
        return {k: len(v) for k, v in sel.items()}

    train_counts = count_map(train_sel)
    val_counts = count_map(val_sel)
    print(f"Built dataset at: {dst}")
    print("Classes:", classes)
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    print(f"Train images: {total_train}, Val images: {total_val}")
    for c in classes:
        print(f"  {c:15s} train={train_counts.get(c, 0):4d}  val={val_counts.get(c, 0):4d}")


if __name__ == "__main__":
    main()

