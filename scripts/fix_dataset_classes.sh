#!/usr/bin/env bash
set -euo pipefail

# Fold 'unripe' and 'underripe' into 'ripe' across splits to align classes.
# Usage: scripts/fix_dataset_classes.sh [DATASET_ROOT]

DATASET_ROOT=${1:-palm-fruit-ripeness-classification-2}

if [ ! -d "$DATASET_ROOT" ]; then
  echo "Dataset root not found: $DATASET_ROOT" >&2
  exit 1
fi

echo "Fixing dataset at: $(realpath "$DATASET_ROOT")"

move_all_images() {
  local src_dir="$1"
  local dst_dir="$2"
  [ -d "$src_dir" ] || return 0
  mkdir -p "$dst_dir"
  shopt -s nullglob nocaseglob
  for f in "$src_dir"/*.{jpg,jpeg,png,bmp,tif,tiff,JPG,JPEG,PNG,BMP,TIF,TIFF} "$src_dir"/*; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    dest="$dst_dir/$base"
    if [ -e "$dest" ]; then
      name="${base%.*}"
      ext="${base##*.}"
      i=1
      while [ -e "$dst_dir/${name}_$i.$ext" ]; do i=$((i+1)); done
      dest="$dst_dir/${name}_$i.$ext"
    fi
    mv "$f" "$dest"
  done
  shopt -u nullglob nocaseglob
}

for split in train valid val test; do
  split_dir="$DATASET_ROOT/$split"
  [ -d "$split_dir" ] || continue
  echo "Processing split: $split"
  # Ensure ripe exists
  mkdir -p "$split_dir/ripe"
  # Move images from unripe/underripe into ripe
  for cname in unripe underripe; do
    if [ -d "$split_dir/$cname" ]; then
      echo "  Folding class '$cname' into 'ripe'"
      move_all_images "$split_dir/$cname" "$split_dir/ripe"
      # Remove empty dir if possible
      rmdir "$split_dir/$cname" 2>/dev/null || true
    fi
  done
done

echo "\nClass counts per split after fix:"
for split in train valid val test; do
  d="$DATASET_ROOT/$split"
  [ -d "$d" ] || continue
  echo "\nSplit: $split"
  for cls in "$d"/*; do
    [ -d "$cls" ] || continue
    count=$(find "$cls" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l)
    echo "  class $(basename "$cls"): $count images"
  done
done

echo "\nDone. Ensure all splits list the same class folders now."

