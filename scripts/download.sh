#!/usr/bin/env bash
set -euo pipefail

python -m src.mode_sawit.data.roboflow --download "$@"

