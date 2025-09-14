#!/usr/bin/env bash
set -euo pipefail

python -m src.mode_sawit.inference.predict "$@"

