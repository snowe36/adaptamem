#!/usr/bin/env bash
# Entire container command when the CUDA image is used. No pip, no GPU_JOB_B64.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/gpu_campaign.py
sleep 120
