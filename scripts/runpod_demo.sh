#!/usr/bin/env bash
# On-GPU (or any OpenMM box): one-shot CHARMM36 throughput.
# Caches assembled.pdb in --out. Second run is just the timed loop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STEPS="${BENCH_STEPS:-4000}"
OUT="${OUT:-runs/gpu}"
PADS="${PADS:-}"

if command -v uv >/dev/null 2>&1; then
  uv sync --extra sim
  PY=(uv run python)
else
  python3 -m pip install -e ".[sim]"
  PY=(python3)
fi

args=(scripts/gpu_job.py --out "$OUT" --steps "$STEPS")
if [[ -n "$PADS" ]]; then
  args+=(--pads "$PADS")
fi
"${PY[@]}" "${args[@]}"
echo "wrote $OUT/bench.json"
