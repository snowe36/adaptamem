#!/usr/bin/env bash
# Runpod 4090-class throughput baseline + optional short produce.
# Bench is ns/day(N). Sample is a separate scientific-efficiency command.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "runpodctl not on PATH" >&2
  exit 1
fi
runpodctl user >/dev/null

GPU="${ADAPTAMEM_GPU:-NVIDIA GeForce RTX 4090}"
STEPS="${BENCH_STEPS:-8000}"
PAD_LADDER="${PAD_LADDER:-1.2 2.5 4.0 6.0}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
if command -v uv >/dev/null 2>&1; then
  uv sync --extra sim --extra dev
  ADAPTAMEM=(uv run adaptamem)
else
  python3 -m pip install -e ".[sim]"
  ADAPTAMEM=(adaptamem)
fi

mkdir -p data/structures runs
if [[ ! -f tests/fixtures/helix.pdb ]]; then
  echo "missing helix fixture" >&2
  exit 1
fi

i=0
points=()
for pad in $PAD_LADDER; do
  work="runs/ladder_${i}"
  mkdir -p "$work"
  # Copy helix; pad is currently the system YAML safety/water — use assemble default
  # and record whatever atom count we got. Varying pad needs a YAML.
  cat > "$work/system.yaml" <<YAML
name: helix_ladder_${i}
structure: $(realpath tests/fixtures/helix.pdb)
objective:
  type: conventional
orientation:
  method: auto
membrane:
  lipids:
    POPC: 1.0
  optimize_size: true
  safety_margin_nm: ${pad}
  water_pad_nm: ${pad}
compute:
  max_gpu_hours: 1
  max_wall_hours: 1
seed: 42
YAML
  "${ADAPTAMEM[@]}" assemble "$work/system.yaml" --out "$work" --force
  "${ADAPTAMEM[@]}" equilibrate "$work" --short || true
  "${ADAPTAMEM[@]}" bench "$work" --steps "$STEPS" --ladder
  i=$((i + 1))
done

"${ADAPTAMEM[@]}" bench runs/ladder_0 --steps "$STEPS"
echo "GPU=$GPU"
echo "bench artifacts under runs/ladder_*"
