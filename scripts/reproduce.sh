#!/usr/bin/env bash
# CPU reproduce: lint + unit tests + doctor/plan on the committed helix fixture.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  uv sync --extra dev
  RUFF=(uv run ruff)
  PYTEST=(uv run pytest)
  ADAPTAMEM=(uv run adaptamem)
else
  PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
  PYTHON="${PYTHON:-python3.11}"
  "$PYTHON" -m pip install -e ".[dev]"
  RUFF=(ruff)
  PYTEST=(pytest)
  ADAPTAMEM=(adaptamem)
fi

"${RUFF[@]}" check .
"${PYTEST[@]}" -q
"${ADAPTAMEM[@]}" doctor tests/fixtures/helix.pdb
"${ADAPTAMEM[@]}" plan tests/fixtures/helix.pdb --objective discover-states

echo ""
echo "CPU path ok. OpenMM assemble/eq is extra: uv sync --extra dev --extra sim"
