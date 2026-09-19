#!/usr/bin/env bash
# GPU oracle only. Assembled system must already be on the pod.
# Always exit 0 so a refusal cannot crash-loop the meter.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/gpu_oracle.py
status=$?
if [ "$status" -ne 0 ]; then
  echo "ORACLE_FAIL rc=$status"
fi
python scripts/runpod_watchdog.py --kill --reason "oracle_exit_${status}" || true
exit 0
