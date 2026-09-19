#!/usr/bin/env python3
"""GPU oracle only. Ship an assembled system. Short MD. Write traces. Leave.

No fetch, assemble, eq campaign, bench, or scout.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    os.chdir(ROOT)
    workdir = Path(os.environ.get("ADAPTAMEM_WORKDIR") or "runs/oracle")
    ns = float(os.environ.get("ORACLE_NS") or "10")
    assembled = workdir / "assembled.pdb"
    xml = workdir / "system.xml"
    eq = workdir / "eq.pdb"
    if not assembled.is_file() or not xml.is_file():
        log("ORACLE_FAIL  no assembled.pdb/system.xml; CPU prepare + ship first")
        print("ORACLE_FAIL", flush=True)
        return 2
    if not eq.is_file():
        log("ORACLE_FAIL  no eq.pdb; addMembrane packing is not a teacher (4 fs NaNs)")
        print("ORACLE_FAIL", flush=True)
        return 2

    from adaptamem.produce import produce

    log(f"oracle    {workdir}  {ns:g} ns")
    result = produce(workdir, ns=ns, force=True, progress=log)
    (workdir / "teacher.json").write_text(
        json.dumps(
            {
                "method": "oracle",
                "values": result.traces,
                "gpu_hours": result.gpu_hours,
                "ns": result.ns,
                "cpu_hours": 0.0,
                "role": "teacher",
                "leaked": False,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"ORACLE    ns={result.ns:g}  gpu-h={result.gpu_hours:.4f}  {workdir / 'teacher.json'}")
    log("ORACLE_DONE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("ORACLE_FAIL", flush=True)
        raise SystemExit(0) from None
