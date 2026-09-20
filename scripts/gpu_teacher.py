#!/usr/bin/env python3
"""Matching-stack eq then a tiny teacher from each crystal start.

No assemble. No CPU fallback. Packing is not a teacher.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def log(msg: str) -> None:
    print(msg, flush=True)


def tiny_cuda() -> None:
    from openmm import Context, Platform, System, VerletIntegrator, unit

    plat = Platform.getPlatformByName("CUDA")
    system = System()
    system.addParticle(1.0)
    integ = VerletIntegrator(0.001 * unit.picoseconds)
    ctx = Context(system, integ, plat)
    ctx.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)
    ctx.getState(getEnergy=True)
    log("TINY_CUDA_OK")


def one_start(workdir: Path, ns: float) -> None:
    from adaptamem.equilibrate import equilibrate
    from adaptamem.produce import produce

    log(f"teacher   {workdir}  eq then {ns:g} ns")
    if not (workdir / "eq.pdb").is_file():
        eq = equilibrate(workdir, short=False, progress=log)
        log(f"EQ        {eq.platform}  {eq.pdb}")
        if not eq.qc.ok:
            raise RuntimeError(f"membrane QC failed in {workdir}")
    else:
        log(f"EQ        skip {workdir / 'eq.pdb'}")
    result = produce(workdir, ns=ns, force=True, progress=log)
    import json

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
                "workdir": str(workdir),
            },
            indent=2,
        )
        + "\n"
    )
    log(f"TEACHER   ns={result.ns:g}  gpu-h={result.gpu_hours:.4f}  {workdir / 'teacher.json'}")


def main() -> int:
    os.chdir(ROOT)
    from adaptamem.correction import require_teacher_gpu
    from adaptamem.errors import RefuseError
    from adaptamem.gpu_contract import enable_no_cpu_fallback, stack_note

    try:
        require_teacher_gpu()
    except RefuseError as exc:
        log(exc.format())
        print("TEACHER_FAIL", flush=True)
        return 2
    enable_no_cpu_fallback()
    log(stack_note())
    try:
        tiny_cuda()
    except Exception as exc:
        log(f"TEACHER_FAIL  CUDA probe: {exc}")
        print("TEACHER_FAIL", flush=True)
        return 2

    raw = os.environ.get("ADAPTAMEM_WORKDIRS") or "runs/2rh1,runs/3sn6"
    ns = float(os.environ.get("ORACLE_NS") or "2")
    workdirs = [Path(p.strip()) for p in raw.split(",") if p.strip()]
    try:
        for wd in workdirs:
            assembled = wd / "assembled.pdb"
            xml = wd / "system.xml"
            if not assembled.is_file() or not xml.is_file():
                log(f"TEACHER_FAIL  no assembled system in {wd}")
                print("TEACHER_FAIL", flush=True)
                return 2
            one_start(wd, ns)
    except Exception:
        traceback.print_exc()
        print("TEACHER_FAIL", flush=True)
        return 2
    print("TEACHER_DONE", flush=True)
    print("CAMPAIGN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
