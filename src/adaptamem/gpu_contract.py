"""When GPU is allowed: matching OpenMM/CUDA, eq.pdb, no CPU fallback."""

from __future__ import annotations

import os
from pathlib import Path

from adaptamem.errors import RefuseError

CUDA_VERSION = "12.8"
OPENMM_STACK = "conda-forge openmm with cuda-version=12.8"
NO_CPU_FALLBACK_ENV = "ADAPTAMEM_NO_CPU_FALLBACK"


def require_eq_pdb(workdir: Path) -> Path:
    eq = Path(workdir) / "eq.pdb"
    if not eq.is_file():
        raise RefuseError(
            f"no eq.pdb in {workdir}; addMembrane packing is not a teacher",
            code="NOT_READY",
        )
    return eq


def no_cpu_fallback() -> bool:
    return os.environ.get(NO_CPU_FALLBACK_ENV) == "1"


def enable_no_cpu_fallback() -> None:
    os.environ[NO_CPU_FALLBACK_ENV] = "1"


def refuse_cpu_platform(platform: str) -> None:
    if no_cpu_fallback() and platform == "CPU":
        raise RefuseError(
            "CPU fallback disabled on a billed GPU; CUDA 12.8 OpenMM required",
            code="NOT_READY",
        )


def stack_note() -> str:
    return f"CUDA {CUDA_VERSION}; {OPENMM_STACK}; eq.pdb required; no CPU fallback"
