"""Local PPM 3.0 via BioMembHub `immers`. Not vendored; refuse if the binary is missing."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from adaptamem.errors import RefuseError

ENV = "ADAPTAMEM_PPM_DIR"
INSTALL = (
    "Clone https://github.com/BioMembHub/ppm3_server_code, `make`, "
    f"set {ENV} to that directory or put `immers` on PATH. "
    "Do not use auto orientation and call it PPM."
)


def find_immers(immers_dir: Path | None = None) -> Path | None:
    if immers_dir is not None:
        cand = Path(immers_dir) / "immers"
        return cand if cand.is_file() else None
    env = os.environ.get(ENV, "").strip()
    if env:
        p = Path(env)
        for cand in (p / "immers", p):
            if cand.is_file() and cand.name == "immers":
                return cand
        if (p / "immers").is_file():
            return p / "immers"
    which = shutil.which("immers")
    if which:
        return Path(which)
    for root in (Path.home() / "ppm3_server_code", Path("/workspace/ppm3"), Path("/tmp/ppm3_server_code")):
        if (root / "immers").is_file():
            return root / "immers"
    return None


def require_immers(immers_dir: Path | None = None) -> Path:
    bin_path = find_immers(immers_dir)
    if bin_path is None:
        raise RefuseError(f"orientation.method=ppm but immers was not found. {INSTALL}", code="PPM_MISSING")
    return bin_path


def run_immers(
    structure: Path,
    dest: Path,
    *,
    topology: str = "in",
    immers_dir: Path | None = None,
    membrane: str = "DOPC",
) -> Path:
    """Run immers; write dest PDB. Leaflet `out` is applied by the caller as a z-flip."""
    immers = require_immers(immers_dir)
    structure = Path(structure)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="adaptamem_ppm_"))
    inp = work / "prot.pdb"
    shutil.copy2(structure, inp)
    # immers CLI is historically: immers <pdb>  (writes in cwd). Keep the call minimal.
    try:
        proc = subprocess.run(
            [str(immers), str(inp)],
            cwd=work,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RefuseError("PPM immers timed out") from exc
    except FileNotFoundError as exc:
        raise RefuseError(f"failed to execute immers: {exc}") from exc
    out_pdb = _find_output_pdb(work, inp)
    if proc.returncode != 0 or out_pdb is None:
        err = (proc.stderr or proc.stdout or "").strip()[:500]
        raise RefuseError(f"PPM immers failed (exit {proc.returncode}). {err or INSTALL}")
    shutil.copy2(out_pdb, dest)
    return dest


def _find_output_pdb(work: Path, inp: Path) -> Path | None:
    candidates = list(work.glob("*.pdb"))
    # Prefer a file that is not the input copy.
    for p in candidates:
        if p.resolve() != inp.resolve() and p.stat().st_size > 0:
            return p
    if inp.exists() and inp.stat().st_size > 0:
        # Some immers builds overwrite the input.
        return inp
    return None
