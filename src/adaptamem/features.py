"""CPU feature extraction. Observables / traces, not more MD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.objective import Observable, parse_objective
from adaptamem.observables import AtomRef, evaluate


def traces_from_payload(data: dict[str, Any]) -> dict[str, list[float]]:
    raw = data.get("values") or data.get("traces") or data.get("features")
    if isinstance(raw, dict) and raw:
        return {str(k): [float(x) for x in v] for k, v in raw.items() if isinstance(v, list)}
    if all(isinstance(v, list) for v in data.values()) and data:
        return {str(k): [float(x) for x in v] for k, v in data.items()}
    raise RefuseError("no observable series (need values/traces/features)", code="NOT_READY")


def traces_from_path(path: Path) -> dict[str, list[float]]:
    return traces_from_payload(json.loads(Path(path).read_text()))


def traces_from_workdir(workdir: Path) -> dict[str, list[float]]:
    root = Path(workdir)
    for name in ("features.json", "produce.json", "scout.json", "teacher.json"):
        p = root / name
        if p.is_file():
            try:
                return traces_from_path(p)
            except RefuseError:
                continue
    pdb = root / "eq.pdb" if (root / "eq.pdb").is_file() else root / "assembled.pdb"
    meta = root / "assemble.json"
    if pdb.is_file() and meta.is_file():
        return traces_from_structure(pdb, observables_from_assemble(meta))
    raise RefuseError(f"no teacher traces in {root}", code="NOT_READY")


def observables_from_assemble(path: Path) -> list[Observable]:
    data = json.loads(Path(path).read_text())
    return list(parse_objective(data.get("objective") or {}).observables)


def traces_from_structure(
    pdb_path: Path,
    observables: list[Observable],
    *,
    chain: str | None = None,
) -> dict[str, list[float]]:
    """One CPU frame. Not an ensemble — coverage is a hole until a teacher shot exists."""
    if not observables:
        raise RefuseError(f"no observables to evaluate on {pdb_path}", code="NOT_READY")
    atoms, xyz = atoms_xyz_from_pdb(pdb_path, chain=chain)
    out: dict[str, list[float]] = {}
    for obs in observables:
        out[obs.name] = [float(evaluate(obs, atoms, xyz))]
    return out


def atoms_xyz_from_pdb(
    pdb_path: Path, *, chain: str | None = None
) -> tuple[list[AtomRef], list[tuple[float, float, float]]]:
    atoms: list[AtomRef] = []
    xyz: list[tuple[float, float, float]] = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        ch = line[21].strip() or "A"
        if chain is not None and ch != chain:
            continue
        name = line[12:16].strip()
        resname = line[17:20].strip()
        resid = int(line[22:26])
        x = float(line[30:38]) / 10.0
        y = float(line[38:46]) / 10.0
        z = float(line[46:54]) / 10.0
        atoms.append(
            AtomRef(index=len(atoms), name=name, resid=resid, chain=ch, resname=resname)
        )
        xyz.append((x, y, z))
    if not atoms:
        raise RefuseError(f"no ATOM records in {pdb_path}", code="STRUCTURE")
    return atoms, xyz


def write_features(traces: dict[str, list[float]], path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps({"features": traces, "gpu_hours": 0.0}, indent=2) + "\n")
    return path


def write_oracle_request(
    path: Path,
    *,
    uncertain: dict[str, list[int]],
    teacher_frames: dict[str, int],
    ns: float = 2.0,
) -> Path:
    """The named GPU shot. CPU already knows it cannot answer from this teacher."""
    payload = {
        "stage": "oracle",
        "ns": ns,
        "uncertain": uncertain,
        "teacher_frames": {str(k): int(v) for k, v in teacher_frames.items()},
        "reason": (
            "CPU teacher has no coverage of the CV. "
            "Named shot: short MD from each crystal start after eq.pdb exists. Not a campaign."
        ),
    }
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
