"""Crystal-frame CVs and CPU-only structural insight. Abort easy wells first."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.features import atoms_xyz_from_pdb
from adaptamem.ladder import STUBBED
from adaptamem.objective import Observable
from adaptamem.observables import AtomRef, evaluate


def parse_ref(spec: str) -> tuple[Path, str | None]:
    raw = spec.strip()
    if ":" in raw:
        path_s, maybe = raw.rsplit(":", 1)
        if maybe and len(maybe) <= 2 and maybe.isalnum() and Path(path_s).suffix.lower() == ".pdb":
            return Path(path_s), maybe
    return Path(raw), None


def traces_from_crystals(
    refs: list[str],
    observables: list[Observable],
) -> dict[str, list[float]]:
    if len(refs) < 2:
        raise RefuseError("crystal gate needs at least two structures", code="NOT_READY")
    merged: dict[str, list[float]] = {o.name: [] for o in observables}
    for spec in refs:
        path, chain = parse_ref(spec)
        if not path.is_file():
            raise RefuseError(f"no structure {path}", code="STRUCTURE")
        atoms, xyz = atoms_xyz_from_pdb(path, chain=chain)
        for obs in observables:
            merged[obs.name].append(float(evaluate(obs, atoms, xyz)))
    return merged


def gate_span(traces: dict[str, list[float]], observables: list[Observable]) -> dict[str, float]:
    """Refuse if every named CV is shorter than its precision across crystals."""
    spans: dict[str, float] = {}
    failed: list[str] = []
    by_name = {o.name: o for o in observables}
    for name, xs in traces.items():
        if len(xs) < 2:
            raise RefuseError(f"{name}: crystal gate needs ≥2 frames", code="NOT_READY")
        span = max(xs) - min(xs)
        spans[name] = span
        prec = by_name.get(name)
        eps = float(prec.precision) if prec is not None and prec.precision is not None else None
        if eps is not None and span < eps:
            failed.append(f"{name} span {span:.3f} nm < ε={eps:g}")
    if failed and len(failed) == len(traces):
        raise RefuseError(
            "crystal gate abort: " + "; ".join(failed) + ". Do not buy GPU.",
            code="EASY_WELL",
        )
    return spans


def ca_contacts(
    atoms: list[AtomRef],
    xyz: list[tuple[float, float, float]],
    *,
    cutoff_nm: float = 0.8,
    min_sep: int = 4,
) -> set[tuple[int, int]]:
    cas = [(a.resid, xyz[i]) for i, a in enumerate(atoms) if a.name == "CA"]
    out: set[tuple[int, int]] = set()
    cut2 = cutoff_nm * cutoff_nm
    for i, (ri, pi) in enumerate(cas):
        for rj, pj in cas[i + 1 :]:
            if abs(ri - rj) < min_sep:
                continue
            d2 = (pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2 + (pi[2] - pj[2]) ** 2
            if d2 <= cut2:
                out.add((ri, rj) if ri < rj else (rj, ri))
    return out


def bfactor_flexibility(pdb_path: Path, *, chain: str | None = None) -> dict[str, float]:
    """PDB B-factors as an RMSF placeholder. Not dynamics."""
    vals: list[float] = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM") or len(line) < 66:
            continue
        ch = line[21].strip() or "A"
        if chain is not None and ch != chain:
            continue
        if line[12:16].strip() != "CA":
            continue
        raw = line[60:66].strip()
        if not raw:
            continue
        vals.append(float(raw))
    if not vals:
        return {"n_ca": 0, "mean_b": 0.0}
    return {"n_ca": len(vals), "mean_b": sum(vals) / len(vals), "max_b": max(vals)}


def crystal_insight(
    refs: list[str],
    observables: list[Observable],
    *,
    cutoff_nm: float = 0.8,
) -> dict[str, Any]:
    traces = traces_from_crystals(refs, observables)
    spans = gate_span(traces, observables)
    frames: list[tuple[Path, str | None, set[tuple[int, int]], dict[str, float]]] = []
    for spec in refs:
        path, chain = parse_ref(spec)
        atoms, xyz = atoms_xyz_from_pdb(path, chain=chain)
        frames.append((path, chain, ca_contacts(atoms, xyz, cutoff_nm=cutoff_nm), bfactor_flexibility(path, chain=chain)))
    contact_sets = [c for _, _, c, _ in frames]
    shared = set.intersection(*contact_sets) if contact_sets else set()
    unique: list[int] = []
    for c in contact_sets:
        unique.append(len(c - shared))
    return {
        "mode": "cpu_only",
        "gpu_hours": 0.0,
        "features": traces,
        "heterogeneity": spans,
        "flexibility": [{"path": str(p), "chain": ch, **flex} for p, ch, _, flex in frames],
        "contacts": {
            "cutoff_nm": cutoff_nm,
            "n_shared": len(shared),
            "n_unique": unique,
            "n_per_frame": [len(c) for c in contact_sets],
        },
        "identified": ["structural", "flexibility", "contacts", "heterogeneity"],
        "unidentified": list(dict.fromkeys(["pi", "transitions", "timescales", *STUBBED])),
    }
