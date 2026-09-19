"""Crystal-frame CVs. Abort easy wells before anyone assembles or buys GPU."""

from __future__ import annotations

from pathlib import Path

from adaptamem.errors import RefuseError
from adaptamem.features import atoms_xyz_from_pdb
from adaptamem.objective import Observable
from adaptamem.observables import evaluate


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
