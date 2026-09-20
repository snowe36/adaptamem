"""Research ladder. Score easy rungs; refuse to invent the rest."""

from __future__ import annotations

from typing import Any

from adaptamem.errors import RefuseError

# Implemented now vs named and refused. Order is the research ladder.
RUNGS: tuple[tuple[str, bool], ...] = (
    ("structural", True),
    ("flexibility", True),
    ("contacts", True),
    ("heterogeneity", True),
    ("populations", True),
    ("distributions", True),
    ("free_energy", False),
    ("transitions", False),
    ("rare_state", False),
    ("pathways", False),
    ("perturbation", False),
)

SCORED = tuple(name for name, ok in RUNGS if ok)
STUBBED = tuple(name for name, ok in RUNGS if not ok)

RATES_LIKE = frozenset({"free_energy", "transitions", "rare_state", "pathways", "perturbation"})


def require_rung(name: str) -> str:
    key = name.replace("-", "_")
    known = {n for n, _ in RUNGS}
    if key not in known:
        raise RefuseError(f"unknown ladder rung {name!r}", code="NOT_IMPLEMENTED")
    if key in STUBBED:
        raise RefuseError(
            f"{key} is not inferred from crystals or a short teacher",
            code="NOT_IMPLEMENTED",
        )
    return key


def identify(model: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """What the model may claim. Kinetics stay unidentified on a crystal prior."""
    kind = str(model.get("kind") or "").replace("-", "_")
    gpu = float(model.get("gpu_hours") or 0.0)
    obs = model.get("observables") or {}
    n_frames = [_n_frames(spec) for spec in obs.values()]
    n = min(n_frames) if n_frames else 0
    disconnected = any(bool(spec.get("disconnected")) for spec in obs.values() if isinstance(spec, dict))
    hops = any(_has_hops(spec) for spec in obs.values() if isinstance(spec, dict))
    kinetics = kind == "msm" and n >= 4 and hops and not disconnected
    occupancy = n >= 1
    pi_ok = kinetics
    identified = ["structural", "heterogeneity"]
    if occupancy:
        identified.append("populations" if pi_ok else "occupancy")
    if n >= 1:
        identified.append("distributions")
    unidentified = list(STUBBED)
    if not pi_ok:
        unidentified = ["pi"] + unidentified
    if not kinetics:
        unidentified = ["transitions", "timescales"] + [u for u in unidentified if u != "transitions"]
    return {
        "mode": mode,
        "kind": kind,
        "gpu_hours": gpu,
        "n_frames": n,
        "kinetics_identified": kinetics,
        "pi_identified": pi_ok,
        "occupancy_identified": occupancy,
        "identified": identified,
        "unidentified": list(dict.fromkeys(unidentified)),
        "rungs_scored": list(SCORED),
        "rungs_stubbed": list(STUBBED),
    }


def _n_frames(spec: object) -> int:
    if not isinstance(spec, dict):
        return 0
    if spec.get("n_frames") is not None:
        return int(spec["n_frames"])
    support = spec.get("support") or []
    return len(support)


def _has_hops(spec: dict[str, Any]) -> bool:
    if spec.get("cross_hops") is not None:
        return int(spec["cross_hops"]) > 0
    counts = spec.get("counts") or []
    n = len(counts)
    return any(counts[i][j] for i in range(n) for j in range(n) if i != j)
