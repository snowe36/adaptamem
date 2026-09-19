"""Where the next GPU-hour goes. select is a plug; first impl may be random velocities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from adaptamem.diagnostics import mean_ci, uniqueness
from adaptamem.errors import RefuseError
from adaptamem.objective import Observable

EXTEND = "extend"
STOP = "stop"
KEEP = "keep"
BRANCH = "branch"

# Unique if this walker occupies bins others do not.
UNIQUE_CUTOFF = 0.3


@dataclass
class Decision:
    walker_id: str
    action: str
    unique: bool
    low_uncertainty: bool
    uniqueness: float
    reason: str


@dataclass
class SelectContext:
    walkers: dict[str, dict[str, list[float]]]
    observables: list[Observable]
    rng_seed: int = 42
    notes: list[str] = field(default_factory=list)


def _series(walkers: dict[str, dict[str, list[float]]], name: str) -> dict[str, list[float]]:
    return {wid: list(vals.get(name) or []) for wid, vals in walkers.items()}


def _low_uncertainty(values: list[float], precision: float) -> bool:
    _, half = mean_ci(values)
    return half <= precision


def decide_axes(low_u: bool, unique: bool) -> str:
    """Uncertainty × uniqueness. Never stop solely because CI is tight."""
    if low_u and not unique:
        return STOP
    if low_u and unique:
        return KEEP
    if (not low_u) and unique:
        return BRANCH
    return EXTEND


def _per_walker(ctx: SelectContext) -> list[Decision]:
    out: list[Decision] = []
    ids = list(ctx.walkers)
    for wid in ids:
        u_flags: list[bool] = []
        uniq_scores: list[float] = []
        traces = ctx.walkers[wid]
        for obs in ctx.observables:
            vals = list(traces.get(obs.name) or [])
            prec = obs.precision if obs.precision is not None else 0.5
            u_flags.append(_low_uncertainty(vals, prec))
            others = [
                list(ctx.walkers[o].get(obs.name) or []) for o in ids if o != wid
            ]
            uniq_scores.append(uniqueness(vals, others))
        if not ctx.observables:
            # No named CV: treat as high-uncertainty / non-unique → extend.
            out.append(
                Decision(
                    walker_id=wid,
                    action=EXTEND,
                    unique=False,
                    low_uncertainty=False,
                    uniqueness=0.0,
                    reason="no named observable; extend",
                )
            )
            continue
        low_u = all(u_flags) if u_flags else False
        uniq = max(uniq_scores) if uniq_scores else 0.0
        unique = uniq >= UNIQUE_CUTOFF
        action = decide_axes(low_u, unique)
        reason = f"uncertainty={'low' if low_u else 'high'} uniqueness={uniq:.2f} → {action}"
        out.append(
            Decision(
                walker_id=wid,
                action=action,
                unique=unique,
                low_uncertainty=low_u,
                uniqueness=uniq,
                reason=reason,
            )
        )
    return out


def select_coverage(ctx: SelectContext) -> list[Decision]:
    return _per_walker(ctx)


def select_random(ctx: SelectContext) -> list[Decision]:
    """Independent velocity replicas: keep everyone until the budget says stop."""
    return [
        Decision(
            walker_id=wid,
            action=EXTEND,
            unique=False,
            low_uncertainty=False,
            uniqueness=0.0,
            reason="select=random (independent velocities)",
        )
        for wid in ctx.walkers
    ]


def select_uncertainty(ctx: SelectContext) -> list[Decision]:
    return _per_walker(ctx)


def select_novelty(ctx: SelectContext) -> list[Decision]:
    dec = _per_walker(ctx)
    for d in dec:
        if d.unique and d.action == EXTEND:
            d.action = BRANCH
            d.reason += " (novelty prefers branch)"
    return dec


def select_information_gain(ctx: SelectContext) -> list[Decision]:
    return select_uncertainty(ctx)


POLICIES: dict[str, Callable[[SelectContext], list[Decision]]] = {
    "random": select_random,
    "uncertainty": select_uncertainty,
    "novelty": select_novelty,
    "information_gain": select_information_gain,
    "coverage": select_coverage,
}


def get_policy(name: str) -> Callable[[SelectContext], list[Decision]]:
    key = name.replace("-", "_")
    if key not in POLICIES:
        raise RefuseError(
            f"unknown select policy {name!r}; use {', '.join(sorted(POLICIES))}",
            code="NOT_IMPLEMENTED",
        )
    return POLICIES[key]
