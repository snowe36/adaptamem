"""Adaptive walkers. CVs come from YAML; walker count is a scheduler output."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.box import BoxPlan
from adaptamem.errors import RefuseError
from adaptamem.objective import Objective
from adaptamem.strategy import PHYSICS_SAME

# Rough 4090-class cost model for scheduling only — not a published ns/day claim.
NS_PER_DAY_PER_100K = 200.0


@dataclass
class WalkerSchedule:
    n_pilot: int
    ns_pilot: float
    n_walkers: int
    ns_per_walker: float
    stop_rule: str
    physics: str
    discover_cvs: bool
    notes: list[str] = field(default_factory=list)

    @property
    def total_ns(self) -> float:
        return self.n_pilot * self.ns_pilot + self.n_walkers * self.ns_per_walker


def mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """Return (mean, 95% CI half-width). Empty → (nan, inf)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("inf")
    mu = sum(values) / n
    if n == 1:
        return mu, float("inf")
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var / n)
    return mu, z * se


def stopped(values: list[float], precision: float) -> bool:
    _, half = mean_ci(values)
    return half <= precision


def schedule(
    objective: Objective,
    box: BoxPlan,
    *,
    budget_hours: float | None = None,
    traces: dict[str, list[float]] | None = None,
) -> WalkerSchedule:
    notes: list[str] = []
    if objective.type == "conventional" and not objective.observables:
        return WalkerSchedule(
            n_pilot=0,
            ns_pilot=0.0,
            n_walkers=1,
            ns_per_walker=100.0,
            stop_rule="fixed 100 ns (conventional)",
            physics=PHYSICS_SAME,
            discover_cvs=False,
            notes=["conventional: one trajectory; not adaptive"],
        )

    ns_day = max(20.0, NS_PER_DAY_PER_100K * (100_000 / max(box.est_atoms, 1)))
    hours = budget_hours if budget_hours is not None else 24.0
    budget_ns = ns_day * (hours / 24.0)

    n_pilot = 4 if objective.discover_cvs else 0
    ns_pilot = 10.0 if n_pilot else 0.0
    remaining = max(0.0, budget_ns - n_pilot * ns_pilot)

    if traces:
        n_keep = 0
        for obs in objective.observables:
            vals = traces.get(obs.name) or []
            prec = obs.precision if obs.precision is not None else 0.5
            if stopped(vals, prec):
                notes.append(f"{obs.name}: CI inside precision={prec:g}; drop walker")
            else:
                n_keep += 1
                notes.append(f"{obs.name}: still uncertain")
        n_walkers = max(n_keep, 1 if objective.observables else 2)
    else:
        n_walkers = 8 if objective.adaptive else 1
        if objective.type == "comparison":
            n_walkers = max(n_walkers, 4)
            notes.append("paired Δ: walkers split across systems")
        notes.append("n_walkers from budget/objective, not a user input")

    ns_each = remaining / max(n_walkers, 1) if n_walkers else 0.0
    if ns_each < 5.0 and budget_hours is not None:
        raise RefuseError(
            f"budget {budget_hours:g} GPU-h → ~{ns_each:.1f} ns/walker; "
            "not enough to reduce U(objective). Raise budget or drop observables."
        )
    ns_each = max(ns_each, 20.0) if budget_hours is None else max(ns_each, 5.0)

    stop = "stop when CI of named observables ≤ precision"
    if objective.discover_cvs:
        stop = "pilot TICA/PCA → branch uncertain states; " + stop
        notes.append("CVs from YAML observables or the pilot, not Python")

    return WalkerSchedule(
        n_pilot=n_pilot,
        ns_pilot=ns_pilot,
        n_walkers=n_walkers,
        ns_per_walker=round(ns_each, 1),
        stop_rule=stop,
        physics=PHYSICS_SAME,
        discover_cvs=objective.discover_cvs,
        notes=notes,
    )


def format_schedule(s: WalkerSchedule) -> str:
    lines = [
        "SAMPLE  scheduler output",
        f"physics     {s.physics}",
        f"pilot       {s.n_pilot} × {s.ns_pilot:g} ns" if s.n_pilot else "pilot       none",
        f"walkers     {s.n_walkers} × {s.ns_per_walker:g} ns",
        f"total       ~{s.total_ns:g} ns",
        f"stop        {s.stop_rule}",
    ]
    for n in s.notes:
        lines.append(f"  note  {n}")
    return "\n".join(lines)


def write_schedule(s: WalkerSchedule, workdir: Path, extra: dict[str, Any] | None = None) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": "0.1",
        "created_utc": datetime.now(UTC).isoformat(),
        "n_pilot": s.n_pilot,
        "ns_pilot": s.ns_pilot,
        "n_walkers": s.n_walkers,
        "ns_per_walker": s.ns_per_walker,
        "total_ns": s.total_ns,
        "stop_rule": s.stop_rule,
        "physics": s.physics,
        "discover_cvs": s.discover_cvs,
        "notes": s.notes,
        **(extra or {}),
    }
    path = workdir / "sample.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
