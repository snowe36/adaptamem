"""Scientific objective — what Adaptamem is allowed to spend GPU time answering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OBJECTIVE_TYPES = ("conventional", "conformational_shift", "discover_states")


@dataclass
class Observable:
    name: str
    kind: str
    selection: str
    precision: float | None = None


@dataclass
class Objective:
    type: str = "conventional"
    observables: list[Observable] = field(default_factory=list)
    discover_cvs: bool = False

    @property
    def adaptive(self) -> bool:
        return self.type != "conventional"


def parse_objective(raw: Any, *, cli_type: str | None = None) -> Objective:
    if cli_type:
        kind = cli_type.replace("-", "_")
    elif isinstance(raw, dict):
        kind = str(raw.get("type", "conventional")).replace("-", "_")
    else:
        kind = "conventional"
    if kind not in OBJECTIVE_TYPES:
        raise ValueError(f"objective.type must be one of {OBJECTIVE_TYPES}")

    data = raw if isinstance(raw, dict) else {}
    obs: list[Observable] = []
    for i, item in enumerate(data.get("observables") or []):
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError(f"objective.observables[{i}] needs at least name")
        obs.append(
            Observable(
                name=str(item["name"]),
                kind=str(item.get("kind", "distance")),
                selection=str(item.get("selection", "")),
                precision=_opt_float(item.get("precision")),
            )
        )
    discover = bool(data.get("discover_cvs", kind == "discover_states"))
    if kind == "discover_states":
        discover = True
    if kind == "conformational_shift" and not obs and not discover:
        raise ValueError(
            "conformational_shift needs observables or discover_cvs: true"
        )
    return Objective(type=kind, observables=obs, discover_cvs=discover)


def _opt_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)
