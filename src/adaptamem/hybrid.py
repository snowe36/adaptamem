"""Hybrid AA/CG plan. Approximation. Annular lipids stay AA for membrane questions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.objective import Objective
from adaptamem.strategy import PHYSICS_APPROX, PHYSICS_SAME, Strategy

ANNULAR_NM = 1.2
IMPLICIT = {"implicit", "u_membrane", "continuum"}


@dataclass
class HybridPlan:
    protein: str
    annular: str
    annular_shell_nm: float
    bulk_membrane: str
    water: str
    physics: str
    refuse: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.refuse is None


def plan_hybrid(
    objective: Objective,
    strategy: Strategy,
    *,
    bulk: str = "CG",
) -> HybridPlan:
    bulk = bulk.upper() if bulk.lower() != "implicit" else "implicit"
    notes: list[str] = []
    protein = "AA"
    annular = "AA"
    water = "CG" if bulk in {"CG", "MARTINI"} else "AA"
    physics = PHYSICS_SAME

    if bulk.lower() in IMPLICIT or bulk == "implicit":
        if objective.type == "membrane_environment":
            raise RefuseError(
                "membrane_environment forbids implicit / U_membrane on first-shell lipids"
            )
        notes.append("implicit bulk is an approximation")
        physics = PHYSICS_APPROX
        bulk_mem = "implicit"
        annular = "AA"
        water = "implicit"
    elif bulk in {"CG", "MARTINI"}:
        physics = PHYSICS_APPROX
        bulk_mem = "CG"
        notes.append("CG bulk is an approximation — do not quote ns/day vs full AA")
        notes.append("CG → AA is backmapping + re-eq, not a restart")
    else:
        bulk_mem = "AA"
        water = "AA"
        notes.append("full AA; hybrid not applied")

    if objective.type == "membrane_environment":
        annular = "AA"
        notes.append(f"annular lipids stay AA within {ANNULAR_NM:g} nm of the protein")
        if bulk_mem != "AA":
            notes.append("only lipids beyond the first shell may be CG")

    if strategy.fidelity == "atomistic" and bulk_mem == "AA":
        physics = PHYSICS_SAME

    return HybridPlan(
        protein=protein,
        annular=annular,
        annular_shell_nm=ANNULAR_NM,
        bulk_membrane=bulk_mem,
        water=water,
        physics=physics,
        notes=notes,
    )


def format_hybrid(p: HybridPlan) -> str:
    if p.refuse:
        return f"HYBRID  REFUSE  {p.refuse}"
    lines = [
        "HYBRID  plan (not a Martini engine)",
        f"physics     {p.physics}",
        f"protein     {p.protein}",
        f"annular     {p.annular}  ({p.annular_shell_nm:g} nm)",
        f"bulk lipids {p.bulk_membrane}",
        f"water       {p.water}",
    ]
    for n in p.notes:
        lines.append(f"  note  {n}")
    if p.physics == PHYSICS_APPROX:
        lines.append("NOTE  approximation — do not quote ns/day against a full-AA baseline")
    return "\n".join(lines)


def write_hybrid(p: HybridPlan, workdir: Path, extra: dict[str, Any] | None = None) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": "0.1",
        "created_utc": datetime.now(UTC).isoformat(),
        "protein": p.protein,
        "annular": p.annular,
        "annular_shell_nm": p.annular_shell_nm,
        "bulk_membrane": p.bulk_membrane,
        "water": p.water,
        "physics": p.physics,
        "executed": False,
        "notes": p.notes,
        **(extra or {}),
    }
    path = workdir / "hybrid.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
