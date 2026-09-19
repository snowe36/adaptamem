"""Choose a computational strategy. Axes are explicit; physics changes are labeled."""

from __future__ import annotations

from dataclasses import dataclass, field

from adaptamem.box import BoxPlan
from adaptamem.doctor import DoctorReport
from adaptamem.objective import Objective

# Claims of higher ns/day are only valid inside same_physics.
# Anything that changes resolution, lipids, or the ensemble is an approximation.
PHYSICS_SAME = "same_physics"
PHYSICS_APPROX = "approximation"


@dataclass
class Strategy:
    physics: str
    cheaper_step: list[str]
    fewer_steps: list[str]
    fewer_expensive_atoms: list[str]
    sampling: str
    fidelity: str
    refuse: str | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.refuse is None


def choose(
    report: DoctorReport,
    objective: Objective,
    box: BoxPlan,
    *,
    budget_gpu_hours: float | None = None,
) -> Strategy:
    reasons: list[str] = []
    if report.action_required:
        items = "; ".join(f.detail for f in report.action_required)
        return Strategy(
            physics=PHYSICS_SAME,
            cheaper_step=[],
            fewer_steps=[],
            fewer_expensive_atoms=[],
            sampling="none",
            fidelity="none",
            refuse=f"structure not production-ready: {items}",
            reasons=["doctor ACTION items must be resolved or explicitly overridden"],
        )
    if not report.tm_spans:
        reasons.append("no TM spans — not treated as a membrane protein")
        return Strategy(
            physics=PHYSICS_SAME,
            cheaper_step=[],
            fewer_steps=[],
            fewer_expensive_atoms=[],
            sampling="none",
            fidelity="none",
            refuse="no candidate transmembrane spans; not a membrane-protein job",
            reasons=reasons,
        )

    cheaper_step = [
        "HMR 4 fs if validation holds",
        "hardware/precision bench on this box",
    ]
    fewer_expensive_atoms = [
        f"geometry box ~{box.lateral_nm:g} nm lateral, ~{box.est_atoms} atoms",
        "water pad from protein extent, not a cubic CHARMM-GUI default",
    ]
    fewer_steps = [
        "equilibration scorecard (stop when ready, not at 50 ns)",
    ]
    physics = PHYSICS_SAME
    sampling = "conventional"
    fidelity = "atomistic"

    if objective.type == "membrane_environment":
        reasons.append("MH-style membrane question: annular lipids stay atomistic")
        fewer_expensive_atoms.append("do not CG or implicit-ize first-shell lipids")
        fewer_expensive_atoms.append("hybrid bulk only beyond ~1.2 nm annular shell")
        sampling = "adaptive" if objective.adaptive else "conventional"
        fewer_steps.append("stop on local thickness/order/water CIs")
    elif objective.type == "comparison":
        reasons.append("Δ between systems: paired sampling, not two independent movies")
        sampling = "adaptive_paired"
        fewer_steps.append("allocate GPU-hours to U(Δ), not to coverage of either system alone")
    elif objective.adaptive:
        sampling = "adaptive"
        fewer_steps.extend(
            [
                "pilot → state/CV discovery",
                "branch uncertain states, drop redundant walkers",
                "stop when U(objective) < precision",
            ]
        )
        if objective.discover_cvs:
            reasons.append("CVs not assumed known")
        reasons.append("same CHARMM36-AA ensemble as a conventional 100 ns baseline")
    else:
        fewer_steps.append("one production trajectory; length from ESS/CI if observables given")
        reasons.append("conventional objective: same-physics inner loop only")

    if box.est_atoms >= 400_000:
        reasons.append(f"~{box.est_atoms} atoms: full-AA campaign is expensive")
        if budget_gpu_hours is not None and budget_gpu_hours < 24:
            physics = PHYSICS_APPROX
            fidelity = "multi"
            fewer_expensive_atoms.append(
                "CG/hybrid exploration then AA promotion (labeled approximation)"
            )
            reasons.append("budget < 1 day: will not claim ns/day vs full AA")
        else:
            fewer_steps.append("consider domain decomposition before the full complex")

    if budget_gpu_hours is not None and sampling == "conventional" and box.est_atoms > 200_000:
        if budget_gpu_hours < 8:
            return Strategy(
                physics=physics,
                cheaper_step=cheaper_step,
                fewer_steps=fewer_steps,
                fewer_expensive_atoms=fewer_expensive_atoms,
                sampling=sampling,
                fidelity=fidelity,
                refuse=(
                    f"~{box.est_atoms} atoms, conventional run, "
                    f"{budget_gpu_hours:g} GPU-h is likely not enough; "
                    "raise budget or change objective to adaptive/discover_states"
                ),
                reasons=reasons,
            )

    return Strategy(
        physics=physics,
        cheaper_step=cheaper_step,
        fewer_steps=fewer_steps,
        fewer_expensive_atoms=fewer_expensive_atoms,
        sampling=sampling,
        fidelity=fidelity,
        refuse=None,
        reasons=reasons,
    )


def format_strategy(s: Strategy) -> str:
    lines = ["STRATEGY"]
    if s.refuse:
        lines.append(f"REFUSE  {s.refuse}")
        for r in s.reasons:
            lines.append(f"  because  {r}")
        return "\n".join(lines)
    lines.append(f"physics     {s.physics}")
    lines.append(f"sampling    {s.sampling}")
    lines.append(f"fidelity    {s.fidelity}")
    lines.append("cheaper timestep")
    lines.extend(f"  - {x}" for x in s.cheaper_step)
    lines.append("fewer timesteps")
    lines.extend(f"  - {x}" for x in s.fewer_steps)
    lines.append("fewer expensive atoms")
    lines.extend(f"  - {x}" for x in s.fewer_expensive_atoms)
    if s.reasons:
        lines.append("why")
        lines.extend(f"  - {x}" for x in s.reasons)
    if s.physics == PHYSICS_APPROX:
        lines.append(
            "NOTE  approximation — do not quote ns/day against a full-AA baseline"
        )
    return "\n".join(lines)
