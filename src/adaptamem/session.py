"""Load a PDB or system YAML into a planned job."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adaptamem.box import BoxPlan, plan_box
from adaptamem.doctor import DoctorReport, audit
from adaptamem.errors import RefuseError
from adaptamem.objective import Objective, parse_objective
from adaptamem.schema import Membrane, Orientation, System, load_protocol, load_system
from adaptamem.strategy import Strategy, choose


@dataclass
class Session:
    system: System
    report: DoctorReport
    objective: Objective
    box: BoxPlan
    strategy: Strategy
    budget_hours: float | None = None

    def gate_assemble(self, *, force: bool = False) -> None:
        r = self.strategy.refuse
        if r is None:
            return
        if "transmembrane" in r:
            raise RefuseError(r, code="NOT_MEMBRANE")
        if r.startswith("structure not production-ready"):
            if force:
                return
            raise RefuseError(r + "  (override with --force)", code="STRUCTURE")
        # Budget / sampling refuses do not block building the system.

    def gate_run(self, *, force: bool = False) -> None:
        if self.strategy.ok:
            return
        r = self.strategy.refuse or "refused"
        if force and r.startswith("structure not production-ready"):
            return
        if force and "transmembrane" not in r:
            return
        code = "NOT_MEMBRANE" if "transmembrane" in r else "BUDGET" if "GPU-h" in r else "STRUCTURE"
        raise RefuseError(r, code=code)


def load_session(
    path: Path,
    *,
    cli_objective: str | None = None,
    budget_hours: float | None = None,
) -> Session:
    path = Path(path)
    proto = load_protocol()
    if path.suffix.lower() in {".yml", ".yaml"}:
        system = load_system(path)
        obj = parse_objective(
            {
                "type": system.objective.type,
                "observables": [
                    {
                        "name": o.name,
                        "kind": o.kind,
                        "selection": o.selection,
                        "precision": o.precision,
                    }
                    for o in system.objective.observables
                ],
                "discover_cvs": system.objective.discover_cvs,
            },
            cli_type=cli_objective,
        )
        report = audit(system.structure)
    else:
        obj = parse_objective(
            {"discover_cvs": True},
            cli_type=cli_objective or "discover_states",
        )
        system = System(
            name=path.stem,
            structure=path,
            orientation=Orientation(),
            membrane=Membrane(),
            objective=obj,
            source=None,
            compute=proto.compute,
        )
        report = audit(path)
    hours = budget_hours
    if hours is None:
        hours = system.compute.max_gpu_hours
        if hours is None:
            hours = proto.compute.max_gpu_hours
    box = plan_box(
        report,
        water_pad_nm=system.membrane.water_pad_nm,
        safety_margin_nm=system.membrane.safety_margin_nm,
    )
    strategy = choose(report, obj, box, budget_gpu_hours=hours)
    return Session(
        system=system,
        report=report,
        objective=obj,
        box=box,
        strategy=strategy,
        budget_hours=hours,
    )
