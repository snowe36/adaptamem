"""Iterative sampling. select is a plug; walker count is an output."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.box import BoxPlan
from adaptamem.campaign import stamp, write_campaign
from adaptamem.diagnostics import Diagnostics, mean_ci, summarize
from adaptamem.errors import RefuseError
from adaptamem.objective import Objective
from adaptamem.produce import ns_from_budget
from adaptamem.select import (
    BRANCH,
    EXTEND,
    KEEP,
    STOP,
    Decision,
    SelectContext,
    get_policy,
)
from adaptamem.strategy import PHYSICS_SAME

# Rough 4090-class cost model for scheduling only — not a published ns/day claim.
NS_PER_DAY_PER_100K = 200.0
LOOP = ("initialize", "pilot", "run_chunk", "analyze", "select", "branch", "stop")

# Re-export for existing tests.
stopped = lambda values, precision: mean_ci(values)[1] <= precision  # noqa: E731


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


@dataclass
class SampleResult:
    schedule: WalkerSchedule
    select: str
    loop: list[str]
    decisions: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    traces: dict[str, dict[str, list[float]]]
    stopped: bool
    refused: str | None
    path: Path | None = None
    executed: bool = False


def schedule(
    objective: Objective,
    box: BoxPlan,
    *,
    budget_hours: float | None = None,
    traces: dict[str, list[float]] | None = None,
    select: str = "random",
) -> WalkerSchedule:
    notes: list[str] = []
    if objective.type == "conventional" and not objective.observables:
        return WalkerSchedule(
            n_pilot=0,
            ns_pilot=0.0,
            n_walkers=1,
            ns_per_walker=100.0,
            stop_rule="fixed length (conventional); budget still caps produce",
            physics=PHYSICS_SAME,
            discover_cvs=False,
            notes=["conventional: one trajectory; not adaptive"],
        )

    ns_day = max(20.0, NS_PER_DAY_PER_100K * (100_000 / max(box.est_atoms, 1)))
    hours = budget_hours if budget_hours is not None else 24.0
    budget_ns = ns_from_budget(hours, ns_day)

    n_pilot = 4 if objective.discover_cvs else 0
    ns_pilot = 1.0 if n_pilot else 0.0
    remaining = max(0.0, budget_ns - n_pilot * ns_pilot)

    if traces:
        # Flatten named traces into a fake single-walker map for a keep-count.
        fake = {"w0": traces}
        ctx = SelectContext(walkers=fake, observables=list(objective.observables))
        dec = get_policy("coverage")(ctx)
        n_keep = sum(1 for d in dec if d.action != STOP)
        notes.extend(d.reason for d in dec)
        n_walkers = max(n_keep, 1 if objective.observables else 2)
    else:
        n_walkers = 8 if objective.adaptive else 1
        if objective.type == "comparison":
            n_walkers = max(n_walkers, 4)
            notes.append("paired Δ: walkers split across systems")
        notes.append("n_walkers from budget/objective, not a user input")
        notes.append("pilot maps an initial landscape; it is not a replica count")

    ns_each = remaining / max(n_walkers, 1) if n_walkers else 0.0
    if ns_each < 5.0 and budget_hours is not None:
        raise RefuseError(
            f"Target precision cannot be reached within the supplied "
            f"{budget_hours:g} GPU-hour budget.",
            code="BUDGET",
        )
    ns_each = max(ns_each, 20.0) if budget_hours is None else max(ns_each, 5.0)

    stop = (
        "stop when CI ≤ precision AND coverage is adequate; "
        "never kill a unique state because its CI is tight"
    )
    if objective.discover_cvs:
        notes.append("CVs from YAML observables or the pilot, not Python")
    notes.append(f"select={select} (plug; random = independent velocities)")

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


def _flatten(traces: dict[str, dict[str, list[float]]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for _wid, named in traces.items():
        for name, vals in named.items():
            out.setdefault(name, []).extend(vals)
    return out


def _coverage_refuse(diag: Diagnostics, n_walkers: int) -> str | None:
    if diag.state_coverage is not None and diag.state_coverage <= 1.0 / 12 and n_walkers >= 1:
        if diag.ci95 is not None and diag.n >= 8:
            return (
                "Only one metastable state has been observed. "
                "A confidence interval would be misleading."
            )
    return None


def analyze(
    traces: dict[str, dict[str, list[float]]],
    objective: Objective,
    *,
    gpu_hours: float | None = None,
    trajectory_ns: float | None = None,
) -> dict[str, Any]:
    flat = _flatten(traces)
    out: dict[str, Any] = {}
    names = [o.name for o in objective.observables] or list(flat)
    for name in names:
        vals = flat.get(name) or []
        out[name] = summarize(
            vals, gpu_hours=gpu_hours, trajectory_ns=trajectory_ns
        ).to_dict()
    return out


def sample(
    objective: Objective,
    box: BoxPlan,
    workdir: Path,
    *,
    budget_hours: float | None = None,
    traces: dict[str, list[float]] | dict[str, dict[str, list[float]]] | None = None,
    select: str = "random",
    execute: bool = False,
    chunk_ns: float | None = None,
    ns_cap: float | None = None,
    seed: int = 42,
    progress: Callable[[str], None] | None = None,
) -> SampleResult:
    """initialize → pilot → run_chunk → analyze → select → branch → stop|REFUSE."""
    log = progress or (lambda _m: None)
    sched = schedule(
        objective, box, budget_hours=budget_hours, traces=_as_flat(traces), select=select
    )
    policy = get_policy(select)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    walkers = _coerce_walkers(traces, sched)
    executed = False
    refused: str | None = None
    decisions: list[Decision] = []

    log("initialize")
    if execute:
        walkers, executed = _execute_loop(
            workdir,
            objective,
            sched,
            select=select,
            chunk_ns=chunk_ns,
            ns_cap=ns_cap,
            seed=seed,
            progress=log,
        )

    log("analyze")
    diags = analyze(walkers, objective, trajectory_ns=sched.total_ns)
    ctx = SelectContext(walkers=walkers, observables=list(objective.observables), rng_seed=seed)
    log(f"select={select}")
    decisions = policy(ctx)
    for d in decisions:
        log(d.reason)

    n_live = sum(1 for d in decisions if d.action in {EXTEND, KEEP, BRANCH})
    if n_live == 0 and decisions:
        n_live = 1
        decisions[0].action = KEEP
        decisions[0].reason += " (keep last walker)"

    for name, d in diags.items():
        diag = Diagnostics(
            estimate=d.get("estimate"),
            ci95=d.get("ci95"),
            ess=d.get("ess"),
            autocorrelation_time=d.get("autocorrelation_time"),
            n=int(d.get("n") or 0),
            gpu_hours=d.get("gpu_hours"),
            trajectory_ns=d.get("trajectory_ns"),
            wall_seconds=d.get("wall_seconds"),
            ci_width_per_gpu_hour=d.get("ci_width_per_gpu_hour"),
            state_coverage=d.get("state_coverage"),
            notes=list(d.get("notes") or []),
        )
        msg = _coverage_refuse(diag, n_walkers=max(len(walkers), 1))
        if msg:
            refused = msg
            raise RefuseError(msg, code="COVERAGE")
        prec = None
        for o in objective.observables:
            if o.name == name:
                prec = o.precision
        if (
            prec is not None
            and diag.ci95 is not None
            and diag.ci95 > prec
            and budget_hours is not None
            and all(x.action == STOP for x in decisions)
        ):
            raise RefuseError(
                f"Target precision cannot be reached within the supplied "
                f"{budget_hours:g} GPU-hour budget.",
                code="BUDGET",
            )

    stopped_flag = bool(decisions) and all(d.action == STOP for d in decisions)
    result = SampleResult(
        schedule=sched,
        select=select,
        loop=list(LOOP),
        decisions=[d.__dict__ for d in decisions],
        diagnostics=diags,
        traces=walkers,
        stopped=stopped_flag,
        refused=refused,
        executed=executed,
    )
    result.path = write_sample(result, workdir, extra={"seed": seed})
    existing = {}
    camp_path = workdir / "campaign.json"
    if camp_path.is_file():
        existing = json.loads(camp_path.read_text())
    camp = stamp(
        campaign_id=existing.get("campaign_id"),
        seed=seed,
        extra={"stage": "sample", "select": select, "executed": executed},
    )
    write_campaign(workdir, camp)
    return result


def _as_flat(
    traces: dict[str, list[float]] | dict[str, dict[str, list[float]]] | None,
) -> dict[str, list[float]] | None:
    if traces is None:
        return None
    if traces and isinstance(next(iter(traces.values())), dict):
        return _flatten(traces)  # type: ignore[arg-type]
    return traces  # type: ignore[return-value]


def _coerce_walkers(
    traces: dict[str, list[float]] | dict[str, dict[str, list[float]]] | None,
    sched: WalkerSchedule,
) -> dict[str, dict[str, list[float]]]:
    if traces and traces and isinstance(next(iter(traces.values())), dict):
        return {str(k): dict(v) for k, v in traces.items()}  # type: ignore[arg-type]
    if traces:
        n = max(sched.n_walkers, 1)
        names = list(traces)
        out: dict[str, dict[str, list[float]]] = {}
        for i in range(n):
            out[f"w{i:03d}"] = {
                name: list(vals[i :: n] if len(vals) >= n else vals)
                for name, vals in traces.items()  # type: ignore[union-attr]
            }
        if not out:
            out["w000"] = {n: list(traces[n]) for n in names}  # type: ignore[index]
        return out
    n = max(sched.n_walkers, 1)
    return {f"w{i:03d}": {} for i in range(n)}


def _execute_loop(
    workdir: Path,
    objective: Objective,
    sched: WalkerSchedule,
    *,
    select: str,
    chunk_ns: float | None,
    ns_cap: float | None,
    seed: int,
    progress: Callable[[str], None],
) -> tuple[dict[str, dict[str, list[float]]], bool]:
    from adaptamem.observables import atoms_from_topology, positions_nm
    from adaptamem.produce import run_chunk
    from adaptamem.schema import load_protocol
    from adaptamem.sim import langevin, pick_platform, require_openmm

    require_openmm()
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    proto = load_protocol()
    pdb_path = workdir / "eq.pdb"
    if not pdb_path.is_file():
        pdb_path = workdir / "assembled.pdb"
    xml = workdir / "system.xml"
    if not pdb_path.is_file() or not xml.is_file():
        raise RefuseError(f"sample --execute needs an assembled system in {workdir}", code="NOT_READY")
    pdb = PDBFile(str(pdb_path))
    system = XmlSerializer.deserialize(xml.read_text())
    plat, _name = pick_platform(proto.platform_preference)
    atoms = atoms_from_topology(pdb.topology)
    ref_xyz = positions_nm(pdb.positions)
    interval = max(1, int(round(proto.save_interval_ps * 1000.0 / proto.timestep_fs)))
    chunk = chunk_ns if chunk_ns is not None else min(sched.ns_per_walker, 1.0)
    chunk_steps = max(1, int(round(chunk * 1_000_000.0 / proto.timestep_fs)))
    n_run = max(sched.n_pilot, sched.n_walkers, 1)
    cap = ns_cap if ns_cap is not None else sched.total_ns
    traces: dict[str, dict[str, list[float]]] = {}
    spent = 0.0
    wdir = workdir / "walkers"
    wdir.mkdir(parents=True, exist_ok=True)
    box = pdb.topology.getPeriodicBoxVectors()
    for i in range(n_run):
        if spent >= cap:
            break
        wid = f"w{i:03d}"
        progress(f"pilot/run_chunk {wid}")
        sim = Simulation(pdb.topology, system, langevin(proto), plat)
        sim.context.setPositions(pdb.positions)
        if box is not None:
            sim.context.setPeriodicBoxVectors(*box)
        sim.context.setVelocitiesToTemperature(proto.temperature_K * unit.kelvin, seed + i)
        wd = wdir / wid
        wd.mkdir(exist_ok=True)
        traces[wid] = {o.name: [] for o in objective.observables}
        spent += run_chunk(
            sim,
            n_steps=chunk_steps,
            interval=interval,
            observables=list(objective.observables),
            atoms=atoms,
            ref_xyz=ref_xyz,
            traces=traces[wid],
            log_path=wd / "walker.log",
            t0_ns=0.0,
            timestep_fs=proto.timestep_fs,
            ckpt_path=wd / "checkpoint.chk",
        )
        (wd / "walker.states").write_text(json.dumps(traces[wid], indent=2) + "\n")
    return traces, True


def format_schedule(s: WalkerSchedule) -> str:
    lines = [
        "SAMPLE  scheduler output",
        f"physics     {s.physics}",
        f"pilot       {s.n_pilot} × {s.ns_pilot:g} ns exploration"
        if s.n_pilot
        else "pilot       none",
        f"walkers     {s.n_walkers} × {s.ns_per_walker:g} ns",
        f"total       ~{s.total_ns:g} ns",
        f"stop        {s.stop_rule}",
    ]
    for n in s.notes:
        lines.append(f"  note  {n}")
    return "\n".join(lines)


def format_sample(r: SampleResult) -> str:
    lines = [format_schedule(r.schedule), f"select      {r.select}", f"loop        {' → '.join(r.loop)}"]
    if r.executed:
        lines.append("executed    true")
    for d in r.decisions:
        lines.append(f"  {d.get('walker_id')}  {d.get('action')}  {d.get('reason')}")
    for name, diag in r.diagnostics.items():
        lines.append(
            f"  {name}  μ={diag.get('estimate')}  CI={diag.get('ci95')}  "
            f"ESS={diag.get('ess')}  U/GPU-h={diag.get('ci_width_per_gpu_hour')}  "
            f"coverage={diag.get('state_coverage')}"
        )
    if r.path:
        lines.append(f"wrote       {r.path}")
    return "\n".join(lines)


def write_schedule(s: WalkerSchedule, workdir: Path, extra: dict[str, Any] | None = None) -> Path:
    dummy = SampleResult(
        schedule=s,
        select="random",
        loop=list(LOOP),
        decisions=[],
        diagnostics={},
        traces={},
        stopped=False,
        refused=None,
    )
    return write_sample(dummy, workdir, extra=extra)


def write_sample(r: SampleResult, workdir: Path, extra: dict[str, Any] | None = None) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    s = r.schedule
    payload: dict[str, Any] = {
        "policy": "0.2",
        "created_utc": datetime.now(UTC).isoformat(),
        "loop": r.loop,
        "select": r.select,
        "n_pilot": s.n_pilot,
        "ns_pilot": s.ns_pilot,
        "n_walkers": s.n_walkers,
        "ns_per_walker": s.ns_per_walker,
        "total_ns": s.total_ns,
        "stop_rule": s.stop_rule,
        "physics": s.physics,
        "discover_cvs": s.discover_cvs,
        "notes": s.notes,
        "decisions": r.decisions,
        "diagnostics": r.diagnostics,
        "executed": r.executed,
        "stopped": r.stopped,
        **(extra or {}),
    }
    path = workdir / "sample.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
