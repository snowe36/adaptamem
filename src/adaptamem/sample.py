"""Iterative sampling. select is a plug; walker count is an output."""

from __future__ import annotations

import json
import shutil
import time
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
    gpu_hours: float | None = None
    spent_ns: float = 0.0


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
    run_chunk_fn: Callable[..., float] | None = None,
    execute_hours: float | None = None,
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
    spent_ns = 0.0
    gpu_hours = None
    if execute:
        walkers, executed, gpu_hours, spent_ns = _execute_loop(
            workdir,
            objective,
            sched,
            select=select,
            chunk_ns=chunk_ns,
            ns_cap=ns_cap,
            seed=seed,
            progress=log,
            budget_hours=execute_hours if execute_hours is not None else budget_hours,
            run_chunk_fn=run_chunk_fn,
        )

    log("analyze")
    diags = analyze(
        walkers,
        objective,
        gpu_hours=gpu_hours,
        trajectory_ns=spent_ns or sched.total_ns,
    )
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

    refuse_code = "BUDGET"
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
            refuse_code = "COVERAGE"
        prec = None
        for o in objective.observables:
            if o.name == name:
                prec = o.precision
        budget_spent = budget_hours is not None or (executed and ns_cap is not None)
        if (
            refused is None
            and prec is not None
            and diag.ci95 is not None
            and diag.ci95 > prec
            and budget_spent
            and (all(x.action == STOP for x in decisions) or executed)
        ):
            hours = budget_hours if budget_hours is not None else gpu_hours
            label = f"{hours:g} GPU-hour" if hours is not None else "ns"
            refused = (
                f"Target precision cannot be reached within the supplied {label} budget."
            )
            refuse_code = "BUDGET"

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
        gpu_hours=gpu_hours,
        spent_ns=spent_ns,
    )
    result.path = write_sample(result, workdir, extra={"seed": seed})
    if refused:
        raise RefuseError(refused, code=refuse_code)
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
    budget_hours: float | None = None,
    run_chunk_fn: Callable[..., float] | None = None,
) -> tuple[dict[str, dict[str, list[float]]], bool, float, float]:
    """EXTEND / BRANCH / STOP / KEEP from checkpoints until GPU-hours or ns_cap."""
    injected = run_chunk_fn is not None
    if not injected:
        from adaptamem.observables import atoms_from_topology, positions_nm
        from adaptamem.produce import run_chunk as _run_chunk
        from adaptamem.schema import load_protocol
        from adaptamem.sim import pick_platform, require_openmm

        require_openmm()
        from openmm import XmlSerializer, unit
        from openmm.app import PDBFile

        run_chunk_fn = _run_chunk
        proto = load_protocol()
        pdb_path = workdir / "eq.pdb"
        if not pdb_path.is_file():
            pdb_path = workdir / "assembled.pdb"
        xml = workdir / "system.xml"
        if not pdb_path.is_file() or not xml.is_file():
            raise RefuseError(
                f"sample --execute needs an assembled system in {workdir}",
                code="NOT_READY",
            )
        pdb = PDBFile(str(pdb_path))
        system = XmlSerializer.deserialize(xml.read_text())
        plat, _name = pick_platform(proto.platform_preference)
        atoms = atoms_from_topology(pdb.topology)
        ref_xyz = positions_nm(pdb.positions)
        interval = max(1, int(round(proto.save_interval_ps * 1000.0 / proto.timestep_fs)))
        chunk = chunk_ns if chunk_ns is not None else min(sched.ns_per_walker, 1.0)
        chunk_steps = max(1, int(round(chunk * 1_000_000.0 / proto.timestep_fs)))
        timestep_fs = proto.timestep_fs
        box = pdb.topology.getPeriodicBoxVectors()
    else:
        proto = pdb = system = plat = atoms = box = None
        unit = None  # type: ignore[assignment]
        ref_xyz = []
        interval = 1
        chunk_steps = 1
        timestep_fs = 4.0

    policy = get_policy(select)
    n_start = max(sched.n_pilot, sched.n_walkers, 1)
    cap = ns_cap if ns_cap is not None else sched.total_ns
    wdir = workdir / "walkers"
    wdir.mkdir(parents=True, exist_ok=True)
    obs = list(objective.observables)

    def _new_sim(rng: int, ckpt: Path | None) -> Any:
        if injected:
            return None
        from openmm.app import Simulation

        from adaptamem.sim import langevin

        sim = Simulation(pdb.topology, system, langevin(proto), plat)
        if ckpt is not None and ckpt.is_file():
            sim.loadCheckpoint(str(ckpt))
        else:
            sim.context.setPositions(pdb.positions)
            if box is not None:
                sim.context.setPeriodicBoxVectors(*box)
        sim.context.setVelocitiesToTemperature(proto.temperature_K * unit.kelvin, rng)
        return sim

    def _record(wid: str, named: dict[str, list[float]]) -> None:
        (wdir / wid / "walker.states").write_text(json.dumps(named, indent=2) + "\n")

    walkers: list[dict[str, Any]] = []
    for i in range(n_start):
        wid = f"w{i:03d}"
        wd = wdir / wid
        wd.mkdir(exist_ok=True)
        ckpt = wd / "checkpoint.chk"
        walkers.append(
            {
                "id": wid,
                "dir": wd,
                "ckpt": ckpt,
                "traces": {o.name: [] for o in obs},
                "t_ns": 0.0,
                "step": True,
                "sim": _new_sim(seed + i, None),
            }
        )
    next_i = n_start
    spent = 0.0
    t0 = time.perf_counter()
    max_walkers = 32

    while True:
        hours_used = (time.perf_counter() - t0) / 3600.0
        if spent >= cap:
            break
        if budget_hours is not None and hours_used >= budget_hours:
            break
        live = [w for w in walkers if w["step"]]
        if not live:
            break
        for w in live:
            hours_used = (time.perf_counter() - t0) / 3600.0
            if spent >= cap or (budget_hours is not None and hours_used >= budget_hours):
                break
            progress(f"run_chunk {w['id']}")
            ns = run_chunk_fn(
                w["sim"],
                n_steps=chunk_steps,
                interval=interval,
                observables=obs,
                atoms=atoms,
                ref_xyz=ref_xyz,
                traces=w["traces"],
                log_path=w["dir"] / "walker.log",
                t0_ns=w["t_ns"],
                timestep_fs=timestep_fs,
                ckpt_path=w["ckpt"],
            )
            w["t_ns"] += float(ns)
            spent += float(ns)
            _record(w["id"], w["traces"])

        traces_now = {w["id"]: w["traces"] for w in walkers}
        ctx = SelectContext(walkers=traces_now, observables=obs, rng_seed=seed)
        progress("analyze")
        progress(f"select={select}")
        decisions = policy(ctx)
        by_id = {w["id"]: w for w in walkers}
        spawned: list[dict[str, Any]] = []
        for d in decisions:
            progress(d.reason)
            w = by_id[d.walker_id]
            if d.action == STOP:
                w["step"] = False
            elif d.action == KEEP:
                w["step"] = False
            elif d.action == EXTEND:
                w["step"] = True
            elif d.action == BRANCH:
                w["step"] = True
                if next_i >= max_walkers:
                    continue
                nid = f"w{next_i:03d}"
                nd = wdir / nid
                nd.mkdir(exist_ok=True)
                nckpt = nd / "checkpoint.chk"
                if w["ckpt"].is_file():
                    shutil.copy2(w["ckpt"], nckpt)
                spawned.append(
                    {
                        "id": nid,
                        "dir": nd,
                        "ckpt": nckpt,
                        "traces": {o.name: [] for o in obs},
                        "t_ns": 0.0,
                        "step": True,
                        "sim": _new_sim(seed + next_i, nckpt if nckpt.is_file() else w["ckpt"]),
                    }
                )
                next_i += 1
        walkers.extend(spawned)

    hours_used = (time.perf_counter() - t0) / 3600.0
    return {w["id"]: w["traces"] for w in walkers}, True, hours_used, spent


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
        if r.spent_ns:
            lines.append(f"spent_ns    {r.spent_ns:g}")
        if r.gpu_hours is not None:
            lines.append(f"gpu_hours   {r.gpu_hours:g}")
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
        "gpu_hours": r.gpu_hours,
        "spent_ns": r.spent_ns,
        "values": _flatten(r.traces) if r.traces else {},
        **(extra or {}),
    }
    path = workdir / "sample.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
