"""Production MD. Streaming CVs are canonical; XTC/checkpoints are compatibility."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.campaign import stamp, write_campaign
from adaptamem.diagnostics import summarize
from adaptamem.errors import RefuseError
from adaptamem.objective import Observable, parse_objective
from adaptamem.observables import atoms_from_topology, evaluate, positions_nm
from adaptamem.schema import Protocol, load_protocol
from adaptamem.sim import hardware_label, langevin, pick_platform, require_openmm

Progress = Callable[[str], None]


@dataclass
class ProduceResult:
    workdir: Path
    ns: float
    steps: int
    platform: str
    traces: dict[str, list[float]]
    diagnostics: dict[str, Any]
    path: Path
    notes: list[str] = field(default_factory=list)
    gpu_hours: float = 0.0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _observables_from_workdir(workdir: Path) -> list[Observable]:
    meta = _load_json(workdir / "assemble.json")
    raw = meta.get("objective") or {}
    obj = parse_objective(raw) if raw else parse_objective({})
    return list(obj.observables)


def ns_from_budget(gpu_hours: float, ns_per_day: float) -> float:
    return float(ns_per_day) * (float(gpu_hours) / 24.0)


def resolve_ns(
    workdir: Path,
    *,
    ns: float | None,
    protocol: Protocol,
    budget_hours: float | None,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    bench = _load_json(workdir / "bench.json")
    perf = bench.get("performance") or bench
    ns_day = float(perf.get("ns_per_day") or 50.0)
    hours = budget_hours
    if hours is None:
        hours = protocol.compute.max_gpu_hours
    budget_ns = ns_from_budget(hours, ns_day) if hours else None
    if ns is not None and budget_ns is not None:
        chosen = min(float(ns), budget_ns)
        if float(ns) > budget_ns:
            notes.append(f"--ns {ns:g} capped by compute budget → {chosen:g} ns")
        return chosen, notes
    if ns is not None:
        return float(ns), notes
    if budget_ns is not None:
        notes.append(f"length from compute budget {hours:g} GPU-h @ {ns_day:.1f} ns/day")
        return budget_ns, notes
    return 1.0, ["no budget and no --ns; default 1 ns"]


def produce(
    workdir: Path,
    *,
    ns: float | None = None,
    protocol: Protocol | None = None,
    budget_hours: float | None = None,
    force: bool = False,
    progress: Progress | None = None,
    seed: int | None = None,
) -> ProduceResult:
    require_openmm()
    proto = protocol or load_protocol()
    log = progress or (lambda _m: None)
    workdir = Path(workdir)
    eq_meta = _load_json(workdir / "eq.json")
    qc = (eq_meta.get("qc") or {}) if eq_meta else {}
    if eq_meta and qc.get("ok") is False and not force:
        raise RefuseError(
            "Membrane has not reached statistical stability. "
            "Production sampling would be invalid.",
            code="MEMBRANE_QC",
        )
    pdb_path = workdir / "eq.pdb"
    if not pdb_path.is_file():
        pdb_path = workdir / "assembled.pdb"
    xml_path = workdir / "system.xml"
    if not pdb_path.is_file() or not xml_path.is_file():
        raise RefuseError(f"no assembled system in {workdir}", code="NOT_READY")

    n_ns, notes = resolve_ns(workdir, ns=ns, protocol=proto, budget_hours=budget_hours)
    steps = max(1, int(round(n_ns * 1_000_000.0 / proto.timestep_fs)))
    log(f"produce {n_ns:g} ns ({steps} steps)")

    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    pdb = PDBFile(str(pdb_path))
    system = XmlSerializer.deserialize(xml_path.read_text())
    plat, plat_name = pick_platform(proto.platform_preference)
    gpu = hardware_label(plat, plat_name)
    sim = Simulation(pdb.topology, system, langevin(proto), plat)
    sim.context.setPositions(pdb.positions)
    box = pdb.topology.getPeriodicBoxVectors()
    if box is not None:
        sim.context.setPeriodicBoxVectors(*box)
    camp = _load_json(workdir / "campaign.json")
    rng = int(seed if seed is not None else camp.get("random_seed") or 42)
    sim.context.setVelocitiesToTemperature(proto.temperature_K * unit.kelvin, rng)

    interval = max(1, int(round(proto.save_interval_ps * 1000.0 / proto.timestep_fs)))
    ckpt_every = max(
        interval,
        int(round(proto.checkpoint_interval_ns * 1_000_000.0 / proto.timestep_fs)),
    )
    obs = _observables_from_workdir(workdir)
    atoms = atoms_from_topology(pdb.topology)
    ref_xyz = positions_nm(pdb.positions)
    traces: dict[str, list[float]] = {o.name: [] for o in obs}
    times: list[float] = []

    xtc_path = workdir / "traj.xtc"
    _attach_traj(sim, xtc_path, interval)
    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "walker.log"
    states_path = workdir / "walker.states"

    t0 = time.perf_counter()
    done = 0
    ckpt_i = 0
    keep = proto.checkpoint_keep_last
    while done < steps:
        n = min(interval, steps - done)
        sim.step(n)
        done += n
        t_ns = done * proto.timestep_fs * 1e-6
        times.append(t_ns)
        xyz = positions_nm(sim.context.getState(getPositions=True).getPositions())
        row: dict[str, Any] = {"t_ns": t_ns}
        for o in obs:
            val = evaluate(o, atoms, xyz, reference=ref_xyz)
            traces[o.name].append(val)
            row[o.name] = val
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if done % ckpt_every < interval or done == steps:
            ckpt = ckpt_dir / f"step_{done}.chk"
            sim.saveCheckpoint(str(ckpt))
            ckpt_i += 1
            old = sorted(ckpt_dir.glob("step_*.chk"), key=lambda p: p.stat().st_mtime)
            for stale in old[:-keep]:
                stale.unlink(missing_ok=True)

    elapsed = time.perf_counter() - t0
    states_path.write_text(json.dumps({"t_ns": times, "observables": traces}, indent=2) + "\n")
    gpu_hours = elapsed / 3600.0
    diags = {
        name: summarize(
            vals, gpu_hours=gpu_hours, trajectory_ns=n_ns, wall_seconds=elapsed
        ).to_dict()
        for name, vals in traces.items()
    }
    camp_stamp = stamp(
        campaign_id=camp.get("campaign_id"),
        protocol=proto,
        gpu=gpu,
        seed=rng,
        extra={"stage": "produce", "ns": n_ns, "steps": steps},
    )
    write_campaign(workdir, camp_stamp)
    payload = {
        "policy": "0.2",
        "created_utc": datetime.now(UTC).isoformat(),
        "ns": n_ns,
        "steps": steps,
        "platform": plat_name,
        "gpu": gpu,
        "seconds": elapsed,
        "gpu_hours": gpu_hours,
        "physics": "same_physics",
        "xtc": str(xtc_path.name),
        "log": str(log_path.name),
        "states": str(states_path.name),
        "diagnostics": diags,
        "notes": notes,
        "campaign": camp_stamp,
    }
    out = workdir / "produce.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return ProduceResult(
        workdir=workdir,
        ns=n_ns,
        steps=steps,
        platform=plat_name,
        traces=traces,
        diagnostics=diags,
        path=out,
        notes=notes,
        gpu_hours=gpu_hours,
    )


def run_chunk(
    sim: Any,
    *,
    n_steps: int,
    interval: int,
    observables: list[Observable],
    atoms: Any,
    ref_xyz: list[tuple[float, float, float]],
    traces: dict[str, list[float]],
    log_path: Path,
    t0_ns: float,
    timestep_fs: float,
    ckpt_path: Path | None = None,
) -> float:
    """Advance an existing Simulation; append streaming CVs. Returns ns simulated."""
    done = 0
    while done < n_steps:
        n = min(interval, n_steps - done)
        sim.step(n)
        done += n
        t_ns = t0_ns + done * timestep_fs * 1e-6
        xyz = positions_nm(sim.context.getState(getPositions=True).getPositions())
        row: dict[str, Any] = {"t_ns": t_ns}
        for o in observables:
            val = evaluate(o, atoms, xyz, reference=ref_xyz)
            traces.setdefault(o.name, []).append(val)
            row[o.name] = val
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        sim.saveCheckpoint(str(ckpt_path))
    return n_steps * timestep_fs * 1e-6


def _attach_traj(sim: Any, path: Path, interval: int) -> None:
    try:
        from openmm.app import XTCReporter

        sim.reporters.append(XTCReporter(str(path), interval))
    except (ImportError, AttributeError):
        from openmm.app import DCDReporter

        sim.reporters.append(DCDReporter(str(path.with_suffix(".dcd")), interval))


def format_produce(r: ProduceResult) -> str:
    lines = [
        f"PRODUCE  {r.ns:g} ns  {r.steps} steps  {r.platform}",
        f"wrote    {r.path}",
    ]
    for name, d in r.diagnostics.items():
        est = d.get("estimate")
        ci = d.get("ci95")
        ess = d.get("ess")
        lines.append(
            f"  {name}  estimate={est}  CI={ci}  ESS={ess}  "
            f"U/GPU-h={d.get('ci_width_per_gpu_hour')}"
        )
    for n in r.notes:
        lines.append(f"  note  {n}")
    return "\n".join(lines)
