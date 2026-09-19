"""How fast this physical system runs. Not a scientific-efficiency number."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.campaign import stamp, write_campaign
from adaptamem.errors import RefuseError
from adaptamem.schema import Protocol, load_protocol
from adaptamem.sim import (
    LIPID_RESIDUES,
    WATER_RESIDUES,
    count_residues,
    hardware_label,
    langevin,
    pick_platform,
    require_openmm,
)

LADDER_ATOMS = (20_000, 50_000, 100_000, 200_000, 400_000, 600_000)


@dataclass
class BenchResult:
    ns_per_day: float
    steps: int
    timestep_fs: float
    platform: str
    n_atoms: int
    seconds: float
    physics: str
    path: Path
    payload: dict[str, Any] = field(default_factory=dict)


def ladder_spec() -> list[dict[str, Any]]:
    return [{"target_atoms": n, "label": f"{n // 1000}k"} for n in LADDER_ATOMS]


def bench(
    workdir: Path,
    *,
    protocol: Protocol | None = None,
    steps: int | None = None,
) -> BenchResult:
    require_openmm()
    from openmm import XmlSerializer
    from openmm.app import PDBFile, Simulation

    proto = protocol or load_protocol()
    workdir = Path(workdir)
    pdb_path = workdir / "eq.pdb"
    if not pdb_path.is_file():
        pdb_path = workdir / "assembled.pdb"
    xml_path = workdir / "system.xml"
    if not pdb_path.is_file() or not xml_path.is_file():
        raise RefuseError(f"no assembled system in {workdir}", code="NOT_READY")

    n_steps = int(steps or proto.bench_steps)
    pdb = PDBFile(str(pdb_path))
    system = XmlSerializer.deserialize(xml_path.read_text())
    plat, plat_name = pick_platform(proto.platform_preference)
    gpu = hardware_label(plat, plat_name)
    sim = Simulation(pdb.topology, system, langevin(proto), plat)
    sim.context.setPositions(pdb.positions)
    box = pdb.topology.getPeriodicBoxVectors()
    if box is not None:
        sim.context.setPeriodicBoxVectors(*box)
    # Stabilize on CPU, then time the preferred platform (CUDA on a 4090).
    try:
        cpu, _ = pick_platform(["CPU"])
        sim_c = Simulation(pdb.topology, system, langevin(proto, timestep_fs=1.0), cpu)
        sim_c.context.setPositions(pdb.positions)
        if box is not None:
            sim_c.context.setPeriodicBoxVectors(*box)
        sim_c.minimizeEnergy(maxIterations=400)
        pos = sim_c.context.getState(getPositions=True).getPositions()
        sim.context.setPositions(pos)
    except Exception:  # noqa: BLE001
        pos = pdb.positions
    from openmm import unit as _unit

    try:
        sim.context.setVelocitiesToTemperature(proto.temperature_K * _unit.kelvin)
        sim.step(min(200, n_steps))
    except Exception:  # noqa: BLE001
        plat, plat_name = pick_platform(["CPU"])
        gpu = hardware_label(plat, plat_name)
        sim = Simulation(pdb.topology, system, langevin(proto), plat)
        sim.context.setPositions(pos)
        if box is not None:
            sim.context.setPeriodicBoxVectors(*box)
        sim.minimizeEnergy(maxIterations=200)
        sim.context.setVelocitiesToTemperature(proto.temperature_K * _unit.kelvin)
        sim.step(min(200, n_steps))
    try:
        gpu = sim.context.getPlatform().getPropertyValue(sim.context, "DeviceName")
    except Exception:  # noqa: BLE001
        pass
    t0 = time.perf_counter()
    sim.step(n_steps)
    elapsed = time.perf_counter() - t0
    ns = n_steps * proto.eq_timestep_fs * 1e-6
    ns_per_day = ns / elapsed * 86400.0 if elapsed > 0 else 0.0
    steps_per_second = n_steps / elapsed if elapsed > 0 else 0.0
    assemble_meta: dict[str, Any] = {}
    meta_path = workdir / "assemble.json"
    if meta_path.is_file():
        assemble_meta = json.loads(meta_path.read_text())
    physics = str(assemble_meta.get("physics") or "same_physics")
    n_atoms = pdb.topology.getNumAtoms()
    n_lipid = count_residues(pdb.topology, LIPID_RESIDUES)
    n_water = count_residues(pdb.topology, WATER_RESIDUES)
    payload = {
        "policy": "0.2",
        "created_utc": datetime.now(UTC).isoformat(),
        "system": {
            "atoms": n_atoms,
            "membrane_lipids": n_lipid,
            "water_atoms": n_water * 3,
            "water_residues": n_water,
            "name": assemble_meta.get("system"),
        },
        "hardware": {
            "gpu": gpu,
            "platform": plat_name,
        },
        "physics": {
            "forcefield": proto.force_field,
            "water": proto.water,
            "timestep_fs": proto.eq_timestep_fs,
            "cutoff_nm": proto.nonbonded_cutoff_nm,
            "pme": proto.pme,
            "precision": proto.precision,
            "hydrogen_mass_amu": proto.hydrogen_mass_amu,
            "class": physics,
        },
        "performance": {
            "ns_per_day": ns_per_day,
            "steps_per_second": steps_per_second,
            "gpu_utilization": None,
            "seconds": elapsed,
            "steps": n_steps,
        },
        "ns_per_day": ns_per_day,
        "n_atoms": n_atoms,
        "note": "bench is throughput only; sample reports scientific efficiency",
    }
    camp = stamp(
        campaign_id=(assemble_meta.get("campaign") or {}).get("campaign_id"),
        protocol=proto,
        gpu=gpu,
        extra={"stage": "bench", "n_atoms": n_atoms},
    )
    payload["campaign"] = camp
    out = workdir / "bench.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    write_campaign(workdir, camp)
    return BenchResult(
        ns_per_day=ns_per_day,
        steps=n_steps,
        timestep_fs=proto.eq_timestep_fs,
        platform=plat_name,
        n_atoms=n_atoms,
        seconds=elapsed,
        physics=physics,
        path=out,
        payload=payload,
    )


def write_ladder(rows: list[dict[str, Any]], path: Path) -> Path:
    payload = {
        "policy": "0.2",
        "created_utc": datetime.now(UTC).isoformat(),
        "metric": "ns_per_day(N)",
        "targets": ladder_spec(),
        "points": rows,
        "note": "same_physics only; do not mix with sample efficiency",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def format_bench(r: BenchResult) -> str:
    gpu = (r.payload.get("hardware") or {}).get("gpu", r.platform)
    phys = r.payload.get("physics") or {}
    perf = r.payload.get("performance") or {}
    sysd = r.payload.get("system") or {}
    lines = [
        f"BENCH  {r.ns_per_day:.1f} ns/day  {r.n_atoms} atoms  {gpu}",
        f"steps/s  {perf.get('steps_per_second', 0):.0f}  "
        f"{r.steps} steps × {r.timestep_fs:g} fs in {r.seconds:.1f} s",
        f"lipids   {sysd.get('membrane_lipids')}  water_atoms {sysd.get('water_atoms')}",
        f"physics  {phys.get('forcefield')}  dt={phys.get('timestep_fs')} fs  "
        f"cutoff={phys.get('cutoff_nm')} nm  PME={phys.get('pme')}  "
        f"precision={phys.get('precision')}",
        f"class    {r.physics}",
    ]
    if r.physics != "same_physics":
        lines.append("NOTE  approximation — do not quote this ns/day against a full-AA baseline")
    return "\n".join(lines)
