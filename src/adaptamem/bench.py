"""ns/day of the assembled (or equilibrated) system. Same-physics number only."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from adaptamem.errors import RefuseError
from adaptamem.schema import Protocol, load_protocol
from adaptamem.sim import langevin, pick_platform, require_openmm


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
        raise RefuseError(f"no assembled system in {workdir}")

    n_steps = int(steps or proto.bench_steps)
    pdb = PDBFile(str(pdb_path))
    system = XmlSerializer.deserialize(xml_path.read_text())
    plat, plat_name = pick_platform(proto.platform_preference)
    sim = Simulation(pdb.topology, system, langevin(proto), plat)
    sim.context.setPositions(pdb.positions)
    box = pdb.topology.getPeriodicBoxVectors()
    if box is not None:
        sim.context.setPeriodicBoxVectors(*box)
    sim.minimizeEnergy(maxIterations=50)
    sim.step(min(200, n_steps))  # warmup
    t0 = time.perf_counter()
    sim.step(n_steps)
    elapsed = time.perf_counter() - t0
    ns = n_steps * proto.eq_timestep_fs * 1e-6
    ns_per_day = ns / elapsed * 86400.0 if elapsed > 0 else 0.0
    assemble_meta = {}
    meta_path = workdir / "assemble.json"
    if meta_path.is_file():
        assemble_meta = json.loads(meta_path.read_text())
    physics = str(assemble_meta.get("physics") or "same_physics")
    payload = {
        "policy": "0.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ns_per_day": ns_per_day,
        "steps": n_steps,
        "timestep_fs": proto.eq_timestep_fs,
        "platform": plat_name,
        "n_atoms": pdb.topology.getNumAtoms(),
        "seconds": elapsed,
        "physics": physics,
        "note": "legal ns/day only inside same_physics",
    }
    out = workdir / "bench.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return BenchResult(
        ns_per_day=ns_per_day,
        steps=n_steps,
        timestep_fs=proto.eq_timestep_fs,
        platform=plat_name,
        n_atoms=pdb.topology.getNumAtoms(),
        seconds=elapsed,
        physics=physics,
        path=out,
    )


def format_bench(r: BenchResult) -> str:
    lines = [
        f"BENCH  {r.ns_per_day:.1f} ns/day  {r.n_atoms} atoms  {r.platform}",
        f"{r.steps} steps × {r.timestep_fs:g} fs in {r.seconds:.1f} s",
        f"physics  {r.physics}",
    ]
    if r.physics != "same_physics":
        lines.append("NOTE  approximation — do not quote this ns/day against a full-AA baseline")
    return "\n".join(lines)
