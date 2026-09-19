#!/usr/bin/env python3
"""CHARMM36 membrane throughput. CPU-minimize, then time CUDA (or OpenCL/CPU).

Reusable GPU entrypoint — no adaptamem install required:

    pip install 'openmm[cuda12]' pdbfixer
    python scripts/gpu_job.py --out /workspace/out --steps 4000

Caches assembled.xml in --out so a second run is just the timed loop.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

LEU = (
    ("N", 0.0, 0.0, 0.0),
    ("CA", 1.46, 0.0, 0.0),
    ("C", 2.0, 1.25, 0.4),
    ("O", 1.4, 2.3, 0.4),
    ("CB", 2.0, -1.25, -0.4),
    ("CG", 1.5, -2.5, 0.2),
    ("CD1", 2.4, -3.6, -0.5),
    ("CD2", 0.05, -2.7, 0.3),
)


def helix_pdb(path: Path, n: int = 20) -> Path:
    """TM poly-Leu: 100 deg/res, 1.5 Å rise, CA ~2.3 Å off z, centered at z=0."""
    lines = ["HEADER    GPUJOB"]
    serial = 1
    rise = 1.5
    twist = math.radians(100.0)
    z0 = -0.5 * (n - 1) * rise
    shift = 2.3 - 1.46
    for i in range(n):
        ang = i * twist
        ca, sa = math.cos(ang), math.sin(ang)
        z = z0 + i * rise
        for name, dx, dy, dz in LEU:
            x0, y0 = dx + shift, dy
            x = x0 * ca - y0 * sa
            y = x0 * sa + y0 * ca
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} LEU A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z + dz:8.3f}  1.00  0.00           {name[0]}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _has_nan(positions) -> bool:
    from openmm import unit

    for vec in positions:
        xyz = vec.value_in_unit(unit.nanometer) if hasattr(vec, "value_in_unit") else vec
        if any(math.isnan(float(c)) for c in xyz):
            return True
    return False


def _pad_schedule(min_pad: float) -> list[float]:
    pads = [float(min_pad)]
    for p in (2.0, 2.5, 3.0):
        if p > min_pad:
            pads.append(p)
    return pads


def pick_platform(order: list[str]):
    from openmm import Platform

    last = None
    for name in order:
        try:
            plat = Platform.getPlatformByName(name)
            if name == "CUDA":
                try:
                    plat.setPropertyDefaultValue("Precision", "mixed")
                except Exception:
                    pass
            return plat, name
        except Exception as exc:
            last = exc
    raise RuntimeError(f"no OpenMM platform from {order}: {last}")


def device_name(sim, plat_name: str) -> str:
    try:
        return str(sim.context.getPlatform().getPropertyValue(sim.context, "DeviceName"))
    except Exception:
        return plat_name


def load_protein(pdb_path: Path, ff):
    from openmm.app import Modeller, PDBFile

    try:
        from pdbfixer import PDBFixer
    except ImportError:
        PDBFixer = None
    if PDBFixer is not None:
        fixer = PDBFixer(filename=str(pdb_path))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        modeller = Modeller(fixer.topology, fixer.positions)
    else:
        pdb = PDBFile(str(pdb_path))
        modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    return modeller


def assemble(out: Path, *, n_leu: int, pad_nm: float) -> None:
    from openmm import MonteCarloMembraneBarostat, XmlSerializer, unit
    from openmm.app import PME, ForceField, HBonds, Modeller, PDBFile

    pdb_path = helix_pdb(out / "helix.pdb", n=n_leu)
    ff = ForceField("charmm36.xml", "charmm36/water.xml")
    protein = load_protein(pdb_path, ff)
    cpu, _ = pick_platform(["CPU"])
    last: Exception | None = None
    modeller = None
    for pad in _pad_schedule(pad_nm):
        trial = Modeller(protein.topology, protein.positions)
        kwargs = dict(
            lipidType="POPC",
            membraneCenterZ=0.0 * unit.nanometer,
            minimumPadding=float(pad) * unit.nanometer,
            positiveIon="K+",
            negativeIon="Cl-",
            ionicStrength=0.15 * unit.molar,
            neutralize=True,
        )
        try:
            print(f"addMembrane pad={pad} nm", flush=True)
            try:
                trial.addMembrane(ff, platform=cpu, **kwargs)
            except TypeError:
                trial.addMembrane(ff, **kwargs)
            if _has_nan(trial.positions):
                raise RuntimeError("addMembrane produced NaN")
            modeller = trial
            break
        except Exception as exc:
            last = exc
            print(f"addMembrane failed pad={pad}: {exc}", flush=True)
    if modeller is None:
        raise RuntimeError(f"addMembrane failed: {last}")
    hmass = 4.0 * unit.amu
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=hmass,
    )
    system.addForce(
        MonteCarloMembraneBarostat(
            1.0 * unit.bar,
            0.0 * unit.bar * unit.nanometer,
            310.0 * unit.kelvin,
            MonteCarloMembraneBarostat.XYIsotropic,
            MonteCarloMembraneBarostat.ZFree,
            15,
        )
    )
    with (out / "assembled.pdb").open("w") as fh:
        PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)
    (out / "system.xml").write_text(XmlSerializer.serialize(system))


def minimize_cpu(topology, system, positions, box):
    from openmm import LangevinMiddleIntegrator, unit
    from openmm.app import Simulation

    cpu, _ = pick_platform(["CPU"])
    integ = LangevinMiddleIntegrator(310.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtoseconds)
    sim = Simulation(topology, system, integ, cpu)
    sim.context.setPositions(positions)
    if box is not None:
        sim.context.setPeriodicBoxVectors(*box)
    try:
        sim.minimizeEnergy(maxIterations=400)
        pos = sim.context.getState(getPositions=True).getPositions()
        if _has_nan(pos):
            raise RuntimeError("minimize produced NaN")
        return pos
    except Exception as exc:
        print(f"CPU minimize failed: {exc}", flush=True)
        if _has_nan(positions):
            raise
        return positions


def timed_steps(topology, system, positions, box, n_steps: int):
    from openmm import LangevinMiddleIntegrator, unit
    from openmm.app import Simulation

    last: Exception | None = None
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            plat, pname = pick_platform([name])
            integ = LangevinMiddleIntegrator(
                310.0 * unit.kelvin, 1.0 / unit.picosecond, 4.0 * unit.femtoseconds
            )
            sim = Simulation(topology, system, integ, plat)
            sim.context.setPositions(positions)
            if box is not None:
                sim.context.setPeriodicBoxVectors(*box)
            sim.context.setVelocitiesToTemperature(310.0 * unit.kelvin)
            warm = min(200, n_steps)
            sim.step(warm)
            t0 = time.perf_counter()
            sim.step(n_steps)
            elapsed = time.perf_counter() - t0
            gpu = device_name(sim, pname)
            return pname, gpu, elapsed
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"{name} failed: {exc}", flush=True)
    raise RuntimeError(f"no platform could step: {last}") from last


def count_res(topology, names: set[str]) -> int:
    return sum(1 for r in topology.residues() if r.name in names)


def run(out: Path, *, steps: int, pad_nm: float, n_leu: int, rebuild: bool) -> dict:
    from openmm import XmlSerializer
    from openmm.app import PDBFile

    out.mkdir(parents=True, exist_ok=True)
    assembled = out / "assembled.pdb"
    xml = out / "system.xml"
    reuse = (
        not rebuild
        and assembled.is_file()
        and xml.is_file()
    )
    if reuse:
        pdb = PDBFile(str(assembled))
        if _has_nan(pdb.positions):
            print(f"cache NaN {assembled}; rebuild", flush=True)
            reuse = False
    if not reuse:
        print(f"assemble n_leu={n_leu} pad={pad_nm} nm", flush=True)
        assemble(out, n_leu=n_leu, pad_nm=pad_nm)
        pdb = PDBFile(str(assembled))
    else:
        print(f"reuse {assembled}", flush=True)
    system = XmlSerializer.deserialize(xml.read_text())
    box = pdb.topology.getPeriodicBoxVectors()
    print("CPU minimize", flush=True)
    pos = minimize_cpu(pdb.topology, system, pdb.positions, box)
    print(f"time {steps} steps", flush=True)
    plat_name, gpu, elapsed = timed_steps(pdb.topology, system, pos, box, steps)
    ns = steps * 4.0e-6
    ns_day = ns / elapsed * 86400.0 if elapsed > 0 else 0.0
    n_atoms = pdb.topology.getNumAtoms()
    payload = {
        "policy": "0.2",
        "created_utc": datetime.now(UTC).isoformat(),
        "system": {
            "atoms": n_atoms,
            "membrane_lipids": count_res(pdb.topology, {"POPC", "POP", "POPE", "POPG"}),
            "water_residues": count_res(pdb.topology, {"HOH", "WAT", "TIP3", "TIP"}),
            "n_leu": n_leu,
            "pad_nm": pad_nm,
        },
        "hardware": {"gpu": gpu, "platform": plat_name},
        "physics": {
            "forcefield": "CHARMM36",
            "timestep_fs": 4.0,
            "cutoff_nm": 1.0,
            "pme": True,
            "precision": "mixed",
            "hydrogen_mass_amu": 4.0,
            "class": "same_physics",
        },
        "performance": {
            "ns_per_day": ns_day,
            "steps_per_second": steps / elapsed if elapsed else 0.0,
            "seconds": elapsed,
            "steps": steps,
        },
        "note": "CPU-minimize, timed on listed platform. Throughput only.",
    }
    (out / "bench.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"BENCH  {ns_day:.1f} ns/day  {n_atoms} atoms  {gpu}  {plat_name}  "
        f"{elapsed:.2f}s",
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OpenMM CHARMM36 GPU throughput")
    p.add_argument("--out", type=Path, default=Path("runs/gpu"))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--pad", type=float, default=2.0)
    p.add_argument("--n-leu", type=int, default=20)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument(
        "--pads",
        default="",
        help="Optional ladder, e.g. '1.2,2.5,4.0' — writes ladder.json",
    )
    args = p.parse_args(argv)
    pads = [float(x) for x in args.pads.split(",") if x.strip()] or [args.pad]
    points = []
    for pad in pads:
        dest = args.out if len(pads) == 1 else args.out / f"pad_{pad:g}"
        payload = run(dest, steps=args.steps, pad_nm=pad, n_leu=args.n_leu, rebuild=args.rebuild)
        points.append(
            {
                "n_atoms": payload["system"]["atoms"],
                "ns_per_day": payload["performance"]["ns_per_day"],
                "gpu": payload["hardware"]["gpu"],
                "platform": payload["hardware"]["platform"],
                "pad_nm": pad,
            }
        )
    if len(points) > 1:
        ladder = {
            "metric": "ns_per_day(N)",
            "points": points,
            "created_utc": datetime.now(UTC).isoformat(),
        }
        path = args.out / "ladder.json"
        args.out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ladder, indent=2) + "\n")
        print(f"ladder {path}", flush=True)
    print("GPU_JOB_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
