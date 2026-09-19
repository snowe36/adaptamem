"""Minimize and short 4 fs equilibration with a membrane QC scorecard."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.schema import Protocol, load_protocol
from adaptamem.sim import (
    QC,
    kinetic_temperature,
    langevin,
    pick_platform,
    require_openmm,
    scorecard,
    strip_barostat,
)

Progress = Callable[[str], None]


@dataclass
class EqResult:
    workdir: Path
    pdb: Path
    qc: QC
    stages: list[str]
    platform: str


def equilibrate(
    workdir: Path,
    *,
    protocol: Protocol | None = None,
    short: bool = False,
    progress: Progress | None = None,
) -> EqResult:
    require_openmm()
    from openmm import XmlSerializer, unit
    from openmm.app import PDBFile, Simulation

    proto = protocol or load_protocol()
    log = progress or (lambda _m: None)
    workdir = Path(workdir)
    assembled = workdir / "assembled.pdb"
    system_xml = workdir / "system.xml"
    if not assembled.is_file() or not system_xml.is_file():
        raise RefuseError(f"no assembled system in {workdir}; run adaptamem assemble first")

    pdb = PDBFile(str(assembled))
    system = XmlSerializer.deserialize(system_xml.read_text())
    plat, plat_name = pick_platform(proto.platform_preference)
    n_atoms = pdb.topology.getNumAtoms()

    k_rest = 1000.0  # kJ/mol/nm² on CA
    stages: list[str] = []
    platforms = _platform_ladder(proto.platform_preference)
    box = pdb.topology.getPeriodicBoxVectors()

    def _sim(
        sys_obj: Any,
        pobj: Any,
        timestep_fs: float,
        positions: Any,
        *,
        precision: str | None = None,
    ) -> Any:
        integ = langevin(proto, timestep_fs=timestep_fs)
        if precision:
            sim = Simulation(pdb.topology, sys_obj, integ, pobj, {"Precision": precision})
        else:
            sim = Simulation(pdb.topology, sys_obj, integ, pobj)
        sim.context.setPositions(positions)
        if box is not None:
            sim.context.setPeriodicBoxVectors(*box)
        return sim

    def _finish(
        pos: Any, energy: float, temp: float, pobj: Any, pname: str, stage: str
    ) -> tuple[Any, float, float]:
        nonlocal plat, plat_name
        plat, plat_name = pobj, pname
        stages.append(stage)
        return pos, float(energy), temp

    def _run(label: str, sys_obj: Any, positions: Any, steps: int, minimize: bool) -> tuple[Any, float, float]:
        last: Exception | None = None
        min_pos = positions
        if minimize:
            ok = False
            for pname, pobj in platforms:
                try:
                    log(f"{label}: minimize ({pname})")
                    sim = _sim(sys_obj, pobj, 1.0, positions)
                    sim.minimizeEnergy(maxIterations=200 if short else 1500)
                    min_pos = sim.context.getState(getPositions=True).getPositions()
                    if _has_nan(min_pos):
                        raise RuntimeError("NaN after minimize")
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    log(f"{label} minimize failed on {pname}: {exc}")
            if not ok:
                raise RefuseError(f"{label} minimize failed: {last}")

        last = None
        cuda_failed = False
        pos = min_pos
        settled = False
        for pname, pobj in platforms:
            settle = settle_steps(short=short, platform=pname, cuda_failed=cuda_failed)
            try:
                prec = "double" if pname == "CUDA" else None
                sim = _sim(sys_obj, pobj, 1.0, min_pos, precision=prec)
                sim.context.setVelocitiesToTemperature(proto.temperature_K * unit.kelvin)
                if settle:
                    log(f"{label}: {settle} steps @ 1 fs ({pname})")
                    sim.step(settle)
                pos = sim.context.getState(getPositions=True).getPositions()
                if _has_nan(pos):
                    raise RuntimeError("NaN after 1 fs settle")
                settled = True
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                log(f"{label} dynamics failed on {pname}: {exc}")
                if pname != "CPU":
                    cuda_failed = True
        if not settled:
            raise RefuseError(f"{label} failed: {last}")

        if steps > 0 and not short:
            last = None
            for pname, pobj in four_fs_platforms(platforms):
                try:
                    sim4 = _sim(sys_obj, pobj, proto.eq_timestep_fs, pos)
                    sim4.context.setVelocitiesToTemperature(proto.temperature_K * unit.kelvin)
                    log(f"{label}: {steps} steps @ {proto.eq_timestep_fs:g} fs ({pname})")
                    sim4.step(steps)
                    state = sim4.context.getState(getPositions=True, getEnergy=True)
                    pos = state.getPositions()
                    if _has_nan(pos):
                        raise RuntimeError("NaN coordinates")
                    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    t = kinetic_temperature(state, n_atoms, sys_obj)
                    return _finish(pos, e, t, pobj, pname, label)
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    log(f"{label} 4 fs failed on {pname}: {exc}")
            raise RefuseError(f"{label} failed: {last}")

        state = sim.context.getState(getPositions=True, getEnergy=True)
        pos = state.getPositions()
        if _has_nan(pos):
            raise RuntimeError("NaN coordinates")
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        t = kinetic_temperature(state, n_atoms, sys_obj)
        return _finish(pos, e, t, pobj, pname, label)

    nvt_sys = strip_barostat(system)
    _restrain_ca(nvt_sys, pdb.topology, pdb.positions, k_rest)

    if short:
        nvt_steps = 100
        npt_steps = 0
        free_steps = 0
    else:
        nvt_steps = _steps(proto.eq_nvt_restrained_ps / 1000.0, proto.eq_timestep_fs)
        npt_steps = _steps(proto.eq_npt_restrained_ns, proto.eq_timestep_fs)
        free_steps = _steps(proto.eq_free_membrane_ns, proto.eq_timestep_fs)

    pos, energy, temp = _run("nvt_restrained", nvt_sys, pdb.positions, nvt_steps, minimize=True)
    if npt_steps:
        npt_sys = _clone(system)
        _restrain_ca(npt_sys, pdb.topology, pos, k_rest)
        pos, energy, temp = _run("npt_restrained", npt_sys, pos, npt_steps, minimize=False)

    qc = scorecard(
        pdb.topology,
        pos,
        potential_kj=energy,
        temperature_K=temp,
        target_T=proto.temperature_K,
        check_temperature=not short,
    )
    if (not qc.ok) and free_steps and proto.eq_stop_on_qc:
        log("QC not ready — free membrane continuation")
        pos, energy, temp = _run("free_membrane", system, pos, free_steps, minimize=False)
        qc = scorecard(
            pdb.topology,
            pos,
            potential_kj=energy,
            temperature_K=temp,
            target_T=proto.temperature_K,
            check_temperature=True,
        )
    elif qc.ok and proto.eq_stop_on_qc:
        log("QC ok — skipping free_membrane_ns")
        stages.append("stop_on_qc")

    out_pdb = workdir / "eq.pdb"
    with out_pdb.open("w") as fh:
        PDBFile.writeFile(pdb.topology, pos, fh, keepIds=True)
    state_xml = workdir / "eq_state.xml"
    integ = langevin(proto)
    sim = Simulation(pdb.topology, system, integ, plat)
    sim.context.setPositions(pos)
    sim.saveState(str(state_xml))

    payload = {
        "policy": "0.1",
        "created_utc": datetime.now(UTC).isoformat(),
        "platform": plat_name,
        "short": short,
        "stages": stages,
        "qc": {
            "ok": qc.ok,
            "potential_kj": qc.potential_kj,
            "temperature_K": qc.temperature_K,
            "box_nm": list(qc.box_nm),
            "n_lipid": qc.n_lipid,
            "apl_nm2": qc.apl_nm2,
            "thickness_nm": qc.thickness_nm,
            "notes": qc.notes,
        },
    }
    (workdir / "eq.json").write_text(json.dumps(payload, indent=2) + "\n")
    return EqResult(workdir=workdir, pdb=out_pdb, qc=qc, stages=stages, platform=plat_name)


def format_eq(r: EqResult) -> str:
    q = r.qc
    apl = "—" if q.apl_nm2 is None else f"{q.apl_nm2:.3f} nm²"
    th = "—" if q.thickness_nm is None else f"{q.thickness_nm:.2f} nm"
    lines = [
        f"EQUILIBRATE  {'ok' if q.ok else 'not ready'}  {r.platform}",
        f"stages   {', '.join(r.stages)}",
        f"box      {q.box_nm[0]:.2f} × {q.box_nm[1]:.2f} × {q.box_nm[2]:.2f} nm",
        f"lipids   {q.n_lipid}  APL {apl}  thickness {th}",
        f"E        {q.potential_kj:.3e} kJ/mol",
        f"wrote    {r.pdb}",
    ]
    for n in q.notes:
        lines.append(f"  qc  {n}")
    return "\n".join(lines)


def settle_steps(*, short: bool, platform: str, cuda_failed: bool) -> int:
    """1 fs settle length. CUDA NaN must not dump 2 ps onto CPU."""
    if short or (platform == "CPU" and cuda_failed):
        return 50
    return 2000


def four_fs_platforms(platforms: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    gpu = [(n, p) for n, p in platforms if n != "CPU"]
    return gpu or platforms


def _steps(time_ns: float, timestep_fs: float) -> int:
    return max(0, int(round(time_ns * 1_000_000.0 / timestep_fs)))


def _clone(system: Any) -> Any:
    from openmm import XmlSerializer

    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def _platform_ladder(preference: list[str]) -> list[tuple[str, Any]]:
    from openmm import Platform

    plat, name = pick_platform(preference)
    out = [(name, plat)]
    if name != "CPU":
        try:
            out.append(("CPU", Platform.getPlatformByName("CPU")))
        except Exception:  # noqa: BLE001
            pass
    return out


def _has_nan(positions: Any) -> bool:
    from openmm import unit

    xyz = positions.value_in_unit(unit.nanometer)
    return any(p[0] != p[0] or p[1] != p[1] or p[2] != p[2] for p in xyz)


def _restrain_ca(system: Any, topology: Any, positions: Any, k: float) -> None:
    from openmm import CustomExternalForce, unit

    force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    pos_nm = positions.value_in_unit(unit.nanometer)
    for atom in topology.atoms():
        if atom.name != "CA":
            continue
        x, y, z = pos_nm[atom.index]
        force.addParticle(atom.index, [float(x), float(y), float(z)])
    system.addForce(force)
