"""OpenMM helpers. Imported only from assemble / equilibrate / bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.schema import Protocol

OPENMM_LIPIDS = ("POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC")
# CHARMM36 addMembrane patches store lipids as 3-letter names.
LIPID_RESIDUES = {
    "POPC",
    "POPE",
    "POP",
    "POPG",
    "DOPC",
    "DOP",
    "DPPC",
    "DPP",
    "DLPC",
    "DLPE",
    "DLP",
    "DMPC",
    "DMP",
}
MISSING_OPENMM = "OpenMM is not installed. pip install 'adaptamem[sim]'"


def openmm_available() -> bool:
    try:
        import openmm  # noqa: F401
    except ImportError:
        return False
    return True


def require_openmm() -> None:
    if not openmm_available():
        raise RefuseError(MISSING_OPENMM)


def openmm_lipid_type(lipids: dict[str, float]) -> tuple[str, str | None]:
    present = {k.upper(): v for k, v in lipids.items() if v > 0}
    if not present:
        raise RefuseError("membrane.lipids is empty")
    name = max(present, key=present.get)
    if name not in OPENMM_LIPIDS:
        raise RefuseError(
            f"{name} is not an OpenMM addMembrane lipid; "
            f"use one of {', '.join(OPENMM_LIPIDS)}"
        )
    note = None
    if len(present) > 1:
        note = f"mixed lipids {present}; Phase 1 builds 100% {name} (approximation)"
    return name, note


def pick_platform(preference: list[str] | None = None) -> tuple[Any, str]:
    from openmm import Platform

    order = list(preference or ["CUDA", "OpenCL", "CPU"])
    if "CPU" not in order:
        order.append("CPU")
    last: Exception | None = None
    for name in order:
        try:
            plat = Platform.getPlatformByName(name)
            if name == "CUDA":
                try:
                    plat.setPropertyDefaultValue("Precision", "mixed")
                except Exception:  # noqa: BLE001
                    pass
            return plat, name
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RefuseError(f"no OpenMM platform from {order}: {last}")


def make_hmr_system(topology: Any, protocol: Protocol) -> Any:
    from openmm import MonteCarloMembraneBarostat, unit
    from openmm.app import PME, HBonds

    ff = charmm36()
    hmass = float(protocol.hydrogen_mass_amu) * unit.amu
    system = ff.createSystem(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=float(protocol.nonbonded_cutoff_nm) * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=hmass,
    )
    system.addForce(
        MonteCarloMembraneBarostat(
            float(protocol.pressure_bar) * unit.bar,
            0.0 * unit.bar * unit.nanometer,
            float(protocol.temperature_K) * unit.kelvin,
            MonteCarloMembraneBarostat.XYIsotropic,
            MonteCarloMembraneBarostat.ZFree,
            15,
        )
    )
    return system


def charmm36() -> Any:
    from openmm.app import ForceField

    return ForceField("charmm36.xml", "charmm36/water.xml")


def strip_barostat(system: Any) -> Any:
    from openmm import XmlSerializer

    clone = XmlSerializer.deserialize(XmlSerializer.serialize(system))
    for i in range(clone.getNumForces() - 1, -1, -1):
        if "Barostat" in type(clone.getForce(i)).__name__:
            clone.removeForce(i)
    return clone


def langevin(protocol: Protocol, *, timestep_fs: float | None = None) -> Any:
    from openmm import LangevinMiddleIntegrator, unit

    dt = float(timestep_fs if timestep_fs is not None else protocol.eq_timestep_fs)
    return LangevinMiddleIntegrator(
        float(protocol.temperature_K) * unit.kelvin,
        1.0 / unit.picosecond,
        dt * unit.femtoseconds,
    )


@dataclass
class QC:
    potential_kj: float
    temperature_K: float | None
    box_nm: tuple[float, float, float]
    n_lipid: int
    apl_nm2: float | None
    thickness_nm: float | None
    ok: bool
    notes: list[str]


def scorecard(
    topology: Any,
    positions: Any,
    *,
    potential_kj: float,
    temperature_K: float | None,
    target_T: float,
    check_temperature: bool = True,
) -> QC:
    from openmm import unit

    box = topology.getPeriodicBoxVectors()
    if box is None:
        raise RefuseError("assembled system has no periodic box")
    a, b, c = box
    lx = a[0].value_in_unit(unit.nanometer)
    ly = b[1].value_in_unit(unit.nanometer)
    lz = c[2].value_in_unit(unit.nanometer)
    n_lipid = sum(1 for res in topology.residues() if res.name in LIPID_RESIDUES)
    apl = (lx * ly) / (n_lipid / 2.0) if n_lipid >= 2 else None
    thickness = _phosphate_thickness_nm(topology, positions)
    notes: list[str] = []
    ok = True
    if not math_isfinite(potential_kj):
        notes.append("potential is not finite")
        ok = False
    if n_lipid < 2:
        notes.append("no lipid residues in topology")
        ok = False
    if apl is not None and not (0.50 <= apl <= 0.85):
        notes.append(f"APL {apl:.3f} nm² outside 0.50–0.85")
        ok = False
    if thickness is not None and not (2.8 <= thickness <= 5.2):
        notes.append(f"thickness {thickness:.2f} nm outside 2.8–5.2")
        ok = False
    if (
        check_temperature
        and temperature_K is not None
        and abs(temperature_K - target_T) > 25
    ):
        notes.append(f"T {temperature_K:.0f} K vs target {target_T:.0f}")
        ok = False
    if not notes:
        notes.append("membrane QC within bounds")
    return QC(
        potential_kj=potential_kj,
        temperature_K=temperature_K,
        box_nm=(lx, ly, lz),
        n_lipid=n_lipid,
        apl_nm2=apl,
        thickness_nm=thickness,
        ok=ok,
        notes=notes,
    )


def _phosphate_thickness_nm(topology: Any, positions: Any) -> float | None:
    from openmm import unit

    zs: list[float] = []
    pos_nm = positions.value_in_unit(unit.nanometer)
    for atom in topology.atoms():
        if atom.name == "P" and atom.residue.name in LIPID_RESIDUES:
            zs.append(float(pos_nm[atom.index][2]))
    if len(zs) < 4:
        return None
    upper = [z for z in zs if z >= 0]
    lower = [z for z in zs if z < 0]
    if not upper or not lower:
        return max(zs) - min(zs)
    return (sum(upper) / len(upper)) - (sum(lower) / len(lower))


def math_isfinite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def kinetic_temperature(state: Any, n_atoms: int, system: Any | None = None) -> float:
    from openmm import unit

    ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    kb = 0.008314462618
    n_c = system.getNumConstraints() if system is not None else 0
    dof = max(3 * n_atoms - n_c - 3, 1)
    return 2.0 * ke / (dof * kb)
