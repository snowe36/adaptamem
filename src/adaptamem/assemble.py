"""Orient the protein and build a compact CHARMM36 bilayer with OpenMM addMembrane."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.box import format_box
from adaptamem.doctor import format_report
from adaptamem.errors import RefuseError
from adaptamem.orient import OrientResult, format_orient, orient_structure
from adaptamem.schema import Protocol, load_protocol
from adaptamem.session import Session
from adaptamem.strategy import PHYSICS_APPROX, format_strategy

Progress = Callable[[str], None]


@dataclass
class AssembleResult:
    workdir: Path
    oriented: Path
    assembled: Path
    system_xml: Path
    lipid: str
    platform: str
    n_atoms: int
    pad_nm: float
    physics: str
    notes: list[str]
    orient: OrientResult


def assemble(
    session: Session,
    workdir: Path,
    *,
    protocol: Protocol | None = None,
    force: bool = False,
    progress: Progress | None = None,
) -> AssembleResult:
    session.gate_assemble(force=force)
    lipid, mix_note = _lipid_for_session(session)
    from adaptamem.sim import charmm36, make_hmr_system, pick_platform, require_openmm

    require_openmm()
    proto = protocol or load_protocol()
    log = progress or (lambda _m: None)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    (workdir / "doctor.txt").write_text(format_report(session.report) + "\n")
    (workdir / "strategy.txt").write_text(format_strategy(session.strategy) + "\n")
    (workdir / "box.txt").write_text(format_box(session.box) + "\n")

    physics = session.strategy.physics
    notes: list[str] = []
    if mix_note:
        notes.append(mix_note)
        physics = PHYSICS_APPROX
    if force and session.strategy.refuse:
        notes.append(f"human veto: {session.strategy.refuse}")

    log("orient TM axis to z, midplane at 0")
    oriented = orient_structure(
        session.system.structure,
        session.report,
        session.system.orientation,
        workdir / "oriented.pdb",
    )
    log(format_orient(oriented).split("\n")[0])

    from openmm import XmlSerializer, unit
    from openmm.app import Modeller, PDBFile

    pad = session.box.safety_margin_nm
    if not session.system.membrane.optimize_size:
        pad = max(pad, 2.0)
        notes.append("optimize_size=false; padding floored at 2.0 nm")

    ff = charmm36()
    log("add missing heavy atoms + CHARMM36 hydrogens")
    protein, fix_notes = _load_protein(oriented.path, ff)
    notes.extend(fix_notes)

    plat, plat_name = pick_platform(proto.platform_preference)
    pads = _pad_schedule(pad)
    built = False
    used_pad = pad
    last: Exception | None = None
    platforms = [(plat_name, plat)]
    if plat_name != "CPU":
        from openmm import Platform

        try:
            platforms.append(("CPU", Platform.getPlatformByName("CPU")))
        except Exception:  # noqa: BLE001
            pass

    ionic = float(session.system.membrane.ionic_strength_M)
    for pname, pobj in platforms:
        for p in pads:
            modeller = Modeller(protein.topology, protein.positions)
            try:
                log(f"addMembrane {lipid} platform={pname} pad={p:g} nm")
                _add_membrane(modeller, ff, pobj, lipid, p, ionic)
                plat_name = pname
                used_pad = p
                built = True
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                log(f"addMembrane failed ({pname}, {p:g} nm): {exc}")
        if built:
            break
    if not built:
        raise RefuseError(f"addMembrane failed: {last}")

    log("createSystem HMR + membrane barostat")
    omm = make_hmr_system(modeller.topology, proto)
    assembled = workdir / "assembled.pdb"
    with assembled.open("w") as fh:
        PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)
    system_xml = workdir / "system.xml"
    system_xml.write_text(XmlSerializer.serialize(omm))

    n_atoms = modeller.topology.getNumAtoms()
    box = modeller.topology.getPeriodicBoxVectors()
    box_nm = None
    if box is not None:
        box_nm = [
            box[0][0].value_in_unit(unit.nanometer),
            box[1][1].value_in_unit(unit.nanometer),
            box[2][2].value_in_unit(unit.nanometer),
        ]
    notes.append(
        f"HMR {proto.hydrogen_mass_amu:g} amu, Δt {proto.eq_timestep_fs:g} fs, "
        f"cutoff {proto.nonbonded_cutoff_nm:g} nm"
    )
    payload: dict[str, Any] = {
        "policy": "0.1",
        "created_utc": datetime.now(UTC).isoformat(),
        "system": session.system.name,
        "structure": str(session.system.structure),
        "lipid": lipid,
        "platform": plat_name,
        "pad_nm": used_pad,
        "n_atoms": n_atoms,
        "box_nm": box_nm,
        "physics": physics,
        "hydrogen_mass_amu": proto.hydrogen_mass_amu,
        "timestep_fs": proto.eq_timestep_fs,
        "cutoff_nm": proto.nonbonded_cutoff_nm,
        "ionic_strength_M": ionic,
        "force": force,
        "notes": notes,
        "orient": {
            "method": oriented.method,
            "topology": oriented.topology,
            "axis_before": list(oriented.axis_before),
            "span_nm": list(oriented.span_nm),
        },
    }
    (workdir / "assemble.json").write_text(json.dumps(payload, indent=2) + "\n")
    return AssembleResult(
        workdir=workdir,
        oriented=oriented.path,
        assembled=assembled,
        system_xml=system_xml,
        lipid=lipid,
        platform=plat_name,
        n_atoms=n_atoms,
        pad_nm=used_pad,
        physics=physics,
        notes=notes,
        orient=oriented,
    )


def format_assemble(r: AssembleResult) -> str:
    lines = [
        f"ASSEMBLE  {r.n_atoms} atoms  {r.lipid}  pad {r.pad_nm:g} nm  {r.platform}",
        f"physics   {r.physics}",
        f"oriented  {r.oriented}",
        f"system    {r.assembled}",
    ]
    for n in r.notes:
        lines.append(f"  note  {n}")
    if r.physics == PHYSICS_APPROX:
        lines.append("NOTE  approximation — do not quote ns/day against a full-AA baseline")
    return "\n".join(lines)


def default_workdir(session: Session, out: Path | None) -> Path:
    if out is not None:
        return Path(out)
    return Path("runs") / session.system.name


def _lipid_for_session(session: Session) -> tuple[str, str | None]:
    from adaptamem.sim import openmm_lipid_type

    lipid, note = openmm_lipid_type(session.system.membrane.lipids)
    if note and session.objective.type == "membrane_environment":
        raise RefuseError(
            "membrane_environment needs the requested lipid mix; "
            "OpenMM addMembrane is single-lipid in Phase 1"
        )
    return lipid, note


def _load_protein(oriented: Path, ff: Any) -> tuple[Any, list[str]]:
    """Heavy atoms via PDBFixer (OXT/sidechains, not missing loops), then CHARMM36 H."""
    from openmm.app import Modeller, PDBFile

    notes: list[str] = []
    try:
        from pdbfixer import PDBFixer
    except ImportError:
        PDBFixer = None  # type: ignore[misc, assignment]

    if PDBFixer is not None:
        fixer = PDBFixer(filename=str(oriented))
        fixer.findMissingResidues()
        n_gap = sum(len(v) for v in (fixer.missingResidues or {}).values())
        if n_gap:
            notes.append(f"pdbfixer saw {n_gap} missing residue(s); not reconstructing")
        fixer.missingResidues = {}
        fixer.findMissingAtoms()
        n_atoms = sum(len(v) for v in (fixer.missingAtoms or {}).values())
        n_term = sum(len(v) for v in (fixer.missingTerminals or {}).values())
        fixer.addMissingAtoms()
        if n_atoms or n_term:
            notes.append(f"pdbfixer added {n_atoms} missing + {n_term} terminal heavy atom(s)")
        modeller = Modeller(fixer.topology, fixer.positions)
    else:
        pdb = PDBFile(str(oriented))
        modeller = Modeller(pdb.topology, pdb.positions)
        notes.append("pdbfixer not installed; hydrogens only")

    try:
        modeller.addHydrogens(ff)
    except ValueError as exc:
        raise RefuseError(
            f"CHARMM36 could not match a residue (missing heavy atoms?): {exc}"
        ) from exc
    return modeller, notes


def _pad_schedule(min_pad: float) -> list[float]:
    pads = [float(min_pad)]
    for p in (2.0, 2.5, 3.0):
        if p > min_pad:
            pads.append(p)
    return pads


def _add_membrane(modeller: Any, ff: Any, platform: Any, lipid: str, pad_nm: float, ionic_M: float) -> None:
    from openmm import unit

    kwargs = dict(
        lipidType=lipid,
        membraneCenterZ=0.0 * unit.nanometer,
        minimumPadding=float(pad_nm) * unit.nanometer,
        positiveIon="K+",
        negativeIon="Cl-",
        ionicStrength=float(ionic_M) * unit.molar,
        neutralize=True,
    )
    try:
        modeller.addMembrane(ff, platform=platform, **kwargs)
    except TypeError:
        modeller.addMembrane(ff, **kwargs)
