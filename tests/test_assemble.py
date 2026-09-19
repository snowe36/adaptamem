from pathlib import Path

import pytest

from adaptamem.assemble import assemble, default_workdir
from adaptamem.cli import main
from adaptamem.errors import RefuseError
from adaptamem.schema import Membrane, System, load_protocol
from adaptamem.session import load_session
from adaptamem.sim import openmm_available, openmm_lipid_type

from pdbutil import helix_pdb


def test_protocol_hmr_is_4_amu():
    p = load_protocol()
    assert p.hydrogen_mass_amu == 4.0
    assert p.eq_timestep_fs == 4.0


def test_majority_lipid_and_mix_note():
    name, note = openmm_lipid_type({"POPC": 0.7, "POPE": 0.3})
    assert name == "POPC"
    assert note is not None
    assert "approximation" in note


def test_unknown_lipid_refuses():
    try:
        openmm_lipid_type({"POPG": 1.0})
    except RefuseError as exc:
        assert "POPG" in exc.message
        return
    raise AssertionError("expected RefuseError")


def test_assemble_refuses_no_tm(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "lys.pdb", n=12, resname="LYS")
    session = load_session(pdb, cli_objective="conventional")
    with pytest.raises(RefuseError, match="transmembrane"):
        assemble(session, tmp_path / "out")


def test_membrane_environment_refuses_lipid_mix(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "leu.pdb")
    session = load_session(pdb, cli_objective="membrane-environment")
    session.system = System(
        name=session.system.name,
        structure=session.system.structure,
        orientation=session.system.orientation,
        membrane=Membrane(lipids={"POPC": 0.7, "POPE": 0.3}),
        objective=session.objective,
    )
    with pytest.raises(RefuseError, match="lipid mix"):
        assemble(session, tmp_path / "out")


def test_assemble_cli_without_openmm(tmp_path: Path):
    if openmm_available():
        pytest.skip("OpenMM present — CLI missing-engine path not exercised")
    pdb = helix_pdb(tmp_path / "leu.pdb")
    code = main(["assemble", str(pdb), "--out", str(tmp_path / "run")])
    assert code == 2


def test_cli_assemble_help():
    try:
        main(["assemble", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
        return
    raise AssertionError("expected help SystemExit")


def test_default_workdir_uses_system_name(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "leu.pdb")
    session = load_session(pdb)
    assert default_workdir(session, None) == Path("runs") / session.system.name
    assert default_workdir(session, tmp_path / "x") == tmp_path / "x"


@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
def test_orient_then_lipid_picker_ready_for_openmm(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "leu.pdb")
    session = load_session(pdb)
    session.gate_assemble()
    lipid, note = openmm_lipid_type(session.system.membrane.lipids)
    assert lipid == "POPC"
    assert note is None
