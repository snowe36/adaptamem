from pathlib import Path

import pytest

from adaptamem.assemble import (
    assemble,
    assert_backbone_packable,
    default_workdir,
    gpu_must_not_pack,
    have_assembled,
)
from adaptamem.cli import main
from adaptamem.errors import RefuseError
from adaptamem.lipids import plan_mix
from adaptamem.schema import load_protocol
from adaptamem.session import load_session
from adaptamem.sim import openmm_available, openmm_lipid_type

from pdbutil import helix_pdb


def test_protocol_hmr_is_4_amu():
    p = load_protocol()
    assert p.hydrogen_mass_amu == 4.0
    assert p.eq_timestep_fs == 4.0


def test_charmm_patch_lipid_residue_names():
    from adaptamem.sim import LIPID_RESIDUES

    assert "POP" in LIPID_RESIDUES
    assert "POPC" in LIPID_RESIDUES
    assert "HOH" not in LIPID_RESIDUES


def test_majority_lipid_single_component():
    name, note = openmm_lipid_type({"POPC": 1.0})
    assert name == "POPC"
    assert note is None
    plan = plan_mix({"POPC": 1.0})
    assert plan.scaffold == "POPC"
    assert plan.n_swap == 0


def test_unknown_lipid_refuses():
    try:
        openmm_lipid_type({"POPG": 1.0})
    except RefuseError as exc:
        assert "POPG" in exc.message
        return
    raise AssertionError("expected RefuseError")


def test_twisted_helix_is_packable(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "leu.pdb")
    assert_backbone_packable(pdb)


def test_stacked_backbone_refused_before_addmembrane(tmp_path: Path):
    lines = ["HEADER    STACK"]
    serial = 1
    for i in range(8):
        z = i * 1.5
        for name, x in (("N", 0.0), ("CA", 1.46), ("C", 2.0)):
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} LEU A{i + 1:4d}    "
                f"{x:8.3f}{0.0:8.3f}{z:8.3f}  1.00  0.00           {name[0]}"
            )
            serial += 1
    lines.append("END")
    stacked = tmp_path / "stack.pdb"
    stacked.write_text("\n".join(lines) + "\n")
    with pytest.raises(RefuseError, match="stacked backbone"):
        assert_backbone_packable(stacked)


def test_assemble_refuses_no_tm(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "lys.pdb", n=12, resname="LYS")
    session = load_session(pdb, cli_objective="conventional")
    with pytest.raises(RefuseError, match="transmembrane"):
        assemble(session, tmp_path / "out")


def test_assemble_cli_without_openmm(tmp_path: Path):
    if openmm_available():
        pytest.skip("OpenMM present — CLI missing-engine path not exercised")
    pdb = helix_pdb(tmp_path / "leu.pdb")
    code = main(["assemble", str(pdb), "--out", str(tmp_path / "run")])
    assert code == 2


def test_gpu_must_not_pack_lipids():
    gpu_must_not_pack(cuda=False)
    with pytest.raises(RefuseError, match="CPU host"):
        gpu_must_not_pack(cuda=True)


def test_have_assembled(tmp_path: Path):
    assert have_assembled(tmp_path) is False
    (tmp_path / "assembled.pdb").write_text("ATOM\n")
    (tmp_path / "system.xml").write_text("<System/>\n")
    (tmp_path / "assemble.json").write_text("{}\n")
    assert have_assembled(tmp_path) is True


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
