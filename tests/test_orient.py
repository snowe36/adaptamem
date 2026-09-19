from pathlib import Path

from adaptamem.doctor import audit, load_atoms
from adaptamem.errors import RefuseError
from adaptamem.orient import orient_structure
from adaptamem.schema import Orientation

from pdbutil import helix_pdb


def test_orient_lays_x_helix_on_z(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "x.pdb", axis="x")
    report = audit(pdb)
    assert report.n_tm >= 1
    dest = tmp_path / "oriented.pdb"
    result = orient_structure(pdb, report, Orientation(method="auto", topology="in"), dest)
    assert result.span_nm[2] > result.span_nm[0]
    assert result.span_nm[2] > result.span_nm[1]
    atoms = load_atoms(dest)
    mean_z = sum(a.z for a in atoms) / len(atoms)
    assert abs(mean_z) < 2.0


def test_orient_z_helix_stays_on_z(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "z.pdb", axis="z")
    report = audit(pdb)
    dest = tmp_path / "oriented.pdb"
    result = orient_structure(pdb, report, Orientation(method="auto"), dest)
    assert result.span_nm[2] > result.span_nm[0]


def test_orient_ppm_refuses(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("adaptamem.ppm.find_immers", lambda immers_dir=None: None)
    pdb = helix_pdb(tmp_path / "z.pdb")
    report = audit(pdb)
    try:
        orient_structure(
            pdb, report, Orientation(method="ppm"), tmp_path / "out.pdb"
        )
    except RefuseError as exc:
        assert "immers" in exc.message.lower() or "ppm" in exc.message.lower()
        return
    raise AssertionError("expected RefuseError")
