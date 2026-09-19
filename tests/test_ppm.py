from pathlib import Path

from adaptamem.doctor import audit
from adaptamem.errors import RefuseError
from adaptamem.orient import orient_structure
from adaptamem.ppm import find_immers, require_immers
from adaptamem.schema import Orientation

from pdbutil import helix_pdb


def test_require_immers_refuses_when_missing(monkeypatch):
    monkeypatch.setattr("adaptamem.ppm.find_immers", lambda immers_dir=None: None)
    try:
        require_immers()
    except RefuseError as exc:
        assert "immers" in exc.message
        assert "auto" in exc.message
        return
    raise AssertionError("expected RefuseError")


def test_orient_ppm_does_not_fall_back_to_auto(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("adaptamem.ppm.find_immers", lambda immers_dir=None: None)
    pdb = helix_pdb(tmp_path / "z.pdb")
    report = audit(pdb)
    try:
        orient_structure(pdb, report, Orientation(method="ppm"), tmp_path / "out.pdb")
    except RefuseError as exc:
        assert "immers" in exc.message.lower() or "ppm" in exc.message.lower()
        return
    raise AssertionError("expected RefuseError")


def test_find_immers_respects_env(tmp_path: Path, monkeypatch):
    dummy = tmp_path / "immers"
    dummy.write_text("#!/bin/sh\n")
    dummy.chmod(0o755)
    monkeypatch.setenv("ADAPTAMEM_PPM_DIR", str(tmp_path))
    found = find_immers()
    assert found == dummy
