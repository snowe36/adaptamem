from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptamem.cli import main
from adaptamem.compress import compress, infer_report
from adaptamem.crystal import ca_contacts, crystal_insight
from adaptamem.errors import RefuseError
from adaptamem.features import atoms_xyz_from_pdb
from adaptamem.ladder import require_rung
from adaptamem.objective import Observable


def _ca_pdb(path: Path, chain: str, d_nm: float, *, extra: bool = False) -> Path:
    ang = d_nm * 10.0
    lines = [
        f"ATOM  {1:5d}  CA  ARG {chain}{131:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C",
        f"ATOM  {2:5d}  CA  LEU {chain}{272:4d}    {ang:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 40.00           C",
    ]
    if extra:
        lines.append(
            f"ATOM  {3:5d}  CA  ALA {chain}{140:4d}    {2.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 10.00           C"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _obs() -> list[Observable]:
    return [
        Observable(
            name="tm6_ic",
            kind="distance",
            selection="name CA and resid 131 ; name CA and resid 272",
            precision=0.2,
        )
    ]


def test_sequence_in_is_not_ready(tmp_path: Path):
    seq = tmp_path / "prot.fasta"
    seq.write_text(">x\nMVLSEGEWQL\n")
    assert main(["features", str(seq), "--out", str(tmp_path / "f.json")]) == 2


def test_cpu_only_crystal_prior_has_no_kinetics(tmp_path: Path):
    a = _ca_pdb(tmp_path / "2rh1.pdb", "A", 0.84, extra=True)
    b = _ca_pdb(tmp_path / "3sn6.pdb", "R", 1.54, extra=True)
    insight = crystal_insight([f"{a}:A", f"{b}:R"], _obs())
    assert insight["mode"] == "cpu_only"
    assert insight["gpu_hours"] == 0.0
    assert insight["heterogeneity"]["tm6_ic"] == pytest.approx(0.70, abs=0.01)
    assert insight["contacts"]["n_per_frame"]
    assert "pi" in insight["unidentified"]
    assert "pathways" in insight["unidentified"]
    dest = tmp_path / "features.json"
    assert (
        main(
            [
                "features",
                "--crystal",
                f"{a}:A",
                "--crystal",
                f"{b}:R",
                "--out",
                str(dest),
            ]
        )
        == 0
    )
    model_path = tmp_path / "model.json"
    assert main(["compress", str(dest), "--out", str(model_path)]) == 0
    pred = tmp_path / "pred.json"
    assert main(["infer", str(model_path), "--mode", "cpu_only", "--out", str(pred)]) == 0
    payload = json.loads(pred.read_text())
    assert payload["mode"] == "cpu_only"
    assert payload["gpu_hours"] == 0.0
    assert payload["identification"]["kinetics_identified"] is False
    assert not (tmp_path / "oracle_request.json").is_file()
    rounded = {round(x, 2) for x in payload["values"]["tm6_ic"]}
    assert rounded <= {0.84, 1.54}


def test_cpu_only_forbids_teacher_gpu_hours():
    model = compress("latent_dynamics", {"tm6_ic": [0.84, 1.54]}, teacher_gpu_hours=1.0)
    with pytest.raises(RefuseError) as ei:
        infer_report(model, mode="cpu_only")
    assert ei.value.code == "NOT_READY"


def test_conventional_is_not_an_infer_mode():
    model = compress("latent_dynamics", {"tm6_ic": [0.84, 1.54]})
    with pytest.raises(RefuseError) as ei:
        infer_report(model, mode="conventional")
    assert ei.value.code == "NOT_READY"


def test_ca_contacts_finds_close_pairs(tmp_path: Path):
    pdb = _ca_pdb(tmp_path / "x.pdb", "A", 0.5, extra=True)
    atoms, xyz = atoms_xyz_from_pdb(pdb, chain="A")
    pairs = ca_contacts(atoms, xyz, cutoff_nm=0.8, min_sep=4)
    assert pairs


def test_rates_rung_is_stubbed():
    with pytest.raises(RefuseError) as ei:
        require_rung("pathways")
    assert ei.value.code == "NOT_IMPLEMENTED"
    assert require_rung("structural") == "structural"
