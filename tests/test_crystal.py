from pathlib import Path

import pytest

from adaptamem.crystal import gate_span, traces_from_crystals
from adaptamem.errors import RefuseError
from adaptamem.objective import Observable


def _ca_pdb(path: Path, chain: str, d_nm: float) -> Path:
    ang = d_nm * 10.0
    lines = [
        f"ATOM  {1:5d}  CA  ARG {chain}{131:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C",
        f"ATOM  {2:5d}  CA  LEU {chain}{272:4d}    {ang:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_crystal_gate_aborts_when_span_below_eps(tmp_path: Path):
    a = _ca_pdb(tmp_path / "inact.pdb", "A", 1.41)
    b = _ca_pdb(tmp_path / "act.pdb", "R", 1.54)
    obs = [
        Observable(
            name="tm6_ic",
            kind="distance",
            selection="name CA and resid 131 ; name CA and resid 272",
            precision=0.2,
        )
    ]
    traces = traces_from_crystals([f"{a}:A", f"{b}:R"], obs)
    with pytest.raises(RefuseError) as ei:
        gate_span(traces, obs)
    assert ei.value.code == "EASY_WELL"


def test_crystal_gate_passes_inactive_active_pair(tmp_path: Path):
    a = _ca_pdb(tmp_path / "2rh1.pdb", "A", 0.84)
    b = _ca_pdb(tmp_path / "3sn6.pdb", "R", 1.54)
    obs = [
        Observable(
            name="tm6_ic",
            kind="distance",
            selection="name CA and resid 131 ; name CA and resid 272",
            precision=0.2,
        )
    ]
    traces = traces_from_crystals([f"{a}:A", f"{b}:R"], obs)
    spans = gate_span(traces, obs)
    assert spans["tm6_ic"] == pytest.approx(0.70, abs=0.01)
