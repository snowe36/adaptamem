from pathlib import Path

import pytest

from adaptamem.errors import RefuseError
from adaptamem.pipeline import EXPERIMENTS
from adaptamem.transfer import (
    crystal_table,
    leave_one_out,
    load_catalog,
    prior_unreliable,
    score_pair,
    teacher_curve,
)

ROOT = Path(__file__).resolve().parents[1]


def _ca_pair(path: Path, d_nm: float) -> Path:
    ang = d_nm * 10.0
    path.write_text(
        "ATOM      1  CA  ARG A 131       0.000   0.000   0.000  1.00  0.00           C\n"
        f"ATOM      2  CA  LEU A 272    {ang:8.3f}   0.000   0.000  1.00  0.00           C\n"
    )
    return path


def test_experiments_are_the_three_demonstrations():
    assert EXPERIMENTS == (
        "mechanistic_teacher",
        "zero_shot_prior",
        "adaptive_correction",
    )


def test_loo_predicts_held_out_from_other_proteins():
    table = {
        "p1": {"tm6_ic": {"inactive": 0.80, "active": 1.50, "span": 0.70}},
        "p2": {"tm6_ic": {"inactive": 0.84, "active": 1.54, "span": 0.70}},
        "p3": {"tm6_ic": {"inactive": 0.78, "active": 1.48, "span": 0.70}},
        "hold": {"tm6_ic": {"inactive": 0.82, "active": 1.52, "span": 0.70}},
    }
    out = leave_one_out(table, "hold")
    assert out["experiment"] == "zero_shot_prior"
    assert out["gpu_hours"] == 0.0
    assert "hold" not in out["train"]
    err = out["predictions"]["tm6_ic"]["error"]["inactive"]
    assert err < 0.2


def test_held_out_traces_are_leakage():
    table = {
        "p1": {"tm6_ic": {"inactive": 0.8, "active": 1.5, "span": 0.7}},
        "p2": {"tm6_ic": {"inactive": 0.8, "active": 1.5, "span": 0.7}},
        "hold": {"tm6_ic": {"inactive": 0.8, "active": 1.5, "span": 0.7}},
    }
    with pytest.raises(RefuseError) as ei:
        leave_one_out(table, "hold", traces={"tm6_ic": [0.8, 1.5]})
    assert ei.value.code == "LEAKAGE"


def test_loo_needs_two_training_proteins():
    table = {
        "p1": {"tm6_ic": {"inactive": 0.8, "active": 1.5, "span": 0.7}},
        "hold": {"tm6_ic": {"inactive": 0.8, "active": 1.5, "span": 0.7}},
    }
    with pytest.raises(RefuseError) as ei:
        leave_one_out(table, "hold")
    assert ei.value.code == "NOT_READY"


def test_teacher_curve_hours_to_eps():
    curve = teacher_curve(
        zero_shot=1.20,
        teacher_mu={0.5: 1.05, 1.0: 0.98, 2.0: 0.97},
        oracle_mu=0.96,
        epsilon=0.1,
        gpu_hours={0.0: 0.0, 0.5: 0.05, 1.0: 0.10, 2.0: 0.20},
    )
    assert curve["curve"][0]["ns"] == 0.0
    assert curve["curve"][0]["reached"] is False
    assert curve["gpu_hours_to_eps"] == 0.05


def test_ood_zero_shot_is_unreliable_not_undersampled():
    assert prior_unreliable(0.6, 0.05) is True
    assert prior_unreliable(0.02, 0.08) is False


def test_score_pair_from_two_pdbs(tmp_path: Path):
    a = _ca_pair(tmp_path / "inact.pdb", 0.84)
    b = _ca_pair(tmp_path / "act.pdb", 1.54)
    scored = score_pair(a, b, {"tm6_ic": [131, 272]})
    assert scored["tm6_ic"]["span"] == pytest.approx(0.70, abs=0.01)


def test_adrb2_alone_is_not_a_zero_shot_prior():
    table = crystal_table(load_catalog(), root=ROOT)
    assert "adrb2" in table
    with pytest.raises(RefuseError) as ei:
        leave_one_out(table, "adrb2")
    assert ei.value.code == "NOT_READY"
