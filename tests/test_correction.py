from pathlib import Path

import pytest

from adaptamem.correction import (
    MECHANISM_COORDINATE,
    adaptive_step,
    classify_observable,
    correction_curve,
    disagreement,
    load_experiment3,
    pairwise_identity,
    posterior_update,
    should_teach,
    transfer_vs_distance,
)
from adaptamem.errors import RefuseError


def test_experiment3_is_frozen_gpu_off():
    proto = load_experiment3()
    assert proto["experiment"] == "adaptive_correction"
    assert proto["gpu"] is False
    assert proto["hold_out"]["primary"] == "adrb2"
    assert proto["hold_out"]["adversarial"] == "acm2"
    assert proto["hold_out"]["hard"] == "htr2a"
    assert proto["teacher"]["budgets_gpu_hours"][0] == 0.0
    assert 1.0 in proto["teacher"]["budgets_gpu_hours"]
    assert proto["observables"]["tm6_ic"]["role"] == "transferable"
    assert proto["observables"]["ionic_lock"]["role"] == "failure"
    assert proto["observables"]["tm3_tm6_pack"]["role"] == "trivial"
    assert proto["observables"][MECHANISM_COORDINATE]["role"] == "mechanism"


def test_packing_small_span_is_trivial_not_transfer():
    pred = {
        "truth": {"inactive": 0.79, "active": 0.82, "span": 0.03},
        "span": {"mu": 0.07},
        "reached": {"span": True},
        "unreliable": {"inactive": False, "active": False},
    }
    assert classify_observable("tm3_tm6_pack", pred) == "trivial"
    assert should_teach("trivial", pred) is False


def test_prior_that_invents_a_switch_is_failure():
    pred = {
        "truth": {"inactive": 1.41, "active": 1.55, "span": 0.14},
        "span": {"mu": 0.68},
        "reached": {"span": False},
        "unreliable": {"inactive": False, "active": False},
    }
    assert classify_observable("ionic_lock", pred) == "failure"
    assert should_teach("failure", pred) is True


def test_tm6_reached_is_transferable_skip_md():
    pred = {
        "truth": {"inactive": 0.84, "active": 1.54, "span": 0.70},
        "span": {"mu": 0.61},
        "reached": {"span": True},
        "unreliable": {"inactive": False, "active": False},
    }
    assert classify_observable("tm6_ic", pred) == "transferable"
    assert should_teach("transferable", pred) is False


def test_ionic_lock_miss_is_failure_call_teacher():
    pred = {
        "truth": {"inactive": 1.11, "active": 1.90, "span": 0.78},
        "span": {"mu": 0.58},
        "reached": {"span": False},
        "unreliable": {"inactive": False, "active": False},
    }
    assert classify_observable("ionic_lock", pred) == "failure"
    assert should_teach("failure", pred) is True


def test_correction_curve_is_gpu_hours_not_ns():
    curve = correction_curve(
        zero_shot=1.20,
        teacher_mu={0.05: 1.05, 0.10: 0.98, 0.25: 0.97, 1.0: 0.96},
        oracle_mu=0.96,
        epsilon=0.1,
        conventional_gpu_hours=20.0,
    )
    assert curve["curve"][0]["gpu_hours"] == 0.0
    assert curve["curve"][0]["reached"] is False
    assert curve["gpu_hours_to_eps"] == 0.05
    assert curve["curve"][1]["error_reduction_per_gpu_hour"] is not None
    assert curve["compression"] == pytest.approx(20.0 / 0.05)


def test_crystal_reference_is_not_an_acceleration():
    curve = correction_curve(
        zero_shot=0.80,
        teacher_mu={0.05: 0.80},
        oracle_mu=0.84,
        epsilon=0.2,
        reference="crystal_endpoints",
    )
    assert curve["compression"] is None


def test_oracle_must_not_update_posterior():
    with pytest.raises(RefuseError) as ei:
        posterior_update(1.6, 0.1, 1.1, 0.05, oracle_mu=1.05)
    assert ei.value.code == "LEAKAGE"


def test_disagreement_inflates_uncertainty_and_moves_toward_teacher():
    post = posterior_update(1.60, 0.08, 1.10, 0.05)
    assert post["disagreement"] is True
    assert post["mu"] < 1.60
    assert post["mu"] > 1.10
    assert post["sd"] >= 0.08
    assert disagreement(1.60, 0.08, 1.10) is True


def test_killer_loop_prior_wrong_teacher_corrects_without_oracle():
    prior = {"mu": 1.58, "sd": 0.19}
    teacher = {"mu": 1.14, "sd": 0.05}
    oracle = 1.12
    step = adaptive_step(prior, role="failure", teacher=teacher, oracle_mu=oracle)
    assert step["action"] == "update"
    assert step["error"] < step["zero_shot_error"]
    assert step["posterior"]["disagreement"] is True


def test_confident_tm6_does_not_request_gpu():
    step = adaptive_step({"mu": 0.84, "sd": 0.13}, role="transferable")
    assert step["action"] == "trust_prior"
    assert step["gpu"] is False


def test_failure_without_teacher_does_not_turn_gpu_on():
    step = adaptive_step({"mu": 1.58, "sd": 0.19}, role="failure")
    assert step["action"] == "request_teacher"
    assert step["gpu"] is False
    assert step["refuse"] == "NOT_READY"


def test_oracle_cli_refuses_while_experiment3_frozen(tmp_path: Path, capsys):
    (tmp_path / "assembled.pdb").write_text("ATOM\n")
    (tmp_path / "system.xml").write_text("<System/>\n")
    (tmp_path / "assemble.json").write_text("{}\n")
    from adaptamem.cli import main

    assert main(["oracle", str(tmp_path), "--ns", "2"]) == 2
    err = capsys.readouterr().err
    assert "experiment 3 protocol is frozen" in err


def test_pairwise_identity_identical_is_one():
    assert pairwise_identity("ARNDCEQGHILKMFPSTWYV", "ARNDCEQGHILKMFPSTWYV") == 1.0
    assert pairwise_identity("AAAA", "WWWW") == 0.0


def test_transfer_vs_distance_rows():
    sweep = {
        "folds": {
            "hold": {
                "train": ["p1", "p2"],
                "predictions": {
                    "tm6_ic": {"error": {"span": 0.09}, "reached": {"span": True}}
                },
            }
        }
    }
    ids = {("hold", "p1"): 0.4, ("hold", "p2"): 0.5}
    rows = transfer_vs_distance(sweep, ids)
    assert rows[0]["mean_train_identity"] == pytest.approx(0.45)
    assert rows[0]["span_error"] == 0.09


def test_mechanism_stat_is_packing_inside_inactive_tm6():
    from adaptamem.correction import mechanism_stat

    tm6 = [0.80, 0.82, 1.50, 1.52]
    pack = [1.10, 1.12, 0.80, 0.81]
    out = mechanism_stat(tm6, pack, inactive_hi=1.0)
    assert out["n"] == 2
    assert out["mu"] == pytest.approx(1.11, abs=0.01)


def test_mechanism_stat_refuses_tm6_only_teacher():
    from adaptamem.correction import mechanism_stat

    with pytest.raises(RefuseError) as ei:
        mechanism_stat([0.8, 0.9], [], inactive_hi=1.0)
    assert ei.value.code == "NOT_READY"
