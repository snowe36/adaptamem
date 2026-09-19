import json
from pathlib import Path

from adaptamem.oracle import freeze, load_oracle, score
from adaptamem.produce import ns_from_budget, resolve_ns
from adaptamem.schema import load_protocol


def test_ns_from_budget():
    assert abs(ns_from_budget(24.0, 120.0) - 120.0) < 1e-9


def test_resolve_ns_caps_at_budget(tmp_path: Path):
    proto = load_protocol()
    (tmp_path / "bench.json").write_text(
        json.dumps({"performance": {"ns_per_day": 48.0}}) + "\n"
    )
    ns, notes = resolve_ns(tmp_path, ns=100.0, protocol=proto, budget_hours=12.0)
    assert ns == 24.0
    assert notes


def test_oracle_score_recovers_mean(tmp_path: Path):
    series = [1.0, 1.1, 0.9, 1.0, 1.05]
    path = tmp_path / "oracle.json"
    freeze({"cv": series}, path)
    oracle = load_oracle(path)
    est = {
        "cv": {
            "estimate": 1.01,
            "ci95": 0.2,
            "values": series[:3],
            "gpu_hours": 2.0,
            "ci_width_per_gpu_hour": 0.2,
            "ess": 3.0,
            "state_coverage": 0.4,
        }
    }
    scored = score(est, oracle)
    assert scored["cv"]["abs_error"] < 0.1
    assert scored["cv"]["oracle_mean_in_ci"] is True
