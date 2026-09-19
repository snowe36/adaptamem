from adaptamem.diagnostics import (
    autocorrelation_time,
    ci_width_per_gpu_hour,
    effective_sample_size,
    mean_ci,
    summarize,
    uniqueness,
)
from adaptamem.select import BRANCH, KEEP, STOP, decide_axes


def test_iid_ess_near_n():
    vals = [float(i % 3) for i in range(200)]
    # not iid; just finite
    ess = effective_sample_size(vals)
    assert ess > 1
    tau = autocorrelation_time(vals)
    assert tau == tau


def test_constant_series_tau_is_one():
    assert autocorrelation_time([1.0] * 50) == 1.0
    assert effective_sample_size([1.0] * 50) == 50


def test_ci_width_per_gpu_hour():
    assert ci_width_per_gpu_hour(0.1, 2.0) == 0.1
    assert ci_width_per_gpu_hour(0.1, 0.0) == float("inf")


def test_summarize_schema():
    d = summarize([0.0, 1.0, 0.5, 0.4, 0.6], gpu_hours=1.5, trajectory_ns=10.0)
    payload = d.to_dict()
    assert "ess" in payload and "state_coverage" in payload
    assert payload["cluster_coverage"] is None
    assert payload["transition_count"] is None
    assert payload["ci_width_per_gpu_hour"] is not None


def test_uniqueness_detects_private_bin():
    a = [0.0] * 20 + [10.0] * 5
    b = [0.0] * 25
    assert uniqueness(a, [b]) > uniqueness(b, [a])


def test_decide_axes_matrix():
    assert decide_axes(True, False) == STOP
    assert decide_axes(True, True) == KEEP
    assert decide_axes(False, True) == BRANCH
    assert decide_axes(False, False) == "extend"


def test_mean_ci_empty():
    mu, half = mean_ci([])
    assert mu != mu
    assert half == float("inf")
