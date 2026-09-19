from adaptamem.compare import (
    MethodRun,
    acceleration_claim,
    compare,
    equal_precision,
    hours_to_oracle_error,
    synthetic_demo,
)
from adaptamem.oracle import Oracle


def test_equal_compute_and_precision_both_present():
    oracle, runs = synthetic_demo()
    payload = compare(oracle, runs)
    assert payload["primary_metric"] == "compression"
    assert payload["oracle_compression"]
    assert payload["equal_compute"]
    assert payload["equal_precision"]
    assert set(payload["methods"]) >= {"single_long", "independent_replicas", "adaptive"}
    adaptive = payload["equal_compute"]["methods"]["adaptive"]["cv"]
    single = payload["equal_compute"]["methods"]["single_long"]["cv"]
    assert adaptive["abs_error"] < single["abs_error"]


def test_hours_to_oracle_error_hits_early():
    stuck = [0.2] * 10 + [1.0] * 90
    hours = hours_to_oracle_error(stuck, oracle_mean=1.0, error=0.15, gpu_hours=10.0)
    assert hours is not None
    assert hours > 4.0
    early = hours_to_oracle_error([1.0] * 100, oracle_mean=1.0, error=0.15, gpu_hours=10.0)
    assert early is not None
    assert early < hours


def test_oracle_compression_scores_adaptive_cheaper():
    oracle, runs = synthetic_demo()
    payload = compare(oracle, runs, error=0.2, oracle_gpu_hours=40.0)
    oc = payload["oracle_compression"]["methods"]
    ad = oc["adaptive"]["cv"]
    sl = oc["single_long"]["cv"]
    assert ad["reached"]
    if sl["reached"]:
        assert ad["gpu_hours_to_error"] <= sl["gpu_hours_to_error"]
    assert ad["compression_vs_oracle"] is not None
    assert ad["compression_vs_oracle"] >= 1.0
    assert ad["acceleration"]["accelerated"] is True
    assert sl["acceleration"]["accelerated"] is False


def test_tight_ci_without_matched_error_is_not_acceleration():
    claim = acceleration_claim(
        reached=False,
        compression_vs_oracle=50.0,
        gpu_hours=1.0,
        cpu_hours=0.2,
    )
    assert claim["accelerated"] is False
    assert "match" in claim["reason"]


def test_leakage_blocks_acceleration_claim():
    claim = acceleration_claim(
        reached=True,
        compression_vs_oracle=20.0,
        gpu_hours=1.0,
        leaked=True,
    )
    assert claim["accelerated"] is False
    assert claim["reason"] == "oracle leakage"


def test_below_tenx_is_not_a_claim():
    oracle = Oracle(values={"cv": [1.0] * 40}, estimates={"cv": {"estimate": 1.0}})
    runs = [
        MethodRun("single_long", {"cv": [1.0] * 40}, gpu_hours=10.0, cpu_hours=0.0),
        MethodRun("msm", {"cv": [1.0] * 40}, gpu_hours=8.0, cpu_hours=0.1),
    ]
    payload = compare(oracle, runs, error=0.2, oracle_gpu_hours=10.0)
    msm = payload["oracle_compression"]["methods"]["msm"]["cv"]
    assert msm["reached"]
    assert msm["compression_vs_oracle"] is not None
    assert msm["compression_vs_oracle"] < 10.0
    assert msm["acceleration"]["accelerated"] is False
    assert msm["cpu_hours"] == 0.1


def test_tenx_is_the_claim_line():
    below = acceleration_claim(reached=True, compression_vs_oracle=9.99, gpu_hours=1.0)
    on_bar = acceleration_claim(reached=True, compression_vs_oracle=10.0, gpu_hours=1.0)
    assert below["accelerated"] is False
    assert on_bar["accelerated"] is True


def test_equal_precision_adaptive_reaches_sooner():
    oracle, runs = synthetic_demo()
    ep = equal_precision(oracle, runs, precision=0.15, coverage_bar=0.25)
    ad = ep["methods"]["adaptive"]["cv"]
    sl = ep["methods"]["single_long"]["cv"]
    assert ad["reached"]
    assert ad["gpu_hours_to_precision"] is not None
    # A tight CI in one well must not count as reaching precision.
    if sl["reached"]:
        assert ad["gpu_hours_to_precision"] <= sl["gpu_hours_to_precision"]
    else:
        assert sl["gpu_hours_to_precision"] is None


def test_cpu_only_msm_matching_epsilon_is_acceleration():
    oracle = Oracle(values={"cv": [1.0] * 40}, estimates={"cv": {"estimate": 1.0}})
    runs = [
        MethodRun("single_long", {"cv": [1.0] * 40}, gpu_hours=40.0),
        MethodRun("msm", {"cv": [1.0] * 40}, gpu_hours=0.0, cpu_hours=0.01),
    ]
    payload = compare(oracle, runs, error=0.2, oracle_gpu_hours=40.0)
    msm = payload["oracle_compression"]["methods"]["msm"]["cv"]
    assert msm["reached"]
    assert msm["cpu_only"]
    assert msm["acceleration"]["accelerated"] is True
    assert "CPU-only" in msm["acceleration"]["reason"]


def test_compression_is_md_avoided_over_oracle_spent():
    oracle = Oracle(values={"cv": [1.0] * 40}, estimates={"cv": {"estimate": 1.0}})
    runs = [
        MethodRun("single_long", {"cv": [1.0] * 40}, gpu_hours=69.0),
        MethodRun("msm", {"cv": [1.0] * 40}, gpu_hours=6.9, cpu_hours=0.01),
    ]
    payload = compare(oracle, runs, error=0.2, oracle_gpu_hours=69.0)
    msm = payload["oracle_compression"]["methods"]["msm"]["cv"]
    assert msm["reached"]
    assert abs(msm["compression"] - 10.0) < 1e-9
    assert msm["acceleration"]["accelerated"] is True
