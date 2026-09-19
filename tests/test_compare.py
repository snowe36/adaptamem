from adaptamem.compare import compare, equal_precision, synthetic_demo


def test_equal_compute_and_precision_both_present():
    oracle, runs = synthetic_demo()
    payload = compare(oracle, runs)
    assert payload["equal_compute"]
    assert payload["equal_precision"]
    assert set(payload["methods"]) >= {"single_long", "independent_replicas", "adaptive"}
    adaptive = payload["equal_compute"]["methods"]["adaptive"]["cv"]
    single = payload["equal_compute"]["methods"]["single_long"]["cv"]
    assert adaptive["abs_error"] < single["abs_error"]


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
