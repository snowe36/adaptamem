from adaptamem.compress import (
    KINDS,
    compress,
    infer,
    leakage_keys,
    msm_from_traces,
    uncertain_regions,
)
from adaptamem.errors import RefuseError


def test_unknown_kind_refuses():
    try:
        compress("more_md", {"cv": [1.0]})
    except RefuseError as exc:
        assert exc.code == "NOT_IMPLEMENTED"
        return
    raise AssertionError("expected RefuseError")


def test_named_plug_does_not_fall_through_to_md():
    try:
        compress("milestoning", {"cv": [1.0, 1.1]})
    except RefuseError as exc:
        assert exc.code == "COMPRESS"
        assert "msm" in KINDS
        assert "active_learning" in KINDS
        return
    raise AssertionError("expected RefuseError")


def test_oracle_mean_is_leakage():
    try:
        compress("latent_dynamics", {"tm6_ic": [0.8, 0.9]}, oracle_mean=1.12)
    except RefuseError as exc:
        assert exc.code == "LEAKAGE"
        assert "oracle_mean" in str(exc)
        return
    raise AssertionError("expected LEAKAGE")


def test_oracle_trace_key_is_leakage():
    try:
        compress("generative_eq", {"oracle_values": [1.0, 1.1]})
    except RefuseError as exc:
        assert exc.code == "LEAKAGE"
        return
    raise AssertionError("expected LEAKAGE")


def test_leakage_keys_ignores_teacher_kwargs():
    assert leakage_keys({"teacher_ns": 20.0, "lag_ps": 200.0}) == []


def test_msm_is_cpu_inference_not_more_md():
    traces = {"cv": [0.8, 0.81, 0.79, 1.5, 1.49, 0.8, 0.82]}
    model = compress("msm", traces)
    assert model["kind"] == "msm"
    assert "observables" in model
    pred = infer(model, n_samples=50, seed=1)
    assert set(pred) == {"cv"}
    assert len(pred["cv"]) == 50
    assert all(isinstance(v, float) for v in pred["cv"])


def test_msm_recovers_two_well_mean():
    # Asymmetric two-state: π(a)≈0.8 → mean ≈ 0.96.
    import random

    a, b = 0.8, 1.6
    p_ab, p_ba = 0.04, 0.16
    rng = random.Random(0)
    x = a
    series = []
    for _ in range(4000):
        u = rng.random()
        if x == a and u < p_ab:
            x = b
        elif x == b and u < p_ba:
            x = a
        series.append(x)
    true_mu = sum(series) / len(series)
    pred = msm_from_traces({"tm6_ic": series}, lag=1, n_bins=4, n_samples=4000, seed=0)
    mu = sum(pred["tm6_ic"]) / len(pred["tm6_ic"])
    assert abs(true_mu - 0.96) < 0.05
    assert abs(mu - true_mu) < 0.08


def test_msm_teacher_too_short_refuses():
    try:
        msm_from_traces({"cv": [1.0, 1.1]}, lag=5)
    except RefuseError as exc:
        assert exc.code == "COMPRESS"
        return
    raise AssertionError("expected RefuseError")


def test_sparse_bins_are_oracle_candidates():
    series = [0.8] * 80 + [1.6] * 2
    model = compress("msm", {"tm6_ic": series}, n_bins=8)
    unsure = uncertain_regions(model, min_count=10)
    assert unsure["tm6_ic"]


def test_two_start_teacher_is_bridge_not_hungry_bins():
    inactive = [
        0.835, 0.822, 0.788, 0.779, 0.775, 0.825, 0.859, 0.835, 0.851, 0.856,
        0.862, 0.848, 0.836, 0.814, 0.811, 0.917, 0.799, 0.807, 0.809, 0.818,
    ]
    active = [
        1.335, 1.288, 1.348, 1.407, 1.343, 1.427, 1.388, 1.303, 1.313, 1.396,
        1.297, 1.249, 1.262, 1.230, 1.226, 1.104, 1.261, 1.265, 1.295, 1.287,
    ]
    model = compress(
        "msm",
        {"tm6_ic": inactive + active},
        n_bins=8,
        shot_lengths=[20, 20],
        teacher_gpu_hours=0.35,
    )
    spec = model["observables"]["tm6_ic"]
    assert spec["bridge_missing"] is True
    assert spec["disconnected"] is True
    assert spec["cross_hops"] == 0
    assert len(spec["wells"]) == 2
    assert spec["gap"]["width"] > 0
    assert spec["gap"]["lo"] < spec["gap"]["hi"]
    assert model["identification"]["kinetics_identified"] is False
    assert model["identification"]["pi_identified"] is False
    assert "occupancy" in model["identification"]["identified"]
    assert uncertain_regions(model, min_count=5)["tm6_ic"] == []
    try:
        compress(
            "active_learning",
            {"tm6_ic": inactive + active},
            inner="msm",
            n_bins=8,
            shot_lengths=[20, 20],
        )
    except RefuseError as exc:
        assert exc.code == "BRIDGE"
        return
    raise AssertionError("expected BRIDGE")


def test_communicating_two_state_is_not_a_bridge_refuse():
    import random

    rng = random.Random(0)
    x = 0.8
    series = []
    for _ in range(400):
        if x == 0.8 and rng.random() < 0.1:
            x = 1.6
        elif x == 1.6 and rng.random() < 0.1:
            x = 0.8
        series.append(x)
    model = compress("msm", {"tm6_ic": series}, n_bins=4)
    spec = model["observables"]["tm6_ic"]
    assert spec["cross_hops"] > 0
    assert spec["bridge_missing"] is False
    assert model["identification"]["kinetics_identified"] is True

