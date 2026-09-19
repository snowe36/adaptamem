from adaptamem.errors import RefuseError
from adaptamem.lipids import integer_counts, leaflet_picks, plan_mix


def test_pope_popg_mix_is_same_physics():
    plan = plan_mix({"POPE": 0.75, "POPG": 0.25}, n_lipid=100, lateral_nm=10.0)
    assert plan.scaffold == "POPE"
    assert plan.swap_to == "POPG"
    assert plan.n_swap == 25
    assert plan.counts is not None
    assert plan.counts["POPE"] + plan.counts["POPG"] == 100
    assert plan.physics == "same_physics"


def test_membrane_environment_popg_mix_does_not_refuse():
    plan = plan_mix({"POPE": 3, "POPG": 1}, n_lipid=80, lateral_nm=12.0)
    assert plan.n_swap == 20


def test_popc_pope_mix_not_implemented():
    try:
        plan_mix({"POPC": 0.7, "POPE": 0.3})
    except RefuseError as exc:
        assert "not implemented" in exc.message
        return
    raise AssertionError("expected RefuseError")


def test_pure_popg_needs_pope_scaffold():
    try:
        plan_mix({"POPG": 1.0})
    except RefuseError as exc:
        assert "scaffold" in exc.message.lower() or "POPG" in exc.message
        return
    raise AssertionError("expected RefuseError")


def test_charged_tiny_box_refuses():
    try:
        plan_mix({"POPE": 0.7, "POPG": 0.3}, n_lipid=40, lateral_nm=5.0)
    except RefuseError as exc:
        assert "charged" in exc.message.lower() or "electrostatic" in exc.message.lower()
        return
    raise AssertionError("expected RefuseError")


def test_integer_counts_sum():
    c = integer_counts({"A": 1 / 3, "B": 2 / 3}, 10)
    assert sum(c.values()) == 10


def test_leaflet_picks_balanced():
    u, lo = leaflet_picks(20, 20, 10, seed=1)
    assert len(u) + len(lo) == 10
    u2, lo2 = leaflet_picks(20, 20, 10, seed=1)
    assert u == u2 and lo == lo2
