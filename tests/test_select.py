from pathlib import Path

from adaptamem.box import plan_box
from adaptamem.doctor import audit
from adaptamem.errors import RefuseError
from adaptamem.objective import parse_objective
from adaptamem.sample import sample, schedule
from adaptamem.select import KEEP, STOP, SelectContext, get_policy

from pdbutil import helix_pdb


def test_loop_keys_in_sample_json(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "leu.pdb")
    session_box = plan_box(audit(pdb))
    obj = parse_objective({"type": "discover_states"})
    r = sample(obj, session_box, tmp_path / "run", budget_hours=48, select="random")
    assert r.loop[0] == "initialize" and r.loop[-1] == "stop"
    assert r.select == "random"
    data = __import__("json").loads(r.path.read_text())
    assert data["select"] == "random"
    assert data["n_pilot"] >= 1


def test_coverage_keeps_unique_tight_ci():
    obj = parse_objective(
        {
            "type": "conformational_shift",
            "observables": [
                {"name": "cv", "kind": "distance", "selection": "a ; b", "precision": 0.5}
            ],
        }
    )
    walkers = {
        "wA": {"cv": [1.0] * 20},
        "wB": {"cv": [8.0] * 20},
    }
    dec = get_policy("coverage")(SelectContext(walkers=walkers, observables=obj.observables))
    by = {d.walker_id: d for d in dec}
    # each occupies a private bin → unique even if CI is tight
    assert by["wA"].unique and by["wB"].unique
    assert by["wA"].action == KEEP
    assert by["wB"].action == KEEP


def test_coverage_stops_redundant_tight_ci():
    obj = parse_objective(
        {
            "type": "conformational_shift",
            "observables": [
                {"name": "cv", "kind": "distance", "selection": "a ; b", "precision": 0.5}
            ],
        }
    )
    walkers = {
        "wA": {"cv": [1.0, 1.01, 0.99] * 10},
        "wB": {"cv": [1.01, 0.99, 1.0] * 10},
    }
    dec = get_policy("coverage")(SelectContext(walkers=walkers, observables=obj.observables))
    assert all(d.action == STOP for d in dec)
    assert all(not d.unique for d in dec)


def test_sample_refuses_one_well_ci(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "leu.pdb")
    box = plan_box(audit(pdb))
    obj = parse_objective(
        {
            "type": "conformational_shift",
            "observables": [
                {"name": "cv", "kind": "distance", "selection": "a ; b", "precision": 0.01}
            ],
        }
    )
    try:
        sample(
            obj,
            box,
            tmp_path / "run",
            traces={"cv": [1.0] * 30},
            select="coverage",
            budget_hours=12,
        )
    except RefuseError as exc:
        assert exc.code == "COVERAGE"
        assert "metastable" in exc.message.lower()
        return
    raise AssertionError("expected COVERAGE refuse")


def test_schedule_still_plans_walkers(tmp_path: Path):
    box = plan_box(audit(helix_pdb(tmp_path / "leu.pdb")))
    s = schedule(parse_objective({"type": "discover_states"}), box, budget_hours=48)
    assert s.n_pilot >= 1
    assert s.n_walkers >= 2
