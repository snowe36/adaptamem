import json
from pathlib import Path

from adaptamem.box import plan_box
from adaptamem.doctor import audit
from adaptamem.errors import RefuseError
from adaptamem.objective import parse_objective
from adaptamem.sample import mean_ci, schedule, stopped, write_schedule
from adaptamem.strategy import PHYSICS_SAME

from pdbutil import helix_pdb


def test_conventional_is_one_walker(tmp_path: Path):
    rep = audit(helix_pdb(tmp_path / "leu.pdb"))
    box = plan_box(rep)
    s = schedule(parse_objective({"type": "conventional"}), box)
    assert s.n_walkers == 1
    assert s.n_pilot == 0
    assert s.physics == PHYSICS_SAME


def test_discover_states_outputs_walkers(tmp_path: Path):
    rep = audit(helix_pdb(tmp_path / "leu.pdb"))
    box = plan_box(rep)
    s = schedule(parse_objective({"type": "discover_states"}), box, budget_hours=48)
    assert s.n_pilot >= 1
    assert s.n_walkers >= 2
    assert s.physics == PHYSICS_SAME


def test_mean_ci_and_stop():
    mu, half = mean_ci([1.0, 1.0, 1.0, 1.0])
    assert abs(mu - 1.0) < 1e-9
    assert half < 0.1
    assert stopped([1.0, 1.02, 0.99, 1.01], precision=0.5)
    assert not stopped([0.0, 10.0], precision=0.1)


def test_traces_drop_converged_observable(tmp_path: Path):
    rep = audit(helix_pdb(tmp_path / "leu.pdb"))
    box = plan_box(rep)
    obj = parse_objective(
        {
            "type": "conformational_shift",
            "observables": [
                {"name": "rmsd", "kind": "rmsd", "selection": "name CA", "precision": 0.5}
            ],
        }
    )
    s = schedule(obj, box, traces={"rmsd": [1.0, 1.01, 0.99, 1.0]})
    assert "inside precision" in " ".join(s.notes)


def test_tiny_budget_refuses(tmp_path: Path):
    rep = audit(helix_pdb(tmp_path / "leu.pdb"))
    box = plan_box(rep)
    try:
        schedule(parse_objective({"type": "discover_states"}), box, budget_hours=0.01)
    except RefuseError as exc:
        assert "budget" in exc.message.lower()
        return
    raise AssertionError("expected RefuseError")


def test_write_sample_json(tmp_path: Path):
    rep = audit(helix_pdb(tmp_path / "leu.pdb"))
    s = schedule(parse_objective({"type": "discover_states"}), plan_box(rep))
    path = write_schedule(s, tmp_path)
    data = json.loads(path.read_text())
    assert data["n_walkers"] == s.n_walkers
    assert data["physics"] == PHYSICS_SAME
