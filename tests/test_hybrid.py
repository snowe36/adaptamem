from pathlib import Path

from adaptamem.box import plan_box
from adaptamem.doctor import audit
from adaptamem.errors import RefuseError
from adaptamem.hybrid import plan_hybrid
from adaptamem.objective import parse_objective
from adaptamem.strategy import PHYSICS_APPROX, choose

from pdbutil import helix_pdb


def _strat(tmp_path: Path, obj_type: str = "discover_states"):
    rep = audit(helix_pdb(tmp_path / "leu.pdb"))
    box = plan_box(rep)
    obj = parse_objective({"type": obj_type, "discover_cvs": True})
    return obj, choose(rep, obj, box)


def test_hybrid_cg_is_approximation(tmp_path: Path):
    obj, strat = _strat(tmp_path)
    plan = plan_hybrid(obj, strat, bulk="CG")
    assert plan.ok
    assert plan.physics == PHYSICS_APPROX
    assert plan.protein == "AA"
    assert plan.annular == "AA"


def test_membrane_environment_keeps_annular_aa(tmp_path: Path):
    obj, strat = _strat(tmp_path, "membrane_environment")
    plan = plan_hybrid(obj, strat, bulk="CG")
    assert plan.annular == "AA"
    assert any("annular" in n.lower() for n in plan.notes)


def test_implicit_membrane_environment_refuses(tmp_path: Path):
    obj, strat = _strat(tmp_path, "membrane_environment")
    try:
        plan_hybrid(obj, strat, bulk="implicit")
    except RefuseError as exc:
        assert "implicit" in exc.message.lower() or "U_membrane" in exc.message
        return
    raise AssertionError("expected RefuseError")


def test_full_aa_hybrid_not_applied(tmp_path: Path):
    obj, strat = _strat(tmp_path)
    plan = plan_hybrid(obj, strat, bulk="AA")
    assert plan.bulk_membrane == "AA"
    assert plan.water == "AA"
