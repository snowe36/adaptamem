from pathlib import Path

from adaptamem.box import plan_box
from adaptamem.doctor import audit
from adaptamem.objective import parse_objective
from adaptamem.schema import load_protocol, load_system
from adaptamem.strategy import PHYSICS_SAME, choose

from pdbutil import helix_pdb

ROOT = Path(__file__).resolve().parents[1]


def test_default_protocol_eq_matches_production():
    p = load_protocol()
    assert p.timestep_fs == 4.0
    assert p.eq_timestep_fs == 4.0
    assert p.hydrogen_mass_amu == 4.0


def test_example_recipes_load():
    dltb = load_system(ROOT / "examples" / "dltb.yaml")
    assert dltb.objective.type == "discover_states"
    assert dltb.objective.discover_cvs
    assert dltb.membrane.optimize_size
    b2ar = load_system(ROOT / "examples" / "b2ar.yaml")
    assert b2ar.objective.type == "conformational_shift"


def test_objective_requires_observables_or_discovery():
    try:
        parse_objective({"type": "conformational_shift"})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_doctor_finds_tm_and_box(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "leu.pdb")
    rep = audit(pdb)
    assert rep.n_residues == 40
    assert rep.n_tm >= 1
    box = plan_box(rep)
    assert box.n_lipids >= 40
    assert box.est_atoms > rep.n_protein_atoms
    obj = parse_objective({"type": "discover_states"})
    strat = choose(rep, obj, box)
    assert strat.ok
    assert strat.physics == PHYSICS_SAME
    assert strat.sampling == "adaptive"


def test_strategy_refuses_without_tm(tmp_path: Path):
    lines = ["HEADER    SOLUBLE"]
    serial = 1
    for i in range(12):
        z = i * 1.5
        for name, dx in (("N", 0.0), ("CA", 1.5), ("C", 2.5)):
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} LYS A{i+1:4d}    "
                f"{dx:8.3f}{0.0:8.3f}{z:8.3f}  1.00  0.00           {name[:1]}"
            )
            serial += 1
    lines.append("END")
    pdb = tmp_path / "lys.pdb"
    pdb.write_text("\n".join(lines) + "\n")
    rep = audit(pdb)
    box = plan_box(rep)
    strat = choose(rep, parse_objective({"type": "conventional"}), box)
    assert not strat.ok
    assert "transmembrane" in (strat.refuse or "")


def test_membrane_environment_keeps_annular_lipids(tmp_path: Path):
    pdb = helix_pdb(tmp_path / "leu.pdb")
    rep = audit(pdb)
    box = plan_box(rep)
    obj = parse_objective({"type": "membrane_environment", "discover_cvs": True})
    strat = choose(rep, obj, box)
    assert strat.ok
    assert "first-shell" in " ".join(strat.fewer_expensive_atoms)
