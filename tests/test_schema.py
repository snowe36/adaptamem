from adaptamem.box import plan_box
from adaptamem.doctor import audit
from adaptamem.objective import parse_objective
from adaptamem.schema import load_protocol, load_system

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_default_protocol_eq_matches_production():
    p = load_protocol()
    assert p.timestep_fs == 4.0
    assert p.eq_timestep_fs == 4.0


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


def _poly_leu_pdb(path: Path) -> Path:
    lines = ["HEADER    TEST"]
    n = 40
    serial = 1
    for i in range(n):
        x, y, z = 0.0, 0.0, i * 1.5
        for name, dx, dy in (("N", 0.0, 0.0), ("CA", 1.5, 0.0), ("C", 2.5, 0.8), ("O", 3.5, 0.8)):
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} LEU A{i+1:4d}    "
                f"{x+dx:8.3f}{y+dy:8.3f}{z:8.3f}  1.00  0.00           {name[:1]}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_doctor_finds_tm_and_box(tmp_path: Path):
    pdb = _poly_leu_pdb(tmp_path / "leu.pdb")
    rep = audit(pdb)
    assert rep.n_residues == 40
    assert rep.n_tm >= 1
    box = plan_box(rep)
    assert box.n_lipids >= 40
    assert box.est_atoms > rep.n_protein_atoms
