from adaptamem.objective import Observable
from adaptamem.observables import AtomRef, evaluate, select_atoms


def _helix_atoms(n: int = 8) -> tuple[list[AtomRef], list[tuple[float, float, float]]]:
    atoms = []
    xyz = []
    for i in range(n):
        atoms.append(AtomRef(index=i, name="CA", resid=i + 1, chain="A", resname="LEU"))
        xyz.append((0.0, 0.0, float(i) * 0.15))
    return atoms, xyz


def test_distance_end_to_end():
    atoms, xyz = _helix_atoms()
    obs = Observable(
        name="ee",
        kind="distance",
        selection="resid 1 and name CA ; resid 8 and name CA",
    )
    d = evaluate(obs, atoms, xyz)
    assert abs(d - 1.05) < 1e-9


def test_rmsd_zero_vs_self():
    atoms, xyz = _helix_atoms()
    obs = Observable(name="r", kind="rmsd", selection="name CA")
    assert evaluate(obs, atoms, xyz, reference=xyz) == 0.0


def test_rmsd_shift():
    atoms, xyz = _helix_atoms()
    moved = [(x, y, z + 0.2) for x, y, z in xyz]
    obs = Observable(name="r", kind="rmsd", selection="name CA")
    # centroid-aligned; uniform z shift → 0
    assert evaluate(obs, atoms, moved, reference=xyz) < 1e-9


def test_select_resid_range():
    atoms, _xyz = _helix_atoms()
    picked = select_atoms(atoms, "resid 2-4 and name CA")
    assert [a.resid for a in picked] == [2, 3, 4]
