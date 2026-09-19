"""YAML collective variables. Evaluated from positions, not by re-reading XTC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.objective import Observable


@dataclass
class AtomRef:
    index: int
    name: str
    resid: int
    chain: str = "A"
    resname: str = ""


def atoms_from_topology(topology: Any) -> list[AtomRef]:
    out: list[AtomRef] = []
    for i, atom in enumerate(topology.atoms()):
        res = atom.residue
        chain = getattr(res, "chain", None)
        chain_id = getattr(chain, "id", "A") if chain is not None else "A"
        resid = int(getattr(res, "id", getattr(res, "index", i)) or 0)
        try:
            resid = int(res.id)
        except (TypeError, ValueError):
            resid = int(res.index) + 1
        out.append(
            AtomRef(
                index=int(getattr(atom, "index", i)),
                name=str(atom.name).strip(),
                resid=resid,
                chain=str(chain_id).strip() or "A",
                resname=str(res.name).strip(),
            )
        )
    return out


def _truth(atom: AtomRef, token: str) -> bool:
    t = token.strip()
    if not t:
        return True
    parts = t.split(None, 1)
    key = parts[0].lower()
    val = parts[1].strip() if len(parts) > 1 else ""
    if key == "name":
        return atom.name == val
    if key in {"resid", "resi"}:
        if "-" in val:
            a, b = val.split("-", 1)
            return int(a) <= atom.resid <= int(b)
        return atom.resid == int(val)
    if key == "chain":
        return atom.chain == val
    if key == "resname":
        return atom.resname == val
    raise RefuseError(f"unknown selection token {token!r}", code="NOT_IMPLEMENTED")


def select_atoms(atoms: list[AtomRef], selection: str) -> list[AtomRef]:
    sel = selection.strip()
    if not sel:
        return list(atoms)
    picked = atoms
    for clause in sel.split(" and "):
        picked = [a for a in picked if _truth(a, clause)]
    return picked


def _groups(selection: str) -> list[str]:
    if ";" in selection:
        return [p.strip() for p in selection.split(";") if p.strip()]
    return [selection.strip()]


def _com(indices: list[int], xyz: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not indices:
        raise RefuseError("selection matched no atoms", code="NOT_READY")
    sx = sy = sz = 0.0
    for i in indices:
        x, y, z = xyz[i]
        sx += x
        sy += y
        sz += z
    n = float(len(indices))
    return (sx / n, sy / n, sz / n)


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _rmsd(
    mobile: list[tuple[float, float, float]], ref: list[tuple[float, float, float]]
) -> float:
    if len(mobile) != len(ref) or not mobile:
        raise RefuseError("rmsd selection/reference size mismatch", code="NOT_READY")
    mx = sum(p[0] for p in mobile) / len(mobile)
    my = sum(p[1] for p in mobile) / len(mobile)
    mz = sum(p[2] for p in mobile) / len(mobile)
    rx = sum(p[0] for p in ref) / len(ref)
    ry = sum(p[1] for p in ref) / len(ref)
    rz = sum(p[2] for p in ref) / len(ref)
    acc = 0.0
    for m, r in zip(mobile, ref, strict=True):
        dx = (m[0] - mx) - (r[0] - rx)
        dy = (m[1] - my) - (r[1] - ry)
        dz = (m[2] - mz) - (r[2] - rz)
        acc += dx * dx + dy * dy + dz * dz
    return (acc / len(mobile)) ** 0.5


def evaluate(
    obs: Observable,
    atoms: list[AtomRef],
    xyz: list[tuple[float, float, float]],
    *,
    reference: list[tuple[float, float, float]] | None = None,
) -> float:
    kind = obs.kind.lower()
    groups = _groups(obs.selection)
    if kind == "distance":
        if len(groups) != 2:
            raise RefuseError(
                f"{obs.name}: distance needs two selections separated by ';'",
                code="NOT_READY",
            )
        a = [x.index for x in select_atoms(atoms, groups[0])]
        b = [x.index for x in select_atoms(atoms, groups[1])]
        return _dist(_com(a, xyz), _com(b, xyz))
    if kind == "rmsd":
        idxs = [x.index for x in select_atoms(atoms, groups[0])]
        if not idxs:
            raise RefuseError(f"{obs.name}: rmsd selection matched nothing", code="NOT_READY")
        mob = [xyz[i] for i in idxs]
        ref = reference
        if ref is None:
            return 0.0
        if len(ref) == len(xyz):
            ref = [ref[i] for i in idxs]
        return _rmsd(mob, ref)
    raise RefuseError(f"observable kind {obs.kind!r} is not implemented", code="NOT_IMPLEMENTED")


def positions_nm(positions: Any) -> list[tuple[float, float, float]]:
    from openmm import unit

    raw = positions.value_in_unit(unit.nanometer)
    return [(float(p[0]), float(p[1]), float(p[2])) for p in raw]
