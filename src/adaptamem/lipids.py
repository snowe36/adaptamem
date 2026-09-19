"""Lipid mix planning. OpenMM addMembrane is single-species; mixes are a post-build swap."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.sim import OPENMM_LIPIDS
from adaptamem.strategy import PHYSICS_SAME

# Not an addMembrane type; built by swapping a POPE (or POP) scaffold.
SWAP_ONLY = {"POPG"}
CHARGED = {"POPG", "POPS", "POPA", "CL", "CDL"}
MIN_LATERAL_NM_CHARGED = 7.0
_TEMPLATE = Path(__file__).parent / "resources" / "lipids" / "popg_charmm36.json"


@dataclass
class MixPlan:
    scaffold: str
    fractions: dict[str, float]
    counts: dict[str, int] | None
    n_swap: int
    swap_to: str | None
    physics: str
    notes: list[str] = field(default_factory=list)


def normalize_fractions(lipids: dict[str, float]) -> dict[str, float]:
    present = {k.upper(): float(v) for k, v in lipids.items() if v > 0}
    if not present:
        raise RefuseError("membrane.lipids is empty")
    total = sum(present.values())
    return {k: v / total for k, v in present.items()}


def integer_counts(fractions: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder allocation; counts sum to n."""
    if n < 1:
        raise RefuseError("need at least one lipid")
    raw = {k: frac * n for k, frac in fractions.items()}
    counts = {k: int(v) for k, v in raw.items()}
    leftover = n - sum(counts.values())
    order = sorted(raw, key=lambda k: (raw[k] - counts[k]), reverse=True)
    i = 0
    while leftover > 0 and order:
        counts[order[i % len(order)]] += 1
        leftover -= 1
        i += 1
    return counts


def plan_mix(
    lipids: dict[str, float],
    *,
    n_lipid: int | None = None,
    lateral_nm: float | None = None,
) -> MixPlan:
    frac = normalize_fractions(lipids)
    charged_frac = sum(v for k, v in frac.items() if k in CHARGED)
    if charged_frac >= 0.15 and lateral_nm is not None and lateral_nm < MIN_LATERAL_NM_CHARGED:
        raise RefuseError(
            f"charged lipid fraction {charged_frac:.2f} in a {lateral_nm:g} nm box; "
            f"need ≥{MIN_LATERAL_NM_CHARGED:g} nm lateral (PBC electrostatics)"
        )

    names = set(frac)
    scaffold_candidates = [k for k in frac if k in OPENMM_LIPIDS]
    if not scaffold_candidates:
        raise RefuseError(
            f"no addMembrane scaffold in {sorted(names)}; "
            f"use one of {', '.join(OPENMM_LIPIDS)} (POPG is a POPE swap)"
        )
    scaffold = max(scaffold_candidates, key=lambda k: frac[k])

    extra = names - {scaffold}
    notes: list[str] = []
    swap_to: str | None = None
    n_swap = 0
    counts = integer_counts(frac, n_lipid) if n_lipid is not None else None

    if extra:
        if extra <= SWAP_ONLY and scaffold == "POPE":
            swap_to = "POPG"
            if counts is not None:
                n_swap = int(counts.get("POPG", 0))
            notes.append(f"build 100% {scaffold}, swap {swap_to} after addMembrane")
        elif extra <= SWAP_ONLY:
            raise RefuseError("POPG mix requires a POPE scaffold (addMembrane has no POPG patch)")
        else:
            raise RefuseError(
                f"lipid mix {frac} is not implemented; Phase 1 swap is POPE:POPG only"
            )
    else:
        notes.append(f"single-component {scaffold}")

    if n_lipid is not None and counts is not None:
        notes.append("counts " + " ".join(f"{k}:{counts[k]}" for k in sorted(counts)))

    return MixPlan(
        scaffold=scaffold,
        fractions=frac,
        counts=counts,
        n_swap=n_swap,
        swap_to=swap_to,
        physics=PHYSICS_SAME,
        notes=notes,
    )


def leaflet_picks(n_upper: int, n_lower: int, n_swap: int, seed: int = 42) -> tuple[list[int], list[int]]:
    """Deterministic indices to convert in each leaflet (upper, lower)."""
    if n_swap <= 0:
        return [], []
    n_u = n_swap // 2
    n_l = n_swap - n_u
    n_u = min(n_u, n_upper)
    n_l = min(n_l, n_lower)
    rng_u = _lcg(seed)
    rng_l = _lcg(seed + 1)
    upper = _sample_indices(n_upper, n_u, rng_u)
    lower = _sample_indices(n_lower, n_l, rng_l)
    return upper, lower


def _lcg(seed: int):
    x = seed % 2147483647 or 1

    def nxt() -> float:
        nonlocal x
        x = (1103515245 * x + 12345) % 2147483647
        return x / 2147483647

    return nxt


def _sample_indices(n: int, k: int, rng) -> list[int]:
    idx = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rng() * (i + 1))
        idx[i], idx[j] = idx[j], idx[i]
    return sorted(idx[:k])


def load_popg_template() -> dict[str, Any]:
    return json.loads(_TEMPLATE.read_text())


def popg_drop_names() -> set[str]:
    return set(load_popg_template()["drop_from_pope"])


def popg_unique_heavies() -> list[str]:
    return list(load_popg_template()["unique_heavies"])


def apply_pope_popg_swap(modeller: Any, n_swap: int, *, seed: int = 42) -> dict[str, Any]:
    """Leaflet-balanced POPE/POP → CHARMM36 POPG (full unique headgroup)."""
    from openmm import Vec3, unit
    from openmm.app import Topology as OmTopology
    from openmm.app.element import Element

    if n_swap <= 0:
        return {"n_swapped": 0}

    tmpl = load_popg_template()
    drop = set(tmpl["drop_from_pope"])
    unique = list(tmpl["unique_heavies"]) + list(tmpl["unique_hydrogens"])
    local = {k: tuple(v) for k, v in tmpl["local_nm"].items()}

    def _pz(res) -> float:
        for a in res.atoms():
            if a.name.strip() == "P":
                return modeller.positions[a.index].value_in_unit(unit.nanometer)[2]
        return 0.0

    lipids = [r for r in modeller.topology.residues() if r.name in {"POP", "POPE"}]
    upper = [r for r in lipids if _pz(r) >= 0]
    lower = [r for r in lipids if _pz(r) < 0]
    ui, li = leaflet_picks(len(upper), len(lower), n_swap, seed=seed)
    pick = [upper[i] for i in ui] + [lower[i] for i in li]
    if not pick:
        return {"n_swapped": 0, "error": "no POPE/POP lipids to swap"}

    new_top = OmTopology()
    chain = new_top.addChain()
    new_pos: list[Any] = []
    built = 0
    placed_unique = 0
    for i, res in enumerate(pick):
        by_name = {a.name.strip(): a for a in res.atoms()}
        rnew = new_top.addResidue("POPG", chain, id=str(i + 1))
        for a in res.atoms():
            if a.name.strip() in drop:
                continue
            new_top.addAtom(a.name, a.element, rnew)
            new_pos.append(modeller.positions[a.index])
        have = {a.name.strip() for a in res.atoms()} - drop
        lab = {}
        if all(k in by_name for k in ("C11", "C12", "P")):
            lab = place_popg_unique(
                _xyz(modeller, by_name["C11"]),
                _xyz(modeller, by_name["C12"]),
                _xyz(modeller, by_name["P"]),
                local,
            )
        for name in unique:
            if name in have:
                continue
            xyz = lab.get(name)
            if xyz is None:
                continue
            new_top.addAtom(name, Element.getBySymbol(name[0]), rnew)
            new_pos.append(Vec3(*xyz) * unit.nanometer)
            placed_unique += 1
        built += 1

    modeller.add(new_top, new_pos)
    modeller.delete(pick)
    return {
        "n_swapped": built,
        "n_upper": len(ui),
        "n_lower": len(li),
        "n_unique_atoms": placed_unique,
        "template": str(_TEMPLATE.name),
    }


def place_popg_unique(
    c11: tuple[float, float, float],
    c12: tuple[float, float, float],
    p: tuple[float, float, float],
    local_nm: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Unique POPG atoms in the C11–C12–P frame (nm)."""
    if local_nm is None:
        local_nm = {k: tuple(v) for k, v in load_popg_template()["local_nm"].items()}
    e1, e2, e3 = _frame(c11, c12, p)
    out: dict[str, tuple[float, float, float]] = {}
    for name, xyz in local_nm.items():
        out[name] = (
            c12[0] + xyz[0] * e1[0] + xyz[1] * e2[0] + xyz[2] * e3[0],
            c12[1] + xyz[0] * e1[1] + xyz[1] * e2[1] + xyz[2] * e3[1],
            c12[2] + xyz[0] * e1[2] + xyz[1] * e2[2] + xyz[2] * e3[2],
        )
    return out


def _frame(
    c11: tuple[float, float, float],
    c12: tuple[float, float, float],
    p: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    e1 = _norm3(_sub3(c12, c11))
    e3 = _cross3(e1, _sub3(p, c12))
    if _dot3(e3, e3) < 1e-12:
        helper = (0.0, 0.0, 1.0) if abs(e1[2]) < 0.9 else (1.0, 0.0, 0.0)
        e3 = _cross3(e1, helper)
    e3 = _norm3(e3)
    e2 = _norm3(_cross3(e3, e1))
    return e1, e2, e3


def _xyz(modeller: Any, atom: Any) -> tuple[float, float, float]:
    from openmm import unit

    p = modeller.positions[atom.index].value_in_unit(unit.nanometer)
    return (float(p[0]), float(p[1]), float(p[2]))


def _popg_unique_heavies(
    c11: tuple[float, float, float], c12: tuple[float, float, float]
) -> dict[str, tuple[float, float, float]]:
    # Kept for tests that called the old stub placer.
    p = _add3(c11, (0.0, 0.0, 0.3))
    placed = place_popg_unique(c11, c12, p)
    return {k: placed[k] for k in ("C13", "OC2", "OC3") if k in placed}


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm3(a):
    n = _dot3(a, a) ** 0.5
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)
