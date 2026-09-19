"""Lipid mix planning. OpenMM addMembrane is single-species; mixes are a post-build swap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.sim import OPENMM_LIPIDS
from adaptamem.strategy import PHYSICS_SAME

# Not an addMembrane type; built by swapping a POPE (or POP) scaffold.
SWAP_ONLY = {"POPG"}
CHARGED = {"POPG", "POPS", "POPA", "CL", "CDL"}
MIN_LATERAL_NM_CHARGED = 7.0


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


def apply_pope_popg_swap(modeller: Any, n_swap: int, *, seed: int = 42) -> dict[str, Any]:
    """Leaflet-balanced POPE/POP → POPG: copy shared atoms, grow C13/OC2/OC3, delete parents."""
    from openmm import Vec3, unit
    from openmm.app import Topology as OmTopology
    from openmm.app.element import Element

    if n_swap <= 0:
        return {"n_swapped": 0}

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
    for i, res in enumerate(pick):
        by_name = {a.name.strip(): a for a in res.atoms()}
        rnew = new_top.addResidue("POPG", chain, id=str(i + 1))
        for a in res.atoms():
            new_top.addAtom(a.name, a.element, rnew)
            new_pos.append(modeller.positions[a.index])
        if "C11" in by_name and "C12" in by_name:
            placed = _popg_unique_heavies(_xyz(modeller, by_name["C11"]), _xyz(modeller, by_name["C12"]))
            have = {a.name.strip() for a in res.atoms()}
            for name, xyz in placed.items():
                if name in have:
                    continue
                new_top.addAtom(name, Element.getBySymbol(name[0]), rnew)
                new_pos.append(Vec3(*xyz) * unit.nanometer)
        built += 1

    modeller.add(new_top, new_pos)
    modeller.delete(pick)
    return {"n_swapped": built, "n_upper": len(ui), "n_lower": len(li)}


def _xyz(modeller: Any, atom: Any) -> tuple[float, float, float]:
    from openmm import unit

    p = modeller.positions[atom.index].value_in_unit(unit.nanometer)
    return (float(p[0]), float(p[1]), float(p[2]))


def _popg_unique_heavies(
    c11: tuple[float, float, float], c12: tuple[float, float, float]
) -> dict[str, tuple[float, float, float]]:
    ax = _norm3(_sub3(c12, c11))
    # Arbitrary perpendicular for a glycerol stub (~1.5 Å bonds).
    helper = (1.0, 0.0, 0.0) if abs(ax[0]) < 0.9 else (0.0, 1.0, 0.0)
    n1 = _norm3(_cross3(ax, helper))
    n2 = _norm3(_cross3(ax, n1))
    c13 = _add3(c12, _scale3(ax, 0.15))
    oc2 = _add3(c13, _scale3(n1, 0.14))
    oc3 = _add3(c13, _scale3(n2, 0.14))
    return {"C13": c13, "OC2": oc2, "OC3": oc3}


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm3(a):
    n = _dot3(a, a) ** 0.5
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)
