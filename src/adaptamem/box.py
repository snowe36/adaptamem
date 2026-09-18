"""Cheapest sufficient box from geometry — not a fixed 10×10 nm membrane."""

from __future__ import annotations

from dataclasses import dataclass

from adaptamem.doctor import DoctorReport

APL_NM2 = 0.65
WATER_NM3 = 0.030  # ~30 Å³ / molecule
HYDROPHOBIC_THICKNESS_NM = 3.2
ATOMS_PER_LIPID = 130  # POPC-ish
ATOMS_PER_WATER = 3


@dataclass
class BoxPlan:
    lateral_nm: float
    z_nm: float
    n_lipids: int
    n_waters: int
    est_atoms: int
    safety_margin_nm: float
    water_pad_nm: float
    notes: str


def plan_box(
    report: DoctorReport,
    *,
    water_pad_nm: float = 1.2,
    safety_margin_nm: float = 1.2,
) -> BoxPlan:
    if report.bbox_nm is None:
        raise ValueError("no coordinates to size a box")
    sx, sy, sz = report.bbox_nm
    # TM proteins: bilayer in XY. Use the two larger extents as lateral.
    dims = sorted((sx, sy, sz), reverse=True)
    lat = dims[0] + 2 * safety_margin_nm
    z = HYDROPHOBIC_THICKNESS_NM + 2 * water_pad_nm
    if not report.tm_spans:
        # soluble-looking: cubic-ish pad
        lat = max(dims[0], dims[1]) + 2 * safety_margin_nm
        z = sz + 2 * water_pad_nm
        notes = "no TM spans; sized as a padded droplet, not a bilayer"
    else:
        notes = f"{report.n_tm} TM spans; lateral pad {safety_margin_nm:g} nm, water {water_pad_nm:g} nm"
    area = lat * lat
    n_lipids = max(40, int(round(2 * area / APL_NM2)))
    n_waters = max(200, int(round(area * 2 * water_pad_nm / WATER_NM3)))
    est = report.n_protein_atoms + n_lipids * ATOMS_PER_LIPID + n_waters * ATOMS_PER_WATER
    # hydrogens roughly double protein heavy count if input lacks H
    est = int(est * 1.4)
    return BoxPlan(
        lateral_nm=round(lat, 2),
        z_nm=round(z, 2),
        n_lipids=n_lipids,
        n_waters=n_waters,
        est_atoms=est,
        safety_margin_nm=safety_margin_nm,
        water_pad_nm=water_pad_nm,
        notes=notes,
    )


def format_box(plan: BoxPlan) -> str:
    return (
        f"BOX PLAN  {plan.lateral_nm:g} × {plan.lateral_nm:g} × {plan.z_nm:g} nm\n"
        f"~{plan.n_lipids} lipids  ~{plan.n_waters} waters  ~{plan.est_atoms} atoms\n"
        f"{plan.notes}"
    )
