"""A few established β2AR activation distances. Not a feature dump."""

from __future__ import annotations

from adaptamem.objective import Observable

# Ballesteros–Weinstein: 3.50–6.34 (TM6 IC), 3.50–6.30 (ionic lock),
# 3.40–6.44 (PIF packing), 7.53–3.50 (NPxxY vs DRY).
SWITCHES: tuple[Observable, ...] = (
    Observable(
        name="tm6_ic",
        kind="distance",
        selection="name CA and resid 131 ; name CA and resid 272",
        precision=0.2,
    ),
    Observable(
        name="ionic_lock",
        kind="distance",
        selection="name CA and resid 131 ; name CA and resid 268",
        precision=0.2,
    ),
    Observable(
        name="tm3_tm6_pack",
        kind="distance",
        selection="name CA and resid 121 ; name CA and resid 282",
        precision=0.2,
    ),
    Observable(
        name="npxxY",
        kind="distance",
        selection="name CA and resid 326 ; name CA and resid 131",
        precision=0.2,
    ),
)
