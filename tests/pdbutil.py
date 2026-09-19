import math
from pathlib import Path

# Approximate LEU heavy geometry, enough for CHARMM36 template matching.
_LEU = (
    ("N", 0.0, 0.0, 0.0),
    ("CA", 1.46, 0.0, 0.0),
    ("C", 2.0, 1.25, 0.4),
    ("O", 1.4, 2.3, 0.4),
    ("CB", 2.0, -1.25, -0.4),
    ("CG", 1.5, -2.5, 0.2),
    ("CD1", 2.4, -3.6, -0.5),
    ("CD2", 0.05, -2.7, 0.3),
)
_LYS_BB = (
    ("N", 0.0, 0.0, 0.0),
    ("CA", 1.46, 0.0, 0.0),
    ("C", 2.0, 1.25, 0.4),
    ("O", 1.4, 2.3, 0.4),
)


def helix_pdb(path: Path, *, axis: str = "z", n: int = 40, resname: str = "LEU") -> Path:
    """TM poly-Leu: 100 deg/res, 1.5 Å rise, CA ~2.3 Å off the long axis, centered."""
    atoms = _LEU if resname == "LEU" else _LYS_BB
    lines = ["HEADER    TEST"]
    serial = 1
    rise = 1.5
    twist = math.radians(100.0)
    along0 = -0.5 * (n - 1) * rise
    shift = 2.3 - 1.46
    for i in range(n):
        ang = i * twist
        ca, sa = math.cos(ang), math.sin(ang)
        along = along0 + i * rise
        for name, dx, dy, dz in atoms:
            x0, y0 = dx + shift, dy
            xr = x0 * ca - y0 * sa
            yr = x0 * sa + y0 * ca
            za = along + dz
            if axis == "x":
                ox, oy, oz = za, xr, yr
            elif axis == "y":
                ox, oy, oz = yr, za, xr
            else:
                ox, oy, oz = xr, yr, za
            elem = name[0]
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} {resname} A{i + 1:4d}    "
                f"{ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           {elem}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path
