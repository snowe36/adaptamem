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
    """Minimal TM-like helix. axis is the long direction of the CA trace."""
    atoms = _LEU if resname == "LEU" else _LYS_BB
    lines = ["HEADER    TEST"]
    serial = 1
    for i in range(n):
        t = i * 1.5
        ox = t if axis == "x" else 0.0
        oy = t if axis == "y" else 0.0
        oz = t if axis == "z" else 0.0
        for name, dx, dy, dz in atoms:
            elem = name[0]
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} {resname} A{i + 1:4d}    "
                f"{ox + dx:8.3f}{oy + dy:8.3f}{oz + dz:8.3f}  1.00  0.00           {elem}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path
