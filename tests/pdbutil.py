from pathlib import Path


def helix_pdb(path: Path, *, axis: str = "z", n: int = 40, resname: str = "LEU") -> Path:
    """Minimal TM-like helix. axis is the long direction of the CA trace."""
    lines = ["HEADER    TEST"]
    serial = 1
    for i in range(n):
        t = i * 1.5
        x = t if axis == "x" else 0.0
        y = t if axis == "y" else 0.0
        z = t if axis == "z" else 0.0
        for name, dx, dy, dz in (
            ("N", 0.0, 0.0, 0.0),
            ("CA", 0.5, 0.4, 0.0),
            ("C", 1.2, 0.2, 0.3),
            ("O", 2.0, 0.2, 0.3),
        ):
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} {resname} A{i + 1:4d}    "
                f"{x + dx:8.3f}{y + dy:8.3f}{z + dz:8.3f}  1.00  0.00           {name[:1]}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path
