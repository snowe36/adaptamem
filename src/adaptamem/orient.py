"""Put the transmembrane axis on z and the midplane at z = 0.

`auto` uses TM-helix endpoints (Kyte–Doolittle spans). `ppm` runs local immers
and refuses if the binary is missing — it is never silently replaced by auto.
`none` only translates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from adaptamem.doctor import Atom, DoctorReport, load_atoms, select_protein, write_pdb
from adaptamem.errors import RefuseError
from adaptamem.schema import Orientation

Vec = tuple[float, float, float]
Mat = tuple[Vec, Vec, Vec]


@dataclass
class OrientResult:
    path: Path
    method: str
    topology: str
    axis_before: Vec
    span_nm: Vec
    midplane_shift_angstrom: Vec
    n_atoms: int


def orient_structure(
    structure: Path,
    report: DoctorReport,
    orientation: Orientation,
    dest: Path,
) -> OrientResult:
    method = (orientation.method or "auto").lower()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if method == "opm":
        raise RefuseError("orientation.method=opm is not implemented; use ppm or auto")
    if method == "ppm":
        from adaptamem.ppm import run_immers

        raw = dest.with_name(dest.stem + "_immers.pdb")
        run_immers(structure, raw, topology="in")
        atoms = select_protein(load_atoms(raw))
        if not atoms:
            raise RefuseError("PPM immers produced no protein ATOM records")
        R = _eye()
        axis = (0.0, 0.0, 1.0)
        used = "ppm3_local_immers"
    else:
        atoms = select_protein(load_atoms(structure))
        if not atoms:
            raise RefuseError("no protein ATOM records to orient")
        if method in {"none", "file", "given"}:
            R = _eye()
            axis = (0.0, 0.0, 1.0)
            used = "none"
        else:
            axis = _tm_axis(atoms, report.tm_spans, report.tm_ca_angstrom)
            R = _rotation_aligning(axis, (0.0, 0.0, 1.0))
            used = "auto_tm_axis"

    if (orientation.topology or "in").lower() == "out":
        R = _mul(_diag(1.0, 1.0, -1.0), R)

    rotated = [_apply(R, (a.x, a.y, a.z)) for a in atoms]
    mid = _centroid(report.tm_ca_angstrom) if report.tm_ca_angstrom else _centroid(rotated)
    mid_r = _apply(R, mid) if report.tm_ca_angstrom else mid
    shift = (-mid_r[0], -mid_r[1], -mid_r[2])

    out_atoms: list[Atom] = []
    for a, p in zip(atoms, rotated, strict=True):
        x, y, z = p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]
        out_atoms.append(
            Atom(
                serial=a.serial,
                name=a.name,
                alt=" ",
                resname=a.resname,
                chain=a.chain,
                resid=a.resid,
                x=x,
                y=y,
                z=z,
                element=a.element,
                het=a.het,
            )
        )
    write_pdb(
        out_atoms,
        dest,
        remarks=[
            f"adaptamem orient method={used} topology={orientation.topology}",
            f"axis_before={axis[0]:.4f},{axis[1]:.4f},{axis[2]:.4f}",
        ],
    )
    xs = [a.x for a in out_atoms]
    ys = [a.y for a in out_atoms]
    zs = [a.z for a in out_atoms]
    return OrientResult(
        path=dest,
        method=used,
        topology=orientation.topology,
        axis_before=axis,
        span_nm=(
            (max(xs) - min(xs)) / 10.0,
            (max(ys) - min(ys)) / 10.0,
            (max(zs) - min(zs)) / 10.0,
        ),
        midplane_shift_angstrom=shift,
        n_atoms=len(out_atoms),
    )


def format_orient(r: OrientResult) -> str:
    return (
        f"ORIENT  {r.method}  topology={r.topology}\n"
        f"axis was ({r.axis_before[0]:.3f}, {r.axis_before[1]:.3f}, {r.axis_before[2]:.3f})\n"
        f"span after  {r.span_nm[0]:.2f} × {r.span_nm[1]:.2f} × {r.span_nm[2]:.2f} nm\n"
        f"wrote {r.path}"
    )


def _tm_axis(
    atoms: list[Atom],
    spans: list[tuple[str, int, int]],
    tm_ca: list[Vec],
) -> Vec:
    ca = {
        (a.chain, a.resid): (a.x, a.y, a.z)
        for a in atoms
        if a.name == "CA" and a.alt in {"", " ", "A"}
    }
    vecs: list[Vec] = []
    for chain, lo, hi in spans:
        pts = [ca[k] for k in sorted(ca) if k[0] == chain and lo <= k[1] <= hi]
        if len(pts) < 5:
            continue
        vecs.append(_sub(pts[-1], pts[0]))
    if vecs:
        ref = vecs[0]
        acc = (0.0, 0.0, 0.0)
        for v in vecs:
            if _dot(v, ref) < 0:
                v = (-v[0], -v[1], -v[2])
            acc = (acc[0] + v[0], acc[1] + v[1], acc[2] + v[2])
        n = _norm(acc)
        if n > 1e-6:
            return (acc[0] / n, acc[1] / n, acc[2] / n)
    pts = tm_ca or list(ca.values())
    return _principal_normal(pts)


def _principal_normal(points: list[Vec]) -> Vec:
    if len(points) < 3:
        return (0.0, 0.0, 1.0)
    c = _centroid(points)
    cov = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for p in points:
        d = _sub(p, c)
        for i in range(3):
            for j in range(3):
                cov[i][j] += d[i] * d[j]
    evals, evecs = _eigh3(cov)
    # Span along each eigenvector; pick the one closest to hydrophobic thickness.
    target = 32.0  # Å
    best_i = 2
    best_d = 1e9
    for i, ev in enumerate(evecs):
        proj = [_dot(_sub(p, c), ev) for p in points]
        span = max(proj) - min(proj)
        d = abs(span - target)
        if span > 8.0 and d < best_d:
            best_d = d
            best_i = i
    if best_d == 1e9:
        best_i = max(range(3), key=lambda i: evals[i])
    return evecs[best_i]


def _centroid(points: list[Vec]) -> Vec:
    n = max(len(points), 1)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def _eye() -> Mat:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _diag(a: float, b: float, c: float) -> Mat:
    return ((a, 0.0, 0.0), (0.0, b, 0.0), (0.0, 0.0, c))


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vec) -> float:
    return math.sqrt(_dot(a, a))


def _cross(a: Vec, b: Vec) -> Vec:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _apply(R: Mat, p: Vec) -> Vec:
    return (
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2],
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2],
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2],
    )


def _mul(A: Mat, B: Mat) -> Mat:
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def _rotation_aligning(v: Vec, target: Vec) -> Mat:
    nv = _norm(v)
    nt = _norm(target)
    if nv < 1e-12 or nt < 1e-12:
        return _eye()
    a = (v[0] / nv, v[1] / nv, v[2] / nv)
    b = (target[0] / nt, target[1] / nt, target[2] / nt)
    c = _dot(a, b)
    if c > 1.0 - 1e-12:
        return _eye()
    if c < -1.0 + 1e-12:
        ortho = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
        axis = _cross(a, ortho)
        n = _norm(axis)
        axis = (axis[0] / n, axis[1] / n, axis[2] / n)
        return _rodrigues(axis, math.pi)
    k = _cross(a, b)
    s = _norm(k)
    k = (k[0] / s, k[1] / s, k[2] / s)
    return _rodrigues(k, math.atan2(s, c))


def _rodrigues(k: Vec, angle: float) -> Mat:
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    kx, ky, kz = k
    return (
        (c + kx * kx * t, kx * ky * t - kz * s, kx * kz * t + ky * s),
        (ky * kx * t + kz * s, c + ky * ky * t, ky * kz * t - kx * s),
        (kz * kx * t - ky * s, kz * ky * t + kx * s, c + kz * kz * t),
    )


def _eigh3(a: list[list[float]]) -> tuple[list[float], list[Vec]]:
    """Jacobi eigen-decomposition of a 3×3 symmetric matrix. evecs as rows."""
    v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    b = [row[:] for row in a]
    for _ in range(32):
        p, q = 0, 1
        m = abs(b[0][1])
        for i, j in ((0, 2), (1, 2)):
            if abs(b[i][j]) > m:
                p, q, m = i, j, abs(b[i][j])
        if m < 1e-14:
            break
        theta = 0.5 * math.atan2(2.0 * b[p][q], b[q][q] - b[p][p])
        c, s = math.cos(theta), math.sin(theta)
        bpp, bqq, bpq = b[p][p], b[q][q], b[p][q]
        b[p][p] = c * c * bpp - 2 * s * c * bpq + s * s * bqq
        b[q][q] = s * s * bpp + 2 * s * c * bpq + c * c * bqq
        b[p][q] = b[q][p] = 0.0
        for r in range(3):
            if r == p or r == q:
                continue
            brp, brq = b[r][p], b[r][q]
            b[r][p] = b[p][r] = c * brp - s * brq
            b[r][q] = b[q][r] = s * brp + c * brq
        for r in range(3):
            vrp, vrq = v[r][p], v[r][q]
            v[r][p] = c * vrp - s * vrq
            v[r][q] = s * vrp + c * vrq
    evals = [b[0][0], b[1][1], b[2][2]]
    evecs = [(v[0][i], v[1][i], v[2][i]) for i in range(3)]
    return evals, evecs
