"""Structure Doctor: inspect a PDB before anyone builds a box.

Does not assign protonation or run PPM. It reports what the file contains
and what a human (or a later engine) must still decide.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

KD = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}

_AA3 = set(KD)
_HEAVY = {"C", "N", "O", "S", "P"}


@dataclass
class Atom:
    serial: int
    name: str
    alt: str
    resname: str
    chain: str
    resid: int
    x: float
    y: float
    z: float
    element: str
    het: bool


@dataclass
class Finding:
    key: str
    detail: str
    action: str | None = None


@dataclass
class DoctorReport:
    path: Path
    n_atoms: int
    n_protein_atoms: int
    n_residues: int
    chains: list[str]
    missing_residues: list[str]
    altlocs: int
    disulfides: int
    ligands: list[str]
    his_cys: int
    tm_spans: list[tuple[str, int, int]]
    clashes: int
    findings: list[Finding] = field(default_factory=list)
    bbox_nm: tuple[float, float, float] | None = None

    @property
    def n_tm(self) -> int:
        return len(self.tm_spans)

    @property
    def action_required(self) -> list[Finding]:
        return [f for f in self.findings if f.action]


def audit(path: Path) -> DoctorReport:
    path = Path(path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        raise ValueError("mmCIF doctor is not implemented yet; convert to PDB")
    atoms, seqres, ssbonds = _parse_pdb(path.read_text(errors="replace").splitlines())
    protein = [a for a in atoms if not a.het and a.resname in _AA3]
    by_res = _residues(protein)
    missing = _missing(seqres, by_res)
    ligands = sorted({a.resname for a in atoms if a.het and a.resname not in {"HOH", "WAT", "NA", "CL", "K"}})
    altlocs = sum(1 for a in atoms if a.alt not in {"", " "})
    his_cys = sum(1 for (c, r), aa in by_res.items() if aa in {"HIS", "CYS"})
    spans = _tm_spans(protein)
    clashes = _count_clashes(protein)
    bbox = _bbox_nm(protein)
    findings: list[Finding] = []
    if missing:
        findings.append(
            Finding(
                "missing_residues",
                f"{len(missing)} unresolved residue(s)",
                "reconstruct before production",
            )
        )
    if altlocs:
        findings.append(Finding("altlocs", f"{altlocs} atoms with alternate locations"))
    if ligands:
        findings.append(Finding("ligands", "cofactors/ligands: " + ", ".join(ligands)))
    if his_cys:
        findings.append(Finding("protonation", f"{his_cys} HIS/CYS — protonation not assigned"))
    if clashes:
        findings.append(
            Finding("clashes", f"{clashes} heavy-atom pairs < 1.5 Å", "minimize or rebuild")
        )
    if not spans:
        findings.append(Finding("tm_helices", "no hydrophobic spans at KD window ≥ 1.6"))
    else:
        findings.append(Finding("tm_helices", f"{len(spans)} candidate TM span(s)"))

    return DoctorReport(
        path=path,
        n_atoms=len(atoms),
        n_protein_atoms=len(protein),
        n_residues=len(by_res),
        chains=sorted({a.chain for a in protein}),
        missing_residues=missing,
        altlocs=altlocs,
        disulfides=len(ssbonds),
        ligands=ligands,
        his_cys=his_cys,
        tm_spans=spans,
        clashes=clashes,
        findings=findings,
        bbox_nm=bbox,
    )


def format_report(rep: DoctorReport) -> str:
    lines = [
        f"STRUCTURE REPORT  {rep.path.name}",
        f"{rep.n_residues} residues  {rep.n_protein_atoms} protein atoms  "
        f"chains {','.join(rep.chains) or '—'}",
        f"{rep.n_tm} candidate TM spans  {rep.disulfides} disulfides  "
        f"{len(rep.ligands)} ligand types",
    ]
    for ch, i, j in rep.tm_spans:
        lines.append(f"  TM  {ch} {i}-{j}")
    for f in rep.findings:
        tag = "ACTION  " if f.action else "        "
        extra = f"  → {f.action}" if f.action else ""
        lines.append(f"{tag}{f.detail}{extra}")
    if not rep.action_required:
        lines.append("No blocking reconstruction items.")
    return "\n".join(lines)


def _parse_pdb(lines: list[str]) -> tuple[list[Atom], dict[str, list[tuple[int, str]]], list[tuple[str, int, str, int]]]:
    atoms: list[Atom] = []
    seqres: dict[str, list[tuple[int, str]]] = defaultdict(list)
    ssbonds: list[tuple[str, int, str, int]] = []
    seq_pos: dict[str, int] = defaultdict(lambda: 1)
    for line in lines:
        rec = line[:6].strip()
        if rec in {"ATOM", "HETATM"} and len(line) >= 54:
            name = line[12:16].strip()
            elem = line[76:78].strip() if len(line) >= 78 else name[:1]
            atoms.append(
                Atom(
                    serial=_safe_int(line[6:11], 0),
                    name=name,
                    alt=line[16:17],
                    resname=line[17:20].strip(),
                    chain=line[21:22].strip() or "A",
                    resid=_safe_int(line[22:26], 0),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    element=elem.upper() or name[:1],
                    het=rec == "HETATM",
                )
            )
        elif rec == "SEQRES" and len(line) > 17:
            chain = line[11:12].strip() or "A"
            for tok in line[19:].split():
                if tok in _AA3:
                    seqres[chain].append((seq_pos[chain], tok))
                    seq_pos[chain] += 1
        elif rec == "SSBOND" and len(line) >= 36:
            ssbonds.append(
                (
                    line[15:16].strip() or "A",
                    _safe_int(line[17:21], 0),
                    line[29:30].strip() or "A",
                    _safe_int(line[31:35], 0),
                )
            )
    return atoms, seqres, ssbonds


def _residues(protein: list[Atom]) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for a in protein:
        out[(a.chain, a.resid)] = a.resname
    return out


def _missing(
    seqres: dict[str, list[tuple[int, str]]],
    present: dict[tuple[str, int], str],
) -> list[str]:
    if not seqres:
        return []
    miss: list[str] = []
    for chain, seq in seqres.items():
        have = {r for c, r in present if c == chain}
        # SEQRES numbering is 1..n; PDB resid may not match. Only flag if
        # the present count is short vs SEQRES length.
        if have and len(have) < len(seq) * 0.95:
            miss.append(f"{chain}: {len(seq) - len(have)} of {len(seq)} absent")
    return miss


def _tm_spans(protein: list[Atom]) -> list[tuple[str, int, int]]:
    window = 19
    thresh = 1.6
    spans: list[tuple[str, int, int]] = []
    chains: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for a in protein:
        key = (a.chain, a.resid)
        if key in seen:
            continue
        seen.add(key)
        chains[a.chain].append((a.resid, a.resname))
    for chain, seq in chains.items():
        seq.sort()
        scores: list[tuple[int, float]] = []
        for i in range(len(seq) - window + 1):
            s = sum(KD.get(seq[i + k][1], 0.0) for k in range(window)) / window
            scores.append((seq[i][0], s))
        in_span = False
        start = 0
        last = 0
        for resid, s in scores:
            if s >= thresh:
                if not in_span:
                    start, in_span = resid, True
                last = resid + window - 1
            elif in_span:
                spans.append((chain, start, last))
                in_span = False
        if in_span:
            spans.append((chain, start, last))
    return spans


def _count_clashes(protein: list[Atom], cutoff: float = 1.5) -> int:
    heavies = [
        a
        for a in protein
        if a.alt in {"", " ", "A"} and (a.element[:1] in _HEAVY or a.name[:1] in _HEAVY)
    ]
    n = 0
    cut2 = cutoff * cutoff
    # Grid in Å
    cell = 2.0
    buckets: dict[tuple[int, int, int], list[Atom]] = defaultdict(list)
    for a in heavies:
        buckets[(int(a.x // cell), int(a.y // cell), int(a.z // cell))].append(a)
    seen: set[tuple[int, int]] = set()
    for (ix, iy, iz), group in buckets.items():
        neigh: list[Atom] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neigh.extend(buckets.get((ix + dx, iy + dy, iz + dz), []))
        for a in group:
            for b in neigh:
                if b.serial <= a.serial:
                    continue
                if a.chain == b.chain and abs(a.resid - b.resid) <= 1:
                    continue
                dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
                if dx * dx + dy * dy + dz * dz < cut2:
                    pair = (a.serial, b.serial)
                    if pair not in seen:
                        seen.add(pair)
                        n += 1
    return n


def _bbox_nm(protein: list[Atom]) -> tuple[float, float, float] | None:
    if not protein:
        return None
    xs = [a.x for a in protein]
    ys = [a.y for a in protein]
    zs = [a.z for a in protein]
    return (
        (max(xs) - min(xs)) / 10.0,
        (max(ys) - min(ys)) / 10.0,
        (max(zs) - min(zs)) / 10.0,
    )


def _safe_int(s: str, default: int) -> int:
    s = s.strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default
