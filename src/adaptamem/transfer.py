"""Leave-one-protein-out dynamics prior. Trajectories of the hold-out are LEAKAGE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from adaptamem.errors import RefuseError
from adaptamem.features import atoms_xyz_from_pdb
from adaptamem.gpcr import SWITCHES
from adaptamem.objective import Observable
from adaptamem.observables import evaluate

TEACHER_NS = (0.0, 0.5, 1.0, 2.0)
_RESOURCE = Path(__file__).parent / "resources" / "gpcr_activation.yaml"


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    data = yaml.safe_load(Path(path or _RESOURCE).read_text())
    if not isinstance(data, dict) or not data.get("proteins"):
        raise RefuseError("GPCR catalog needs proteins", code="NOT_READY")
    return data


def observables_for(residues: dict[str, list[int]]) -> list[Observable]:
    out: list[Observable] = []
    by_name = {o.name: o for o in SWITCHES}
    for name, pair in residues.items():
        if name not in by_name or len(pair) != 2:
            continue
        proto = by_name[name]
        a, b = int(pair[0]), int(pair[1])
        out.append(
            Observable(
                name=name,
                kind="distance",
                selection=f"name CA and resid {a} ; name CA and resid {b}",
                precision=proto.precision,
            )
        )
    return out


def score_pair(
    inactive_pdb: Path,
    active_pdb: Path,
    residues: dict[str, list[int]],
    *,
    inactive_chain: str | None = None,
    active_chain: str | None = None,
) -> dict[str, dict[str, float]]:
    obs = observables_for(residues)
    if not obs:
        raise RefuseError("no activation coordinates in residue map", code="NOT_READY")
    ia, ix = atoms_xyz_from_pdb(inactive_pdb, chain=inactive_chain)
    aa, ax = atoms_xyz_from_pdb(active_pdb, chain=active_chain)
    out: dict[str, dict[str, float]] = {}
    for o in obs:
        lo = float(evaluate(o, ia, ix))
        hi = float(evaluate(o, aa, ax))
        out[o.name] = {"inactive": lo, "active": hi, "span": abs(hi - lo)}
    return out


def crystal_table(catalog: dict[str, Any], *, root: Path | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """Score every protein that has both structures on disk. Skip the rest."""
    base = Path(root) if root is not None else Path.cwd()
    table: dict[str, dict[str, dict[str, float]]] = {}
    for prot in catalog.get("proteins") or []:
        pid = str(prot["id"])
        try:
            ina = _resolve(prot["inactive"], base)
            act = _resolve(prot["active"], base)
            table[pid] = score_pair(
                ina["path"],
                act["path"],
                prot.get("residues") or {},
                inactive_chain=ina.get("chain"),
                active_chain=act.get("chain"),
            )
        except RefuseError:
            continue
    return table


def switch_coupling(
    table: dict[str, dict[str, dict[str, float]]],
    *,
    epsilon: float = 0.2,
) -> dict[str, Any]:
    """Do endpoint switches move together? Hypothesis C is lock slaved to TM6."""
    rows = []
    for pid, cvs in table.items():
        tm6 = (cvs.get("tm6_ic") or {}).get("span")
        lock = (cvs.get("ionic_lock") or {}).get("span")
        pack = (cvs.get("tm3_tm6_pack") or {}).get("span")
        npy = (cvs.get("npxxY") or {}).get("span")
        rows.append(
            {
                "id": pid,
                "tm6_span": tm6,
                "lock_span": lock,
                "pack_span": pack,
                "npy_span": npy,
                "lock_not_slaved": (
                    tm6 is not None and lock is not None and tm6 >= epsilon and lock < epsilon
                ),
                "pack_hidden": pack is not None and pack < epsilon,
            }
        )
    paired = [
        (r["tm6_span"], r["lock_span"])
        for r in rows
        if r["tm6_span"] is not None and r["lock_span"] is not None
    ]
    return {
        "experiment": "switch_coupling",
        "gpu_hours": 0.0,
        "n": len(rows),
        "pearson_tm6_lock": _pearson([a for a, _ in paired], [b for _, b in paired]),
        "lock_not_slaved": [r["id"] for r in rows if r["lock_not_slaved"]],
        "pack_moves": [
            r["id"] for r in rows if r["pack_span"] is not None and r["pack_span"] >= epsilon
        ],
        "rows": rows,
    }


def leave_one_out(
    table: dict[str, dict[str, dict[str, float]]],
    hold_out: str,
    *,
    traces: dict[str, list[float]] | None = None,
    epsilon: float = 0.2,
) -> dict[str, Any]:
    """Predict the hold-out protein from other proteins' crystals. No trajectories."""
    if hold_out not in table:
        raise RefuseError(f"hold-out {hold_out!r} has no crystal scores", code="NOT_READY")
    if traces:
        raise RefuseError(
            "held-out protein traces must not enter the prior",
            code="LEAKAGE",
        )
    train_ids = [p for p in table if p != hold_out]
    if len(train_ids) < 2:
        raise RefuseError(
            f"zero-shot prior needs ≥2 training proteins; have {len(train_ids)}",
            code="NOT_READY",
        )
    truth = table[hold_out]
    names = sorted(truth)
    pred: dict[str, Any] = {}
    for name in names:
        train_lo = [table[p][name]["inactive"] for p in train_ids if name in table[p]]
        train_hi = [table[p][name]["active"] for p in train_ids if name in table[p]]
        if len(train_lo) < 2:
            continue
        lo_mu, lo_sd = _mean_sd(train_lo)
        hi_mu, hi_sd = _mean_sd(train_hi)
        err_lo = abs(lo_mu - truth[name]["inactive"])
        err_hi = abs(hi_mu - truth[name]["active"])
        err_sp = abs(abs(hi_mu - lo_mu) - truth[name]["span"])
        pred[name] = {
            "inactive": {"mu": lo_mu, "sd": lo_sd},
            "active": {"mu": hi_mu, "sd": hi_sd},
            "span": {"mu": abs(hi_mu - lo_mu), "sd": (lo_sd**2 + hi_sd**2) ** 0.5},
            "truth": truth[name],
            "error": {"inactive": err_lo, "active": err_hi, "span": err_sp},
            "reached": {
                "inactive": err_lo <= epsilon,
                "active": err_hi <= epsilon,
                "span": err_sp <= epsilon,
            },
            "unreliable": {
                "inactive": prior_unreliable(err_lo, lo_sd),
                "active": prior_unreliable(err_hi, hi_sd),
            },
        }
    return {
        "experiment": "zero_shot_prior",
        "hold_out": hold_out,
        "train": train_ids,
        "gpu_hours": 0.0,
        "predictions": pred,
        "leaked": False,
        "epsilon_nm": epsilon,
    }


def leave_one_out_sweep(
    table: dict[str, dict[str, dict[str, float]]],
    *,
    epsilon: float = 0.2,
) -> dict[str, Any]:
    """Hold out each protein in turn. GPU-hours stay zero."""
    folds: dict[str, Any] = {}
    for pid in table:
        folds[pid] = leave_one_out(table, pid, epsilon=epsilon)
    return {
        "experiment": "zero_shot_prior",
        "gpu_hours": 0.0,
        "n_proteins": len(table),
        "epsilon_nm": epsilon,
        "proteins": sorted(table),
        "folds": folds,
    }


def teacher_curve(
    *,
    zero_shot: float,
    teacher_mu: dict[float, float],
    oracle_mu: float,
    epsilon: float,
    gpu_hours: dict[float, float],
) -> dict[str, Any]:
    """ns ladder. Experiment 3 score is correction_curve (error reduction per GPU-hour)."""
    rows = []
    hours_to_eps = None
    for ns in TEACHER_NS:
        if ns == 0.0:
            mu = float(zero_shot)
        elif ns in teacher_mu:
            mu = float(teacher_mu[ns])
        else:
            continue
        err = abs(mu - oracle_mu)
        spent = float(gpu_hours.get(ns, 0.0))
        reached = err <= epsilon
        rows.append({"ns": ns, "mu": mu, "error": err, "gpu_hours": spent, "reached": reached})
        if reached and hours_to_eps is None:
            hours_to_eps = spent
    return {
        "experiment": "adaptive_correction",
        "oracle_mu": oracle_mu,
        "epsilon": epsilon,
        "curve": rows,
        "gpu_hours_to_eps": hours_to_eps,
    }


def surprise(error: float, sd: float) -> float:
    return float(error) / (float(sd) + 1e-9)


def prior_unreliable(error: float, sd: float, *, k: float = 2.0) -> bool:
    """Zero-shot error far outside the prior's own width: the prior is wrong, not under-sampled."""
    return surprise(error, sd) > k


def _resolve(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    raw = spec.get("file")
    if not raw:
        raise RefuseError("catalog entry missing file", code="NOT_READY")
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise RefuseError(f"no structure {path}", code="STRUCTURE")
    chain = spec.get("chain")
    return {"path": path, "chain": str(chain) if chain else None}


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, var**0.5
