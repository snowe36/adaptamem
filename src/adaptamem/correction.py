"""Experiment 3: tiny MD only where the prior is wrong. GPU stays off until named."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from adaptamem.errors import RefuseError
from adaptamem.transfer import surprise

GPU_HOURS = (0.0, 0.05, 0.10, 0.25, 1.0)
MECHANISM_COORDINATE = "pack_in_inactive_tm6"
_RESOURCE = Path(__file__).parent / "resources" / "experiment3.yaml"

_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def load_experiment3(path: Path | None = None) -> dict[str, Any]:
    data = yaml.safe_load(Path(path or _RESOURCE).read_text())
    if not isinstance(data, dict) or data.get("experiment") != "adaptive_correction":
        raise RefuseError("experiment 3 protocol missing", code="NOT_READY")
    for key in ("hold_out", "observables", "oracle", "teacher", "stopping", "epsilon_nm"):
        if key not in data:
            raise RefuseError(f"experiment 3 protocol missing {key}", code="NOT_READY")
    return data


def classify_observable(
    name: str,
    pred: dict[str, Any],
    *,
    epsilon: float = 0.2,
) -> str:
    """Per hold-out. A small crystal span is a trivial prior, not a transfer win."""
    if name == MECHANISM_COORDINATE:
        return "mechanism"
    span = float((pred.get("truth") or {}).get("span") or 0.0)
    prior_span = float(((pred.get("span") or {}).get("mu") or 0.0))
    if span < epsilon:
        if prior_span >= epsilon:
            return "failure"
        return "trivial"
    reached = bool((pred.get("reached") or {}).get("span"))
    if reached:
        return "transferable"
    return "failure"


def classify_fold(fold: dict[str, Any], *, epsilon: float = 0.2) -> dict[str, str]:
    return {
        name: classify_observable(name, spec, epsilon=epsilon)
        for name, spec in (fold.get("predictions") or {}).items()
    }


def mechanism_stat(
    tm6: list[float],
    pack: list[float],
    *,
    inactive_hi: float,
) -> dict[str, float]:
    """Mean packing while TM6 stays in the inactive well. Crystals cannot score this."""
    if len(tm6) != len(pack) or not tm6:
        raise RefuseError(
            "pack_in_inactive_tm6 needs paired tm6_ic and tm3_tm6_pack traces",
            code="NOT_READY",
        )
    in_well = [p for t, p in zip(tm6, pack, strict=True) if t <= inactive_hi]
    if not in_well:
        raise RefuseError("no frames in the inactive TM6 well", code="NOT_READY")
    return {
        "mu": sum(in_well) / len(in_well),
        "n": float(len(in_well)),
        "n_total": float(len(tm6)),
    }


def should_teach(role: str, pred: dict[str, Any] | None = None) -> bool:
    if role in {"trivial", "transferable"}:
        if role == "transferable" and pred:
            bad = (pred.get("unreliable") or {}).get("active") or (
                pred.get("unreliable") or {}
            ).get("inactive")
            return bool(bad)
        return False
    return role in {"failure", "mechanism"}


def gated_observables(fold: dict[str, Any], *, epsilon: float = 0.2) -> list[str]:
    roles = classify_fold(fold, epsilon=epsilon)
    pred = fold.get("predictions") or {}
    return [n for n, role in roles.items() if should_teach(role, pred.get(n))]


def disagreement(prior_mu: float, prior_sd: float, teacher_mu: float, *, k: float = 2.0) -> bool:
    return surprise(abs(float(teacher_mu) - float(prior_mu)), prior_sd) > k


def posterior_update(
    prior_mu: float,
    prior_sd: float,
    teacher_mu: float,
    teacher_sd: float,
    *,
    oracle_mu: float | None = None,
) -> dict[str, Any]:
    """Blend prior with the teacher. Oracle is evaluation-only — passing it here is LEAKAGE."""
    if oracle_mu is not None:
        raise RefuseError("oracle must not update the posterior", code="LEAKAGE")
    pp = 1.0 / (float(prior_sd) ** 2 + 1e-12)
    tp = 1.0 / (float(teacher_sd) ** 2 + 1e-12)
    mu = (pp * float(prior_mu) + tp * float(teacher_mu)) / (pp + tp)
    sd = (1.0 / (pp + tp)) ** 0.5
    disagreed = disagreement(prior_mu, prior_sd, teacher_mu)
    if disagreed:
        sd = max(sd, abs(float(teacher_mu) - float(prior_mu)) / 2.0)
    return {"mu": mu, "sd": sd, "disagreement": disagreed}


def correction_curve(
    *,
    zero_shot: float,
    teacher_mu: dict[float, float],
    oracle_mu: float,
    epsilon: float,
    conventional_gpu_hours: float | None = None,
    reference: str = "conventional_oracle",
) -> dict[str, Any]:
    """Error vs held-out oracle on a GPU-hour axis. Oracle never trains the prior."""
    rows = []
    hours_to_eps = None
    prev_err = None
    prev_h = None
    for h in GPU_HOURS:
        if h == 0.0:
            mu = float(zero_shot)
        elif h in teacher_mu:
            mu = float(teacher_mu[h])
        else:
            continue
        err = abs(mu - oracle_mu)
        reached = err <= epsilon
        reduction = None
        if prev_err is not None and prev_h is not None and h > prev_h:
            reduction = (prev_err - err) / (h - prev_h)
        rows.append(
            {
                "gpu_hours": h,
                "mu": mu,
                "error": err,
                "reached": reached,
                "error_reduction_per_gpu_hour": reduction,
            }
        )
        if reached and hours_to_eps is None:
            hours_to_eps = h
        prev_err, prev_h = err, h
    payload = {
        "experiment": "adaptive_correction",
        "reference": reference,
        "oracle_mu": oracle_mu,
        "epsilon": epsilon,
        "curve": rows,
        "gpu_hours_to_eps": hours_to_eps,
        "conventional_gpu_hours": conventional_gpu_hours,
    }
    if (
        reference == "conventional_oracle"
        and hours_to_eps is not None
        and conventional_gpu_hours
        and conventional_gpu_hours > 0
    ):
        payload["compression"] = conventional_gpu_hours / hours_to_eps if hours_to_eps else None
    elif reference != "conventional_oracle":
        payload["compression"] = None
        payload["note"] = "crystal endpoints are not an oracle; no acceleration claim"
    return payload


def adaptive_step(
    prior: dict[str, float],
    *,
    role: str,
    teacher: dict[str, float] | None = None,
    oracle_mu: float | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Trust the prior, request a teacher, or update. Scoring against oracle is last and optional."""
    proto = protocol or load_experiment3()
    teach = should_teach(role, None)
    if not teach:
        out: dict[str, Any] = {
            "action": "trust_prior",
            "gpu": False,
            "posterior": {"mu": prior["mu"], "sd": prior["sd"], "disagreement": False},
        }
        if oracle_mu is not None:
            out["error"] = abs(prior["mu"] - oracle_mu)
        return out
    if teacher is None:
        if not proto.get("gpu"):
            return {
                "action": "request_teacher",
                "gpu": False,
                "refuse": "NOT_READY",
                "reason": "experiment 3 protocol is frozen; GPU off until a named shot has eq.pdb",
                "posterior": prior,
            }
        require_teacher_gpu(proto)
        return {"action": "request_teacher", "gpu": True, "posterior": prior}
    post = posterior_update(prior["mu"], prior["sd"], teacher["mu"], teacher.get("sd", 0.05))
    out = {"action": "update", "gpu": False, "posterior": post}
    if oracle_mu is not None:
        out["error"] = abs(post["mu"] - oracle_mu)
        out["zero_shot_error"] = abs(prior["mu"] - oracle_mu)
    return out


def require_teacher_gpu(protocol: dict[str, Any] | None = None, *, eq_pdb: Path | None = None) -> None:
    proto = protocol or load_experiment3()
    if not proto.get("gpu"):
        raise RefuseError(
            "experiment 3 protocol is frozen; GPU off until a named shot has eq.pdb",
            code="NOT_READY",
        )
    if eq_pdb is None or not Path(eq_pdb).is_file():
        raise RefuseError("no eq.pdb; packing is not a teacher", code="NOT_READY")


def transfer_vs_distance(
    sweep: dict[str, Any],
    identities: dict[tuple[str, str], float],
    *,
    cv: str = "tm6_ic",
) -> list[dict[str, Any]]:
    """When does the prior stop transferring? Sequence identity vs LOO span error."""
    rows = []
    folds = sweep.get("folds") or {}
    for pid, fold in folds.items():
        train = fold.get("train") or []
        ids = []
        for t in train:
            key = (pid, t) if (pid, t) in identities else (t, pid)
            if key in identities:
                ids.append(identities[key])
        pred = (fold.get("predictions") or {}).get(cv) or {}
        err = (pred.get("error") or {}).get("span")
        if err is None or not ids:
            continue
        rows.append(
            {
                "hold_out": pid,
                "cv": cv,
                "mean_train_identity": sum(ids) / len(ids),
                "span_error": err,
                "reached": bool((pred.get("reached") or {}).get("span")),
                "n_train": len(ids),
            }
        )
    return rows


def ca_sequence(pdb: Path, chain: str | None = None) -> str:
    seen: dict[int, str] = {}
    for line in Path(pdb).read_text(errors="replace").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        ch = line[21].strip() or "A"
        if chain is not None and ch != chain:
            continue
        try:
            resid = int(line[22:26])
        except ValueError:
            continue
        if resid >= 1000:
            continue
        aa = _AA1.get(line[17:20].strip())
        if aa:
            seen[resid] = aa
    return "".join(seen[i] for i in sorted(seen))


def pairwise_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    gap = -1
    prev = list(range(0, -m - 1, -1))
    match_prev = [0] * (m + 1)
    for i, ca in enumerate(a, 1):
        cur = [ -i] + [0] * m
        match_cur = [0] * (m + 1)
        for j, cb in enumerate(b, 1):
            diag = prev[j - 1] + (1 if ca == cb else 0)
            up = prev[j] + gap
            left = cur[j - 1] + gap
            if diag >= up and diag >= left:
                cur[j] = diag
                match_cur[j] = match_prev[j - 1] + (1 if ca == cb else 0)
            elif up >= left:
                cur[j] = up
                match_cur[j] = match_prev[j]
            else:
                cur[j] = left
                match_cur[j] = match_cur[j - 1]
        prev, match_prev = cur, match_cur
    return match_prev[m] / min(n, m)


def catalog_identities(catalog: dict[str, Any], *, root: Path) -> dict[tuple[str, str], float]:
    from adaptamem.transfer import _resolve

    seqs: dict[str, str] = {}
    for prot in catalog.get("proteins") or []:
        try:
            spec = _resolve(prot["inactive"], root)
        except RefuseError:
            continue
        seqs[str(prot["id"])] = ca_sequence(spec["path"], spec.get("chain"))
    out: dict[tuple[str, str], float] = {}
    ids = sorted(seqs)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            out[(a, b)] = pairwise_identity(seqs[a], seqs[b])
    return out
