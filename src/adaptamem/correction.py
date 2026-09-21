"""Experiment 3: tiny MD only where the prior is wrong. GPU stays off until named."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from adaptamem.errors import RefuseError
from adaptamem.transfer import surprise

FORBIDDEN_STARTS = ("2rh1", "3sn6")

GPU_HOURS = (0.0, 0.05, 0.10, 0.25, 1.0)
MECHANISM_COORDINATE = "pack_in_inactive_tm6"
_RESOURCE = Path(__file__).parent / "resources" / "experiment3.yaml"
_PRIOR_CORRECTION = Path(__file__).parent / "resources" / "prior_correction_v1.yaml"

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


def load_prior_correction(path: Path | None = None) -> dict[str, Any]:
    data = yaml.safe_load(Path(path or _PRIOR_CORRECTION).read_text())
    if not isinstance(data, dict) or data.get("experiment") != "prior_correction_v1":
        raise RefuseError("prior_correction_v1 manifest missing", code="NOT_READY")
    for key in ("targets", "teacher", "oracle", "success", "epsilon_nm", "controls"):
        if key not in data:
            raise RefuseError(f"prior_correction_v1 missing {key}", code="NOT_READY")
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
    if eq_pdb is not None and not Path(eq_pdb).is_file():
        raise RefuseError("no eq.pdb; packing is not a teacher", code="NOT_READY")


def named_teacher_workdirs(manifest: dict[str, Any] | None = None) -> list[Path]:
    man = manifest or load_prior_correction()
    out: list[Path] = []
    for target in man.get("targets") or []:
        wds = target.get("workdirs") or {}
        for role in ("inactive", "active"):
            if wds.get(role):
                out.append(Path(wds[role]))
    return out


def refuse_forbidden_starts(workdirs: list[Path]) -> None:
    for wd in workdirs:
        name = Path(wd).name.lower()
        if any(tag in name for tag in FORBIDDEN_STARTS):
            raise RefuseError(
                "do not start another 1D TM6 shot from 2RH1/3SN6",
                code="MECHANISM",
            )


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        raise RefuseError("empty teacher trace", code="NOT_READY")
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, var**0.5


def _floats(xs: Any) -> list[float]:
    return [float(x) for x in xs]


def correction_direction(
    teacher_val: float,
    prior_val: float,
    crystal_val: float,
    *,
    epsilon: float,
) -> str:
    t_err = abs(float(teacher_val) - float(crystal_val))
    p_err = abs(float(prior_val) - float(crystal_val))
    if t_err + 1e-12 < p_err:
        return "toward_crystal"
    if abs(float(teacher_val) - float(prior_val)) <= epsilon:
        return "supports_prior"
    return "neither"


def load_target_teachers(
    manifest: dict[str, Any] | None = None,
    *,
    root: Path | None = None,
) -> tuple[dict[str, dict[str, dict[str, list[float]]]], float]:
    """Read teacher.json from named workdirs. features.json is not a teacher."""
    man = manifest or load_prior_correction()
    base = Path(root) if root is not None else Path.cwd()
    teachers: dict[str, dict[str, dict[str, list[float]]]] = {}
    hours = 0.0
    for target in man.get("targets") or []:
        wds = target.get("workdirs") or {}
        pair: dict[str, dict[str, list[float]]] = {}
        for role in ("inactive", "active"):
            rel = wds.get(role)
            if not rel:
                continue
            path = Path(rel)
            if not path.is_absolute():
                path = base / path
            refuse_forbidden_starts([path])
            tj = path / "teacher.json"
            if not tj.is_file():
                continue
            data = json.loads(tj.read_text())
            traces = data.get("values") or data.get("traces") or {}
            pair[role] = {k: _floats(v) for k, v in traces.items()}
            hours += float(data.get("gpu_hours") or 0.0)
        if not pair:
            continue
        if set(pair) != {"inactive", "active"}:
            raise RefuseError(
                f"{target['protein']}: both crystal starts required",
                code="NOT_READY",
            )
        teachers[str(target["protein"])] = pair
    return teachers, hours


def _score_one(
    target: dict[str, Any],
    pair: dict[str, dict[str, list[float]]] | None,
    *,
    epsilon: float,
    gpu_hours: float,
    score_cv: str | None,
) -> dict[str, Any]:
    cv = str(target["cv"])
    if score_cv is not None and score_cv != cv:
        raise RefuseError(
            f"do not retarget {cv} to {score_cv} after seeing the trajectory",
            code="MECHANISM",
        )
    prior_error = {
        "inactive": float(target["error"]["inactive"]),
        "active": float(target["error"]["active"]),
        "span": float(target["error"]["span"]),
    }
    base: dict[str, Any] = {
        "protein": target["protein"],
        "cv": cv,
        "class": target.get("class"),
        "prior_error_before": prior_error,
        "expected_correction": target.get("expected_correction"),
        "gpu_hours": float(gpu_hours),
    }
    if pair is None:
        base.update(
            {
                "teacher": None,
                "teacher_corrected_error": None,
                "direction": None,
                "reached": target.get("reached"),
                "status": "awaiting_eq",
                "curve": correction_curve(
                    zero_shot=float(target["prior"]["span"]["mu"]),
                    teacher_mu={},
                    oracle_mu=float(target["crystal"]["span_nm"]),
                    epsilon=epsilon,
                    reference="crystal_endpoints",
                ),
            }
        )
        return base
    if cv not in pair.get("inactive", {}) or cv not in pair.get("active", {}):
        raise RefuseError(f"teacher missing named CV {cv}", code="NOT_READY")
    ina_mu, ina_sd = _mean_sd(pair["inactive"][cv])
    act_mu, act_sd = _mean_sd(pair["active"][cv])
    span = abs(act_mu - ina_mu)
    span_sd = (ina_sd**2 + act_sd**2) ** 0.5
    crystal = target["crystal"]
    prior = target["prior"]
    teacher_error = {
        "inactive": abs(ina_mu - float(crystal["inactive_nm"])),
        "active": abs(act_mu - float(crystal["active_nm"])),
        "span": abs(span - float(crystal["span_nm"])),
    }
    direction = {
        "inactive": correction_direction(
            ina_mu, float(prior["inactive"]["mu"]), float(crystal["inactive_nm"]), epsilon=epsilon
        ),
        "active": correction_direction(
            act_mu, float(prior["active"]["mu"]), float(crystal["active_nm"]), epsilon=epsilon
        ),
        "span": correction_direction(
            span, float(prior["span"]["mu"]), float(crystal["span_nm"]), epsilon=epsilon
        ),
    }
    post_span = posterior_update(
        float(prior["span"]["mu"]),
        float(prior["span"]["sd"]),
        span,
        span_sd if span_sd > 0 else 0.05,
    )
    reached = {k: teacher_error[k] <= epsilon for k in teacher_error}
    hours_to = float(gpu_hours) if reached["span"] and gpu_hours > 0 else None
    if reached["span"] and gpu_hours == 0.0:
        hours_to = 0.0
    base.update(
        {
            "teacher": {
                "inactive": {"mu": ina_mu, "sd": ina_sd},
                "active": {"mu": act_mu, "sd": act_sd},
                "span": {"mu": span, "sd": span_sd},
            },
            "teacher_corrected_error": teacher_error,
            "posterior_span": post_span,
            "direction": direction,
            "reached": reached,
            "matched_expectation": direction["span"] == target.get("expected_correction"),
            "gpu_hours_to_correction": hours_to,
            "status": "scored",
            "curve": correction_curve(
                zero_shot=float(prior["span"]["mu"]),
                teacher_mu={float(gpu_hours): span} if gpu_hours > 0 else {},
                oracle_mu=float(crystal["span_nm"]),
                epsilon=epsilon,
                reference="crystal_endpoints",
            ),
            "calibration": {
                role: {
                    "prior_unreliable": bool((target.get("unreliable") or {}).get(role)),
                    "teacher_surprise": surprise(
                        abs(
                            (ina_mu if role == "inactive" else act_mu)
                            - float(prior[role]["mu"])
                        ),
                        float(prior[role]["sd"]),
                    ),
                }
                for role in ("inactive", "active")
            },
        }
    )
    return base


def score_prior_correction(
    manifest: dict[str, Any] | None = None,
    teachers: dict[str, dict[str, dict[str, list[float]]]] | None = None,
    *,
    gpu_hours: float = 0.0,
    oracle_mu: float | None = None,
    score_cv: str | None = None,
) -> dict[str, Any]:
    """Score named-CV teachers against the frozen crystals. Oracle never trains."""
    if oracle_mu is not None:
        raise RefuseError("oracle must not update the posterior", code="LEAKAGE")
    man = manifest or load_prior_correction()
    eps = float(man.get("epsilon_nm") or 0.2)
    traces = teachers or {}
    for pid in traces:
        if pid == "adrb2":
            raise RefuseError("β2AR TM6 is the no-teacher control", code="MECHANISM")
    rows = []
    for target in man.get("targets") or []:
        pid = str(target["protein"])
        rows.append(
            _score_one(
                target,
                traces.get(pid),
                epsilon=eps,
                gpu_hours=gpu_hours,
                score_cv=score_cv,
            )
        )
    scored_rows = [r for r in rows if r.get("status") == "scored"]
    all_reached = (
        bool(rows)
        and len(scored_rows) == len(rows)
        and all((r.get("reached") or {}).get("span") for r in scored_rows)
    )
    hours_to = float(gpu_hours) if all_reached else None
    controls = []
    for ctrl in man.get("controls") or []:
        controls.append(
            {
                "protein": ctrl["protein"],
                "cv": ctrl["cv"],
                "teacher": ctrl.get("teacher"),
                "prior_error_before": ctrl.get("error"),
                "reached": ctrl.get("reached"),
                "status": "skip",
            }
        )
    return {
        "experiment": "prior_correction_v1",
        "gpu": False,
        "gpu_hours": float(gpu_hours),
        "epsilon_nm": eps,
        "reference": "crystal_endpoints",
        "compression": None,
        "note": "crystal endpoints are not an oracle; no acceleration claim",
        "targets": rows,
        "controls": controls,
        "gpu_hours_to_correction": hours_to,
        "awaiting_eq": all(r.get("status") == "awaiting_eq" for r in rows),
    }


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
