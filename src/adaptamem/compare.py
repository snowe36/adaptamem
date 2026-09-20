"""Compression: baseline MD avoided per GPU-hour of oracle, not CI/hour."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.compress import KINDS as COMPRESS_KINDS
from adaptamem.diagnostics import summarize
from adaptamem.errors import RefuseError
from adaptamem.oracle import Oracle, score

ACCELERATION_BAR = 10.0

METHODS = (
    "single_long",
    "independent_replicas",
    "adaptive",
    "random_branch",
    "latent_dynamics",
    "cpu_only",
    "msm",
    "surrogate",
    "hybrid_correction",
    "active_learning",
)


@dataclass
class MethodRun:
    name: str
    traces: dict[str, list[float]]
    gpu_hours: float
    trajectory_ns: float | None = None
    cpu_hours: float | None = None
    leaked: bool = False


def _diag(run: MethodRun) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, vals in run.traces.items():
        d = summarize(
            vals, gpu_hours=run.gpu_hours, trajectory_ns=run.trajectory_ns
        ).to_dict()
        d["values"] = list(vals)
        out[name] = d
    return out


def equal_compute(oracle: Oracle, runs: list[MethodRun]) -> dict[str, Any]:
    hours = {r.name: r.gpu_hours for r in runs}
    if len(set(round(h, 6) for h in hours.values())) > 1:
        # still compare; label the mismatch
        note = "gpu_hours differ across methods"
    else:
        note = "equal GPU-hours"
    scored = {r.name: score(_diag(r), oracle) for r in runs}
    return {"experiment": "equal_compute", "note": note, "methods": scored}


def hours_to_oracle_error(
    values: list[float],
    oracle_mean: float,
    error: float,
    gpu_hours: float,
) -> float | None:
    """GPU-hours until |running mean − oracle μ| ≤ error (prefix of the series)."""
    if not values or gpu_hours <= 0 or error <= 0:
        return None
    n = len(values)
    dt = gpu_hours / n
    acc = 0.0
    for i, x in enumerate(values, start=1):
        acc += float(x)
        if i < 8:
            continue
        if abs(acc / i - oracle_mean) <= error:
            return dt * i
    return None


def is_compress_method(name: str) -> bool:
    key = name.replace("-", "_")
    return key in COMPRESS_KINDS or key in {"surrogate", "cpu_only", "cpu_first"}


def acceleration_claim(
    *,
    reached: bool,
    compression_vs_oracle: float | None,
    gpu_hours: float,
    cpu_hours: float | None = None,
    leaked: bool = False,
    bar: float = ACCELERATION_BAR,
) -> dict[str, Any]:
    """True at matched ε with ≥10× fewer GPU-hours, or CPU-only inference that matches.

    Lower CI or earlier stopping without matched oracle error is not acceleration.
    """
    base = {"bar": bar, "gpu_hours": gpu_hours, "cpu_hours": cpu_hours}
    if leaked:
        return {**base, "accelerated": False, "reason": "oracle leakage"}
    if not reached:
        return {**base, "accelerated": False, "reason": "did not match oracle ε"}
    if gpu_hours == 0:
        return {
            **base,
            "accelerated": True,
            "reason": "CPU-only inference at matched ε",
        }
    if compression_vs_oracle is None:
        return {
            **base,
            "accelerated": False,
            "reason": "missing conventional GPU-hours; cannot claim a ratio",
        }
    if compression_vs_oracle < bar:
        return {
            **base,
            "accelerated": False,
            "reason": f"{compression_vs_oracle:.2f}× < {bar:g}× bar",
        }
    return {
        **base,
        "accelerated": True,
        "reason": f"{compression_vs_oracle:.1f}× MD avoided per oracle GPU-hour",
    }


def oracle_compression(
    oracle: Oracle,
    runs: list[MethodRun],
    *,
    error: float,
    oracle_gpu_hours: float | None = None,
) -> dict[str, Any]:
    """compression = baseline MD avoided / GPU oracle spent. speedup includes CPU."""
    methods: dict[str, Any] = {}
    baseline = oracle_gpu_hours
    for r in runs:
        per_obs: dict[str, Any] = {}
        for name, vals in r.traces.items():
            o_vals = oracle.values.get(name) or []
            if not o_vals:
                continue
            o_mu = sum(o_vals) / len(o_vals)
            if is_compress_method(r.name):
                mu = sum(vals) / len(vals) if vals else None
                reached = mu is not None and abs(mu - o_mu) <= error
                hours = r.gpu_hours if reached else None
            else:
                hours = hours_to_oracle_error(vals, o_mu, error, r.gpu_hours)
                reached = hours is not None
            spent = float(hours) if hours is not None else float(r.gpu_hours)
            avoided = float(baseline) if (reached and baseline) else 0.0
            ratio = (avoided / spent) if (reached and spent > 0 and baseline) else None
            cpu = float(r.cpu_hours or 0.0)
            denom = spent + cpu
            speedup = (float(baseline) / denom) if (reached and baseline and denom > 0) else None
            per_obs[name] = {
                "error": error,
                "oracle_mean": o_mu,
                "gpu_hours_to_error": hours,
                "total_gpu_hours": r.gpu_hours,
                "cpu_hours": r.cpu_hours,
                "reached": reached,
                "baseline_md_avoided_hours": avoided if reached else 0.0,
                "oracle_gpu_hours_spent": spent if reached else r.gpu_hours,
                "compression": ratio,
                "compression_vs_oracle": ratio,
                "speedup": speedup,
                "cpu_only": bool(reached and r.gpu_hours == 0),
                "leaked": r.leaked,
                "acceleration": acceleration_claim(
                    reached=reached,
                    compression_vs_oracle=ratio,
                    gpu_hours=r.gpu_hours,
                    cpu_hours=r.cpu_hours,
                    leaked=r.leaked,
                ),
            }
        methods[r.name] = per_obs
    return {
        "experiment": "oracle_compression",
        "error": error,
        "oracle_gpu_hours": oracle_gpu_hours,
        "acceleration_bar": ACCELERATION_BAR,
        "note": (
            "compression = baseline MD compute avoided / GPU oracle spent. "
            "speedup = baseline / (CPU inference + oracle). "
            "Acceleration is ≥10× at matched ε, or CPU-only inference that matches. "
            "Lower CI is not a claim."
        ),
        "methods": methods,
    }


def hours_to_precision(
    values: list[float],
    precision: float,
    gpu_hours: float,
    *,
    coverage_bar: float = 0.25,
    span: tuple[float, float] | None = None,
) -> float | None:
    """GPU-hours until CI ≤ precision and coverage ≥ bar (prefix of the series)."""
    if not values or gpu_hours <= 0:
        return None
    n = len(values)
    dt = gpu_hours / n
    for i in range(2, n + 1):
        d = summarize(values[:i], gpu_hours=dt * i, span=span)
        if d.ci95 is None:
            continue
        if d.n < 8 or (d.ess is not None and d.ess < 8):
            continue
        cov = d.state_coverage or 0.0
        if d.ci95 <= precision and cov >= coverage_bar:
            return dt * i
    return None


def equal_precision(
    oracle: Oracle,
    runs: list[MethodRun],
    *,
    precision: float | None = None,
    coverage_bar: float = 0.25,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for r in runs:
        per_obs: dict[str, Any] = {}
        for name, vals in r.traces.items():
            if precision is None:
                o = oracle.estimates.get(name) or {}
                prec = float(o.get("ci95") or 0.5)
            else:
                prec = precision
            o_vals = oracle.values.get(name) or []
            span = (min(o_vals), max(o_vals)) if len(o_vals) >= 2 else None
            hours = hours_to_precision(
                vals, prec, r.gpu_hours, coverage_bar=coverage_bar, span=span
            )
            per_obs[name] = {
                "precision": prec,
                "gpu_hours_to_precision": hours,
                "total_gpu_hours": r.gpu_hours,
                "reached": hours is not None,
            }
        methods[r.name] = per_obs
    return {
        "experiment": "equal_precision",
        "coverage_bar": coverage_bar,
        "methods": methods,
    }


def compare(
    oracle: Oracle,
    runs: list[MethodRun],
    *,
    precision: float | None = None,
    error: float | None = None,
    oracle_gpu_hours: float | None = None,
) -> dict[str, Any]:
    if len(runs) < 2:
        raise RefuseError("compare needs at least two methods", code="NOT_READY")
    names = [r.name for r in runs]
    eps = error if error is not None else (precision if precision is not None else 0.05)
    payload = {
        "policy": "0.4",
        "created_utc": datetime.now(UTC).isoformat(),
        "physics": "same_physics",
        "methods": names,
        "oracle_compression": oracle_compression(
            oracle, runs, error=eps, oracle_gpu_hours=oracle_gpu_hours
        ),
        "equal_compute": equal_compute(oracle, runs),
        "equal_precision": equal_precision(oracle, runs, precision=precision),
        "primary_metric": "compression",
        "acceleration_bar": ACCELERATION_BAR,
        "note": (
            "primary score is baseline MD avoided per GPU-hour of oracle; "
            "acceleration is ≥10× at matched ε, or CPU-only inference that matches. "
            "CI/hour is diagnostic. More MD that stops sooner is not compression."
        ),
    }
    return payload


def load_method(path: Path, name: str | None = None) -> MethodRun:
    data = json.loads(Path(path).read_text())
    label = name or str(data.get("method") or Path(path).stem)
    traces = data.get("values") or data.get("traces") or {}
    if not traces:
        raise RefuseError(f"{path} has no observable series (need values/traces)", code="NOT_READY")
    hours = float(data.get("gpu_hours") or 0.0)
    if hours <= 0:
        diags = data.get("diagnostics") or {}
        for d in diags.values() if isinstance(diags, dict) else []:
            if isinstance(d, dict) and d.get("gpu_hours"):
                hours = float(d["gpu_hours"])
                break
    ns = data.get("ns") or data.get("trajectory_ns")
    cpu = data.get("cpu_hours")
    leaked = bool(data.get("leaked"))
    return MethodRun(
        name=label,
        traces=traces,
        gpu_hours=hours,
        trajectory_ns=ns,
        cpu_hours=None if cpu is None else float(cpu),
        leaked=leaked,
    )


def write_compare(payload: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def format_compare(payload: dict[str, Any]) -> str:
    lines = [
        "COMPARE  GPU-hours to oracle ε (MD is the teacher)",
        f"methods   {', '.join(payload.get('methods') or [])}",
    ]
    oc = payload.get("oracle_compression") or {}
    lines.append(f"A oracle compression  ε={oc.get('error')}  ({oc.get('note')})")
    for method, obs in (oc.get("methods") or {}).items():
        for name, d in obs.items():
            acc = d.get("acceleration") or {}
            claim = "ACCELERATED" if acc.get("accelerated") else "no claim"
            lines.append(
                f"  {method:24s} {name}  hours={d.get('gpu_hours_to_error')}  "
                f"reached={d.get('reached')}  compression={d.get('compression')}  "
                f"speedup={d.get('speedup')}  cpu_h={d.get('cpu_hours')}  {claim}"
            )
    eq = payload.get("equal_compute") or {}
    lines.append(f"B equal compute  ({eq.get('note')})")
    for method, obs in (eq.get("methods") or {}).items():
        for name, d in obs.items():
            lines.append(
                f"  {method:24s} {name}  |err|={d.get('abs_error')}  "
                f"U/GPU-h={d.get('ci_width_per_gpu_hour')}  "
                f"ESS={d.get('ess')}"
            )
    ep = payload.get("equal_precision") or {}
    lines.append("C equal precision")
    for method, obs in (ep.get("methods") or {}).items():
        for name, d in obs.items():
            lines.append(
                f"  {method:24s} {name}  hours={d.get('gpu_hours_to_precision')}  "
                f"reached={d.get('reached')}"
            )
    return "\n".join(lines)


def synthetic_demo(n: int = 400) -> tuple[Oracle, list[MethodRun]]:
    """CI-only unit-test fixture: adaptive recovers the oracle mean with less waste."""
    import random

    rng = random.Random(0)
    oracle_vals = [rng.gauss(1.0, 0.25) for _ in range(n * 4)]
    oracle = Oracle(
        values={"cv": oracle_vals},
        estimates={"cv": summarize(oracle_vals).to_dict()},
    )
    # Single long: stuck then a late jump — tight CI, wrong answer at equal hours.
    single = [rng.gauss(0.2, 0.05) for _ in range(int(n * 0.85))]
    single += [rng.gauss(1.0, 0.25) for _ in range(n - len(single))]
    # Independent replicas: mixture, slower to mix.
    replicas: list[float] = []
    for i in range(4):
        mu = 0.2 if i < 2 else 1.0
        replicas.extend(rng.gauss(mu, 0.15) for _ in range(n // 4))
    # Adaptive: spends time near the oracle mean.
    adaptive = [rng.gauss(1.0, 0.25) for _ in range(n)]
    # Random-state branching ablation: better than single, worse than adaptive.
    branch = [rng.gauss(0.7, 0.4) for _ in range(n)]
    hours = 10.0
    runs = [
        MethodRun("single_long", {"cv": single}, hours, trajectory_ns=100.0),
        MethodRun("independent_replicas", {"cv": replicas}, hours, trajectory_ns=100.0),
        MethodRun("adaptive", {"cv": adaptive}, hours, trajectory_ns=100.0),
        MethodRun("random_branch", {"cv": branch}, hours, trajectory_ns=100.0),
    ]
    return oracle, runs
