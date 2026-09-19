"""Same Hamiltonian, different sampling strategy. Two experiments, three methods."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.diagnostics import summarize
from adaptamem.errors import RefuseError
from adaptamem.oracle import Oracle, score

METHODS = ("single_long", "independent_replicas", "adaptive", "random_branch")


@dataclass
class MethodRun:
    name: str
    traces: dict[str, list[float]]
    gpu_hours: float
    trajectory_ns: float | None = None


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
) -> dict[str, Any]:
    if len(runs) < 2:
        raise RefuseError("compare needs at least two methods", code="NOT_READY")
    names = [r.name for r in runs]
    payload = {
        "policy": "0.2",
        "created_utc": datetime.now(UTC).isoformat(),
        "physics": "same_physics",
        "methods": names,
        "equal_compute": equal_compute(oracle, runs),
        "equal_precision": equal_precision(oracle, runs, precision=precision),
        "primary_metric": "ci_width_per_gpu_hour",
        "note": (
            "same Hamiltonian, system, integrator, timestep, analysis; "
            "only sampling strategy changes"
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
    return MethodRun(name=label, traces=traces, gpu_hours=hours, trajectory_ns=ns)


def write_compare(payload: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def format_compare(payload: dict[str, Any]) -> str:
    lines = [
        "COMPARE  same physics, different sampling",
        f"methods   {', '.join(payload.get('methods') or [])}",
    ]
    eq = payload.get("equal_compute") or {}
    lines.append(f"A equal compute  ({eq.get('note')})")
    for method, obs in (eq.get("methods") or {}).items():
        for name, d in obs.items():
            lines.append(
                f"  {method:24s} {name}  |err|={d.get('abs_error')}  "
                f"U/GPU-h={d.get('ci_width_per_gpu_hour')}  "
                f"ESS={d.get('ess')}"
            )
    ep = payload.get("equal_precision") or {}
    lines.append("B equal precision")
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
