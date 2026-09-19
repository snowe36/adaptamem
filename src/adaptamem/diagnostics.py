"""Statistical precision vs sampling adequacy. CI alone is not convergence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostics:
    estimate: float | None
    ci95: float | None
    ess: float | None
    autocorrelation_time: float | None
    n: int
    gpu_hours: float | None
    trajectory_ns: float | None
    wall_seconds: float | None
    ci_width_per_gpu_hour: float | None
    state_coverage: float | None = None
    cluster_coverage: float | None = None
    transition_count: int | None = None
    n_unique_bins: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci95": self.ci95,
            "ess": self.ess,
            "autocorrelation_time": self.autocorrelation_time,
            "n": self.n,
            "gpu_hours": self.gpu_hours,
            "trajectory_ns": self.trajectory_ns,
            "wall_seconds": self.wall_seconds,
            "ci_width_per_gpu_hour": self.ci_width_per_gpu_hour,
            "state_coverage": self.state_coverage,
            "cluster_coverage": self.cluster_coverage,
            "transition_count": self.transition_count,
            "n_unique_bins": self.n_unique_bins,
            "notes": self.notes,
        }


def mean_ci(values: list[float], z: float = 1.96) -> tuple[float, float]:
    """(mean, 95% CI half-width). Empty → (nan, inf)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("inf")
    mu = sum(values) / n
    if n == 1:
        return mu, float("inf")
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var / n)
    return mu, z * se


def autocorrelation_time(values: list[float], max_lag: int | None = None) -> float:
    """Integrated autocorrelation time τ = 1 + 2 Σ ρ(k), Geyer cutoff."""
    n = len(values)
    if n < 4:
        return float("nan")
    mu = sum(values) / n
    var = sum((x - mu) ** 2 for x in values) / n
    if var <= 1e-18:
        return 1.0
    cap = max_lag if max_lag is not None else min(n // 3, 200)
    acc = 0.0
    for k in range(1, cap + 1):
        c = sum((values[i] - mu) * (values[i + k] - mu) for i in range(n - k)) / (n * var)
        if k > 1 and c <= 0:
            break
        acc += c
    return 1.0 + 2.0 * acc


def effective_sample_size(values: list[float]) -> float:
    n = len(values)
    if n < 4:
        return float(n)
    tau = autocorrelation_time(values)
    if not math.isfinite(tau) or tau <= 0:
        return float(n)
    return n / tau


def occupancy_bins(values: list[float], n_bins: int = 12, span: tuple[float, float] | None = None) -> list[int]:
    if not values:
        return [0] * n_bins
    lo, hi = span if span is not None else (min(values), max(values))
    if hi <= lo:
        counts = [0] * n_bins
        counts[0] = len(values)
        return counts
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for x in values:
        i = int((x - lo) / width)
        if i < 0:
            i = 0
        if i >= n_bins:
            i = n_bins - 1
        counts[i] += 1
    return counts


def uniqueness(this: list[float], others: list[list[float]], n_bins: int = 12) -> float:
    """Fraction of this series' occupied bins not occupied by others."""
    pool = list(this)
    for o in others:
        pool.extend(o)
    if len(pool) < 2:
        return 1.0
    span = (min(pool), max(pool))
    mine = occupancy_bins(this, n_bins=n_bins, span=span)
    rest = occupancy_bins([x for o in others for x in o], n_bins=n_bins, span=span)
    n_mine = sum(1 for c in mine if c > 0)
    if n_mine == 0:
        return 0.0
    n_unique = sum(1 for a, b in zip(mine, rest, strict=True) if a > 0 and b == 0)
    return n_unique / n_mine


def state_coverage(
    values: list[float],
    n_bins: int = 12,
    min_count: int = 1,
    span: tuple[float, float] | None = None,
) -> float:
    """Occupied-bin fraction. Pass a reference span (oracle) when scoring adequacy."""
    if not values:
        return 0.0
    counts = occupancy_bins(values, n_bins=n_bins, span=span)
    return sum(1 for c in counts if c >= min_count) / n_bins


def ci_width_per_gpu_hour(ci95: float, gpu_hours: float) -> float:
    if gpu_hours <= 0 or not math.isfinite(ci95):
        return float("inf")
    return (2.0 * ci95) / gpu_hours


def summarize(
    values: list[float],
    *,
    gpu_hours: float | None = None,
    trajectory_ns: float | None = None,
    wall_seconds: float | None = None,
    n_bins: int = 12,
    span: tuple[float, float] | None = None,
) -> Diagnostics:
    mu, half = mean_ci(values)
    ess = effective_sample_size(values) if values else 0.0
    tau = autocorrelation_time(values) if values else float("nan")
    cov = state_coverage(values, n_bins=n_bins, span=span)
    n_unique = sum(1 for c in occupancy_bins(values, n_bins=n_bins, span=span) if c > 0) if values else 0
    u_per = None
    if gpu_hours is not None:
        u_per = ci_width_per_gpu_hour(half, gpu_hours)
    notes: list[str] = []
    if values and cov < 0.2:
        notes.append("state_coverage low — CI may describe one well")
    if math.isfinite(ess) and ess < 8:
        notes.append(f"ESS={ess:.1f}; statistical precision is weak")
    return Diagnostics(
        estimate=None if not values else mu,
        ci95=None if not values else half,
        ess=ess if values else None,
        autocorrelation_time=tau if values else None,
        n=len(values),
        gpu_hours=gpu_hours,
        trajectory_ns=trajectory_ns,
        wall_seconds=wall_seconds,
        ci_width_per_gpu_hour=u_per,
        state_coverage=cov if values else None,
        cluster_coverage=None,
        transition_count=None,
        n_unique_bins=n_unique if values else None,
        notes=notes,
    )
