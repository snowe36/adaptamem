"""Held-out AA oracle. Adaptive sampling is scored against this distribution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptamem.diagnostics import mean_ci, occupancy_bins, summarize
from adaptamem.errors import RefuseError


@dataclass
class Oracle:
    values: dict[str, list[float]]
    estimates: dict[str, dict[str, Any]]
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "0.2",
            "created_utc": datetime.now(UTC).isoformat(),
            "role": "held_out_aa_reference",
            "observables": self.estimates,
            "n": {k: len(v) for k, v in self.values.items()},
        }


def freeze(
    traces: dict[str, list[float]],
    path: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Oracle:
    if not traces:
        raise RefuseError("oracle freeze needs observable traces", code="NOT_READY")
    estimates = {name: summarize(vals).to_dict() for name, vals in traces.items()}
    oracle = Oracle(values=traces, estimates=estimates, path=Path(path))
    payload = oracle.to_dict()
    if extra:
        payload.update(extra)
    # Keep the raw series for scoring; this is the oracle dataset.
    payload["values"] = traces
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return oracle


def freeze_workdir(workdir: Path, *, out: Path | None = None) -> Oracle:
    workdir = Path(workdir)
    traces = _traces_from_workdir(workdir)
    dest = out or (workdir / "oracle.json")
    extra = {}
    prod = workdir / "produce.json"
    if prod.is_file():
        extra["source"] = "produce.json"
        extra["produce"] = json.loads(prod.read_text()).get("ns")
    return freeze(traces, dest, extra=extra)


def load_oracle(path: Path) -> Oracle:
    data = json.loads(Path(path).read_text())
    values = data.get("values") or {}
    estimates = data.get("observables") or {k: summarize(v).to_dict() for k, v in values.items()}
    return Oracle(values=values, estimates=estimates, path=Path(path))


def score(estimate: dict[str, Any], oracle: Oracle) -> dict[str, Any]:
    """How close is this campaign's estimate to the held-out AA distribution?"""
    out: dict[str, Any] = {}
    for name, series in oracle.values.items():
        o_mu, o_ci = mean_ci(series)
        got = estimate.get(name) or {}
        mu = got.get("estimate")
        ci = got.get("ci95")
        err = None if mu is None else abs(float(mu) - o_mu)
        covers = None
        if mu is not None and ci is not None:
            covers = abs(float(mu) - o_mu) <= float(ci) + 1e-12
        bins_o = occupancy_bins(series)
        samp = got.get("values") or []
        bins_s = occupancy_bins(samp, span=(min(series), max(series))) if samp else None
        recovery = None
        if bins_s is not None:
            n_o = sum(1 for c in bins_o if c > 0)
            n_hit = sum(1 for a, b in zip(bins_o, bins_s, strict=True) if a > 0 and b > 0)
            recovery = n_hit / n_o if n_o else 0.0
        out[name] = {
            "oracle_estimate": o_mu,
            "oracle_ci95": o_ci,
            "estimate": mu,
            "ci95": ci,
            "abs_error": err,
            "oracle_mean_in_ci": covers,
            "bin_recovery": recovery,
            "gpu_hours": got.get("gpu_hours"),
            "ci_width_per_gpu_hour": got.get("ci_width_per_gpu_hour"),
            "ess": got.get("ess"),
            "state_coverage": got.get("state_coverage"),
        }
    return out


def format_score(scored: dict[str, Any]) -> str:
    lines = ["ORACLE  held-out AA reference"]
    for name, d in scored.items():
        lines.append(
            f"  {name}  oracle={d.get('oracle_estimate')}  "
            f"estimate={d.get('estimate')}  |err|={d.get('abs_error')}  "
            f"covers={d.get('oracle_mean_in_ci')}  recovery={d.get('bin_recovery')}"
        )
    return "\n".join(lines)


def _traces_from_workdir(workdir: Path) -> dict[str, list[float]]:
    states = workdir / "walker.states"
    if states.is_file():
        data = json.loads(states.read_text())
        obs = data.get("observables") or data
        if isinstance(obs, dict) and obs and isinstance(next(iter(obs.values())), list):
            return {k: list(v) for k, v in obs.items() if k != "t_ns"}
    prod = workdir / "produce.json"
    if prod.is_file():
        diags = json.loads(prod.read_text()).get("diagnostics") or {}
        # diagnostics don't carry raw series; require walker.states
        if diags and not states.is_file():
            raise RefuseError(
                "produce.json has diagnostics but no walker.states series to freeze",
                code="NOT_READY",
            )
    sample = workdir / "sample.json"
    if sample.is_file():
        raise RefuseError(
            "freeze a conventional produce/oracle workdir, not an adaptive sample.json",
            code="NOT_READY",
        )
    log = workdir / "walker.log"
    if log.is_file():
        traces: dict[str, list[float]] = {}
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for k, v in row.items():
                if k == "t_ns":
                    continue
                traces.setdefault(k, []).append(float(v))
        if traces:
            return traces
    raise RefuseError(f"no observable series in {workdir}", code="NOT_READY")
