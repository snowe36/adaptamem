"""MD is the teacher. CPU plugs consume short traces; they do not integrate missing time.

MSM, latent occupancy, and uncertainty-gated active_learning are implemented.
Other named kinds REFUSE — those REFUSEs are controls, not more OpenMM.
"""

from __future__ import annotations

import random
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.ladder import identify

KINDS = (
    "msm",
    "milestoning",
    "weighted_ensemble",
    "learned_propagator",
    "latent_dynamics",
    "generative_eq",
    "hybrid_correction",
    "active_learning",
)

# Inference must not see the answer. Evaluation uses these after compress returns.
LEAKAGE_KEYS = frozenset(
    {
        "oracle",
        "oracle_mean",
        "oracle_mu",
        "oracle_values",
        "oracle_populations",
        "oracle_estimates",
        "oracle_pi",
        "future_frames",
        "held_out_frames",
        "eval_frames",
        "eval_values",
    }
)


def leakage_keys(kwargs: dict[str, object], traces: dict[str, list[float]] | None = None) -> list[str]:
    leaked = [k for k in kwargs if k in LEAKAGE_KEYS or k.startswith("oracle_")]
    if traces:
        leaked.extend(k for k in traces if k in LEAKAGE_KEYS or k.startswith("oracle_"))
    return sorted(set(leaked))


def compress(kind: str, traces: dict[str, list[float]], **kwargs: object) -> dict[str, Any]:
    """Fit a CPU model from teacher traces. Does not sample; use infer()."""
    key = kind.replace("-", "_")
    if key not in KINDS:
        raise RefuseError(
            f"unknown compression {kind!r}; use {', '.join(KINDS)}",
            code="NOT_IMPLEMENTED",
        )
    leaked = leakage_keys(kwargs, traces)
    if leaked:
        raise RefuseError(
            "no oracle leakage: "
            + ", ".join(leaked)
            + " must not be passed into inference",
            code="LEAKAGE",
        )
    if key == "msm":
        model = fit_msm(traces, **kwargs)
    elif key == "latent_dynamics":
        model = fit_latent(traces, **kwargs)
    elif key == "active_learning":
        model = fit_active(traces, **kwargs)
    else:
        raise RefuseError(
            f"{key} is a named plug: learn from short MD, then predict. "
            "Do not spend the remaining budget integrating walkers.",
            code="COMPRESS",
        )
    gpu = float(model.get("gpu_hours") or 0.0)
    model["identification"] = identify(
        model, mode="cpu_only" if gpu == 0.0 else "cpu_first"
    )
    return model


def infer_report(
    model: dict[str, Any],
    *,
    mode: str = "cpu_first",
    n_samples: int = 500,
    seed: int = 0,
    min_count: int = 5,
) -> dict[str, Any]:
    """Infer values plus identification. cpu_only forbids teacher GPU-hours."""
    from adaptamem.pipeline import MODES

    key = mode.replace("-", "_")
    if key not in MODES:
        raise RefuseError(f"unknown mode {mode!r}; use {', '.join(MODES)}", code="NOT_READY")
    if key == "conventional":
        raise RefuseError(
            "conventional is held-out MD (oracle-freeze + analyze), not infer",
            code="NOT_READY",
        )
    gpu = float(model.get("gpu_hours") or 0.0)
    if key == "cpu_only" and gpu > 0:
        raise RefuseError("cpu_only forbids teacher GPU-hours", code="NOT_READY")
    values = infer(model, n_samples=n_samples, seed=seed)
    uncertain: dict[str, list[int]] = {}
    try:
        uncertain = uncertain_regions(model, min_count=min_count)
    except RefuseError:
        uncertain = {}
    ident = identify(model, mode=key)
    if key == "cpu_only":
        ident["kinetics_identified"] = False
        ident["pi_identified"] = False
        extra = ["pi", "transitions", "timescales"]
        ident["unidentified"] = list(dict.fromkeys(extra + list(ident["unidentified"])))
    return {
        "mode": key,
        "method": model.get("kind"),
        "kind": model.get("kind"),
        "values": values,
        "gpu_hours": gpu,
        "uncertain": uncertain,
        "identification": ident,
        "identified": ident["identified"],
        "unidentified": ident["unidentified"],
        "wells": _model_wells(model),
        "bridge_missing": _bridge_missing(model),
    }


def infer(model: dict[str, Any], **kwargs: object) -> dict[str, list[float]]:
    """CPU production-scale search from a compressed model. No MD."""
    kind = str(model.get("kind") or "").replace("-", "_")
    if kind == "active_learning":
        inner = model.get("inner_model")
        if not isinstance(inner, dict):
            raise RefuseError("active_learning model missing inner_model", code="COMPRESS")
        return infer(inner, **kwargs)
    if kind == "msm":
        return sample_msm(model, **kwargs)
    if kind == "latent_dynamics":
        return sample_latent(model, **kwargs)
    if kind in KINDS:
        raise RefuseError(
            f"{kind} has no infer yet; fit is a control REFUSE until built",
            code="COMPRESS",
        )
    raise RefuseError(f"unknown model kind {kind!r}", code="NOT_IMPLEMENTED")


def uncertain_regions(model: dict[str, Any], *, min_count: int = 5) -> dict[str, list[int]]:
    """Bins the CPU model has barely seen. Those are the only GPU-oracle candidates."""
    kind = str(model.get("kind") or "").replace("-", "_")
    if kind == "active_learning":
        inner = model.get("inner_model")
        if not isinstance(inner, dict):
            raise RefuseError("active_learning model missing inner_model", code="COMPRESS")
        return uncertain_regions(inner, min_count=min_count)
    if kind == "latent_dynamics":
        return _uncertain_latent(model, min_count=min_count)
    if kind != "msm":
        raise RefuseError(f"uncertain_regions not implemented for {kind}", code="COMPRESS")
    out: dict[str, list[int]] = {}
    for name, spec in (model.get("observables") or {}).items():
        if spec.get("constant") is not None:
            out[name] = [0]
            continue
        if spec.get("bridge_missing"):
            out[name] = []
            continue
        counts = spec.get("counts") or []
        out[name] = [i for i, row in enumerate(counts) if sum(row) < min_count]
    return out


def fit_msm(traces: dict[str, list[float]], **kwargs: object) -> dict[str, Any]:
    if not traces:
        raise RefuseError("msm needs teacher traces", code="COMPRESS")
    lag = _int_kw(kwargs, "lag", 1)
    n_bins = max(2, _int_kw(kwargs, "n_bins", 8))
    obs: dict[str, Any] = {}
    for name, series in traces.items():
        xs = [float(v) for v in series]
        if len(xs) < lag + 2:
            raise RefuseError(
                f"msm teacher {name!r} has {len(xs)} frames; need lag+2 ({lag + 2})",
                code="COMPRESS",
            )
        lens = _shot_lengths(kwargs, name, len(xs))
        obs[name] = _fit_1d(xs, lag=lag, n_bins=n_bins, shot_lengths=lens)
    return {
        "kind": "msm",
        "gpu_hours": _gpu_hours(kwargs),
        "observables": obs,
        "lag": lag,
        "shot_lengths": kwargs.get("shot_lengths"),
    }


def fit_latent(traces: dict[str, list[float]], **kwargs: object) -> dict[str, Any]:
    """Empirical occupancy of crystal or teacher frames. Not a neural SDE."""
    if not traces:
        raise RefuseError(
            "latent_dynamics needs crystal or teacher frames (structure prior)",
            code="COMPRESS",
        )
    obs: dict[str, Any] = {}
    for name, series in traces.items():
        xs = [float(v) for v in series]
        if not xs:
            raise RefuseError(f"latent teacher {name!r} is empty", code="COMPRESS")
        obs[name] = {
            "support": xs,
            "lo": min(xs),
            "hi": max(xs),
            "n_frames": len(xs),
            "kinetics_identified": False,
        }
    return {"kind": "latent_dynamics", "gpu_hours": _gpu_hours(kwargs), "observables": obs}


def fit_active(traces: dict[str, list[float]], **kwargs: object) -> dict[str, Any]:
    """Query only hungry bins. If everything is uncertain, REFUSE — not more MD."""
    inner_kind = str(kwargs.get("inner") or "msm").replace("-", "_")
    if inner_kind == "active_learning":
        raise RefuseError("active_learning inner cannot be itself", code="COMPRESS")
    extra = kwargs.get("extra_shots")
    extra_map: dict[str, list[list[float]]] = {}
    if isinstance(extra, dict):
        extra_map = {str(k): [list(map(float, shot)) for shot in v] for k, v in extra.items()}
    gpu_per = float(kwargs.get("gpu_hours_per_shot") or 1.0)
    max_shots = max(0, _int_kw(kwargs, "max_shots", 8))
    spent = _gpu_hours(kwargs)
    current = {k: [float(x) for x in v] for k, v in traces.items()}
    shot_lens: dict[str, list[int]] = {
        name: _shot_lengths(kwargs, name, len(xs)) for name, xs in current.items()
    }
    inner_kw = {
        k: v
        for k, v in kwargs.items()
        if k
        not in {
            "inner",
            "extra_shots",
            "gpu_hours_per_shot",
            "max_shots",
            "teacher_gpu_hours",
            "gpu_hours",
            "shot_lengths",
        }
    }
    queried = 0
    model: dict[str, Any] | None = None
    while True:
        inner_kw["shot_lengths"] = shot_lens
        inner_kw["teacher_gpu_hours"] = spent
        model = compress(inner_kind, current, **inner_kw)
        if _bridge_missing(model) and not _has_extra(extra_map):
            raise RefuseError(
                "two wells observed, no sampled transition; "
                "filling 1D bins is not a mechanism teacher",
                code="BRIDGE",
            )
        unc = uncertain_regions(model)
        if not _any_hungry(unc) and not _bridge_missing(model):
            break
        if _all_hungry(model, unc) and not _has_extra(extra_map):
            raise RefuseError(
                "every bin is uncertain; refusing to become conventional MD",
                code="COVERAGE",
            )
        if queried >= max_shots or not _has_extra(extra_map):
            raise RefuseError(
                "uncertain bins remain and no extra teacher shot is available",
                code="COVERAGE",
            )
        name, shot = _pop_shot(extra_map)
        current.setdefault(name, []).extend(shot)
        shot_lens.setdefault(name, []).append(len(shot))
        spent += gpu_per
        queried += 1
    assert model is not None
    return {
        "kind": "active_learning",
        "inner": inner_kind,
        "inner_model": model,
        "gpu_hours": spent,
        "n_queries": queried,
        "observables": model.get("observables") or {},
    }


def sample_msm(model: dict[str, Any], **kwargs: object) -> dict[str, list[float]]:
    n_samples = max(1, _int_kw(kwargs, "n_samples", 500))
    seed = _int_kw(kwargs, "seed", 0)
    out: dict[str, list[float]] = {}
    for i, (name, spec) in enumerate((model.get("observables") or {}).items()):
        out[name] = _sample_1d(spec, n_samples=n_samples, seed=seed + i)
    if not out:
        raise RefuseError("msm model has no observables", code="COMPRESS")
    return out


def sample_latent(model: dict[str, Any], **kwargs: object) -> dict[str, list[float]]:
    n_samples = max(1, _int_kw(kwargs, "n_samples", 500))
    seed = _int_kw(kwargs, "seed", 0)
    out: dict[str, list[float]] = {}
    rng = random.Random(seed)
    for name, spec in (model.get("observables") or {}).items():
        support = [float(x) for x in (spec.get("support") or [])]
        if not support:
            raise RefuseError(f"latent model {name!r} has empty support", code="COMPRESS")
        out[name] = [support[rng.randrange(len(support))] for _ in range(n_samples)]
    return out


def msm_from_traces(traces: dict[str, list[float]], **kwargs: object) -> dict[str, list[float]]:
    """Fit then sample. Tests and one-shot CLI."""
    n_samples = kwargs.get("n_samples", 500)
    seed = kwargs.get("seed", 0)
    fit_kw = {k: v for k, v in kwargs.items() if k not in {"n_samples", "seed"}}
    return infer(fit_msm(traces, **fit_kw), n_samples=n_samples, seed=seed)


def _gpu_hours(kwargs: dict[str, object]) -> float:
    v = kwargs.get("teacher_gpu_hours", kwargs.get("gpu_hours", 0.0))
    if v is None:
        return 0.0
    return float(v)  # type: ignore[arg-type]


def _shot_lengths(kwargs: dict[str, object], name: str, n: int) -> list[int]:
    raw = kwargs.get("shot_lengths")
    if raw is None:
        return [n]
    if isinstance(raw, dict):
        lens = list(raw.get(name) or raw.get("default") or [])
    else:
        lens = [int(x) for x in raw]  # type: ignore[arg-type]
    if not lens:
        return [n]
    if sum(lens) != n:
        raise RefuseError(
            f"shot_lengths for {name!r} sum to {sum(lens)}, series has {n} frames",
            code="COMPRESS",
        )
    return [int(x) for x in lens]


def _uncertain_latent(model: dict[str, Any], *, min_count: int) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for name, spec in (model.get("observables") or {}).items():
        lo = float(spec.get("lo") or 0.0)
        hi = float(spec.get("hi") or 0.0)
        support = [float(x) for x in (spec.get("support") or [])]
        if hi - lo < 1e-12 or len(support) < min_count:
            out[name] = [0]
            continue
        n_bins = 8
        edges = _quantile_edges(support, n_bins)
        counts = [0] * max(1, len(edges) - 1)
        for x in support:
            counts[_assign(x, edges)] += 1
        out[name] = [i for i, c in enumerate(counts) if c < min_count]
    return out


def _n_bins(spec: dict[str, Any]) -> int:
    if spec.get("constant") is not None:
        return 1
    if spec.get("support") is not None:
        lo = float(spec.get("lo") or 0.0)
        hi = float(spec.get("hi") or 0.0)
        return 1 if hi - lo < 1e-12 else 8
    return len(spec.get("counts") or spec.get("pi") or spec.get("centers") or [0])


def _any_hungry(unc: dict[str, list[int]]) -> bool:
    return any(bins for bins in unc.values())


def _bridge_missing(model: dict[str, Any]) -> bool:
    obs = model.get("observables") or {}
    return any(
        isinstance(spec, dict) and spec.get("bridge_missing") for spec in obs.values()
    )


def _model_wells(model: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in (model.get("observables") or {}).items():
        if not isinstance(spec, dict):
            continue
        out[name] = {
            "wells": spec.get("wells") or [],
            "gap": spec.get("gap"),
            "cross_hops": spec.get("cross_hops"),
            "bridge_missing": bool(spec.get("bridge_missing")),
        }
    return out


def _wells(xs: list[float], shot_lengths: list[int]) -> list[dict[str, float]]:
    by_shot = _shot_supports(xs, shot_lengths)
    if len(shot_lengths) >= 2 and len(by_shot) >= 2:
        raw = by_shot
    else:
        raw = _value_clusters(xs)
    return [{"lo": lo, "hi": hi, "n": n} for lo, hi, n in raw]


def _value_clusters(xs: list[float]) -> list[tuple[float, float, int]]:
    uniq = sorted(set(xs))
    if not uniq:
        return []
    if len(uniq) == 1:
        return [(uniq[0], uniq[0], len(xs))]
    if len(uniq) == 2:
        a, b = uniq
        return [(a, a, xs.count(a)), (b, b, xs.count(b))]
    gaps = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
    mx = max(gaps)
    med = sorted(gaps)[len(gaps) // 2]
    cuts = {gaps.index(mx)} if mx > 0 and mx >= 2.0 * med else set()
    groups: list[list[float]] = [[uniq[0]]]
    for i in range(len(gaps)):
        if i in cuts:
            groups.append([uniq[i + 1]])
        else:
            groups[-1].append(uniq[i + 1])
    out: list[tuple[float, float, int]] = []
    for g in groups:
        lo, hi = g[0], g[-1]
        n = sum(1 for x in xs if lo <= x <= hi)
        out.append((lo, hi, n))
    return out


def _shot_supports(xs: list[float], shot_lengths: list[int]) -> list[tuple[float, float, int]]:
    ranges: list[tuple[float, float, int]] = []
    off = 0
    for slen in shot_lengths:
        seg = xs[off : off + slen]
        off += slen
        if not seg:
            continue
        ranges.append((min(seg), max(seg), len(seg)))
    return _merge_ranges(ranges)


def _merge_ranges(ranges: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: r[0])
    merged = [list(ordered[0])]
    for lo, hi, n in ordered[1:]:
        plo, phi, pn = merged[-1]
        if lo <= phi:
            merged[-1] = [plo, max(phi, hi), pn + n]
        else:
            merged.append([lo, hi, n])
    return [(float(a), float(b), int(c)) for a, b, c in merged]


def _well_id(x: float, wells: list[dict[str, float]]) -> int:
    for i, w in enumerate(wells):
        if float(w["lo"]) <= x <= float(w["hi"]):
            return i
    return -1


def _cross_hops(labels: list[int], lag: int, shot_lengths: list[int]) -> int:
    n = 0
    off = 0
    for slen in shot_lengths:
        seg = labels[off : off + slen]
        for t in range(len(seg) - lag):
            a, b = seg[t], seg[t + lag]
            if a >= 0 and b >= 0 and a != b:
                n += 1
        off += slen
    return n


def _gap(wells: list[dict[str, float]]) -> dict[str, float] | None:
    if len(wells) < 2:
        return None
    lo = float(wells[0]["hi"])
    hi = float(wells[1]["lo"])
    if hi <= lo:
        return None
    return {"lo": lo, "hi": hi, "width": hi - lo}


def _all_hungry(model: dict[str, Any], unc: dict[str, list[int]]) -> bool:
    obs = model.get("observables") or {}
    if not obs:
        return True
    for name, spec in obs.items():
        n = _n_bins(spec)
        hungry = set(unc.get(name) or [])
        if spec.get("constant") is not None:
            if 0 not in hungry:
                return False
            continue
        if n and hungry != set(range(n)):
            return False
    return True


def _has_extra(extra_map: dict[str, list[list[float]]]) -> bool:
    return any(shots for shots in extra_map.values())


def _pop_shot(extra_map: dict[str, list[list[float]]]) -> tuple[str, list[float]]:
    for name, shots in extra_map.items():
        if shots:
            return name, shots.pop(0)
    raise RefuseError("no extra teacher shot left", code="COVERAGE")


def _int_kw(kwargs: dict[str, object], name: str, default: int) -> int:
    v = kwargs.get(name, default)
    if v is None:
        return default
    return int(v)  # type: ignore[arg-type]


def _fit_1d(
    xs: list[float], *, lag: int, n_bins: int, shot_lengths: list[int] | None = None
) -> dict[str, Any]:
    unique: list[float] = []
    for x in sorted(xs):
        if not unique or x > unique[-1]:
            unique.append(x)
    if len(unique) == 1:
        return {"constant": unique[0], "n_frames": len(xs), "lag": lag, "counts": []}
    if len(unique) <= n_bins:
        index = {v: i for i, v in enumerate(unique)}
        bins = [index[x] for x in xs]
        n = len(unique)
        centers = unique
        edges = unique
    else:
        edges = _quantile_edges(xs, n_bins)
        bins = [_assign(x, edges) for x in xs]
        n = max(bins) + 1
        centers = _bin_centers(edges, n)
    counts = _transition_counts(bins, n, lag, shot_lengths or [len(bins)])
    hops = sum(counts[i][j] for i in range(n) for j in range(n) if i != j)
    wells = _wells(xs, shot_lengths or [len(xs)])
    labels = [_well_id(x, wells) for x in xs]
    cross = _cross_hops(labels, lag, shot_lengths or [len(xs)])
    bridge_missing = len(wells) > 1 and cross == 0
    disconnected = bridge_missing or (hops == 0 and n > 1)
    trans = _row_stochastic(counts)
    if disconnected:
        occ = [0.0] * n
        for b in bins:
            occ[b] += 1.0
        tot = sum(occ) or 1.0
        pi = [c / tot for c in occ]
    else:
        pi = _stationary(trans)
    return {
        "edges": edges,
        "pi": pi,
        "centers": centers,
        "trans": trans,
        "counts": counts,
        "lag": lag,
        "n_frames": len(xs),
        "disconnected": disconnected,
        "wells": wells,
        "gap": _gap(wells),
        "cross_hops": cross,
        "bridge_missing": bridge_missing,
    }


def _transition_counts(
    bins: list[int], n: int, lag: int, shot_lengths: list[int]
) -> list[list[float]]:
    counts = [[0.0] * n for _ in range(n)]
    off = 0
    for slen in shot_lengths:
        seg = bins[off : off + slen]
        for t in range(len(seg) - lag):
            counts[seg[t]][seg[t + lag]] += 1.0
        off += slen
    return counts


def _sample_1d(spec: dict[str, Any], *, n_samples: int, seed: int) -> list[float]:
    if spec.get("constant") is not None:
        return [float(spec["constant"])] * n_samples
    pi = spec["pi"]
    centers = spec["centers"]
    rng = random.Random(seed)
    return [centers[_choice(pi, rng.random())] for _ in range(n_samples)]


def _quantile_edges(xs: list[float], n_bins: int) -> list[float]:
    s = sorted(xs)
    n = len(s)
    edges = [s[min(n - 1, k * (n - 1) // n_bins)] for k in range(n_bins + 1)]
    out = [edges[0]]
    for e in edges[1:]:
        if e > out[-1]:
            out.append(e)
    if len(out) == 1:
        out.append(out[0])
    return out


def _assign(x: float, edges: list[float]) -> int:
    for i in range(1, len(edges) - 1):
        if x <= edges[i]:
            return i - 1
    return len(edges) - 2


def _bin_centers(edges: list[float], n: int) -> list[float]:
    centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)]
    while len(centers) < n:
        centers.append(centers[-1])
    return centers[:n]


def _row_stochastic(counts: list[list[float]]) -> list[list[float]]:
    n = len(counts)
    trans: list[list[float]] = []
    for i, row in enumerate(counts):
        tot = sum(row)
        if tot <= 0:
            stay = [0.0] * n
            stay[i] = 1.0
            trans.append(stay)
            continue
        trans.append([c / tot for c in row])
    return trans


def _stationary(trans: list[list[float]], steps: int = 4000) -> list[float]:
    n = len(trans)
    pi = [1.0 / n] * n
    for _ in range(steps):
        nxt = [0.0] * n
        for i, p in enumerate(pi):
            for j, t in enumerate(trans[i]):
                nxt[j] += p * t
        s = sum(nxt) or 1.0
        pi = [x / s for x in nxt]
    return pi


def _choice(weights: list[float], u: float) -> int:
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if u <= acc:
            return i
    return len(weights) - 1
