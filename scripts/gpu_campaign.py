#!/usr/bin/env python3
"""1AFO glycophorin: same CHARMM36 HMR 4 fs Hamiltonian, pre vs post sampling.

One process, no install steps. CUDA must already be the OpenMM platform
(see docker/gpu.Dockerfile). Writes runs/1afo/compare.json.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "examples" / "glycophorin.yaml"
WORKDIR = ROOT / "runs" / "1afo"
PDB_ID = "1AFO"
ORACLE_NS = 20.0
METHOD_NS = 5.0
MAX_GPU_HOURS = 4.0
PRECISION = 0.05
STAGING = (
    "eq.pdb",
    "assembled.pdb",
    "system.xml",
    "assemble.json",
    "eq.json",
    "bench.json",
    "campaign.json",
)

sys.path.insert(0, str(ROOT / "src"))


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_1afo(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        log(f"fetch     skip {dest}")
        return dest
    script = ROOT / "scripts" / "fetch_pdb.sh"
    subprocess.check_call(["bash", str(script), PDB_ID, str(dest)])
    return dest


def plan_lengths(ns_day: float, remaining_hours: float) -> tuple[float, float]:
    """Oracle ~20 ns, methods ~5 ns, scaled to the remaining GPU-hour cap."""
    if ns_day <= 0 or remaining_hours <= 0:
        return 0.0, 0.0
    oracle_h = ORACLE_NS / ns_day * 24.0
    methods_h = 3.0 * METHOD_NS / ns_day * 24.0
    need = oracle_h + methods_h
    scale = 1.0 if need <= remaining_hours else remaining_hours / need
    return ORACLE_NS * scale, METHOD_NS * scale


def stage(src: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in STAGING:
        p = src / name
        if p.is_file():
            shutil.copy2(p, dest / name)
    return dest


def dump_method(path: Path, name: str, traces: dict[str, list[float]], gpu_hours: float, ns: float, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "method": name,
        "values": traces,
        "traces": traces,
        "gpu_hours": gpu_hours,
        "ns": ns,
        "trajectory_ns": ns,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def traces_from_produce(workdir: Path) -> dict[str, list[float]]:
    from adaptamem.oracle import _traces_from_workdir

    return _traces_from_workdir(workdir)


def traces_from_sample(workdir: Path) -> dict[str, list[float]]:
    data = _load(workdir / "sample.json")
    values = data.get("values") or {}
    if values:
        return {k: list(v) for k, v in values.items() if isinstance(v, list)}
    walkers = data.get("walkers") or {}
    out: dict[str, list[float]] = {}
    for named in walkers.values():
        if not isinstance(named, dict):
            continue
        for k, v in named.items():
            if isinstance(v, list):
                out.setdefault(k, []).extend(v)
    return out


def verdict(compare: dict[str, Any], refused: str | None) -> str:
    if refused:
        return f"REFUSE ({refused})"
    eq = (compare.get("equal_compute") or {}).get("methods") or {}
    pre = (eq.get("single_long") or {}).get("g83_sep") or {}
    ctrl = (eq.get("independent_replicas") or {}).get("g83_sep") or {}
    post = (eq.get("adaptive") or {}).get("g83_sep") or {}
    pre_u = pre.get("ci_width_per_gpu_hour")
    ctrl_u = ctrl.get("ci_width_per_gpu_hour")
    post_u = post.get("ci_width_per_gpu_hour")
    in_ci = post.get("oracle_mean_in_ci")
    rec_pre = pre.get("bin_recovery")
    rec_post = post.get("bin_recovery")
    if post_u is None or pre_u is None:
        return "incomplete — missing U/GPU-h"
    post_beats_pre = float(post_u) < float(pre_u)
    covered = bool(in_ci) or (
        rec_post is not None and rec_pre is not None and float(rec_post) > float(rec_pre)
    )
    if post_beats_pre and covered:
        return "post wins (adaptive lower U/GPU-h than single_long; oracle mean in CI or better recovery)"
    if ctrl_u is not None and float(ctrl_u) < float(pre_u) and not (
        post_u is not None and float(post_u) < float(ctrl_u)
    ):
        return "no acceleration — replicas beat single_long; adaptive did not beat replicas"
    return "post did not beat pre at equal GPU-hours"


def format_block(
    ns_day: float,
    n_atoms: int,
    oracle: dict[str, Any],
    methods: dict[str, dict[str, Any]],
    decision: str,
) -> str:
    o = (oracle.get("observables") or {}).get("g83_sep") or oracle.get("g83_sep") or {}
    pre = methods.get("single_long") or {}
    ctrl = methods.get("independent_replicas") or {}
    post = methods.get("adaptive") or {}

    def _row(tag: str, label: str, d: dict[str, Any]) -> str:
        return (
            f"{tag:<9}{label:<22} U/GPU-h={d.get('ci_width_per_gpu_hour')}  "
            f"CI={d.get('ci95')}  μ={d.get('estimate')}"
        )

    return "\n".join(
        [
            f"PRE/POST  1AFO g83_sep  {ns_day:.1f} ns/day  {n_atoms} atoms",
            f"oracle    μ={o.get('estimate')}  CI={o.get('ci95')}",
            _row("pre", "single_long", pre),
            _row("ctrl", "independent_replicas", ctrl),
            _row("post", "adaptive", post),
            f"verdict   {decision}",
        ]
    )


def _run_eq(workdir: Path) -> None:
    from adaptamem.equilibrate import equilibrate

    try:
        equilibrate(workdir, short=False, progress=log)
        return
    except Exception as exc:  # noqa: BLE001
        log(f"full eq failed ({exc}); --short fallback")
        equilibrate(workdir, short=True, progress=log)


def _sample_arm(
    session: Any,
    src: Path,
    dest: Path,
    *,
    select: str,
    ns: float,
    seed: int,
    chunk_ns: float,
    budget_hours: float | None,
) -> tuple[dict[str, list[float]], float, float, str | None]:
    from adaptamem.errors import RefuseError
    from adaptamem.sample import sample

    stage(src, dest)
    refused = None
    try:
        result = sample(
            session.objective,
            session.box,
            dest,
            budget_hours=None,
            select=select,
            execute=True,
            chunk_ns=chunk_ns,
            ns_cap=ns,
            seed=seed,
            progress=log,
            execute_hours=budget_hours,
        )
        traces = traces_from_sample(dest) or _flatten_walkers(result.traces)
        hours = float(result.gpu_hours or 0.0)
        spent = float(result.spent_ns or ns)
        return traces, hours, spent, None
    except RefuseError as exc:
        refused = exc.code
        log(exc.format())
        traces = traces_from_sample(dest)
        data = _load(dest / "sample.json")
        hours = float(data.get("gpu_hours") or 0.0)
        spent = float(data.get("spent_ns") or ns)
        return traces, hours, spent, refused


def _flatten_walkers(walkers: dict[str, dict[str, list[float]]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for named in walkers.values():
        for k, v in named.items():
            out.setdefault(k, []).extend(v)
    return out


def main() -> int:
    os.chdir(ROOT)
    t0 = time.perf_counter()
    WORKDIR.mkdir(parents=True, exist_ok=True)
    refused: str | None = None

    pdb = ROOT / "data" / "structures" / f"{PDB_ID}.pdb"
    fetch_1afo(pdb)

    from adaptamem.assemble import assemble
    from adaptamem.bench import bench
    from adaptamem.compare import MethodRun, compare, write_compare
    from adaptamem.oracle import freeze, load_oracle
    from adaptamem.produce import produce
    from adaptamem.session import load_session

    session = load_session(YAML)
    if not (WORKDIR / "system.xml").is_file():
        assemble(session, WORKDIR, progress=log)
    else:
        log(f"assemble  skip {WORKDIR / 'system.xml'}")

    if not (WORKDIR / "eq.pdb").is_file():
        _run_eq(WORKDIR)
    else:
        log(f"eq        skip {WORKDIR / 'eq.pdb'}")

    if not (WORKDIR / "bench.json").is_file():
        br = bench(WORKDIR)
        ns_day = float(br.ns_per_day)
        n_atoms = int(br.n_atoms)
    else:
        log(f"bench     skip {WORKDIR / 'bench.json'}")
        payload = _load(WORKDIR / "bench.json")
        perf = payload.get("performance") or payload
        ns_day = float(perf.get("ns_per_day") or payload.get("ns_per_day") or 0.0)
        n_atoms = int(payload.get("n_atoms") or (payload.get("system") or {}).get("atoms") or 0)

    elapsed_h = (time.perf_counter() - t0) / 3600.0
    remaining = max(0.05, MAX_GPU_HOURS - elapsed_h)
    oracle_ns, method_ns = plan_lengths(ns_day, remaining)
    log(f"clock     {ns_day:.1f} ns/day  {n_atoms} atoms")
    log(f"plan      oracle {oracle_ns:.3f} ns  methods {method_ns:.3f} ns  remaining {remaining:.2f} GPU-h")

    oracle_dir = WORKDIR / "oracle"
    if not (WORKDIR / "oracle.json").is_file():
        stage(WORKDIR, oracle_dir)
        pr = produce(oracle_dir, ns=oracle_ns, seed=42, force=True, progress=log)
        traces = pr.traces or traces_from_produce(oracle_dir)
        freeze(traces, WORKDIR / "oracle.json", extra={"ns": pr.ns, "gpu_hours": pr.gpu_hours, "seed": 42})
        dump_method(
            WORKDIR / "oracle_run.json",
            "oracle",
            traces,
            pr.gpu_hours,
            pr.ns,
            extra={"seed": 42, "role": "held_out"},
        )
    oracle = load_oracle(WORKDIR / "oracle.json")

    chunk_ns = min(0.5, max(0.05, method_ns / 4.0))
    method_hours = method_ns / ns_day * 24.0 if ns_day else 0.0

    sl_dir = WORKDIR / "single_long"
    if not (WORKDIR / "single_long.json").is_file():
        stage(WORKDIR, sl_dir)
        pr = produce(sl_dir, ns=method_ns, seed=1, force=True, progress=log)
        dump_method(
            WORKDIR / "single_long.json",
            "single_long",
            pr.traces or traces_from_produce(sl_dir),
            pr.gpu_hours,
            pr.ns,
            extra={"seed": 1},
        )
    sl = _load(WORKDIR / "single_long.json")

    # control: independent replicas
    rep_dir = WORKDIR / "replicas"
    if not (WORKDIR / "replicas.json").is_file():
        traces, hours, spent, code = _sample_arm(
            session, WORKDIR, rep_dir, select="random", ns=method_ns, seed=2, chunk_ns=chunk_ns, budget_hours=method_hours
        )
        dump_method(WORKDIR / "replicas.json", "independent_replicas", traces, hours, spent, extra={"seed": 2, "select": "random", "refuse": code})
        if code:
            refused = code
    rep = _load(WORKDIR / "replicas.json")

    # post: adaptive
    ad_dir = WORKDIR / "adaptive"
    if not (WORKDIR / "adaptive.json").is_file():
        traces, hours, spent, code = _sample_arm(
            session, WORKDIR, ad_dir, select="uncertainty", ns=method_ns, seed=3, chunk_ns=chunk_ns, budget_hours=method_hours
        )
        dump_method(WORKDIR / "adaptive.json", "adaptive", traces, hours, spent, extra={"seed": 3, "select": "uncertainty", "refuse": code})
        if code:
            refused = code
    ad = _load(WORKDIR / "adaptive.json")

    runs = [
        MethodRun("single_long", sl.get("values") or {}, float(sl.get("gpu_hours") or 0.0), sl.get("ns")),
        MethodRun("independent_replicas", rep.get("values") or {}, float(rep.get("gpu_hours") or 0.0), rep.get("ns")),
        MethodRun("adaptive", ad.get("values") or {}, float(ad.get("gpu_hours") or 0.0), ad.get("ns")),
    ]
    payload = compare(oracle, runs, precision=PRECISION)
    payload["bench"] = {"ns_per_day": ns_day, "n_atoms": n_atoms}
    payload["plan"] = {"oracle_ns": oracle_ns, "method_ns": method_ns, "max_gpu_hours": MAX_GPU_HOURS}
    payload["refuse"] = refused
    decision = verdict(payload, refused)
    payload["verdict"] = decision
    write_compare(payload, WORKDIR / "compare.json")

    eqm = (payload.get("equal_compute") or {}).get("methods") or {}
    block = format_block(
        ns_day,
        n_atoms,
        oracle.to_dict() | {"observables": oracle.estimates},
        {
            "single_long": (eqm.get("single_long") or {}).get("g83_sep") or {},
            "independent_replicas": (eqm.get("independent_replicas") or {}).get("g83_sep") or {},
            "adaptive": (eqm.get("adaptive") or {}).get("g83_sep") or {},
        },
        decision,
    )
    print(block, flush=True)
    print("CAMPAIGN_DONE", flush=True)
    return 0 if refused is None else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
