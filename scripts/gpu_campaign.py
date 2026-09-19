#!/usr/bin/env python3
"""5EH4 glycophorin: same CHARMM36 HMR 4 fs Hamiltonian, pre vs post sampling.

One process, no install steps. CUDA must already be the OpenMM platform
(see docker/gpu.Dockerfile). Writes runs/5eh4/compare.json.
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
WORKDIR = ROOT / "runs" / "5eh4"
PDB_ID = "5EH4"
CHAINS = "AB"
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


def fetch_pdb(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        log(f"fetch     skip {dest}")
    else:
        script = ROOT / "scripts" / "fetch_pdb.sh"
        subprocess.check_call(["bash", str(script), PDB_ID, str(dest)])
    keep_chains(dest, CHAINS)
    return dest


def keep_chains(path: Path, chains: str) -> None:
    """Keep protein ATOM/TER for one biological copy. Drop crystal lipids (HETATM)."""
    allowed = set(chains)
    out: list[str] = []
    for line in Path(path).read_text().splitlines(keepends=True):
        if line.startswith(("ATOM", "TER")):
            chain = line[21] if len(line) > 21 else ""
            if chain in allowed:
                out.append(line)
            continue
        if line.startswith(("HETATM", "ANISOU", "MODEL", "ENDMDL")):
            continue
        out.append(line)
    Path(path).write_text("".join(out))


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
    oc = (compare.get("oracle_compression") or {}).get("methods") or {}
    if not oc:
        return "incomplete — missing oracle_compression"
    for method, obs in oc.items():
        for d in obs.values():
            acc = d.get("acceleration") or {}
            if acc.get("accelerated"):
                return f"10× acceleration claim ({method})"
            if d.get("leaked") or acc.get("reason") == "oracle leakage":
                return "no acceleration claim — oracle leakage"
    first = next(iter(oc.values()), {})
    obs_name = next(iter(first), None)
    pre_h = ((oc.get("single_long") or {}).get(obs_name) or {}).get("gpu_hours_to_error")
    ctrl_h = ((oc.get("independent_replicas") or {}).get(obs_name) or {}).get("gpu_hours_to_error")
    post_h = ((oc.get("adaptive") or {}).get(obs_name) or {}).get("gpu_hours_to_error")
    if pre_h is None:
        if post_h is not None:
            return "no acceleration claim — method hit ε; conventional baseline did not"
        return "no acceleration claim — conventional never hit oracle ε"
    if post_h is not None and float(post_h) < float(pre_h):
        return "no acceleration claim — below 10× bar (matched ε is required, 10× is the claim)"
    if ctrl_h is not None and float(ctrl_h) < float(pre_h) and (
        post_h is None or float(post_h) >= float(ctrl_h)
    ):
        return "no acceleration claim — replicas beat single_long; adaptive did not (more MD, not a teacher)"
    return "no acceleration claim — method did not beat conventional GPU-hours to oracle ε"


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
        hours = d.get("gpu_hours_to_error")
        return (
            f"{tag:<9}{label:<22} hours_to_ε={hours}  "
            f"U/GPU-h={d.get('ci_width_per_gpu_hour')}  μ={d.get('estimate')}"
        )

    return "\n".join(
        [
            f"COMPRESS  {PDB_ID}  {ns_day:.1f} ns/day  {n_atoms} atoms",
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
    fetch_pdb(pdb)

    from adaptamem.assemble import assemble, have_assembled
    from adaptamem.bench import bench
    from adaptamem.compare import MethodRun, compare, write_compare
    from adaptamem.oracle import freeze, load_oracle
    from adaptamem.produce import produce
    from adaptamem.session import load_session
    from adaptamem.sim import cuda_available

    session = load_session(YAML)
    cuda = cuda_available()
    if (WORKDIR / "eq.pdb").is_file():
        log(f"assemble  skip {WORKDIR / 'eq.pdb'}")
    elif have_assembled(WORKDIR):
        log(f"assemble  skip shipped {WORKDIR / 'assembled.pdb'}")
        if not cuda:
            log("ASSEMBLE_DONE  CPU assembled system is ready to ship; no CUDA here")
            return 0
    elif cuda:
        log(
            "CAMPAIGN_FAIL  no assembled.pdb; addMembrane is CPU work. "
            "Assemble on a CPU host and ship assembled.pdb + system.xml."
        )
        print("CAMPAIGN_FAIL", flush=True)
        return 2
    else:
        assemble(session, WORKDIR, force=True, progress=log)
        log("ASSEMBLE_DONE  ship assembled.pdb system.xml assemble.json; GPU eq is a later step")
        return 0

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
    oracle_run = _load(WORKDIR / "oracle_run.json")

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
    payload = compare(oracle, runs, precision=PRECISION, error=PRECISION, oracle_gpu_hours=float(oracle_run.get("gpu_hours") or 0.0) or None)
    payload["bench"] = {"ns_per_day": ns_day, "n_atoms": n_atoms}
    payload["plan"] = {"oracle_ns": oracle_ns, "method_ns": method_ns, "max_gpu_hours": MAX_GPU_HOURS}
    payload["refuse"] = refused
    decision = verdict(payload, refused)
    payload["verdict"] = decision
    write_compare(payload, WORKDIR / "compare.json")

    eqm = (payload.get("equal_compute") or {}).get("methods") or {}
    ocm = (payload.get("oracle_compression") or {}).get("methods") or {}

    def _arm(name: str) -> dict[str, Any]:
        row = dict((eqm.get(name) or {}).get("g83_sep") or {})
        row.update((ocm.get(name) or {}).get("g83_sep") or {})
        return row

    block = format_block(
        ns_day,
        n_atoms,
        oracle.to_dict() | {"observables": oracle.estimates},
        {
            "single_long": _arm("single_long"),
            "independent_replicas": _arm("independent_replicas"),
            "adaptive": _arm("adaptive"),
        },
        decision,
    )
    print(block, flush=True)
    print("CAMPAIGN_DONE", flush=True)
    return 0 if refused is None else 2


def _shutdown(reason: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from runpod_watchdog import terminate_self
    except ImportError:
        log(f"watchdog  skip import ({reason})")
        return
    terminate_self(reason)


if __name__ == "__main__":
    try:
        code = main()
        _shutdown("campaign_done")
        raise SystemExit(code)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("CAMPAIGN_FAIL", flush=True)
        _shutdown("campaign_fail")
        raise SystemExit(0)
