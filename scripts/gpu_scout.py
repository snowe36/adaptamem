#!/usr/bin/env python3
"""b2AR 3P0G Phase 0 scout: 20 ns conventional on tm6_ic.

Not an oracle. Not a method. If the CV span is already within eps, abort.
T4L lock / easy well, do not buy 1 us. Prints SCOUT_GO or SCOUT_ABORT, then
self-terminates the Runpod when RUNPOD_API_KEY is in the env.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKDIR = ROOT / "runs" / "3p0g_scout"
PDB_ID = "3P0G"
CHAINS = "A"
SCOUT_NS = 20.0
PRECISION = 0.2
MAX_GPU_HOURS = 3.0
OBS = "tm6_ic"

sys.path.insert(0, str(ROOT / "src"))


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_pdb(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        log(f"fetch     skip {dest}")
        return dest
    url = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    req = urllib.request.Request(url, headers={"User-Agent": "adaptamem-scout/1.0"})
    log(f"fetch     {url}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def keep_chains(path: Path, chains: str) -> None:
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


def next_stage(*, assembled: bool, cuda: bool) -> str:
    """CPU packs the membrane. CUDA only sees a shipped assembled system."""
    if assembled and cuda:
        return "eq"
    if assembled and not cuda:
        return "ship"
    if not assembled and cuda:
        return "refuse"
    return "assemble"


def span_of(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values) - min(values))


def scout_verdict(span: float | None, precision: float = PRECISION) -> str:
    if span is None:
        return "SCOUT_FAIL  no tm6_ic series"
    if span < precision:
        return (
            f"SCOUT_ABORT  tm6_ic span {span:.4f} nm < eps={precision} nm "
            "(easy well or T4L lock; 1 us oracle forbidden)"
        )
    return (
        f"SCOUT_GO  tm6_ic span {span:.4f} nm >= eps={precision} nm "
        "(landscape is not an easy well; held-out oracle is allowed)"
    )


def _run_eq(workdir: Path) -> None:
    from adaptamem.equilibrate import equilibrate

    try:
        equilibrate(workdir, short=False, progress=log)
        return
    except Exception as exc:  # noqa: BLE001
        log(f"full eq failed ({exc}); --short fallback")
        equilibrate(workdir, short=True, progress=log)


def terminate_self(reason: str) -> bool:
    pod_id = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    log(f"watchdog  {reason}")
    if not pod_id or not api_key:
        log("WATCHDOG_CANNOT_KILL missing RUNPOD_POD_ID or RUNPOD_API_KEY")
        return False
    req = urllib.request.Request(
        f"https://rest.runpod.io/v2/pods/{pod_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            log(f"watchdog  terminated {pod_id} HTTP {resp.status}")
            return True
    except urllib.error.HTTPError as exc:
        log(f"watchdog  terminate failed HTTP {exc.code} {exc.reason}")
        return False
    except urllib.error.URLError as exc:
        log(f"watchdog  terminate failed {exc.reason}")
        return False


def main() -> int:
    os.chdir(ROOT)
    t0 = time.perf_counter()
    WORKDIR.mkdir(parents=True, exist_ok=True)

    pdb = ROOT / "data" / "structures" / f"{PDB_ID}.pdb"
    fetch_pdb(pdb)
    keep_chains(pdb, CHAINS)

    yaml_path = WORKDIR / "system.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "name: b2ar_3p0g_scout",
                f"structure: {pdb}",
                "objective:",
                "  type: conformational_shift",
                "  observables:",
                f"    - name: {OBS}",
                "      kind: distance",
                '      selection: "name CA and resid 131 ; name CA and resid 272"',
                f"      precision: {PRECISION}",
                "orientation:",
                "  method: auto",
                "  topology: out",
                "membrane:",
                "  lipids:",
                "    POPC: 1.0",
                "  optimize_size: true",
                "  safety_margin_nm: 1.2",
                "  water_pad_nm: 1.2",
                "  ionic_strength_M: 0.15",
                "compute:",
                f"  max_gpu_hours: {MAX_GPU_HOURS}",
                "seed: 42",
                "",
            ]
        )
    )

    from adaptamem.assemble import assemble, have_assembled
    from adaptamem.bench import bench
    from adaptamem.produce import produce
    from adaptamem.session import load_session
    from adaptamem.sim import cuda_available

    session = load_session(yaml_path)
    assembled = have_assembled(WORKDIR)
    cuda = cuda_available()
    stage = next_stage(assembled=assembled, cuda=cuda)
    if stage == "refuse":
        log(
            "SCOUT_FAIL  no assembled.pdb; addMembrane is CPU work. "
            "Run this script on a CPU host, then ship assembled.pdb + system.xml."
        )
        print("SCOUT_FAIL", flush=True)
        return 2
    if stage == "assemble":
        assemble(session, WORKDIR, force=True, progress=log)
        log("ASSEMBLE_DONE  ship assembled.pdb system.xml assemble.json; GPU eq is a later step")
        return 0
    if stage == "ship":
        log("ASSEMBLE_DONE  CPU assembled system is ready to ship; no CUDA here")
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
        payload = json.loads((WORKDIR / "bench.json").read_text())
        perf = payload.get("performance") or payload
        ns_day = float(perf.get("ns_per_day") or 0.0)
        n_atoms = int(payload.get("n_atoms") or 0)

    elapsed_h = (time.perf_counter() - t0) / 3600.0
    remaining = max(0.05, MAX_GPU_HOURS - elapsed_h)
    scout_ns = SCOUT_NS
    if ns_day > 0:
        cap = remaining * ns_day / 24.0
        scout_ns = min(SCOUT_NS, cap)
    log(f"clock     {ns_day:.1f} ns/day  {n_atoms} atoms")
    log(f"plan      scout {scout_ns:.3f} ns  remaining {remaining:.2f} GPU-h")

    traces: dict[str, list[float]] = {}
    gpu_hours = 0.0
    ran_ns = 0.0
    if not (WORKDIR / "scout.json").is_file():
        pr = produce(WORKDIR, ns=scout_ns, seed=42, force=True, progress=log)
        traces = pr.traces or {}
        gpu_hours = float(pr.gpu_hours or 0.0)
        ran_ns = float(pr.ns or scout_ns)
        (WORKDIR / "scout.json").write_text(
            json.dumps(
                {
                    "method": "conventional_scout",
                    "values": traces,
                    "gpu_hours": gpu_hours,
                    "ns": ran_ns,
                    "cpu_hours": 0.0,
                    "role": "phase0_scout",
                    "leaked": False,
                },
                indent=2,
            )
            + "\n"
        )
    else:
        data = json.loads((WORKDIR / "scout.json").read_text())
        traces = data.get("values") or {}
        gpu_hours = float(data.get("gpu_hours") or 0.0)
        ran_ns = float(data.get("ns") or 0.0)

    vals = list(traces.get(OBS) or [])
    span = span_of(vals)
    mu = (sum(vals) / len(vals)) if vals else None
    decision = scout_verdict(span, PRECISION)
    summary: dict[str, Any] = {
        "pdb": PDB_ID,
        "ns_per_day": ns_day,
        "n_atoms": n_atoms,
        "scout_ns": ran_ns,
        "gpu_hours": gpu_hours,
        "observable": OBS,
        "n": len(vals),
        "mean": mu,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "span": span,
        "precision": PRECISION,
        "verdict": decision.split()[0],
        "leaked": False,
    }
    (WORKDIR / "scout_verdict.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(
        f"SCOUT     {PDB_ID}  {ns_day:.1f} ns/day  {n_atoms} atoms  "
        f"ns={ran_ns:.2f}  span={span}  mu={mu}"
    )
    log(decision)
    log("SCOUT_DONE")
    return 0 if decision.startswith("SCOUT_GO") or decision.startswith("SCOUT_ABORT") else 2


if __name__ == "__main__":
    try:
        code = main()
        terminate_self("scout_done")
        raise SystemExit(code)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("SCOUT_FAIL", flush=True)
        terminate_self("scout_fail")
        raise SystemExit(0)
