from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from adaptamem.assemble import assemble, default_workdir, format_assemble
from adaptamem.box import format_box
from adaptamem.campaign import reproduce_report
from adaptamem.doctor import audit, format_report
from adaptamem.errors import RefuseError
from adaptamem.orient import format_orient
from adaptamem.schema import default_system_template_path, load_protocol, load_system
from adaptamem.session import load_session
from adaptamem.strategy import format_strategy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adaptamem",
        description=(
            "Decide the cheapest membrane-protein simulation that can answer "
            "the objective, then run the Phase 1 OpenMM path."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Write a blank system YAML")
    p_init.add_argument("out", type=Path)

    p_doc = sub.add_parser("doctor", help="Structure report from a PDB")
    p_doc.add_argument("structure", type=Path)

    p_plan = sub.add_parser("plan", help="Doctor + cheapest box + next experiment")
    _job_args(p_plan)

    p_val = sub.add_parser("validate", help="Load a system YAML")
    p_val.add_argument("system", type=Path)

    p_asm = sub.add_parser("assemble", help="Orient + compact OpenMM membrane + 4 fs HMR system")
    _job_args(p_asm)
    p_asm.add_argument("--out", type=Path, default=None)
    p_asm.add_argument("--force", action="store_true", help="Override doctor ACTION items")

    p_eq = sub.add_parser("equilibrate", help="Minimize + 4 fs eq with membrane QC")
    p_eq.add_argument("workdir", type=Path)
    p_eq.add_argument("--short", action="store_true", help="Minimize + 100 steps (tests / smoke)")

    p_bench = sub.add_parser("bench", help="Throughput of the assembled system (not science)")
    p_bench.add_argument("workdir", type=Path)
    p_bench.add_argument("--steps", type=int, default=None)
    p_bench.add_argument(
        "--ladder",
        action="store_true",
        help="Append this workdir as one ns/day(N) point to ladder.json",
    )

    p_prod = sub.add_parser("produce", help="Production MD; streaming CVs, XTC secondary")
    p_prod.add_argument("workdir", type=Path)
    p_prod.add_argument("--ns", type=float, default=None, help="Hard cap on simulated ns")
    p_prod.add_argument("--budget-hours", type=float, default=None)
    p_prod.add_argument("--force", action="store_true", help="Ignore membrane QC refuse")

    p_run = sub.add_parser("run", help="Plan + assemble + equilibrate")
    _job_args(p_run)
    p_run.add_argument("--out", type=Path, default=None)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--short", action="store_true")
    p_run.add_argument("--bench", action="store_true")
    p_run.add_argument("--produce", action="store_true")
    p_run.add_argument("--ns", type=float, default=None)

    p_sample = sub.add_parser(
        "sample",
        help="Adaptive loop: initialize→pilot→chunk→analyze→select→branch→stop",
    )
    _job_args(p_sample)
    p_sample.add_argument("--out", type=Path, default=None)
    p_sample.add_argument("--traces", type=Path, default=None, help="JSON observable series")
    p_sample.add_argument(
        "--select",
        default="random",
        help="random | uncertainty | novelty | information_gain",
    )
    p_sample.add_argument("--execute", action="store_true", help="Run OpenMM chunks")
    p_sample.add_argument("--chunk-ns", type=float, default=None)
    p_sample.add_argument("--ns", type=float, default=None, help="Cap on executed ns")

    p_hyb = sub.add_parser("hybrid", help="AA/CG plan; annular lipids stay AA")
    _job_args(p_hyb)
    p_hyb.add_argument("--out", type=Path, default=None)
    p_hyb.add_argument("--bulk", default="CG", help="CG | AA | implicit")

    p_repro = sub.add_parser("reproduce", help="Replay campaign hashes + protocol")
    p_repro.add_argument("campaign", help="campaign_id or workdir")

    p_of = sub.add_parser("oracle-freeze", help="Freeze conventional traces as held-out AA oracle")
    p_of.add_argument("workdir", type=Path)
    p_of.add_argument("--out", type=Path, default=None)

    p_os = sub.add_parser("oracle-score", help="Score estimates against a frozen oracle")
    p_os.add_argument("estimate", type=Path, help="JSON with diagnostics or values")
    p_os.add_argument("--oracle", type=Path, required=True)

    p_cmp = sub.add_parser("compare", help="Equal GPU-hours and equal precision vs oracle")
    p_cmp.add_argument("--oracle", type=Path, required=True)
    p_cmp.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable. NAME is single_long | independent_replicas | adaptive | random_branch",
    )
    p_cmp.add_argument("--out", type=Path, default=None)
    p_cmp.add_argument("--demo", action="store_true", help="Synthetic fixture (tests / docs)")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "init":
            return _init(args.out)
        if args.cmd == "doctor":
            return _doctor(args.structure)
        if args.cmd == "plan":
            return _plan(args.input, args.objective, args.budget_hours)
        if args.cmd == "validate":
            return _validate(args.system)
        if args.cmd == "assemble":
            return _assemble(args)
        if args.cmd == "equilibrate":
            return _equilibrate(args.workdir, args.short)
        if args.cmd == "bench":
            return _bench(args.workdir, args.steps, args.ladder)
        if args.cmd == "produce":
            return _produce(args)
        if args.cmd == "run":
            return _run(args)
        if args.cmd == "sample":
            return _sample(args)
        if args.cmd == "hybrid":
            return _hybrid(args)
        if args.cmd == "reproduce":
            return _reproduce(args.campaign)
        if args.cmd == "oracle-freeze":
            return _oracle_freeze(args.workdir, args.out)
        if args.cmd == "oracle-score":
            return _oracle_score(args.estimate, args.oracle)
        if args.cmd == "compare":
            return _compare(args)
    except RefuseError as exc:
        print(exc.format(), file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.cmd}")
    return 2


def _job_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="PDB or system YAML")
    p.add_argument(
        "--objective",
        default=None,
        help="conventional | conformational-shift | discover-states | comparison | membrane-environment",
    )
    p.add_argument("--budget-hours", type=float, default=None)


def _init(out: Path) -> int:
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(default_system_template_path(), out)
    print(out)
    return 0


def _doctor(path: Path) -> int:
    print(format_report(audit(path)))
    return 0


def _plan(path: Path, cli_objective: str | None, budget_hours: float | None = None) -> int:
    session = load_session(path, cli_objective=cli_objective, budget_hours=budget_hours)
    print(format_report(session.report))
    print()
    print(format_box(session.box))
    print()
    print(format_strategy(session.strategy))
    print()
    print("EXPERIMENT")
    print(f"objective     {session.objective.type}")
    if session.budget_hours is not None:
        print(f"budget        {session.budget_hours:g} GPU-h")
    if session.objective.observables:
        for o in session.objective.observables:
            prec = f"  precision={o.precision}" if o.precision is not None else ""
            print(f"  observable  {o.name} ({o.kind}){prec}")
    elif session.objective.discover_cvs:
        print("  CVs from a short pilot (TICA/PCA) — not user-specified")
    else:
        print("  conventional single trajectory")
    print("walker count and ns/walker are scheduler outputs, not inputs")
    return 0 if session.strategy.ok else 2


def _assemble(args: argparse.Namespace) -> int:
    session = load_session(
        args.input, cli_objective=args.objective, budget_hours=args.budget_hours
    )
    print(format_report(session.report))
    print()
    print(format_box(session.box))
    print()
    print(format_strategy(session.strategy))
    print()
    workdir = default_workdir(session, args.out)
    result = assemble(session, workdir, force=args.force, progress=_progress)
    print()
    print(format_orient(result.orient))
    print()
    print(format_assemble(result))
    return 0


def _equilibrate(workdir: Path, short: bool) -> int:
    from adaptamem.equilibrate import equilibrate, format_eq

    result = equilibrate(workdir, short=short, progress=_progress)
    print(format_eq(result))
    return 0 if result.qc.ok or short else 1


def _bench(workdir: Path, steps: int | None, ladder: bool) -> int:
    from adaptamem.bench import bench, format_bench, write_ladder

    r = bench(workdir, steps=steps)
    print(format_bench(r))
    if ladder:
        row = {
            "n_atoms": r.n_atoms,
            "ns_per_day": r.ns_per_day,
            "gpu": (r.payload.get("hardware") or {}).get("gpu"),
            "workdir": str(workdir),
        }
        path = Path(workdir) / "ladder.json"
        existing: list = []
        if path.is_file():
            existing = list((json.loads(path.read_text()).get("points")) or [])
        existing.append(row)
        write_ladder(existing, path)
        print(f"ladder    {path}")
    return 0


def _produce(args: argparse.Namespace) -> int:
    from adaptamem.produce import format_produce, produce

    r = produce(
        args.workdir,
        ns=args.ns,
        budget_hours=args.budget_hours,
        force=args.force,
        progress=_progress,
    )
    print(format_produce(r))
    return 0


def _run(args: argparse.Namespace) -> int:
    from adaptamem.equilibrate import equilibrate, format_eq

    session = load_session(
        args.input, cli_objective=args.objective, budget_hours=args.budget_hours
    )
    session.gate_run(force=args.force)
    print(format_report(session.report))
    print()
    print(format_box(session.box))
    print()
    print(format_strategy(session.strategy))
    print()
    workdir = default_workdir(session, args.out)
    result = assemble(session, workdir, force=args.force, progress=_progress)
    print()
    print(format_assemble(result))
    print()
    eq = equilibrate(workdir, short=args.short, progress=_progress)
    print(format_eq(eq))
    if args.bench:
        from adaptamem.bench import bench, format_bench

        print()
        print(format_bench(bench(workdir)))
    if args.produce:
        from adaptamem.produce import format_produce, produce

        print()
        print(
            format_produce(
                produce(
                    workdir,
                    ns=args.ns,
                    budget_hours=args.budget_hours,
                    force=args.force or args.short,
                    progress=_progress,
                )
            )
        )
    return 0 if eq.qc.ok or args.short else 1


def _sample(args: argparse.Namespace) -> int:
    from adaptamem.sample import format_sample, sample

    session = load_session(
        args.input, cli_objective=args.objective, budget_hours=args.budget_hours
    )
    traces = None
    if args.traces is not None:
        traces = json.loads(Path(args.traces).read_text())
    workdir = default_workdir(session, args.out)
    result = sample(
        session.objective,
        session.box,
        workdir,
        budget_hours=args.budget_hours if args.budget_hours is not None else session.budget_hours,
        traces=traces,
        select=args.select,
        execute=args.execute,
        chunk_ns=args.chunk_ns,
        ns_cap=args.ns,
        seed=session.system.seed,
        progress=_progress,
    )
    print(format_sample(result))
    return 0


def _hybrid(args: argparse.Namespace) -> int:
    from adaptamem.hybrid import format_hybrid, plan_hybrid, write_hybrid

    session = load_session(
        args.input, cli_objective=args.objective, budget_hours=args.budget_hours
    )
    plan = plan_hybrid(session.objective, session.strategy, bulk=args.bulk)
    workdir = default_workdir(session, args.out)
    write_hybrid(plan, workdir, extra={"objective": session.objective.type})
    print(format_hybrid(plan))
    print(f"wrote {workdir / 'hybrid.json'}")
    return 0 if plan.ok else 2


def _reproduce(target: str) -> int:
    print(reproduce_report(target, protocol=load_protocol()))
    return 0


def _oracle_freeze(workdir: Path, out: Path | None) -> int:
    from adaptamem.oracle import freeze_workdir

    oracle = freeze_workdir(workdir, out=out)
    print(f"ORACLE  froze { {k: len(v) for k, v in oracle.values.items()} }")
    print(oracle.path)
    return 0


def _oracle_score(estimate: Path, oracle_path: Path) -> int:
    from adaptamem.oracle import format_score, load_oracle, score

    oracle = load_oracle(oracle_path)
    data = json.loads(Path(estimate).read_text())
    est = data.get("diagnostics") or data.get("observables") or data
    print(format_score(score(est, oracle)))
    return 0


def _compare(args: argparse.Namespace) -> int:
    from adaptamem.compare import (
        compare,
        format_compare,
        load_method,
        synthetic_demo,
        write_compare,
    )
    from adaptamem.oracle import load_oracle

    if args.demo:
        oracle, runs = synthetic_demo()
    else:
        oracle = load_oracle(args.oracle)
        runs = []
        for item in args.method:
            if "=" not in item:
                raise RefuseError("--method needs NAME=PATH", code="NOT_READY")
            name, path = item.split("=", 1)
            runs.append(load_method(Path(path), name=name))
        if len(runs) < 2:
            raise RefuseError("compare needs at least two --method NAME=PATH", code="NOT_READY")
    payload = compare(oracle, runs)
    print(format_compare(payload))
    dest = args.out
    if dest is None and not args.demo:
        dest = Path("compare.json")
    if dest is not None:
        write_compare(payload, dest)
        print(f"wrote {dest}")
    return 0


def _validate(system_path: Path) -> int:
    system = load_system(system_path)
    lipids = " ".join(f"{k}:{v:g}" for k, v in system.membrane.lipids.items())
    print(f"system     {system.name}")
    print(f"structure  {system.structure}")
    print(f"objective  {system.objective.type}")
    print(f"discover   {system.objective.discover_cvs}")
    print(f"obs        {len(system.objective.observables)}")
    print(f"lipids     {lipids}")
    print(f"size       optimize={system.membrane.optimize_size}")
    print(f"budget     gpu-h={system.compute.max_gpu_hours} wall-h={system.compute.max_wall_hours}")
    return 0


def _progress(msg: str) -> None:
    print(f"  … {msg}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
