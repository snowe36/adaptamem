from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from adaptamem.assemble import assemble, default_workdir, format_assemble
from adaptamem.box import format_box
from adaptamem.doctor import audit, format_report
from adaptamem.errors import RefuseError
from adaptamem.orient import format_orient
from adaptamem.schema import default_system_template_path, load_system
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

    p_bench = sub.add_parser("bench", help="ns/day of the assembled system")
    p_bench.add_argument("workdir", type=Path)
    p_bench.add_argument("--steps", type=int, default=None)

    p_run = sub.add_parser("run", help="Plan + assemble + equilibrate")
    _job_args(p_run)
    p_run.add_argument("--out", type=Path, default=None)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--short", action="store_true")
    p_run.add_argument("--bench", action="store_true")

    p_sample = sub.add_parser("sample", help="Walker schedule (YAML CVs; n_walkers is an output)")
    _job_args(p_sample)
    p_sample.add_argument("--out", type=Path, default=None)
    p_sample.add_argument("--traces", type=Path, default=None, help="JSON map of observable → values")

    p_hyb = sub.add_parser("hybrid", help="AA/CG plan; annular lipids stay AA")
    _job_args(p_hyb)
    p_hyb.add_argument("--out", type=Path, default=None)
    p_hyb.add_argument("--bulk", default="CG", help="CG | AA | implicit")

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
            return _bench(args.workdir, args.steps)
        if args.cmd == "run":
            return _run(args)
        if args.cmd == "sample":
            return _sample(args)
        if args.cmd == "hybrid":
            return _hybrid(args)
    except RefuseError as exc:
        print(f"REFUSE  {exc.message}", file=sys.stderr)
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


def _bench(workdir: Path, steps: int | None) -> int:
    from adaptamem.bench import bench, format_bench

    print(format_bench(bench(workdir, steps=steps)))
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
    return 0 if eq.qc.ok or args.short else 1


def _sample(args: argparse.Namespace) -> int:
    import json

    from adaptamem.sample import format_schedule, schedule, write_schedule

    session = load_session(
        args.input, cli_objective=args.objective, budget_hours=args.budget_hours
    )
    traces = None
    if args.traces is not None:
        traces = json.loads(Path(args.traces).read_text())
    sched = schedule(
        session.objective, session.box, budget_hours=args.budget_hours, traces=traces
    )
    workdir = default_workdir(session, args.out)
    write_schedule(sched, workdir, extra={"objective": session.objective.type})
    print(format_schedule(sched))
    print(f"wrote {workdir / 'sample.json'}")
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
    return 0


def _progress(msg: str) -> None:
    print(f"  … {msg}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
