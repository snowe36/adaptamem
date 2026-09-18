from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from adaptamem.box import format_box, plan_box
from adaptamem.doctor import audit, format_report
from adaptamem.objective import parse_objective
from adaptamem.schema import default_system_template_path, load_system


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adaptamem",
        description=(
            "Decide the cheapest membrane-protein simulation that can answer "
            "the objective. Not a fixed MD recipe."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Write a blank system YAML")
    p_init.add_argument("out", type=Path)

    p_doc = sub.add_parser("doctor", help="Structure report from a PDB")
    p_doc.add_argument("structure", type=Path)

    p_plan = sub.add_parser("plan", help="Doctor + cheapest box + next experiment")
    p_plan.add_argument("input", type=Path, help="PDB or system YAML")
    p_plan.add_argument(
        "--objective",
        default=None,
        help="conventional | conformational-shift | discover-states",
    )

    p_val = sub.add_parser("validate", help="Load a system YAML")
    p_val.add_argument("system", type=Path)

    p_run = sub.add_parser("run", help="Execute the planned experiment (physics not built)")
    p_run.add_argument("input", type=Path)
    p_run.add_argument("--objective", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "init":
        return _init(args.out)
    if args.cmd == "doctor":
        return _doctor(args.structure)
    if args.cmd == "plan":
        return _plan(args.input, args.objective)
    if args.cmd == "validate":
        return _validate(args.system)
    if args.cmd == "run":
        _plan(args.input, args.objective)
        print(
            "\nphysics engine is not built — this is the decision layer only",
            file=sys.stderr,
        )
        return 2
    parser.error(f"unknown command {args.cmd}")
    return 2


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


def _resolve(path: Path, cli_objective: str | None):
    if path.suffix.lower() in {".yml", ".yaml"}:
        system = load_system(path)
        obj = parse_objective(
            {
                "type": system.objective.type,
                "observables": [
                    {
                        "name": o.name,
                        "kind": o.kind,
                        "selection": o.selection,
                        "precision": o.precision,
                    }
                    for o in system.objective.observables
                ],
                "discover_cvs": system.objective.discover_cvs,
            },
            cli_type=cli_objective,
        )
        return system, audit(system.structure), obj
    from adaptamem.schema import Membrane, Orientation, System

    obj = parse_objective({}, cli_type=cli_objective or "discover_states")
    system = System(
        name=path.stem,
        structure=path,
        orientation=Orientation(),
        membrane=Membrane(),
        objective=obj,
        source=None,
    )
    return system, audit(path), obj


def _plan(path: Path, cli_objective: str | None) -> int:
    system, report, obj = _resolve(path, cli_objective)
    print(format_report(report))
    print()
    box = plan_box(
        report,
        water_pad_nm=system.membrane.water_pad_nm,
        safety_margin_nm=system.membrane.safety_margin_nm,
    )
    print(format_box(box))
    print()
    print("EXPERIMENT")
    print(f"objective     {obj.type}")
    if obj.observables:
        for o in obj.observables:
            prec = f"  precision={o.precision}" if o.precision is not None else ""
            print(f"  observable  {o.name} ({o.kind}){prec}")
    elif obj.discover_cvs:
        print("  CVs from a short pilot (TICA/PCA) — not user-specified")
    else:
        print("  conventional single trajectory")
    print("next          short pilot → dynamical cartography → allocate walkers")
    print("stop          when uncertainty on the objective is below precision")
    print("not decided   walker count, ns/walker (those are scheduler outputs)")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
