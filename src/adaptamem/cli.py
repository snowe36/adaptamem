from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from adaptamem.assemble import assemble, default_workdir, format_assemble, have_assembled
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
            "CPU-first MD inference for membrane proteins. "
            "What can we learn without a long trajectory? "
            "Modes: cpu_only | cpu_first | conventional."
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

    p_asm = sub.add_parser("assemble", help="CPU: orient + membrane (alias of prepare)")
    _job_args(p_asm)
    p_asm.add_argument("--out", type=Path, default=None)
    p_asm.add_argument("--force", action="store_true", help="Override doctor ACTION items")

    p_prep = sub.add_parser("prepare", help="CPU: fetch inputs, orient, assemble")
    _job_args(p_prep)
    p_prep.add_argument("--out", type=Path, default=None)
    p_prep.add_argument("--force", action="store_true", help="Override doctor ACTION items")

    p_feat = sub.add_parser(
        "features",
        help="CPU: observables from crystals or teacher traces",
    )
    p_feat.add_argument("input", type=Path, nargs="?", default=None, help="workdir or JSON traces")
    p_feat.add_argument("--out", type=Path, default=None)
    p_feat.add_argument(
        "--crystal",
        action="append",
        default=[],
        metavar="PDB[:CHAIN]",
        help="Inactive/active X-ray frames. Repeat. Gate aborts if span < ε.",
    )
    p_feat.add_argument("--name", default="tm6_ic")
    p_feat.add_argument(
        "--selection",
        default="name CA and resid 131 ; name CA and resid 272",
    )
    p_feat.add_argument("--precision", type=float, default=0.2)
    p_feat.add_argument(
        "--sequence",
        type=Path,
        default=None,
        help="Amino-acid sequence. Moonshot input; currently NOT_READY without a structure.",
    )

    p_comp = sub.add_parser(
        "compress",
        help="CPU: fit a prior from crystals or a surrogate from teacher traces",
    )
    p_comp.add_argument("traces", type=Path)
    p_comp.add_argument("--kind", default="auto", help="auto | msm | latent_dynamics | active_learning | …")
    p_comp.add_argument("--out", type=Path, required=True)
    p_comp.add_argument("--lag", type=int, default=1)
    p_comp.add_argument("--n-bins", type=int, default=8)
    p_comp.add_argument(
        "--teacher-gpu-hours",
        type=float,
        default=0.0,
        help="GPU-hours already spent on the teacher (billed, not inferred)",
    )

    p_inf = sub.add_parser("infer", help="CPU: predict the ensemble from a compressed model")
    p_inf.add_argument("model", type=Path)
    p_inf.add_argument("--out", type=Path, required=True)
    p_inf.add_argument("--n-samples", type=int, default=500)
    p_inf.add_argument("--seed", type=int, default=0)
    p_inf.add_argument(
        "--mode",
        default="cpu_first",
        help="cpu_only (no MD request) | cpu_first (oracle if uncertain) | conventional (REFUSE)",
    )

    p_dec = sub.add_parser(
        "decide",
        help="CPU: COVERAGE vs BRIDGE vs MECHANISM. Does not launch MD.",
    )
    p_dec.add_argument(
        "teacher",
        type=Path,
        nargs="?",
        default=None,
        help="Frozen teacher JSON (default: bundled 2RH1/3SN6 teacher)",
    )
    p_dec.add_argument("--out", type=Path, required=True)
    p_dec.add_argument(
        "--crystal",
        action="append",
        default=[],
        metavar="PDB[:CHAIN]",
        help="Endpoint crystals for extra GPCR switches. Repeat.",
    )

    p_loo = sub.add_parser(
        "loo",
        help="CPU: leave-one-protein-out crystal prior. Hold-out traces are LEAKAGE.",
    )
    p_loo.add_argument("--hold-out", default="adrb2")
    p_loo.add_argument(
        "--sweep",
        action="store_true",
        help="Hold out each catalog protein in turn",
    )
    p_loo.add_argument("--catalog", type=Path, default=None)
    p_loo.add_argument("--out", type=Path, required=True)
    p_loo.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repo root for catalog-relative PDB paths",
    )

    p_corr = sub.add_parser(
        "correct",
        help="CPU: Experiment 3 protocol. GPU stays off until a named shot has eq.pdb.",
    )
    p_corr.add_argument("--hold-out", default=None)
    p_corr.add_argument("--catalog", type=Path, default=None)
    p_corr.add_argument("--out", type=Path, required=True)
    p_corr.add_argument("--root", type=Path, default=None)

    p_or = sub.add_parser(
        "oracle",
        help="GPU: short trustworthy MD after eq.pdb. Explicit. Not a campaign.",
    )
    p_or.add_argument("workdir", type=Path)
    p_or.add_argument("--ns", type=float, required=True, help="Teacher length. Keep it tiny.")
    p_or.add_argument("--budget-hours", type=float, default=None)
    p_or.add_argument("--force", action="store_true")

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

    p_gpu = sub.add_parser("gpu", help="One-shot CHARMM36 throughput (CPU min, CUDA time)")
    p_gpu.add_argument("--out", type=Path, default=Path("runs/gpu"))
    p_gpu.add_argument("--steps", type=int, default=4000)
    p_gpu.add_argument("--pad", type=float, default=1.2)
    p_gpu.add_argument("--n-leu", type=int, default=20)
    p_gpu.add_argument("--rebuild", action="store_true")
    p_gpu.add_argument("--pads", default="")

    p_prod = sub.add_parser("produce", help="Production MD; streaming CVs, XTC secondary")
    p_prod.add_argument("workdir", type=Path)
    p_prod.add_argument("--ns", type=float, default=None, help="Hard cap on simulated ns")
    p_prod.add_argument("--budget-hours", type=float, default=None)
    p_prod.add_argument("--force", action="store_true", help="Ignore membrane QC refuse")

    sub.add_parser(
        "run",
        help="REFUSE: stages are explicit (prepare|features|compress|infer|oracle|analyze)",
    )

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
    p_of.add_argument(
        "--also",
        action="append",
        default=[],
        type=Path,
        help="Additional conventional start to concatenate into the held-out reference",
    )

    p_os = sub.add_parser("oracle-score", help="Score estimates against a frozen oracle")
    p_os.add_argument("estimate", type=Path, help="JSON with diagnostics or values")
    p_os.add_argument("--oracle", type=Path, required=True)

    p_an = sub.add_parser(
        "analyze",
        help="CPU: compression = MD avoided / GPU oracle spent (cpu_only vs cpu_first vs conventional)",
    )
    p_an.add_argument("--oracle", type=Path, default=None)
    p_an.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable. NAME is msm | latent_dynamics | single_long | adaptive | ...",
    )
    p_an.add_argument("--out", type=Path, default=None)
    p_an.add_argument("--demo", action="store_true")
    p_an.add_argument("--error", type=float, default=None, help="ε vs oracle mean")
    p_an.add_argument("--oracle-gpu-hours", type=float, default=None)

    p_cmp = sub.add_parser("compare", help="alias of analyze")
    p_cmp.add_argument("--oracle", type=Path, default=None)
    p_cmp.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable. NAME is msm | latent_dynamics | single_long | adaptive | ...",
    )
    p_cmp.add_argument("--out", type=Path, default=None)
    p_cmp.add_argument("--demo", action="store_true", help="Synthetic fixture (tests / docs)")
    p_cmp.add_argument("--error", type=float, default=None, help="ε vs oracle mean")
    p_cmp.add_argument("--oracle-gpu-hours", type=float, default=None)

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
        if args.cmd == "prepare":
            return _assemble(args)
        if args.cmd == "features":
            return _features(args)
        if args.cmd == "compress":
            return _compress(args)
        if args.cmd == "infer":
            return _infer(args)
        if args.cmd == "decide":
            return _decide(args)
        if args.cmd == "loo":
            return _loo(args)
        if args.cmd == "correct":
            return _correct(args)
        if args.cmd == "oracle":
            return _oracle(args)
        if args.cmd == "equilibrate":
            return _equilibrate(args.workdir, args.short)
        if args.cmd == "bench":
            return _bench(args.workdir, args.steps, args.ladder)
        if args.cmd == "gpu":
            return _gpu(args)
        if args.cmd == "produce":
            return _produce(args)
        if args.cmd == "run":
            return _run()
        if args.cmd == "sample":
            return _sample(args)
        if args.cmd == "hybrid":
            return _hybrid(args)
        if args.cmd == "reproduce":
            return _reproduce(args.campaign)
        if args.cmd == "oracle-freeze":
            return _oracle_freeze(args.workdir, args.out, also=args.also)
        if args.cmd == "oracle-score":
            return _oracle_score(args.estimate, args.oracle)
        if args.cmd in {"compare", "analyze"}:
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
    print("PREPARE  CPU assembled system; GPU oracle is a later, explicit step")
    return 0


def _features(args: argparse.Namespace) -> int:
    from adaptamem.features import traces_from_path, traces_from_workdir, write_features

    seq = args.sequence
    src = Path(args.input) if args.input is not None else None
    if seq is not None or (src is not None and src.suffix.lower() in {".fa", ".fasta", ".faa", ".seq"}):
        raise RefuseError(
            "sequence-in is the moonshot; need a structure first",
            code="NOT_READY",
        )

    if args.crystal:
        from adaptamem.crystal import crystal_insight
        from adaptamem.objective import Observable

        obs = [
            Observable(
                name=args.name,
                kind="distance",
                selection=args.selection,
                precision=args.precision,
            )
        ]
        insight = crystal_insight(list(args.crystal), obs)
        dest = args.out or Path("features.json")
        traces = insight["features"]
        write_features(traces, dest, extra={k: v for k, v in insight.items() if k != "features"})
        print(f"FEATURES  crystals n={ {k: len(v) for k, v in traces.items()} }")
        for k, v in traces.items():
            span = (insight.get("heterogeneity") or {}).get(k)
            print(f"  {k}  {[f'{x:.3f}' for x in v]}  span={span:.3f} nm")
        print(f"  identified    {insight['identified']}")
        print(f"  unidentified  {insight['unidentified']}")
        print(dest)
        return 0

    if args.input is None:
        raise RefuseError("features needs a workdir/JSON or --crystal PDB:CHAIN", code="NOT_READY")
    src = Path(args.input)
    traces = traces_from_workdir(src) if src.is_dir() else traces_from_path(src)
    dest = args.out or (src / "features.json" if src.is_dir() else src.with_name("features.json"))
    write_features(traces, dest)
    print(f"FEATURES  { {k: len(v) for k, v in traces.items()} }")
    for k, v in traces.items():
        if v:
            print(f"  {k}  n={len(v)}  last={v[-1]:.4f}")
    print(dest)
    return 0


def _compress(args: argparse.Namespace) -> int:
    from adaptamem.compress import compress
    from adaptamem.features import traces_from_path, traces_from_workdir

    src = Path(args.traces)
    traces = traces_from_workdir(src) if src.is_dir() else traces_from_path(src)
    kind = str(args.kind)
    if kind == "auto":
        n = min((len(v) for v in traces.values()), default=0)
        kind = "latent_dynamics" if n < args.lag + 2 else "msm"
    t0 = time.perf_counter()
    model = compress(
        kind,
        traces,
        lag=args.lag,
        n_bins=args.n_bins,
        teacher_gpu_hours=args.teacher_gpu_hours,
    )
    model["cpu_hours"] = (time.perf_counter() - t0) / 3600.0
    Path(args.out).write_text(json.dumps(model, indent=2) + "\n")
    print(f"COMPRESS  {model.get('kind')}  gpu-h={model.get('gpu_hours', 0)}  cpu-h={model['cpu_hours']:.6f}")
    print(args.out)
    return 0


def _infer(args: argparse.Namespace) -> int:
    from adaptamem.compress import infer_report

    model = json.loads(Path(args.model).read_text())
    t0 = time.perf_counter()
    payload = infer_report(
        model, mode=args.mode, n_samples=args.n_samples, seed=args.seed
    )
    cpu = (time.perf_counter() - t0) / 3600.0
    payload["cpu_hours"] = cpu
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    uncertain = payload.get("uncertain") or {}
    n_unc = sum(len(v) for v in uncertain.values())
    print(
        f"INFER  {payload['mode']}  {payload['method']}  "
        f"gpu-h={payload['gpu_hours']}  cpu-h={cpu:.6f}  uncertain_bins={n_unc}"
    )
    print(f"  identified    {payload.get('identified')}")
    print(f"  unidentified  {payload.get('unidentified')}")
    if payload.get("bridge_missing"):
        print("  BRIDGE  two wells, no sampled transition — not a 1D bin fill")
        for name, spec in (payload.get("wells") or {}).items():
            gap = spec.get("gap") or {}
            wells = spec.get("wells") or []
            print(
                f"  {name}  wells={len(wells)}  cross_hops={spec.get('cross_hops')}  "
                f"gap={gap.get('lo')}–{gap.get('hi')}"
            )
    if n_unc and payload.get("mode") != "cpu_only" and not payload.get("bridge_missing"):
        from adaptamem.features import write_oracle_request

        obs = model.get("observables") or {}
        teacher_frames = {
            k: int((obs.get(k) or {}).get("n_frames") or 0) for k in uncertain
        }
        req = write_oracle_request(
            Path(args.out).with_name("oracle_request.json"),
            uncertain=uncertain,
            teacher_frames=teacher_frames,
        )
        print(f"  GPU oracle only for those bins → {req}")
    print(args.out)
    return 0


def _decide(args: argparse.Namespace) -> int:
    from adaptamem.decide import decide_b2ar_teacher

    d = decide_b2ar_teacher(teacher=args.teacher, crystals=list(args.crystal or []))
    payload = d.to_dict()
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"DECIDE  coverage={d.coverage}  bridge={d.bridge}  gpu={d.gpu}")
    if d.refuse:
        print(f"  refuse     {d.refuse}")
    print(f"  question   {d.question}")
    if d.mechanism:
        print(f"  MECHANISM  hidden={d.mechanism.get('hidden_at_endpoints')}")
        print(f"  do_not_start {d.mechanism.get('do_not_start')}")
    print(args.out)
    return 0


def _loo(args: argparse.Namespace) -> int:
    from adaptamem.transfer import (
        crystal_table,
        leave_one_out,
        leave_one_out_sweep,
        load_catalog,
    )

    cat = load_catalog(args.catalog)
    root = Path(args.root) if args.root else Path.cwd()
    table = crystal_table(cat, root=root)
    eps = float(cat.get("epsilon_nm") or 0.2)
    if args.sweep:
        payload = leave_one_out_sweep(table, epsilon=eps)
        print(f"LOO  sweep n={payload['n_proteins']}  gpu-h=0")
        for pid, fold in payload["folds"].items():
            tm6 = (fold.get("predictions") or {}).get("tm6_ic") or {}
            err = (tm6.get("error") or {}).get("span")
            reached = (tm6.get("reached") or {}).get("span")
            print(f"  hold-out={pid}  tm6_span_err={err}  reached={reached}")
    else:
        payload = leave_one_out(table, str(args.hold_out), epsilon=eps)
        print(f"LOO  hold-out={payload['hold_out']}  train={payload['train']}  gpu-h=0")
        for name, spec in (payload.get("predictions") or {}).items():
            err = spec["error"]
            reached = spec.get("reached") or {}
            print(
                f"  {name}  inactive_err={err['inactive']:.3f}  "
                f"active_err={err['active']:.3f}  span_err={err['span']:.3f}  "
                f"span_reached={reached.get('span')}"
            )
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(args.out)
    return 0


def _correct(args: argparse.Namespace) -> int:
    from adaptamem.correction import (
        MECHANISM_COORDINATE,
        catalog_identities,
        classify_fold,
        correction_curve,
        gated_observables,
        load_experiment3,
        load_prior_correction,
        load_target_teachers,
        score_prior_correction,
        transfer_vs_distance,
    )
    from adaptamem.transfer import crystal_table, leave_one_out, leave_one_out_sweep, load_catalog

    proto = load_experiment3()
    cat = load_catalog(args.catalog)
    root = Path(args.root) if args.root else Path.cwd()
    table = crystal_table(cat, root=root)
    eps = float(proto.get("epsilon_nm") or cat.get("epsilon_nm") or 0.2)
    hold = str(args.hold_out or proto["hold_out"]["primary"])
    fold = leave_one_out(table, hold, epsilon=eps)
    sweep = leave_one_out_sweep(table, epsilon=eps)
    roles = classify_fold(fold, epsilon=eps)
    roles[MECHANISM_COORDINATE] = "mechanism"
    gated = gated_observables(fold, epsilon=eps)
    if MECHANISM_COORDINATE not in gated:
        gated = [*gated, MECHANISM_COORDINATE]
    identities = catalog_identities(cat, root=root)
    dist = {
        cv: transfer_vs_distance(sweep, identities, cv=cv)
        for cv in ("tm6_ic", "ionic_lock", "tm3_tm6_pack", "npxxY")
    }
    sweep_roles = {pid: classify_fold(f, epsilon=eps) for pid, f in sweep["folds"].items()}
    curves = {}
    for name in gated:
        spec = (fold.get("predictions") or {}).get(name)
        if not spec:
            continue
        curves[name] = correction_curve(
            zero_shot=float(spec["span"]["mu"]),
            teacher_mu={},
            oracle_mu=float(spec["truth"]["span"]),
            epsilon=eps,
            reference="crystal_endpoints",
        )
    payload = {
        "experiment": "adaptive_correction",
        "gpu": False,
        "hold_out": hold,
        "adversarial": proto["hold_out"]["adversarial"],
        "hard": proto["hold_out"].get("hard"),
        "roles": roles,
        "gated": gated,
        "sweep_roles": sweep_roles,
        "named_questions": proto.get("named_questions"),
        "teacher_budgets_gpu_hours": proto["teacher"]["budgets_gpu_hours"],
        "oracle": proto["oracle"],
        "zero_shot_curves": curves,
        "fold": fold,
        "distance": {
            f"{a}|{b}": v for (a, b), v in identities.items()
        },
        "transfer_vs_distance": dist,
        "refuse": "NOT_READY",
        "reason": "experiment 3 protocol is frozen; GPU off until a named shot has eq.pdb",
    }
    man = load_prior_correction()
    traces, hours = load_target_teachers(man, root=root)
    payload["prior_correction"] = score_prior_correction(man, traces, gpu_hours=hours)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"CORRECT  hold-out={hold}  gpu=False  gated={gated}")
    for name, role in roles.items():
        print(f"  {name}  {role}")
    pc = payload["prior_correction"]
    print(f"  prior_correction  awaiting_eq={pc['awaiting_eq']}  gpu-h={pc['gpu_hours']}")
    print(args.out)
    return 0


def _oracle(args: argparse.Namespace) -> int:
    from adaptamem.correction import require_teacher_gpu
    from adaptamem.gpu_contract import require_eq_pdb

    require_teacher_gpu()
    if not have_assembled(args.workdir):
        raise RefuseError(
            f"no assembled system in {args.workdir}; CPU prepare first, then oracle",
            code="NOT_READY",
        )
    require_eq_pdb(args.workdir)
    from adaptamem.produce import format_produce, produce

    r = produce(
        args.workdir,
        ns=args.ns,
        budget_hours=args.budget_hours,
        force=args.force,
        progress=_progress,
    )
    print(format_produce(r))
    print("ORACLE_DONE")
    return 0


def _equilibrate(workdir: Path, short: bool) -> int:
    from adaptamem.equilibrate import equilibrate, format_eq

    result = equilibrate(workdir, short=short, progress=_progress)
    print(format_eq(result))
    return 0 if result.qc.ok or short else 1


def _gpu(args: argparse.Namespace) -> int:
    from adaptamem.correction import require_teacher_gpu

    require_teacher_gpu()
    from importlib.util import module_from_spec, spec_from_file_location

    candidates = [
        Path(__file__).resolve().parents[2] / "scripts" / "gpu_job.py",
        Path.cwd() / "scripts" / "gpu_job.py",
    ]
    script = next((p for p in candidates if p.is_file()), None)
    if script is None:
        raise RefuseError("scripts/gpu_job.py not found; run from the repo", code="NOT_READY")
    spec = spec_from_file_location("adaptamem_gpu_job", script)
    if spec is None or spec.loader is None:
        raise RefuseError(f"could not load {script}", code="NOT_READY")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = [
        "--out",
        str(args.out),
        "--steps",
        str(args.steps),
        "--pad",
        str(args.pad),
        "--n-leu",
        str(args.n_leu),
    ]
    if args.rebuild:
        argv.append("--rebuild")
    if args.pads:
        argv.extend(["--pads", args.pads])
    return int(mod.main(argv) or 0)


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
    from adaptamem.correction import require_teacher_gpu
    from adaptamem.produce import format_produce, produce

    require_teacher_gpu()

    r = produce(
        args.workdir,
        ns=args.ns,
        budget_hours=args.budget_hours,
        force=args.force,
        progress=_progress,
    )
    print(format_produce(r))
    return 0


def _run() -> int:
    from adaptamem.pipeline import run_is_not_a_campaign

    print(run_is_not_a_campaign(), file=sys.stderr)
    return 2


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


def _oracle_freeze(workdir: Path, out: Path | None, also: list[Path] | None = None) -> int:
    from adaptamem.oracle import freeze_workdir, freeze_workdirs

    extras = list(also or [])
    if extras:
        dest = out or Path("oracle.json")
        oracle = freeze_workdirs([workdir, *extras], out=dest)
    else:
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
        if args.oracle is None:
            raise RefuseError("analyze needs --oracle PATH or --demo", code="NOT_READY")
        oracle = load_oracle(args.oracle)
        runs = []
        for item in args.method:
            if "=" not in item:
                raise RefuseError("--method needs NAME=PATH", code="NOT_READY")
            name, path = item.split("=", 1)
            runs.append(load_method(Path(path), name=name))
        if len(runs) < 2:
            raise RefuseError("compare needs at least two --method NAME=PATH", code="NOT_READY")
    payload = compare(
        oracle, runs, error=args.error, oracle_gpu_hours=args.oracle_gpu_hours
    )
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
