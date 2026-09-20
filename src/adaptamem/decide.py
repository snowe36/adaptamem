"""What information is missing. COVERAGE / BRIDGE / MECHANISM — not more TM6."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adaptamem.compress import compress, uncertain_regions
from adaptamem.crystal import traces_from_crystals
from adaptamem.errors import RefuseError
from adaptamem.gpcr import SWITCHES

_RESOURCE = Path(__file__).parent / "resources" / "b2ar_teacher.json"


@dataclass
class Decision:
    coverage: str
    bridge: str
    mechanism: dict[str, Any] | None
    gpu: bool
    next_start: str | None
    question: str
    identified: list[str]
    unidentified: list[str]
    wells: list[dict[str, Any]]
    refuse: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_b2ar_teacher(path: Path | None = None) -> dict[str, Any]:
    data = json.loads(Path(path or _RESOURCE).read_text())
    shots = data["shots"]
    traces = {"tm6_ic": [float(x) for s in shots for x in s["tm6_ic"]]}
    data["traces"] = traces
    data["shot_lengths"] = [len(s["tm6_ic"]) for s in shots]
    return data


def decide(
    traces: dict[str, list[float]],
    *,
    shot_lengths: list[int] | None = None,
    crystal_traces: dict[str, list[float]] | None = None,
    teacher_gpu_hours: float = 0.0,
    epsilon: float = 0.2,
    do_not_start: tuple[str, ...] = ("2RH1", "3SN6"),
) -> Decision:
    """CPU policy. Does not launch MD."""
    lens = shot_lengths or [len(next(iter(traces.values())))]
    model = compress(
        "msm",
        traces,
        lag=1,
        n_bins=8,
        shot_lengths=lens,
        teacher_gpu_hours=teacher_gpu_hours,
    )
    obs = model.get("observables") or {}
    primary = obs.get("tm6_ic") or next(iter(obs.values()), {})
    ident = model.get("identification") or {}
    wells = list(primary.get("wells") or [])
    bridge_missing = bool(primary.get("bridge_missing"))
    hungry = uncertain_regions(model, min_count=5)
    n_hungry = sum(len(v) for v in hungry.values())

    if bridge_missing:
        mech = mechanism_request(
            crystal_traces or {},
            wells,
            epsilon=epsilon,
            do_not_start=do_not_start,
        )
        return Decision(
            coverage="no_1d_shot",
            bridge="refuse",
            mechanism=mech,
            gpu=False,
            next_start=None,
            question=mech["question"],
            identified=list(ident.get("identified") or []),
            unidentified=list(ident.get("unidentified") or []),
            wells=wells,
            refuse="BRIDGE",
        )
    if n_hungry:
        return Decision(
            coverage="shot",
            bridge="ok",
            mechanism=None,
            gpu=False,
            next_start=None,
            question="sparse bins inside a communicating chain",
            identified=list(ident.get("identified") or []),
            unidentified=list(ident.get("unidentified") or []),
            wells=wells,
            refuse="COVERAGE" if _all_bins_hungry(obs, hungry) else None,
        )
    return Decision(
        coverage="enough",
        bridge="ok",
        mechanism=None,
        gpu=False,
        next_start=None,
        question="CPU model has local coverage and a communicating chain",
        identified=list(ident.get("identified") or []),
        unidentified=list(ident.get("unidentified") or []),
        wells=wells,
    )


def mechanism_request(
    crystal_traces: dict[str, list[float]],
    wells: list[dict[str, Any]],
    *,
    epsilon: float = 0.2,
    do_not_start: tuple[str, ...] = ("2RH1", "3SN6"),
) -> dict[str, Any]:
    """Hypotheses from crystal spans. Not a pathway claim."""
    spans = {name: (max(xs) - min(xs) if len(xs) >= 2 else 0.0) for name, xs in crystal_traces.items()}
    endpoint = [n for n, s in spans.items() if n != "tm6_ic" and s >= epsilon]
    hidden = [n for n, s in spans.items() if n != "tm6_ic" and s < epsilon]
    hyps = [
        {
            "id": "A",
            "name": "pack_first",
            "claim": "TM3–TM6 packing rearranges before TM6 IC displacement",
            "coordinate": "tm3_tm6_pack",
            "crystal_span_nm": spans.get("tm3_tm6_pack"),
            "note": "packing is the same at both crystals; any pack-first path is hidden at the endpoints",
        },
        {
            "id": "B",
            "name": "microswitch_first",
            "claim": "NPxxY rearrangement precedes TM6 IC displacement",
            "coordinate": "npxxY",
            "crystal_span_nm": spans.get("npxxY"),
            "note": "NPxxY already differs at the crystals; endpoints do not order the events",
        },
        {
            "id": "C",
            "name": "lock_with_tm6",
            "claim": "ionic lock is slaved to TM6 IC; it is not an independent switch",
            "coordinate": "ionic_lock",
            "crystal_span_nm": spans.get("ionic_lock"),
            "note": "lock span matches TM6 at the crystals; a broken lock in the inactive TM6 well would refute this",
        },
    ]
    return {
        "code": "MECHANISM",
        "gpu": False,
        "do_not_start": list(do_not_start),
        "endpoint_switches": endpoint,
        "hidden_at_endpoints": hidden,
        "hypotheses": hyps,
        "question": (
            "Does NPxxY or TM3–TM6 packing rearrange while TM6 stays in the inactive well?"
        ),
        "discriminating_region": {
            "tm6_ic": "well_0",
            "and_any_of": hidden + [n for n in ("npxxY", "ionic_lock") if n in endpoint],
        },
        "wells": wells,
        "spans": spans,
        "reason": (
            "two wells, no sampled transition; filling 1D TM6 bins is not a mechanism teacher. "
            "No eq.pdb in the discriminating region — do not buy GPU."
        ),
    }


def decide_b2ar_teacher(
    *,
    teacher: Path | None = None,
    crystals: list[str] | None = None,
) -> Decision:
    payload = load_b2ar_teacher(teacher)
    crystal_traces: dict[str, list[float]] = {}
    if crystals:
        crystal_traces = traces_from_crystals(crystals, list(SWITCHES))
    return decide(
        payload["traces"],
        shot_lengths=payload["shot_lengths"],
        crystal_traces=crystal_traces,
        teacher_gpu_hours=float(payload.get("gpu_hours_produce") or 0.0),
        epsilon=float(payload.get("epsilon_nm") or 0.2),
    )


def require_not_another_tm6_shot(decision: Decision) -> None:
    if decision.bridge == "refuse":
        raise RefuseError(
            decision.mechanism["reason"] if decision.mechanism else "BRIDGE",
            code="BRIDGE",
        )
    if decision.refuse == "COVERAGE":
        raise RefuseError("every bin is uncertain; refusing to become conventional MD", code="COVERAGE")
    if decision.mechanism and not decision.gpu:
        raise RefuseError(decision.mechanism["reason"], code="MECHANISM")


def _all_bins_hungry(obs: dict[str, Any], hungry: dict[str, list[int]]) -> bool:
    for name, spec in obs.items():
        n = len(spec.get("counts") or spec.get("pi") or [])
        if set(hungry.get(name) or []) != set(range(n)):
            return False
    return bool(obs)
