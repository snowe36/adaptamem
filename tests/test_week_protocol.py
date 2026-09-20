"""Weeks 3–6 as CPU protocol tests. No billed GPU."""

from __future__ import annotations

import json
import random
from pathlib import Path

from adaptamem.cli import main
from adaptamem.compare import MethodRun, compare
from adaptamem.compress import compress, infer, infer_report, uncertain_regions
from adaptamem.errors import RefuseError

A, B = 0.8, 1.6
EPS = 0.2
BASELINE_H = 69.0
TEACHER_H = 6.9
OBS = "tm6_ic"


def _walk(n: int, rng: random.Random, *, start: float, p_ab: float, p_ba: float) -> list[float]:
    x = start
    out: list[float] = []
    for _ in range(n):
        u = rng.random()
        if x == A and u < p_ab:
            x = B
        elif x == B and u < p_ba:
            x = A
        out.append(x)
    return out


def test_week3_two_start_teacher_does_not_invent_rates():
    shot_a = [A] * 40
    shot_b = [B] * 40
    model = compress(
        "latent_dynamics",
        {OBS: shot_a + shot_b},
        teacher_gpu_hours=0.2,
        shot_lengths=[40, 40],
    )
    report = infer_report(model, mode="cpu_first", n_samples=200, seed=0)
    assert report["identification"]["kinetics_identified"] is False
    assert "transitions" in report["unidentified"]
    assert set(report["values"][OBS]) <= {A, B}


def test_week4_coverage_refuse_when_everything_hungry():
    try:
        compress("active_learning", {OBS: [A] * 80}, inner="latent_dynamics")
    except RefuseError as exc:
        assert exc.code == "COVERAGE"
        return
    raise AssertionError("expected COVERAGE")


def test_week4_occupancy_vs_pi_disconnected():
    xs = [A] * 60 + [B] * 60
    model = compress("msm", {OBS: xs}, n_bins=4, shot_lengths=[60, 60])
    spec = model["observables"][OBS]
    assert spec["disconnected"] is True
    ident = model["identification"]
    assert ident["pi_identified"] is False
    assert ident["occupancy_identified"] is True
    assert uncertain_regions(model)


def test_week5_held_out_freeze_concatenates_starts(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "walker.states").write_text(json.dumps({"observables": {OBS: [A] * 50}}) + "\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "walker.states").write_text(json.dumps({"observables": {OBS: [B] * 50}}) + "\n")
    dest = tmp_path / "oracle.json"
    assert (
        main(
            [
                "oracle-freeze",
                str(tmp_path / "a"),
                "--also",
                str(tmp_path / "b"),
                "--out",
                str(dest),
            ]
        )
        == 0
    )
    data = json.loads(dest.read_text())
    assert data["held_out"] is True
    assert len(data["values"][OBS]) == 100
    assert set(data["values"][OBS]) == {A, B}


def test_week6_three_modes_scored_against_held_out_oracle():
    rng = random.Random(0)
    oracle_vals = _walk(8_000, rng, start=A, p_ab=0.04, p_ba=0.16)
    teacher = _walk(4_000, random.Random(1), start=A, p_ab=0.04, p_ba=0.16)
    crystals = [A, B]
    mu_oracle = sum(oracle_vals) / len(oracle_vals)
    from adaptamem.oracle import Oracle

    held = Oracle(values={OBS: oracle_vals}, estimates={OBS: {"estimate": mu_oracle}})
    cpu_only = infer(compress("latent_dynamics", {OBS: crystals}), n_samples=4_000, seed=0)
    cpu_first = infer(compress("msm", {OBS: teacher}, n_bins=4), n_samples=4_000, seed=0)
    payload = compare(
        held,
        [
            MethodRun("cpu_only", cpu_only, 0.0, cpu_hours=0.001),
            MethodRun("msm", cpu_first, TEACHER_H, cpu_hours=0.001),
            MethodRun("single_long", {OBS: oracle_vals}, BASELINE_H),
        ],
        error=EPS,
        oracle_gpu_hours=BASELINE_H,
    )
    methods = payload["oracle_compression"]["methods"]
    msm = methods["msm"][OBS]
    only = methods["cpu_only"][OBS]
    conv = methods["single_long"][OBS]
    assert msm["reached"]
    assert msm["compression"] is not None and msm["compression"] >= 10.0
    assert only["total_gpu_hours"] == 0.0
    assert conv["reached"]
    go = {
        "beta2ar": "keep if teacher stack is stable",
        "msm_accelerated": msm["acceleration"]["accelerated"],
        "not_inferred": ["rates", "pathways", "sequence_in"],
    }
    assert go["msm_accelerated"] is True
    assert go["not_inferred"]
