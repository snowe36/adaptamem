"""Held-out synthetic two-basin oracle. CPU compression; no GPU."""

from __future__ import annotations

import json
import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from adaptamem.cli import main
from adaptamem.compare import MethodRun, compare
from adaptamem.compress import compress, infer, uncertain_regions
from adaptamem.errors import RefuseError
from adaptamem.oracle import Oracle, freeze

A, B = 0.8, 1.6
P_AB, P_BA = 0.04, 0.16
EPS = 0.1
BASELINE_H = 69.0
TEACHER_H = 6.9
OBS = "tm6_ic"
ROOT = Path(__file__).resolve().parents[1]


def two_state(
    n: int,
    rng: random.Random,
    *,
    start: float = A,
    p_ab: float = P_AB,
    p_ba: float = P_BA,
) -> list[float]:
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


def oracle_series() -> list[float]:
    return two_state(8_000, random.Random(0))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def test_msm_compresses_mixed_teacher_vs_held_out_oracle():
    oracle_vals = oracle_series()
    teacher = two_state(4_000, random.Random(1))
    unmixed = two_state(4_000, random.Random(2), p_ab=0.0, p_ba=0.0, start=A)
    oracle = Oracle(
        values={OBS: oracle_vals},
        estimates={OBS: {"estimate": mean(oracle_vals)}},
    )
    pred = infer(compress("msm", {OBS: teacher}, n_bins=4), n_samples=4_000, seed=0)
    payload = compare(
        oracle,
        [
            MethodRun("single_long", {OBS: unmixed}, TEACHER_H),
            MethodRun("msm", pred, TEACHER_H, cpu_hours=0.001),
        ],
        error=EPS,
        oracle_gpu_hours=BASELINE_H,
    )
    msm = payload["oracle_compression"]["methods"]["msm"][OBS]
    ctrl = payload["oracle_compression"]["methods"]["single_long"][OBS]
    assert abs(mean(oracle_vals) - 0.96) < 0.05
    assert abs(mean(unmixed) - A) < 1e-9
    assert not ctrl["reached"]
    assert msm["reached"]
    assert msm["compression"] is not None
    assert msm["compression"] >= 10.0
    assert msm["acceleration"]["accelerated"] is True


def test_unmixed_teacher_mean_fails_and_msm_does_not_invent_the_other_well():
    oracle_vals = oracle_series()
    unmixed = [A] * 800
    oracle = Oracle(values={OBS: oracle_vals}, estimates={OBS: {"estimate": mean(oracle_vals)}})
    model = compress("msm", {OBS: unmixed}, n_bins=4)
    pred = infer(model, n_samples=200, seed=0)
    assert set(pred[OBS]) == {A}
    unsure = uncertain_regions(model)
    assert unsure[OBS]
    payload = compare(
        oracle,
        [
            MethodRun("single_long", {OBS: unmixed}, TEACHER_H),
            MethodRun("msm", pred, TEACHER_H, cpu_hours=0.001),
        ],
        error=EPS,
        oracle_gpu_hours=BASELINE_H,
    )
    assert not payload["oracle_compression"]["methods"]["msm"][OBS]["reached"]
    assert not payload["oracle_compression"]["methods"]["single_long"][OBS]["reached"]


def test_msm_lag_does_not_cross_shot_boundaries():
    shot_a = [A] * 60
    shot_b = [B] * 60
    xs = shot_a + shot_b
    fake = compress("msm", {OBS: xs}, n_bins=4)
    honest = compress("msm", {OBS: xs}, n_bins=4, shot_lengths=[60, 60])
    spec_fake = fake["observables"][OBS]
    spec_ok = honest["observables"][OBS]
    hops_fake = sum(
        spec_fake["counts"][i][j]
        for i in range(len(spec_fake["counts"]))
        for j in range(len(spec_fake["counts"]))
        if i != j
    )
    hops_ok = sum(
        spec_ok["counts"][i][j]
        for i in range(len(spec_ok["counts"]))
        for j in range(len(spec_ok["counts"]))
        if i != j
    )
    assert hops_fake >= 1
    assert hops_ok == 0
    assert spec_ok["disconnected"] is True
    pred = infer(honest, n_samples=200, seed=0)
    assert set(pred[OBS]) <= {A, B}


def test_latent_occupancy_hits_tenx_on_mixed_teacher():
    oracle_vals = oracle_series()
    mixed = [A] * 800 + [B] * 200
    unmixed = [A] * 1000
    oracle = Oracle(values={OBS: oracle_vals}, estimates={OBS: {"estimate": mean(oracle_vals)}})
    pred = infer(compress("latent_dynamics", {OBS: mixed}), n_samples=4_000, seed=0)
    payload = compare(
        oracle,
        [
            MethodRun("single_long", {OBS: unmixed}, TEACHER_H),
            MethodRun("latent_dynamics", pred, TEACHER_H, cpu_hours=0.001),
        ],
        error=EPS,
        oracle_gpu_hours=BASELINE_H,
    )
    lat = payload["oracle_compression"]["methods"]["latent_dynamics"][OBS]
    assert lat["reached"]
    assert lat["compression"] >= 10.0
    assert lat["acceleration"]["accelerated"] is True


def test_latent_unmixed_is_ood_uncertain_and_does_not_invent():
    model = compress("latent_dynamics", {OBS: [A] * 100})
    pred = infer(model, n_samples=50, seed=0)
    assert set(pred[OBS]) == {A}
    assert uncertain_regions(model)[OBS] == [0]


def test_active_learning_refuses_when_everything_is_uncertain():
    try:
        compress("active_learning", {OBS: [A] * 80}, inner="latent_dynamics")
    except RefuseError as exc:
        assert exc.code == "COVERAGE"
        return
    raise AssertionError("expected COVERAGE refuse")


def test_active_learning_queries_only_an_extra_shot():
    unmixed = [A] * 400
    extra = {OBS: [[B] * 100]}
    model = compress(
        "active_learning",
        {OBS: unmixed},
        inner="latent_dynamics",
        extra_shots=extra,
        teacher_gpu_hours=6.0,
        gpu_hours_per_shot=0.9,
        max_shots=3,
    )
    assert model["kind"] == "active_learning"
    assert model["n_queries"] == 1
    assert abs(model["gpu_hours"] - 6.9) < 1e-9
    pred = infer(model, n_samples=4_000, seed=0)
    oracle_vals = oracle_series()
    oracle = Oracle(values={OBS: oracle_vals}, estimates={OBS: {"estimate": mean(oracle_vals)}})
    payload = compare(
        oracle,
        [
            MethodRun("single_long", {OBS: unmixed}, 6.9),
            MethodRun("active_learning", pred, float(model["gpu_hours"]), cpu_hours=0.001),
        ],
        error=EPS,
        oracle_gpu_hours=BASELINE_H,
    )
    arm = payload["oracle_compression"]["methods"]["active_learning"][OBS]
    assert arm["reached"]
    assert arm["compression"] >= 10.0


def test_cli_features_compress_infer_analyze(tmp_path: Path):
    oracle_vals = oracle_series()
    teacher = two_state(4_000, random.Random(3))
    unmixed = two_state(4_000, random.Random(4), p_ab=0.0, start=A)
    freeze({OBS: oracle_vals}, tmp_path / "oracle.json")
    (tmp_path / "teacher.json").write_text(json.dumps({"values": {OBS: teacher}}) + "\n")
    (tmp_path / "unmixed.json").write_text(
        json.dumps({"method": "single_long", "values": {OBS: unmixed}, "gpu_hours": TEACHER_H})
        + "\n"
    )
    assert main(["features", str(tmp_path / "teacher.json"), "--out", str(tmp_path / "features.json")]) == 0
    assert (
        main(
            [
                "compress",
                str(tmp_path / "features.json"),
                "--kind",
                "msm",
                "--n-bins",
                "4",
                "--teacher-gpu-hours",
                str(TEACHER_H),
                "--out",
                str(tmp_path / "model.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "infer",
                str(tmp_path / "model.json"),
                "--n-samples",
                "2000",
                "--out",
                str(tmp_path / "pred.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "analyze",
                "--oracle",
                str(tmp_path / "oracle.json"),
                "--method",
                f"msm={tmp_path / 'pred.json'}",
                "--method",
                f"single_long={tmp_path / 'unmixed.json'}",
                "--error",
                str(EPS),
                "--oracle-gpu-hours",
                str(BASELINE_H),
                "--out",
                str(tmp_path / "compare.json"),
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "compare.json").read_text())
    msm = payload["oracle_compression"]["methods"]["msm"][OBS]
    ctrl = payload["oracle_compression"]["methods"]["single_long"][OBS]
    assert payload["primary_metric"] == "compression"
    assert msm["reached"]
    assert msm["compression"] >= 10.0
    assert not ctrl["reached"]
    assert ctrl["acceleration"]["accelerated"] is False


def test_gpu_oracle_refuses_empty_workdir(tmp_path: Path, monkeypatch):
    script = ROOT / "scripts" / "gpu_oracle.py"
    spec = spec_from_file_location("adaptamem_gpu_oracle", script)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv("ADAPTAMEM_WORKDIR", str(tmp_path))
    assert mod.main() == 2


def test_gpu_oracle_refuses_assembled_without_eq(tmp_path: Path, monkeypatch):
    script = ROOT / "scripts" / "gpu_oracle.py"
    spec = spec_from_file_location("adaptamem_gpu_oracle", script)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "assembled.pdb").write_text("ATOM\n")
    (tmp_path / "system.xml").write_text("<System/>\n")
    monkeypatch.setenv("ADAPTAMEM_WORKDIR", str(tmp_path))
    assert mod.main() == 2


def test_runpod_boot_is_oracle_only():
    text = (ROOT / "scripts" / "runpod_boot.sh").read_text()
    assert "gpu_oracle.py" in text
    assert "gpu_campaign.py" not in text
    assert "gpu_scout.py" not in text
