from pathlib import Path

from adaptamem.bench import BenchResult, format_bench, ladder_spec


def test_ladder_has_six_rungs():
    spec = ladder_spec()
    assert [s["target_atoms"] for s in spec] == [20_000, 50_000, 100_000, 200_000, 400_000, 600_000]


def test_format_bench_uses_rich_payload(tmp_path: Path):
    payload = {
        "system": {"atoms": 40000, "membrane_lipids": 120, "water_atoms": 20000},
        "hardware": {"gpu": "RTX 4090", "platform": "CUDA"},
        "physics": {
            "forcefield": "CHARMM36",
            "timestep_fs": 4.0,
            "cutoff_nm": 1.0,
            "pme": True,
            "precision": "mixed",
        },
        "performance": {"ns_per_day": 123.4, "steps_per_second": 35000},
    }
    r = BenchResult(
        ns_per_day=123.4,
        steps=20000,
        timestep_fs=4.0,
        platform="CUDA",
        n_atoms=40000,
        seconds=10.0,
        physics="same_physics",
        path=tmp_path / "bench.json",
        payload=payload,
    )
    text = format_bench(r)
    assert "123.4" in text
    assert "CHARMM36" in text
    assert "RTX 4090" in text
    assert "PME=True" in text
