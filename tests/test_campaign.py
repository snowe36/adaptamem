from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from adaptamem.compare import MethodRun, compare
from adaptamem.oracle import Oracle


def _load_campaign():
    script = Path(__file__).resolve().parents[1] / "scripts" / "gpu_campaign.py"
    spec = spec_from_file_location("adaptamem_gpu_campaign", script)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_lengths_fits_budget():
    camp = _load_campaign()
    oracle_ns, method_ns = camp.plan_lengths(480.0, 4.0)
    assert abs(oracle_ns - 20.0) < 1e-9
    assert abs(method_ns - 5.0) < 1e-9


def test_plan_lengths_scales_when_tight():
    camp = _load_campaign()
    oracle_ns, method_ns = camp.plan_lengths(50.0, 1.0)
    assert oracle_ns < 20.0
    assert method_ns < 5.0
    assert abs(oracle_ns / method_ns - 4.0) < 1e-6


def test_verdict_and_block():
    camp = _load_campaign()
    oracle = Oracle(
        values={"g83_sep": [1.0] * 40},
        estimates={"g83_sep": {"estimate": 1.0, "ci95": 0.01}},
    )
    hours = 1.0
    runs = [
        MethodRun("single_long", {"g83_sep": [0.2] * 40}, hours, 5.0),
        MethodRun("independent_replicas", {"g83_sep": [0.5] * 40}, hours, 5.0),
        MethodRun("adaptive", {"g83_sep": [1.0] * 40}, hours, 5.0),
    ]
    payload = compare(oracle, runs, precision=0.05)
    decision = camp.verdict(payload, None)
    block = camp.format_block(200.0, 50000, oracle.to_dict(), {
        "single_long": (payload["equal_compute"]["methods"]["single_long"]["g83_sep"]),
        "independent_replicas": (payload["equal_compute"]["methods"]["independent_replicas"]["g83_sep"]),
        "adaptive": (payload["equal_compute"]["methods"]["adaptive"]["g83_sep"]),
    }, decision)
    assert block.count("\n") == 5
    assert "COMPRESS" in block
    assert "CAMPAIGN_DONE" not in block
    assert "acceleration" in decision or "REFUSE" in camp.verdict(payload, "BUDGET")


def test_keep_chains_drops_crystal_copy(tmp_path: Path):
    camp = _load_campaign()
    pdb = tmp_path / "xtal.pdb"
    pdb.write_text(
        "ATOM      1  CA  GLY A  83       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  GLY B  83       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  GLY C  83       9.000   9.000   9.000  1.00  0.00           C\n"
        "ATOM      4  CA  GLY D  83       8.000   9.000   9.000  1.00  0.00           C\n"
        "HETATM    5  C1  OLB A 201       0.500   0.500   0.500  1.00  0.00           C\n"
    )
    camp.keep_chains(pdb, "AB")
    text = pdb.read_text()
    assert "GLY A  83" in text
    assert "GLY B  83" in text
    assert "GLY C  83" not in text
    assert "GLY D  83" not in text
    assert "OLB" not in text


def test_pyproject_has_gpu_extra():
    text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
    assert "openmm[cuda12]" in text
    assert "gpu =" in text or 'gpu =' in text


def test_watchdog_missing_env(monkeypatch):
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    script = Path(__file__).resolve().parents[1] / "scripts" / "runpod_watchdog.py"
    spec = spec_from_file_location("adaptamem_runpod_watchdog", script)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.terminate_self("test") is False
