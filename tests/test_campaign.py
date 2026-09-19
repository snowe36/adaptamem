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
    assert "PRE/POST" in block
    assert "CAMPAIGN_DONE" not in block
    assert "post wins" in decision or "REFUSE" in camp.verdict(payload, "BUDGET")


def test_pyproject_has_gpu_extra():
    text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
    assert "openmm[cuda12]" in text
    assert "gpu =" in text or 'gpu =' in text
