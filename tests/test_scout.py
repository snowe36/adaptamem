from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_scout():
    script = Path(__file__).resolve().parents[1] / "scripts" / "gpu_scout.py"
    spec = spec_from_file_location("adaptamem_gpu_scout", script)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_easy_well_aborts():
    camp = _load_scout()
    # 5EH4-class: span already inside ε
    assert camp.scout_verdict(0.05, 0.2).startswith("SCOUT_ABORT")
    assert camp.scout_verdict(0.199, 0.2).startswith("SCOUT_ABORT")


def test_slow_cv_is_go():
    camp = _load_scout()
    assert camp.scout_verdict(0.2, 0.2).startswith("SCOUT_GO")
    assert camp.scout_verdict(1.1, 0.2).startswith("SCOUT_GO")


def test_missing_series_fails():
    camp = _load_scout()
    assert camp.scout_verdict(None).startswith("SCOUT_FAIL")


def test_span_of():
    camp = _load_scout()
    assert abs(camp.span_of([0.8, 1.1, 0.9]) - 0.3) < 1e-9
    assert camp.span_of([]) is None


def test_cpu_assemble_then_ship_to_cuda():
    camp = _load_scout()
    assert camp.next_stage(assembled=False, cuda=False) == "assemble"
    assert camp.next_stage(assembled=True, cuda=False) == "ship"
    assert camp.next_stage(assembled=True, cuda=True) == "eq"
    assert camp.next_stage(assembled=False, cuda=True) == "refuse"
