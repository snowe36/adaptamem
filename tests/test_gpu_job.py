from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from adaptamem.cli import main


def _load_gpu_job():
    script = Path(__file__).resolve().parents[1] / "scripts" / "gpu_job.py"
    spec = spec_from_file_location("adaptamem_gpu_job", script)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_helix_pdb_writes_leu(tmp_path: Path):
    gpu = _load_gpu_job()
    path = gpu.helix_pdb(tmp_path / "helix.pdb", n=20)
    text = path.read_text()
    assert text.count("ATOM") == 160
    assert "LEU" in text


def test_gpu_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["gpu", "--help"])
    assert exc.value.code == 0
