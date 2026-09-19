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


def _ca_xyz(text: str):
    xs, ys, zs = [], [], []
    for line in text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        xs.append(float(line[30:38]))
        ys.append(float(line[38:46]))
        zs.append(float(line[46:54]))
    return xs, ys, zs


def _assert_rotated_centered_z(xs, ys, zs):
    assert max(zs) - min(zs) > 20
    assert abs((max(zs) + min(zs)) / 2) < 3
    assert max(xs) - min(xs) > 2
    assert max(ys) - min(ys) > 2


def test_helix_pdb_writes_leu(tmp_path: Path):
    gpu = _load_gpu_job()
    path = gpu.helix_pdb(tmp_path / "helix.pdb", n=20)
    text = path.read_text()
    assert text.count("ATOM") == 160
    assert "LEU" in text
    _assert_rotated_centered_z(*_ca_xyz(text))


def test_both_helix_generators_rotated_and_centered(tmp_path: Path):
    from pdbutil import helix_pdb as pdbutil_helix

    gpu = _load_gpu_job()
    gpu_text = gpu.helix_pdb(tmp_path / "gpu.pdb", n=20).read_text()
    util_text = pdbutil_helix(tmp_path / "util.pdb", n=20).read_text()
    _assert_rotated_centered_z(*_ca_xyz(gpu_text))
    _assert_rotated_centered_z(*_ca_xyz(util_text))


def test_gpu_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["gpu", "--help"])
    assert exc.value.code == 0
