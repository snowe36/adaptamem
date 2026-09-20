from pathlib import Path

from adaptamem.cli import main
from adaptamem.errors import RefuseError
from adaptamem.gpu_contract import CUDA_VERSION, require_eq_pdb, stack_note


def test_stack_pin_is_cuda_12_8():
    assert CUDA_VERSION == "12.8"
    assert "eq.pdb" in stack_note()
    assert "no CPU fallback" in stack_note()


def test_require_eq_pdb(tmp_path: Path):
    try:
        require_eq_pdb(tmp_path)
    except RefuseError as exc:
        assert exc.code == "NOT_READY"
        return
    raise AssertionError("expected NOT_READY")


def test_cli_oracle_refuses_assembled_without_eq(tmp_path: Path):
    (tmp_path / "assembled.pdb").write_text("ATOM\n")
    (tmp_path / "system.xml").write_text("<System/>\n")
    (tmp_path / "assemble.json").write_text("{}\n")
    assert main(["oracle", str(tmp_path), "--ns", "2"]) == 2
