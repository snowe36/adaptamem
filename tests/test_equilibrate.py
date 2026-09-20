from adaptamem.equilibrate import (
    CA_RESTRAINT_ENERGY,
    NPT_WARMUP_1FS,
    four_fs_platforms,
    settle_steps,
    skip_npt,
)
from adaptamem.produce import production_strip_barostat


def test_cuda_nan_does_not_dump_2ps_on_cpu():
    assert settle_steps(short=False, platform="CUDA", cuda_failed=False) == 2000
    assert settle_steps(short=False, platform="CPU", cuda_failed=True) == 50
    assert settle_steps(short=True, platform="CUDA", cuda_failed=False) == 50


def test_four_fs_skips_cpu_when_cuda_present():
    plats = [("CUDA", "gpu"), ("CPU", "cpu")]
    assert four_fs_platforms(plats) == [("CUDA", "gpu")]
    assert four_fs_platforms([("CPU", "cpu")]) == [("CPU", "cpu")]


def test_ca_restraints_use_periodic_distance():
    assert "periodicdistance" in CA_RESTRAINT_ENERGY
    assert "(x-x0)" not in CA_RESTRAINT_ENERGY


def test_npt_warmup_is_10ps_at_1fs():
    assert NPT_WARMUP_1FS == 10_000


def test_skip_npt_when_nvt_qc_already_ok():
    assert skip_npt(qc_ok=True, stop_on_qc=True) is True
    assert skip_npt(qc_ok=False, stop_on_qc=True) is False
    assert skip_npt(qc_ok=True, stop_on_qc=False) is False


def test_nvt_teacher_strips_barostat():
    assert production_strip_barostat(["nvt_restrained", "skip_npt", "stop_on_qc"]) is True
    assert production_strip_barostat(["nvt_restrained", "npt_restrained"]) is False
