from adaptamem.equilibrate import four_fs_platforms, settle_steps


def test_cuda_nan_does_not_dump_2ps_on_cpu():
    assert settle_steps(short=False, platform="CUDA", cuda_failed=False) == 2000
    assert settle_steps(short=False, platform="CPU", cuda_failed=True) == 50
    assert settle_steps(short=True, platform="CUDA", cuda_failed=False) == 50


def test_four_fs_skips_cpu_when_cuda_present():
    plats = [("CUDA", "gpu"), ("CPU", "cpu")]
    assert four_fs_platforms(plats) == [("CUDA", "gpu")]
    assert four_fs_platforms([("CPU", "cpu")]) == [("CPU", "cpu")]
