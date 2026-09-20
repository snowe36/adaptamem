from pathlib import Path

import pytest

from adaptamem.cli import main
from adaptamem.compress import compress
from adaptamem.crystal import traces_from_crystals
from adaptamem.decide import (
    decide,
    decide_b2ar_teacher,
    load_b2ar_teacher,
    require_not_another_tm6_shot,
)
from adaptamem.errors import RefuseError
from adaptamem.gpcr import SWITCHES

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_teacher_refuses_bridge_not_coverage():
    payload = load_b2ar_teacher()
    traces = payload["traces"]
    assert payload["shot_lengths"] == [20, 20]
    assert payload["cross_hops"] == 0
    lo = min(traces["tm6_ic"][:20])
    hi0 = max(traces["tm6_ic"][:20])
    lo1 = min(traces["tm6_ic"][20:])
    assert hi0 - lo < 0.2
    assert (lo1 - hi0) < float(payload["epsilon_nm"])
    d = decide(
        traces,
        shot_lengths=payload["shot_lengths"],
        teacher_gpu_hours=float(payload["gpu_hours_produce"]),
        epsilon=float(payload["epsilon_nm"]),
    )
    assert d.coverage == "no_1d_shot"
    assert d.bridge == "refuse"
    assert d.gpu is False
    assert d.next_start is None
    assert d.refuse == "BRIDGE"
    assert d.mechanism is not None
    assert d.mechanism["code"] == "MECHANISM"
    assert "2RH1" in d.mechanism["do_not_start"]
    try:
        compress(
            "active_learning",
            traces,
            inner="msm",
            n_bins=8,
            shot_lengths=payload["shot_lengths"],
        )
    except RefuseError as exc:
        assert exc.code == "BRIDGE"
        assert "1D" in exc.message
        return
    raise AssertionError("expected BRIDGE")


def test_crystal_switches_hide_packing_not_npxxY():
    refs = [
        str(ROOT / "data/structures/2RH1.pdb") + ":A",
        str(ROOT / "data/structures/3SN6_R.pdb") + ":R",
    ]
    xtal = traces_from_crystals(refs, list(SWITCHES))
    payload = load_b2ar_teacher()
    d = decide(
        payload["traces"],
        shot_lengths=payload["shot_lengths"],
        crystal_traces=xtal,
        teacher_gpu_hours=float(payload["gpu_hours_produce"]),
    )
    assert "tm3_tm6_pack" in d.mechanism["hidden_at_endpoints"]
    assert "npxxY" in d.mechanism["endpoint_switches"]
    assert "ionic_lock" in d.mechanism["endpoint_switches"]
    assert xtal["tm6_ic"][1] - xtal["tm6_ic"][0] == pytest.approx(0.70, abs=0.02)
    assert xtal["tm3_tm6_pack"][1] - xtal["tm3_tm6_pack"][0] < 0.2


def test_decide_cli_does_not_name_gpu(tmp_path: Path):
    out = tmp_path / "decide.json"
    assert main(["decide", "--out", str(out)]) == 0
    text = out.read_text()
    assert '"gpu": false' in text
    assert "BRIDGE" in text
    assert "MECHANISM" in text


def test_require_not_another_tm6_shot():
    d = decide_b2ar_teacher()
    with pytest.raises(RefuseError) as ei:
        require_not_another_tm6_shot(d)
    assert ei.value.code == "BRIDGE"
