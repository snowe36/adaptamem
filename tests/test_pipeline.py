from pathlib import Path

from adaptamem.cli import main
from adaptamem.features import traces_from_payload, write_features
from adaptamem.pipeline import CPU_STAGES, GPU_STAGES, STAGES, run_is_not_a_campaign


def test_cpu_is_default_gpu_is_oracle_only():
    assert STAGES[0] == "prepare"
    assert "compress" in CPU_STAGES
    assert "infer" in CPU_STAGES
    assert GPU_STAGES == ("oracle",)
    assert "oracle" not in CPU_STAGES


def test_run_refuses_the_mega_pipeline(capsys):
    assert main(["run"]) == 2
    err = capsys.readouterr().err
    assert "mega-run" in err
    assert "adaptamem compress" in err


def test_run_message_names_the_loop():
    text = run_is_not_a_campaign()
    assert "GPU teaches" in text
    assert "oracle on a tiny subset" in text


def test_features_from_values_json(tmp_path: Path):
    payload = {"values": {"tm6_ic": [0.8, 0.9, 1.1]}}
    traces = traces_from_payload(payload)
    dest = write_features(traces, tmp_path / "features.json")
    assert dest.is_file()
    assert traces["tm6_ic"][-1] == 1.1


def test_features_from_assembled_pdb(tmp_path: Path):
    from adaptamem.features import traces_from_structure, write_oracle_request
    from adaptamem.objective import Observable

    pdb = tmp_path / "assembled.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A 131       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A 272      10.000   0.000   0.000  1.00  0.00           C\n"
    )
    obs = [
        Observable(
            name="tm6_ic",
            kind="distance",
            selection="name CA and resid 131 ; name CA and resid 272",
            precision=0.2,
        )
    ]
    traces = traces_from_structure(pdb, obs)
    assert abs(traces["tm6_ic"][0] - 1.0) < 1e-9
    req = write_oracle_request(
        tmp_path / "oracle_request.json",
        uncertain={"tm6_ic": [0]},
        teacher_frames={"tm6_ic": 1},
    )
    assert "short MD from each crystal start" in req.read_text()
