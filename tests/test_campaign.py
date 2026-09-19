from pathlib import Path

from adaptamem.campaign import find_campaign, reproduce_report, stamp, write_campaign
from adaptamem.errors import RefuseError
from adaptamem.schema import load_protocol


def test_stamp_and_reproduce(tmp_path: Path):
    proto = load_protocol()
    pdb = tmp_path / "x.pdb"
    pdb.write_text("HEADER\nEND\n")
    payload = stamp(structure=pdb, protocol=proto, gpu="CPU", seed=7)
    assert payload["campaign_id"]
    assert payload["system_hash"]
    assert payload["protocol_hash"]
    assert payload["forcefield_hash"]
    assert payload["random_seed"] == 7
    path = write_campaign(tmp_path, payload)
    assert path.is_file()
    text = reproduce_report(str(tmp_path), protocol=proto)
    assert payload["campaign_id"] in text
    assert "match" in text
    text_id = reproduce_report(payload["campaign_id"], search=tmp_path)
    assert payload["campaign_id"] in text_id


def test_missing_campaign_refuses(tmp_path: Path):
    try:
        find_campaign("nope", search=tmp_path)
    except RefuseError as exc:
        assert exc.code == "NOT_READY"
        return
    raise AssertionError("expected RefuseError")
