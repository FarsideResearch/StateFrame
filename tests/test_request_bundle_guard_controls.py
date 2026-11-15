"""Independent decoded-string controls and successful HOME-overlay bundle exports."""
from __future__ import annotations

import json

import pytest

from stateframe import request_records, runner
from stateframe.engine import kernel_checkpoint as checkpoint
from test_checkpoint_home_boundary import _isolated_home, _repo, _scripted_real_run


@pytest.mark.parametrize("location", ["object-key", "duplicate-value", "nested-array"])
def test_decoded_json_secret_locations_are_independently_refused(monkeypatch, location):
    secret = b"A" * 15 + b"\n" + b"B" * 16
    encoded = json.dumps(secret.decode("ascii")).encode("ascii")
    records = {
        "object-key": b'{"nested":{' + encoded + b':"safe"}}',
        "duplicate-value": b'{"content":' + encoded + b',"content":"safe"}',
        "nested-array": b'{"messages":[[null,1,true,' + encoded + b']]}',
    }
    record = records[location]
    assert secret not in record
    assert secret.hex().encode("ascii") not in record
    monkeypatch.setattr(checkpoint, "_SECRETS", [secret, secret.hex().encode("ascii")])

    with pytest.raises(request_records.RequestRecordsError,
                       match="checkpoint refused: registered secret in state"):
        request_records._check_bundle_records({"0001.json": record})


@pytest.mark.parametrize("home_source", ["profile", "extra_env"])
def test_home_overlay_publishes_readable_request_bundle(tmp_path, monkeypatch, home_source):
    home, _secret = _isolated_home(tmp_path, monkeypatch)
    monkeypatch.delenv("STATEFRAME_HOME")
    repo = _repo(tmp_path / "repo")
    run = tmp_path / "run"
    profile = None
    extra_env = {}
    if home_source == "profile":
        switches = runner.profiles.load(runner.profiles.DEFAULT)
        switches["STATEFRAME_HOME"] = str(home)
        profile = tmp_path / "profile.json"
        profile.write_text(json.dumps({
            "schema": "stateframe-product-profile/1", "switches": switches,
        }), encoding="utf-8")
    else:
        extra_env["STATEFRAME_HOME"] = str(home)

    result = _scripted_real_run(
        repo, run, lambda *_args: None, monkeypatch=monkeypatch,
        profile=profile, extra_env=extra_env,
    )

    assert not result.get("final_diff_error")
    assert (run / request_records.PACK_NAME).is_file(), "request-bundle export was refused"
    packed = request_records._pack_records(run / request_records.PACK_NAME)
    assert packed, "successful scripted run must export preflight records"
    for name, data in packed.items():
        assert request_records.read_request_record(run, name) == data
        assert isinstance(json.loads(data), dict)
