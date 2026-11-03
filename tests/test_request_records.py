"""Preflight request records remain byte exact through lossless sealing."""
from __future__ import annotations

import json

import pytest

from stateframe import inspect_run, request_records, runner


def _run_with_requests(tmp_path):
    run = tmp_path / "run"
    records = run / "requests"
    records.mkdir(parents=True)
    (run / "wire").mkdir()
    originals = {
        "0001.json": b'{"body":{"messages":[{"role":"user"}]},"turn":1}\n',
        "0002.json": b'{ "body": {"tools": []}, "turn": 2 }\n',
    }
    for name, data in originals.items():
        (records / name).write_bytes(data)
    return run, originals


def test_request_record_reader_returns_original_bytes_before_and_after_seal(tmp_path):
    run, originals = _run_with_requests(tmp_path)
    wire_record = run / "wire/0001.request.json"
    wire_record.write_bytes(b'{"wire":"untouched"}\n')
    wire_before = wire_record.read_bytes()

    assert request_records.list_request_records(run) == sorted(originals)
    before = {name: request_records.read_request_record(run, name) for name in originals}
    assert before == originals
    count_before = len(request_records.list_request_records(run))

    request_records.seal_request_records(run)

    assert (run / "requests.pack.xz").is_file()
    assert not list((run / "requests").glob("*.json"))
    assert request_records.list_request_records(run) == sorted(originals)
    after = {name: request_records.read_request_record(run, name) for name in originals}
    assert after == before
    assert len(request_records.list_request_records(run)) == count_before
    assert wire_record.read_bytes() == wire_before


def test_interrupted_request_seal_leaves_loose_records_readable(tmp_path, monkeypatch):
    run, originals = _run_with_requests(tmp_path)

    def interrupt(*_args, **_kwargs):
        raise OSError("injected crash before pack publication")

    monkeypatch.setattr(request_records.os, "replace", interrupt)
    with pytest.raises(OSError, match="injected crash"):
        request_records.seal_request_records(run)

    assert not (run / "requests.pack.xz").exists()
    assert request_records.list_request_records(run) == sorted(originals)
    assert {name: request_records.read_request_record(run, name) for name in originals} == originals


def test_damaged_sealed_pack_raises_request_records_error(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / request_records.PACK_NAME).write_bytes(b"\xfd7zXZ\x00invalid")

    with pytest.raises(request_records.RequestRecordsError, match="damaged"):
        request_records.list_request_records(run)


def test_inspect_and_summary_count_sealed_preflight_records(tmp_path, monkeypatch):
    run, originals = _run_with_requests(tmp_path)
    request_records.seal_request_records(run)
    (run / "summary.json").write_text(json.dumps({"engine_result": "SUBMITTED"}))

    overview = inspect_run.overview(run)
    assert "preflight records: 2" in overview

    monkeypatch.setattr(runner, "_read_send_metrics", lambda *_args: [])
    result = runner.summarize(run, 0.0)
    assert result["requests"] == len(originals)


def test_reading_a_sealed_run_decompresses_the_pack_once(tmp_path, monkeypatch):
    """Readers list then read every record; each read must not rescan the whole xz archive."""
    import tarfile

    run = tmp_path / "run"
    (run / "requests").mkdir(parents=True)
    originals = {f"{index:04d}.json": f'{{"turn":{index}}}\n'.encode() for index in range(1, 41)}
    for name, data in originals.items():
        (run / "requests" / name).write_bytes(data)
    opened = []
    real_open = tarfile.open

    def counting_open(*args, **kwargs):
        opened.append(kwargs.get("mode") or (args[1] if len(args) > 1 else "r"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(request_records.tarfile, "open", counting_open)
    request_records.seal_request_records(run)
    assert len(opened) <= 4
    opened.clear()
    names = request_records.list_request_records(run)
    assert {name: request_records.read_request_record(run, name) for name in names} == originals
    assert len(opened) <= 1
