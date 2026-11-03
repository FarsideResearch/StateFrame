"""Sealed measurement journals and append-only blob packs preserve old evidence."""
from __future__ import annotations

import hashlib
import json
import lzma
import pickle
import tarfile
import zlib
from types import SimpleNamespace

import pytest

import kernel_journal_store as journal_store
import kernel_measurement as measurement
import kernel_checkpoint as checkpoint
from kernel_checkpoint import _journal_offset, _load_journal_chunk
from kernel_timeline import TimelineIndex
from stateframe import outcomes, resume, runner


def _journal(tmp_path):
    directory = tmp_path / "measurement"
    directory.mkdir()
    path = directory / "events.jsonl"
    path.write_bytes(b'{"sequence":1}\n{"sequence":2}\n')
    return directory, path


def test_journal_store_globals_are_checkpoint_serializable():
    state = checkpoint._module_state("kernel_journal_store", journal_store)
    pickle.dumps(state)


def test_open_journal_and_paths_prefer_live_then_sealed(tmp_path):
    directory, path = _journal(tmp_path)
    (directory / "events.jsonl.xz").write_bytes(lzma.compress(b'wrong\n'))
    with measurement.open_journal(directory) as lines:
        assert list(lines) == ['{"sequence":1}\n', '{"sequence":2}\n']
    assert measurement.journal_paths(tmp_path) == [path]
    path.unlink()
    (directory / "events.jsonl.xz").write_bytes(lzma.compress(b'first\nsecond\n'))
    assert measurement.journal_paths(tmp_path) == [directory / "events.jsonl.xz"]
    with measurement.open_journal(path) as lines:
        assert list(lines) == ["first\n", "second\n"]
    with measurement.open_journal(path, binary=True) as stream:
        stream.seek(6)
        assert stream.readline() == b"second\n"


def test_seal_is_idempotent_and_preserves_checkpoint_offsets(tmp_path):
    directory, path = _journal(tmp_path)
    original = path.read_bytes()
    assert _journal_offset(path, 1) == len(b'{"sequence":1}\n')
    measurement.seal_measurement_run(directory)
    assert not path.exists()
    assert (directory / "events.jsonl.xz").exists()
    assert _journal_offset(path, 1) == len(b'{"sequence":1}\n')
    assert measurement.seal_measurement_run(directory)["journal_bytes"] == len(original)


def test_seal_refuses_live_engine_writer(tmp_path):
    import fcntl

    run = tmp_path / "run"
    directory = run / "worker" / "measurement-runs" / "one"
    directory.mkdir(parents=True)
    path = directory / "events.jsonl"
    path.write_bytes(b'{"sequence":1}\n')
    lock_path = run / ".engine.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = measurement.seal_measurement_run(directory)
        assert result == {"sealed": False, "reason": "live engine holds .engine.lock"}
        assert path.is_file()
        assert not (directory / "events.jsonl.xz").exists()
        fcntl.flock(lock, fcntl.LOCK_UN)


def test_checkpoint_chunk_reads_sealed_journal_at_plain_offset(tmp_path):
    directory, path = _journal(tmp_path)
    data = path.read_bytes()
    offset = data.index(b"\n") + 1
    suffix = data[offset:]
    part = {"journal": str(path), "offset": offset, "end": len(data),
            "sha": hashlib.sha256(suffix).hexdigest(), "entries": [("J", None)], "blobs": {}}
    assert _load_journal_chunk(part) == [{"sequence": 2}]
    measurement.seal_measurement_run(directory)
    assert _load_journal_chunk(part) == [{"sequence": 2}]


@pytest.mark.parametrize("stage", ["compress", "verify", "rename", "directory_fsync", "unlink"])
def test_seal_interruption_keeps_readable_journal(tmp_path, monkeypatch, stage):
    directory, path = _journal(tmp_path)
    expected = path.read_text()
    target = {"compress": "_write_sealed_journal", "verify": "_verify_sealed_journal",
              "rename": "_publish_sealed_journal", "directory_fsync": "_fsync_directory",
              "unlink": "_unlink_plain_journal"}[stage]
    original = getattr(journal_store, target)

    def interrupted(*args, **kwargs):
        if stage in {"directory_fsync", "unlink"}:
            original(*args, **kwargs)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(journal_store, target, interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        measurement.seal_measurement_run(directory)
    with measurement.open_journal(directory) as stream:
        assert stream.read() == expected


def _blob(directory, data):
    digest = hashlib.sha256(data).hexdigest()
    blobs = directory / "blobs"
    blobs.mkdir(exist_ok=True)
    (blobs / f"{digest}.zlib").write_bytes(zlib.compress(data))
    return digest


def test_pack_refuses_live_measurement_journal(tmp_path):
    directory, _ = _journal(tmp_path)
    _blob(directory, b'"first"')
    with pytest.raises(ValueError, match="sealed"):
        measurement.pack_blobs(directory)


def test_pack_reads_old_mixed_and_rebuilds_missing_index(tmp_path):
    directory, journal = _journal(tmp_path)
    first = _blob(directory, b'"first"')
    second = _blob(directory, b'"second"')
    assert measurement._read_journal_blob(journal, first) == b'"first"'
    measurement.seal_measurement_run(directory)
    measurement.pack_blobs(directory)
    assert not list((directory / "blobs").glob("*.zlib"))
    assert measurement._read_journal_blob(journal, first) == b'"first"'
    (directory / "blobs" / "pack-0001.idx").unlink()
    assert measurement._read_journal_blob(journal, second) == b'"second"'
    assert (directory / "blobs" / "pack-0001.idx").is_file()
    third = _blob(directory, b'"third"')
    assert measurement._read_journal_blob(journal, third) == b'"third"'
    assert measurement._read_journal_blob(journal, first) == b'"first"'
    measurement.pack_blobs(directory)
    assert measurement._read_journal_blob(journal, third) == b'"third"'


def test_pack_torn_tail_retry_succeeds_and_reads_every_blob(tmp_path):
    directory, journal = _journal(tmp_path)
    first = _blob(directory, b'"first"')
    measurement.seal_measurement_run(directory)
    measurement.pack_blobs(directory)
    second = _blob(directory, b'"second"')
    third = _blob(directory, b'"third"')
    pack = directory / "blobs" / "pack-0001.sfpk"
    with pack.open("ab") as stream:
        stream.write(b"SFPK\x00\x00")

    measurement.pack_blobs(directory)

    assert not list((directory / "blobs").glob("*.zlib"))
    assert [measurement._read_journal_blob(journal, digest) for digest in (first, second, third)] == [
        b'"first"', b'"second"', b'"third"']


def test_pack_torn_tail_and_crc_corruption(tmp_path):
    directory, journal = _journal(tmp_path)
    first = _blob(directory, b'"first"')
    second = _blob(directory, b'"second"')
    measurement.seal_measurement_run(directory)
    measurement.pack_blobs(directory)
    pack = directory / "blobs" / "pack-0001.sfpk"
    with pack.open("ab") as stream:
        stream.write(b"SFPK\x00\x00")
    (directory / "blobs" / "pack-0001.idx").unlink()
    assert measurement._read_journal_blob(journal, first) == b'"first"'
    assert measurement._read_journal_blob(journal, second) == b'"second"'
    data = bytearray(pack.read_bytes())
    data[40] ^= 1
    pack.write_bytes(data)
    with pytest.raises((ValueError, FileNotFoundError)):
        measurement._read_journal_blob(journal, first)


def test_stale_pack_index_rebuilds(tmp_path):
    directory, journal = _journal(tmp_path)
    digest = _blob(directory, b'"first"')
    measurement.seal_measurement_run(directory)
    measurement.pack_blobs(directory)
    index = directory / "blobs" / "pack-0001.idx"
    index.write_text(json.dumps({"pack_size": 0, "records": {}}))
    assert measurement._read_journal_blob(journal, digest) == b'"first"'


def test_stale_same_size_index_rebuilds(tmp_path):
    directory, journal = _journal(tmp_path)
    digest = _blob(directory, b'"first"')
    measurement.seal_measurement_run(directory)
    measurement.pack_blobs(directory)
    index = directory / "blobs" / "pack-0001.idx"
    saved = json.loads(index.read_text())
    saved["records"] = {}
    index.write_text(json.dumps(saved))
    assert measurement._read_journal_blob(journal, digest) == b'"first"'


def test_pack_interruption_before_index_keeps_loose_blob(tmp_path, monkeypatch):
    directory, journal = _journal(tmp_path)
    digest = _blob(directory, b'"first"')
    measurement.seal_measurement_run(directory)

    def interrupted(*args):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(journal_store, "_write_pack_index", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        measurement.pack_blobs(directory)
    assert (directory / "blobs" / f"{digest}.zlib").is_file()
    assert measurement._read_journal_blob(journal, digest) == b'"first"'


def test_old_archive_tar_member_layout_still_reads(tmp_path):
    directory, journal = _journal(tmp_path)
    digest = _blob(directory, b'"first"')
    archive = tmp_path / "old.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(journal, arcname="run/events.jsonl")
        target.add(directory / "blobs" / f"{digest}.zlib",
                   arcname=f"run/blobs/{digest}.zlib")
    assert measurement._read_journal_blob(f"{archive}::run/events.jsonl", digest) == b'"first"'


def test_host_readers_accept_sealed_measurement_journal(tmp_path):
    run = tmp_path / "run"
    directory = run / "worker" / "measurement-runs" / "one"
    directory.mkdir(parents=True)
    digest = "sha256:" + "a" * 64
    events = [
        {"sequence": 1, "previous_hash": None, "record_hash": "one", "kind": "scope_widen",
         "path": "file.py", "pre_edit_sha256": "base"},
        {"sequence": 2, "previous_hash": "one", "record_hash": "two", "kind": "done_sealed",
         "stored": True, "seal_id": digest, "artifact_digest": digest},
        {"sequence": 3, "previous_hash": "two", "record_hash": "three",
         "kind": "done_submission_stored", "stored": True, "artifact_digest": digest},
    ]
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    measurement.seal_measurement_run(directory)
    assert resume._journal_lineage_ok(run) is None
    assert runner._widen_admissions(run, {"file.py": "base"}) == ["file.py"]
    assert outcomes.journal_submission(run)["result"] == "SUBMITTED"


def test_timeline_uses_uncompressed_offsets_after_seal(tmp_path):
    directory, path = _journal(tmp_path)
    path.write_text('{"event_id":"a","kind":"run_started"}\n')
    timeline = TimelineIndex(journal=SimpleNamespace(path=path))
    timeline.refresh()
    assert len(timeline._events) == 1
    assert timeline.refresh() == 0
    measurement.seal_measurement_run(directory)
    assert timeline.refresh() == 0
    rebuilt = TimelineIndex(journal=SimpleNamespace(path=path))
    rebuilt.refresh()
    assert len(rebuilt._events) == 1
