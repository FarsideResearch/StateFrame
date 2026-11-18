"""Reading all preflight records lists the loose directory once and keeps the sealed bytes exact."""
from __future__ import annotations

import hashlib

import pytest

from stateframe import request_records

PACK_NAME = request_records.PACK_NAME

# sha256 of the sealed pack for the three-record fixture below, pinned on the pre-change tree.
PINNED_PACK_SHA256 = "e638fbcd77b2bcbf5fd2e53b7629af378ad3e5012127631648be7fa173159715"


@pytest.fixture(autouse=True)
def _legacy_direct_engine_checkpoint(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE", "1")


def _loose_run(tmp_path, records: dict[str, bytes]):
    run = tmp_path / "run"
    (run / "requests").mkdir(parents=True)
    (run / "wire").mkdir()
    for name, data in records.items():
        (run / "requests" / name).write_bytes(data)
    return run


def _count_loose_listings(monkeypatch) -> list[object]:
    listings: list[object] = []
    original = request_records._loose_files

    def counting(run_dir):
        listings.append(run_dir)
        return original(run_dir)

    monkeypatch.setattr(request_records, "_loose_files", counting)
    return listings


def _mixed_run(tmp_path):
    """Sealed + loose records with precedence, plus entries that are not records."""
    run = tmp_path / "run"
    (run / "requests").mkdir(parents=True)
    (run / "wire").mkdir()
    sealed = {
        "0001.json": b"sealed-1\n",
        "0003.json": b"sealed-3\n",
        "0010.json": b"sealed-10\n",
    }
    with (run / PACK_NAME).open("wb") as stream:
        request_records._write_archive(stream, sealed)
    loose = {
        "0001.json": b"loose-1\n",  # loose copy must win over the sealed copy
        "0002.json": b"loose-2\n",
        "0010.json": b"sealed-10\n",
        "12345.json": b"loose-12345\n",  # sorts after the four-digit names
    }
    for name, data in loose.items():
        (run / "requests" / name).write_bytes(data)
    requests = run / "requests"
    (requests / "abc.json").write_bytes(b"not a record\n")
    (requests / "1.json").write_bytes(b"too short\n")
    (requests / "0004.txt").write_bytes(b"wrong suffix\n")
    (requests / "0005.json").mkdir()
    target = run / "elsewhere.bin"
    target.write_bytes(b"symlink target\n")
    (requests / "0006.json").symlink_to(target)
    return run


def test_all_records_lists_the_loose_directory_once(tmp_path, monkeypatch):
    originals = {f"{index:04d}.json": f'{{"turn":{index}}}\n'.encode() for index in range(1, 6)}
    run = _loose_run(tmp_path, originals)
    listings = _count_loose_listings(monkeypatch)

    assert request_records._all_records(run) == originals
    assert len(listings) == 1


def test_request_pack_size_lists_the_loose_directory_once(tmp_path, monkeypatch):
    originals = {f"{index:04d}.json": f'{{"turn":{index}}}\n'.encode() for index in range(1, 6)}
    run = _loose_run(tmp_path, originals)
    listings = _count_loose_listings(monkeypatch)

    assert request_records.request_pack_size(run) > 0
    assert len(listings) == 1


def test_all_records_matches_per_record_reads_on_mixed_storage(tmp_path):
    run = _mixed_run(tmp_path)
    expected_names = ["0001.json", "0002.json", "0003.json", "0010.json", "12345.json"]

    per_record = {name: request_records.read_request_record(run, name)
                  for name in request_records.list_request_records(run)}

    assert request_records.list_request_records(run) == expected_names
    assert list(per_record) == expected_names
    all_records = request_records._all_records(run)
    assert list(all_records) == expected_names
    assert all_records == per_record
    assert all_records["0001.json"] == b"loose-1\n"
    assert all_records["0003.json"] == b"sealed-3\n"


def test_all_records_reads_sealed_copy_when_listed_loose_file_is_gone(tmp_path, monkeypatch):
    run = _mixed_run(tmp_path)
    original = request_records._loose_files

    def listed_then_removed(root):
        listing = dict(original(root))
        listing["0003.json"] = root / "requests" / "gone.json"  # listed, then unlinked before read
        return listing

    monkeypatch.setattr(request_records, "_loose_files", listed_then_removed)

    records = request_records._all_records(run)
    assert records["0003.json"] == b"sealed-3\n"
    assert records["0001.json"] == b"loose-1\n"


def test_sealed_pack_bytes_and_size_are_unchanged(tmp_path, monkeypatch):
    from stateframe.engine import kernel_checkpoint as checkpoint

    monkeypatch.delenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE")
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", False)
    originals = {
        "0001.json": b'{"turn":1}\n',
        "0002.json": b'{"turn":2}\n',
        "0010.json": b'{"turn":10}\n',
    }
    run = _loose_run(tmp_path, originals)
    predicted_size = request_records.request_pack_size(run)

    removed = request_records.seal_request_records(run)

    pack = run / PACK_NAME
    assert removed == sum(len(data) for data in originals.values())
    assert pack.stat().st_size == predicted_size
    assert hashlib.sha256(pack.read_bytes()).hexdigest() == PINNED_PACK_SHA256
    assert not (run / "requests" / "0001.json").exists()
    assert {name: request_records.read_request_record(run, name) for name in originals} == originals
