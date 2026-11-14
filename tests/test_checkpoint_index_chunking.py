"""Append-only checkpoint indexes use bounded journal-cache chunks."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import types
import zlib
from pathlib import Path

import pytest

from detreason.world_knowledge import WorldKnowledgeStore
from kernel_continuation_journal import ContinuationJournal
from kernel_generated_artifacts import GeneratedArtifactIndex
from kernel_history_spool import EventLookup, HistorySpool
from kernel_recall import RecallStore
from kernel_timeline import TimelineIndex
from stateframe import _child, resume
from stateframe.engine import kernel_checkpoint as checkpoint


@pytest.fixture(autouse=True)
def _legacy_direct_engine_checkpoint(tmp_path, monkeypatch):
    isolated_home = tmp_path / "stateframe-home"
    isolated_home.mkdir(mode=0o700)
    isolated_user_home = tmp_path / "user-home"
    isolated_user_home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(isolated_user_home))
    monkeypatch.setenv("STATEFRAME_HOME", str(isolated_home))
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_CONFIGURED", False)
    monkeypatch.setattr(checkpoint, "_RUNNER_HOME_IDENTITY_PIN", None)
    monkeypatch.setattr(checkpoint, "_PRODUCT_HOST", False)
    monkeypatch.setattr(checkpoint, "_FINGERPRINT", None)
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [])
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_VERIFIED_POOL_IDENTITIES", set())
    monkeypatch.setattr(checkpoint, "_POOL_KNOWN", {})


_CONTEXT = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
_CASES = (
    ("journal_hashes", "ContinuationJournal._record_hashes", 128, 72),
    ("journal_offsets", "ContinuationJournal._record_offsets", 128, 72),
    ("timeline_positions", "TimelineIndex._position_by_id", 256, 96),
    ("generated_events", "GeneratedArtifactIndex._events", 128, 96),
    ("world_events", "WorldKnowledgeStore._events", 16, 640),
    ("recall_items", "RecallStore._items", 16, 640),
)


def _new_owner(case: str, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    if case.startswith("journal_"):
        owner = object.__new__(ContinuationJournal)
        owner.path = directory / "events.jsonl"
        owner._fd = None
        owner._record_cache = []
        owner._record_cache_sizes = []
        owner._record_cache_start = 0
        owner._record_cache_bytes = 0
        owner._record_cache_limit = None
        owner._record_hashes = []
        owner._record_offsets = []
        owner._sequence = 0
        owner._head = None
        owner._size = 0
        owner._tool_reconciliation_records = {}
        return owner
    if case == "timeline_positions":
        owner = TimelineIndex("profile-session")
        owner._events = HistorySpool(directory=directory)
        owner._position_by_id = {}
        owner._by_id = EventLookup(owner._events, owner._position_by_id)
        return owner
    if case == "generated_events":
        return GeneratedArtifactIndex("profile-run")
    if case == "world_events":
        return WorldKnowledgeStore({"sandbox_id": "profile-sandbox"})
    if case == "recall_items":
        return RecallStore()
    raise AssertionError(f"unknown candidate {case}")


def _index_key(case: str, owner, root: Path) -> str:
    if case.startswith("journal_"):
        kind = "journal-hashes" if case == "journal_hashes" else "journal-offsets"
        return checkpoint._stable_key(kind, checkpoint._journal_key(owner))
    if case == "timeline_positions":
        return checkpoint._stable_key("timeline-position", checkpoint._stable_key("timeline", owner.session_id))
    if case == "generated_events":
        return checkpoint._stable_key("generated-artifacts", owner.run_id)
    if case == "world_events":
        scope = json.dumps(owner.scope, sort_keys=True, separators=(",", ":"))
        return checkpoint._stable_key("world-knowledge-events", root, scope)
    if case == "recall_items":
        first = owner._items[0].get("id", "") if owner._items else ""
        return checkpoint._stable_key("recall-items", root, first)
    raise AssertionError(f"unknown candidate {case}")


def _noise(case: str, serial: int) -> str:
    return hashlib.shake_256(f"{case}:{serial}".encode()).hexdigest(256)


def _append_batch(owner, case: str, step: int, count: int) -> None:
    for offset in range(count):
        serial = (step - 1) * count + offset + 1
        sample = _noise(case, serial)
        if case.startswith("journal_"):
            owner._record_hashes.append(hashlib.sha256(f"{serial}:{sample}".encode()).hexdigest())
            owner._record_offsets.append(serial * 173)
            owner._sequence = len(owner._record_hashes)
            owner._head = owner._record_hashes[-1]
        elif case == "timeline_positions":
            event_id = hashlib.sha256(f"timeline:{serial}".encode()).hexdigest()
            owner._position_by_id[event_id] = len(owner._position_by_id)
        elif case == "generated_events":
            event = {
                "schema": "stateframe-audit/1", "run_id": owner.run_id,
                "sequence": serial, "event_id": f"{owner.run_id}:{serial}",
                "kind": "profile_sample", "sample": sample,
            }
            owner.mark_unobserved(event, error_type="ProfileSample")
        elif case == "world_events":
            owner._emit({"event_id": f"world:{serial}", "kind": "profile_sample", "sample": sample})
        elif case == "recall_items":
            owner.append("observation", {"sample": sample}, gist=f"sample {serial}",
                         item_id=f"sample-{serial}")
        else:
            raise AssertionError(f"unknown candidate {case}")


def _dump(owner, root: Path, previous: dict | None = None) -> tuple[bytes, dict]:
    checkpoint._SAVE.clear()
    checkpoint._SAVE.update(root=root, index=previous or {})
    try:
        raw = checkpoint.dumps(owner, _CONTEXT)
        return raw, dict(checkpoint._SAVE.get("index_out", {}))
    finally:
        checkpoint._SAVE.clear()


def _load(raw: bytes, root: Path, index: dict):
    checkpoint._SAVE.clear()
    checkpoint._SAVE.update(load_root=root, load_index=index)
    try:
        return checkpoint._Unpickler(io.BytesIO(raw), _CONTEXT).load()
    finally:
        checkpoint._SAVE.clear()


def test_stale_index_chunk_temporary_is_removed_only_after_writer_dies(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    cache = root / checkpoint.CACHE_DIR
    cache.mkdir(parents=True)
    live_pid = os.getpid()
    dead_pid = max(live_pid + 1_000_000, 1_000_000)
    while checkpoint._pid_alive(dead_pid):
        dead_pid += 1
    assert checkpoint._pid_alive(live_pid)

    held_temps: list[Path] = []
    real_replace = checkpoint.os.replace

    def hold_chunk_temp(source, destination):
        source_path = Path(source)
        if source_path.parent == cache and source_path.name.endswith(
                checkpoint.CHUNK_SUFFIX + ".tmp"):
            held_temps.append(source_path)
            return
        real_replace(source, destination)

    def leave_writer_temp(key: str, writer_pid: int | None = None) -> Path:
        before = len(held_temps)
        if writer_pid is None:
            writer_pid = os.getpid()
            checkpoint._chunk_list(key, ["interrupted chunk payload"])
        else:
            with monkeypatch.context() as writer:
                writer.setattr(checkpoint.os, "getpid", lambda: writer_pid)
                checkpoint._chunk_list(key, ["interrupted chunk payload"])
        assert len(held_temps) == before + 1
        temporary = held_temps[-1]
        prefix = f".{key}-0000000-0000001-{writer_pid}-"
        random_suffix = temporary.name[len(prefix):-len(checkpoint.CHUNK_SUFFIX + ".tmp")]
        assert temporary.name.startswith(prefix)
        assert len(random_suffix) == 16 and all(char in "0123456789abcdef" for char in random_suffix)
        assert temporary.is_file()
        return temporary

    monkeypatch.setattr(checkpoint.os, "replace", hold_chunk_temp)
    monkeypatch.setattr(checkpoint, "_STATE", {"child": None})
    checkpoint._SAVE.clear()
    checkpoint._SAVE.update(root=root, index={})
    try:
        dead_temp = leave_writer_temp("dead-writer-index", dead_pid)
        live_temp = leave_writer_temp("live-writer-index")
        assert not checkpoint._pid_alive(dead_pid)

        assert checkpoint._sweep_stale(root) == 1

        assert not dead_temp.exists()
        assert live_temp.is_file()
    finally:
        checkpoint._SAVE.clear()


def _field(case: str, owner):
    name = {
        "journal_hashes": "_record_hashes",
        "journal_offsets": "_record_offsets",
        "timeline_positions": "_position_by_id",
        "generated_events": "_events",
        "world_events": "_events",
        "recall_items": "_items",
    }[case]
    return getattr(owner, name)


def _measure_series(case: str, directory: Path) -> list[int]:
    root = directory / "checkpoints"
    owner = _new_owner(case, directory)
    previous: dict = {}
    old_names: set[str] = set()
    count = next(spec[2] for spec in _CASES if spec[0] == case)
    samples = []
    for step in range(1, 41):
        _append_batch(owner, case, step, count)
        raw, current = _dump(owner, root, previous)
        key = _index_key(case, owner, root)
        if current:
            assert key in current
        current_names = {row[0] for rows in current.values() for row in rows}
        added = sum((root / checkpoint.CACHE_DIR / name).stat().st_size
                    for name in current_names - old_names)
        samples.append(len(zlib.compress(raw, 1)) + added)
        old_names = current_names
        previous = current
    return samples


def _bounded(samples: list[int], append_bytes: int) -> None:
    assert len(samples) == 40
    assert max(samples[-5:]) <= 3 * max(samples[:5]) + 2 * append_bytes


@pytest.mark.parametrize("case,label,count,item_bytes", _CASES, ids=[row[1] for row in _CASES])
def test_index_chunk_growth_is_bounded_and_whole_pickle_control_fails(
        tmp_path, monkeypatch, case, label, count, item_bytes):
    optimized = _measure_series(case, tmp_path / "optimized")
    append_bytes = count * item_bytes
    _bounded(optimized, append_bytes)

    with monkeypatch.context() as control:
        control.setattr(checkpoint, "_chunk_list", lambda *_args, **_kwargs: None)
        whole_pickle = _measure_series(case, tmp_path / "whole-pickle")
    with pytest.raises(AssertionError):
        _bounded(whole_pickle, append_bytes)


@pytest.mark.parametrize("case,label,count,_item_bytes", _CASES, ids=[row[1] for row in _CASES])
def test_chunked_index_round_trip_is_exact_and_preserves_container_type(
        tmp_path, case, label, count, _item_bytes):
    owner = _new_owner(case, tmp_path)
    _append_batch(owner, case, 1, min(count, 4))
    expected = _field(case, owner)
    original_type = type(expected)
    root = tmp_path / "checkpoints"
    raw, index = _dump(owner, root)
    assert _index_key(case, owner, root) in index

    restored = _load(raw, root, index)
    actual = _field(case, restored)
    assert type(actual) is original_type
    assert checkpoint._same(actual, expected)
    if case == "timeline_positions":
        assert isinstance(restored._by_id, EventLookup)
        assert restored._by_id.positions is restored._position_by_id


def _truncate(owner, case: str) -> None:
    value = _field(case, owner)
    if type(value) is list:
        setattr(owner, {
            "journal_hashes": "_record_hashes", "journal_offsets": "_record_offsets",
            "world_events": "_events", "recall_items": "_items",
        }[case], value[: max(1, len(value) // 2)])
    else:
        rebuilt = dict(list(value.items())[: max(1, len(value) // 2)])
        setattr(owner, "_position_by_id" if case == "timeline_positions" else "_events", rebuilt)
        if case == "timeline_positions":
            owner._by_id.positions = rebuilt


def _rebuild_same_length(owner, case: str) -> None:
    value = _field(case, owner)
    if type(value) is list:
        rebuilt = list(value)
        tail = rebuilt[-1]
        if case == "journal_hashes":
            rebuilt[-1] = hashlib.sha256(b"rebuilt journal hash").hexdigest()
        elif case == "journal_offsets":
            rebuilt[-1] += 1_000_003
        elif case == "world_events":
            rebuilt[-1] = {**tail, "event_id": "world:rebuilt", "sample": "rebuilt"}
        elif case == "recall_items":
            rebuilt[-1] = {**tail, "id": "sample-rebuilt", "gist": "rebuilt"}
        setattr(owner, {"journal_hashes": "_record_hashes", "journal_offsets": "_record_offsets",
                        "world_events": "_events", "recall_items": "_items"}[case], rebuilt)
    else:
        rebuilt = dict(value)
        key = next(reversed(rebuilt))
        item = rebuilt.pop(key)
        replacement = key + 1_000_003 if type(key) is int else f"{key}:rebuilt"
        rebuilt[replacement] = item
        setattr(owner, "_position_by_id" if case == "timeline_positions" else "_events", rebuilt)
        if case == "timeline_positions":
            owner._by_id.positions = rebuilt


@pytest.mark.parametrize("case,label,count,_item_bytes", _CASES, ids=[row[1] for row in _CASES])
def test_truncation_and_rebuild_start_fresh_chunk_lineages(tmp_path, case, label, count, _item_bytes):
    owner = _new_owner(case, tmp_path)
    _append_batch(owner, case, 1, min(count, 8))
    root = tmp_path / "checkpoints"
    _raw, first = _dump(owner, root)
    key = _index_key(case, owner, root)
    old_row = first[key][-1]
    old_chunk = root / checkpoint.CACHE_DIR / old_row[0]
    old_bytes = old_chunk.read_bytes()

    _truncate(owner, case)
    _raw, truncated = _dump(owner, root, first)
    truncated_row = truncated[key][-1]
    assert truncated_row[1] == 0
    assert truncated_row[0] != old_row[0]
    assert old_chunk.read_bytes() == old_bytes


    _rebuild_same_length(owner, case)
    _raw, rebuilt = _dump(owner, root, truncated)
    rebuilt_row = rebuilt[key][-1]
    assert rebuilt_row[1] == 0
    assert rebuilt_row[0] not in {old_row[0], truncated_row[0]}
    assert old_chunk.read_bytes() == old_bytes


def test_saved_chunked_indexes_pass_preflight_and_authentication_with_signed_control(
        tmp_path, monkeypatch, _legacy_direct_engine_checkpoint):
    # This writer feeds the product resume selector, so use the child startup contract.
    checkpoint.configure_product_host()
    run = tmp_path / "run"
    root = run / "checkpoints"
    journal = _new_owner("journal_hashes", tmp_path / "journal")
    _append_batch(journal, "journal_hashes", 1, 2)
    timeline = _new_owner("timeline_positions", tmp_path / "timeline")
    _append_batch(timeline, "timeline_positions", 1, 2)
    generated = _new_owner("generated_events", tmp_path / "generated")
    _append_batch(generated, "generated_events", 1, 2)
    world = _new_owner("world_events", tmp_path / "world")
    _append_batch(world, "world_events", 1, 2)
    recall = _new_owner("recall_items", tmp_path / "recall")
    _append_batch(recall, "recall_items", 1, 2)

    monkeypatch.setattr(checkpoint, "_is_state_module", lambda *_args: False)
    monkeypatch.setattr(checkpoint, "_context_items", lambda: [])
    monkeypatch.setattr(checkpoint, "_STATE", {"setup": {
        "code": test_saved_chunked_indexes_pass_preflight_and_authentication_with_signed_control.__code__,
        "line": 0, "funcs": frozenset(),
    }})
    signing_key = b"index-chunk-test-key-0123456789abcdef"
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [signing_key])
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_PARTICIPANTS", {})
    frame = types.SimpleNamespace(
        f_code=test_saved_chunked_indexes_pass_preflight_and_authentication_with_signed_control.__code__,
        f_locals={"indexes": {"journal": journal, "timeline": timeline,
                              "generated": generated, "world": world, "recall": recall}},
    )

    folder = checkpoint.save(frame, 1, run)
    meta_path = folder / checkpoint.META_FILE
    meta = json.loads(meta_path.read_text())
    required_keys = {
        _index_key("journal_hashes", journal, root),
        _index_key("journal_offsets", journal, root),
        _index_key("timeline_positions", timeline, root),
        _index_key("generated_events", generated, root),
        _index_key("world_events", world, root),
        _index_key("recall_items", recall, root),
    }
    assert required_keys <= set(meta["chunk_index"])
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [])
    monkeypatch.setattr(_child, "checkpoint_key", lambda: signing_key)
    selected = resume.select_checkpoint(run, [{"send": 1}], run / "wire")
    assert selected is not None and selected["dir"] == str(folder)
    assert checkpoint.authenticate(folder)["signature"] == meta["signature"]

    tampered_key = _index_key("world_events", world, root)
    meta["chunk_index"][tampered_key][0][3] = "0" * 64
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    with pytest.raises(checkpoint.CheckpointError, match="signature does not verify"):
        checkpoint.authenticate(folder)
    assert resume.select_checkpoint(run, [{"send": 1}], run / "wire") is None


def _write_signed_checkpoint(tmp_path, monkeypatch, raw: bytes, index: dict,
                             missing_name: str | None = None, *,
                             checkpoint_name: str = "0001-index-chunk", seq: int = 1) -> Path:
    checkpoint.configure_product_host()
    root = tmp_path / "run" / "checkpoints"
    folder = root / checkpoint_name
    checkpoint._private_dir(root)
    checkpoint._private_dir(folder)
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"index-chunk-test-key-0123456789abcdef"])
    stored = zlib.compress(raw, 1)
    (folder / checkpoint.STATE_FILE).write_bytes(stored)
    (folder / checkpoint.STATE_FILE).chmod(0o600)
    meta = {
        "schema": checkpoint.SCHEMA,
        "step": seq,
        "seq": seq,
        "sends": 0,
        "created": seq,
        "entries": [],
        "python": list(sys.version_info[:3]),
        "engine_files_sha256": checkpoint.engine_fingerprint(),
        "state_bytes": len(raw),
        "state_sha256": hashlib.sha256(raw).hexdigest(),
        "state_codec": checkpoint.STATE_CODEC,
        "state_stored_bytes": len(stored),
        "state_stored_sha256": hashlib.sha256(stored).hexdigest(),
        "chunk_index": index,
        "chunks": sorted({row[0] for rows in index.values() for row in rows}),
    }
    meta["signature"] = checkpoint._signature(meta, meta["state_sha256"])
    meta_path = folder / checkpoint.META_FILE
    meta_path.write_text(json.dumps(meta))
    meta_path.chmod(0o600)
    if missing_name is not None:
        (root / checkpoint.CACHE_DIR / missing_name).unlink()
    return folder


@pytest.mark.parametrize("case,label,count,_item_bytes", _CASES, ids=[row[1] for row in _CASES])
def test_missing_chunk_for_each_new_index_is_refused_by_validate_and_selection(
        tmp_path, monkeypatch, case, label, count, _item_bytes):
    owner = _new_owner(case, tmp_path)
    _append_batch(owner, case, 1, min(count, 2))
    root = tmp_path / "run" / "checkpoints"
    raw, index = _dump(owner, root)
    key = _index_key(case, owner, root)
    missing_name = index[key][-1][0]
    folder = _write_signed_checkpoint(tmp_path, monkeypatch, raw, index, missing_name)

    with pytest.raises(checkpoint.CheckpointError, match="journal cache chunk .* missing"):
        checkpoint.validate(folder)
    selected = resume.select_checkpoint(folder.parent.parent, [{"send": 1}], tmp_path / "wire")
    assert selected is None


@pytest.mark.parametrize("case", ("world_events", "recall_items"))
def test_registered_secret_in_chunked_index_is_refused_before_chunk_write(
        tmp_path, monkeypatch, case, _legacy_direct_engine_checkpoint):
    secret = f"registered-secret-fixture-{case}-2026"
    owner = _new_owner(case, tmp_path / case)
    assert (type(owner).__module__, type(owner).__name__) in checkpoint.CHUNKED_TYPES, (
        "the secret regression must exercise a newly chunked index, not only the state.pkl guard"
    )
    if case == "world_events":
        owner._emit({"event_id": "world:secret", "kind": "profile_sample", "detail": secret})
    else:
        owner.append("observation", {"detail": secret}, gist="secret fixture", item_id="secret-recall")

    monkeypatch.setattr(checkpoint, "_is_state_module", lambda *_args: False)
    monkeypatch.setattr(checkpoint, "_context_items", lambda: [])
    monkeypatch.setattr(checkpoint, "_STATE", {"setup": {
        "code": test_registered_secret_in_chunked_index_is_refused_before_chunk_write.__code__,
        "line": 0, "funcs": frozenset(),
    }})
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"index-chunk-test-key-0123456789abcdef"])
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)
    monkeypatch.setattr(checkpoint, "_PARTICIPANTS", {})
    frame = types.SimpleNamespace(
        f_code=test_registered_secret_in_chunked_index_is_refused_before_chunk_write.__code__,
        f_locals={"indexes": {case: owner}},
    )

    run = tmp_path / "run"
    with pytest.raises(checkpoint.CheckpointError,
                       match="a registered secret would be written into the checkpoint"):
        checkpoint.save(frame, 1, run)

    cache = run / "checkpoints" / checkpoint.CACHE_DIR
    if cache.is_dir():
        for chunk in cache.glob(f"*{checkpoint.CHUNK_SUFFIX}"):
            stored = chunk.read_bytes()
            assert secret.encode() not in stored
            try:
                raw = zlib.decompress(stored)
            except zlib.error:
                raw = b""
            assert secret.encode() not in raw


def test_registered_secret_in_audit_chunk_is_refused_before_chunk_write(tmp_path, monkeypatch):
    secret = "registered-secret-audit-fixture-2026"
    relative = "worker/measurement-runs/run/events.jsonl"
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(secret.encode())
    target = tmp_path / "snapshot" / relative
    target.parent.mkdir(parents=True)
    root = tmp_path / "checkpoints"
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    checkpoint.register_secret(secret)

    with pytest.raises(checkpoint.CheckpointError,
                       match="a registered secret would be written into the checkpoint"):
        checkpoint._snapshot_appended_file(source, target, relative, root, [], {}, 1)

    cache = root / checkpoint.CACHE_DIR
    if cache.is_dir():
        for chunk in cache.iterdir():
            stored = chunk.read_bytes()
            assert secret.encode() not in stored
            try:
                raw = zlib.decompress(stored)
            except zlib.error:
                raw = b""
            assert secret.encode() not in raw


def _tree_bytes(root: Path) -> dict[str, bytes | None]:
    result = {}
    for path in sorted(root.rglob("*")):
        result[str(path.relative_to(root))] = path.read_bytes() if path.is_file() else None
    return result


@pytest.mark.parametrize("damage", ("byte-flipped", "truncated"))
def test_corrupt_chunk_is_refused_in_preflight_without_changing_run_and_uses_previous(
        tmp_path, monkeypatch, damage):
    run = tmp_path / "run"
    run.mkdir()
    (run / "request.json").write_bytes(b"live request bytes")
    (run / "setup-output").mkdir()
    (run / "setup-output" / "result.txt").write_bytes(b"live setup output")
    root = run / "checkpoints"
    owner = _new_owner("recall_items", tmp_path / "recall")
    _append_batch(owner, "recall_items", 1, 2)
    good_raw, good_index = _dump(owner, root)
    damaged_raw, damaged_index = _dump(owner, root)
    key = _index_key("recall_items", owner, root)
    good = _write_signed_checkpoint(
        tmp_path, monkeypatch, good_raw, good_index, checkpoint_name="0001-good", seq=1)
    damaged = _write_signed_checkpoint(
        tmp_path, monkeypatch, damaged_raw, damaged_index,
        checkpoint_name="0002-damaged", seq=2)
    chunk_name = damaged_index[key][0][0]
    chunk = root / checkpoint.CACHE_DIR / chunk_name
    assert chunk.is_file()
    stored = chunk.read_bytes()
    if damage == "byte-flipped":
        changed = bytearray(stored)
        changed[len(changed) // 2] ^= 1
        chunk.write_bytes(changed)
    else:
        chunk.write_bytes(stored[:-1])
    chunk.chmod(0o600)

    before = _tree_bytes(run)
    with pytest.raises(checkpoint.CheckpointError) as refusal:
        checkpoint.validate(damaged)
    assert chunk_name in str(refusal.value)
    assert _tree_bytes(run) == before

    refusals = []
    monkeypatch.setattr(
        resume, "_record_checkpoint_refusal",
        lambda _root, folder, detail: refusals.append((folder.name, detail)),
    )
    selected = resume.select_checkpoint(run, [{"send": 1}], run / "wire")
    assert selected is not None and selected["dir"] == str(good)
    assert any(name == damaged.name and chunk_name in detail for name, detail in refusals)
    assert _tree_bytes(run) == before
