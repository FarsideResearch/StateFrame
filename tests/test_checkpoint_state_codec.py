"""Checkpoint state encoding, compatibility and corruption guards."""
from __future__ import annotations

import contextvars
import hashlib
import io
import json
import pickle
import sys
import zlib

import pytest

from stateframe import resume
from stateframe.engine import kernel_checkpoint as checkpoint


SIGNING_KEY = b"checkpoint-codec-test-key-0123456789abcdef"


def _signed_meta(raw: bytes, stored: bytes, *, codec="zlib1", state_bytes: int | None = None) -> dict:
    meta = {
        "schema": checkpoint.SCHEMA,
        "state_bytes": len(raw) if state_bytes is None else state_bytes,
        "state_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if codec is not None:
        meta.update(state_codec=codec,
                    state_stored_sha256=hashlib.sha256(stored).hexdigest(),
                    state_stored_bytes=len(stored))
    meta["signature"] = checkpoint._signature(meta, meta["state_sha256"])
    return meta


def _write_checkpoint(root, raw: bytes, stored: bytes, *, codec="zlib1", state_bytes=None):
    folder = root / "0001-test"
    checkpoint._private_dir(root)
    checkpoint._private_dir(folder)
    state = folder / checkpoint.STATE_FILE
    state.write_bytes(stored)
    state.chmod(0o600)
    meta = _signed_meta(raw, stored, codec=codec, state_bytes=state_bytes)
    meta_path = folder / checkpoint.META_FILE
    meta_path.write_text(json.dumps(meta))
    meta_path.chmod(0o600)
    return folder, meta


def _rewrite_as_legacy_raw(folder) -> bytes:
    meta_path = folder / checkpoint.META_FILE
    meta = json.loads(meta_path.read_text())
    raw = checkpoint.decode_state((folder / checkpoint.STATE_FILE).read_bytes(), meta)
    (folder / checkpoint.STATE_FILE).write_bytes(raw)
    for field in ("state_codec", "state_stored_sha256", "state_stored_bytes"):
        meta.pop(field, None)
    meta["signature"] = checkpoint._signature(meta, meta["state_sha256"])
    meta_path.write_text(json.dumps(meta))
    return raw


def test_state_writer_round_trips_zlib1_and_keeps_raw_and_stored_digests(monkeypatch):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    value = {"label": "checkpoint state", "repeated": "abc123" * 4096}
    raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream = io.BytesIO()
    writer = checkpoint._StateWriter(stream)
    writer.write(raw)
    writer.finish()
    stored = stream.getvalue()

    meta = _signed_meta(raw, stored)
    assert meta["state_codec"] == "zlib1"
    assert meta["state_bytes"] == len(raw)
    assert meta["state_sha256"] == hashlib.sha256(raw).hexdigest()
    assert meta["state_stored_bytes"] == len(stored)
    assert meta["state_stored_sha256"] == hashlib.sha256(stored).hexdigest()
    assert writer.count == len(raw)
    assert writer.stored_count == len(stored)
    assert len(stored) < len(raw)
    decoded = checkpoint.decode_state(stored, meta)
    assert decoded == raw
    assert pickle.loads(decoded) == value


def test_state_writer_refuses_raw_state_over_the_restore_ceiling(monkeypatch):
    monkeypatch.setattr(checkpoint, "MAX_STATE_BYTES", 4)
    writer = checkpoint._StateWriter(io.BytesIO())
    assert writer.write(b"1234") == 4
    with pytest.raises(checkpoint.CheckpointError, match="raw state exceeds the 4-byte limit"):
        writer.write(b"5")


def _mixed_raw_to_compressed_resume(run):
    value = "saved in raw format"
    checkpoint._STATE["setup"] = {"code": sys._getframe().f_code, "line": 0, "funcs": frozenset()}
    legacy = checkpoint.save(sys._getframe(), 1, run)
    raw = _rewrite_as_legacy_raw(legacy)

    selected = resume.select_checkpoint(run, [{"send": 1}], run / "wire")
    assert selected is not None and selected["dir"] == str(legacy)
    value = "changed before raw restore"
    checkpoint.restore(sys._getframe(), legacy, run)
    assert value == "saved in raw format"

    value = "saved after raw resume"
    compressed = checkpoint.save(sys._getframe(), 2, run)
    meta = json.loads((compressed / checkpoint.META_FILE).read_text())
    assert meta["state_codec"] == "zlib1"
    assert meta["state_bytes"] == len(checkpoint.decode_state(
        (compressed / checkpoint.STATE_FILE).read_bytes(), meta))
    assert meta["state_stored_bytes"] == (compressed / checkpoint.STATE_FILE).stat().st_size

    value = "changed before compressed restore"
    checkpoint.restore(sys._getframe(), compressed, run)
    assert value == "saved after raw resume"

    value = "changed before second raw restore"
    checkpoint.restore(sys._getframe(), legacy, run)
    assert value == "saved in raw format"
    return raw, legacy, compressed


def test_legacy_raw_restore_then_compressed_save_and_restore_each(monkeypatch, tmp_path):
    previous_state = checkpoint._STATE.copy()
    previous_key = checkpoint._SIGNING_KEY.copy()
    previous_secrets = checkpoint._SECRETS.copy()
    previous_scans = checkpoint._SCANS.copy()
    try:
        monkeypatch.setattr(checkpoint, "_is_state_module", lambda *_args: False)
        monkeypatch.setattr(checkpoint, "_context_items", lambda: [])
        monkeypatch.setattr(checkpoint.contextvars, "copy_context", contextvars.Context)
        monkeypatch.setattr(checkpoint, "install_continuous_clock", lambda _saved: None)
        monkeypatch.setenv(checkpoint.KEEP_ENV, "2")
        checkpoint._STATE.clear()
        checkpoint._SCANS.clear()
        checkpoint._SIGNING_KEY.clear()
        checkpoint._SECRETS.clear()
        checkpoint.register_signing_key(SIGNING_KEY)
        run = tmp_path / "run"
        run.mkdir()

        raw, legacy, compressed = _mixed_raw_to_compressed_resume(run)

        legacy_meta = json.loads((legacy / checkpoint.META_FILE).read_text())
        assert "state_codec" not in legacy_meta
        assert (legacy / checkpoint.STATE_FILE).read_bytes() == raw
        assert checkpoint.authenticate(legacy)["state_bytes"] == len(raw)
        assert checkpoint.authenticate(compressed)["state_codec"] == "zlib1"
        assert legacy.is_dir() and compressed.is_dir()
    finally:
        checkpoint._STATE.clear()
        checkpoint._STATE.update(previous_state)
        checkpoint._SIGNING_KEY[:] = previous_key
        checkpoint._SECRETS[:] = previous_secrets
        checkpoint._SCANS.clear()
        checkpoint._SCANS.update(previous_scans)


def test_same_size_stored_tampering_is_rejected_before_zlib(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    raw = pickle.dumps({"state": "verified"}, protocol=pickle.HIGHEST_PROTOCOL)
    stored = zlib.compress(raw, level=1)
    folder, _meta = _write_checkpoint(tmp_path / "checkpoints", raw, stored)
    state = folder / checkpoint.STATE_FILE
    changed = bytearray(state.read_bytes())
    changed[len(changed) // 2] ^= 1
    state.write_bytes(changed)
    assert len(changed) == len(stored)

    def unexpected_decompressor(*_args, **_kwargs):
        pytest.fail("stored checksum guard must run before zlib")

    monkeypatch.setattr(checkpoint.zlib, "decompressobj", unexpected_decompressor)
    with pytest.raises(checkpoint.CheckpointError, match="state checksum mismatch"):
        checkpoint.authenticate(folder)


def test_truncated_stored_file_is_rejected_before_zlib(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    raw = pickle.dumps({"state": "truncated"}, protocol=pickle.HIGHEST_PROTOCOL)
    stored = zlib.compress(raw, level=1)
    folder, _meta = _write_checkpoint(tmp_path / "checkpoints", raw, stored)
    (folder / checkpoint.STATE_FILE).write_bytes(stored[:-1])

    def unexpected_decompressor(*_args, **_kwargs):
        pytest.fail("stored size guard must reject truncation before zlib")

    monkeypatch.setattr(checkpoint.zlib, "decompressobj", unexpected_decompressor)
    with pytest.raises(checkpoint.CheckpointError, match="state checksum mismatch"):
        checkpoint.authenticate(folder)


def test_consistently_hashed_truncated_zlib_stream_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    raw = pickle.dumps({"state": "truncated stream"}, protocol=pickle.HIGHEST_PROTOCOL)
    stored = zlib.compress(raw, level=1)[:-1]
    folder, _meta = _write_checkpoint(tmp_path / "checkpoints", raw, stored)

    with pytest.raises(checkpoint.CheckpointError, match="compressed stream is truncated"):
        checkpoint.authenticate(folder)


def test_small_zip_bomb_payload_is_rejected_by_absolute_cap_before_inflation(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    raw = b"z" * (1 << 20)
    stored = zlib.compress(raw, level=1)
    assert len(stored) < 10_000
    folder, _meta = _write_checkpoint(tmp_path / "checkpoints", raw, stored)
    monkeypatch.setattr(checkpoint, "MAX_STATE_BYTES", 1024)

    def unexpected_decompressor(*_args, **_kwargs):
        pytest.fail("absolute output cap must reject before allocating inflate output")

    monkeypatch.setattr(checkpoint.zlib, "decompressobj", unexpected_decompressor)
    with pytest.raises(checkpoint.CheckpointError, match="exceeds the .*byte limit"):
        checkpoint.authenticate(folder)


def test_understated_state_bytes_bounds_each_inflate(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    raw = b"x" * (1 << 20)
    stored = zlib.compress(raw, level=1)
    folder, _meta = _write_checkpoint(tmp_path / "checkpoints", raw, stored, state_bytes=64)
    real_decompressobj = zlib.decompressobj
    calls = []

    class BoundedDecompressor:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def decompress(self, data, max_length=0):
            output = self.wrapped.decompress(data, max_length)
            calls.append((max_length, len(output)))
            return output

        @property
        def unconsumed_tail(self):
            return self.wrapped.unconsumed_tail

        @property
        def eof(self):
            return self.wrapped.eof

        @property
        def unused_data(self):
            return self.wrapped.unused_data

    monkeypatch.setattr(checkpoint.zlib, "decompressobj",
                        lambda *args, **kwargs: BoundedDecompressor(real_decompressobj(*args, **kwargs)))
    with pytest.raises(checkpoint.CheckpointError, match="exceeds its declared"):
        checkpoint.authenticate(folder)
    assert calls == [(65, 65)]


def test_unknown_signed_codec_has_a_named_checkpoint_error(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    raw = pickle.dumps({"state": "unknown codec"}, protocol=pickle.HIGHEST_PROTOCOL)
    stored = zlib.compress(raw, level=1)
    folder, _meta = _write_checkpoint(tmp_path / "checkpoints", raw, stored, codec="mystery")
    with pytest.raises(checkpoint.CheckpointError, match="unknown checkpoint state codec 'mystery'"):
        checkpoint.authenticate(folder)


def test_codec_metadata_is_hmac_signed(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    raw = pickle.dumps({"state": "signed codec"}, protocol=pickle.HIGHEST_PROTOCOL)
    stored = zlib.compress(raw, level=1)
    folder, meta = _write_checkpoint(tmp_path / "checkpoints", raw, stored)
    meta["state_codec"] = "unknown"
    (folder / checkpoint.META_FILE).write_text(json.dumps(meta))

    def unexpected_decompressor(*_args, **_kwargs):
        pytest.fail("the HMAC guard must run before decompression")

    monkeypatch.setattr(checkpoint.zlib, "decompressobj", unexpected_decompressor)
    with pytest.raises(checkpoint.CheckpointError, match="signature does not verify"):
        checkpoint.authenticate(folder)


def test_resume_preflight_checks_compressed_stored_bytes_before_zlib(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [SIGNING_KEY])
    raw = pickle.dumps({"state": "preflight"}, protocol=pickle.HIGHEST_PROTOCOL)
    stored = zlib.compress(raw, level=1)
    run = tmp_path / "run"
    root = run / "checkpoints"
    folder, meta = _write_checkpoint(root, raw, stored)
    meta.update(seq=1, sends=0, created=1)
    (folder / checkpoint.META_FILE).write_text(json.dumps(meta))
    state = folder / checkpoint.STATE_FILE
    changed = bytearray(state.read_bytes())
    changed[-2] ^= 1
    state.write_bytes(changed)

    def unexpected_decompressor(*_args, **_kwargs):
        pytest.fail("resume preflight must verify stored bytes before zlib")

    monkeypatch.setattr(checkpoint.zlib, "decompressobj", unexpected_decompressor)
    assert resume.select_checkpoint(run, [{"send": 1}], run / "wire") is None
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert events[-1]["detail"] == "checkpoint state checksum mismatch"
