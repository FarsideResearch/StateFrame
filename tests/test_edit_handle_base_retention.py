"""Edit handle base retention: the bytes a handle delivered survive a later file move.

Drives the production mint path (kernel_edit_request_map.compose_request_map with the
production workspace reader) and the production base accessor (HandleRecord.base_view).
HandleRecord.text() keeps refusing once the file revision moves; base_view() never raises.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import kernel_edit_handles as handles  # noqa: E402
import kernel_edit_request_map as request_map  # noqa: E402
from kernel_edit_handles import (  # noqa: E402
    DeliveredSpan,
    HandleRecord,
    SpanKind,
    compose_request_map,
    mint_handles,
)
from kernel_paged_source import source_revision  # noqa: E402

EXPECTED_AS_DICT_KEYS = {
    "handle", "request_ordinal", "ordinal", "path", "revision", "kind",
    "start_line", "end_line", "total_lines", "byte_start", "byte_end",
}


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write(root: Path, relative: str, raw: bytes) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def _mint(root: Path, relative: str, shown: str, ordinal: int = 1):
    """Mint through the production path; returns (handle_map, messages)."""
    messages = [{"role": "user", "content": f"[Current source: {relative}]\n{shown}"}]
    placed, handle_map = compose_request_map(
        messages, request_ordinal=ordinal, read_source=request_map._workspace_reader(root))
    assert handle_map is not None, "fixture must mint at least one handle"
    return handle_map, placed


def test_base_survives_outside_edit_and_text_still_refuses(tmp_path: Path) -> None:
    original = b"alpha\nbeta\ngamma\ndelta\n"
    _write(tmp_path, "src/app.py", original)
    handle_map, _ = _mint(tmp_path, "src/app.py", "beta\ngamma\n")
    record = handle_map.lookup("q1h1")
    assert record is not None and record.kind is SpanKind.SPAN
    old_revision = record.revision

    _write(tmp_path, "src/app.py", b"alpha\nBETA\ngamma\ndelta\n")
    moved = (tmp_path / "src/app.py").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        record.text(moved)

    revision, text, start, end = record.base_view()
    assert revision == old_revision
    assert text == "beta\ngamma\n"
    assert text.encode("utf-8") == original[start:end]
    assert (start, end) == (6, 17)


def test_base_view_needs_no_disk_read_after_mint(tmp_path: Path) -> None:
    _write(tmp_path, "src/app.py", b"alpha\nbeta\ngamma\n")
    handle_map, _ = _mint(tmp_path, "src/app.py", "beta\ngamma\n")
    record = handle_map.lookup("q1h1")
    assert record is not None
    (tmp_path / "src/app.py").unlink()
    assert record.base_view()[1] == "beta\ngamma\n"


def test_crlf_bom_and_non_ascii_bytes_preserved_exactly(tmp_path: Path) -> None:
    raw = "﻿café\r\nnaïve — ok\r\nthird line\r\n".encode("utf-8")
    _write(tmp_path, "src/text.py", raw)
    # Whole file shown: REWRITE, so the base is the full file including BOM and CRLF.
    handle_map, _ = _mint(tmp_path, "src/text.py", raw.decode("utf-8"))
    record = handle_map.lookup("q1h1")
    assert record is not None and record.kind is SpanKind.REWRITE
    _, text, start, end = record.base_view()
    assert text is not None
    assert text.encode("utf-8") == raw[start:end] == raw
    assert _sha(raw) == record.base_digest == record.revision


def test_partial_span_with_crlf_and_non_ascii_is_byte_exact(tmp_path: Path) -> None:
    raw = "﻿café\r\nnaïve — ok\r\nthird line\r\nfourth\r\n".encode("utf-8")
    _write(tmp_path, "src/text.py", raw)
    handle_map, _ = _mint(tmp_path, "src/text.py", "naïve — ok\r\nthird line\r\n")
    record = handle_map.lookup("q1h1")
    assert record is not None and record.kind is SpanKind.SPAN
    _, text, start, end = record.base_view()
    assert text == "naïve — ok\r\nthird line\r\n"
    assert raw[start:end] == text.encode("utf-8")


def test_rewrite_base_retained_below_bound(tmp_path: Path) -> None:
    raw = b"one\ntwo\n"
    _write(tmp_path, "src/small.py", raw)
    handle_map, _ = _mint(tmp_path, "src/small.py", "one\ntwo\n")
    record = handle_map.lookup("q1h1")
    assert record is not None and record.kind is SpanKind.REWRITE
    assert record.base_view()[1] == "one\ntwo\n"
    assert record.base_digest == _sha(raw)


def test_rewrite_above_bound_keeps_digest_only(tmp_path: Path) -> None:
    lines = [f"line {index:05d} {'x' * 80}\n" for index in range(240)]
    raw = "".join(lines).encode("utf-8")
    assert len(raw) > handles.MAX_RETAINED_REWRITE_BASE_BYTES
    _write(tmp_path, "src/big.py", raw)
    handle_map, _ = _mint(tmp_path, "src/big.py", "".join(lines))
    record = handle_map.lookup("q1h1")
    assert record is not None and record.kind is SpanKind.REWRITE
    revision, text, start, end = record.base_view()
    assert text is None
    assert revision == record.revision == _sha(raw)
    assert record.base_digest == _sha(raw)
    assert (start, end) == (0, len(raw))


def test_bound_matches_source_window_constant() -> None:
    # The REWRITE retention bound is the F4 source working-set byte limit, kept local so
    # the pure handle module does not import the engine stateframe layer.
    import kernel_stateframe_f4 as f4

    assert handles.MAX_RETAINED_REWRITE_BASE_BYTES == f4.MAX_SOURCE_WORKING_SET_BYTES


def test_request_bytes_unchanged_by_base_retention(tmp_path: Path) -> None:
    source = "alpha\nbeta\ngamma\n"
    _write(tmp_path, "src/app.py", source.encode("utf-8"))
    content = "[Current source: src/app.py]\nbeta\ngamma\n"
    messages = [{"role": "user", "content": content}]
    placed, handle_map = compose_request_map(
        messages, request_ordinal=1, read_source=request_map._workspace_reader(tmp_path))
    assert handle_map is not None
    revision = source_revision(source)
    expected_block = (
        "[CURRENT SOURCE]\n"
        "request q1: 1 handle over 1 file. Each handle covers exactly the lines listed; "
        "lines marked not shown were not delivered in this request and cannot be edited "
        "through these handles.\n"
        f"file src/app.py revision {revision} (3 lines)\n"
        "  q1h1 lines 2-3 (span)\n"
        "  not shown: lines 1"
    )
    # Content ends with a newline, so the map follows after one newline separator.
    assert placed[-1]["content"] == content + "\n" + expected_block


def test_as_dict_shape_unchanged_for_repair_request_bytes(tmp_path: Path) -> None:
    _write(tmp_path, "src/app.py", b"alpha\nbeta\ngamma\n")
    handle_map, _ = _mint(tmp_path, "src/app.py", "beta\ngamma\n")
    record = handle_map.lookup("q1h1")
    assert record is not None
    assert set(record.as_dict()) == EXPECTED_AS_DICT_KEYS
    assert set(handle_map.as_dict()["handles"][0]) == EXPECTED_AS_DICT_KEYS


def test_old_shape_record_without_base_loads_and_reports_none() -> None:
    # Records minted before base retention (or built by callers that do not supply a base)
    # must construct and answer base_view without raising.
    span = DeliveredSpan(path="src/app.py", revision=_sha(b"a\nb\n"), start_line=1,
                         end_line=2, total_lines=2, byte_start=0, byte_end=4,
                         kind=SpanKind.REWRITE)
    record = HandleRecord(handle="q1h1", request_ordinal=1, ordinal=1, span=span)
    assert record.base_view() == (None, None, 0, 4)
    assert record.base_digest is None


def test_base_text_not_leaked_into_repr(tmp_path: Path) -> None:
    _write(tmp_path, "src/app.py", b"secret_token_value\nsecond\n")
    handle_map, _ = _mint(tmp_path, "src/app.py", "secret_token_value\nsecond\n")
    record = handle_map.lookup("q1h1")
    assert record is not None
    assert "secret_token_value" not in repr(record)


def test_mint_handles_direct_path_captures_base_from_span() -> None:
    source = "alpha\nbeta\n"
    span = DeliveredSpan.from_source("src/app.py", source, start_line=2, end_line=2)
    handle_map = mint_handles(3, [span])
    record = handle_map.lookup("q3h1")
    assert record is not None
    assert record.base_view() == (span.revision, "beta\n", 6, 11)
