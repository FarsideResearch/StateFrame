"""Stale read window: a file read before it changed on disk must not get handles at the new revision.

Drives the real worker under STATEFRAME_EDIT_PROTOCOL=handles (the runtime used by
test_edit_protocol_switch_wiring). The model reads the rules file (bytes B1). The engine's
workspace copy is rewritten on disk (bytes B2) before the provider body that follows the
read is sent. Assertions run on the bodies the scripted endpoint actually received.
"""
from __future__ import annotations

import json

import pytest

from kernel_edit_handles import CURRENT_SOURCE_PREFIX, parse_map
from kernel_paged_source import source_revision
from test_edit_handles_wire_once import _engine_sends, received  # noqa: F401  (fixture)
from test_edit_protocol_switch_wiring import RULES, RULES_PATH, _drive

B1 = RULES
# Two ways the disk can change under a read. Both keep some B1 lines verbatim, which is what
# lets a verbatim scan of the new bytes find "delivered" text.
B2_PREPEND = "# changed on disk after the read\n" + RULES   # old lines shift down by one
B2_APPEND = RULES + "    # changed on disk after the read\n"  # old lines keep their numbers
READ = {"tool": "read", "required": True, "arguments": {"path": RULES_PATH}}


def _map_text(body: dict) -> str | None:
    """The map block of a body, or None. Asserts the map sits once, in string content."""
    found: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str) and CURRENT_SOURCE_PREFIX in content:
            found.append(content[content.index(CURRENT_SOURCE_PREFIX):])
        elif CURRENT_SOURCE_PREFIX in json.dumps(content):
            pytest.fail("map placed in non-string message content")
    assert len(found) <= 1, "more than one [CURRENT SOURCE] map in one body"
    return found[0] if found else None


def _delivered(body: dict) -> str:
    """Every string message the request carried, with the map block removed."""
    parts: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content.split(CURRENT_SOURCE_PREFIX, 1)[0])
    return "\n".join(parts)


def _revision_source(revision: str, disk_after: str) -> str | None:
    for candidate in (B1, disk_after):
        if source_revision(candidate) == revision:
            return candidate
    return None


def _span_text(source: str, start: int, end: int) -> str:
    return "".join(source.splitlines(keepends=True)[start - 1:end])


def _engine_workspace_files(tmp_path):
    """The run's own working copy that the read and edit tools resolve paths against."""
    return sorted((tmp_path / "worker-tmp").glob(f"react_*/repo/{RULES_PATH}"))


@pytest.mark.parametrize("disk_after", [B2_PREPEND, B2_APPEND], ids=["prepend-line", "append-line"])
def test_disk_change_after_read_mints_no_handle_at_new_revision_over_stale_window(
        received, monkeypatch, tmp_path, disk_after):
    from stateframe.engine.probes import react_repair

    real_ask = react_repair.ask_tools
    sends_seen: list[int] = []

    def ask_then_change_disk(messages, *args, **kwargs):
        # Send 0 carries no source. Before send 1 the file changes on disk; send 1 still holds
        # only the B1 read result from the transcript. Send 2 follows a fresh read of B2.
        if len(sends_seen) == 1:
            engine_files = _engine_workspace_files(tmp_path)
            assert len(engine_files) == 1, f"expected one engine workspace copy, found {engine_files}"
            engine_files[0].write_text(disk_after, encoding="utf-8")
        sends_seen.append(1)
        return real_ask(messages, *args, **kwargs)

    monkeypatch.setattr(react_repair, "ask_tools", ask_then_change_disk)
    _drive(tmp_path, monkeypatch, protocol="handles", stop_at=4, steps=[READ, READ, READ, READ])

    assert len(sends_seen) >= 2, "the disk change never happened before the second send"
    bodies = _engine_sends(received())[:3]
    assert len(bodies) == 3, "expected three engine bodies at the endpoint"

    # Precondition: the stale body carries B1 bytes only, and the fresh body carries B2 bytes.
    stale = bodies[1]
    assert "minimum_spend = 25" in _delivered(stale), "precondition: send 1 must deliver B1 bytes"
    assert "changed on disk" not in _delivered(stale), "precondition: send 1 must not deliver B2 bytes"
    fresh = bodies[2]
    assert "changed on disk" in _delivered(fresh), "precondition: send 2 must deliver the fresh B2 read"

    # (a) Every mapped span is backed by bytes delivered in that request, at the revision it names.
    stale_map = _map_text(stale)
    parsed = parse_map(stale_map) if stale_map else None
    for entry in (parsed or {}).get("files", []):
        source = _revision_source(entry["revision"], disk_after)
        assert source is not None, f"map revision {entry['revision']} matches neither B1 nor B2"
        for handle in entry["handles"]:
            span = _span_text(source, handle["start_line"], handle["end_line"])
            assert span in _delivered(stale), f"{handle['handle']} covers text not delivered in that request"

    # (b) No handle is minted at the B2 revision over the stale B1 window. Acceptable outcomes:
    # no mapped window for this file, or a map that carries the B1 revision.
    for entry in (parsed or {}).get("files", []):
        if entry["path"] == RULES_PATH:
            assert entry["revision"] == source_revision(B1), (
                f"stale window mapped at the new disk revision {entry['revision']}\n{stale_map}")

    # (c) Positive control: the fresh B2 read is mapped at the B2 revision over the whole file.
    fresh_map = _map_text(fresh)
    fresh_parsed = parse_map(fresh_map) if fresh_map else None
    rows = [entry for entry in (fresh_parsed or {}).get("files", []) if entry["path"] == RULES_PATH]
    assert rows, "fresh B2 window was not mapped"
    assert rows[0]["revision"] == source_revision(disk_after)
    total = len(disk_after.splitlines())
    assert any(h["start_line"] == 1 and h["end_line"] == total for h in rows[0]["handles"]), (
        "fresh B2 window is not mapped as the whole file")
