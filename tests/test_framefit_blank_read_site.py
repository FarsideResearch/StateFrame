"""Clipped blank lines are recoverable only from unchanged read evidence."""

import hashlib
import json

import pytest

from kernel_stateframe_action_ledger import (
    ActionLedgerError,
    _read_current_open_site,
    build_action_context,
)


def _source_with_clipped_blank_lines() -> str:
    return "".join(
        "\n" if line in (100, 101) else f"line {line}\n"
        for line in range(1, 131)
    )


def _blank_read_locator(source: str) -> dict[str, object]:
    return {
        "path": "source.py",
        "lines": {"start": 100, "end": 100},
        "last_action": "read",
        "last_read_revision": "sha256:" + hashlib.sha256(source.encode()).hexdigest(),
        "anchor_line": "",
        "anchor_offset": 0,
        "window_line_count": 1,
        "observed_text": "\n",
        "expected_text": "",
    }


@pytest.mark.parametrize("changed", [False, True])
def test_clipped_blank_read_uses_line_only_with_matching_revision(tmp_path, changed):
    source = _source_with_clipped_blank_lines()
    target = tmp_path / "source.py"
    target.write_text(source)
    revision = "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    read_reply = (
        f"[read page] path=source.py revision={revision} lines=100-100/130 "
        "requested=lines:100-130 effective=lines:100-100@0:1 "
        "clip=max_lines chars=1 bytes=1 truncated=true "
        "decoded_with_replacement=false\n129 more lines\n\n"
    )
    messages = [
        {"role": "user", "content": "Inspect source.py."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "read-1", "type": "function", "function": {
                "name": "read", "arguments": json.dumps({
                    "path": "source.py", "start_line": 100, "end_line": 130}),
            }}]},
        {"role": "tool", "tool_call_id": "read-1", "content": read_reply},
    ]
    if changed:
        target.write_text(source.replace("line 1\n", "changed 1\n", 1))
        with pytest.raises(ActionLedgerError, match="could not be anchored"):
            build_action_context(messages, workspace_root=tmp_path)
    else:
        context = build_action_context(messages, workspace_root=tmp_path)
        assert context["open_edit_site"]["lines"]["start"] == 100
        assert context["open_edit_site"]["lines"]["end"] == 100
        assert context["open_edit_site"]["current_bytes"] == "\n"


@pytest.mark.parametrize(
    "invalid_evidence",
    [
        "modified_bytes",
        "wrong_revision",
        "wrong_action",
        "mismatched_observation",
        "invalid_locator",
        "out_of_range_locator",
        "mismatched_range_count",
    ],
)
def test_clipped_blank_read_refuses_unproved_or_invalid_locator(tmp_path, invalid_evidence):
    source = _source_with_clipped_blank_lines()
    target = tmp_path / "source.py"
    target.write_text(source)
    locator = _blank_read_locator(source)

    if invalid_evidence == "modified_bytes":
        target.write_text(source.replace("line 1\n", "changed 1\n", 1))
    elif invalid_evidence == "wrong_revision":
        locator["last_read_revision"] = "sha256:" + hashlib.sha256(b"other").hexdigest()
    elif invalid_evidence == "wrong_action":
        locator["last_action"] = "edit"
    elif invalid_evidence == "mismatched_observation":
        locator["observed_text"] = "unseen source text\n"
    elif invalid_evidence == "invalid_locator":
        locator["lines"] = {"start": 0, "end": 0}
    elif invalid_evidence == "out_of_range_locator":
        locator["lines"] = {"start": 131, "end": 131}
    elif invalid_evidence == "mismatched_range_count":
        locator["window_line_count"] = 2

    with pytest.raises(ActionLedgerError):
        _read_current_open_site(tmp_path, locator)
