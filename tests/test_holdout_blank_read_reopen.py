"""Blank-only clipped reads must reopen from exact current file evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_stateframe_action_ledger import ActionLedgerError, build_action_context  # noqa: E402


def _read_messages(path: str, snapshot: str, start: int, end: int) -> list[dict[str, object]]:
    lines = snapshot.splitlines(keepends=True)
    assert 1 <= start <= end <= len(lines)
    visible = "".join(lines[start - 1:end])
    raw = snapshot.encode("utf-8")
    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    call_id = "holdout-read-1"
    result = (
        f"[read page] path={path} revision={revision} lines={start}-{end}/{len(lines)} "
        f"chars={len(visible)} bytes={len(visible.encode('utf-8'))} "
        "clip=max_lines truncated=true\ncurrent source\n"
        f"{visible}"
    )
    return [
        {"role": "assistant", "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"path": path})},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _write(root: Path, path: str, content: str) -> Path:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8"))
    return target


def _current_site(context: dict[str, object], raw: bytes) -> dict[str, object]:
    site = context.get("open_edit_site")
    assert isinstance(site, dict), "a usable current open edit site should be returned"
    current = raw.decode("utf-8")
    lines = current.splitlines(keepends=True)
    site_lines = site["lines"]
    assert isinstance(site_lines, dict)
    start, end = site_lines["start"], site_lines["end"]
    assert type(start) is int and type(end) is int
    if lines:
        assert 1 <= start <= end <= len(lines)
        expected = "".join(lines[start - 1:end])
    else:
        expected = ""
    assert site["current_bytes"] == expected
    assert site["revision"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert site["content_sha256"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    return site


def _refused_or_current(
    root: Path, path: str, messages: list[dict[str, object]],
) -> dict[str, object] | None:
    try:
        context = build_action_context(messages, workspace_root=root)
    except ActionLedgerError:
        return None
    target = root / path
    assert target.is_file(), "a deleted read target must be refused"
    return _current_site(context, target.read_bytes())


@pytest.mark.parametrize(
    ("blank_line", "tail"),
    [
        ("   \n", "TAIL = 1\n"),
        ("\t\t\n", "TAIL = 1\n"),
        ("\f", ""),
    ],
    ids=("spaces", "tabs", "form-feed"),
)
def test_unique_whitespace_only_clipped_read_reopens_current_bytes(
    tmp_path: Path, blank_line: str, tail: str,
) -> None:
    path = "sample.py"
    snapshot = "HEAD = 1\n" + blank_line + tail
    target = _write(tmp_path, path, snapshot)

    messages = [{"role": "user", "content": "Task: inspect the source."}]
    messages.extend(_read_messages(path, snapshot, 2, 2))
    context = build_action_context(messages, workspace_root=tmp_path)

    site = _current_site(context, target.read_bytes())
    assert site["lines"] == {"start": 2, "end": 2, "total": len(snapshot.splitlines(keepends=True))}
    assert site["current_bytes"] == blank_line


@pytest.mark.parametrize(
    ("snapshot", "start", "end", "visible"),
    [
        (" \nVALUE = 1\n", 1, 1, " \n"),
        ("VALUE = 1\n\t", 2, 2, "\t"),
    ],
    ids=("first-line", "last-line"),
)
def test_blank_first_and_last_file_lines_reopen(
    tmp_path: Path, snapshot: str, start: int, end: int, visible: str,
) -> None:
    path = "boundary.py"
    target = _write(tmp_path, path, snapshot)
    messages = [{"role": "user", "content": "Task: inspect the source."}]
    messages.extend(_read_messages(path, snapshot, start, end))

    context = build_action_context(messages, workspace_root=tmp_path)

    site = _current_site(context, target.read_bytes())
    assert site["current_bytes"] == visible
    assert site["lines"]["start"] == start


def test_repeated_multi_blank_window_reopens_at_its_exact_line_range(tmp_path: Path) -> None:
    path = "blank-run.py"
    snapshot = "\n\n\n\n"
    target = _write(tmp_path, path, snapshot)
    messages = [{"role": "user", "content": "Task: inspect the source."}]
    messages.extend(_read_messages(path, snapshot, 2, 3))

    context = build_action_context(messages, workspace_root=tmp_path)

    site = _current_site(context, target.read_bytes())
    assert site["lines"] == {"start": 2, "end": 3, "total": 4}
    assert site["current_bytes"] == "\n\n"


def test_mixed_blank_and_text_clipped_window_keeps_current_behavior(tmp_path: Path) -> None:
    path = "mixed.py"
    snapshot = " \nVALUE = 1\n\t\n"
    target = _write(tmp_path, path, snapshot)
    messages = [{"role": "user", "content": "Task: inspect the source."}]
    messages.extend(_read_messages(path, snapshot, 1, 3))

    context = build_action_context(messages, workspace_root=tmp_path)

    site = _current_site(context, target.read_bytes())
    assert site["lines"] == {"start": 1, "end": 3, "total": 3}
    assert site["current_bytes"] == snapshot


@pytest.mark.parametrize(
    "case",
    (
        "same-size-changed-bytes",
        "deleted-file",
        "different-range",
        "different-revision",
        "crlf-against-lf",
        "missing-trailing-newline",
        "empty-file",
    ),
)
def test_mismatched_read_is_refused_or_returns_only_current_bytes(
    tmp_path: Path, case: str,
) -> None:
    path = "stale.py"
    if case == "missing-trailing-newline":
        snapshot, start, end = "START = 1\nEND = 1\n \n", 3, 3
    else:
        snapshot, start, end = "START = 1\n \nEND = 1\n", 2, 2
    target = _write(tmp_path, path, snapshot)
    messages = [{"role": "user", "content": "Task: inspect the source."}]
    messages.extend(_read_messages(path, snapshot, start, end))

    if case == "same-size-changed-bytes":
        changed = snapshot.replace(" \n", "X\n", 1)
        assert len(changed.encode("utf-8")) == len(snapshot.encode("utf-8"))
        target.write_bytes(changed.encode("utf-8"))
    elif case == "deleted-file":
        target.unlink()
    elif case == "different-range":
        target.write_bytes(("PREFIX = 0\n" + snapshot).encode("utf-8"))
    elif case == "different-revision":
        target.write_bytes(snapshot.replace("START = 1", "START = 2", 1).encode("utf-8"))
    elif case == "crlf-against-lf":
        target.write_bytes(snapshot.replace("\n", "\r\n").encode("utf-8"))
    elif case == "missing-trailing-newline":
        target.write_bytes(snapshot[:-1].encode("utf-8"))
    elif case == "empty-file":
        target.write_bytes(b"")
    else:
        raise AssertionError(f"unhandled case: {case}")

    site = _refused_or_current(tmp_path, path, messages)
    if site is not None and case == "different-range":
        assert site["lines"]["start"] == 3
    if site is not None and case == "different-revision":
        assert site["lines"]["start"] == 2
