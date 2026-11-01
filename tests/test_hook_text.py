import base64
import json
import sys

import pytest

from kernel_hooks import (
    HookDefinition,
    PanelEntry,
    build_payload,
    event_id,
    matches,
    render_panel,
    resolve_argv,
    sanitize,
    parse_outcome,
    HookArgvError,
)


def definition(event="post_tool", **changes):
    values = dict(
        name="note",
        event=event,
        argv=(sys.executable, "{paths}"),
        files=(),
        tools=("shell", "mcp_*"),
        changed=True,
        paths=("*.py",),
        timeout=30,
        network="none",
        on_error="block",
        max_blocks=3,
        source_index=0,
        executable=sys.executable,
        executable_digest="abc",
    )
    values.update(changes)
    return HookDefinition(**values)


def test_matchers_and_expanded_paths(tmp_path):
    hook = definition()
    assert matches(hook, tool="shell", changed_paths=("good.py",))
    assert matches(hook, tool="mcp_server", changed_paths=("good.py",))
    assert not matches(hook, tool="done", changed_paths=("good.py",))
    assert not matches(
        hook, tool="shell", changed_paths=("gone.py",), deleted_paths=("gone.py",)
    )
    assert not matches(hook, tool="shell", changed_paths=("readme.md",))
    argv = resolve_argv(hook, tmp_path, paths=("-x.py", "b.py"))
    assert argv[-2:] == ["./-x.py", "./b.py"]
    with pytest.raises(HookArgvError, match="no matching"):
        resolve_argv(hook, tmp_path)
    with pytest.raises(HookArgvError, match="invalid"):
        resolve_argv(hook, tmp_path, paths=("bad\n.py",))
    with pytest.raises(HookArgvError, match="256"):
        resolve_argv(hook, tmp_path, paths=tuple(f"f{i}.py" for i in range(257)))
    with pytest.raises(HookArgvError, match="64 KiB"):
        resolve_argv(
            hook,
            tmp_path,
            paths=tuple(f"{i:03d}" + "x" * 300 + ".py" for i in range(240)),
        )


def test_payload_overflow_and_stable_identity():
    hook = definition(argv=(sys.executable,))
    ident = event_id("seed", "post_tool:call7", 3)
    assert ident == event_id("seed", "post_tool:call7", 3)
    large = "x" * (300 << 10)
    result = build_payload(
        "session_start",
        hook,
        event_id=ident,
        run_id="run",
        cwd="/tmp/work-a",
        data={"task": large},
        version="0.2.1",
    )
    assert result.error is None and result.reduced["task"] > 0
    assert len(result.payload) <= 256 << 10
    gate = build_payload(
        "pre_tool",
        hook,
        event_id=ident,
        run_id="run",
        cwd="/tmp/work-a",
        data={"args": large},
        version="0.2.1",
    )
    assert gate.payload is None and gate.error == "input_size"
    other = build_payload(
        "session_start",
        hook,
        event_id=ident,
        run_id="run",
        cwd="/tmp/work-b",
        data={"task": large},
        version="0.2.1",
    )
    assert result.identity_digest == other.identity_digest


def test_post_tool_output_is_bounded_before_transport():
    hook = definition(argv=(sys.executable,))
    result = build_payload(
        "post_tool",
        hook,
        event_id="event",
        run_id="run",
        cwd="/tmp/workspace",
        data={"output": "é" * 20000},
        version="0.2.1",
    )
    data = json.loads(result.payload)["data"]
    assert len(data["output"].encode()) <= 16 << 10
    assert data["output_truncated"] is True


def test_nonfinite_payload_fails_closed():
    hook = definition(argv=(sys.executable,))
    result = build_payload(
        "pre_tool",
        hook,
        event_id="event",
        run_id="run",
        cwd="/tmp/workspace",
        data={"args": {"n": float("nan")}},
        version="0.2.1",
    )
    assert result.payload is None and result.error == "input_invalid"


def test_sanitizer_and_panel_budget():
    cleaned = sanitize(
        "\x1b[31mred\x1b[0m\rnext\nSTATEFRAME_FRAMEFIT_KEEP_K\x00", limit=2048
    )
    assert (
        "\x1b" not in cleaned.text and "STATEFRAME_FRAMEFIT_KEEP_K" not in cleaned.text
    )
    assert cleaned.replaced_literals == 1
    entries = [
        PanelEntry("start", "issue", "session_start", "", "context"),
        PanelEntry(
            "note", "lint", "post_tool", "after shell, call 7", "failed\nfix this"
        ),
    ]
    panel = render_panel(
        entries, room_tokens=2048, count_tokens=lambda text: len(text) // 4
    )
    assert panel.message["_request_budget_section"] == "user_hooks"
    assert "| failed\n| fix this" in panel.message["content"]
    assert render_panel(entries, room_tokens=1, count_tokens=len).message is None
    less = render_panel(entries, room_tokens=1000, count_tokens=len, max_bytes=170)
    assert less.message is None or "session_start issue" in less.message["content"]


@pytest.mark.parametrize("event,code,stdout,stderr,field,expected", [
    ("session_start", 0, "start context  \n", "", "context", "start context"),
    ("post_tool", 0, "post note  \n", "", "context", "post note"),
    ("pre_tool", 2, "", "denied  \n", "reason", "denied"),
    ("stop", 0, '{"decision":"deny","reason":"veto  \\n"}', "", "reason", "veto"),
])
def test_model_visible_hook_output_trims_terminal_whitespace(
        event, code, stdout, stderr, field, expected):
    out, err = stdout.encode(), stderr.encode()
    outcome = parse_outcome(event, {
        "schema": "stateframe-hook-envelope/1", "exit_code": code,
        "signal": None, "timed_out": False, "launch_error": None,
        "stdout_b64": base64.b64encode(out).decode(),
        "stderr_b64": base64.b64encode(err).decode(),
        "stdout_total": len(out), "stderr_total": len(err),
        "cleanup_ok": True,
    })
    assert getattr(outcome, field) == expected


def test_panel_omitted_bytes_are_counted_once_including_dropped_entries():
    entries = [
        PanelEntry("start", "issue", "session_start", "", "abcdefghij"),
        PanelEntry("note", "lint", "post_tool", "", "1234567890"),
    ]
    full = render_panel(entries, room_tokens=None, count_tokens=len)
    panel = render_panel(
        entries, room_tokens=None, count_tokens=len,
        max_bytes=len(full.message["content"].encode()) - 1,
    )
    assert panel.message is not None
    retained = sum(
        len(line[2:].encode())
        for line in panel.message["content"].splitlines()
        if line.startswith("| ") and "bytes omitted" not in line
    )
    assert 0 < retained < 20
    assert panel.omitted_bytes == 20 - retained
    dropped = render_panel(entries, room_tokens=1, count_tokens=len)
    assert dropped.message is None and dropped.omitted_bytes == 20
