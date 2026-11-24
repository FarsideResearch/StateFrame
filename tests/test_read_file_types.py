"""Regression coverage for filesystem types and byte decoding in the read tool."""
from __future__ import annotations

import errno
import builtins
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

import kernel_read_exec as reads
import kernel_source_delivery as source

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path, *, args: dict[str, object] | None = None,
          measurement: object | None = None):
    delivery: dict[str, object] = {}
    measurement = measurement or SimpleNamespace(journal=None, retained=None, step_ledger=None)
    result = reads.execute_read(
        path.parent, path, path.name, path.name, 0,
        args=args or {"path": path.name}, measurement=measurement,
        step=1, delivery=delivery,
    )
    return result, delivery


def _fifo_child(path: Path, *, pinned: bool) -> subprocess.CompletedProcess[str]:
    code = """\
import json, sys
from pathlib import Path
from types import SimpleNamespace
from kernel_read_exec import execute_read
p = Path(sys.argv[1])
a = {"path": p.name}
if sys.argv[2] == "pinned": a["expected_revision"] = "sha256:" + "0" * 64
d = {}
r = execute_read(p.parent, p, p.name, p.name, 0, args=a,
    measurement=SimpleNamespace(journal=None, retained=None, step_ledger=None),
    step=1, delivery=d)
print(json.dumps({"result": r[0], "observation": r[1], "metadata": d}), flush=True)
"""
    env = os.environ.copy()
    paths = (ROOT / "src", ROOT / "src/stateframe/engine", ROOT / "tests")
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code, str(path), "pinned" if pinned else "ordinary"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=2,
    )


def _fifo_swap_child(path: Path) -> subprocess.CompletedProcess[str]:
    code = """\
import os, sys
from pathlib import Path
from kernel_source_delivery import NonRegularReadPath, read_regular_file
p = Path(sys.argv[1])
real_open = os.open
def replace_with_fifo(target, flags, *args, **kwargs):
    if Path(target) == p:
        p.unlink()
        os.mkfifo(p)
    return real_open(target, flags, *args, **kwargs)
os.open = replace_with_fifo
try:
    read_regular_file(p)
except NonRegularReadPath:
    print("refused", flush=True)
else:
    raise AssertionError("FIFO replacement was read as a regular file")
"""
    env = os.environ.copy()
    paths = (ROOT / "src", ROOT / "src/stateframe/engine", ROOT / "tests")
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=2,
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes require POSIX")
@pytest.mark.parametrize("pinned", [False, True], ids=["cache-refill", "revision-check"])
def test_fifo_read_is_typed_refusal_without_blocking(tmp_path, pinned):
    path = tmp_path / "source.fifo"
    os.mkfifo(path)

    try:
        completed = _fifo_child(path, pinned=pinned)
    except subprocess.TimeoutExpired:
        pytest.fail("read blocked opening a FIFO instead of returning a typed refusal")

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["result"] == "NON_REGULAR_FILE"
    assert "REFUSED NON_REGULAR_FILE" in result["observation"]
    assert result["metadata"]["refusal_cause"] == "non_regular_file"
    assert result["metadata"]["delivery_status"] == "WITHHELD"


def test_invalid_utf8_uses_replacement_text_and_typed_note(tmp_path):
    raw = b"first \xff byte\nsecond \xfe byte\nvalid \xef\xbf\xbd; broken \xe2(\xa1\n"
    path = tmp_path / "source.txt"
    path.write_bytes(raw)

    (status, observation, _), metadata = _read(path)

    assert status == "OK"
    assert ("first \ufffd byte\nsecond \ufffd byte\n"
            "valid \ufffd; broken \ufffd(\ufffd\n") in observation
    assert "\\xff" not in observation
    note = "not UTF-8: 4 undecodable bytes are shown as U+FFFD; do not copy them into an edit"
    assert note in observation
    assert metadata["decoding_note"] == note


def test_invalid_utf8_count_matches_ignored_utf8_bytes():
    raw = b"real \xef\xbf\xbd; broken \xe2\x82\xff"
    _, replaced, note = source.decode_source(raw)
    count = len(raw) - len(raw.decode("utf-8", "ignore").encode("utf-8"))
    assert count == 3 and replaced and note and f"{count} undecodable bytes" in note


def test_invalid_utf8_count_does_not_call_python_once_per_character(monkeypatch):
    calls = 0
    original = builtins.ord

    def counted(char):
        nonlocal calls
        calls += 1
        return original(char)

    monkeypatch.setattr(builtins, "ord", counted)
    source.decode_source((b"valid text " + b"\xff" * 2000) * 4)
    assert calls == 0


def test_byte_capped_pages_keep_0830_cursor_budget_and_hash():
    from kernel_source_delivery import deliver_source, normalize_read_request

    raw = "éabcdefgh\nnext\n".encode("utf-8")
    request = normalize_read_request({"path": "f", "max_chars": 512}, page_byte_limit=512)
    first = deliver_source(raw, "f", 0, args=request)
    assert first.content == ""
    assert first.metadata["clip_reason"] == "max_bytes"
    assert first.metadata["page_bytes_sha256"] == "sha256:" + hashlib.sha256(b"").hexdigest()
    cursor = first.metadata["next_cursor"]
    assert cursor.endswith(":1:0")
    resumed = deliver_source(raw, "f", 0, args=normalize_read_request(
        {"path": "f", "cursor": cursor, "max_chars": 512}, page_byte_limit=512))
    assert resumed.content == first.content and resumed.metadata["next_cursor"] == cursor
    assert [normalize_read_request({"path": "f"}, page_byte_limit=cap).max_bytes
            for cap in (1, 2, 511)] == [1, 2, 511]


def test_binary_read_reports_size_without_returning_content(tmp_path):
    path = tmp_path / "source.dat"
    path.write_bytes(b"prefix\0secret\xff")

    (status, observation, payload), metadata = _read(path)

    assert status == "BINARY"
    assert "binary file, 14 bytes" in observation
    assert payload is None
    assert metadata["binary_file"] is True
    assert metadata["file_byte_length"] == 14
    assert metadata["delivery_status"] == "WITHHELD"
    assert metadata["requested_bytes_delivered"] == 0


def test_valid_replacement_character_remains_valid_utf8(tmp_path):
    from kernel_source_delivery import deliver_source

    raw = "real � character\n".encode("utf-8")
    page = deliver_source(raw, "source.txt", 0, args={"path": "source.txt"})

    assert page.result_class == "OK"
    assert page.content == "real � character\n"
    assert "�" in page.text and "not UTF-8:" not in page.text
    assert "decoding_note" not in page.metadata
    assert page.metadata["revision"] == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_invalid_utf8_keeps_base_columns_hashes_and_advancing_cursors(tmp_path):
    from kernel_source_delivery import deliver_source, normalize_read_request

    raw = b"first\nsecond \xff byte\nthird\n"
    path = tmp_path / "source.txt"
    path.write_bytes(raw)
    (status, observation, _), metadata = _read(
        path, args={"path": path.name, "line": 2, "max_lines": 1})
    assert status == "OK"
    assert "second \ufffd byte\n" in observation
    assert "first\n" not in observation and "third\n" not in observation
    assert metadata["decoding_note"] == (
        "not UTF-8: 1 undecodable bytes are shown as U+FFFD; do not copy them into an edit")

    raw = bytes((255,)) * 400
    page = deliver_source(raw, "source.txt", 0,
                          args={"path": "source.txt", "max_chars": 512})
    assert page.result_class == "OK"
    assert page.content == "\ufffd"
    assert len(page.text) <= 512
    assert page.metadata["page_bytes_sha256"] == "sha256:" + hashlib.sha256(
        page.content.encode()).hexdigest()
    cursor = page.metadata["next_cursor"]
    assert cursor and int(cursor.split(":")[3]) == 1
    next_page = deliver_source(raw, "source.txt", 0,
                               args={"path": "source.txt", "cursor": cursor, "max_chars": 512})
    assert next_page.metadata["start_column"] == 1 and next_page.content == "\ufffd"
    byte_limited = deliver_source(
        raw, "source.txt", 0,
        args=normalize_read_request({"path": "source.txt", "max_chars": 512}, page_byte_limit=512),
    )
    assert byte_limited.content == ""
    byte_cursor = byte_limited.metadata["next_cursor"]
    assert int(byte_cursor.split(":")[3]) == 0
    byte_next = deliver_source(raw, "source.txt", 0, args=normalize_read_request(
        {"path": "source.txt", "cursor": byte_cursor, "max_chars": 512}, page_byte_limit=512))
    assert byte_next.metadata["start_column"] == 0 and byte_next.content == ""
    assert normalize_read_request({"path": "source.txt"}, page_byte_limit=511).max_bytes == 511
    note = "not UTF-8: 1 undecodable bytes are shown as U+FFFD; do not copy them into an edit"
    narrow = deliver_source(b"\xff\n", "p" * 100, 0,
                            args={"path": "p" * 100, "max_chars": 512})
    assert narrow.metadata["decoding_note"] == note and note not in narrow.text
    path = tmp_path / ("p" * 100)
    path.write_bytes(b"\xff\n")
    (status, observation, _), metadata = _read(
        path, args={"path": path.name, "max_chars": 512})
    assert status == "OK" and metadata["decoding_note"] == note and note not in observation
    assert len(narrow.text) <= 512

    saved = "read1:" + hashlib.sha256(b"a\xffbc\n").hexdigest() + ":1:1"
    restored = deliver_source(b"a\xffbc\n", "source.txt", 0,
                              args={"path": "source.txt", "cursor": saved})
    assert restored.content.startswith("\ufffd")


def test_directory_missing_and_regular_symlink_keep_existing_results(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    assert _read(folder)[0][0] == "DIRECTORY"
    assert _read(tmp_path / "missing.txt")[0][0] == "MISSING"

    target = tmp_path / "target.txt"
    target.write_text("linked text\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target.name)
    result, _ = _read(link)
    assert result[0] == "OK" and "linked text\n" in result[1]


def test_device_path_is_typed_refusal(tmp_path):
    device = Path("/dev/null")
    if not device.exists() or not hasattr(os, "symlink"):
        pytest.skip("a readable device path is unavailable")
    link = tmp_path / "device"
    link.symlink_to(device)

    result, metadata = _read(link)

    assert result[0] == "NON_REGULAR_FILE"
    assert metadata["refusal_cause"] == "non_regular_file"
    assert "REFUSED NON_REGULAR_FILE" in result[1]


def test_socket_open_failure_is_typed_refusal(tmp_path, monkeypatch):
    path = tmp_path / "socket"
    path.write_bytes(b"")
    real_open = source.os.open

    def socket_open(target, flags, *args, **kwargs):
        if Path(target) == path:
            raise OSError(errno.ENXIO, "socket paths cannot be opened as files")
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(source.os, "open", socket_open)
    result, metadata = _read(path)

    assert result[0] == "NON_REGULAR_FILE"
    assert metadata["refusal_cause"] == "non_regular_file"


def test_regular_read_works_without_nonblocking_flag(tmp_path, monkeypatch):
    path = tmp_path / "source.txt"
    path.write_bytes(b"regular file\n")
    real_open = source.os.open
    opened_flags = []
    def track_flags(target, flags, *args, **kwargs):
        opened_flags.append(flags)
        return real_open(target, flags, *args, **kwargs)
    monkeypatch.setattr(source.os, "open", track_flags)
    if hasattr(os, "O_NONBLOCK"):
        reads._read_regular_file(path)
        assert opened_flags[-1] & os.O_NONBLOCK
    monkeypatch.delattr(source.os, "O_NONBLOCK", raising=False)

    events = []
    real_stat = Path.stat
    def checked_stat(target, *args, **kwargs):
        if target == path:
            events.append("stat")
        return real_stat(target, *args, **kwargs)
    def checked_open(target, flags, *args, **kwargs):
        if Path(target) == path:
            events.append("open")
        return real_open(target, flags, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", checked_stat)
    monkeypatch.setattr(source.os, "open", checked_open)

    content, signature = reads._read_regular_file(path)

    assert content == b"regular file\n"
    assert signature is not None
    assert events.index("stat") < events.index("open")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes require POSIX")
def test_path_changed_to_fifo_between_stat_and_open_is_refused(tmp_path):
    path = tmp_path / "source.txt"
    path.write_bytes(b"regular\n")
    completed = _fifo_swap_child(path)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "refused"


def test_append_between_stat_and_open_returns_appended_content(tmp_path, monkeypatch):
    path = tmp_path / "source.txt"
    path.write_bytes(b"before\n")
    real_open = source.os.open

    def append_before_open(target, flags, *args, **kwargs):
        if Path(target) == path:
            with path.open("ab") as output:
                output.write(b"appended\n")
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(source.os, "open", append_before_open)
    content, signature = reads._read_regular_file(path)

    assert content == b"before\nappended\n"
    assert signature == source._signature_for_stat(path.stat())


def test_atomic_regular_replacement_between_stat_and_open_returns_replacement(tmp_path, monkeypatch):
    path = tmp_path / "source.txt"
    replacement = tmp_path / "replacement.txt"
    path.write_bytes(b"original\n")
    replacement.write_bytes(b"replacement\n")
    real_open = source.os.open

    def replace_before_open(target, flags, *args, **kwargs):
        if Path(target) == path:
            os.replace(replacement, path)
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(source.os, "open", replace_before_open)
    content, signature = reads._read_regular_file(path)

    assert content == b"replacement\n"
    assert signature == source._signature_for_stat(path.stat())


@pytest.mark.parametrize("kind", ["fifo", "device", "socket", "directory"])
def test_special_paths_are_refused_before_open(tmp_path, monkeypatch, kind):
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("named pipes require POSIX")
        os.mkfifo(path)
    elif kind == "device":
        if not Path("/dev/null").exists():
            pytest.skip("device unavailable")
        path.symlink_to("/dev/null")
    else:
        if not hasattr(socket, "AF_UNIX"):
            pytest.skip("Unix sockets unavailable")
        server = socket.socket(socket.AF_UNIX)
        try:
            server.bind(str(path))
        except OSError as error:
            server.close()
            pytest.skip(f"socket creation unavailable: {error.errno}")
        server.close()
    monkeypatch.delattr(source.os, "O_NONBLOCK", raising=False)
    monkeypatch.setattr(source.os, "open", lambda *_a, **_k: pytest.fail("opened special path"))
    with pytest.raises(reads._NonRegularReadPath):
        reads._read_regular_file(path)


def test_fdopen_failure_closes_raw_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "source.txt"
    path.write_bytes(b"regular\n")
    real_open = source.os.open
    descriptors = []
    def track_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor
    monkeypatch.setattr(source.os, "open", track_open)
    monkeypatch.setattr(source.os, "fdopen", lambda *_a, **_k: (_ for _ in ()).throw(OSError("fdopen")))
    with pytest.raises(OSError, match="fdopen"):
        reads._read_regular_file(path)
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_binary_cursor_from_text_revision_is_stale_before_binary_refusal(tmp_path):
    path = tmp_path / "source.dat"
    old_text = b"old text\n"
    cursor = "read1:" + hashlib.sha256(old_text).hexdigest() + ":1:0"
    path.write_bytes(b"prefix\0secret\xff")
    ordinary, _ = _read(path)
    assert ordinary[0] == "BINARY" and "secret" not in ordinary[1]
    stale, _ = _read(path, args={"path": path.name, "cursor": cursor})
    assert stale[0] == "STALE_REVISION"
    pinned, _ = _read(path, args={"path": path.name, "expected_revision": "sha256:" + "0" * 64})
    assert pinned[0] == "STALE_REVISION"


def test_plain_read_uses_shared_safe_text_and_binary_rules(tmp_path):
    from kernel_plain_host import PlainHost
    def plain_read(path):
        records = []
        host = object.__new__(PlainHost)
        host.authority = SimpleNamespace(resolve=lambda _raw: path, root=path.parent)
        host.hooks = SimpleNamespace(widen=lambda _p, _t, start, end, **_k: (start, end))
        host.book = SimpleNamespace(reads=0)
        host.journal = SimpleNamespace(receipt=lambda *_a, **fields: records.append(fields))
        return host._read_file({"path": path.name}), records
    path = tmp_path / "source.txt"
    path.write_bytes(b"plain \xff\n")
    text, records = plain_read(path)
    assert "plain \ufffd" in text
    note = "not UTF-8: 1 undecodable bytes are shown as U+FFFD; do not copy them into an edit"
    assert note in text and records[0]["decoding_note"] == note
    from kernel_plain_loop import OUT_CAP
    path.write_bytes(b"\xff" * OUT_CAP)
    long_text, long_metadata = plain_read(path)
    assert "not UTF-8:" not in long_text and long_metadata[0]["decoding_note"].startswith("not UTF-8:")
    path.write_bytes(b"prefix\0secret")
    binary, _ = plain_read(path)
    assert "BINARY_FILE" in binary and "13 bytes" in binary and "secret" not in binary
    folder = tmp_path / "folder"
    folder.mkdir()
    with pytest.raises(IsADirectoryError):
        plain_read(folder)
    link = tmp_path / "folder-link"
    link.symlink_to(folder, target_is_directory=True)
    with pytest.raises(IsADirectoryError):
        plain_read(link)


def test_plain_loop_audit_marks_withheld_reads_refused(monkeypatch):
    import kernel_plain_loop as plain

    class Journal:
        def __init__(self):
            self.events = []

        def write(self, kind, **fields):
            self.events.append((kind, fields))

    def turn(call_id, name, arguments):
        return {"choices": [{"message": {"tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)},
        }]}, "finish_reason": "tool_calls"}]}

    for observation, code in (("BINARY_FILE: x is binary, 3 bytes", "binary_file"),
                              ("REFUSED NON_REGULAR_FILE: x is not a regular file.",
                               "non_regular_file")):
        monkeypatch.setenv("STATEFRAME_TOOL_REPLY_CONSTRUCTOR", "on")
        calls = 0

        def provider(_body, _timeout):
            nonlocal calls
            calls += 1
            return (turn("read", "read_file", {"path": "x"}) if calls == 1
                    else turn("done", "done", {"summary": "finished"}))

        journal = Journal()
        plain.run_loop("task", provider=provider, execute=lambda *_: observation,
                       limits=plain.Limits(model="test", max_requests=2), journal=journal,
                       sleep=lambda _seconds: None)
        [receipt] = [fields["outcome"] for kind, fields in journal.events
                     if kind == "tool_reply_receipt" and fields.get("tool") == "read_file"]
        assert receipt == {"kind": "REFUSED", "code": code}


def test_react_repair_read_dispatch_types_withheld_results():
    import ast

    path = ROOT / "src/stateframe/engine/probes/react_repair.py"
    tree = ast.parse(path.read_text())
    nodes = list(ast.walk(tree))
    assignments = {target.id: node.value for node in nodes if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    withheld = assignments["_withheld_read"]
    assert {item.value for item in ast.walk(withheld)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)} >= {
                "BINARY", "NON_REGULAR_FILE"}
    outcome = assignments["_outcome"]
    assert isinstance(outcome, ast.IfExp) and isinstance(outcome.test, ast.Name)
    assert outcome.test.id == "_withheld_read"
    assert isinstance(outcome.body, ast.Attribute) and outcome.body.attr == "REFUSED"
    assert isinstance(outcome.body.value, ast.Name) and outcome.body.value.id == "_ToolReplyOutcome"
    assert any(isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name)
               and node.func.id == "_tool_reply_for_call"
               and {keyword.arg for keyword in node.keywords} >= {"outcome", "receipt"}
               and any(keyword.arg == "outcome" and isinstance(keyword.value, ast.Name)
                       and keyword.value.id == "_outcome" for keyword in node.keywords)
               and any(keyword.arg == "receipt" and isinstance(keyword.value, ast.Name)
                       and keyword.value.id == "_receipt" for keyword in node.keywords)
               for node in nodes)


def test_react_repair_read_receipts_do_not_leak_across_missing_path(tmp_path, monkeypatch):
    import tempfile
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_verdict_identity import _ACTIVE_PROVIDER
    from stateframe.engine.probes import react_repair
    from stateframe.task_row import build_row
    from stateframe.workcopy import create

    class StopAfterReadSequence(BaseException):
        pass

    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "answer.py").write_text(
        "def answer():\n    return 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_answer.py").write_text(
        "def test_answer():\n    assert False\n", encoding="utf-8")
    row = build_row(create(repo, tmp_path / "copy"), task="Inspect answer.py",
                    target="app/answer.py", task_id="read-result-audit")
    row["files"]["binary.dat"] = {"parent_src": "prefix\0secret"}

    output = tmp_path / "run" / "worker" / "measurement-runs"
    output.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("STATEFRAME_TOOL_REPLY_CONSTRUCTOR", "on")
    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_IDEALPATH", "off")
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    original_init = StateFrameRuntime.__init__

    def scoped_init(self, owner, **kwargs):
        kwargs["output"] = output
        original_init(self, owner, **kwargs)

    monkeypatch.setattr(StateFrameRuntime, "__init__", scoped_init)
    actions = [
        {"tool": "read", "required": True, "arguments": {"path": "binary.dat"}},
        {"tool": "read", "required": True, "arguments": {"path": "missing.dat"}},
        {"tool": "read", "required": True, "arguments": {"path": "binary.dat"}},
        {"tool": "read", "required": True, "arguments": {"path": "app/answer.py"}},
    ]
    monkeypatch.setattr(react_repair, "MAX_STEPS", len(actions) + 2)
    outcomes = []
    original_reply = react_repair._tool_reply_for_call
    sends = 0

    def capture_reply(messages, call_id, observation, **kwargs):
        message = original_reply(messages, call_id, observation, **kwargs)
        marker = getattr(message, "_stateframe_tool_reply_receipt", None)
        if isinstance(marker, dict):
            outcomes.append((call_id, marker["outcome"]))
        return message

    monkeypatch.setattr(react_repair, "_tool_reply_for_call", capture_reply)

    def capture(messages, *args, **kwargs):
        nonlocal sends
        sends += 1
        if sends == len(actions) + 1:
            raise StopAfterReadSequence
        action = actions[sends - 1]
        return "", [{"id": f"read-{sends}", "type": "function", "function": {
            "name": action["tool"], "arguments": json.dumps(action["arguments"]),
        }}]

    monkeypatch.setattr(react_repair, "ask_tools", capture)
    contexts = (ACTIVE_PARENT, ACTIVE_RUNTIME, ACTIVE_REQUEST, _ACTIVE_PROVIDER)
    tokens = [context.set(None) for context in contexts]
    try:
        with pytest.raises(StopAfterReadSequence):
            react_repair.work(row)
    finally:
        for context, token in zip(contexts, tokens):
            context.reset(token)

    assert [outcome for _call, outcome in outcomes] == [
        {"kind": "REFUSED", "code": "binary_file"},
        {"kind": "ACCEPTED"},
        {"kind": "REFUSED", "code": "binary_file"},
        {"kind": "ACCEPTED"},
    ]
    import ast

    tree = ast.parse((ROOT / "src/stateframe/engine/probes/react_repair.py").read_text())
    step_loop = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Name) and node.target.id == "step"
                     and isinstance(node.iter, ast.Call)
                     and isinstance(node.iter.func, ast.Name)
                     and node.iter.func.id == "step_indices")
    first_statement = step_loop.body[0]
    assert (isinstance(first_statement, ast.Assign)
            and len(first_statement.targets) == 1
            and isinstance(first_statement.targets[0], ast.Name)
            and first_statement.targets[0].id == "_read_result"
            and isinstance(first_statement.value, ast.Constant)
            and first_statement.value.value is None)


def test_read_uses_the_opened_regular_descriptor_after_path_replacement(tmp_path, monkeypatch):
    path = tmp_path / "source.txt"
    path.write_text("opened regular file\n", encoding="utf-8")
    actual_open = source.os.open

    def replace_after_open(target, flags, *args, **kwargs):
        descriptor = actual_open(target, flags, *args, **kwargs)
        if Path(target) == path:
            path.rename(tmp_path / "opened-source.txt")
            path.mkdir()
        return descriptor

    monkeypatch.setattr(source.os, "open", replace_after_open)
    result, _ = _read(path)

    assert result[0] == "OK"
    assert "opened regular file\n" in result[1]


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets unavailable")
def test_unix_socket_path_is_refused_when_host_allows_creation(tmp_path):
    path = tmp_path / "socket"
    server = socket.socket(socket.AF_UNIX)
    try:
        try:
            server.bind(str(path))
        except OSError as error:
            pytest.skip(f"sandbox does not allow Unix socket creation: {error.errno}")
        result, metadata = _read(path)
        assert result[0] == "NON_REGULAR_FILE"
        assert metadata["refusal_cause"] == "non_regular_file"
    finally:
        server.close()


def test_paged_reads_use_checked_disk_loads_and_exact_cache_paths(tmp_path, monkeypatch):
    raw = b"first\nsecond\nthird\n"
    path = tmp_path / "source.txt"
    path.write_bytes(raw)
    runtime = SimpleNamespace(journal=None, retained=None, step_ledger=None)
    actual = reads._read_regular_file
    loads: list[Path] = []

    def count_load(target):
        loads.append(target)
        return actual(target)

    monkeypatch.setattr(reads, "_read_regular_file", count_load)
    first, _ = _read(path, args={"path": path.name, "max_lines": 1}, measurement=runtime)
    assert first[0] == "OK" and len(loads) == 1
    cursor = re.search(r"cursor=(read1:[0-9a-f]{64}:\d+:\d+)", first[1]).group(1)
    second, _ = _read(path, args={"path": path.name, "cursor": cursor}, measurement=runtime)
    assert second[0] == "OK" and "second\n" in second[1]
    assert len(loads) == 1

    cache = next(iter(runtime._read_contract_source_cache.values()))
    cache.content = None
    third, _ = _read(path, args={"path": path.name, "line": 3, "max_lines": 1},
                     measurement=runtime)
    assert third[0] == "OK" and "third\n" in third[1]
    assert len(loads) == 2

    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    pinned, _ = _read(path, args={"path": path.name, "expected_revision": revision},
                      measurement=runtime)
    assert pinned[0] == "OK" and len(loads) == 3
