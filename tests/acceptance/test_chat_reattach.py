"""Session continuity at the chat CLI, with independent processes and a socketless model."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, closing, contextmanager, suppress
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
from threading import Event
import time
from unittest.mock import patch

import pytest

from _timeouts import scaled
from fixtures.process_signals import signal_run_owned
from tools.pinned_tree import pinned_tree
from ._support import PYTHON, ROOT, copy_repo, environment


FIRST = "Read app.py and report the first value."
SECOND = "Read app.py and report the second value."
QUEUED = "Report the queued value."


def _capture_first_wire(run, destination):
    """Enter the production child with its saved spec and intercept the first provider send."""
    from stateframe import _child
    install_environment = _child.install_environment

    def capture(wire, endpoint, _key, **kwargs):
        from kernel_step_contract import system_prompt_variant
        assert endpoint == "http://127.0.0.1:1/v1/chat/completions"
        destination.with_suffix(".prompt").write_text(system_prompt_variant())
        destination.write_bytes(wire)
        os._exit(0)  # Dedicated capture process. No model reply or subsequent steps.

    def install_offline_environment(*args, **kwargs):
        install_environment(*args, **kwargs)
        # Install the fake transport before tool schema measurement starts its tokenizer thread.
        import kernel_bounded_http
        import kernel_request_diet

        kernel_bounded_http.request_bytes = capture
        kernel_request_diet._http_tokenize = lambda content, *_a, **_kw: range(len(content) // 4)

    with patch.object(_child, "install_environment", install_offline_environment):
        status = _child.main(["_child.py", str(run)])
    raise RuntimeError(f"worker exited before first model request: {status}")


def _offline_execute(options, _started, _printer, run, _work, spec):
    """Stand in for task execution only. Chat owns admission, history, and all session rows."""
    from stateframe import runner

    if "WIRE_CAPTURE" in os.environ:
        env = runner._child_env(options)
        env.update(PYTHONPATH=os.environ["PYTHONPATH"], PYTHONHASHSEED=spec["environment"]["PYTHONHASHSEED"])
        subprocess.run([PYTHON, "-m", __spec__.name, "capture", str(run), os.environ["WIRE_CAPTURE"]],
                       env=env, cwd=run, check=True)
    # Follow-up prompts can also carry typed session facts after the user's text.
    objective = options.task.splitlines()[0]
    with Path("model-calls.jsonl").open("a") as stream:
        print(json.dumps(objective), file=stream)
    if objective == os.environ.get("BLOCK_AT_TASK"):
        Path("model-waiting").touch()
        while not Path("release-model").exists():
            time.sleep(0.02)
    check = {"passed": True, "command": "offline", "exit_code": 0}
    summary = {
        "run_dir": str(run), "runtime_outcome": "completed", "check_outcome": "pass",
        "artifact": "none", "engine_result": "SUBMITTED", "child_exit_code": 0,
        "engine_terminal": {"result": "SUBMITTED"}, "exit_code": 0,
        "tests_before": check, "tests_after": check, "files_changed": [],
    }
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "final.diff").write_bytes(b"")
    return summary


def _offline_cli():
    def deny_network(event, args):
        if event == "socket.connect":
            raise SystemExit(f"unexpected network access in offline chat: {args[1]}")

    sys.addaudithook(deny_network)
    if sys.argv[1] == "capture":
        return _capture_first_wire(Path(sys.argv[2]), Path(sys.argv[3]))
    from stateframe import cli, runner
    from stateframe.engine import kernel_provider_dialect as dialect

    def provider_probe(origin, key, path, **kwargs):
        return {"/props": (200, {"default_generation_settings": {"n_ctx": 65536}}),
                "/slots": (404, {})}[path]

    args = sys.argv[1:]
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner.Endpoint, "probe", return_value={"model_found": True}))
        stack.enter_context(patch.object(dialect, "bounded_json_probe", provider_probe))
        stack.enter_context(patch.object(runner, "_execute", _offline_execute))
        if "PAUSE_AT_ACK" in os.environ:
            _pause_on_ack(stack, os.environ["PAUSE_AT_ACK"])
        return cli.main(args)


def _pause_on_ack(stack, task):
    """Hold at the first visible output after the REPL reads this input."""
    armed = False
    readline = sys.stdin.readline

    def read(*args):
        nonlocal armed
        line = readline(*args)
        armed = line.rstrip("\r\n") == task
        return line

    def observe(stream):
        # Give piped output the terminal's line buffering. print() may write the
        # text and newline separately, or explicitly flush a prompt with no newline.
        stream.reconfigure(line_buffering=True)
        write, flush = stream.write, stream.flush
        pending = False

        def flush_and_pause():
            flush()
            if armed and pending:
                Path("ack-waiting").touch()
                Event().wait(scaled(30))
                raise RuntimeError("test did not interrupt the paused chat")

        def write_and_check(text):
            nonlocal pending
            size = write(text)
            if armed:
                pending = pending or bool(text.strip())
                if "\n" in text:
                    flush_and_pause()
            return size

        stack.enter_context(patch.object(stream, "write", write_and_check))
        stack.enter_context(patch.object(stream, "flush", flush_and_pause))

    stack.enter_context(patch.object(sys.stdin, "readline", read))
    observe(sys.stdout)
    observe(sys.stderr)


class Chat:
    def __init__(self, root):
        self.root = root
        self.repo = copy_repo(root)
        self.home = root / "home" / ".stateframe"
        self.session = self.home / "sessions" / "reattach-example"
        self.db = self.session / "session.sqlite3"

    @contextmanager
    def process(self, command="chat", flags=(), *, attach=True, **settings):
        env = environment()
        child_env = {k: env[k] for k in ("PATH", "LANG", "PYTHONPATH") if k in env}
        child_env.update(HOME=str(self.home.parent), STATEFRAME_HOME=str(self.home),
                         TMPDIR=str(self.root), PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                         STATEFRAME_N_CTX="65536", **settings)
        child_env["PYTHONPATH"] += os.pathsep + str(ROOT / "tests")
        args = [PYTHON, "-m", "acceptance.test_chat_reattach", command, "--repo", str(self.repo),
                "--endpoint", "http://127.0.0.1:1/v1", "--model", "scripted", "--target", "app.py"]
        if "WIRE_CAPTURE" not in settings:
            args.append("--no-run-env")  # The simulated task executor needs no project interpreter.
        if attach:
            args.extend(["--session-dir", str(self.session)])
        with subprocess.Popen([*args, *flags], cwd=self.root, env=child_env, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              start_new_session=True) as child:
            try:
                yield child
            finally:
                if child.poll() is None:
                    with suppress(ProcessLookupError):
                        self.interrupt(child, "chat test cleanup")
                child.wait(timeout=scaled(5))

    def interrupt(self, child, what):
        # The offline chat uses a simulated executor and holds at a session barrier.
        signal_run_owned(self.session, child, signal.SIGKILL, deadline=time.monotonic() + scaled(5),
                         what=what, require_checkpoint_saved=False, group=True)

    def run(self, *lines, **kwargs):
        with self.process(**kwargs) as child:
            timeout = scaled(45 if "WIRE_CAPTURE" in kwargs else 25)
            output, _ = child.communicate("\n".join((*lines, "")), timeout=timeout)
            return child.returncode, output

    def ok(self, *lines, **kwargs):
        status, output = self.run(*lines, **kwargs)
        if status != 0 or "Traceback (most recent call last)" in output:
            pytest.fail(output)
        return output

    def calls(self):
        path = self.root / "model-calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def chat(tmp_path):
    return Chat(tmp_path)


class ReattachUnavailable(AssertionError):
    """Only an observed legacy behavior can consume a strict xfail."""


def _check_for_legacy_refusal(output):
    if "stateframe: error: working copy destination already exists:" in output:
        raise ReattachUnavailable(output)


def _row_counts(db, marker=None):
    """Count all user-table rows, or just rows containing the given marker."""
    if not db.is_file():
        raise ReattachUnavailable(f"no durable session database at {db}")
    counts = Counter()
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        tables = connection.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT GLOB 'sqlite_*'")
        for (name,) in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            if marker is None:
                counts[name] = connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
                continue
            for row in connection.execute(f"SELECT * FROM {quoted}"):
                if any((isinstance(value, str) and marker in value) or
                       (isinstance(value, bytes) and marker.encode() in value) for value in row):
                    counts[name] += 1
    return counts


def _history(output, task):
    # Piped input is not echoed. Match only history entries, excluding prompts and summaries.
    return re.findall(r"(?m)^\s*(?:stateframe> )?\d+\. \[([^\]\n]+)\] " + re.escape(task) + "$",
                      output)


def _reattached(chat, *lines):
    status, output = chat.run(*lines, "/quit")
    _check_for_legacy_refusal(output)
    assert status == 0, output
    assert "Traceback (most recent call last)" not in output
    return output


def _not_shipped(section):
    return pytest.mark.xfail(strict=True, raises=ReattachUnavailable,
                             reason=f"DESIGN25 §{section}: not shipped in 0.10.2")


def test_first_turn_history_in_live_session(chat):
    output = chat.ok(FIRST, "/history", "/quit")
    states = _history(output, FIRST)
    assert len(states) == 1 and "runtime=completed" in states[0], output
    assert chat.calls() == [FIRST]


@_not_shipped("per-session SQLite authority")
def test_clean_exit_reopen_and_append(chat):
    chat.ok(FIRST, "/quit")
    assert chat.calls() == [FIRST]
    original = _row_counts(chat.db, FIRST)
    assert original
    one_turn = _row_counts(chat.db)
    before = _reattached(chat, "/history")
    assert len(_history(before, FIRST)) == 1, before
    assert _row_counts(chat.db, FIRST) == original
    assert _row_counts(chat.db) == one_turn
    assert chat.calls() == [FIRST]

    after = _reattached(chat, SECOND, "/history")
    assert len(_history(after, FIRST)) == len(_history(after, SECOND)) == 1, after
    assert after.rindex(FIRST) < after.rindex(SECOND)
    assert _row_counts(chat.db, FIRST) == original
    appended = _row_counts(chat.db, SECOND)
    assert appended
    two_turns = _row_counts(chat.db)
    again = _reattached(chat, "/history")
    assert _row_counts(chat.db) == two_turns
    assert _history(again, FIRST) == _history(after, FIRST)
    assert _history(again, SECOND) == _history(after, SECOND)
    assert _row_counts(chat.db, FIRST) == original and _row_counts(chat.db, SECOND) == appended
    assert chat.calls() == [FIRST, SECOND]

    # An uninterrupted two-turn conversation is the control for how many rows a
    # new turn adds. An implementation may legitimately store several row types.
    control = chat.session.with_name("uninterrupted")
    chat.ok(FIRST, SECOND, "/quit", attach=False, flags=("--session-dir", str(control)))
    assert _row_counts(control / "session.sqlite3") == two_turns
    assert _row_counts(control / "session.sqlite3", FIRST) == original
    assert _row_counts(control / "session.sqlite3", SECOND) == appended


@contextmanager
def _in_flight(chat, *lines, boundary="model"):
    if boundary == "ack":
        marker = "ack-waiting"
        settings = {"PAUSE_AT_ACK": SECOND}
    else:
        marker = "model-waiting"
        settings = {"BLOCK_AT_TASK": SECOND}
    with chat.process(**settings) as writer:
        writer.stdin.write("\n".join((*lines, "")))
        writer.stdin.flush()
        deadline = time.monotonic() + scaled(15)
        while not (chat.root / marker).exists():
            if writer.poll() is not None:
                pytest.fail(writer.stdout.read())
            if time.monotonic() > deadline:
                pytest.fail(f"writer never reached the {boundary} barrier")
            time.sleep(0.02)
        yield writer


@_not_shipped("durable input before acknowledgement")
@pytest.mark.parametrize("boundary", ["ack", "model"])
def test_kill_after_input_admission_before_reply(chat, boundary):
    # For the acknowledgement case, queue the next line before the writer pauses.
    queued = (QUEUED,) if boundary == "ack" else ()
    with _in_flight(chat, FIRST, "/history", SECOND, *queued, boundary=boundary) as writer:
        if boundary == "model":
            if chat.db.is_file():
                assert _row_counts(chat.db, SECOND), "input was not visible during the model call"
            writer.stdin.write(QUEUED + "\n")
            writer.stdin.flush()
        chat.interrupt(writer, f"chat interruption at {boundary}")
        transcript, _ = writer.communicate(timeout=scaled(5))
        assert writer.returncode == -signal.SIGKILL, transcript
    expected_calls = [FIRST] if boundary == "ack" else [FIRST, SECOND]
    assert chat.calls() == expected_calls
    acknowledged = _history(transcript, FIRST)
    assert len(acknowledged) == 1 and "runtime=completed" in acknowledged[0], transcript
    admitted = _row_counts(chat.db, SECOND) if chat.db.is_file() else {}
    if chat.db.is_file():
        assert admitted, f"input was not committed at the {boundary} boundary"
    output = _reattached(chat, "/history")
    assert admitted and _row_counts(chat.db, SECOND)
    assert _history(output, FIRST) == acknowledged
    pending = _history(output, SECOND)
    assert len(pending) == 1, output
    for state in [*pending, *_history(output, QUEUED)]:
        assert not re.search(r"\b(completed|submitted|done|succeeded)\b", state, re.I), state
    assert chat.calls() == expected_calls


@_not_shipped("per-session SQLite authority")
def test_two_attaches_share_a_consistent_session(chat):
    with _in_flight(chat, FIRST, SECOND, "/quit") as writer:
        before = _row_counts(chat.db, FIRST) if chat.db.is_file() else {}
        status, output = chat.run("/history", "/quit")
        assert writer.poll() is None
        assert "Traceback (most recent call last)" not in output
        _check_for_legacy_refusal(output)
        if status:
            assert re.search(r"\b(?:SESSION|CHAT)_[A-Z_]*(?:BUSY|LOCKED|IN_USE|ATTACHED)\b", output, re.I), output
        else:
            assert len(_history(output, FIRST)) == 1, output
            assert "runtime=completed" in _history(output, FIRST)[0]
            pending = _history(output, SECOND)
            assert len(pending) <= 1 and all("runtime=completed" not in state for state in pending)
        assert before and _row_counts(chat.db, FIRST) == before
        (chat.root / "release-model").touch()
        output, _ = writer.communicate(timeout=scaled(15))
        assert writer.returncode == 0, output
    output = _reattached(chat, "/history")
    for task in (FIRST, SECOND):
        states = _history(output, task)
        assert len(states) == 1 and "runtime=completed" in states[0], output
    assert _row_counts(chat.db, FIRST) == before
    assert chat.calls() == [FIRST, SECOND]


@pytest.fixture(params=["missing", "corrupt"])
def damaged_session(chat, request):
    chat.ok(FIRST, "/quit")
    if request.param == "missing":
        chat.db.unlink(missing_ok=True)
    else:
        chat.db.write_bytes(b"damaged SQLite header")
    return request.param


def test_unreadable_authority_never_replaces_session(chat, damaged_session):
    status, output = chat.run("/history", "/quit")
    assert status != 0 and chat.session.name in output, output
    assert "Traceback (most recent call last)" not in output
    assert list(chat.session.parent.iterdir()) == [chat.session]
    if damaged_session == "missing":
        assert not chat.db.exists()
    else:
        assert chat.db.read_bytes() == b"damaged SQLite header"
    assert chat.calls() == [FIRST]


@_not_shipped("per-session SQLite authority")
def test_unreadable_authority_gives_typed_error(chat, damaged_session):
    status, output = chat.run("/history", "/quit")
    _check_for_legacy_refusal(output)
    assert status != 0 and chat.session.name in output, output
    assert re.search(r"\b(?:SESSION|CHAT)_[A-Z_]*(?:MISSING|NOT_FOUND|CORRUPT|UNREADABLE|INVALID)\b",
                     output, re.I), output


@pytest.fixture
def older_session(chat, tmp_path_factory, monkeypatch):
    # Materialize the v0.10.1 tag from local git objects. A missing writer must fail the test.
    monkeypatch.setenv("PINNED_TREE_REQUIRED", "1")
    source = pinned_tree("8c0a28c9a80d1b4745bcc9b311e5d921adad1734", tmp_path_factory) / "src"
    engine = source / "stateframe" / "engine"
    history = chat.ok(FIRST, "/history", "/quit", PYTHONPATH=os.pathsep.join(map(str, (source, engine))))
    states = _history(history, FIRST)
    assert len(states) == 1 and "runtime=completed" in states[0], history
    assert chat.calls() == [FIRST]

    # The old CLI kept /history in memory, but wrote the task text to its run spec.
    spec = json.loads((chat.session / "task-01" / "spec.json").read_text())
    assert Path(spec["engine_dir"]) == engine
    assert spec["public_row"]["msg"] == FIRST
    assert not chat.db.exists()
    return chat


@_not_shipped("per-session SQLite authority (RNR25 cross-release adoption)")
def test_older_writer_session_opens_without_replaying_model_call(older_session):
    chat = older_session
    status, output = chat.run("/history", "/quit")
    assert chat.calls() == [FIRST]
    assert "Traceback (most recent call last)" not in output, output
    if status:
        _check_for_legacy_refusal(output)
    assert status == 0, output
    drop = re.search(
        r"\b(?:[A-Z_]+_DROP(?:PED)?|DROP(?:PED)?_[A-Z_]+)\b", output, re.I)
    assert len(_history(output, FIRST)) == 1 or drop, output
    appended = chat.ok(SECOND, "/history", "/quit")
    assert len(_history(appended, SECOND)) == 1, appended
    assert chat.calls() == [FIRST, SECOND]


def test_no_session_flag_preserves_first_request_bytes(chat):
    task = "Read app.py and report the café value."
    plain_wire, chat_wire = chat.root / "plain.request", chat.root / "chat.request"
    chat.ok(command="run", attach=False, flags=("--task", task, "--run-dir", str(chat.root / "run")),
            WIRE_CAPTURE=str(plain_wire))
    chat.ok(task, "/quit", attach=False, WIRE_CAPTURE=str(chat_wire))
    assert plain_wire.with_suffix(".prompt").read_text() == "v3"
    assert chat_wire.with_suffix(".prompt").read_bytes() == plain_wire.with_suffix(".prompt").read_bytes()
    expected = plain_wire.read_bytes()
    assert any(task in message["content"] for message in json.loads(expected)["messages"])
    assert chat_wire.read_bytes() == expected


if __name__ == "__main__":
    raise SystemExit(_offline_cli())
