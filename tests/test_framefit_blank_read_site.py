"""Clipped blank lines are recoverable only from unchanged read evidence."""

import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

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
        context = build_action_context(messages, workspace_root=tmp_path)
        assert context["site_context"] == {
            "status": "UNVERIFIABLE", "reason": "missing_exact_current_site",
        }
        assert context["open_edit_site"] is None
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


def _messages_for_read(root: Path, source: bytes, arguments: dict[str, object]):
    from kernel_source_delivery import deliver_source

    path = str(arguments["path"])
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
    page = deliver_source(source, path, 0, args=arguments)
    call_id = "read-1"
    return [
        {"role": "user", "content": "Read the requested source range."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": "read", "arguments": arguments,
            },
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": page.text},
    ], target


@pytest.mark.parametrize(
    ("source", "start", "end", "expected"),
    [
        (b"\nhead\nmore\n", 1, 1, "\n"),
        (b"head\n\n", 2, 2, "\n"),
        (b"\n\n", 1, 2, "\n\n"),
        (b"head\r\n\r\nend\r\n", 2, 2, "\r\n"),
    ],
    ids=("blank-at-start", "blank-at-end", "whole-blank-file", "crlf"),
)
def test_blank_ranges_and_crlf_remain_exact_when_the_site_is_readable(
    tmp_path, source, start, end, expected,
):
    messages, _target = _messages_for_read(
        tmp_path, source, {"path": "pkg/window.txt", "start_line": start, "end_line": end},
    )
    context = build_action_context(messages, workspace_root=tmp_path)
    assert context["open_edit_site"]["current_bytes"] == expected
    assert "site_context" not in context


@pytest.mark.parametrize("mutation", ["deleted", "changed", "binary"])
def test_unreadable_current_site_degrades_to_existing_typed_context(tmp_path, mutation):
    messages, target = _messages_for_read(
        tmp_path, b"head\n\nend\n",
        {"path": "pkg/window.txt", "start_line": 2, "max_lines": 1},
    )
    if mutation == "deleted":
        target.unlink()
    elif mutation == "changed":
        target.write_bytes(b"head changed\n\nend\n")
    else:
        target.write_bytes(b"\xff\x00\xfe")

    context = build_action_context(messages, workspace_root=tmp_path)
    assert context["open_edit_site"] is None
    assert context["site_context"] == {
        "status": "UNVERIFIABLE", "reason": "missing_exact_current_site",
    }


def test_unexpected_site_read_error_still_raises(tmp_path, monkeypatch):
    import kernel_stateframe_action_ledger as ledger

    messages, _target = _messages_for_read(
        tmp_path, b"head\n\nend\n",
        {"path": "pkg/window.txt", "start_line": 2, "max_lines": 1},
    )

    def mutated_read(*_args, **_kwargs):
        raise RuntimeError("unexpected mutation")

    monkeypatch.setattr(ledger, "_read_current_open_site", mutated_read)
    with pytest.raises(RuntimeError, match="unexpected mutation"):
        build_action_context(messages, workspace_root=tmp_path)


def test_random_read_shapes_and_ranges_do_not_escape_site_errors(tmp_path):
    rng = random.Random(20261123)
    separators = ("\n", "\r\n", "\u2028", "\u2029")
    for index in range(48):
        rows = [rng.choice(("", "x", "value", "z" * rng.randrange(1, 96)))
                for _ in range(rng.randrange(1, 9))]
        source = rng.choice(separators).join(rows) + rng.choice(("", "\n", "\r\n"))
        if not source:
            source = "\n"
        source_bytes = source.encode("utf-8")
        start = rng.randint(1, len(source.splitlines(keepends=True)))
        arguments = {"path": "pkg/window.txt", "start_line": start,
                     "max_lines": rng.randrange(1, 5)}
        messages, target = _messages_for_read(tmp_path, source_bytes, arguments)
        mutation = rng.choice(("same", "same", "changed", "deleted", "binary"))
        if mutation == "changed":
            target.write_bytes(source_bytes + b"changed\n")
        elif mutation == "deleted":
            target.unlink()
        elif mutation == "binary":
            target.write_bytes(b"\xff\x00")
        try:
            build_action_context(messages, workspace_root=tmp_path)
        except Exception as error:
            pytest.fail(f"random case {index} ({mutation}) raised {type(error).__name__}: {error}")


CHECKOUT = Path(__file__).resolve().parents[1]
BASE = "7540148c9c7b58cd76dfec3b004dc3763451a3d3"


def _isolate_profile_environment(monkeypatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.startswith(("STATEFRAME_", "REACT_")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "state-home"))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")


def _scripted_read_repo(tmp_path: Path, shape: str):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    if shape == "unicode_separator":
        source = "prefix\u2028continued\n\n" + "".join(f"line{i}\n" for i in range(120))
        arguments = {"path": "pkg/window.txt", "start_line": 3, "max_lines": 1}
    elif shape == "long_line":
        source = "\n" + "x" * 70_000 + "\n" + "".join(f"line{i}\n" for i in range(120))
        arguments = {"path": "pkg/window.txt", "start_line": 1}
    else:
        source = "first line\n\n\nlast line\n"
        arguments = {"path": "pkg/window.txt", "start_line": 2, "max_lines": 1}
    (repo / "pkg/window.txt").write_text(source, encoding="utf-8")
    return repo, {"tool": "read", "arguments": arguments}


def _scripted_open_site_run(tmp_path, monkeypatch, shape, *, checkout=CHECKOUT, context_mode=None):
    from _timeouts import scaled
    from stateframe.scripted_endpoint import Script, serve

    repo, read = _scripted_read_repo(tmp_path, shape)
    profile_args = []
    if context_mode is not None:
        profile_data = json.loads((checkout / "src/stateframe/profiles/r2hc3.json").read_text())
        profile_data["switches"]["STATEFRAME_FRAMEFIT5_ACTION_CONTEXT"] = context_mode
        profile = tmp_path / "profile.json"
        profile.write_text(json.dumps(profile_data))
        profile_args = ["--profile", str(profile)]

    script = Script({"steps": [
        read,
        {"tool": "done", "arguments": {"summary": "Read completed."}},
    ]})
    endpoint = serve(script)
    run_dir = tmp_path / "run"
    _isolate_profile_environment(monkeypatch, tmp_path)
    try:
        result = subprocess.run([
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Inspect app.py and report VALUE.",
            "--target", "app.py", "--endpoint", f"http://127.0.0.1:{endpoint.server_port}/v1",
            "--model", script.model, "--run-dir", str(run_dir), *profile_args,
            "--max-turns", "2", "--runner-kind", "generic", "--test-cmd", "true",
            "--quiet", "--json",
        ], env={**os.environ, "PYTHONPATH": str(checkout / "src")}, cwd=tmp_path,
            capture_output=True, text=True, timeout=scaled(180))
    finally:
        endpoint.shutdown()
        endpoint.server_close()
    assert (run_dir / "summary.json").is_file(), result.stdout + result.stderr
    return json.loads((run_dir / "summary.json").read_text()), script, run_dir


def _run_events(run_dir: Path) -> list[dict[str, object]]:
    from tests._journal import measurement_events

    locations = [run_dir, *sorted((run_dir / "attempts").glob("*"))]
    return [
        event
        for location in locations
        if (location / "worker/measurement-runs").is_dir()
        for event in measurement_events(location)
    ]


def _typed_frame(request):
    (frame,) = [
        json.loads(message["content"].split("\n", 1)[1])["state"]["framefit5"]
        for message in request["messages"]
        if (message.get("content") or "").startswith("[StateFrame]\n")
    ]
    return frame


def _assert_unverifiable_site(request):
    frame = _typed_frame(request)
    assert frame["open_sites"] == []
    assert frame["open_work"]["site_context_status"] == "UNVERIFIABLE"
    assert frame["open_work"]["site_context_reason"] == "latest_file_action_has_no_exact_current_site"


def _assert_once_before_next_send(run_dir):
    events = _run_events(run_dir)
    first, second = [i for i, event in enumerate(events) if event["kind"] == "request_submitted"]
    degraded_kind = "framefit5_action_context_degraded"
    assert all(event["kind"] != degraded_kind for event in events[:first])
    (degraded,) = [event for event in events[first:second] if event["kind"] == degraded_kind]
    assert degraded["status"] == "UNVERIFIABLE"
    assert degraded["reason"] == "missing_exact_current_site"


@pytest.mark.parametrize("shape", ["unicode_separator", "long_line"])
def test_shipped_profile_builds_next_request_after_blank_read(tmp_path, monkeypatch, shape):
    summary, script, run_dir = _scripted_open_site_run(tmp_path, monkeypatch, shape)
    assert summary["engine_result"] != "ENGINE_ERROR", summary.get("engine_terminal")
    assert len(script.requests) == 2
    _assert_unverifiable_site(script.requests[1])
    _assert_once_before_next_send(run_dir)
    script.assert_consumed()


@pytest.mark.parametrize("context_mode", ["off", "on"])
def test_valid_site_request_bytes_match_base(tmp_path, tmp_path_factory, monkeypatch, context_mode):
    from tools.pinned_tree import pinned_tree

    wires = []
    for name, checkout in (("base", pinned_tree(BASE, tmp_path_factory)), ("current", CHECKOUT)):
        case = tmp_path / name
        case.mkdir()
        summary, script, run_dir = _scripted_open_site_run(
            case, monkeypatch, "ordinary", checkout=checkout, context_mode=context_mode,
        )
        assert summary["engine_result"] != "ENGINE_ERROR", summary.get("engine_terminal")
        assert len(script.requests) == 2
        assert _typed_frame(script.requests[1])["open_sites"][0]["path"] == "pkg/window.txt"
        script.assert_consumed()
        wires.append([(run_dir / "wire" / f"{send:04}.request.json").read_bytes() for send in (1, 2)])
    base_wire, current_wire = wires
    assert current_wire == base_wire


def test_resume_from_read_journal_builds_the_next_request(tmp_path, monkeypatch):
    from _timeouts import scaled
    from engine_run import start_monitored
    from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
    from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve

    waiting = threading.Event()
    release = threading.Event()

    class InterruptedAfterRead(Script):
        def next_call(self, offered):
            if self.position == 1 and not release.is_set():
                waiting.set()
                release.wait(scaled(120))
                raise ScriptStepUnavailable("interrupted after delivering the read")
            return super().next_call(offered)

    repo, read = _scripted_read_repo(tmp_path, "unicode_separator")
    script = InterruptedAfterRead({"steps": [
        read,
        {"tool": "done", "arguments": {"summary": "Read completed."}},
    ]})
    endpoint = serve(script)
    _isolate_profile_environment(monkeypatch, tmp_path)
    env = {**os.environ, "PYTHONPATH": str(CHECKOUT / "src")}
    run_dir = tmp_path / "run"
    url = f"http://127.0.0.1:{endpoint.server_port}/v1"
    process = None
    try:
        launched_after = time.time()
        process = start_monitored([
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Inspect app.py and report VALUE.",
            "--target", "app.py", "--endpoint", url, "--model", script.model,
            "--run-dir", str(run_dir), "--max-turns", "2", "--keep-checkpoints",
            "--runner-kind", "generic", "--test-cmd", "true", "--quiet", "--json",
        ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, run_dir=run_dir,
            expected_seconds=scaled(180), step="interrupt after blank read")
        deadline = time.monotonic() + scaled(120)
        while not waiting.wait(0.25):
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "post-read request did not reach the endpoint"
        assert [path.name for path in sorted((run_dir / "wire").glob("*.response.json"))] == [
            "0001.response.json",
        ]
        pending = (run_dir / "wire/0002.request.json").read_bytes()
        (engine_pid,) = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
        engine_pid = int(engine_pid)
        identity = record_identity(engine_pid, launched_after=launched_after, command_fragment="_child.py")
        signal_run_verified(run_dir, engine_pid, identity, signal.SIGKILL,
                            deadline=deadline, what="post-read interruption")
        release.set()
        process.communicate(timeout=scaled(60))
        resumed = subprocess.run([
            sys.executable, "-m", "stateframe", "resume", str(run_dir), "--endpoint", url,
            "--model", script.model, "--quiet", "--json",
        ], env=env, capture_output=True, text=True, timeout=scaled(180))
        assert (run_dir / "wire/0002.response.json").is_file(), resumed.stdout + resumed.stderr
    finally:
        release.set()
        if process is not None and process.poll() is None:
            signal_run_owned(run_dir, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="post-read test cleanup", require_checkpoint_saved=False, group=True)
            process.communicate(timeout=scaled(10))
        endpoint.shutdown()
        endpoint.server_close()
    assert len(script.requests) == 3
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["engine_result"] != "ENGINE_ERROR", summary
    assert summary["resume"]["recorded_exchanges"] == 1
    engine_events = [json.loads(line.removeprefix("[sf] "))
                     for line in (run_dir / "engine.log").read_text().splitlines()
                     if line.startswith("[sf] ")]
    assert [event["send"] for event in engine_events if event["kind"] == "checkpoint_verified"] == [2]
    assert (run_dir / "wire/0002.request.json").read_bytes() == pending
    _assert_unverifiable_site(script.requests[-1])
    script.assert_consumed()


def test_chat_task_continues_after_an_unreadable_site(tmp_path, monkeypatch):
    from stateframe.chat import Session
    from stateframe.endpoint import Endpoint
    from stateframe.runner import RunOptions
    from stateframe.scripted_endpoint import Script, serve

    repo, read = _scripted_read_repo(tmp_path, "unicode_separator")
    script = Script({"steps": [
        read,
        {"tool": "done", "arguments": {"summary": "Read completed."}},
    ]})
    endpoint = serve(script)
    _isolate_profile_environment(monkeypatch, tmp_path)
    try:
        url = f"http://127.0.0.1:{endpoint.server_port}/v1"
        options = RunOptions(
            repo=repo, task="", endpoint=Endpoint.from_args(url, script.model, None),
            max_turns=2, test_cmd="true", runner_kind="generic", target="app.py",
            python=sys.executable, no_project_hooks=True, quiet=True,
        )
        session = Session(repo=repo, home=tmp_path / "session", base=options)
        session.start(printer=lambda *_args: None)
        record = session.run_task("Inspect app.py and report VALUE.", printer=lambda *_args: None)
    finally:
        endpoint.shutdown()
        endpoint.server_close()
    assert len(script.requests) == 2
    _assert_unverifiable_site(script.requests[1])
    _assert_once_before_next_send(Path(record.run_dir))
    script.assert_consumed()
