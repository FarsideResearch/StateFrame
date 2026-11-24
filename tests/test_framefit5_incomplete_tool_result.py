"""Incomplete transcript pairs still receive typed settled action state."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from _timeouts import scaled
from kernel_stateframe_action_ledger import build_action_context
from kernel_stateframe_framefit5 import FrameFit5Error, _opaque_actions, build_typed_state
from kernel_native_tools import NativeToolBatch
from _stateframe_home import isolated_stateframe_home
from fixtures.process_signals import signal_run_owned
import stateframe.scripted_endpoint as scripted_endpoint
from stateframe.scripted_endpoint import Script, serve


class _Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append({"kind": kind, **fields})
        return self.events[-1]


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.mark.parametrize(
    "path",
    [
        "missing/synthetic.py",
        "task-input:" + hashlib.sha256(b"synthetic task").hexdigest(),
    ],
    ids=["missing-path", "held-input-handle"],
)
def test_incomplete_final_call_is_typed_after_many_task_pages(path):
    messages = [
        {"role": "system", "content": "Synthetic request."},
        {"role": "user", "content": "Synthetic task."},
    ]
    for index in range(100):
        call_id = f"page-{index}"
        messages.extend((
            {"role": "assistant", "content": None, "tool_calls": [
                _call(call_id, "read_check_receipt", {"handle": "synthetic", "offset": index}),
            ]},
            {"role": "tool", "tool_call_id": call_id, "content": "page settled"},
        ))
    missing_id = "grep-final"
    messages.append({"role": "assistant", "content": None, "tool_calls": [
        _call(missing_id, "grep", {
            "pattern": "synthetic literal",
            "path": path,
            "mode": "literal",
            "page_size": 50,
        }),
    ]})

    journal = _Journal()
    context = build_action_context(messages, journal=journal)
    assert any(
        event.get("kind") == "action_group_dropped"
        and event.get("reason") == "incomplete_tool_results"
        for event in journal.events
    )
    context["journal_receipt_records"] = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=Path.cwd(), task_row={}),
        journal=journal,
        binding=lambda: {},
    )
    typed_state = build_typed_state(messages, context, runtime)
    state_actions = typed_state["action_ledger"]["unclassified_actions"]
    assert any(row.get("call_id") == missing_id for row in state_actions)

    turns = context["unclassified_turns"]
    final_turn = next((
        turn for turn in turns
        if any(call.get("id") == missing_id
               for message in turn if message.get("role") == "assistant"
               for call in message.get("tool_calls", ()))
    ), None)
    assert final_turn is not None, f"call {missing_id} was dropped without typed state"
    result = next(
        message for message in final_turn
        if message.get("role") == "tool" and message.get("tool_call_id") == missing_id
    )
    assert result["content"] == (
        "ERROR: tool result was missing from the transcript; execution outcome is unknown."
    )

    rows, covered_ids = _opaque_actions(turns, journal)
    assert missing_id in covered_ids
    receipt = next(row for row in rows if row.get("call_id") == missing_id)
    assert receipt["result_receipt"] == result["content"]


def test_queued_result_stays_with_its_call_before_the_next_request():
    messages = [
        {"role": "system", "content": "Synthetic request."},
        {"role": "user", "content": "Synthetic task."},
    ]
    for index in range(100):
        call_id = f"page-{index}"
        messages.extend((
            {"role": "assistant", "content": None, "tool_calls": [
                _call(call_id, "read_check_receipt", {"handle": "synthetic", "offset": index}),
            ]},
            {"role": "tool", "tool_call_id": call_id, "content": "page settled"},
        ))
    grep_id = "grep-final"
    grep_call = _call(grep_id, "grep", {
        "pattern": "synthetic literal",
        "path": "task-input:" + hashlib.sha256(b"synthetic task").hexdigest(),
        "mode": "literal",
        "page_size": 50,
    })
    messages.extend((
        {"role": "assistant", "content": None, "tool_calls": [grep_call]},
        {"role": "user", "content": "Intervening host note."},
    ))
    batch = NativeToolBatch()
    batch._pending_tool_result_messages = [
        {"role": "assistant", "content": None, "tool_calls": [grep_call]},
        {"role": "tool", "tool_call_id": grep_id, "content": "ERROR: outcome unknown"},
    ]

    assert batch.append_unexecuted_tool_results(messages) == 1
    assistant_index = next(
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and any(call.get("id") == grep_id for call in message.get("tool_calls", ()))
    )
    assert messages[assistant_index + 1].get("role") == "tool"
    assert messages[assistant_index + 1].get("tool_call_id") == grep_id

    journal = _Journal()
    context = build_action_context(messages, journal=journal)
    context["journal_receipt_records"] = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=Path.cwd(), task_row={}),
        journal=journal,
        binding=lambda: {},
    )
    typed_state = build_typed_state(messages, context, runtime)
    assert any(
        row.get("call_id") == grep_id
        for row in typed_state["action_ledger"]["unclassified_actions"]
    )


def test_malformed_group_preserves_each_identifiable_call():
    call_id = "identifiable-call"
    messages = [
        {"role": "system", "content": "Synthetic request."},
        {"role": "user", "content": "Synthetic task."},
        {"role": "assistant", "content": None, "tool_calls": [
            _call(call_id, "grep", {"pattern": "literal", "path": "missing.py"}),
            "malformed sibling",
        ]},
    ]
    journal = _Journal()
    context = build_action_context(messages, journal=journal)
    context["journal_receipt_records"] = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=Path.cwd(), task_row={}),
        journal=journal,
        binding=lambda: {},
    )

    typed_state = build_typed_state(messages, context, runtime)
    action = next(
        row for row in typed_state["action_ledger"]["unclassified_actions"]
        if row.get("call_id") == call_id
    )
    assert action["result_receipt"] == (
        "ERROR: tool result was missing from the transcript; execution outcome is unknown."
    )


def test_result_without_text_is_settled_as_unknown():
    call_id = "empty-result-call"
    messages = [
        {"role": "system", "content": "Synthetic request."},
        {"role": "user", "content": "Synthetic task."},
        {"role": "assistant", "content": None, "tool_calls": [
            _call(call_id, "grep", {"pattern": "literal", "path": "missing.py"}),
        ]},
        {"role": "tool", "tool_call_id": call_id, "content": None},
    ]
    journal = _Journal()
    context = build_action_context(messages, journal=journal)
    context["journal_receipt_records"] = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=Path.cwd(), task_row={}),
        journal=journal,
        binding=lambda: {},
    )

    typed_state = build_typed_state(messages, context, runtime)
    action = next(
        row for row in typed_state["action_ledger"]["unclassified_actions"]
        if row.get("call_id") == call_id
    )
    assert action["result_receipt"] == (
        "ERROR: tool result was missing from the transcript; execution outcome is unknown."
    )


def test_reused_call_id_does_not_crash_when_later_result_is_missing():
    call_id = "reused-call"
    messages = [
        {"role": "system", "content": "Synthetic request."},
        {"role": "user", "content": "Synthetic task."},
        {"role": "assistant", "content": None, "tool_calls": [
            _call(call_id, "grep", {"pattern": "first"}),
        ]},
        {"role": "tool", "tool_call_id": call_id, "content": "first result"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call(call_id, "grep", {"pattern": "second"}),
        ]},
    ]
    journal = _Journal()
    context = build_action_context(messages, journal=journal)
    context["journal_receipt_records"] = []
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=Path.cwd(), task_row={}),
        journal=journal,
        binding=lambda: {},
    )

    typed_state = build_typed_state(messages, context, runtime)
    actions = typed_state["action_ledger"]["unclassified_actions"]
    assert [row.get("result_receipt") for row in actions] == [
        "first result",
        "ERROR: tool result was missing from the transcript; execution outcome is unknown.",
    ]


def test_cli_bad_grep_scope_gets_a_result_before_the_next_request(tmp_path):
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    run = tmp_path / "run"
    script = Script({"steps": [
        {"tool": "grep", "arguments": {
            "pattern": "synthetic literal",
            "path": "task-input:" + hashlib.sha256(b"synthetic task").hexdigest(),
            "mode": "literal",
            "page_size": 50,
        }},
        {"tool": "done", "arguments": {
            "summary": "Inspection complete.",
            "no_change_paths": [{"path": "app.py", "reason": "Inspection only."}],
        }},
    ], "tokenize": {"mode": "ok"}})
    second_response_started = threading.Event()
    release_second_response = threading.Event()
    handler = scripted_endpoint.make_handler(script)
    send = handler._send

    def hold_second_response(self, status, payload):
        if (status == 200 and isinstance(payload, dict) and payload.get("choices")
                and len(script.requests) == 2):
            second_response_started.set()
            release_second_response.wait(timeout=scaled(30))
        send(self, status, payload)

    handler._send = hold_second_response
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    environment = {
        **{key: value for key, value in os.environ.items()
           if not key.startswith("STATEFRAME_")},
        "STATEFRAME_HOME": isolated_stateframe_home(),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root / "src"),
    }
    command = [
        sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
        "--task", "Inspect the sample repository.", "--test-python", sys.executable,
        "--check-cmd", "true", "--endpoint",
        f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
        "--run-dir", str(run), "--max-turns", "2", "--no-run-env", "--quiet",
    ]
    process = None
    stdout = stderr = ""
    try:
        process = subprocess.Popen(
            command, cwd=root, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        deadline = time.monotonic() + scaled(180)
        while (not second_response_started.is_set() and process.poll() is None
               and time.monotonic() < deadline):
            time.sleep(0.02)
        if not second_response_started.is_set():
            if process.poll() is None:
                signal_run_owned(
                    run, process, signal.SIGKILL, deadline=time.monotonic(),
                    what="bad-grep CLI request timeout", require_checkpoint_saved=False,
                    group=True,
                )
            stdout, stderr = process.communicate(timeout=scaled(10))
            pytest.fail("bad-grep CLI did not reach the following request: " + stdout + stderr)

        script.assert_consumed()
        assert len(script.requests) == 2
        following_messages = script.requests[1]["messages"]
        assert any(
            row.get("role") == "tool" and row.get("tool_call_id")
            for row in following_messages
        )
    finally:
        release_second_response.set()
        try:
            if process is not None:
                if process.poll() is None:
                    signal_run_owned(
                        run, process, signal.SIGKILL, deadline=time.monotonic(),
                        what="bad-grep CLI cleanup", require_checkpoint_saved=False,
                        group=True,
                    )
                stdout, stderr = process.communicate(timeout=scaled(10))
        finally:
            server.shutdown()
            server.server_close()

    assert "frame-only typed state could not be constructed" not in stderr
    assert "one or more prior tool actions lack typed state" not in stderr


def _opaque_turn(call_id, pattern, result):
    call = _call(call_id, "grep", {"pattern": pattern})
    return [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _assistant_message(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def _pending_turn(call, result):
    return [
        _assistant_message(call),
        {"role": "tool", "tool_call_id": call["id"], "content": result},
    ]


def _result_pairs(messages, call_id):
    pairs = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = [call for call in message.get("tool_calls", ())
                 if isinstance(call, dict) and call.get("id") == call_id]
        for call in calls:
            result = None
            for following in messages[index + 1:]:
                if following.get("role") == "assistant":
                    break
                if (following.get("role") == "tool"
                        and following.get("tool_call_id") == call_id):
                    result = following.get("content")
                    break
            function = call.get("function") or {}
            arguments = json.loads(function.get("arguments") or "{}")
            pairs.append((arguments.get("pattern"), result))
    return pairs


def _opaque_action_rows(messages):
    marker = '"unclassified_actions":'
    decoder = json.JSONDecoder()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        start = content.find(marker)
        if start < 0:
            continue
        try:
            rows, _end = decoder.raw_decode(content, start + len(marker))
        except json.JSONDecodeError:
            continue
        if isinstance(rows, list):
            return rows
    return []


def test_opaque_actions_accepts_same_id_in_two_later_turns():
    turns = [
        _opaque_turn("reused", "first", "first settled result"),
        _opaque_turn("reused", "second", "second settled result"),
    ]

    rows, covered_ids = _opaque_actions(turns)

    assert covered_ids == {"reused"}
    assert [(row["turn_index"], row["result_receipt"]) for row in rows] == [
        (0, "first settled result"),
        (1, "second settled result"),
    ]


def test_opaque_actions_accepts_reuse_after_a_settled_unknown():
    turns = [
        _opaque_turn(
            "reused", "first",
            "ERROR: tool result was missing from the transcript; execution outcome is unknown.",
        ),
        _opaque_turn("reused", "second", "second settled result"),
    ]

    rows, _ = _opaque_actions(turns)

    assert [row["result_receipt"] for row in rows] == [
        "ERROR: tool result was missing from the transcript; execution outcome is unknown.",
        "second settled result",
    ]


def test_opaque_actions_accepts_call_id_reused_three_times():
    turns = [
        _opaque_turn("reused", "first", "first result"),
        _opaque_turn("reused", "second", "second result"),
        _opaque_turn("reused", "third", "third result"),
    ]

    rows, covered_ids = _opaque_actions(turns)

    assert covered_ids == {"reused"}
    assert [row["turn_index"] for row in rows] == [0, 1, 2]
    assert [row["result_receipt"] for row in rows] == [
        "first result", "second result", "third result",
    ]


@pytest.mark.parametrize("duplicate_shape", ["same-call", "same-assistant"])
def test_opaque_actions_rejects_duplicate_id_inside_one_turn(duplicate_shape):
    call = _call("duplicate", "grep", {"pattern": "same action"})
    if duplicate_shape == "same-call":
        assistant = _assistant_message(call, call)
        turn = [assistant, {"role": "tool", "tool_call_id": "duplicate", "content": "settled"}]
    else:
        turn = [
            _assistant_message(call),
            _assistant_message(dict(call)),
            {"role": "tool", "tool_call_id": "duplicate", "content": "settled"},
        ]

    with pytest.raises(FrameFit5Error):
        _opaque_actions([turn])


def test_opaque_actions_rejects_the_same_turn_object_supplied_twice():
    turn = _opaque_turn("reused", "same action", "settled result")

    with pytest.raises(FrameFit5Error):
        _opaque_actions([turn, turn])


def test_opaque_actions_rejects_the_same_call_object_supplied_in_later_turn():
    call = _call("reused", "grep", {"pattern": "same action"})
    result = {"role": "tool", "tool_call_id": "reused", "content": "settled result"}
    turns = [
        [_assistant_message(call), dict(result)],
        [_assistant_message(call), dict(result)],
    ]

    with pytest.raises(FrameFit5Error):
        _opaque_actions(turns)


def test_opaque_actions_allows_a_later_identical_call_as_a_second_row():
    turns = [
        _opaque_turn("reused", "same action", "same result"),
        _opaque_turn("reused", "same action", "same result"),
    ]

    rows, _ = _opaque_actions(turns)

    assert [row["turn_index"] for row in rows] == [0, 1]
    assert [row["result_receipt"] for row in rows] == ["same result", "same result"]


def test_reused_call_result_is_inserted_after_the_newest_assistant():
    first = _call("reused", "grep", {"pattern": "first"})
    second = _call("reused", "grep", {"pattern": "second"})
    messages = [
        {"role": "user", "content": "Synthetic task."},
        _assistant_message(first),
        {"role": "tool", "tool_call_id": "reused", "content": "first result"},
        _assistant_message(second),
    ]
    batch = NativeToolBatch()
    batch._pending_tool_result_messages = [
        {"role": "tool", "tool_call_id": "reused", "content": "second result"},
    ]

    assert batch.append_unexecuted_tool_results(messages) == 1

    assert _result_pairs(messages, "reused") == [
        ("first", "first result"),
        ("second", "second result"),
    ]


def test_reused_call_result_already_in_newest_turn_is_not_duplicated():
    first = _call("reused", "grep", {"pattern": "first"})
    second = _call("reused", "grep", {"pattern": "second"})
    messages = [
        _assistant_message(first),
        {"role": "tool", "tool_call_id": "reused", "content": "first result"},
        _assistant_message(second),
        {"role": "tool", "tool_call_id": "reused", "content": "second result"},
    ]
    batch = NativeToolBatch()
    batch._pending_tool_result_messages = _pending_turn(second, "second result")

    assert batch.append_unexecuted_tool_results(messages) == 0
    assert _result_pairs(messages, "reused") == [
        ("first", "first result"),
        ("second", "second result"),
    ]


def test_reused_call_ids_inside_one_pending_batch_stay_with_each_turn():
    first = _call("reused", "grep", {"pattern": "first"})
    second = _call("reused", "grep", {"pattern": "second"})
    messages = [_assistant_message(first)]
    batch = NativeToolBatch()
    batch._pending_tool_result_messages = [
        *_pending_turn(first, "first result"),
        *_pending_turn(second, "second result"),
    ]

    assert batch.append_unexecuted_tool_results(messages) == 2
    assert _result_pairs(messages, "reused") == [
        ("first", "first result"),
        ("second", "second result"),
    ]


def test_multiple_queued_results_preserve_call_order():
    first = _call("first", "grep", {"pattern": "first"})
    second = _call("second", "grep", {"pattern": "second"})
    assistant = _assistant_message(first, second)
    messages = [assistant]
    batch = NativeToolBatch()
    batch._pending_tool_result_messages = [
        assistant,
        {"role": "tool", "tool_call_id": "first", "content": "first result"},
        {"role": "tool", "tool_call_id": "second", "content": "second result"},
    ]

    assert batch.append_unexecuted_tool_results(messages) == 2
    assistant_index = messages.index(assistant)
    assert [message.get("tool_call_id") for message in messages[assistant_index + 1:]] == [
        "first", "second",
    ]


def _parent_source_tree(root, destination):
    parent = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    archive = subprocess.run(
        ["git", "-C", str(root), "archive", "--format=tar", parent],
        capture_output=True, check=True,
    ).stdout
    destination.mkdir()
    subprocess.run(
        ["tar", "-xf", "-", "-C", str(destination)],
        input=archive, capture_output=True, check=True,
    )
    return destination


def _unique_call_request_bytes(engine_root, repo, run_dir, home):
    script = Script({"steps": [
        {"tool": "grep", "arguments": {
            "pattern": "VALUE", "path": "app.py", "mode": "literal", "page_size": 50,
        }},
        {"tool": "grep", "arguments": {
            "pattern": "1", "path": "app.py", "mode": "literal", "page_size": 50,
        }},
        {"tool": "done", "arguments": {
            "summary": "Inspection complete.",
            "no_change_paths": [{"path": "app.py", "reason": "Inspection only."}],
        }},
    ], "tokenize": {"mode": "ok"}})
    third_response_started = threading.Event()
    release_third_response = threading.Event()
    handler = scripted_endpoint.make_handler(script)
    send = handler._send

    def hold_third_response(self, status, payload):
        if (status == 200 and isinstance(payload, dict) and payload.get("choices")
                and len(script.requests) == 3):
            third_response_started.set()
            release_third_response.wait(timeout=30)
        send(self, status, payload)

    handler._send = hold_third_response
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    environment = {
        **{key: value for key, value in os.environ.items()
           if not key.startswith("STATEFRAME_")},
        "STATEFRAME_HOME": isolated_stateframe_home(),
        "REACT_NATIVE_TOOL_BATCHES": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(engine_root / "src"),
    }
    command = [
        sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
        "--task", "Inspect the sample repository.", "--check-cmd", "true", "--endpoint",
        f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
        "--run-dir", str(run_dir), "--max-turns", "4", "--no-run-env",
        "--test-python", sys.executable, "--profile", "r2hc3",
        "--shell-network", "loopback", "--quiet",
    ]
    process = subprocess.Popen(
        command, cwd=engine_root, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 180
        while (not third_response_started.is_set() and process.poll() is None
               and time.monotonic() < deadline):
            time.sleep(0.02)
        if not third_response_started.is_set():
            if process.poll() is None:
                signal_run_owned(
                    run_dir, process, signal.SIGKILL, deadline=time.monotonic(),
                    what="unique-call CLI cleanup", require_checkpoint_saved=False,
                    group=True,
                )
            stdout, stderr = process.communicate(timeout=10)
            pytest.fail("unique-call CLI did not reach the complete-group request: "
                        + stdout + stderr)
        script.assert_consumed()
        requests = sorted((run_dir / "wire").glob("*.request.json"))
        assert len(requests) == 3
        call_ids = [
            call.get("id")
            for message in script.requests[-1].get("messages", ())
            if message.get("role") == "assistant"
            for call in message.get("tool_calls", ())
        ]
        assert len(call_ids) == len(set(call_ids))
        result = [path.read_bytes() for path in requests]
        signal_run_owned(
            run_dir, process, signal.SIGKILL, deadline=time.monotonic() + 180,
            what="unique-call CLI mid-run interruption", group=True,
        )
        process.communicate(timeout=10)
        assert process.returncode == -signal.SIGKILL
        return result
    finally:
        release_third_response.set()
        if process.poll() is None:
            signal_run_owned(
                run_dir, process, signal.SIGKILL, deadline=time.monotonic(),
                what="unique-call CLI cleanup", require_checkpoint_saved=False,
                group=True,
            )
            process.communicate(timeout=10)
        server.shutdown()
        server.server_close()


def test_complete_unique_id_groups_match_parent_request_bytes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("parent request parity needs this repository's Git history")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    home = tmp_path / "home"
    parent_tree = _parent_source_tree(root, tmp_path / "parent")

    parent_requests = _unique_call_request_bytes(
        parent_tree, repo, tmp_path / "parent-run", home)
    current_requests = _unique_call_request_bytes(
        root, repo, tmp_path / "current-run", home)

    assert current_requests == parent_requests


def _serve_with_reused_ids(
    script, third_response_started, release_third_response,
    fourth_response_started, release_fourth_response,
):
    handler = scripted_endpoint.make_handler(script)
    send = handler._send

    def send_with_reused_ids(self, status, payload):
        if status == 200 and isinstance(payload, dict):
            choices = payload.get("choices") or ()
            if choices:
                calls = choices[0].get("message", {}).get("tool_calls") or ()
                request_count = len(script.requests)
                if request_count <= 2:
                    for call in calls:
                        function = call.get("function") or {}
                        if function.get("name") == "grep":
                            call["id"] = "local-reused"
                elif request_count == 3:
                    third_response_started.set()
                    release_third_response.wait(timeout=30)
                elif request_count == 4:
                    fourth_response_started.set()
                    release_fourth_response.wait(timeout=30)
        send(self, status, payload)

    handler._send = send_with_reused_ids
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_cli_resume_after_kill_keeps_reused_call_groups_settled(tmp_path):
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    run = tmp_path / "run"
    script = Script({"steps": [
        {"tool": "grep", "arguments": {
            "pattern": "VALUE", "path": "app.py", "mode": "literal", "page_size": 50,
        }},
        {"tool": "grep", "arguments": {
            "pattern": "NOT_PRESENT", "path": "app.py", "mode": "literal", "page_size": 50,
        }},
        {"tool": "done", "arguments": {
            "summary": "Inspection complete.",
            "no_change_paths": [{"path": "app.py", "reason": "Inspection only."}],
        }},
        {"tool": "done", "arguments": {
            "summary": "Inspection complete.",
            "no_change_paths": [{"path": "app.py", "reason": "Inspection only."}],
        }},
    ], "tokenize": {"mode": "ok"}})
    third_response_started = threading.Event()
    release_third_response = threading.Event()
    fourth_response_started = threading.Event()
    release_fourth_response = threading.Event()
    server = _serve_with_reused_ids(
        script, third_response_started, release_third_response,
        fourth_response_started, release_fourth_response,
    )
    environment = {
        **{key: value for key, value in os.environ.items()
           if not key.startswith("STATEFRAME_")},
        "STATEFRAME_HOME": isolated_stateframe_home(),
        "REACT_NATIVE_TOOL_BATCHES": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root / "src"),
    }
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    command = [
        sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
        "--task", "Inspect the sample repository.", "--check-cmd", "true",
        "--endpoint", endpoint, "--model", "scripted",
        "--run-dir", str(run), "--max-turns", "8", "--no-run-env",
        "--test-python", sys.executable, "--profile", "r2hc3",
        "--shell-network", "loopback", "--quiet",
    ]
    process = subprocess.Popen(
        command, cwd=root, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    resumed_process = None
    try:
        deadline = time.monotonic() + 180
        while (not third_response_started.is_set() and process.poll() is None
               and time.monotonic() < deadline):
            time.sleep(0.02)
        if not third_response_started.is_set():
            if process.poll() is None:
                signal_run_owned(
                    run, process, signal.SIGKILL, deadline=time.monotonic(),
                    what="reused-call initial run cleanup",
                    require_checkpoint_saved=False, group=True,
                )
            stdout, stderr = process.communicate(timeout=10)
            pytest.fail("third scripted request was not reached: " + stdout + stderr)
        request_path = run / "wire" / "0003.request.json"
        assert request_path.is_file()
        original_request = request_path.read_bytes()
        signal_run_owned(
            run, process, signal.SIGKILL, deadline=time.monotonic() + 180,
            what="reused-call initial run checkpoint interruption", group=True,
        )
        process.communicate(timeout=10)
        assert process.returncode == -signal.SIGKILL
        release_third_response.set()

        resumed_process = subprocess.Popen(
            [sys.executable, "-m", "stateframe", "resume", str(run),
             "--endpoint", endpoint, "--model", "scripted", "--max-turns", "8", "--quiet"],
            cwd=root, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        resume_deadline = time.monotonic() + 180
        while (not fourth_response_started.is_set() and resumed_process.poll() is None
               and time.monotonic() < resume_deadline):
            time.sleep(0.02)
        if not fourth_response_started.is_set():
            if resumed_process.poll() is None:
                signal_run_owned(
                    run, resumed_process, signal.SIGKILL,
                    deadline=time.monotonic(), what="reused-call resumed run cleanup",
                    require_checkpoint_saved=False, group=True,
                )
            stdout, stderr = resumed_process.communicate(timeout=10)
            pytest.fail("resumed CLI did not reach the reused-call request: " + stdout + stderr)
        resumed_request_path = run / "wire" / "0001.request.json"
        assert resumed_request_path.is_file()
        signal_run_owned(
            run, resumed_process, signal.SIGKILL, deadline=time.monotonic() + 180,
            what="reused-call resumed run checkpoint interruption", group=True,
        )
        resumed_process.communicate(timeout=10)
        assert resumed_process.returncode == -signal.SIGKILL
        script.assert_consumed()
        assert len(script.requests) >= 4
        for request in script.requests[2:4]:
            rows = [
                row for row in _opaque_action_rows(request["messages"])
                if isinstance(row, dict) and row.get("tool") == "grep"
            ]
            assert [row["arguments"]["pattern"] for row in rows] == [
                "VALUE", "NOT_PRESENT",
            ]
            assert rows[0]["result_sha256"] != rows[1]["result_sha256"]
            assert rows[0]["turn_index"] < rows[1]["turn_index"]
        prior_wire = run / "attempts" / "1" / "wire"
        repeated_provider_ids = []
        for filename in ("0001.response.json", "0002.response.json"):
            response = json.loads((prior_wire / filename).read_text(encoding="utf-8"))
            repeated_provider_ids.append([
                call.get("id")
                for call in response.get("choices", [{}])[0]
                .get("message", {}).get("tool_calls", ())
                if call.get("id")
            ])
        assert repeated_provider_ids == [["local-reused"], ["local-reused"]]
        assert original_request == (
            run / "attempts" / "1" / "wire" / "0003.request.json"
        ).read_bytes()
    finally:
        release_third_response.set()
        release_fourth_response.set()
        if process.poll() is None:
            signal_run_owned(
                run, process, signal.SIGKILL, deadline=time.monotonic(),
                what="reused-call initial run cleanup", require_checkpoint_saved=False,
                group=True,
            )
            process.communicate(timeout=10)
        if resumed_process is not None and resumed_process.poll() is None:
            signal_run_owned(
                run, resumed_process, signal.SIGKILL, deadline=time.monotonic(),
                what="reused-call resumed run cleanup",
                require_checkpoint_saved=False, group=True,
            )
            resumed_process.communicate(timeout=10)
        server.shutdown()
        server.server_close()
