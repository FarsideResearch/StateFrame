"""Repeated-query regression tests at the history and provider boundaries."""
from __future__ import annotations

import copy
import gzip
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from http.server import ThreadingHTTPServer
from inspect import getsource
from itertools import pairwise
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

FIXTURE = Path(__file__).resolve()
ROOT = FIXTURE.parents[1]
BASE = "47db805430cb312c05b5b487cc9ff78771723003"
UNTOUCHED = "29765b39820fd39800fd7292f52fb5bcb81e6069"  # 0.9.9, the immediate branch base
MODEL = "repeat-compaction-mock"
TASK = "Find NEEDLE in app.py and report its value without changing files."
MATCH = "app.py:1:NEEDLE = 7"
BATCH_TEXT = "I will search for NEEDLE twice in this response."
NOTICE = "[host-capacity] fixed test notice"
SUMMARY = ("Task: inspect app.py. Decisions: search for NEEDLE. Files changed: none. "
           "Current test status: not run. Open problems: none. Next steps: report the match.")
FORCED_SUMMARY = SUMMARY.replace("report the match", "report the value found in app.py")
DONE = {"tool": "done", "arguments": {"summary": "Located NEEDLE."}}


def _call(call_id, pattern="NEEDLE"):
    return {"id": call_id, "type": "function", "function": {
        "name": "grep", "arguments": json.dumps({"pattern": pattern, "path": "app.py"}),
    }}


def _result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _assistant(*calls, text=None):
    return {"role": "assistant", "content": text, "tool_calls": list(calls)}


def test_repeat_cut_stale_offset_inside_two_call_exchange():
    from kernel_stateframe_framefit5 import FrameFit5Error, _opaque_actions
    from probes import react_repair
    from probes.history_compaction import compact_messages, split_turns

    def exchange(call_id):
        return [_assistant(_call(call_id)), _result(call_id, f"real result {call_id}")]

    original = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "user", "content": "older leading note"},
        *exchange("old-1"), *exchange("old-2"), *exchange("anchor-call"),
    ]
    stale_offset = len(original) - 1
    assert stale_offset == 8
    messages = copy.deepcopy(compact_messages(
        original, "continued", keep_turns=1, original_prefix=original[:2],
    ))
    first, second = "batch-1", "batch-2"
    calls = [_call(first), _call(second)]
    delivered = [_result(call_id, f"real result {call_id}") for call_id in (first, second)]
    messages.extend([*exchange("current-1"), _assistant(*calls), *delivered])
    assert messages[7]["tool_calls"] == calls
    assert messages[8:10] == delivered

    def assert_complete(candidate):
        rows, covered = _opaque_actions(split_turns(candidate))
        assert {first, second} <= covered
        assert {row["call_id"]: row["result_receipt"] for row in rows
                if row.get("call_id") in {first, second}} == {
                    item["tool_call_id"]: item["content"] for item in delivered
                }

    assert_complete(messages)
    old_cut = copy.deepcopy(messages)
    del old_cut[stale_offset:]
    assert len(old_cut[7]["tool_calls"]) == 2
    assert old_cut[8:] == []
    with pytest.raises(FrameFit5Error, match="unclassified tool result is missing"):
        assert_complete(old_cut)

    active = {
        "occurrence": 5, "origin": (None, None), "text": None, "selected": 1,
        "calls": [
            {"ordinal": ordinal, "host_id": call["id"], "provider_id": call["id"],
             "declaration": call, "delivery": {"message": result, "marker": None},
             "archive": (False, False)}
            for ordinal, (call, result) in enumerate(zip(calls, delivered, strict=True))
        ],
        "advisories": [], "suppressed": None, "note": None, "wire_ids": {},
    }
    tags = {7: ((5, 0, "call"), (5, 1, "call")),
            8: ((5, 0, "result"),), 9: ((5, 1, "result"),)}
    def assert_candidate(planner):
        cut = planner(
            messages, boundary={"side": "after_action", "legacy_offset": stale_offset,
                                "occurrence": 3, "edge": None},
            active=active, selected=1, tags=tags, rewrite_applied=True,
        )
        assert_complete([message for index, message in enumerate(messages) if index not in cut["drop"]])
        assert cut["exchanges_removed"] == 0

    def legacy_planner(messages, *, boundary, **_):
        return {"drop": tuple(range(boundary["legacy_offset"], len(messages))),
                "exchanges_removed": 0}

    assert_candidate(react_repair._plan_repeat_cut)
    with pytest.raises(FrameFit5Error, match="unclassified tool result is missing"):
        assert_candidate(legacy_planner)


def _grep(pattern="NEEDLE"):
    return {"tool": "grep", "arguments": {"pattern": pattern, "path": "app.py"}}


def _batch(*patterns):
    first, *siblings = map(_grep, patterns)
    return {**first, "also": siblings}


def _repeats(noise=0):
    return ([_grep(f"NOISE_{index}") for index in range(noise)] + [_grep()] * 7
            + [_batch("NEEDLE", "NEEDLE")])


@contextmanager
def _provider(steps, *, responses=None):
    from stateframe.scripted_endpoint import Script, make_handler

    script = Script({"model": MODEL, "tokenize": {"mode": "ok"}, "steps": [*steps, DONE]})
    script.decisions = []
    script.summaries = []
    script.pause = False
    script.arrived = Event()
    script.release = Event()

    class Handler(make_handler(script)):
        def _send(self, status, payload):
            if (responses and isinstance(payload, dict)
                    and str(payload.get("id", "")).startswith("chatcmpl-")
                    and (response := responses.get(script.position))):
                text, ids = response
                message = payload["choices"][0]["message"]
                message["content"] = text
                for call, call_id in zip(message["tool_calls"], ids, strict=True):
                    call["id"] = call_id
            super()._send(status, payload)

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return super().do_POST()
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(gzip.decompress(raw) if self.headers.get("Content-Encoding") == "gzip"
                              else raw)
            if any(_content(message).startswith("Summarize the work so far for another coding-agent turn.")
                   for message in body["messages"]):
                script.summaries.append(raw)
                summary = SUMMARY if len(script.summaries) == 1 else FORCED_SUMMARY
                self._send(200, {
                    "id": f"summary-{len(script.summaries)}", "model": MODEL,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": summary}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 24, "total_tokens": 25},
                })
                return
            script.decisions.append(raw)
            if script.pause and script.position == len(steps):
                script.arrived.set()
                script.release.wait(timeout=60)
                self.close_connection = True
                return
            self.rfile = io.BytesIO(raw)
            return super().do_POST()

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        script.endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield script
        finally:
            script.release.set()
            server.shutdown()
            thread.join()


def _content(row):
    return str(row.get("content") or "")


def _messages(raw):
    return json.loads(raw)["messages"]


def _notes(messages):
    return [row for row in messages if row.get("role") == "user"
            and _content(row).startswith(("[kernel] You repeated `",
                                          "[kernel] You have been cycling"))]


def _grep_ids(messages, pattern="NEEDLE"):
    return [call["id"] for row in messages for call in row.get("tool_calls", ())
            if call["function"]["name"] == "grep"
            and json.loads(call["function"]["arguments"]).get("pattern") == pattern]


def _exchange(messages, text):
    indices = [index for index, row in enumerate(messages)
               if row.get("role") == "assistant" and _content(row) == text]
    assert len(indices) == 1, f"expected one assistant row with {text!r}, found {len(indices)}"
    [index] = indices
    ids = [call["id"] for call in messages[index]["tool_calls"]]
    assert len(ids) == len(set(ids))
    results = messages[index + 1:index + 1 + len(ids)]
    assert [row.get("tool_call_id") for row in results if row.get("role") == "tool"] == ids
    for call_id in ids:
        assert sum(call["id"] == call_id for row in messages
                   for call in row.get("tool_calls", ())) == 1
        assert sum(row.get("tool_call_id") == call_id for row in messages) == 1
    return ids, results


def _assert_batch(messages, expected_result=None, *, typed=None):
    ids, results = _exchange(messages, BATCH_TEXT)
    assert len(ids) == 2
    actual, suppressed = map(_content, results)
    assert MATCH in actual and "NOT EXECUTED" not in actual
    if expected_result is not None:
        assert actual == expected_result
    assert "NOT EXECUTED" in suppressed and "ACCEPTED" not in suppressed
    assert sum("NOT EXECUTED" in _content(row) for row in messages
               if row.get("role") == "tool") == 1
    if typed:
        assert suppressed.startswith("grep REFUSED repeat_query_suppressed:")
    return ids, actual


def _wire_metrics(label, decisions):
    adjacent = [len(os.path.commonprefix((left, right))) for left, right in pairwise(decisions)]
    print(f"{label} decision bytes {[len(raw) for raw in decisions]}; adjacent LCP bytes {adjacent}")


def _env(root, scratch):
    return {
        **{name: value for name, value in os.environ.items()
           if not name.startswith(("STATEFRAME_", "REACT_", "KERNEL_", "LONG_HORIZON_"))},
        "PYTHONPATH": os.pathsep.join(map(str, (root / "src", root / "src/stateframe/engine"))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": str(scratch),
        "STATEFRAME_HOME": str(scratch / "stateframe-home"),
    }


def _append_event(path, **event):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")


def _read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def _observe_product(rr, probe_path, *, notice, old_checkpoint, settled_checkpoint):
    """Install observation taps after genuine product environment setup."""
    from inspect import unwrap

    import kernel_checkpoint
    import kernel_navigation_pages
    import kernel_stall_observation
    from kernel_pilot import ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime

    if notice:
        from capability_release_runner import prepare_request

        plan = SimpleNamespace(enabled=True, rendered_notice=NOTICE)
        rr.ask_tools.prepare_request_inputs = lambda messages: prepare_request(messages, plan)

    prepare = StateFrameRuntime.prepare.__code__
    search = unwrap(kernel_navigation_pages.search_repository).__code__
    tool_reply = rr._tool_reply_for_call.__code__
    stall = kernel_stall_observation.observe_stall_in_place.__code__
    frame_state = kernel_checkpoint._frame_state.__code__
    save_point = kernel_checkpoint.save_point.__code__

    # Observe calls without replacing functions the checkpointer has to serialize.
    def observe(frame, event, result):
        code = frame.f_code
        if code is prepare and event == "call":
            fields = frame.f_locals
            if fields["provider_request"]:
                _append_event(probe_path, kind="prepare", capped=fields["messages"],
                              uncapped=fields["uncapped"],
                              step=getattr(fields["self"].owner, "current_step", 0))
        elif code is search and event == "call":
            runtime = ACTIVE_RUNTIME.get()
            batch = runtime.native_batch
            _append_event(probe_path, kind="grep-route", pattern=frame.f_locals["pattern"],
                          call_id=runtime.current_tool_call_id,
                          ordinal=batch.index - 1 if batch is not None else 0)
        elif code is tool_reply and event == "return" and isinstance(result, dict):
            if MATCH in _content(result):
                _append_event(probe_path, kind="grep-delivery", call_id=result["tool_call_id"],
                              message=result)
        elif code is stall and event == "return" and result:
            _append_event(probe_path, kind="stall", observation=result["observation"])
        elif settled_checkpoint and code is save_point and event == "call":
            # Save the settled response at its next boundary even though the
            # normal cadence saves only once per provider send.
            active = frame.f_locals["frame"].f_locals.get("_repeat_active")
            if active is not None and active["suppressed"] is not None:
                kernel_checkpoint._STATE["last_sends"] = None
        elif old_checkpoint and code is frame_state and event == "return" and result:
            for key in tuple(result):
                if key.startswith("_repeat_"):
                    del result[key]

    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        sys.setprofile(observe)
    else:
        # Restrict instrumentation to four functions when the interpreter supports it.
        tool = monitoring.PROFILER_ID
        monitoring.use_tool_id(tool, "repeat-query audit")
        start, returned = monitoring.events.PY_START, monitoring.events.PY_RETURN
        monitoring.register_callback(tool, start, lambda _code, _offset:
                                     observe(sys._getframe(1), "call", None))
        monitoring.register_callback(tool, returned, lambda _code, _offset, value:
                                     observe(sys._getframe(1), "return", value))
        for code in (prepare, search):
            monitoring.set_local_events(tool, code, start)
        if settled_checkpoint:
            monitoring.set_local_events(tool, save_point, start)
        for code in (tool_reply, stall):
            monitoring.set_local_events(tool, code, returned)
        if old_checkpoint:
            monitoring.set_local_events(tool, frame_state, returned)


# The host launcher keeps the real product child, profile, recorder and replay.
# Only the test process's launch arguments carry the observation callback.
_PRODUCT_DRIVER = """
import sys
from stateframe import runner
from stateframe.cli import main

observer_source, audit, old, notice, settled = sys.argv[1:6]
launch = runner._engine_child_launch

def observed_launch(options, run):
    command, cwd = launch(options, run)
    child = str(runner.CHILD)
    bootstrap = f'''
import os, runpy, sys
sys.path[0] = {str(runner.CHILD.parent)!r}
sys.argv = [{child!r}, {str(run)!r}]
os.chdir({str(run)!r})
def finish_install(frame, event, result):
    if event == "return":
        sys.settrace(None)
        from pathlib import Path
        observer = {{}}
        exec({observer_source!r}, observer)
        observer["_observe_product"](frame.f_locals["rr"], Path({audit!r}),
                                     notice={notice}, old_checkpoint={old}, settled_checkpoint={settled})
def install_probe(frame, event, arg):
    if (event == "call" and frame.f_code.co_name == "install_checked_ask"
            and frame.f_code.co_filename == {child!r}):
        frame.f_trace_lines = False
        return finish_install
    return None
sys.settrace(install_probe)
runpy.run_path({child!r}, run_name="__main__")
'''
    return [command[0], "-B", "-c", bootstrap, "stateframe._child"], cwd

runner._engine_child_launch = observed_launch
sys.argv = ["stateframe", *sys.argv[6:]]
raise SystemExit(main())
"""


def _cli_commands(tmp_path, endpoint, *, trim=4, switches=None, task=TASK,
                  advisory=None, checkpoint=False, old_checkpoint=False, minimal_profile=False,
                  root=ROOT, source="NEEDLE = 7\n", notice=False, test_command=None,
                  settled_checkpoint=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(source, encoding="utf-8")
    profile = {"schema": "stateframe-product-profile/1", "switches": {}}
    if not minimal_profile:
        profile = json.loads((root / "src/stateframe/profiles/r2hc3.json").read_text())
    profile["switches"].update({
        "REACT_TRIM": str(trim), "STATEFRAME_CHECKPOINT": "on" if checkpoint else "off",
        "STATEFRAME_CHECKPOINT_EVERY": "1", "KERNEL_INTERVENTIONS": "1",
        "REACT_NATIVE_TOOL_BATCHES": "1", "STATEFRAME_PROVIDER": "llama.cpp", **(switches or {}),
    })
    if checkpoint:
        # Publish the checkpoint after the batch-producing send before the
        # second sibling is selected. A prior checkpoint could replay the
        # response and recreate the very record this test needs to restore.
        profile["switches"]["STATEFRAME_CHECKPOINT_ASYNC"] = "off"
    if advisory is not None:
        profile["switches"]["STATEFRAME_ZERO_PROGRESS_NUDGE_STEP"] = str(advisory)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    run = tmp_path / "run"
    audit = tmp_path / "audit.jsonl"

    def launcher(*args, observe=False, old=False, settled=False):
        if not (observe or old):
            return [sys.executable, "-m", "stateframe", *args]
        # Importing pytest in the engine would create a ContextVar the product
        # checkpoint cannot serialize. Load only the observer and its helpers.
        observer_sources = (
            "import json, sys\nfrom types import SimpleNamespace\n",
            f"NOTICE = {NOTICE!r}\nMATCH = {MATCH!r}\n",
            *(getsource(function) for function in (_content, _append_event, _observe_product)),
        )
        return [sys.executable, "-c", _PRODUCT_DRIVER, "\n".join(observer_sources), str(audit),
                repr(old), repr(notice), repr(settled), *args]

    common = ("--endpoint", endpoint, "--keep-checkpoints", "--max-turns", "40")
    checks = ("--runner-kind", "generic", "--test-cmd", test_command) if test_command else ()
    start = launcher("run", "--repo", str(repo), "--task", task, "--model", MODEL,
                     "--profile", str(profile_path), "--run-dir", str(run), "--no-run-env",
                     "--quiet", *checks, *common, observe=not minimal_profile,
                     old=old_checkpoint, settled=settled_checkpoint)
    resume = launcher("resume", str(run), *common, observe=checkpoint and not old_checkpoint)
    return start, resume, run, audit


def _run_cli(command, root, scratch, run):
    from engine_run import run_monitored

    return run_monitored(command, cwd=root, env=_env(root, scratch), capture_output=True,
                         text=True, expected_seconds=180, run_dir=run, check=False)


def _cli_output(process, run):
    traceback = run / "child-traceback.txt"
    engine_log = run / "engine.log"
    return process.stdout + process.stderr + "".join(
        path.read_text() for path in (traceback, engine_log) if path.is_file()
    )


def _interrupt_at_decision(command, scratch, run, script):
    from _timeouts import scaled
    from engine_run import host_load_factor
    from fixtures.process_signals import (record_identity, signal_run_owned, signal_run_verified,
                                          wait_for_checkpoint_saved)

    script.pause = True
    finished = Event()
    launched_after = time.time()
    with (scratch / "interrupted.log").open("w+") as output, subprocess.Popen(
        command, cwd=ROOT, env=_env(ROOT, scratch), stdout=output,
        stderr=subprocess.STDOUT, start_new_session=True,
    ) as child:
        watcher = Thread(target=lambda: (child.wait(), finished.set(), script.arrived.set()))
        watcher.start()
        try:
            assert script.arrived.wait(scaled(120 * host_load_factor())), (
                "initial run never reached the affected decision"
            )
            output.seek(0)
            assert not finished.is_set(), output.read()
            latest = run / "checkpoints" / "LATEST"
            deadline = time.monotonic() + scaled(40)
            wait_for_checkpoint_saved(
                run, deadline=deadline, what="repeated-query decision checkpoint", process=child,
            )
            assert latest.is_file(), _read_events(run / "checkpoints/events.jsonl")
            saved = json.loads((latest.parent / latest.read_text().strip() / "checkpoint.json")
                               .read_text())
            assert saved["sends"] == script.position
            engine_pid = int((run / ".engine.lock").read_text())
            engine_identity = record_identity(
                engine_pid, launched_after=launched_after, command_fragment="_child.py",
            )
            signal_run_verified(
                run, engine_pid, engine_identity, signal.SIGKILL, deadline=deadline,
                what="repeated-query decision interruption", require_checkpoint_saved=False,
            )
            child.wait(timeout=scaled(30))
        finally:
            script.pause = False
            script.release.set()
            if child.poll() is None:
                signal_run_owned(
                    run, child, signal.SIGTERM, deadline=time.monotonic(),
                    what="repeated-query wrapper cleanup", require_checkpoint_saved=False,
                )
                try:
                    child.wait(timeout=scaled(5))
                except subprocess.TimeoutExpired:
                    signal_run_owned(
                        run, child, signal.SIGKILL, deadline=time.monotonic(),
                        what="stuck repeated-query wrapper cleanup", require_checkpoint_saved=False,
                    )
                    child.wait(timeout=scaled(5))
            watcher.join(timeout=scaled(30))
    return script.decisions[-1]


def _assert_sibling_advisory(messages, expected):
    [batch] = [i for i, row in enumerate(messages) if _content(row) == BATCH_TEXT]
    [advisory] = [i for i, row in enumerate(messages) if row.get("role") == "user"
                  and expected in _content(row)]
    [note] = _notes(messages)
    assert batch + 2 < advisory < messages.index(note)


def _assert_product_boundary(events, *, expected_noise):
    dispatches = [event for event in events if event["kind"] == "grep-route"]
    assert [event["pattern"] for event in dispatches] == [
        f"NOISE_{index}" for index in range(expected_noise)] + ["NEEDLE"] * 8
    assert all(event["ordinal"] == 0 for event in dispatches)
    prepares = [event for event in events if event["kind"] == "prepare"]
    assert any("earlier exchanges trimmed to fit context" in _content(row)
               for event in prepares for row in event["capped"])
    [delivery] = [event["message"] for event in events if event["kind"] == "grep-delivery"
                  and event["call_id"] == dispatches[-1]["call_id"]]
    affected_index = next((index for index, event in enumerate(prepares)
                           if _notes(event["capped"])), None)
    assert affected_index, "suppression did not reach provider preparation"
    _assert_affected_prepare(prepares[affected_index], dispatches, delivery, expected_noise)
    if expected_noise:
        # This complete singleton was still live immediately before the batch.
        # The trim limit leaves enough room for it even after the first sibling.
        eligible_id = _grep_ids(prepares[affected_index - 1]["capped"])[-1]
        for view in ("capped", "uncapped"):
            _assert_exchange_removed(prepares[affected_index][view], {eligible_id})
    return dispatches, delivery, prepares[affected_index]


def _assert_affected_prepare(affected, dispatches, delivery, expected_noise):
    heading = f"[kernel] No edit or check yet after {expected_noise + 8} turns:"
    for messages in (affected["capped"], affected["uncapped"]):
        ids, _ = _assert_batch(messages, _content(delivery))
        assert ids[0] == dispatches[-1]["call_id"]
        assert ids[1] not in {event["call_id"] for event in dispatches}
        _assert_sibling_advisory(messages, heading)
        [note] = _notes(messages)
        if expected_noise:
            assert "saved history boundary is no longer available" not in _content(note)
        else:
            assert "saved history boundary is no longer available" in _content(note)


def _assert_exchange_removed(messages, removed_ids):
    assert removed_ids
    assert removed_ids.isdisjoint(call["id"] for row in messages for call in row.get("tool_calls", ()))
    assert removed_ids.isdisjoint(row.get("tool_call_id") for row in messages)


def _assert_only_submission_retries(run, decisions, affected_index):
    from _journal import measurement_events

    # The shipped profile can refuse an unchanged, unchecked submission twice.
    # Any decision after the first affected send must account for one such refusal.
    refusals = [event for event in measurement_events(run) if event["kind"] == "early_done_refused"]
    tail = decisions[affected_index:]
    assert len(tail) == 1 + len(refusals)
    assert len(refusals) <= 2
    for raw in tail[1:]:
        messages = _messages(raw)
        latest = next(row for row in reversed(messages) if row.get("tool_calls"))
        [call] = latest["tool_calls"]
        assert call["function"]["name"] == "done"
        [result] = [row for row in messages if row.get("tool_call_id") == call["id"]]
        assert "[host] done refused:" in _content(result)


@pytest.mark.parametrize(("trim", "noise", "resume"), [(4, 0, None), (24, 20, None),
                                                       (4, 0, "queued"), (4, 0, "settled")],
                         ids=("lost-pin", "found-pin", "checkpoint-resume", "settled-resume"))
def test_next_decision_keeps_executed_and_suppressed_siblings(tmp_path, trim, noise, resume):
    steps = _repeats(noise)
    responses = {len(steps): (BATCH_TEXT, ("declared-grep-8", "declared-grep-9"))}
    with _provider(steps, responses=responses) as script:
        initial, restart, run, audit = _cli_commands(
            tmp_path, script.endpoint, trim=trim, advisory=len(steps), checkpoint=resume is not None,
            settled_checkpoint=resume == "settled",
        )
        if resume:
            interrupted = _interrupt_at_decision(initial, tmp_path, run, script)
            initial_events = _read_events(audit)
            dispatches, delivery, interrupted_prepare = _assert_product_boundary(
                initial_events, expected_noise=noise,
            )
            script.decisions.clear()
            result = _run_cli(restart, ROOT, tmp_path, run)
            assert result.returncode == 0, _cli_output(result, run)
            assert script.decisions[0] == interrupted
            _assert_only_submission_retries(run, script.decisions, 0)
            affected_wire = script.decisions[0]
            restored = _read_events(run / "checkpoints/events.jsonl")
            assert any(row["event"] == "checkpoint_restored" for row in restored)
            resume_spec = json.loads((run / "spec.json").read_text())["resume"]
            assert resume_spec["attempt"] == 2 and resume_spec["checkpoint"]
            assert resume_spec["checkpoint"]["sends"] == len(steps)
            assert resume_spec["checkpoint"]["step"] == len(steps) + (resume == "settled")
            assert '"kind":"checkpoint_verified"' in (run / "engine.log").read_text()
            resumed_events = _read_events(audit)[len(initial_events):]
            assert not any(event["kind"] in {"grep-route", "grep-delivery"}
                           for event in resumed_events), "restore repeated grep execution"
            restored_prepares = [event for event in resumed_events if event["kind"] == "prepare"]
            assert restored_prepares, "resume never rebuilt the provider input views"
            restored_prepare = restored_prepares[0]
            _assert_affected_prepare(restored_prepare, dispatches, delivery, noise)
            for view in ("capped", "uncapped"):
                assert restored_prepare[view] == interrupted_prepare[view], view
            _wire_metrics("interrupted / restored", [interrupted, *script.decisions])
        else:
            result = _run_cli(initial, ROOT, tmp_path, run)
            assert result.returncode == 0, _cli_output(result, run)
            _assert_only_submission_retries(run, script.decisions, len(steps))
            affected_wire = script.decisions[len(steps)]
            _, delivery, _ = _assert_product_boundary(_read_events(audit), expected_noise=noise)
            _wire_metrics(f"product trim={trim}", script.decisions)
        script.assert_consumed()
        assert not script.summaries
        environment = json.loads((run / "spec.json").read_text())["environment"]
        assert environment["STATEFRAME_FRAMEFIT"] == environment["STATEFRAME_WIRE_TRUTH_GATE"] == "on"
        messages = _messages(affected_wire)
        _assert_batch(messages, _content(delivery))
        advisory = f"[kernel] No edit or check yet after {len(steps)} turns:"
        assert sum(_content(row).count(advisory) for row in messages) == 1


def test_product_cli_history_arm_is_blocked_before_model_compaction(tmp_path):
    with _provider(_repeats()) as script:
        initial, _, run, _ = _cli_commands(
            tmp_path, script.endpoint, minimal_profile=True,
            switches={"KERNEL_MODE": "render_selection:OFF", "REACT_HISTORY_COMPACTION": "on",
                      "REACT_COMPACTION_THRESHOLD": "2", "REACT_COMPACTION_KEEP_TURNS": "1",
                      "STATEFRAME_VIEW": "operational", "STATEFRAME_PROJECTOR": "rich"},
        )
        result = _run_cli(initial, ROOT, tmp_path, run)
        output = _cli_output(result, run)
        assert result.returncode != 0
        assert "frame-only preflight degraded to history; refusing" in output
        assert script.decisions == script.summaries == script.requests == []


@pytest.mark.parametrize("trim", [1000000, 4], ids=["legacy", "trimmed"])
def test_legacy_suppression_checkpoint_after_a_later_sibling(tmp_path, trim):
    steps = [*[_grep()] * 7, _batch("NEEDLE", "NEEDLE", "LATER")]
    source = "NEEDLE = 7\nLATER = 8\n"
    settled_step = len(steps) + 2
    with _provider(steps, responses={len(steps): (BATCH_TEXT, ("a", "b", "c"))}) as script:
        initial, restart, run, audit = _cli_commands(
            tmp_path, script.endpoint, trim=trim, checkpoint=True, source=source,
            switches={"STATEFRAME_CHECKPOINT_EVERY": str(settled_step)},
        )
        interrupted = _interrupt_at_decision(initial, tmp_path, run, script)
        saved = [event["step"] for event in _read_events(run / "checkpoints/events.jsonl")
                 if event["event"] == "checkpoint_saved"]
        assert saved == [settled_step], "only the settled suppression boundary needs a synchronous save"
        events = _read_events(audit)
        assert [event["pattern"] for event in events if event["kind"] == "grep-route"] == [
            *["NEEDLE"] * 8, "LATER",
        ]
        prepare = next(event for event in reversed(events) if event["kind"] == "prepare")
        for view in ("capped", "uncapped"):
            [note] = _notes(prepare[view])
            if trim == 4:
                assert "saved history boundary is no longer available" in _content(note)
                _, (first, suppressed, later) = _exchange(prepare[view], BATCH_TEXT)
                assert MATCH in _content(first)
                assert "NOT EXECUTED" in _content(suppressed)
                assert "app.py:2:LATER = 8" in _content(later)
            else:
                assert "those exchanges have been removed" in _content(note)
        script.decisions.clear()
        result = _run_cli(restart, ROOT, tmp_path, run)
        assert result.returncode == 0, _cli_output(result, run)
        restored = _read_events(run / "checkpoints/events.jsonl")
        assert any(event["event"] == "checkpoint_restored" for event in restored)
        resume_spec = json.loads((run / "spec.json").read_text())["resume"]
        assert resume_spec["attempt"] == 2
        assert resume_spec["checkpoint"]["step"] == settled_step
        assert script.decisions[0] == interrupted
        assert '"kind":"checkpoint_verified"' in (run / "engine.log").read_text()
        assert not any(event["kind"] == "grep-route" for event in _read_events(audit)[len(events):])
        _assert_only_submission_retries(run, script.decisions, 0)
        _wire_metrics(f"trim={trim} interrupted / restored", [interrupted, script.decisions[0]])
        script.assert_consumed()


def test_older_checkpoint_with_missing_repeat_evidence_uses_verified_replay(tmp_path):
    steps = _repeats()
    with _provider(steps, responses={len(steps): (BATCH_TEXT, ("grep-8", "grep-9"))}) as script:
        initial, restart, run, _ = _cli_commands(
            tmp_path, script.endpoint, checkpoint=True, old_checkpoint=True,
        )
        interrupted = _interrupt_at_decision(initial, tmp_path, run, script)
        script.decisions.clear()
        result = _run_cli(restart, ROOT, tmp_path, run)
        assert result.returncode == 0, _cli_output(result, run)
        refusal_log = run / "attempts/2/checkpoints/events.jsonl"
        refusal = [event for event in _read_events(refusal_log)
                   if event["event"] == "checkpoint_restore_refused"]
        assert refusal and any("repeat" in event["detail"].lower() for event in refusal), refusal_log
        resume_spec = json.loads((run / "spec.json").read_text())["resume"]
        assert resume_spec["attempt"] == 3
        assert resume_spec.get("checkpoint") is None
        assert not resume_spec.get("restart_live")
        assert len(resume_spec["tape"]) == len(steps)
        assert f'"replayed":{len(steps)}' in (run / "engine.log").read_text()
        assert script.decisions[0] == interrupted
        _assert_only_submission_retries(run, script.decisions, 0)
        script.assert_consumed()


def _worker(scratch, monkeypatch, script, *, switches=None, source="NEEDLE = 7\n",
            task=TASK, notice=False, advisory=None, provider_limit=None, force=False,
            summary_at=None, stall_at=None, probe=None):
    """Drive the real worker and compactor with a local OpenAI-compatible endpoint."""
    import history_compaction
    import kernel_navigation_pages
    import kernel_stall_observation
    from capability_capacity_preflight import CapacityRefusal
    from capability_release_runner import prepare_request
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME, ParentMeter
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_verdict_identity import _ACTIVE_PROVIDER
    from probes import react_repair

    from stateframe.task_row import build_row
    from stateframe.workcopy import create

    repo = scratch / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(source, encoding="utf-8")
    row = build_row(create(repo, scratch / "workspace"), target="app.py", task=task,
                    task_id="repeat-query", runner_kind="generic")
    run = scratch / "run"
    results = run / "worker/results"
    journals = run / "worker/measurement-runs"
    journals.mkdir(parents=True)
    for name in list(os.environ):
        if name.startswith(("STATEFRAME_", "REACT_", "KERNEL_", "LONG_HORIZON_")):
            monkeypatch.delenv(name)
    for name, value in {
        "STATEFRAME_HOME": str(scratch / "stateframe-home"),
        "STATEFRAME_RESULTS_DIR": str(results),
        "STATEFRAME_PROJECTOR": "rich",
        "STATEFRAME_VIEW": "operational",
        "STATEFRAME_IDEALPATH": "off",
        "STATEFRAME_PROVIDER": "llama.cpp",
        "STATEFRAME_WIRE_TRUTH_GATE": "off",
        "STATEFRAME_IDLE_GOVERNOR": "off",
        "REACT_TOOLCALL": "1",
        "REACT_NATIVE_TOOL_BATCHES": "1",
        "REACT_TRIM": "1000000",
        "KERNEL_INTERVENTIONS": "1",
        "KERNEL_MODE": "render_selection:OFF",
        **(switches or {}),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    if advisory is not None:
        monkeypatch.setenv("STATEFRAME_ZERO_PROGRESS_NUDGE_STEP", str(advisory))
    monkeypatch.setattr(react_repair, "API", script.endpoint + "/chat/completions")
    monkeypatch.setattr(react_repair, "MODEL", MODEL)
    monkeypatch.setattr(react_repair, "KEY", "")

    init = StateFrameRuntime.__init__

    def scoped_init(self, owner, **kwargs):
        init(self, owner, **{**kwargs, "output": journals})

    monkeypatch.setattr(StateFrameRuntime, "__init__", scoped_init)
    prepare = StateFrameRuntime.prepare
    compact = history_compaction.HistoryCompactor.maybe_compact
    stall = kernel_stall_observation.observe_stall_in_place
    original_ask = react_repair.ask_tools
    if probe is None:
        probe = {}
    probe.update(prepare=[], compact=[], applied_compactions=[], stall=[], capacity=[], delivered=[],
                 dispatched=[])

    tool_reply = react_repair._tool_reply_for_call
    search = kernel_navigation_pages.search_repository

    def capture_reply(messages, call_id, observation, **kwargs):
        result = tool_reply(messages, call_id, observation, **kwargs)
        if MATCH in _content(result):
            probe["delivered"].append(copy.deepcopy(result))
        return result

    def capture_search(root, pattern, *args, **kwargs):
        probe["dispatched"].append(pattern)
        return search(root, pattern, *args, **kwargs)

    def capture(self, messages, *, uncapped, **kwargs):
        before = (copy.deepcopy(messages), copy.deepcopy(uncapped))
        prepared = prepare(self, messages, uncapped=uncapped, **kwargs)
        if kwargs.get("provider_request", True):
            probe["prepare"].append((before, copy.deepcopy(prepared.messages)))
        return prepared

    def observe_compaction(self, messages, *args, force=False, **kwargs):
        before = copy.deepcopy(messages)
        old_threshold = self.threshold
        # Select the ordinary compaction input by its transcript. The actual
        # estimate, summary charge, model send, validation, and rewrite run unchanged.
        if summary_at is not None and not force:
            self.threshold = 1 if (
                not probe.get("ordinary_applied") and summary_at(messages)) else 10**9
        try:
            result = compact(self, messages, *args, force=force, **kwargs)
        finally:
            self.threshold = old_threshold
        if result.applied and not force:
            probe["ordinary_applied"] = True
        if result.applied:
            probe["applied_compactions"].append((force, before, copy.deepcopy(result.messages)))
        probe["compact"].append((force, before, result.applied))
        return result

    def observe_stall(runtime, messages, requirement, **kwargs):
        if stall_at is not None and not stall_at(messages):
            # Let the observer track progress before the selected interval.
            # Its normal output bound prevents it from emitting this oversized brief.
            requirement *= 64
        observed = stall(runtime, messages, requirement, **kwargs)
        if observed:
            probe["stall"].append(observed["observation"])
        return observed

    def ask(messages, **kwargs):
        if force and not probe["capacity"] and _notes(messages):
            probe["capacity"].append(copy.deepcopy(messages))
            raise CapacityRefusal("fixture exact capacity refusal", prompt_tokens=65000,
                                  context_limit=65536, reroutable=True)
        return original_ask(messages, **kwargs)

    monkeypatch.setattr(StateFrameRuntime, "prepare", capture)
    monkeypatch.setattr(react_repair, "_tool_reply_for_call", capture_reply)
    monkeypatch.setattr(kernel_navigation_pages, "search_repository", capture_search)
    monkeypatch.setattr(history_compaction.HistoryCompactor, "maybe_compact", observe_compaction)
    monkeypatch.setattr(kernel_stall_observation, "observe_stall_in_place", observe_stall)
    if notice:
        plan = SimpleNamespace(enabled=True, rendered_notice=NOTICE)
        ask.prepare_request_inputs = lambda messages: prepare_request(messages, plan)
    if force or notice:
        monkeypatch.setattr(react_repair, "ask_tools", ask)
    if provider_limit is not None:
        closure_profile = ParentMeter._closure_profile
        monkeypatch.setattr(ParentMeter, "_closure_profile", lambda self, journal: {
            **closure_profile(self, journal), "provider_slots": 0,
        })
    with ExitStack() as contexts:
        for active in (ACTIVE_PARENT, ACTIVE_RUNTIME, ACTIVE_REQUEST, _ACTIVE_PROVIDER):
            contexts.callback(active.reset, active.set(None))
        if provider_limit is not None:
            parent = ParentMeter(attempts=1, steps=None, enforce=True,
                                 provider_limit=provider_limit)
            contexts.callback(ACTIVE_PARENT.reset, ACTIVE_PARENT.set(parent))
        result = react_repair.work(row)
    assert not result["result"].startswith("ERR:"), result
    script.assert_consumed()
    return run


@pytest.mark.parametrize(("keep", "found"), [(0, False), (1, False), (1, True)],
                         ids=("keep-0-lost", "keep-1-lost", "keep-1-found"))
def test_real_summary_and_forced_retry_preserve_one_exchange(tmp_path, monkeypatch, keep, found):
    from _journal import measurement_events

    noise = 5 if found else 0
    steps = _repeats(noise)
    probe = {}

    def ordinary_summary_at(messages):
        if keep == 1:
            # The second matching singleton installs the streak pin. With keep=1
            # it survives only if this is the last turn at the ordinary rewrite.
            needed = 2 if found else 4
            return len(_grep_ids(messages)) >= needed
        return any(_content(row) == BATCH_TEXT for row in messages)

    with _provider(steps, responses={len(steps): (BATCH_TEXT, ("grep-8", "grep-9"))}) as script:
        run = _worker(tmp_path, monkeypatch, script, force=True, notice=True, advisory=len(steps),
                      summary_at=ordinary_summary_at, probe=probe, switches={
                          "REACT_HISTORY_COMPACTION": "on", "REACT_COMPACTION_KEEP_TURNS": str(keep),
                          "REACT_COMPACTION_THRESHOLD": "2", "LONG_HORIZON_PROTOCOL": "on",
                          "LONG_HORIZON_NO_COMPACTION": "off",
                          "STATEFRAME_TOOL_REPLY_CONSTRUCTOR": "on",
                      })
    events = measurement_events(run)
    summaries = [event for event in events if event["kind"] == "history_compaction"]
    assert any(applied for force, _, applied in probe["compact"] if not force)
    assert len(probe["capacity"]) == 1
    assert any(force and applied for force, _, applied in probe["compact"])
    assert len(summaries) == len(script.summaries) == 2
    assert all(event["summary_request_id"] and event["summary_tokens"] > 0 for event in summaries)
    ids = {event["summary_request_id"] for event in summaries}
    charged = [event for event in events if event["kind"] == "history_compaction_accounting"
               and event.get("counted_in_run_totals")]
    assert len(charged) == len(ids) == 2
    assert {event["request_id"] for event in charged} == ids
    assert any(event["kind"] == "capacity_reroute"
               and event.get("action") == "history_force_summary" for event in events)
    [(ordinary, before, after), (forced, _, forced_after)] = probe["applied_compactions"]
    assert not ordinary and forced
    _assert_summary(after, SUMMARY)
    _assert_summary(forced_after, FORCED_SUMMARY)
    heading = f"[kernel] No edit or check yet after {len(steps)} turns:"
    exact = _content(probe["delivered"][-1])
    [first_attempt] = probe["capacity"]
    retry = _messages(script.decisions[len(steps)])
    for messages, summary in ((first_attempt, SUMMARY), (retry, FORCED_SUMMARY)):
        _assert_summary(messages, summary)
        _assert_batch(messages, exact, typed=True)
        assert sum(_content(row) == NOTICE for row in messages) == 1
        _assert_sibling_advisory(messages, heading)
        [note] = _notes(messages)
        assert ("saved history boundary is no longer available" in _content(note)) != found
    affected = [(index, views) for index, (views, _) in enumerate(probe["prepare"])
                if _notes(views[0])]
    assert len(affected) >= 2
    first_index, first_views = affected[0]
    _, retry_views = affected[1]
    for views, summary in ((first_views, SUMMARY), (retry_views, FORCED_SUMMARY)):
        for messages in views:
            _assert_summary(messages, summary)
            _assert_batch(messages, exact, typed=True)
            assert sum(_content(row) == NOTICE for row in messages) == 1
            _assert_sibling_advisory(messages, heading)
    if found:
        _, pin_id = _grep_ids(before)
        assert _grep_ids(after) == [pin_id]
        assert any(row.get("tool_call_id") == pin_id for row in after)
        # The six singleton repeats from the pin through the last pre-batch
        # decision are complete, still live, and eligible for removal.
        previous_views, _ = probe["prepare"][first_index - 1]
        [pin, *eligible] = _grep_ids(previous_views[0])
        assert pin == pin_id and len(eligible) == 5
        removed_ids = {pin_id, *eligible}
        assert sorted(row["tool_call_id"] for row in previous_views[0]
                      if row.get("tool_call_id") in removed_ids) == sorted(removed_ids)
        for messages in (*first_views, first_attempt):
            _assert_exchange_removed(messages, removed_ids)
    executed_id = probe["delivered"][-1]["tool_call_id"]
    for _, before, _ in probe["compact"]:
        batch_rows = [row for row in before if _content(row) == BATCH_TEXT]
        assert len(batch_rows) <= 1
        batch_ids = [call["id"] for row in batch_rows for call in row.get("tool_calls", ())]
        for call_id in batch_ids:
            assert sum(row.get("tool_call_id") == call_id for row in before) <= 1
        assert sum(row.get("tool_call_id") == executed_id for row in before) <= 1
    if keep == 0:
        [forced_input] = [before for force, before, _ in probe["compact"] if force]
        assert not any(row.get("tool_call_id") == executed_id for row in forced_input)
    assert len(script.decisions) == len(steps) + 2
    _wire_metrics(f"history keep={keep} found={found}", script.decisions)


def _assert_summary(messages, expected):
    summary_rows = [_content(row) for row in messages if row.get("role") == "user"]
    assert sum(text.endswith("\n" + expected) for text in summary_rows) == 1
    stale = FORCED_SUMMARY if expected == SUMMARY else SUMMARY
    assert not any(text.endswith("\n" + stale) for text in summary_rows)


@pytest.mark.parametrize("later", [False, True], ids=["last-sibling", "later-sibling-delivered"])
def test_forced_summary_activates_legacy_suppression_record(tmp_path, monkeypatch, later):
    steps = _repeats(noise=2)
    ids = ("grep-8", "grep-9")
    if later:
        steps[-1] = _batch("NEEDLE", "NEEDLE", "LATER")
        ids += ("grep-later",)
    probe = {}
    with _provider(steps, responses={len(steps): (BATCH_TEXT, ids)}) as script:
        _worker(tmp_path, monkeypatch, script, force=True, notice=True, advisory=len(steps),
                probe=probe, switches={
                    "REACT_HISTORY_COMPACTION": "on", "REACT_COMPACTION_KEEP_TURNS": "0",
                    "REACT_COMPACTION_THRESHOLD": "1000000", "LONG_HORIZON_PROTOCOL": "on",
                    "LONG_HORIZON_NO_COMPACTION": "off",
                    "STATEFRAME_TOOL_REPLY_CONSTRUCTOR": "on",
                })
    [first_attempt] = probe["capacity"]
    assert _notes(first_attempt)
    assert not any(_content(row) == BATCH_TEXT or "NOT EXECUTED" in _content(row)
                   for row in first_attempt)
    [(forced, summary_input, _)] = probe["applied_compactions"]
    assert forced and len(script.summaries) == 1
    assert not any(_content(row) == BATCH_TEXT for row in summary_input)
    assert probe["dispatched"] == ["NOISE_0", "NOISE_1", *["NEEDLE"] * 8, *(["LATER"] if later else [])]
    later_result = next((row for row in summary_input if row.get("role") == "tool"
                         and "pattern='LATER'" in _content(row)), None)
    assert (later_result is not None) == later
    retry = _messages(script.decisions[len(steps)])
    exact = _content(probe["delivered"][-1])
    [(views, _)] = [(views, prepared) for views, prepared in probe["prepare"]
                    if any(_content(row).endswith("\n" + SUMMARY) for row in views[0])
                    and any(_content(row) == BATCH_TEXT for row in views[0])]
    for messages in (*views, retry):
        _assert_summary(messages, SUMMARY)
        if later:
            _, results = _exchange(messages, BATCH_TEXT)
            first, refused, last = results
            assert _content(first) == exact
            assert _content(refused).startswith("grep REFUSED repeat_query_suppressed: NOT EXECUTED")
            assert last == later_result
        else:
            _assert_batch(messages, exact, typed=True)
        assert sum(_content(row) == NOTICE for row in messages) == 1
        _assert_sibling_advisory(messages, f"[kernel] No edit or check yet after {len(steps)} turns:")
    assert len(script.decisions) == len(steps) + 2
    _wire_metrics("forced activation", script.decisions)


def test_repeat_suppression_settles_later_queue_siblings(tmp_path, monkeypatch):
    batch = _batch("NEEDLE", "NEEDLE", "LATER_A", "LATER_B")
    batch["also"].append({"tool": "read", "arguments": {"path": "app.py"}})
    steps = [*[_grep()] * 7, batch]
    probe = {}
    with _provider(steps, responses={len(steps): (BATCH_TEXT, ("a", "b", "c", "d", "e"))}) as script:
        _worker(tmp_path, monkeypatch, script, probe=probe,
                source="NEEDLE = 7\nLATER_A = 8\nLATER_B = 9\nREAD_ONLY = 10\n",
                switches={"REACT_TRIM": "4", "STATEFRAME_TOOL_REPLY_CONSTRUCTOR": "on"})
    [(views, _)] = [(views, prepared) for views, prepared in probe["prepare"]
                    if _notes(views[0]) and any(_content(row) == BATCH_TEXT for row in views[0])]
    for messages in (*views, _messages(script.decisions[len(steps)])):
        ids, results = _exchange(messages, BATCH_TEXT)
        assert len(ids) == 5
        executed, selected, later_a, later_b, read = map(_content, results)
        assert executed == _content(probe["delivered"][-1])
        assert selected.startswith("grep REFUSED repeat_query_suppressed: NOT EXECUTED")
        assert "READ_ONLY = 10" in read
        assert "app.py:2:LATER_A = 8" in later_a
        assert "app.py:3:LATER_B = 9" in later_b
        assert all("NOT EXECUTED" not in text for text in (later_a, later_b, read))
    assert len(probe["delivered"]) == 8
    assert probe["dispatched"] == [*["NEEDLE"] * 8, "LATER_A", "LATER_B"]
    assert len(script.decisions) == len(steps) + 2


def test_trimmed_repeat_streak_keeps_batch_prefix(tmp_path, monkeypatch):
    read = {"tool": "read", "arguments": {"path": "app.py"}}
    batch = {**read, "also": [_grep(), _grep()]}
    steps = [*[_grep(f"NOISE_{index}") for index in range(18)], batch, *[_grep()] * 7]
    probe = {}
    with _provider(steps, responses={19: (BATCH_TEXT, ("read-first", "grep-first", "grep-repeat"))}) as script:
        _worker(tmp_path, monkeypatch, script, probe=probe, switches={"REACT_TRIM": "44"})

    before = _messages(script.decisions[len(steps) - 1])
    assert any("earlier exchanges trimmed to fit context" in _content(row) for row in before)
    [read_call] = [call for row in before for call in row.get("tool_calls", ())
                   if call["function"]["name"] == "read"]
    [read_result] = [row for row in before if row.get("tool_call_id") == read_call["id"]]
    assert "NEEDLE = 7" in _content(read_result)
    views, _ = next((views, prepared) for views, prepared in probe["prepare"] if _notes(views[0]))
    for messages in (*views, _messages(script.decisions[len(steps)])):
        assert any(read_call in row.get("tool_calls", ()) for row in messages)
        assert read_result in messages
        [note] = _notes(messages)
        assert "Removed 6 earlier complete query exchanges" in _content(note)


def test_stall_advisory_preserves_exact_sibling_delivery(tmp_path, monkeypatch):
    steps = _repeats()
    probe = {}
    with _provider(steps, responses={len(steps): (BATCH_TEXT, ("grep-8", "grep-9"))}) as script:
        _worker(tmp_path, monkeypatch, script, probe=probe, advisory=len(steps), switches={
            "REACT_TRIM": "16", "STATEFRAME_REPEAT_GATE": "off",
            "STATEFRAME_TOOL_REPLY_CONSTRUCTOR": "off",
        }, stall_at=lambda messages: not _notes(messages)
                and any(_content(row) == BATCH_TEXT for row in messages))
    [observation] = probe["stall"]
    exact = _content(probe["delivered"][-1])
    views, _ = probe["prepare"][len(steps)]
    heading = f"[kernel] No edit or check yet after {len(steps)} turns:"
    for messages in (*views, _messages(script.decisions[len(steps)])):
        _assert_batch(messages, exact)
        _assert_sibling_advisory(messages, observation)
        _assert_sibling_advisory(messages, heading)
        stall_index = next(index for index, row in enumerate(messages) if observation in _content(row))
        nudge_index = next(index for index, row in enumerate(messages) if heading in _content(row))
        assert stall_index < nudge_index


@pytest.fixture(scope="module")
def control_trees(tmp_path_factory):
    from tools.pinned_tree import pinned_tree

    baseline = pinned_tree(BASE, tmp_path_factory)
    if checkout := os.environ.get("SF_TEST_A4_UNTOUCHED"):
        target = Path(checkout).resolve()
        version = "0.9.8"
    else:
        # The immediate branch base has the pinned engine and profile and can be
        # archived independently in branch-only checkouts.
        release = subprocess.run(["git", "cat-file", "-e", f"{UNTOUCHED}^{{commit}}"],
                                 cwd=ROOT, check=False, capture_output=True, timeout=30)
        assert release.returncode == 0, "the immediate branch base or SF_TEST_A4_UNTOUCHED is required"
        target = pinned_tree(UNTOUCHED, tmp_path_factory)
        version = "0.9.9"
    assert not target.samefile(baseline) and not target.samefile(ROOT)
    assert '[project]' in (target / "pyproject.toml").read_text()
    assert f'version = "{version}"' in (target / "pyproject.toml").read_text()
    return {"pinned": baseline, "untouched": target, "candidate": ROOT}


def _parity_case(scratch, case, presentation):
    """The same driver is executed in isolated Python processes for all three trees."""
    steps = [_grep(), _grep("ABSENT")]
    notice = False
    source = "NEEDLE = 7\nABSENT = 0\n"
    task = TASK
    limit = None
    test_command = None
    switches = {"REACT_HISTORY_COMPACTION": "off"}
    if case in {"consecutive", "quoted", "stall"}:
        steps = _repeats()
        if case == "quoted":
            task += ' The phrase "3 earlier exchanges trimmed to fit context." is a quote.'
        if case == "stall":
            # Observe genuinely identical result bytes while leaving consecutive
            # query suppression enabled. The ledger's repeat advice varies per call.
            switches.update(STATEFRAME_STALL_OBSERVATION="on", STATEFRAME_REPEAT_GATE="off")
    elif case == "cycling":
        steps = [_grep(pattern) for pattern in ("NEEDLE", "OTHER") * 10]
        notice = True
        source += "OTHER = 8\n"
        if presentation == "product":
            # Pin after a real action so the legacy cut retains the task even
            # with the capacity prepender's extra system row.
            steps.insert(0, {"tool": "run_tests", "arguments": {}})
            test_command = "python3 -B -c 'from app import NEEDLE; assert NEEDLE == 7'"
    elif case == "four":
        steps = [_batch("PADDING", "NEEDLE", "ABSENT_A", "ABSENT_B")]
        # These short hits keep the exact request bytes independent of the
        # temporary workspace path emitted as the scope of a no-hit result.
        source += "ABSENT_A = 1\nABSENT_B = 2\n" + ("# PADDING " + "x" * 160 + "\n") * 30
        limit = 3
        switches.update(REACT_HISTORY_COMPACTION="on", REACT_COMPACTION_THRESHOLD="4400",
                        REACT_COMPACTION_KEEP_TURNS="1", KERNEL_INTERVENTIONS="0")
    else:
        assert case == "no-repeat"
    probe = {}
    with _provider(steps) as script:
        if presentation == "product":
            assert limit is None
            run, audit = _product_parity_run(scratch, script, switches=switches, source=source,
                                             task=task, notice=notice, test_command=test_command)
            _assert_only_submission_retries(run, script.decisions, len(steps))
            probe["stall"] = [event for event in audit if event["kind"] == "stall"]
        else:
            with pytest.MonkeyPatch.context() as monkeypatch:
                run = _worker(scratch, monkeypatch, script, switches=switches, source=source,
                              task=task, notice=notice, provider_limit=limit, probe=probe)
    if presentation == "history" and case != "four":
        assert len(script.decisions) == len(steps) + 2
    if case in {"consecutive", "quoted", "stall", "cycling"}:
        transcripts = [_messages(raw) for raw in script.decisions]
        if presentation == "product":
            # FrameFit renders host advisories into the frame. The audit shows
            # which preparation reached that decision without decoding its prose.
            transcripts = [event["capped"] for event in audit if event["kind"] == "prepare"]
        notes = [note for messages in transcripts for note in _notes(messages)]
        prefix = ("[kernel] You have been cycling" if case == "cycling"
                  else "[kernel] You repeated")
        assert any(_content(note).startswith(prefix) for note in notes)
        if presentation == "product":
            sent = _messages(script.decisions[len(steps)])
            assert any(prefix in _content(row) for row in sent)
            if notice:
                assert sum(_content(row).count(NOTICE) for row in sent) == 1
        assert script.summaries == []
        if case == "consecutive":
            assert any(_notes(messages)
                       and not any("NOT EXECUTED" in _content(row) for row in messages)
                       for messages in transcripts), "baseline note-only decision was never exercised"
        if notice:
            assert any(_notes(messages) and any(_content(row) == NOTICE for row in messages)
                       for messages in transcripts)
        if case == "stall":
            assert probe["stall"], "the real stall observer never copied the transcript"
    elif case == "four":
        from _journal import measurement_events

        events = measurement_events(run)
        assert len(script.summaries) == 1
        [summary] = [event for event in events if event["kind"] == "history_compaction"]
        assert summary["summary_tokens"] > 0 and summary["summary_request_id"]
        assert len(script.decisions) == 2
        assert all(not _notes(_messages(raw)) for raw in script.decisions)
        second = script.decisions[1]
        assert b"ABSENT_B" in second and b"PADDING" not in second
        assert any(SUMMARY in _content(row) for row in _messages(second))
    else:
        assert all(not _notes(_messages(raw)) for raw in script.decisions)
        assert not script.summaries
    return script.decisions, script.summaries


def _product_parity_run(scratch, script, *, switches, source, task, notice, test_command):
    from _journal import measurement_events

    from stateframe.request_records import list_request_records, read_request_record

    root = Path.cwd()
    initial, _, run, audit = _cli_commands(
        scratch, script.endpoint, root=root, trim=1000000, switches=switches, source=source,
        task=task, notice=notice, test_command=test_command,
    )
    result = _run_cli(initial, root, scratch, run)
    assert result.returncode == 0, _cli_output(result, run)
    script.assert_consumed()
    environment = json.loads((run / "spec.json").read_text())["environment"]
    assert environment["STATEFRAME_FRAMEFIT"] == environment["STATEFRAME_WIRE_TRUTH_GATE"] == "on"
    events = measurement_events(run)
    checked = {event["body_sha256"].removeprefix("sha256:"): event["request_id"] for event in events
               if event["kind"] == "wire_truth_checked" and event["final"] and not event["failed"]}
    requests = {record["request_id"]: record for name in list_request_records(run)
                for record in [json.loads(read_request_record(run, name))]}
    assert script.decisions
    sent = [requests[checked[sha256(raw).hexdigest()]] for raw in script.decisions]
    assert all(record["presentation"]["framefit5_mode"] == "FRAME_ONLY" for record in sent)
    sent_turns = {record["turn"] for record in sent}
    return run, [event for event in _read_events(audit)
                 if event["kind"] != "prepare" or event["step"] in sent_turns]


_PARITY_DRIVER = """
import json
import runpy
import sys
from pathlib import Path

fixture, scratch, case, presentation = sys.argv[1:]
scratch = Path(scratch)
scenario = runpy.run_path(fixture)["_parity_case"]
decisions, summaries = scenario(scratch, case, presentation)
(scratch / "wire.json").write_text(json.dumps(
    [[body.hex() for body in group] for group in (decisions, summaries)]))
"""


def _compare_control_bytes(tmp_path, control_trees, case, presentation):
    from _timeouts import scaled
    from engine_run import host_load_factor

    captured = {}
    for label, root in control_trees.items():
        scratch = tmp_path / label
        scratch.mkdir()
        environment = _env(root, scratch)
        environment["PYTHONPATH"] += os.pathsep + str(ROOT / "tests")
        result = subprocess.run([sys.executable, "-c", _PARITY_DRIVER, str(FIXTURE), str(scratch), case,
                                 presentation],
                                cwd=root, env=environment, capture_output=True, text=True,
                                timeout=scaled(240 * host_load_factor()), check=False)
        assert result.returncode == 0, f"{case}/{presentation}/{label}\n{result.stdout}{result.stderr}"
        captured[label] = tuple([bytes.fromhex(raw) for raw in group] for group in
                                json.loads((scratch / "wire.json").read_text()))
        decisions, summaries = captured[label]
        _wire_metrics(f"{case}/{presentation}/{label}", decisions)
        if summaries:
            print(f"{case}/{label} summary bytes {[len(raw) for raw in summaries]}")
    baseline = captured["pinned"]
    for label in ("untouched", "candidate"):
        for role, before, after in zip(("decision", "summary"), baseline, captured[label], strict=True):
            assert len(before) == len(after), f"{case}/{label}: {role} count changed"
            for index, (expected, actual) in enumerate(zip(before, after, strict=True), start=1):
                prefix = len(os.path.commonprefix((expected, actual)))
                assert actual == expected, (
                    f"{case}/{label}: {role} {index}: expected {len(expected)} bytes, "
                    f"got {len(actual)}; common prefix {prefix} bytes"
                )


@pytest.mark.parametrize("case", ["no-repeat", "consecutive", "quoted", "stall", "cycling"])
@pytest.mark.parametrize("presentation", ["history", "product"])
def test_unaffected_requests_match_pinned_wire_bytes(tmp_path, control_trees, case, presentation):
    _compare_control_bytes(tmp_path, control_trees, case, presentation)


@pytest.mark.parametrize("typed", ["on", "off"])
@pytest.mark.parametrize("reuse", [False, True], ids=["native-looking", "reused-ID"])
def test_batch_identity_receipts_and_archive_are_occurrence_scoped(tmp_path, monkeypatch, typed, reuse):
    older_id = "native_call_1"
    newer_id = older_id if reuse else "native_call_2"
    older_text = "This response ran its own search."
    newer_text = "This response independently repeated the search."
    responses = {8: (older_text, (older_id,)), 9: (newer_text, (newer_id,))}
    probe = {}
    with _provider([_grep()] * 9, responses=responses) as script:
        _worker(tmp_path, monkeypatch, script, probe=probe, switches={
            "REACT_TRIM": "8", "REACT_NATIVE_TOOL_BATCHES": "0",
            "STATEFRAME_TOOL_REPLY_CONSTRUCTOR": typed,
        })
    messages = _messages(script.decisions[9])
    [first_id], [first] = _exchange(messages, older_text)
    [second_id], [second] = _exchange(messages, newer_text)
    assert first_id == older_id
    if reuse:
        assert second_id.startswith(older_id + "__a4_")
    else:
        assert second_id == newer_id
    assert MATCH in _content(first) and "NOT EXECUTED" not in _content(first)
    [delivery] = [row for row in probe["delivered"] if row["tool_call_id"] == older_id]
    assert _content(first) == _content(delivery)
    refusal = _content(second)
    assert "NOT EXECUTED" in refusal and "ACCEPTED" not in refusal
    if typed == "on":
        assert refusal.startswith("grep REFUSED repeat_query_suppressed:")
    else:
        assert not refusal.startswith("grep REFUSED ")
    _, prior = probe["prepare"][8]
    # The executed response is new to the affected decision. Its two outgoing
    # views must agree on the exact result and the final wire must preserve it.
    views, _ = probe["prepare"][9]
    assert all(_content(_exchange(view, older_text)[1][0]) == _content(delivery) for view in views)
    assert not any(_content(row) == newer_text for row in prior)


def test_repeat_archive_reconstructs_each_grep_once(tmp_path, monkeypatch):
    from detreason.attempt_memory import _actions, _reconstruct_messages

    with _provider(_repeats(), responses={8: (BATCH_TEXT, ("grep-8", "grep-9"))}) as script:
        run = _worker(tmp_path, monkeypatch, script, switches={"REACT_TRIM": "4"})
    results = run / "worker/results"
    [transcript] = (results / "react_transcripts").glob("*.json")
    [log] = (results / "react_events").glob("*.jsonl")
    events = _read_events(log)
    assert any(event["kind"] == "trimmed" for event in events)
    rows = _reconstruct_messages(transcript, json.loads(transcript.read_text())["messages"], events)
    actions, _ = _actions(rows)
    assert sum(action.kind == "grep" for action in actions) == 9
    calls = [call["id"] for row in rows for call in row.get("tool_calls", ())
             if call["function"]["name"] == "grep"]
    assert len(set(calls)) == len(calls) == 9
    deliveries = [row for row in rows if row.get("tool_call_id") in calls]
    assert len({row["tool_call_id"] for row in deliveries}) == len(deliveries) == 9
    seen = [_content(row) for row in deliveries]
    assert sum(MATCH in result for result in seen) == 8
    assert sum("NOT EXECUTED" in result for result in seen) == 1


def test_repeat_module_import_preserves_fresh_clone_temp_state(tmp_path):
    clone = tmp_path / "clone"
    clone.mkdir()
    tests = clone / "tests"
    tests.mkdir()
    copied_test = tests / Path(__file__).name
    copied_test.write_bytes(Path(__file__).read_bytes())
    driver = """
import os
import runpy
import sys
import tempfile
from pathlib import Path

cached = tempfile.tempdir
configured = os.environ.get("TMPDIR")
before = set(Path.cwd().rglob("*"))
runpy.run_path(sys.argv[1])
assert tempfile.tempdir == cached
assert os.environ.get("TMPDIR") == configured
assert set(Path.cwd().rglob("*")) == before
"""
    result = subprocess.run([sys.executable, "-c", driver, str(copied_test)], cwd=clone,
                            env=_env(ROOT, tmp_path), capture_output=True, text=True,
                            timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unsuppressed_four_call_batch_costs_one_summary(tmp_path, control_trees):
    _compare_control_bytes(tmp_path, control_trees, "four", "history")
