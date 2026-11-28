"""R052 continuity and raw provider-byte parity against StateFrame 0.9.9."""

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from _timeouts import scaled
from kernel_stateframe_runtime import ViableAlternatives, viable_alternatives_enabled
from tools.pinned_tree import pinned_tree

from stateframe.runner import OPT_IN_SWITCHES, resolve_opt_in_switches
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve

CHECKOUT = Path(__file__).resolve().parents[1]
PINNED = "29765b39820fd39800fd7292f52fb5bcb81e6069"
SWITCH = "STATEFRAME_VIABLE_ALTERNATIVES"
PAIR = {
    "evidence_path": "contract.json", "key": "format",
    "alternatives": ["legacy", "modern"], "edit_path": "app.py",
    "dependent_edit": "Set the decoder format in app.py.",
}
CONTRACT = b'{\n  "format": "modern"\n}\n'
DONE = {"tool": "done", "arguments": {"summary": "Inspected the contract and decoder."}}


def _read(path, **arguments):
    return {"tool": "read", "arguments": {"path": path, **arguments}}


def _request_messages(state_message, *, tool_result="read result"):
    """A minimal StateFrame request with the frame immediately before its exchange."""
    return [
        {"role": "system", "content": "System instructions."},
        {"role": "user", "content": "Task: inspect the current request."},
        state_message,
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "read-call", "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "read-call", "name": "read",
         "content": tool_result},
    ]


def _repo(case):
    repo = case / "repo"
    repo.mkdir(parents=True)
    (repo / "app.py").write_text('FORMAT = "legacy"\n', encoding="utf-8")
    (repo / "contract.json").write_bytes(CONTRACT)
    return repo


def _env(case, checkout, switch=None):
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(("STATEFRAME_", "REACT_"))}
    env.update(
        HOME=str(case / "home"), STATEFRAME_HOME=str(case / "state"),
        PYTHONPATH=str(checkout / "src"), PYTHONDONTWRITEBYTECODE="1",
    )
    if switch is not None:
        env[SWITCH] = switch
    return env


def _serve_raw(script, *, capacity_retry):
    server = serve(script)
    bodies = []

    class CaptureRaw(server.RequestHandlerClass):
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return super().do_POST()
            assert not self.headers.get("Content-Encoding")
            stream = self.rfile
            payload = stream.read(int(self.headers["Content-Length"]))
            bodies.append(payload)
            # The first request on resume has already proved checkpoint byte parity.
            if capacity_retry and len(bodies) == 3:
                return self._send(400, {"error": {"code": "context_length_exceeded",
                                                 "message": "maximum context length is 32768 tokens"}})
            self.rfile = io.BytesIO(payload)
            try:
                super().do_POST()
            finally:
                self.rfile = stream

    server.RequestHandlerClass = CaptureRaw
    return server, bodies


def _run(case, checkout, script, *, switch=None, interrupt=False, capacity_retry=False,
         pre_resume_check=None):
    repo = _repo(case)
    env = _env(case, checkout, switch)
    server, bodies = _serve_raw(script, capacity_retry=capacity_retry)
    run_dir = case / "run"
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    common = ["--endpoint", endpoint, "--model", script.model, "--quiet", "--json"]
    command = [
        sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
        "--task", "Inspect app.py and contract.json to determine the decoder format.",
        "--target", "app.py", "--run-dir", str(run_dir), "--keep-checkpoints",
        "--max-turns", str(len(script.steps)), "--runner-kind", "generic",
        "--test-cmd", "true", *common,
    ]
    process = None
    try:
        if interrupt:
            from engine_run import start_monitored
            from fixtures.process_signals import record_identity, signal_run_verified

            launched = time.time()
            process = start_monitored(
                command, env=env, cwd=case, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, run_dir=run_dir, expected_seconds=scaled(180), step="R052 resume parity",
            )
            deadline = time.monotonic() + scaled(120)
            while not script.waiting.wait(0.25):
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "the in-flight request never reached the provider"
            (pid,) = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
            identity = record_identity(int(pid), launched_after=launched, command_fragment="_child.py")
            signal_run_verified(run_dir, int(pid), identity, signal.SIGKILL,
                                deadline=deadline, what="R052 in-flight request interruption")
            script.release.set()
            process.communicate(timeout=scaled(60))
            if pre_resume_check is not None:
                from _journal import measurement_events

                pre_resume_check(run_dir, bodies, measurement_events(run_dir))
            command = [sys.executable, "-m", "stateframe", "resume", str(run_dir), *common]
        result = subprocess.run(command, env=env, cwd=case, capture_output=True, text=True,
                                timeout=scaled(180), check=False)
    finally:
        if interrupt:
            script.release.set()
        if process is not None and process.poll() is None:
            from fixtures.process_signals import signal_run_owned

            signal_run_owned(run_dir, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="R052 test cleanup", require_checkpoint_saved=False, group=True)
            process.communicate(timeout=scaled(10))
        server.shutdown()
        server.server_close()
    assert (run_dir / "summary.json").is_file(), result.stdout + result.stderr
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["engine_result"] != "ENGINE_ERROR", (summary, result.stderr)
    script.assert_consumed()
    return run_dir, bodies


class _InterruptAfterFirstRead(Script):
    def __init__(self, steps):
        super().__init__({"steps": steps})
        self.waiting = threading.Event()
        self.release = threading.Event()

    def next_call(self, offered):
        if self.position == 1 and not self.release.is_set():
            self.waiting.set()
            self.release.wait(scaled(120))
            raise ScriptStepUnavailable("request interrupted before response")
        return super().next_call(offered)


def _typed_frame(raw):
    (state,) = [
        json.loads(message["content"].split("\n", 1)[1])["state"]["framefit5"]
        for message in json.loads(raw)["messages"]
        if (message.get("content") or "").startswith("[StateFrame]\n")
    ]
    return state


@pytest.mark.parametrize("switch", [None, "false"], ids=["unset", "explicit-false"])
def test_r052_still_viable_interpretations_off_request_bytes_match_pinned_099(
    tmp_path, tmp_path_factory, switch,
):
    from _journal import measurement_events

    steps = [_read("app.py"), DONE]
    baseline = pinned_tree(PINNED, tmp_path_factory)
    capacity_retry = switch == "false"
    original, expected = _run(tmp_path / "base", baseline, _InterruptAfterFirstRead(steps),
                              interrupt=True, capacity_retry=capacity_retry)
    current, actual = _run(tmp_path / "current", CHECKOUT, _InterruptAfterFirstRead(steps),
                           switch=switch, interrupt=True, capacity_retry=capacity_retry)
    assert len(expected) == len(actual) == 3 + capacity_retry
    assert expected[1] == expected[2]
    assert actual == expected
    for run_dir in (original, current):
        log = (run_dir / "engine.log").read_text(encoding="utf-8")
        assert '"kind":"checkpoint_verified"' in log.replace(" ", "")
        retries = [event for event in measurement_events(run_dir)
                   if event["kind"] == "capacity_retry"]
        assert len(retries) == int(capacity_retry)


def test_r052_still_viable_interpretations_switch_defaults_off(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    assert OPT_IN_SWITCHES[SWITCH] == ("off", "on")
    assert SWITCH not in resolve_opt_in_switches()
    assert not viable_alternatives_enabled()
    monkeypatch.setenv(SWITCH, "false")
    assert resolve_opt_in_switches()[SWITCH] == "off"
    assert not viable_alternatives_enabled()
    monkeypatch.setenv(SWITCH, "on")
    assert resolve_opt_in_switches()[SWITCH] == "on"
    assert viable_alternatives_enabled()


def test_r052_still_viable_interpretations_on_path_is_bounded_and_idempotent(tmp_path):
    from _journal import measurement_events

    steps = [
        _read("app.py", decision_delta={"viable_pair": PAIR}),
        _read("contract.json", max_lines=1),
        _read("app.py"),
        _read("contract.json"),
        _read("app.py"),
        _read("contract.json"),
        _read("app.py"),
        DONE,
    ]

    def require_interrupted_pair_receipt(_run_dir, wire_bodies, events):
        interrupted_hash = "sha256:" + hashlib.sha256(wire_bodies[1]).hexdigest()
        assert any(event.get("kind") == "request_submitted"
                   and isinstance(event.get("viable_alternatives_delivery"), dict)
                   and event["viable_alternatives_delivery"].get("payload_sha256")
                   == interrupted_hash
                   for event in events), (
            "the interrupted pair-bearing submission needs its receipt before resume"
        )

    run_dir, bodies = _run(tmp_path / "enabled", CHECKOUT, _InterruptAfterFirstRead(steps),
                           switch="on", interrupt=True,
                           pre_resume_check=require_interrupted_pair_receipt)
    assert len(bodies) == len(steps) + 1
    assert bodies[1] == bodies[2]
    interrupted_pair_sha256 = "sha256:" + hashlib.sha256(bodies[1]).hexdigest()
    bodies = bodies[:1] + bodies[2:]
    # The request after the one-line contract read must keep the pair visible
    # while that evidence is incomplete, even though only contract.json remains
    # the active read site at that point.
    assert _typed_frame(bodies[2])["viable_alternatives"] == PAIR
    (read_tool,) = [tool for tool in json.loads(bodies[0])["tools"]
                   if tool["function"]["name"] == "read"]
    assert "viable_pair" in read_tool["function"]["parameters"]["properties"][
        "decision_delta"]["properties"]
    for index in (1, 3):
        assert _typed_frame(bodies[index])["viable_alternatives"] == PAIR
    resolved = {**PAIR, "observed": "modern", "contradicted": "legacy",
                "evidence_revision": "sha256:" + hashlib.sha256(CONTRACT).hexdigest()}
    for index in (5, 7):
        assert _typed_frame(bodies[index])["viable_alternatives"] == resolved
    locations = (run_dir, *sorted((run_dir / "attempts").glob("*")))
    recorded_events = [(location, event) for location in locations
                       for event in measurement_events(location)]
    events = [event for _, event in recorded_events]
    delivery_kinds = {"request_final_wire_prepared", "request_submitted"}
    delivery_events_by_id = {}
    delivery_events_without_id = []
    for location, event in recorded_events:
        if event.get("kind") not in delivery_kinds:
            continue
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            semantic_event = {key: value for key, value in event.items()
                              if key not in {"previous_hash", "record_hash", "timestamp_ns"}}
            key = (location, event_id)
            previous = delivery_events_by_id.get(key)
            assert previous is None or previous[0] == semantic_event
            delivery_events_by_id.setdefault(key, (semantic_event, (location, event)))
        else:
            delivery_events_without_id.append((location, event))
    delivery_events = [*(row[1] for row in delivery_events_by_id.values()),
                       *delivery_events_without_id]
    delivery_receipts = [
        (location, event, event["viable_alternatives_delivery"])
        for location, event in delivery_events
        if event["kind"] == "request_submitted"
        and isinstance(event.get("viable_alternatives_delivery"), dict)
    ]
    assert delivery_receipts, "submitted viable pairs need a durable delivery receipt"
    interrupted_submissions = [
        (location, event) for location, event in delivery_events
        if location == run_dir and event["kind"] == "request_submitted"
        and any(prepared_location == location
                and prepared["kind"] == "request_final_wire_prepared"
                and prepared.get("request_id") == event.get("request_id")
                and prepared.get("payload_sha256") == interrupted_pair_sha256
                and prepared["sequence"] < event["sequence"]
                for prepared_location, prepared in delivery_events)
    ]
    assert len(interrupted_submissions) == 1
    original_location, original_submission = interrupted_submissions[0]
    assert any(location == original_location
               and submission.get("event_id") == original_submission.get("event_id")
               and receipt.get("payload_sha256") == interrupted_pair_sha256
               for location, submission, receipt in delivery_receipts), (
        "the interrupted pair-bearing submission must have its own durable delivery receipt"
    )
    (resolution,) = [event["update_result"] for event in events
                     if event.get("reason") == "viable_pair_resolved"]
    assert resolution["identity"] == {
        "path": "contract.json", "revision": resolved["evidence_revision"],
        "first_line": 1, "last_line": 3, "byte_start": 0, "byte_end": len(CONTRACT),
        "payload_sha256": hashlib.sha256(CONTRACT).hexdigest(), "lossy_span": None,
    }
    wire_pairs = [raw for raw in bodies if "viable_alternatives" in _typed_frame(raw)]
    assert all(raw.count(rb'\"viable_alternatives\"') == 1 for raw in wire_pairs)
    wire_pair_hashes = {
        "sha256:" + hashlib.sha256(raw).hexdigest() for raw in wire_pairs
    }
    pair_submission_ids = {
        (location, event["event_id"])
        for location, event in delivery_events
        if event["kind"] == "request_submitted"
        and event.get("payload_hash") in wire_pair_hashes
        and any(source == location
                and prepared["kind"] == "request_final_wire_prepared"
                and prepared.get("request_id") == event.get("request_id")
                and prepared.get("transport_attempt") == event.get("transport_attempt")
                and prepared.get("payload_sha256") == event.get("payload_hash")
                and prepared["sequence"] < event["sequence"]
                for source, prepared in delivery_events)
    }
    receipt_submission_ids = {
        (location, submission["event_id"])
        for location, submission, _receipt in delivery_receipts
    }
    assert receipt_submission_ids == pair_submission_ids
    submitted_ids = {event.get("request_id") for event in events
                     if event["kind"] == "request_submitted"}
    deliveries = [
        event for event in events
        if event["kind"] == "request_final_wire_prepared"
        and event.get("request_id") in submitted_ids
        and event.get("payload_sha256") in wire_pair_hashes
    ]
    delivery_hashes = {event["payload_sha256"] for event in deliveries}
    delivery_request_ids = {event["request_id"] for event in deliveries}
    assert delivery_hashes == wire_pair_hashes
    assert 0 < len(delivery_request_ids) <= len(delivery_hashes)
    receipt_keys = set()
    for location, submission, receipt in delivery_receipts:
        key = (location, submission["request_id"], submission["event_id"])
        assert key not in receipt_keys
        receipt_keys.add(key)
        request_preparations = [event for source, event in delivery_events
                                if source == location
                                if event["kind"] == "request_final_wire_prepared"
                                and event.get("request_id") == submission["request_id"]]
        (prepared,) = [event for event in request_preparations
                       if event.get("transport_attempt") == receipt["transport_attempt"]
                       and event.get("payload_sha256") == receipt["payload_sha256"]
                       and event["sequence"] < submission["sequence"]
                       and not any(other["sequence"] > event["sequence"]
                                   and other["sequence"] < submission["sequence"]
                                   for other in request_preparations)]
        assert prepared["sequence"] < submission["sequence"]
        matching_bodies = [raw for raw in bodies
                           if "sha256:" + hashlib.sha256(raw).hexdigest()
                           == receipt["payload_sha256"]]
        assert matching_bodies
        assert all(_typed_frame(raw)["viable_alternatives"] == receipt["pair"]
                   for raw in matching_bodies)
        panel = receipt["pair"]
        assert receipt["pair_sha256"] == "sha256:" + hashlib.sha256(
            json.dumps(panel, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        ).hexdigest()
        assert receipt["payload_sha256"] in wire_pair_hashes


def test_viable_pair_capacity_keeps_every_open_sibling(tmp_path):
    repo = _repo(tmp_path)
    store = ViableAlternatives(repo)
    runtime = SimpleNamespace()
    for key in ("format", "reader", "version", "protocol"):
        declaration = {**PAIR, "key": key}
        assert store.declare(declaration, runtime)["accepted"]
    original = store.snapshot()["pairs"]
    assert store.declare({**PAIR, "alternatives": ["modern", "legacy"]}, runtime)["reason"] == (
        "already_declared"
    )
    assert store.declare({**PAIR, "alternatives": ["current", "modern"]}, runtime)["reason"] == (
        "pair_conflict"
    )
    assert store.declare({**PAIR, "key": "fifth"}, runtime)["reason"] == "viable_pair_capacity"
    assert store.snapshot() == {"pairs": original, "capacity_misses": 1}


def test_viable_pair_delivery_matches_one_submission_per_transport_attempt(tmp_path):
    from kernel_host_authorship import HostAuthorship, author_row
    from kernel_stateframe_framefit5 import viable_pair_delivery_receipt

    class Journal:
        def __init__(self, events):
            self.events = list(events)

        def request_records(self, request_id):
            return tuple(event for event in self.events
                         if event.get("request_id") == request_id)

    repo = _repo(tmp_path)
    alternatives = ViableAlternatives(repo)
    runtime = SimpleNamespace(viable_alternatives=alternatives)
    assert alternatives.declare(PAIR, runtime)["accepted"]
    message = {
        "role": "user",
        "content": "[StateFrame]\n" + json.dumps({"state": {"framefit5": {
            "mode": "FRAME_ONLY", "next_site": {"path": "app.py"},
            "viable_alternatives": PAIR,
        }}}),
    }
    ledger = HostAuthorship(step=1)
    owned_state = author_row(ledger, "user", message["content"],
                             kind="state", site="test_fixture")
    wire_messages = _request_messages(owned_state)
    runtime._host_authorship = ledger
    request_id = "request-with-transport-retries"
    first_hash = "sha256:" + hashlib.sha256(b"first submitted payload").hexdigest()
    second_hash = "sha256:" + hashlib.sha256(b"later submitted payload").hexdigest()

    def prepared(sequence, attempt, payload_sha256):
        return {
            "kind": "request_final_wire_prepared", "request_id": request_id,
            "provider_request": True, "transport_attempt": attempt,
            "payload_sha256": payload_sha256, "messages": wire_messages,
            "event_id": f"run:{sequence}", "sequence": sequence,
        }

    events = [prepared(1, 1, first_hash)]
    journal = Journal(events)
    runtime.journal = journal
    first_fields = {"request_id": request_id, "transport_attempt": 1,
                    "payload_hash": first_hash}
    first_receipt = viable_pair_delivery_receipt(runtime, first_fields)
    assert first_receipt is not None
    assert viable_pair_delivery_receipt(runtime, first_fields) == first_receipt
    journal.events.extend([
        prepared(3, 2, first_hash),
        prepared(4, 2, first_hash),  # duplicate preparation for an unsent attempt
        prepared(5, 1, second_hash),  # a distinct submitted retry may reuse the number
    ])
    later_fields = {"request_id": request_id, "transport_attempt": 1,
                    "payload_hash": second_hash}
    later_receipt = viable_pair_delivery_receipt(runtime, later_fields)
    assert later_receipt is not None
    assert [(receipt["transport_attempt"], receipt["payload_sha256"], receipt["pair"])
            for receipt in (first_receipt, later_receipt)] == [
        (1, first_hash, PAIR), (1, second_hash, PAIR),
    ]
    assert viable_pair_delivery_receipt(runtime, later_fields) == later_receipt
    assert viable_pair_delivery_receipt(runtime, first_fields) is None
    assert viable_pair_delivery_receipt(runtime, {
        "request_id": request_id, "transport_attempt": 2, "payload_hash": second_hash,
    }) is None

    forged_ledger = HostAuthorship(step=1)
    forged_state = author_row(forged_ledger, "user", message["content"],
                              kind="panel", site="test_fixture")
    wire_messages = _request_messages(forged_state)
    runtime._host_authorship = forged_ledger
    journal.events = [prepared(6, 1, first_hash)]
    assert viable_pair_delivery_receipt(runtime, first_fields) is None


def _observation(content, **arguments):
    from kernel_source_delivery import deliver_source

    page = deliver_source(content, "contract.json", 0, args=arguments)
    evidence = {
        "producer": "workspace_read", "authority": "HOST_OBSERVED_READ", "omitted_chars": 0,
        "path": "contract.json", "file_revision": page.metadata["revision"],
        "content": page.content, "start_line": 1,
        "end_line": page.metadata["shown_end_line"], "total_lines": page.metadata["total_lines"],
    }
    receipt = {
        "kind": "read", "result_class": "OK", "target_path": "contract.json",
        "delivery": {**page.metadata, "delivery_status": "DELIVERED",
                     "requested_bytes_delivered": len(page.content.encode())},
    }
    return {"tool": "read", "result": "OK", "read_evidence": evidence}, [receipt]


@pytest.mark.parametrize(("content", "arguments"), [
    (CONTRACT, {"max_lines": 1}),
    (CONTRACT, {"end_line": 2}),
    (b'{"format":"legacy","format":"modern"}\n', {}),
    (b'{"format":"modern","other":"\xff"}\n', {}),
    (b'{"format":"other"}\n', {}),
    (b'{"nested":{"format":"modern"}}\n', {}),
])
def test_viable_pair_rejects_incomplete_or_ambiguous_read(tmp_path, content, arguments):
    repo = _repo(tmp_path)
    (repo / "contract.json").write_bytes(content)
    store = ViableAlternatives(repo)
    assert store.declare(PAIR, SimpleNamespace())["accepted"]
    assert list(store.observe(*_observation(content, **arguments))) == []
    assert store.for_edit(["app.py"]) == PAIR


def test_viable_pair_projects_on_open_evidence_site(tmp_path):
    from kernel_stateframe_framefit5 import _viable_pair_for_frame

    repo = _repo(tmp_path)
    store = ViableAlternatives(repo)
    assert store.declare(PAIR, SimpleNamespace())["accepted"]
    frame = {"next_site": {"path": "contract.json"},
             "open_sites": [{"path": "contract.json"}]}
    assert _viable_pair_for_frame(store, frame) == PAIR


def test_viable_pair_restores_both_siblings_when_evidence_revision_changes(tmp_path):
    repo = _repo(tmp_path)
    store = ViableAlternatives(repo)
    assert store.declare(PAIR, SimpleNamespace())["accepted"]
    read = _observation(CONTRACT)
    assert len(list(store.observe(*read))) == 1
    assert not list(store.observe(*read))
    assert store.for_edit(["app.py"])["observed"] == "modern"
    restored = ViableAlternatives(repo, store.snapshot())
    (repo / "contract.json").write_bytes(b'{"format":"legacy"}\n')
    assert not list(restored.observe(*read))
    assert restored.for_edit(["app.py"]) == PAIR


def test_viable_pair_leaves_required_source_and_reasoning_intact_under_pressure(
    tmp_path, monkeypatch,
):
    import kernel_request_budget
    from kernel_host_authorship import HostAuthorship, author_row
    from kernel_stateframe_framefit5 import _fit_viable_pair

    repo = _repo(tmp_path)
    capture = SimpleNamespace(
        request_id="request", _capacity_retry_original_body={"model": "scripted", "max_tokens": 10},
        final_tool_surface={"request_id": "request", "tools": []},
        request_budget_final_wire={"max_request_bytes": 1200, "max_prompt_tokens": 500},
    )
    store = ViableAlternatives(repo)
    ledger = HostAuthorship(step=1)
    runtime = SimpleNamespace(viable_alternatives=store, _active_capture=capture,
                              current_request_id="request", frame_ceiling=500,
                              _host_authorship=ledger)
    assert store.declare(PAIR, runtime)["accepted"]
    frame = {"framefit5": {"mode": "FRAME_ONLY", "next_site": {"path": "app.py"}},
             "reasoning": ["The format must agree in both files."]}
    state_message = {
        "role": "user", "content": "[StateFrame]\n" + json.dumps({"state": frame}),
    }
    state_message = author_row(ledger, "user", state_message["content"],
                               kind="state", site="test_fixture")
    messages = _request_messages(
        state_message, tool_result="required read bytes " + "x" * 750,
    )
    original = [message.copy() for message in messages]
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda *_: SimpleNamespace(measure=lambda body: (100, "tokenize")))
    _fit_viable_pair(runtime, messages, None)
    assert messages == original
    capture.request_budget_final_wire["max_request_bytes"] = 2000
    _fit_viable_pair(runtime, messages, None)
    with_panel = [message.copy() for message in messages]
    _fit_viable_pair(runtime, messages, None)
    assert messages == with_panel
    assert messages[:2] == original[:2]
    assert messages[3:] == original[3:]
    resulting_state = json.loads(messages[2]["content"].split("\n", 1)[1])["state"]
    assert resulting_state["framefit5"].pop("viable_alternatives") == PAIR
    assert resulting_state == frame


def test_viable_pair_fit_rejects_state_marker_authored_as_panel(tmp_path, monkeypatch):
    import kernel_request_budget
    from kernel_host_authorship import HostAuthorship, author_row
    from kernel_stateframe_framefit5 import _fit_viable_pair

    repo = _repo(tmp_path)
    capture = SimpleNamespace(
        request_id="request", _capacity_retry_original_body={"model": "scripted", "max_tokens": 10},
        final_tool_surface={"request_id": "request", "tools": []},
        request_budget_final_wire={"max_request_bytes": 12000, "max_prompt_tokens": 500},
    )
    store = ViableAlternatives(repo)
    ledger = HostAuthorship(step=1)
    frame = {"framefit5": {"mode": "FRAME_ONLY", "next_site": {"path": "app.py"}},
             "reasoning": ["The format must agree in both files."]}
    content = "[StateFrame]\n" + json.dumps({"state": frame})
    state_message = author_row(ledger, "user", content, kind="panel", site="test_fixture")
    messages = _request_messages(state_message)
    original = [message.copy() for message in messages]
    runtime = SimpleNamespace(
        viable_alternatives=store, _active_capture=capture, current_request_id="request",
        frame_ceiling=500, _host_authorship=ledger,
    )
    assert store.declare(PAIR, runtime)["accepted"]
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    monkeypatch.setattr(kernel_request_budget, "_request_token_meter",
                        lambda *_: SimpleNamespace(measure=lambda body: (100, "tokenize")))

    _fit_viable_pair(runtime, messages, None)

    assert messages == original
