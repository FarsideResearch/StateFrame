"""The wire recorder authorizes exactly live network sends."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import kernel_bounded_http
import pytest
from kernel_tool_tape import ToolReplayRefused
from stateframe import _child
from stateframe._child import SubframeBudgetExhausted, install_wire_recorder
from stateframe.parallel_resources import SendLedger
from stateframe.replay_tape import Replayer, build, normalized_digest


def _isolate_child_state(monkeypatch):
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))


def test_non_subframe_route_keeps_existing_unrecorded_transport_path(tmp_path, monkeypatch):
    _isolate_child_state(monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    primary = "http://127.0.0.1:8080/v1/chat/completions"
    alternate = "http://127.0.0.1:8081/v1/chat/completions"
    wire = b'{"model":"scripted","messages":[]}'
    calls = []

    def send(request, endpoint, key, **kwargs):
        calls.append((request, endpoint, key, kwargs))
        return b'{"choices":[{}]}'

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", send)
    install_wire_recorder(run, primary)
    assert kernel_bounded_http.request_bytes(wire, alternate, "") == b'{"choices":[{}]}'
    assert calls == [(wire, alternate, "", {})]
    assert list((run / "wire").iterdir()) == []


def test_live_sends_are_charged_and_replay_is_free(tmp_path, monkeypatch):
    _isolate_child_state(monkeypatch)
    run = tmp_path / "run" / "parallel" / "call" / "units" / "a"
    run.mkdir(parents=True)
    ledger = SendLedger(run / "sends-ledger.jsonl")
    wire = b'{"model":"scripted","messages":[]}'
    response = b'{"usage":{"prompt_tokens":7,"completion_tokens":3},"choices":[]}'
    network = []

    def send(request, endpoint, key, **kwargs):
        kwargs["before_attempt"](1)
        kwargs["record_attempts"]([{"http_status": 200}])
        network.append((request, endpoint, key, kwargs))
        return response

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", send)
    install_wire_recorder(run, "http://127.0.0.1:8080/v1/chat/completions",
                          send_ledger=ledger, send_cap=1)
    assert kernel_bounded_http.request_bytes(wire, "http://127.0.0.1:8080/v1/chat/completions", "") == response
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert [item["state"] for item in rows] == ["AUTHORIZED", "SETTLED"]
    assert rows[1]["usage"]["prompt_tokens"] == 7
    assert ledger.remaining(1) == 0
    with pytest.raises(SubframeBudgetExhausted, match="BUDGET_EXHAUSTED"):
        kernel_bounded_http.request_bytes(wire, "http://127.0.0.1:8080/v1/chat/completions", "")
    assert len(network) == 1

    recorded_request = run / "recorded.request.json"
    recorded_response = run / "recorded.response.json"
    recorded_request.write_bytes(wire)
    recorded_response.write_bytes(response)
    tape = [{"send": 1, "request": str(recorded_request), "response": str(recorded_response),
             "request_digest": normalized_digest(wire),
             "response_sha256": hashlib.sha256(response).hexdigest(),
             "endpoint_identity": {"url": "http://127.0.0.1:8080/v1", "model": "scripted"}}]
    replayer = Replayer(tape)
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", send)
    install_wire_recorder(run, "http://127.0.0.1:8080/v1/chat/completions", replayer,
                          send_ledger=ledger, send_cap=1)
    assert kernel_bounded_http.request_bytes(wire, "http://127.0.0.1:8080/v1/chat/completions", "") == response
    assert len(network) == 1
    assert [json.loads(line)["state"] for line in ledger.path.read_text().splitlines()] == [
        "AUTHORIZED", "SETTLED"]


def test_ambiguous_authorization_is_spent_on_resume(tmp_path, monkeypatch):
    _isolate_child_state(monkeypatch)
    run = tmp_path / "run" / "parallel" / "call" / "units" / "a"
    run.mkdir(parents=True)
    ledger = SendLedger(run / "sends-ledger.jsonl")
    ledger.authorize("prior-process:1:1:ambiguous")
    assert ledger.remaining(1) == 0
    network = []

    def send(_wire, _endpoint, _key, **kwargs):
        kwargs["before_attempt"](1)
        network.append("sent")
        return b'{}'

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", send)
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    install_wire_recorder(run, endpoint, send_ledger=SendLedger(ledger.path), send_cap=1)
    with pytest.raises(SubframeBudgetExhausted, match="BUDGET_EXHAUSTED"):
        kernel_bounded_http.request_bytes(b'{"model":"scripted","messages":[]}', endpoint, "")
    assert network == []
    assert [json.loads(line)["state"] for line in ledger.path.read_text().splitlines()] == ["AUTHORIZED"]


def test_routed_subframe_send_is_charged_and_recorded(tmp_path, monkeypatch):
    _isolate_child_state(monkeypatch)
    run = tmp_path / "run" / "parallel" / "call" / "units" / "a"
    run.mkdir(parents=True)
    ledger = SendLedger(run / "sends-ledger.jsonl")
    primary = "http://127.0.0.1:8080/v1/chat/completions"
    routed = "http://127.0.0.1:8081/v1/chat/completions"
    wire = b'{"model":"routed-model","messages":[]}'
    calls = []

    def send(request, endpoint, key, **kwargs):
        kwargs["before_attempt"](1)
        kwargs["record_attempts"]([{"http_status": 200}])
        calls.append((request, endpoint, key))
        return b'{"usage":{"prompt_tokens":3,"completion_tokens":2},"choices":[{}]}'

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", send)
    install_wire_recorder(run, primary, send_ledger=ledger, send_cap=1)
    kernel_bounded_http.request_bytes(wire, routed, "")
    assert calls == [(wire, routed, "")]
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert [row["state"] for row in rows] == ["AUTHORIZED", "SETTLED"]
    assert json.loads((run / "wire/0001.endpoint.json").read_bytes()) == {
        "url": "http://127.0.0.1:8081/v1", "model": "routed-model"}
    with pytest.raises(SubframeBudgetExhausted, match="BUDGET_EXHAUSTED"):
        kernel_bounded_http.request_bytes(wire, routed, "")
    assert len(calls) == 1
    assert (run / "wire/0002.error.txt").read_text() == "SubframeBudgetExhausted\n"

    tape = build(run / "wire", run / "sends.jsonl")
    assert tape[0]["endpoint_identity"] == {"url": "http://127.0.0.1:8081/v1",
                                             "model": "routed-model"}
    resumed = tmp_path / "resumed"
    resumed.mkdir()
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", send)
    install_wire_recorder(resumed, primary, Replayer(tape),
                          send_ledger=SendLedger(resumed / "sends-ledger.jsonl"), send_cap=1)
    with pytest.raises(ToolReplayRefused, match="subframe route"):
        kernel_bounded_http.request_bytes(wire, primary, "")
    assert len(calls) == 1


@pytest.mark.parametrize("failure_kind", ["transport", "budget"])
def test_settlement_failure_preserves_primary_send_error(tmp_path, monkeypatch, capsys, failure_kind):
    _isolate_child_state(monkeypatch)
    run = tmp_path / "run" / "parallel" / "call" / "units" / "a"
    run.mkdir(parents=True)
    ledger = SendLedger(run / "sends-ledger.jsonl")

    def fail_settlement(_send_id, _usage):
        raise OSError("private ledger detail must not replace the send error")

    monkeypatch.setattr(ledger, "settle", fail_settlement)
    transport_error = RuntimeError("transport failed")
    transport_error.error_detail = "connection reset"
    budget_errors = []

    def send(_wire, _endpoint, _key, **kwargs):
        kwargs["before_attempt"](1)
        kwargs["record_attempts"]([{"http_status": 503}])
        if failure_kind == "transport":
            raise transport_error
        try:
            kwargs["before_attempt"](2)
        except SubframeBudgetExhausted as error:
            budget_errors.append(error)
            raise
        raise AssertionError("the second physical attempt should exceed the cap")

    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", send)
    install_wire_recorder(run, endpoint, send_ledger=ledger, send_cap=1)

    expected_error = RuntimeError if failure_kind == "transport" else SubframeBudgetExhausted
    with pytest.raises(expected_error) as failure:
        kernel_bounded_http.request_bytes(b'{"model":"scripted","messages":[]}', endpoint, "")

    if failure_kind == "transport":
        assert failure.value is transport_error
        assert (run / "wire/0001.error.txt").read_text() == "RuntimeError: connection reset\n"
    else:
        assert budget_errors == [failure.value]
        assert (run / "wire/0001.error.txt").read_text() == "SubframeBudgetExhausted\n"
    assert (run / "wire/0001.settle-error.txt").read_text() == "OSError\n"
    assert ledger.remaining(1) == 0
    assert [json.loads(line)["state"] for line in ledger.path.read_text().splitlines()] == ["AUTHORIZED"]

    events = [json.loads(line.removeprefix("[sf] ")) for line in capsys.readouterr().out.splitlines()
              if line.startswith("[sf] ")]
    assert any(event["kind"] == "send_settlement_error" and event["error"] == "OSError"
               for event in events)
    if failure_kind == "transport":
        assert any(event["kind"] == "send_error" and event["error"] == "RuntimeError"
                   for event in events)
        send_records = [json.loads(line) for line in (run / "sends.jsonl").read_text().splitlines()]
        assert send_records[0]["error"] == "RuntimeError"


def test_concurrent_sends_get_distinct_wire_records(tmp_path, monkeypatch):
    _isolate_child_state(monkeypatch)
    run = tmp_path / "run" / "parallel" / "call" / "units" / "a"
    run.mkdir(parents=True)
    ledger = SendLedger(run / "sends-ledger.jsonl")
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    send_count = 8
    physical_send_barrier = threading.Barrier(send_count)

    def send(_wire, _endpoint, _key, **kwargs):
        kwargs["before_attempt"](1)
        kwargs["record_attempts"]([{"http_status": 200}])
        physical_send_barrier.wait(timeout=10)
        return b'{"usage":{"prompt_tokens":1,"completion_tokens":1},"choices":[]}'

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", send)
    install_wire_recorder(run, endpoint, send_ledger=ledger, send_cap=send_count)
    requests = [json.dumps({"model": f"model-{index}", "messages": []}).encode()
                for index in range(send_count)]

    with ThreadPoolExecutor(max_workers=send_count) as executor:
        responses = list(executor.map(
            lambda wire: kernel_bounded_http.request_bytes(wire, endpoint, ""), requests))

    assert len(responses) == send_count
    request_files = sorted((run / "wire").glob("*.request.json"))
    assert [path.name for path in request_files] == [f"{index:04d}.request.json"
                                                       for index in range(1, send_count + 1)]
    for request_file in request_files:
        index = request_file.name.split(".", 1)[0]
        wire = json.loads(request_file.read_bytes())
        endpoint_identity = json.loads((run / "wire" / f"{index}.endpoint.json").read_bytes())
        assert endpoint_identity["model"] == wire["model"]
        assert (run / "wire" / f"{index}.response.json").is_file()

    send_records = [json.loads(line) for line in (run / "sends.jsonl").read_text().splitlines()]
    assert sorted(record["send"] for record in send_records) == list(range(1, send_count + 1))
    authorized_ids = [json.loads(line)["send_id"] for line in ledger.path.read_text().splitlines()
                      if json.loads(line)["state"] == "AUTHORIZED"]
    assert len(authorized_ids) == send_count
    assert len(set(authorized_ids)) == send_count


def test_send_cap_is_checked_with_authorization_under_one_ledger_lock(tmp_path):
    path = tmp_path / "run" / "parallel" / "call" / "units" / "a" / "sends-ledger.jsonl"
    path.parent.mkdir(parents=True)
    first = SendLedger(path)
    second = SendLedger(path)
    assert first.authorize_with_cap("first:1", 1)
    assert not second.authorize_with_cap("second:1", 1)
    assert not first.authorize_with_cap("first:1", 1)
    assert second.remaining(1) == 0
    assert [json.loads(line)["send_id"] for line in path.read_text().splitlines()] == ["first:1"]


@pytest.mark.host_sandbox
def test_model_router_alternate_scripted_endpoint_uses_subframe_wire_and_cap(tmp_path, monkeypatch):
    _isolate_child_state(monkeypatch)
    from kernel_stateframe_model_router import (
        EndpointConfig,
        ModelRouter,
        ModelRouterConfig,
    )

    class ScriptedEndpoint(BaseHTTPRequestHandler):
        requests = []

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            self.requests.append(json.loads(body))
            response = b'{"usage":{"prompt_tokens":5,"completion_tokens":2},"choices":[{}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedEndpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        alternate = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        primary = "http://127.0.0.1:8080/v1/chat/completions"
        router = ModelRouter(mode="on", config=ModelRouterConfig(endpoints=(
            EndpointConfig("alternate", "scripted-alternate", role="moe", url=alternate,
                           allow_empty_key=True, context_tokens=65536),)), base_model="scripted",
            replay_enabled=False)
        capture = SimpleNamespace(step=1)
        prepared = router.prepare_call(capture, {"model": "scripted", "messages": []}, primary, "")
        assert (prepared.endpoint, prepared.body["model"]) == (alternate, "scripted-alternate")

        run = tmp_path / "run" / "parallel" / "call" / "units" / "a"
        run.mkdir(parents=True)
        ledger = SendLedger(run / "sends-ledger.jsonl")
        request = json.dumps(prepared.body).encode()
        original_request = kernel_bounded_http.request_bytes
        monkeypatch.setattr(kernel_bounded_http, "request_bytes", original_request)
        install_wire_recorder(run, primary, send_ledger=ledger, send_cap=1)
        kwargs = {"deadline": time.monotonic() + 20, "socket_timeout": 5}
        response = kernel_bounded_http.request_bytes(request, prepared.endpoint, prepared.key, **kwargs)
        assert json.loads(response)["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}
        assert [row["state"] for row in map(json.loads, ledger.path.read_text().splitlines())] == [
            "AUTHORIZED", "SETTLED"]
        assert json.loads((run / "wire/0001.endpoint.json").read_bytes()) == {
            "url": f"http://127.0.0.1:{server.server_port}/v1", "model": "scripted-alternate"}
        assert ScriptedEndpoint.requests[0]["model"] == "scripted-alternate"
        with pytest.raises(SubframeBudgetExhausted, match="BUDGET_EXHAUSTED"):
            kernel_bounded_http.request_bytes(request, prepared.endpoint, prepared.key, **kwargs)
        assert len(ScriptedEndpoint.requests) == 1
        tape = build(run / "wire", run / "sends.jsonl")
        assert tape[0]["endpoint_identity"] == {
            "url": f"http://127.0.0.1:{server.server_port}/v1", "model": "scripted-alternate"}
        resumed = tmp_path / "replay" / "parallel" / "call" / "units" / "a"
        resumed.mkdir(parents=True)
        monkeypatch.setattr(kernel_bounded_http, "request_bytes", original_request)
        install_wire_recorder(resumed, primary, Replayer(tape),
                              send_ledger=SendLedger(resumed / "sends-ledger.jsonl"), send_cap=1)
        with pytest.raises(ToolReplayRefused, match="subframe route"):
            kernel_bounded_http.request_bytes(request, primary, prepared.key, **kwargs)
        assert len(ScriptedEndpoint.requests) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("cap, expected_requests, expected_states", [
    (2, 2, ["AUTHORIZED", "AUTHORIZED", "SETTLED", "SETTLED"]),
    (1, 1, ["AUTHORIZED", "SETTLED"]),
])
def test_gzip_415_retry_authorizes_each_physical_attempt(
        tmp_path, monkeypatch, cap, expected_requests, expected_states):
    _isolate_child_state(monkeypatch)
    class Handler(BaseHTTPRequestHandler):
        requests = []

        def do_POST(self):
            self.requests.append(self.headers.get("Content-Encoding"))
            size = int(self.headers["Content-Length"])
            self.rfile.read(size)
            if self.headers.get("Content-Encoding") == "gzip":
                self.send_response(415)
                body = b"gzip unsupported"
            else:
                self.send_response(200)
                body = b'{"usage":{"prompt_tokens":7,"completion_tokens":3},"choices":[]}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        run = tmp_path / "run" / "parallel" / "call" / "units" / "a"
        run.mkdir(parents=True)
        ledger = SendLedger(run / "sends-ledger.jsonl")
        monkeypatch.setenv("STATEFRAME_HTTP_GZIP", "on")
        original_request = kernel_bounded_http.request_bytes
        monkeypatch.setattr(kernel_bounded_http, "request_bytes", original_request)
        install_wire_recorder(run, f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                              send_ledger=ledger, send_cap=cap)
        wire = b'{"model":"scripted","messages":[]}' + b" " * 32768
        request = kernel_bounded_http.request_bytes
        kwargs = {"deadline": time.monotonic() + 20, "socket_timeout": 5}
        if cap == 1:
            with pytest.raises(SubframeBudgetExhausted, match="BUDGET_EXHAUSTED"):
                request(wire, f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "", **kwargs)
        else:
            response = request(wire, f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "", **kwargs)
            assert json.loads(response)["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}
        rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
        assert [row["state"] for row in rows] == expected_states
        assert len({row["send_id"] for row in rows if row["state"] == "AUTHORIZED"}) == cap
        assert Handler.requests == ["gzip", None][:expected_requests]
        assert ledger.remaining(cap) == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.host_sandbox
def test_timed_out_physical_attempt_remains_spent_without_settlement(tmp_path, monkeypatch):
    _isolate_child_state(monkeypatch)
    # Without a user queue cap a silent but alive provider has no total cap (see test_queue_wait_user_only).
    # This case times out on purpose, so it sets the user cap that ends the wait before the attempt deadline.
    monkeypatch.setenv("STATEFRAME_QUEUE_WAIT_MAX_S", "0.5")
    class SlowHandler(BaseHTTPRequestHandler):
        requests = 0

        def do_POST(self):
            type(self).requests += 1
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(2)

        def log_message(self, _format, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        run = tmp_path / "run" / "parallel" / "call" / "units" / "a"
        run.mkdir(parents=True)
        ledger = SendLedger(run / "sends-ledger.jsonl")
        original_request = kernel_bounded_http.request_bytes
        monkeypatch.setattr(kernel_bounded_http, "request_bytes", original_request)
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        install_wire_recorder(run, endpoint, send_ledger=ledger, send_cap=1)
        with pytest.raises(TimeoutError, match="HTTP absolute deadline exceeded"):
            kernel_bounded_http.request_bytes(
                b'{"model":"scripted","messages":[]}', endpoint, "",
                deadline=time.monotonic() + 1, socket_timeout=5, liveness_timeout=10)
        assert SlowHandler.requests == 1
        assert [json.loads(line)["state"] for line in ledger.path.read_text().splitlines()] == [
            "AUTHORIZED"]
        assert ledger.remaining(1) == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.host_sandbox
def test_real_subframe_child_records_f4_policy_ledger_and_releases_slot(tmp_path, monkeypatch):
    from test_parallel_host import (
        _worker_diagnostics,
        test_real_scripted_worker_submission_seals_recorded_bytes_and_receipt,
    )

    from stateframe import parallel_resources
    monkeypatch.setattr(
        parallel_resources, "LEASE_ROOT",
        tmp_path / "child-home" / ".stateframe" / "slot-leases")
    worker = tmp_path / "run" / "parallel" / "call" / "units" / "a"
    try:
        test_real_scripted_worker_submission_seals_recorded_bytes_and_receipt(tmp_path)
        prepared = json.loads((worker / "spec.json").read_bytes())
        assert prepared["profile"] == "r2hc3"
        request = json.loads((worker / "wire" / "0001.request.json").read_bytes())
        names = {item["function"]["name"] for item in request["tools"]}
        assert "subframe_blocked" in names and "parallel" not in names
        ledger = SendLedger(worker / "sends-ledger.jsonl")
        rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
        authorized = {row["send_id"] for row in rows if row["state"] == "AUTHORIZED"}
        settled = {row["send_id"] for row in rows if row["state"] == "SETTLED"}
        assert authorized and authorized == settled
        assert (Path(prepared["parallel_permit_root"]) / "0.lock").is_file()
        historical_slot = json.loads((worker / "parallel-slot.json").read_bytes())["slot"]
        lease = parallel_resources.lease_slot(
            prepared["parallel_parent_endpoint"], [historical_slot])
        assert lease is not None and lease.slot == historical_slot
        lease.close()
    except Exception as error:
        pytest.fail(f"{error}\n{_worker_diagnostics(worker)}", pytrace=False)


@pytest.mark.parametrize("interrupted", [False, True])
def test_subframe_endpoint_identity_and_stable_wire_reserves_coexist(tmp_path, monkeypatch, interrupted):
    from kernel_measurement import ACTIVE_REQUEST
    from stateframe.replay_tape import pending_request

    _isolate_child_state(monkeypatch)
    run = tmp_path / "run/parallel/call/units/a"
    run.mkdir(parents=True)
    endpoint = "http://127.0.0.1:8080/v1/chat/completions"
    wire = b'{"model":"scripted","messages":[]}'
    response = b'{"choices":[{"message":{"content":"ok"}}]}'
    reserves = {"reserves": [[28, 256]], "trim_count": 1, "epoch": 1}
    identity = {"url": "http://127.0.0.1:8080/v1", "model": "scripted"}

    class Interrupted(BaseException):
        pass

    def transport(sent, *_args, **kwargs):
        assert sent == wire
        assert json.loads((run / "wire/0001.stable-wire.json").read_bytes()) == reserves
        assert json.loads((run / "wire/0001.endpoint.json").read_bytes()) == identity
        if interrupted:
            raise Interrupted
        kwargs["before_attempt"](1)
        kwargs["record_attempts"]([{"http_status": 200}])
        return response

    monkeypatch.setattr(kernel_bounded_http, "request_bytes", transport)
    ledger = SendLedger(run / "sends-ledger.jsonl")
    install_wire_recorder(run, endpoint, send_ledger=ledger, send_cap=1)
    token = ACTIVE_REQUEST.set(SimpleNamespace(_stable_wire_reserves_record=reserves))
    try:
        if interrupted:
            with pytest.raises(Interrupted):
                kernel_bounded_http.request_bytes(wire, endpoint, "")
        else:
            assert kernel_bounded_http.request_bytes(wire, endpoint, "") == response
    finally:
        ACTIVE_REQUEST.reset(token)
    record = (pending_request(run / "wire") if interrupted
              else build(run / "wire", run / "sends.jsonl")[0])
    assert record["stable_wire_reserves"] == reserves
    assert record["endpoint_identity"] == identity
