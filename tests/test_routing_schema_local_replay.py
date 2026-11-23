"""In-process loopback proof for routing schema 1 parity and schema 2 replay."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

import kernel_context_capacity
import kernel_decide2
import kernel_endpoint_size
import kernel_final_wire_allocation
import kernel_frame_routes as frame_routes
import kernel_measurement
import kernel_request_diet
import kernel_route_decision
from stateframe import routes, runner
from stateframe.endpoint import Endpoint


ROUTING_SCHEMA_ENV = "STATEFRAME_ROUTING_SCHEMA"
BASELINE = {
    "endpoint": "http://127.0.0.1:1/v1",
    "model": "base-model",
    "api_key_env": None,
    "temperature": 1.0,
    "max_tokens": 24000,
    "timeout_s": 1800.0,
    "slot": None,
    "context_tokens": 65536,
    "allow_remote": False,
}
PARITY_VOLATILE_FIELDS = frozenset({
    "timestamp_ns", "previous_hash", "record_hash", "run_id", "task_id", "event_id",
    "sequence", "tokenize_latency_ms", "tokenize_route", "tokenizer_identity",
})


class RecordingDeployment:
    """A tiny OpenAI-compatible loopback server with separate chat/tokenize captures."""

    def __init__(self, name: str, *, context_tokens: int = 65536,
                 model_ids: tuple[str, ...] = ("base-model",)) -> None:
        self.name = name
        self.context_tokens = context_tokens
        self.model_ids = model_ids
        self.requests: list[tuple[str, str, bytes]] = []
        self.posts: list[tuple[str, bytes]] = []
        self.chat_wires: list[bytes] = []
        self.chat_bodies: list[dict[str, object]] = []
        self.tokenize_wires: list[bytes] = []
        deployment = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: object) -> None:
                return

            def _reply(self, status: int, payload: object) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                deployment.requests.append(("GET", self.path, b""))
                if self.path.rstrip("/") == "/v1/models":
                    self._reply(200, {"object": "list", "data": [
                        {"id": model, "object": "model"} for model in deployment.model_ids
                    ]})
                elif self.path.rstrip("/") == "/slots":
                    self._reply(200, [{"id": 0, "n_ctx": deployment.context_tokens,
                                       "is_processing": False}])
                elif self.path.rstrip("/") == "/props":
                    self._reply(200, {"default_generation_settings": {
                        "n_ctx": deployment.context_tokens,
                    }})
                else:
                    self._reply(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                deployment.requests.append(("POST", self.path, raw))
                deployment.posts.append((self.path, raw))
                if self.path.rstrip("/").endswith("/tokenize"):
                    deployment.tokenize_wires.append(raw)
                    self._reply(200, {"tokens": [1, 2, 3]})
                    return
                if self.path.rstrip("/") == "/v1/chat/completions":
                    deployment.chat_wires.append(raw)
                    deployment.chat_bodies.append(json.loads(raw))
                    self._reply(200, {"choices": [{"message": {
                        "role": "assistant", "content": "recorded",
                    }}]})
                    return
                self._reply(404, {"error": {"message": "not found"}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    @property
    def chat_endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def tokenize_endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/tokenize"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def __enter__(self) -> RecordingDeployment:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class MemoryJournal:
    """The decision and capacity events created by this test stay in process."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, kind: str, **fields: object) -> None:
        self.events.append({"kind": kind, **fields})

    def records(self, *, rebuild: bool = True):
        del rebuild
        yield from self.events

    def of_kind(self, kind: str, request_id: str | None = None) -> list[dict[str, object]]:
        return [event for event in self.events
                if event.get("kind") == kind
                and (request_id is None or event.get("request_id") == request_id)]


class Capture:
    def __init__(self, request_id: str, journal: MemoryJournal, *,
                 frame_route: object | None = None, model_router: object | None = None,
                 purpose: str | None = None) -> None:
        self.request_id = request_id
        self.step = 1
        self.journal = journal
        self.frame_route = frame_route
        self.model_router = model_router
        self.call_purpose = purpose
        self.model_router_call_purpose = purpose
        self.model_router_child_purpose = purpose


@pytest.fixture(autouse=True)
def isolated_route_state(monkeypatch):
    names = (
        ROUTING_SCHEMA_ENV, "STATEFRAME_MODEL_NAME", "DS_MODEL", "STATEFRAME_DECIDE2_MODEL",
        "STATEFRAME_WORK_FRAME_MODEL", "STATEFRAME_ROUTE_READ_ENDPOINT",
        "STATEFRAME_ROUTE_READ_MODEL", "LLAMA_CPP_TOKENIZE_URL", "STATEFRAME_N_CTX",
        "STATEFRAME_CONTEXT_TOKENS", "STATEFRAME_MIN_OUTPUT_TOKENS", "STATEFRAME_ROUTE_SCOPE",
        "STATEFRAME_ROUTE_UNIT_CLASS", "STATEFRAME_ROUTE_SLOT",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
                 "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STATEFRAME_CONTEXT_PREFLIGHT", "on")
    monkeypatch.setattr(kernel_endpoint_size, "ENDPOINT_SIZE",
                        kernel_endpoint_size.resolve_endpoint_size({}))
    kernel_context_capacity._cache.clear()
    frame_routes.install(None)
    yield
    frame_routes.install(None)
    kernel_context_capacity._cache.clear()


def _baseline_for(deployment: RecordingDeployment) -> dict[str, object]:
    return {**BASELINE, "endpoint": deployment.base_url}


def _install_routes(default: RecordingDeployment, *entries: str,
                    environ: dict[str, str] | None = None) -> dict[str, object]:
    baseline = _baseline_for(default)
    table = routes.resolve(baseline, cli=list(entries), environ=environ or {})
    frame_routes.install(table.public(), baseline=baseline)
    return baseline


def _body(model: str, content: str = "local route request", *, max_tokens: int = 256
          ) -> dict[str, object]:
    return {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens}


def _micro_body() -> dict[str, object]:
    request = kernel_decide2.DecisionRequest(
        decision_id="decision-1", question_kind="relevance",
        question="Is the source relevant?", options=("yes", "no", "ABSTAIN"),
    )
    return kernel_decide2.build_micro_request(
        kernel_decide2.build_frame(request), rung="L1",
        tool_surface=({"type": "function", "function": {"name": "read"}},),
    )


def _work_body() -> dict[str, object]:
    request = kernel_decide2.DecisionRequest(
        decision_id="decision-2", question_kind="relevance",
        question="Is the source relevant?", options=("yes", "no", "ABSTAIN"),
    )
    return kernel_decide2.build_work_request(request, reason="low confidence", max_tokens=64)


def _wire_and_dispatch(capture: Capture, body: dict[str, object], *, fallback_endpoint: str,
                       fallback_key: str, expected_endpoint: str, expected_model: str,
                       expected_source: str, expected_role: str | None,
                       expected_chat_server: RecordingDeployment,
                       expected_tokenizer_server: RecordingDeployment,
                       expected_context: int,
                       deployments: tuple[RecordingDeployment, ...],
                       recovery_prepared: bool = False,
                       expect_capacity_event: bool = False) -> tuple[bytes, object]:
    request_before = {deployment.name: len(deployment.requests)
                      for deployment in deployments}
    route = capture.frame_route
    if route is not None:
        capture._tokenize_endpoint = str(route.settings.get("tokenizer") or route.settings["endpoint"])
        frame_routes.apply_body(body, route, default_endpoint=fallback_endpoint)

    decision, body, endpoint, key = kernel_route_decision.resolve_route_decision(
        capture, body, fallback_endpoint, fallback_key,
        recovery_prepared=recovery_prepared,
    )

    tokenizer_binding = kernel_final_wire_allocation.routed_tokenize_endpoint(capture) or endpoint
    tokenizer_capture = SimpleNamespace(
        frame_route=route,
        _tokenize_endpoint=tokenizer_binding,
    )
    tokenizer_url = kernel_request_diet.resolve_tokenize_url(
        capture=tokenizer_capture, endpoint=tokenizer_binding,
    )
    assert tokenizer_url == expected_tokenizer_server.tokenize_endpoint
    tokenizer = kernel_request_diet.RequestTokenMeter(
        endpoint=tokenizer_url, route_allow_remote=False,
    )
    tokenizer_before = {deployment.name: len(deployment.tokenize_wires)
                        for deployment in deployments}
    _, measurement = tokenizer.measure(body)
    assert measurement == "tokenize"

    live_context = kernel_context_capacity.probe_n_ctx(endpoint, key)
    assert live_context is not None
    assert live_context == expected_chat_server.context_tokens
    route_context = route.settings.get("context_tokens") if route is not None else live_context
    assert min(live_context, route_context) == expected_context
    kernel_context_capacity.enforce_context_capacity(capture, body, endpoint=endpoint, key=key)

    wire = kernel_measurement._serialize_final_request_body(body, None, capture.request_id)
    before = {deployment.name: len(deployment.chat_wires) for deployment in deployments}
    response = kernel_measurement._request_bytes_for_capture(
        capture, wire, endpoint, key,
        deadline=time.monotonic() + 5,
        socket_timeout=5,
        route_allow_remote=False,
    )
    assert json.loads(response)["choices"][0]["message"]["content"] == "recorded"

    arrivals = [deployment.name for deployment in deployments
                if len(deployment.chat_wires) == before[deployment.name] + 1]
    assert arrivals == [expected_chat_server.name]
    assert all(len(deployment.chat_wires) - before[deployment.name] ==
               (1 if deployment is expected_chat_server else 0)
               for deployment in deployments)
    assert len(expected_tokenizer_server.tokenize_wires) == (
        tokenizer_before[expected_tokenizer_server.name] + 1)
    allowed_servers = {expected_chat_server, expected_tokenizer_server}
    request_deltas = {
        deployment: deployment.requests[request_before[deployment.name]:]
        for deployment in deployments
    }
    assert all(not requests or deployment in allowed_servers
               for deployment, requests in request_deltas.items())
    unexpected_requests = [
        (deployment.name, method, path)
        for deployment, requests in request_deltas.items()
        for method, path, _ in requests
        if not (
            (method == "POST" and path.rstrip("/").endswith("/tokenize")
             and deployment is expected_tokenizer_server)
            or (method == "POST" and path.rstrip("/") == "/v1/chat/completions"
                and deployment is expected_chat_server)
            or (method == "GET" and path.rstrip("/") in {"/slots", "/props"}
                and deployment in allowed_servers)
        )
    ]
    assert not unexpected_requests, (unexpected_requests, endpoint, tokenizer_url)
    assert json.loads(expected_chat_server.chat_wires[-1]) == json.loads(wire)
    assert json.loads(wire)["model"] == expected_model
    assert (decision.model, decision.endpoint, decision.source, decision.role) == (
        expected_model, expected_endpoint, expected_source, expected_role)

    selected_events = capture.journal.of_kind("route_decision", capture.request_id)
    assert selected_events
    assert {("model", event.get("model")) for event in selected_events} == {
        ("model", expected_model),
    }
    assert {("endpoint", event.get("endpoint")) for event in selected_events} == {
        ("endpoint", expected_endpoint),
    }
    assert {("role", event.get("role")) for event in selected_events} == {
        ("role", expected_role),
    }
    frame_events = capture.journal.of_kind("frame_route", capture.request_id)
    if expected_source == "static_route":
        assert len(frame_events) == 1
        frame_event = frame_events[0]
        assert (frame_event.get("role"), frame_event.get("model"),
                frame_event.get("endpoint_host"), frame_event.get("dispatch")) == (
            expected_role, expected_model, urlsplit(expected_endpoint).hostname, False)

    capacity_events = capture.journal.of_kind("context_capacity_shrunk", capture.request_id)
    if expect_capacity_event:
        assert capacity_events
        assert all(event.get("n_ctx") == expected_context for event in capacity_events)
    return wire, decision


def test_unset_and_one_keep_request_bytes_and_journal_identical(monkeypatch):
    with RecordingDeployment("default", model_ids=("base-model", "micro-override")) as default:
        deployments = (default,)
        captures: list[tuple[bytes, list[dict[str, object]]]] = []
        for schema in (None, "1"):
            if schema is None:
                monkeypatch.delenv(ROUTING_SCHEMA_ENV, raising=False)
            else:
                monkeypatch.setenv(ROUTING_SCHEMA_ENV, schema)
            monkeypatch.setenv("DS_MODEL", "configured-model")
            monkeypatch.setenv("STATEFRAME_DECIDE2_MODEL", "micro-override")
            journal = MemoryJournal()
            capture = Capture("parity-request", journal)
            wire, _ = _wire_and_dispatch(
                capture, _micro_body(),
                fallback_endpoint=default.chat_endpoint, fallback_key="",
                expected_endpoint=default.chat_endpoint,
                expected_model="decision-tier", expected_source="default", expected_role=None,
                expected_chat_server=default, expected_tokenizer_server=default,
                expected_context=65536, deployments=deployments,
            )
            normalized_journal = [
                {key: value for key, value in event.items()
                 if key not in PARITY_VOLATILE_FIELDS}
                for event in journal.events
            ]
            captures.append((wire, normalized_journal))
        assert captures[0][0] == captures[1][0]
        assert captures[0][1] == captures[1][1]


def test_schema_two_default_route_uses_one_decided_loopback_deployment(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "2")
    with RecordingDeployment("default") as default:
        journal = MemoryJournal()
        capture = Capture("default-request", journal)
        _wire_and_dispatch(
            capture, _body("base-model"),
            fallback_endpoint=default.chat_endpoint, fallback_key="",
            expected_endpoint=default.chat_endpoint, expected_model="base-model",
            expected_source="default", expected_role=None,
            expected_chat_server=default, expected_tokenizer_server=default,
            expected_context=65536, deployments=(default,),
        )


def test_per_operation_consultation_and_child_override_use_their_decisions(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "2")
    monkeypatch.setenv("STATEFRAME_DECIDE2_MODEL", "consult-model")
    monkeypatch.setenv("STATEFRAME_WORK_FRAME_MODEL", "child-model")
    with (RecordingDeployment("default") as default,
          RecordingDeployment("consultation", model_ids=("consult-model",)) as consultation,
          RecordingDeployment("child", model_ids=("child-model",)) as child):
        baseline = _install_routes(
            default,
            f"decide2.endpoint={consultation.base_url}", "decide2.model=consult-model",
            f"unit.endpoint={child.base_url}", "unit.model=child-model",
        )
        deployments = (default, consultation, child)
        consultation_route = frame_routes.resolve("decide2")
        consultation_journal = MemoryJournal()
        _wire_and_dispatch(
            Capture("consultation-request", consultation_journal, frame_route=consultation_route,
                    purpose="decide2"),
            _micro_body(), fallback_endpoint=f"{baseline['endpoint']}/chat/completions",
            fallback_key="", expected_endpoint=consultation.chat_endpoint,
            expected_model="consult-model", expected_source="static_route", expected_role="decide2",
            expected_chat_server=consultation, expected_tokenizer_server=consultation,
            expected_context=65536, deployments=deployments,
        )

        child_route = frame_routes.resolve("unit")
        child_journal = MemoryJournal()
        _wire_and_dispatch(
            Capture("child-request", child_journal, frame_route=child_route, purpose="unit"),
            _work_body(), fallback_endpoint=f"{baseline['endpoint']}/chat/completions",
            fallback_key="", expected_endpoint=child.chat_endpoint,
            expected_model="child-model", expected_source="static_route", expected_role="unit",
            expected_chat_server=child, expected_tokenizer_server=child,
            expected_context=65536, deployments=deployments,
        )


def test_small_context_local_deployment_is_the_capacity_authority(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "2")
    with (RecordingDeployment("default") as default,
          RecordingDeployment("small", context_tokens=32768, model_ids=("small-local",)) as small):
        _install_routes(
            default,
            f"unit.endpoint={small.base_url}", "unit.model=small-local",
            "unit.context_tokens=32768", "unit.max_tokens=32767",
        )
        route = frame_routes.resolve("unit")
        journal = MemoryJournal()
        _wire_and_dispatch(
            Capture("small-context-request", journal, frame_route=route, purpose="unit"),
            _body("small-local", "x" * 1024, max_tokens=32767),
            fallback_endpoint=default.chat_endpoint, fallback_key="",
            expected_endpoint=small.chat_endpoint, expected_model="small-local",
            expected_source="static_route", expected_role="unit",
            expected_chat_server=small, expected_tokenizer_server=small,
            expected_context=32768, deployments=(default, small), expect_capacity_event=True,
        )


def test_settings_only_route_keeps_its_body_setting_on_the_decided_server(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "2")
    with RecordingDeployment("default") as default:
        baseline = _install_routes(default, "read.temperature=0.25")
        route = frame_routes.resolve("read")
        journal = MemoryJournal()
        capture = Capture("settings-only-request", journal, frame_route=route, purpose="read")
        _wire_and_dispatch(
            capture, _body("base-model", max_tokens=24000),
            fallback_endpoint=f"{baseline['endpoint']}/chat/completions", fallback_key="",
            expected_endpoint=default.chat_endpoint, expected_model="base-model",
            expected_source="static_route", expected_role="read",
            expected_chat_server=default, expected_tokenizer_server=default,
            expected_context=65536, deployments=(default,),
        )
        assert default.chat_bodies[-1]["temperature"] == 0.25
        frame_events = journal.of_kind("frame_route", "settings-only-request")
        assert len(frame_events) == 1
        assert frame_events[0]["overridden_fields"] == ["temperature"]


def test_override_uses_its_deployment_and_configured_tokenizer_origin(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "2")
    with (RecordingDeployment("default") as default,
          RecordingDeployment("override", model_ids=("reader-model",)) as override,
          RecordingDeployment("override-tokenizer", context_tokens=8192,
                              model_ids=("tokenizer-model",)) as tokenizer):
        monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", default.base_url)
        _install_routes(
            default,
            f"read.endpoint={override.base_url}", "read.model=reader-model",
            f"read.tokenizer={tokenizer.base_url}",
        )
        route = frame_routes.resolve("read")
        journal = MemoryJournal()
        _wire_and_dispatch(
            Capture("tokenizer-override-request", journal, frame_route=route, purpose="read"),
            _body("base-model"), fallback_endpoint=default.chat_endpoint, fallback_key="",
            expected_endpoint=override.chat_endpoint, expected_model="reader-model",
            expected_source="static_route", expected_role="read",
            expected_chat_server=override, expected_tokenizer_server=tokenizer,
            expected_context=65536, deployments=(default, override, tokenizer),
        )
        frame_events = journal.of_kind("frame_route", "tokenizer-override-request")
        assert "tokenizer" in frame_events[0]["overridden_fields"]
        assert default.chat_wires == [] and default.tokenize_wires == []
        assert override.tokenize_wires == []


class EnvironmentRouter:
    """A replay fixture whose proposed route follows only its current environment."""

    def __init__(self) -> None:
        self.prepare_calls = 0

    def prepare_call(self, _capture: object, body: dict[str, object],
                     _endpoint: str, _key: str) -> SimpleNamespace:
        self.prepare_calls += 1
        body = dict(body)
        body["model"] = os.environ["STATEFRAME_ROUTE_READ_MODEL"]
        endpoint = os.environ["STATEFRAME_ROUTE_READ_ENDPOINT"].rstrip("/")
        return SimpleNamespace(
            body=body,
            endpoint=f"{endpoint}/chat/completions",
            key="",
            applied=True,
            decision=SimpleNamespace(endpoint=SimpleNamespace(api_key_env=None)),
        )

    def validate_final_request(self, _capture: object, _body: dict[str, object]) -> None:
        return None


def test_schema_two_replay_uses_journaled_decision_after_route_environment_changes(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "2")
    with (RecordingDeployment("recorded", context_tokens=16384,
                              model_ids=("recorded-model",)) as recorded,
          RecordingDeployment("changed", context_tokens=8192,
                              model_ids=("changed-model",)) as changed):
        router = EnvironmentRouter()
        monkeypatch.setenv("STATEFRAME_ROUTE_READ_ENDPOINT", recorded.base_url)
        monkeypatch.setenv("STATEFRAME_ROUTE_READ_MODEL", "recorded-model")
        journal = MemoryJournal()
        first = Capture("resume-request", journal, model_router=router, purpose="read")
        _wire_and_dispatch(
            first, _body("base-model", "x" * 1024, max_tokens=16383),
            fallback_endpoint=changed.chat_endpoint, fallback_key="",
            expected_endpoint=recorded.chat_endpoint, expected_model="recorded-model",
            expected_source="model_router", expected_role="read",
            expected_chat_server=recorded, expected_tokenizer_server=recorded,
            expected_context=16384, deployments=(recorded, changed),
            expect_capacity_event=True,
        )

        monkeypatch.setenv("STATEFRAME_ROUTE_READ_ENDPOINT", changed.base_url)
        monkeypatch.setenv("STATEFRAME_ROUTE_READ_MODEL", "changed-model")
        resumed = Capture("resume-request", journal, model_router=router, purpose="read")
        _wire_and_dispatch(
            resumed, _body("changed-model", "y" * 1024, max_tokens=16383),
            fallback_endpoint=changed.chat_endpoint, fallback_key="",
            expected_endpoint=recorded.chat_endpoint, expected_model="recorded-model",
            expected_source="replayed", expected_role="read",
            expected_chat_server=recorded, expected_tokenizer_server=recorded,
            expected_context=16384, deployments=(recorded, changed),
            recovery_prepared=True, expect_capacity_event=True,
        )
        decisions = journal.of_kind("route_decision", "resume-request")
        assert [(event["model"], event["endpoint"], event["role"]) for event in decisions] == [
            ("recorded-model", recorded.chat_endpoint, "read"),
            ("recorded-model", recorded.chat_endpoint, "read"),
        ]
        assert router.prepare_calls == 1
        assert len(recorded.chat_wires) == 2
        assert changed.chat_wires == [] and changed.tokenize_wires == []


def test_conflicting_tokenizer_environment_is_refused_before_any_send(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "2")
    with (RecordingDeployment("default") as default,
          RecordingDeployment("override", model_ids=("reader-model",)) as override):
        monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", default.base_url)
        options = runner.RunOptions(
            repo=None,
            task="synthetic route conflict",
            endpoint=Endpoint(default.base_url, "base-model"),
            route_cli=[f"read.endpoint={override.base_url}", "read.model=reader-model"],
        )
        options._context_tokens = 65536
        options._context_env = {}
        with pytest.raises(runner.RunError, match="tokenizer"):
            runner._probe_routes(options, lambda *_args: None)
        assert default.requests == [] and override.requests == []
        assert default.chat_wires == [] and override.chat_wires == []
        assert default.tokenize_wires == [] and override.tokenize_wires == []
