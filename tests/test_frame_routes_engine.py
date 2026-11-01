"""Engine route authority: public digest, unit masks, action and body decisions."""
from __future__ import annotations

import copy
import ast
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.request import Request

import pytest

from stateframe import routes
import kernel_frame_routes as frame_routes


DEFAULT = {"endpoint": "http://127.0.0.1:8080/v1", "model": "base", "api_key_env": None,
           "temperature": 1.0, "max_tokens": 24000, "timeout_s": 1800.0,
           "slot": None, "context_tokens": 65536, "allow_remote": False}


@pytest.fixture(autouse=True)
def clear_routes():
    frame_routes.install(None)
    yield
    frame_routes.install(None)


def table(*entries):
    return routes.resolve(DEFAULT, cli=list(entries), environ={}).public()


def test_digest_and_unit_mask(monkeypatch):
    spec = table("read.model=reader", "unit.model=unit", "unit.patch.model=patch")
    frame_routes.install(spec)
    monkeypatch.setenv("STATEFRAME_ROUTE_SCOPE", "unit")
    monkeypatch.setenv("STATEFRAME_ROUTE_UNIT_CLASS", "patch")
    monkeypatch.setenv("STATEFRAME_ROUTE_SLOT", "3")
    chosen = frame_routes.resolve("read")
    assert chosen.model == "patch"
    assert chosen.settings["slot"] == 3
    assert chosen.settings["temperature"] == 1.0
    bad = copy.deepcopy(spec)
    bad["unit_overrides"]["unit.patch"] = []
    with pytest.raises(ValueError, match="digest"):
        frame_routes.install(bad)


def test_plan_uses_complete_current_failure_and_last_action():
    frame_routes.install(table("read.model=reader"))
    assert frame_routes.role_for_request("rev") == "plan"
    check = {"execution_ids": ["a", "b"], "check_receipt_ids": ["r1", "r2"], "status": "FAIL"}
    frame_routes.close_action({"tool": "run_tests", "tool_call_id": "call"}, revision="rev",
                              canonical=check, receipts=[{"execution_id": "a", "receipt_id": "r1", "revision": "rev"},
                                                         {"execution_id": "b", "receipt_id": "r2", "revision": "rev"}])
    assert frame_routes.LAST_CLOSED_ACTION["call_id"] == "call"
    assert frame_routes.LAST_CLOSED_ACTION["execution_ids"] == ["a", "b"]
    assert frame_routes.role_for_request("rev") == "plan"
    assert frame_routes.role_for_request("new-rev") == "read"
    frame_routes.close_action({"tool": "read"}, revision="rev")
    assert frame_routes.role_for_request("rev") == "read"
    frame_routes.close_action({"tool": "run_tests"}, revision="rev", canonical=check,
                              receipts=[{"execution_id": "a", "receipt_id": "r1", "revision": "rev"},
                                        {"execution_id": "b", "receipt_id": "r2", "revision": "old"}])
    assert frame_routes.role_for_request("rev") == "read"


def test_every_call_classified_before_dispatch():
    from probes import react_repair

    dispatcher_tools = {tool["function"]["name"] for tool in react_repair.TOOLS}
    assert not frame_routes.unclassified_dispatcher(dispatcher_tools)
    def calls(*names):
        return [{"function": {"name": name}} for name in names]
    assert frame_routes.classify_calls(calls("read", "edit", "done")) == "edit"
    assert frame_routes.classify_calls(calls("read", "done")) == "finish"
    assert frame_routes.classify_calls(calls("run_tests", "read")) == "read"
    assert frame_routes.classify_calls(calls("custom"), custom_writes={"custom"}) == "edit"
    assert frame_routes.unclassified_dispatcher({"read", "edit", "done"}) == set()
    assert frame_routes.unclassified_dispatcher({"new_dispatcher_tool"}) == {"new_dispatcher_tool"}


def test_body_changes_only_effective_fields_and_strips_invalid_pin():
    frame_routes.install(table("read.endpoint=http://127.0.0.1:8090/v1", "read.model=reader",
                               "read.temperature=0.3", "read.max_tokens=500"))
    body = {"model": "base", "temperature": 1.0, "max_tokens": 24000,
            "id_slot": 4, "cache_prompt": True, "seed": 7}
    frame_routes.apply_body(body, frame_routes.resolve("read"),
                            default_endpoint="http://127.0.0.1:8080/v1/chat/completions")
    assert body == {"model": "reader", "temperature": 0.3, "max_tokens": 500, "seed": 7}


def test_locality_refuses_remote_without_permission():
    frame_routes.require_local("http://127.0.0.1:8090/v1/chat/completions", allow_remote=False)
    with pytest.raises(ValueError, match="not on this machine"):
        frame_routes.require_local("https://example.com/v1/chat/completions", allow_remote=False)
    frame_routes.require_local("https://example.com/v1/chat/completions", allow_remote=True)


def test_routed_redirect_refuses_remote_hop():
    from stateframe.engine import kernel_bounded_http

    handler = kernel_bounded_http._RoutedRedirects(False)
    request = Request("http://127.0.0.1:8090/v1/chat/completions")
    with pytest.raises(ValueError, match="not on this machine"):
        handler.redirect_request(request, None, 302, "Found", {},
                                 "https://example.com/v1/chat/completions")


def test_capture_tokenizer_identity_ignores_ambient_override(monkeypatch):
    from stateframe.engine import kernel_request_diet

    frame_routes.install(table("read.endpoint=http://127.0.0.1:8090/v1", "read.model=reader"))
    route = frame_routes.resolve("read")
    capture = SimpleNamespace(frame_route=route, _tokenize_endpoint=route.settings["endpoint"])
    monkeypatch.setenv("LLAMA_CPP_TOKENIZE_URL", "http://127.0.0.1:8080/tokenize")
    assert kernel_request_diet.resolve_tokenize_url(capture=capture) == "http://127.0.0.1:8090/tokenize"
    assert '"model":"reader"' in kernel_request_diet.tokenizer_contract(None, capture)


def test_native_batch_is_uncommitted_when_route_rejects_full_batch():
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_native_tools import NativeToolBatch

    frame_routes.install(table("read.model=reader"))

    class Journal:
        def emit(self, *args, **kwargs):
            raise AssertionError("rejected batch must not emit a selection")

    capture = SimpleNamespace(journal=Journal(), request_id="request", snapshot_id="snapshot",
                              selected_response_event_id="response")
    token = ACTIVE_REQUEST.set(capture)
    batch = NativeToolBatch()
    calls = [{"id": "r", "function": {"name": "read", "arguments": "{}"}},
             {"id": "w", "function": {"name": "edit", "arguments": "{}"}}]

    def reject(decoded):
        assert decoded == calls
        raise frame_routes.RouteEscalation("edit", decoded)

    try:
        with pytest.raises(frame_routes.RouteEscalation):
            batch.next(lambda *_args, **_kwargs: ("", calls), [], before_commit=reject)
    finally:
        ACTIVE_REQUEST.reset(token)
    assert batch.pending == [] and batch.origin is None and batch.selected_ref is None


def test_child_keeps_routed_key_secret_through_environment_purge(monkeypatch):
    from stateframe import _child

    fake = {"STATEFRAME_ROUTED_KEY": "secret", "STATEFRAME_ROUTE_SCOPE": "ambient"}
    monkeypatch.setattr(_child.os, "environ", fake)
    _child.install_environment({"environment": {}, "routes": {"roles": {
        "read": {"api_key_env": "STATEFRAME_ROUTED_KEY"}}}})
    assert fake == {"STATEFRAME_ROUTED_KEY": "secret"}


def test_rejected_capture_restores_prior_accepted_and_cannot_be_retained():
    from stateframe.engine.kernel_stateframe_runtime import StateFrameRuntime

    frame_routes.install(table("read.model=reader"))

    prior = SimpleNamespace(response_event_id="accepted")
    rejected = SimpleNamespace(response_event_id="discarded", passive_retention_context={})
    runtime = SimpleNamespace(_last_capture=prior, _active_capture=rejected,
                              passive_retention=SimpleNamespace(ingest_capture=lambda *_: pytest.fail(
                                  "discarded response reached passive retention")))
    frame_routes.reject_capture(runtime, rejected)
    assert rejected.route_response_discarded is True
    assert runtime._active_capture is prior
    assert StateFrameRuntime._record_passive_retention_capture(runtime, rejected) is None


def test_inactive_tool_call_branch_has_original_single_send():
    from probes import react_repair

    frame_routes.install(None)
    assert frame_routes.ROUTES_ACTIVE is False
    source = Path(react_repair.__file__).read_text()
    tree = ast.parse(source)
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "not _frame_routes.ROUTES_ACTIVE"]
    assert len(branches) == 1
    assert ast.unparse(branches[0].body[0]) == "content, calls = _provider_request_once()"
    assert len(branches[0].body) == 1
    from stateframe.engine import kernel_stateframe_runtime
    runtime = ast.parse(Path(kernel_stateframe_runtime.__file__).read_text())
    inactive = [node for node in ast.walk(runtime) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "not _frame_routes.ROUTES_ACTIVE"
                and "build_typed_state" in " ".join(ast.unparse(stmt) for stmt in node.body)]
    assert len(inactive) == 1
    assert "kernel_frame_routes" not in " ".join(ast.unparse(stmt) for stmt in inactive[0].body)


def test_shipped_route_modules_have_no_crash_pause_switch():
    root = Path(__file__).resolve().parents[1] / "src" / "stateframe"
    for path in (root / "routes.py", root / "_child.py",
                 root / "engine" / "kernel_frame_routes.py",
                 root / "engine" / "probes" / "react_repair.py"):
        assert "STATEFRAME_ROUTE_TEST_PAUSE" not in path.read_text(), path


@pytest.mark.parametrize("point", ["after_discard", "after_route:2"])
def test_test_only_crash_hook_blocks_at_exact_boundary(tmp_path, point):
    hook = Path(__file__).parent / "acceptance" / "route_crash_hook"
    marker = tmp_path / "ready"
    if point == "after_discard":
        (tmp_path / "kernel_measurement.py").write_text(
            "class AuditJournal:\n    def emit(self, kind, **fields):\n        return {}\n")
        code = "import kernel_measurement; kernel_measurement.AuditJournal().emit('route_escalation_discarded')"
    else:
        code = ("import os; from pathlib import Path; "
                f"p=Path({str(tmp_path / '0002.route.json')!r}); "
                "p.with_suffix('.tmp').write_text('route'); "
                "os.replace(p.with_suffix('.tmp'), p)")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(hook), str(tmp_path))),
           "ROUTE_CRASH_POINT": point, "ROUTE_CRASH_MARKER": str(marker)}
    process = subprocess.Popen([sys.executable, "-c", code], env=env)
    try:
        cutoff = time.monotonic() + 5
        while time.monotonic() < cutoff and process.poll() is None and not marker.is_file():
            time.sleep(0.01)
        assert marker.is_file() and process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=5)


def test_selected_tokenizer_and_capacity_probe_use_same_routed_endpoint(monkeypatch):
    from stateframe.scripted_endpoint import Script, serve
    from stateframe.engine import kernel_context_capacity, kernel_request_diet

    monkeypatch.delenv("STATEFRAME_N_CTX", raising=False)
    monkeypatch.delenv("STATEFRAME_CONTEXT_PREFLIGHT", raising=False)
    script = Script({"model": "reader", "slots": [65536], "tokenize": {"mode": "ok"}})
    server = serve(script)
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/v1"
        frame_routes.install(table(f"read.endpoint={base}", "read.model=reader"))
        route = frame_routes.resolve("read")
        capture = SimpleNamespace(frame_route=route, _tokenize_endpoint=base)
        tokenize_url = kernel_request_diet.resolve_tokenize_url(capture=capture)
        meter = kernel_request_diet.RequestTokenMeter(endpoint=tokenize_url,
                                                       route_allow_remote=False)
        body = {"model": "reader", "messages": [{"role": "user", "content": "routed prompt"}],
                "max_tokens": 100}
        assert meter.measure(body)[1] == "tokenize"
        assert meter.count_text_exact("routed prompt") is not None
        kernel_context_capacity.enforce_context_capacity(capture, body, endpoint=route.endpoint)
        assert script.tokenize_calls >= 1
        assert script.slots_calls >= 1
    finally:
        server.shutdown()
        server.server_close()


def test_routed_http_redirect_is_refused_on_actual_transport():
    from stateframe.engine import kernel_bounded_http, kernel_request_diet

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "https://example.com/v1/chat/completions")
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        with pytest.raises(kernel_bounded_http.BoundedTransportError) as caught:
            kernel_bounded_http.request_bytes(b'{"model":"reader","messages":[]}', endpoint, "",
                deadline=time.monotonic() + 10, socket_timeout=5, route_meta={"role": "read"})
        assert caught.value.error_type == "ValueError"
        with pytest.raises(ValueError, match="not on this machine"):
            kernel_request_diet._http_tokenize("routed prompt", endpoint,
                                               route_allow_remote=False)
    finally:
        server.shutdown()
        server.server_close()


def test_source_replan_wrapper_forwards_route_options_on_every_path():
    # Regression: FrameFit (on in the shipped profile) wraps prepare with a fixed signature that
    # dropped forced_route/frozen_route_capture, so every routed second leg died with a TypeError.
    import kernel_stateframe_runtime as runtime_mod

    seen = []

    def prepare(self, messages, *, uncapped, tools, provider_request=True,
                request_host_text_preparer=None, forced_route=None, frozen_route_capture=None):
        seen.append((forced_route, frozen_route_capture))
        return SimpleNamespace(capture=None)

    wrapped = runtime_mod._reprepare_on_source_change(prepare)
    first = object()
    for mode in ("off", "on"):
        owner = SimpleNamespace(stateframe_framefit_mode=mode)
        wrapped(owner, [], uncapped=[], tools=[], forced_route="edit", frozen_route_capture=first)
        wrapped(owner, [], uncapped=[], tools=[])
    assert seen == [("edit", first), (None, None), ("edit", first), (None, None)]


def test_routed_seal_keeps_token_meter_on_capture_only():
    # Regression: sealing a routed request read context._request_token_meter, which a routed
    # context does not carry (AttributeError on the shipped profile).
    import kernel_final_wire_allocation as allocation

    routed = SimpleNamespace(frame_route={"role": "read"})
    plain = SimpleNamespace()
    assert allocation.publishes_runtime_token_meter(routed)  # routes inactive: unchanged behavior
    frame_routes.install(table("read.model=reader"))
    assert not allocation.publishes_runtime_token_meter(routed)
    assert allocation.publishes_runtime_token_meter(plain)


def test_notice_dedup_reentry_keeps_forced_role():
    # Regression: notice dedup (on in the shipped profile) re-enters prepare and dropped the
    # forced role, so an escalated edit step was re-asked of the read model.
    import kernel_stateframe_notice_dedup as dedup

    seen = []

    def prepare(runtime, messages, *, uncapped, tools, provider_request=True,
                request_host_text_preparer=None, forced_route=None, frozen_route_capture=None):
        seen.append((forced_route, frozen_route_capture, request_host_text_preparer))
        return SimpleNamespace(messages=[])

    first, preparer = object(), (lambda *args, **kwargs: None)
    runtime = SimpleNamespace()
    dedup.prepare_with_commit(runtime, prepare, [], uncapped=[], tools=[],
                              forced_route="edit", frozen_route_capture=first)
    dedup.prepare_with_commit(runtime, prepare, [], uncapped=[], tools=[],
                              request_host_text_preparer=preparer,
                              forced_route="finish", frozen_route_capture=first)
    dedup.prepare_with_commit(runtime, prepare, [], uncapped=[], tools=[])
    assert seen == [("edit", first, None), ("finish", first, preparer), (None, None, None)]


def test_forced_leg_options_rebuild_only_escalated_legs():
    frame_routes.install(table("read.model=reader"))
    first = object()
    escalated = SimpleNamespace(frame_route=frame_routes.resolve("edit", forced_by="r1"),
                                _frame_route_frozen_capture=first)
    ordinary = SimpleNamespace(frame_route=frame_routes.resolve("read"),
                               _frame_route_frozen_capture=None)
    assert frame_routes.forced_leg_options(escalated) == {"forced_route": "edit",
                                                          "frozen_route_capture": first}
    assert frame_routes.forced_leg_options(ordinary) == {}
    assert frame_routes.forced_leg_options(SimpleNamespace()) == {}


def _bound_names(nodes, *, skip=()):
    """Names a block binds (assignments, except-as, imports, loops), not counting nested functions."""
    bound, stack = set(), list(nodes)
    while stack:
        node = stack.pop()
        if any(node is skipped for skipped in skip):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
                             ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue  # own scope: cannot rebind work() locals
        stack.extend(ast.iter_child_nodes(node))
    return bound


def test_routed_step_branch_never_rebinds_work_locals():
    # Regression: the escalation loop reused `target`, overwriting work()'s target file path, so
    # every routed run that escalated crashed at the end ('str' object has no attribute 'relative_to').
    from probes import react_repair

    tree = ast.parse(Path(react_repair.__file__).read_text())
    work = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "work")
    branch = next(node for node in ast.walk(work) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "not _frame_routes.ROUTES_ACTIVE")
    routed = _bound_names(branch.orelse)
    elsewhere = _bound_names(work.body, skip=branch.orelse)
    # The step's own results are shared with the unrouted branch by design.
    shared = {"content", "calls", "_prepared", "_capture_token"}
    clashes = sorted(name for name in routed & elsewhere
                     if name not in shared and not name.startswith("_route"))
    assert clashes == []


@contextmanager
def _post_server(responder):
    seen = []

    class StockServer(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # llama.cpp framing: Content-Length, keep-alive

        def log_message(self, *_args):
            pass

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            headers = {name.lower(): value for name, value in self.headers.items()}
            seen.append((headers, raw))
            status, reply = responder(headers, raw)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), StockServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        yield endpoint, seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _large_wire():
    return ('{"model":"reader","messages":[{"role":"user","content":"' + "x" * 40000 + '"}]}').encode()


def test_gzip_415_after_deadline_is_not_resent():
    from stateframe.engine import kernel_bounded_http

    wire = _large_wire()

    def delayed_refusal(_headers, _raw):
        time.sleep(0.2)
        return 415, b""

    with _post_server(delayed_refusal) as (endpoint, seen):
        # Invoke the same child directly so the parent's deadline watchdog cannot mask its
        # own decision after it receives the empty 415 response.
        packet = {"wire": wire.decode(), "endpoint": endpoint, "key": "",
                  "deadline": time.monotonic() + 0.05, "socket_timeout": 5, "gzip": True}
        result = subprocess.run(
            [sys.executable, "-I", "-B", kernel_bounded_http.__file__, "--http-child"],
            input=json.dumps(packet).encode(), capture_output=True, timeout=5, check=False)
    assert result.returncode != 0
    diagnostic = json.loads(result.stderr)
    assert diagnostic["error_type"] == "HTTPError"
    assert diagnostic["http_status"] == 415
    assert len(seen) == 1
    assert seen[0][0].get("content-encoding") == "gzip"
