"""Exact count reuse and ordered consumption through tokenizer entry points."""
from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

import kernel_request_diet as diet
from tape_server import TapeServer


@pytest.fixture
def pilot(monkeypatch, tmp_path):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path))
    monkeypatch.delenv("LLAMA_CPP_TOKENIZE_URL", raising=False)
    monkeypatch.delenv("STATEFRAME_EXACT_TOKEN_COUNT_CACHE", raising=False)
    module = ModuleType("stateframe.send1_wave4")
    module.enabled = lambda: True
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _server():
    server = TapeServer({"turns": [{"message": {"content": "unused"}}]})
    server._token_count = lambda _text: 9
    return server


_PROCESS = """
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from kernel_agent_tools import AgentTools
from kernel_custom_tools import CustomToolSet
from kernel_final_wire_budget import reconcile_frozen_budget
import kernel_request_diet as diet
from kernel_provider_dialect import capabilities_for
from kernel_request_budget import _request_token_meter
from kernel_shell_tool import SHELL_SCHEMA
from kernel_tool_settings import ToolSettings
from stateframe.engine.kernel_request_auth_identity import protected_home_identity

base, variant = sys.argv[1:]
home = Path(os.environ["STATEFRAME_HOME"])
home.mkdir(mode=0o700, exist_ok=True)
protected_home_identity(home, initialize=True)
os.environ["STATEFRAME_SEND1_WAVE4"] = "off" if variant == "off" else "on"
# Keep deliberate direct-map collisions in the dedicated storage test.
diet._EXACT_TOKEN_COUNT_CACHE_ENTRIES = 2**256
endpoint = base + "/chat/completions"
key = "rotated-key" if variant == "auth" else "fixture-key"
os.environ["TEST_API_KEY"] = key
sys.modules["probes.react_repair"] = SimpleNamespace(API=endpoint, KEY=key)
pin = "b" * 64 if variant == "pin" else "a" * 64
manifest = {"model": "alias", "tokenizer": pin}
spec = {"chat_url": endpoint, "model": "alias", "api_key_env": "TEST_API_KEY",
        "environment": {"REACT_TOKENIZER": pin}}
if variant != "off":
    spec["send1_wave4"] = True
if variant == "alias":
    manifest["tokenizer"] = spec["environment"]["REACT_TOKENIZER"] = "alias"
assert capabilities_for(endpoint, key, require_slots=False).continuation

clock = "0250.0" if variant == "countdown" else "0300.0"
notices = [{"role": "user", "content": "Notice one"}, {"role": "user", "content": "Notice two"}]
if variant == "notice":
    notices.reverse()
draft = {"model": "alias", "messages": [
    {"role": "system", "content": (
        "[CONTINUATION LIMITS | stateframe-continuation-limits/1] "
        "sandbox_seconds_remaining=" + clock + ";")},
    *notices], "tools": []}
if variant == "options":
    draft["chat_template_kwargs"] = {"enable_thinking": False}
cap = SimpleNamespace(request_id="send-1", _tokenize_endpoint=endpoint,
                      _tokenize_logical_request_id="lineage-1", journal=SimpleNamespace(manifest=manifest))
runtime = SimpleNamespace()
measure = _request_token_meter(runtime, cap).measure(draft)
final = copy.deepcopy(draft)
final["messages"].append({"role": "user", "content": "Final placement"})
if variant == "placement":
    final["messages"].insert(1, final["messages"].pop())
cap.request_action_contract = {"admitted_tool_schemas": []}
cap._frame_candidate_inputs = {"token_ceiling": 100 if variant == "capacity" else 2000}
receipt = reconcile_frozen_budget(cap, runtime, final)
assert receipt["measurement"]["request_id"] == "send-1"
assert receipt["measurement"]["logical_request_id"] == "lineage-1"

if variant == "schema":
    shell = SHELL_SCHEMA["function"]
    shell["description"] = "X" + shell["description"][1:]
existing_reports = set(home.glob("tool-schema-budget-*.json"))
hub = AgentTools(SimpleNamespace(spec=spec, run_dir=home, settings=ToolSettings(web=True)),
                 CustomToolSet((), "empty"), defer_schema_budget=True)
assert hub.schemas()
assert len(set(home.glob("tool-schema-budget-*.json")) - existing_reports) == 1
report = hub.schema_budget
assert hub.schema_budget is report
assert report["shell"] == {"tokens": 9, "method": "tokenize"}
print(json.dumps({"measure": measure, "measurement": receipt["measurement"],
                  "status": receipt["status"], "schemas": report}))
"""


def test_separate_launches_reuse_draft_final_and_first_use_schema_counts(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_TOKENIZE_URL", raising=False)
    source = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, STATEFRAME_HOME=str(tmp_path / "home"),
               PYTHONPATH=os.pathsep.join((str(source), str(source / "stateframe/engine"))))
    with _server() as server:
        def launch(variant, calls):
            before = server.tokenize_request_count
            run = subprocess.run([sys.executable, "-c", _PROCESS, server.url, variant],
                                 env=env, capture_output=True, text=True, timeout=40)
            assert run.returncode == 0, run.stderr
            assert server.tokenize_request_count - before == calls, variant
            return json.loads(run.stdout)

        original = launch("cold", 9)
        assert original == launch("warm", 0)
        countdown = launch("countdown", 0)
        assert countdown["measurement"]["body_sha256"] != original["measurement"]["body_sha256"]
        assert countdown["measure"] == original["measure"]
        capacity = launch("capacity", 0)
        assert capacity["measurement"]["body_sha256"] == original["measurement"]["body_sha256"]
        assert capacity["measurement"]["max_prompt_tokens"] == 100
        assert capacity["status"] == "HOST_BLOCKED"
        launch("placement", 1)
        launch("schema", 1)
        launch("notice", 2)
        launch("options", 2)
        launch("pin", 9)
        launch("auth", 9)
        launch("alias", 9)
        launch("alias", 9)
        assert original == launch("off", 9)


def _pinned_meter(base, *, secret="a"):
    endpoint = base + "/chat/completions"
    meter = diet.RequestTokenMeter(endpoint=endpoint, generation_endpoint=endpoint, auth_key=secret)
    meter._persistent_provider_identity = diet.tokenizer_contract(
        None, None, persistent=True, recorded={
            "model": "alias", "chat_url": endpoint, "environment": {"REACT_TOKENIZER": "c" * 64},
        },
    )
    return meter


def _prompt(content):
    return {"model": "alias", "messages": [{"role": "user", "content": content}], "tools": []}


def test_signed_slots_reject_wrong_key_count_and_permissions(tmp_path, monkeypatch, pilot):
    from kernel_provider_dialect import capabilities_for

    monkeypatch.setattr(diet, "_EXACT_TOKEN_COUNT_CACHE_ENTRIES", 2**256)
    with _server() as server:
        assert capabilities_for(server.url, "a", require_slots=False).continuation
        first, second = _prompt("one"), _prompt("two")
        meter = _pinned_meter(server.url)
        expected = meter.measure(first)
        meter.measure(second)
        slots = tmp_path / "token-counts-v1"
        paths = list(slots.iterdir())
        assert len(paths) == 2
        record = paths[0].read_bytes()
        paths[0].write_bytes(paths[1].read_bytes())
        paths[1].write_bytes(record)
        meter = _pinned_meter(server.url)
        assert meter.measure(first) == meter.measure(second) == expected
        assert server.tokenize_request_count == 4
        for path in paths:
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b" 9\n", b" 1\n"))
        assert _pinned_meter(server.url).measure(first) == expected
        assert server.tokenize_request_count == 5
        for path in paths:
            path.chmod(0o644)
        assert _pinned_meter(server.url).measure(first) == expected
        assert server.tokenize_request_count == 6

        # A warm token count must never suppress a scheduled recovery probe.
        meter = _pinned_meter(server.url)
        meter.begin_request("outage")
        meter._availability = "unavailable"
        meter._start_reprobe_schedule()
        for index in range(diet.TOKENIZE_REPROBE_AFTER_REQUESTS):
            meter.begin_request(f"next-{index}")
        assert meter.measure(first) == expected
        assert meter._reprobe_status == "recovered"
        assert server.tokenize_request_count == 7


def test_signed_slot_key_preserves_field_boundaries(tmp_path):
    from stateframe.engine.kernel_request_auth_identity import protected_home_identity

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    protected_home_identity(home, initialize=True)
    _, key, _ = diet._strong_count_slot(home / "token-counts-v1", "provider", "content", "prompt")
    payload = json.dumps(
        (diet._COUNT_CONTRACT_VERSION, "prompt", "provider", "content"), separators=(",", ":"),
    ).encode("ascii")
    assert key == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("changed", ["key", "directory"])
def test_signed_cache_rejects_authority_replaced_during_read(tmp_path, monkeypatch, pilot, changed):
    from kernel_provider_dialect import capabilities_for
    from stateframe.startup_cache_writes import writer_scope

    with _server() as server:
        assert capabilities_for(server.url, "a", require_slots=False).continuation
        body = _prompt("Exact prompt")
        old_tokens, _ = _pinned_meter(server.url).measure(body)
        directory = tmp_path / "token-counts-v1"
        entry, = directory.iterdir()
        signature = entry.read_bytes().partition(b"\n")[2]
        compare = diet.hmac.compare_digest
        replaced = False

        def compare_and_replace(actual, expected):
            nonlocal replaced
            match = compare(actual, expected)
            if actual == signature and not replaced:
                replaced = True
                if changed == "key":
                    replacement = tmp_path / "new-key"
                    replacement.touch(mode=0o600)
                    replacement.write_bytes(b"x" * 32)
                    replacement.replace(tmp_path / "identity.key")
                else:
                    directory.rename(tmp_path / "old-counts")
                    directory.mkdir(mode=0o700)
            return match

        monkeypatch.setattr(diet.hmac, "compare_digest", compare_and_replace)
        server._token_count = lambda _text: 17
        with writer_scope(lambda *_args, **_kwargs: None):
            tokens, method = _pinned_meter(server.url).measure(body)
        assert replaced
        assert method == "tokenize"
        assert tokens == old_tokens + 8
        assert server.tokenize_request_count == 2


@pytest.mark.parametrize("changed", ["key", "directory"])
def test_signed_count_in_memory_requires_current_authority(tmp_path, pilot, changed):
    from kernel_provider_dialect import capabilities_for
    from stateframe.engine.kernel_request_auth_identity import protected_home_identity

    protected_home_identity(tmp_path, initialize=True)
    with _server() as server:
        assert capabilities_for(server.url, "", require_slots=False).continuation
        body = _prompt("Exact prompt")
        old = _pinned_meter(server.url, secret=None).measure(body)
        meter = _pinned_meter(server.url, secret=None)
        meter.begin_request("first")
        assert meter.measure(body) == old
        assert server.tokenize_request_count == 1

        if changed == "key":
            replacement = tmp_path / "new-key"
            replacement.touch(mode=0o600)
            replacement.write_bytes(b"x" * 32)
            replacement.replace(tmp_path / "identity.key")
        else:
            directory = tmp_path / "token-counts-v1"
            directory.rename(tmp_path / "old-counts")
            directory.mkdir(mode=0o700)
        server._token_count = lambda _text: 17

        meter.begin_request("second")
        assert meter.measure(body) == (old[0] + 8, "tokenize")
        assert server.tokenize_request_count == 2


def test_text_and_prompt_use_separate_slots_even_for_identical_utf8(tmp_path, monkeypatch, pilot):
    from kernel_provider_dialect import capabilities_for

    monkeypatch.setattr(diet, "_EXACT_TOKEN_COUNT_CACHE_ENTRIES", 2**256)
    with _server() as server:
        assert capabilities_for(server.url, "a", require_slots=False).continuation
        body = _prompt("Exact ä")
        for _ in range(2):
            meter = _pinned_meter(server.url)
            assert meter.measure(body)[1] == "tokenize"
            assert meter.count_text_exact(diet._tokenization_content(body)) == 9
        assert server.tokenize_request_count == 2
        assert len(list((tmp_path / "token-counts-v1").iterdir())) == 2


def test_route_namespace_reuse_requires_the_primary_pin_to_apply(monkeypatch, pilot):
    from kernel_frame_routes import Route
    from kernel_provider_dialect import capabilities_for
    from kernel_request_budget import _request_token_meter

    monkeypatch.setenv("TOKEN_TEST_KEY", "a")
    monkeypatch.setattr(diet._frame_routes, "ROUTES_ACTIVE", True)
    monkeypatch.setattr(diet, "_EXACT_TOKEN_COUNT_CACHE_ENTRIES", 2**256)
    with _server() as server:
        endpoint = server.url + "/chat/completions"
        monkeypatch.setitem(sys.modules, "probes.react_repair", SimpleNamespace(API=endpoint, KEY="a"))
        assert capabilities_for(endpoint, "a", require_slots=False).continuation
        journal = SimpleNamespace(manifest={"model": "alias", "tokenizer": "a" * 64})

        def measure(role, base):
            route = Route(role, "supervisor", None, {
                "endpoint": base, "model": "alias", "api_key_env": "TOKEN_TEST_KEY",
            })
            capture = SimpleNamespace(frame_route=route, journal=journal, _tokenize_endpoint=base)
            return _request_token_meter(None, capture).measure(_prompt("same"))

        assert measure("read", server.url) == measure("read", server.url)
        assert server.tokenize_request_count == 1
        assert measure("edit", server.url) == measure("edit", server.url)
        assert server.tokenize_request_count == 2
        assert measure("edit", server.url + "/override") == measure("edit", server.url + "/override")
        assert server.tokenize_request_count == 4


@pytest.mark.parametrize("fault", ("redirect", "malformed", "closed", "timeout"))
def test_faults_keep_off_verdict_and_cannot_publish_disk_counts(tmp_path, monkeypatch, pilot, fault):
    from kernel_provider_dialect import capabilities_for

    posts = []

    class FaultyTokenizer(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            if self.path.endswith("/props"):
                payload = {"default_generation_settings": {"n_ctx": 65536}, "model_path": "fixture"}
            else:
                payload = {"tokens": [1]}
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            posts.append(self.path)
            if fault == "redirect":
                self.send_response(302)
                self.send_header("Location", "/target")
                self.end_headers()
            elif fault == "malformed":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"tokens":[[]]}')
            elif fault == "closed":
                self.connection.shutdown(socket.SHUT_RDWR)
            else:
                time.sleep(0.5)

    with ThreadingHTTPServer(("127.0.0.1", 0), FaultyTokenizer) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}/v1"
            assert capabilities_for(base, "a", require_slots=False).continuation
            if fault == "timeout":
                monkeypatch.setattr(diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", 0.15)
            pilot.enabled = lambda: False
            meter = _pinned_meter(base)
            expected = meter.measure(_prompt(fault)), meter.unavailable_reason()
            method = "tokenize" if fault == "redirect" else "calibrated_bytes"
            assert expected[0][1] == method
            pilot.enabled = lambda: True
            for _ in range(2):
                meter = _pinned_meter(base)
                assert (meter.measure(_prompt(fault)), meter.unavailable_reason()) == expected
            attempts = 1 if fault == "redirect" else diet.TOKENIZE_MAX_CONSECUTIVE_FAILURES
            assert posts == ["/tokenize"] * (3 * attempts)
            assert not (tmp_path / "token-counts-v1").exists()
        finally:
            server.shutdown()
            thread.join()


def test_overlap_uses_two_single_requests_and_does_not_join_unused_wait(monkeypatch, pilot):
    started = threading.Barrier(3)
    release_first, release_unused = threading.Event(), threading.Event()
    unused_done = threading.Event()
    posts = []

    def post(content, url, **kwargs):
        label = json.loads(content)["messages"][0]["content"]
        posts.append((label, url, kwargs["auth_key"]))
        started.wait(5)
        if label == "source retained":
            assert release_first.wait(5)
        else:
            assert release_unused.wait(5)
            unused_done.set()
        return [1]

    monkeypatch.setattr(diet, "_http_tokenize", post)
    monkeypatch.setattr(diet.RequestTokenMeter, "_tokenizer_capability_proven", lambda _self: True)
    meter = _pinned_meter("http://127.0.0.1:49")
    retained, omitted = _prompt("source retained"), _prompt("source omitted")
    try:
        with meter.overlap_http_counts((retained, omitted), pilot_only=True):
            started.wait(5)
            release_first.set()
            assert meter.measure(retained)[1] == "tokenize"
        assert not unused_done.is_set()
        assert sorted(posts) == [
            ("source omitted", meter.endpoint, "a"),
            ("source retained", meter.endpoint, "a"),
        ]
        assert meter._tokenizer_calls_by_request == {None: 1}
    finally:
        release_unused.set()
    assert unused_done.wait(5)


def test_prefetch_respects_input_byte_cap_and_skips_deadline_or_replay(monkeypatch, pilot):
    posts = []
    monkeypatch.setattr(diet.RequestTokenMeter, "_tokenizer_capability_proven", lambda _self: True)

    def post(content, _url, **_kwargs):
        posts.append(json.loads(content)["messages"][0]["content"])
        return [1]

    monkeypatch.setattr(diet, "_http_tokenize", post)
    meter = _pinned_meter("http://127.0.0.1:53")
    small, large = _prompt("small"), _prompt("X" * 1000)
    with meter.overlap_http_counts((small, large), pilot_only=True, max_request_bytes=200):
        assert len(diet._PREFETCHED_HTTP_COUNTS.get()) == 1
        meter.measure(small)
    assert posts == ["small"]

    meter.bind_request("deadline", "deadline", deadline=time.monotonic() + 3)
    with meter.overlap_http_counts((large,), pilot_only=True):
        assert diet._PREFETCHED_HTTP_COUNTS.get() is None
    meter.bind_request("replay", "replay")
    monkeypatch.setattr(diet, "_replaying_counts", lambda: True)
    with meter.overlap_http_counts((large,), pilot_only=True):
        assert diet._PREFETCHED_HTTP_COUNTS.get() is None
    assert posts == ["small"]


@pytest.mark.parametrize("exception", (TimeoutError, ValueError, ConnectionResetError))
def test_consumed_failure_retries_before_later_candidate(monkeypatch, pilot, exception):
    results = []
    for enabled in (False, True):
        pilot.enabled = lambda: enabled
        calls = []
        first = _prompt("first")
        second = _prompt("second")
        monkeypatch.setattr(diet.RequestTokenMeter, "_tokenizer_capability_proven", lambda _self: True)

        def post(content, _url, **_kwargs):
            label = json.loads(content)["messages"][0]["content"]
            calls.append(label)
            if label == "first":
                raise exception("failed candidate")
            return [1]

        monkeypatch.setattr(diet, "_http_tokenize", post)
        meter = _pinned_meter("http://127.0.0.1:51")
        with meter.overlap_http_counts((first, second), pilot_only=True):
            results.append((meter.measure(first), meter.unavailable_reason(), meter.measure(second)))
        assert calls.count("first") == diet.TOKENIZE_MAX_CONSECUTIVE_FAILURES
        assert meter._tokenizer_calls_by_request == {None: diet.TOKENIZE_MAX_CONSECUTIVE_FAILURES}
    assert results[0] == results[1]
