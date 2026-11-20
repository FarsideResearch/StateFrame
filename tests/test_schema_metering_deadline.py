"""A trickling tokenizer cannot hold startup schema admission open."""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Event, Thread, current_thread
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from _journal import measurement_journal_paths
from kernel_agent_schemas import measure_schemas
from kernel_agent_tools import AgentTools
from kernel_custom_tools import CustomToolSet
from kernel_tool_settings import ToolSettings
from stateframe.scripted_endpoint import Script, make_handler


CAP = 0.25
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox


@contextmanager
def tokenizer(*, trickle, trickle_posts=1):
    script = Script(json.loads((ROOT / "examples/discount_bug.script.json").read_text()))
    schema_posts = []

    class Handler(make_handler(script)):
        def _tokenize(self, raw):
            content = json.loads(json.loads(raw)["content"])
            if content.get("messages") != [] or not content.get("tools"):
                return super()._tokenize(raw)
            schema_posts.append(content)
            if trickle == "after-exact" and content["tools"][0]["function"]["name"] == "shell":
                time.sleep(CAP / 3)
                self._send(200, {"tokens": [1, 2, 3]})
                return
            if not trickle or len(schema_posts) > trickle_posts:
                self._send(200, {"tokens": [1, 2, 3]})
                return
            # Each byte arrives within the socket timeout, but the full body takes over a second.
            body = b'{"tokens": [1, 2, 3]}'
            if trickle == "headers":
                body = b"HTTP/1.1 200 OK\r\nContent-Length: 21\r\n\r\n" + body
            else:
                self.send_response(200)
                if trickle == "chunked":
                    self.send_header("Transfer-Encoding", "chunked")
                    body = b"15;extension=abcdefghijklmno\r\n" + body + b"\r\n0\r\n\r\n"
                else:
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
            try:
                for byte in body:
                    self.wfile.write(bytes([byte]))
                    time.sleep(0.065)
            except (BrokenPipeError, ConnectionResetError):
                pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", schema_posts
        finally:
            server.shutdown()
            worker.join()


@pytest.mark.parametrize(("trickle", "cap"), [(True, CAP), (False, None), (False, 0)],
                         ids=["trickle", "fast-default-cap", "zero-cap"])
def test_bind_bounds_schema_metering_and_keeps_fast_counts(tmp_path, monkeypatch, trickle, cap):
    import kernel_agent_tools
    import kernel_request_diet
    from kernel_provider_dialect import LLAMA_CPP, RUN_CAPABILITY_CACHE, ProviderCapabilities, origin_of

    if cap is not None:
        monkeypatch.setattr(kernel_request_diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", cap)
    monkeypatch.setattr(kernel_agent_tools, "time", SimpleNamespace(monotonic=time.monotonic))
    fallback = measure_schemas()
    with tokenizer(trickle=trickle) as (endpoint, schema_posts):
        # Product runs seed the child's provider proof from the host preflight.
        monkeypatch.setitem(RUN_CAPABILITY_CACHE, origin_of(endpoint), ProviderCapabilities(dialect=LLAMA_CPP))
        shell = SimpleNamespace(spec={"chat_url": endpoint + "/chat/completions"},
                                run_dir=tmp_path, settings=ToolSettings())
        start = time.monotonic()
        tools = AgentTools(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
        tools.bind()
        elapsed = time.monotonic() - start

        if cap == 0:
            assert elapsed < 0.35
            assert tools.schema_budget == fallback
            assert schema_posts == []
        elif trickle:
            assert elapsed < CAP + 0.35
            assert tools.schema_budget == fallback
            assert len(schema_posts) == 1
        else:
            assert len(schema_posts) == len(fallback)
            assert tools.schema_budget == {name: {"tokens": 3, "method": "tokenize"} for name in fallback}
        assert json.loads(tools.schema_budget_path.read_text()) == tools.schema_budget


@pytest.mark.parametrize("trickle", [True, "headers", "chunked", "after-exact"],
                         ids=["body", "headers", "chunked", "after-exact"])
def test_parallel_binds_share_one_tokenizer_with_bounded_workers(tmp_path, monkeypatch, trickle):
    import kernel_request_diet
    from kernel_provider_dialect import LLAMA_CPP, RUN_CAPABILITY_CACHE, ProviderCapabilities, origin_of

    monkeypatch.setattr(kernel_request_diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", CAP)
    original_tokenize = kernel_request_diet._http_tokenize
    requests = []

    def record_worker(*args, **kwargs):
        requests.append((current_thread(), kwargs["deadline"]))
        return original_tokenize(*args, **kwargs)

    monkeypatch.setattr(kernel_request_diet, "_http_tokenize", record_worker)
    fallback = measure_schemas()
    callers = 4
    posts_per_bind = 2 if trickle == "after-exact" else 1
    ready = Barrier(callers)
    with tokenizer(trickle=trickle, trickle_posts=callers * posts_per_bind) as (endpoint, schema_posts):
        monkeypatch.setitem(RUN_CAPABILITY_CACHE, origin_of(endpoint), ProviderCapabilities(dialect=LLAMA_CPP))

        def bind(index):
            run_dir = tmp_path / str(index)
            run_dir.mkdir()
            shell = SimpleNamespace(spec={"chat_url": endpoint + "/chat/completions"},
                                    run_dir=run_dir, settings=ToolSettings())
            ready.wait(timeout=5)
            start = time.monotonic()
            tools = AgentTools(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
            tools.bind()
            return time.monotonic() - start, tools.schema_budget

        with ThreadPoolExecutor(max_workers=callers) as pool:
            for elapsed, counts in pool.map(bind, range(callers)):
                assert elapsed < CAP + 0.35
                assert counts == fallback
        assert len(schema_posts) == len(requests) == callers * posts_per_bind
        workers = {worker for worker, _ in requests}
        assert len(workers) == callers
        for worker in workers:
            assert len({deadline for caller, deadline in requests if caller is worker}) == 1
        cleanup_deadline = time.monotonic() + 0.35
        for worker in workers:
            worker.join(timeout=max(0, cleanup_deadline - time.monotonic()))
        assert not any(worker.is_alive() for worker in workers)


@pytest.mark.parametrize("stall", ["resolution", "first-address"])
def test_expired_schema_worker_does_not_start_a_connection(tmp_path, monkeypatch, stall):
    import kernel_request_diet
    from kernel_provider_dialect import LLAMA_CPP, RUN_CAPABILITY_CACHE, ProviderCapabilities, origin_of

    monkeypatch.setattr(kernel_request_diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", CAP)
    original_resolve = socket.getaddrinfo
    original_connect = socket.socket.connect
    started = Event()
    release = Event()
    workers = []
    connections = []

    with tokenizer(trickle=True) as (endpoint, _):
        port = urlsplit(endpoint).port
        monkeypatch.setitem(RUN_CAPABILITY_CACHE, origin_of(endpoint), ProviderCapabilities(dialect=LLAMA_CPP))

        def wait_for_bind():
            workers.append(current_thread())
            started.set()
            assert release.wait(5), "test did not release the tokenizer worker"

        def resolve(host, service, *args, **kwargs):
            addresses = original_resolve(host, service, *args, **kwargs)
            if (host, service) != ("127.0.0.1", port):
                return addresses
            if stall == "resolution":
                wait_for_bind()
                return addresses
            family, kind, protocol, canonical, _ = addresses[0]
            return [(family, kind, protocol, canonical, ("127.0.0.2", port)), *addresses]

        def connect(sock, address):
            if address == ("127.0.0.2", port):
                wait_for_bind()
                raise TimeoutError("first address did not respond")
            if address == ("127.0.0.1", port):
                connections.append(address)
            return original_connect(sock, address)

        monkeypatch.setattr(socket, "getaddrinfo", resolve)
        monkeypatch.setattr(socket.socket, "connect", connect)
        shell = SimpleNamespace(spec={"chat_url": endpoint + "/chat/completions"},
                                run_dir=tmp_path, settings=ToolSettings())
        start = time.monotonic()
        tools = AgentTools(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
        try:
            assert started.wait(1)
            tools.bind()
            assert time.monotonic() - start < CAP + 0.35
            assert tools.schema_budget == measure_schemas()
        finally:
            release.set()
            for worker in workers:
                worker.join(timeout=1)
        assert not any(worker.is_alive() for worker in workers)
        assert connections == []


def test_stuck_address_resolution_does_not_hold_interpreter_open(tmp_path):
    child = """
import socket
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

sys.path[:0] = sys.argv[2:]
import kernel_request_diet
from kernel_agent_tools import AgentTools
from kernel_custom_tools import CustomToolSet
from kernel_provider_dialect import LLAMA_CPP, RUN_CAPABILITY_CACHE, ProviderCapabilities
from kernel_tool_settings import ToolSettings

started = Event()
blocked = Event()

def resolve(*args, **kwargs):
    started.set()
    blocked.wait()

socket.getaddrinfo = resolve
kernel_request_diet.TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS = 0.25
origin = "http://127.0.0.1:9"
RUN_CAPABILITY_CACHE[origin] = ProviderCapabilities(dialect=LLAMA_CPP)
shell = SimpleNamespace(spec={"chat_url": origin + "/v1/chat/completions"},
                        run_dir=Path(sys.argv[1]), settings=ToolSettings())
tools = AgentTools(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
assert started.wait(3), "tokenizer did not reach address resolution"
tools.bind()
assert tools.startup_schema_budget["reason"] == "deadline"
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", child, str(tmp_path), str(ROOT / "src"), str(ROOT / "src/stateframe/engine")],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_resume_before_first_send_can_start_without_a_journal(tmp_path, monkeypatch):
    import kernel_request_diet

    monkeypatch.setattr(kernel_request_diet, "_http_tokenize", lambda *_args, **_kwargs: [1, 2, 3])
    archive = tmp_path / "attempts" / "1"
    archive.mkdir(parents=True)
    shell = SimpleNamespace(spec={"chat_url": "http://127.0.0.1:9/v1/chat/completions", "resume": {
        "previous_attempt": str(archive), "tape": [],
        "checkpoint": {"status": "skipped", "reason": "hooks_configured"},
    }}, run_dir=tmp_path, settings=ToolSettings())

    tools = AgentTools(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
    tools.bind()

    assert tools.schema_budget == {name: {"tokens": 3, "method": "tokenize"} for name in measure_schemas()}


def test_replay_refuses_a_journal_without_startup_counts(tmp_path, monkeypatch):
    import kernel_request_diet
    from kernel_continuation_journal import ContinuationJournal
    from kernel_tool_tape import ToolReplayRefused

    calls = []

    def tokenize(*args, **kwargs):
        calls.append(args)
        return [1, 2, 3]

    monkeypatch.setattr(kernel_request_diet, "_http_tokenize", tokenize)
    archive = tmp_path / "attempts" / "1"
    wire = archive / "wire"
    wire.mkdir(parents=True)
    request = wire / "0001.request.json"
    request.write_text("{}")
    journals = archive / "worker" / "measurement-runs"
    journals.mkdir(parents=True)
    ContinuationJournal(journals, "task", {}).close()
    shell = SimpleNamespace(spec={"chat_url": "http://127.0.0.1:9/v1/chat/completions", "resume": {
        "previous_attempt": str(archive), "tape": [{"request": str(request)}],
    }}, run_dir=tmp_path, settings=ToolSettings())

    with pytest.raises(ToolReplayRefused, match="Recorded startup schema counts are missing"):
        tools = AgentTools(shell, CustomToolSet((), "empty"), defer_schema_budget=True)
        tools.bind()
    assert calls == []


@pytest.mark.parametrize("archive_interrupted", [False, True], ids=["normal", "interrupted-archive"])
def test_startup_fallback_is_journaled_and_replayed_byte_for_byte(tmp_path, monkeypatch, archive_interrupted):
    from kernel_journal_store import open_journal
    from kernel_measurement import resolve_journal_blob_refs
    from stateframe import cli, runner

    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    run = tmp_path / "run"
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    monkeypatch.delenv("LLAMA_CPP_TOKENIZE_URL", raising=False)
    child = tmp_path / "child.py"
    child.write_text(f"""
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, {str(runner.CHILD.parent)!r})
engine = runpy.run_path({str(runner.CHILD)!r}, run_name="schema_deadline_child")
state = engine["main"].__globals__
install_identity = state["install_run_identity"]

def install_test_cap(seed):
    install_identity(seed)
    spec = json.loads((Path(sys.argv[1]) / "spec.json").read_text())
    sys.path.insert(0, spec["engine_dir"])
    import kernel_request_diet
    kernel_request_diet.TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS = {CAP!r}

state["install_run_identity"] = install_test_cap
raise SystemExit(engine["_run_cli"](sys.argv))
""")
    monkeypatch.setattr(runner, "CHILD", child)

    def startup_record():
        journal, = measurement_journal_paths(run)
        with open_journal(journal) as stream:
            started = json.loads(stream.readline())
        assert started["kind"] == "run_started"
        return resolve_journal_blob_refs(started["manifest"], journal).get("startup_schema_budget")

    with tokenizer(trickle=True) as (endpoint, schema_posts):
        assert cli.main(["run", "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
                         "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run),
                         "--test-python", sys.executable, "--no-run-env",
                         "--max-turns", "1", "--quiet"]) == 1
        wire = next((run / "wire").glob("*.request.json"))
        live = wire.read_bytes()
        record = startup_record()
        assert record == {"reason": "deadline", "counts": measure_schemas()}
        posts_after_live = len(schema_posts)
        assert posts_after_live == 1
        # Reclassify the completed first exchange as interrupted so resume replays it.
        (run / "engine-result.json").write_text(json.dumps({"result": {"result": "CANCELLED"}}))
        if archive_interrupted:
            # Resume creates the archive and moves the journal before moving the wire tape.
            partial = run / "attempts" / "1"
            partial.mkdir(parents=True)
            shutil.copy2(run / "spec.json", partial / "spec.json")
            (run / "worker").rename(partial / "worker")
        assert cli.main(["resume", str(run), "--quiet"]) == 1
        assert json.loads((run / "sends.jsonl").read_text().splitlines()[0])["replayed"] is True
        assert wire.read_bytes() == live
        assert startup_record() == record
        assert len(schema_posts) == posts_after_live
