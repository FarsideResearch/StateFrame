"""A trickling tokenizer cannot hold startup schema admission open."""
from __future__ import annotations

import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Thread, current_thread
from types import SimpleNamespace

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


@pytest.mark.parametrize("trickle", [True, False], ids=["trickle", "fast"])
def test_bind_bounds_schema_metering_and_keeps_fast_counts(tmp_path, monkeypatch, trickle):
    import kernel_request_diet
    from kernel_provider_dialect import LLAMA_CPP, RUN_CAPABILITY_CACHE, ProviderCapabilities, origin_of

    monkeypatch.setattr(kernel_request_diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", CAP)
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

        assert schema_posts
        if trickle:
            assert elapsed < CAP + 0.35
            assert tools.schema_budget == fallback
            assert len(schema_posts) == 1
        else:
            assert tools.schema_budget == {name: {"tokens": 3, "method": "tokenize"} for name in fallback}
        assert json.loads(tools.schema_budget_path.read_text()) == tools.schema_budget


@pytest.mark.parametrize("trickle", [True, "headers", "chunked"], ids=["body", "headers", "chunked"])
def test_parallel_binds_share_one_tokenizer_with_bounded_workers(tmp_path, monkeypatch, trickle):
    import kernel_request_diet
    from kernel_provider_dialect import LLAMA_CPP, RUN_CAPABILITY_CACHE, ProviderCapabilities, origin_of

    monkeypatch.setattr(kernel_request_diet, "TOKENIZE_REQUEST_TIMEOUT_CAP_SECONDS", CAP)
    original_tokenize = kernel_request_diet._http_tokenize
    workers = []

    def record_worker(*args, **kwargs):
        workers.append(current_thread())
        return original_tokenize(*args, **kwargs)

    monkeypatch.setattr(kernel_request_diet, "_http_tokenize", record_worker)
    fallback = measure_schemas()
    callers = 4
    ready = Barrier(callers)
    with tokenizer(trickle=trickle, trickle_posts=callers) as (endpoint, schema_posts):
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
        assert len(schema_posts) == len(workers) == callers
        cleanup_deadline = time.monotonic() + 0.35
        for worker in workers:
            worker.join(timeout=max(0, cleanup_deadline - time.monotonic()))
        assert not any(worker.is_alive() for worker in workers)


def test_startup_fallback_is_journaled_and_replayed_byte_for_byte(tmp_path, monkeypatch):
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
        assert cli.main(["resume", str(run), "--quiet"]) == 1
        assert json.loads((run / "sends.jsonl").read_text().splitlines()[0])["replayed"] is True
        assert wire.read_bytes() == live
        assert startup_record() == record
        assert len(schema_posts) == posts_after_live
