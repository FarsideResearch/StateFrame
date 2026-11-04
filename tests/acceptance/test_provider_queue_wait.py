"""A queued first model send survives its request cap in a complete run."""

from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

from stateframe.engine.kernel_journal_store import journal_paths, open_journal

from ._support import Endpoint, PYTHON, ROOT, environment


@pytest.mark.host_sandbox
def test_first_queued_send_submits_once_and_run_succeeds(tmp_path):
    posts: list[bytes] = []
    holding = threading.Event()
    with Endpoint(ROOT / "examples/discount_bug.script.json") as scripted:
        upstream = urlsplit(scripted.url)

        class QueueProxy(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _reply(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if holding.is_set() and self.path in {"/slots", "/health"}:
                    self._reply(401 if self.path == "/slots" else 503, b"busy")
                    return
                self._forward(None)

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == "/v1/chat/completions":
                    posts.append(body)
                    if len(posts) == 1:
                        holding.set()
                        time.sleep(4.2)
                self._forward(body)

            def _forward(self, body: bytes | None) -> None:
                connection = http.client.HTTPConnection(
                    upstream.hostname, upstream.port, timeout=20)
                try:
                    headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
                    if self.headers.get("Authorization"):
                        headers["Authorization"] = self.headers["Authorization"]
                    connection.request("POST" if body is not None else "GET",
                                       self.path, body=body, headers=headers)
                    response = connection.getresponse()
                    self._reply(response.status, response.read())
                finally:
                    connection.close()

        proxy = ThreadingHTTPServer(("127.0.0.1", 0), QueueProxy)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        try:
            repo = tmp_path / "repo"
            shutil.copytree(ROOT / "examples/discount_bug", repo)
            run = tmp_path / "run"
            profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
            profile["switches"].update(
                STATEFRAME_PER_REQUEST_TIMEOUT_S="2",
                STATEFRAME_QUEUE_WAIT_MAX_S="8",
                STATEFRAME_PROVIDER_LIVENESS_S="0.5")
            profile_path = tmp_path / "queue-profile.json"
            profile_path.write_text(json.dumps(profile))
            # The run-level budget is generous: loaded CI hosts (load 60-80) exhausted 60 s on the
            # whole run. The queue timings under test (2 s cap, 4.2 s hold, 8 s max) are unchanged.
            result = subprocess.run([
                PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
                "--task", "Fix the pricing code so the tests pass.",
                "--endpoint", f"http://127.0.0.1:{proxy.server_port}/v1",
                "--model", "scripted", "--profile", str(profile_path),
                "--run-dir", str(run),
                "--test-python", PYTHON, "--max-turns", "16", "--timeout", "240",
                "--quiet",
            ], capture_output=True, text=True, env=environment(), timeout=300, check=False)
            if ("engine interpreter" in result.stderr
                    and "sandbox_apply: Operation not permitted" in result.stderr):
                pytest.skip("nested host sandbox prevents the engine startup probe")
            assert (run / "summary.json").is_file(), result.stdout + result.stderr
            summary = json.loads((run / "summary.json").read_text())
            assert result.returncode == 0, result.stdout + result.stderr + str(summary)
            assert summary["succeeded"] is True
            assert posts
            assert posts.count(posts[0]) == 1
            assert "warning: the model server queued requests" in result.stdout
            # A finished run seals its journal as events.jsonl.xz; read through the store so either layout counts.
            journals = journal_paths(run / "worker" / "measurement-runs")
            assert len(journals) == 1
            with open_journal(journals[0]) as source:
                events = [json.loads(line) for line in source if line.strip()]
            waits = [event for event in events if event.get("kind") == "provider_queue_wait"]
            assert len(waits) == 1
            assert waits[0]["seconds_waited"] >= 2
            assert waits[0]["probe_http_status"] == 401
        finally:
            proxy.shutdown()
            proxy.server_close()
            thread.join(timeout=5)
