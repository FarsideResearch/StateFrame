"""A profile proxy cannot intercept model requests from a real run."""

import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ._support import Endpoint, PYTHON, ROOT, environment, read_requests


pytestmark = pytest.mark.host_sandbox


def test_profile_proxy_does_not_receive_model_requests(tmp_path):
    requests = []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _reject(self):
            length = int(self.headers.get("Content-Length", "0"))
            requests.append((self.path, self.rfile.read(length)))
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):  # noqa: N802
            self._reject()

        def do_GET(self):  # noqa: N802
            self._reject()

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    try:
        proxy_url = f"http://127.0.0.1:{proxy.server_port}"
        profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
        profile["switches"].update(HTTP_PROXY=proxy_url, http_proxy=proxy_url)
        profile_path = tmp_path / "proxy-profile.json"
        profile_path.write_text(json.dumps(profile))
        repo = tmp_path / "repo"
        shutil.copytree(ROOT / "examples/discount_bug", repo)
        run_dir = tmp_path / "run"
        env = environment()
        for name in ("HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy"):
            env.pop(name, None)
        with Endpoint(ROOT / "examples/discount_bug.script.json") as endpoint:
            process = subprocess.run([
                PYTHON, "-m", "stateframe", "run", "--repo", str(repo),
                "--task", "Fix the pricing code so the tests pass.",
                "--endpoint", endpoint.url, "--model", "scripted",
                "--profile", str(profile_path), "--run-dir", str(run_dir),
                "--test-python", PYTHON, "--max-turns", "16", "--quiet",
            ], capture_output=True, text=True, env=env, timeout=240, check=False)
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        assert process.returncode == 0, process.stdout + process.stderr + str(summary)
        assert summary["succeeded"] is True
        assert read_requests(run_dir)
        assert requests == []
    finally:
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=5)
