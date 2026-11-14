"""Timeout sites (batch TB) report one stderr line per cause per run, and repeats stay silent."""
import io
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import kernel_env_preflight as preflight
import kernel_reason_log as rl
from stateframe.endpoint import Endpoint, EndpointError

LINE = re.compile(r"^\[stateframe\] (WARN|ERROR) (?P<component>[a-z_]+): .+", re.M)


@pytest.fixture(autouse=True)
def stream(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    out = io.StringIO()
    rl.configure(level="WARN", stream=out)
    yield out
    rl.reset_for_tests()


def _components(out):
    return [m.group("component") for m in LINE.finditer(out.getvalue())]


def _raise_timeout(command, root, timeout):
    raise subprocess.TimeoutExpired(list(command), timeout)


def test_preflight_command_timeout_logs_once_per_run(stream):
    command = ("python3", "-m", "pytest")
    for _ in range(2):
        evidence = preflight._run(command, Path("."), 2.5, _raise_timeout)
        # The caller still sees the same timed-out evidence; logging changes nothing.
        assert evidence.timed_out is True
        assert evidence.exit_code == 124
    assert _components(stream) == ["env_preflight"]
    assert "exceeded its 2.5s timeout" in stream.getvalue() and "no env override" in stream.getvalue()
    assert rl.run_summary()["by_cause"]["timeout:env_preflight:timeout"]["count"] == 2


def test_endpoint_probe_stall_logs_one_line_and_still_fails(stream):
    class StallingHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):  # noqa: N802
            time.sleep(1.0)  # both attempts outlive the 0.3s client timeout
            try:
                self.send_response(200)
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StallingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        with pytest.raises(EndpointError, match="unreachable"):
            Endpoint(url, "served", timeout_s=0.3).probe()
    finally:
        server.shutdown()
        server.server_close()
    assert _components(stream) == ["endpoint"]
    assert "endpoint probe timed out" in stream.getvalue()
