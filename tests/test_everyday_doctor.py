"""Exercise doctor through its public CLI with isolated host and provider fixtures."""
from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from _stateframe_home import isolated_stateframe_home
from stateframe import cli

MODEL_LIST = {"data": [{"id": "local-model"}]}
MARKER = "private-fixture-marker"


@pytest.fixture
def probe_module():
    from stateframe.engine import kernel_safe_probe
    return kernel_safe_probe


@pytest.fixture
def host(tmp_path, monkeypatch, probe_module):
    from stateframe.engine import kernel_test_isolation

    for name in ("ENDPOINT", "MODEL", "ALLOW_REMOTE_ENDPOINT", "CONTEXT_TOKENS", "PROMPT_BUDGET",
                 "ENDPOINT_MAX_JSON_BYTES"):
        monkeypatch.delenv("STATEFRAME_" + name, raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("STATEFRAME_HOME", isolated_stateframe_home())
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    system_bin = tmp_path / "system-bin"
    system_bin.mkdir()
    bwrap = system_bin / "bwrap"
    bwrap.touch(mode=0o755)
    monkeypatch.setenv("PATH", str(system_bin))
    monkeypatch.setattr(kernel_test_isolation, "_SANDBOX_DIRS", (str(system_bin),))
    probes = []

    def probe(argv, directory, **kwargs):
        assert argv[:4] == [sys.executable, "-I", "-B", "-c"]
        assert directory.is_dir() and not directory.is_relative_to(workspace)
        probes.append(directory)
        return SimpleNamespace(exit_code=0, output=b"", timed_out=False, output_truncated=False)

    monkeypatch.setattr(probe_module, "run_probe", probe)
    monkeypatch.setattr(cli, "run", lambda *_a, **_kw: pytest.fail("doctor started a run"))
    yield probes
    assert list(workspace.iterdir()) == []
    assert all(not path.exists() for path in probes)


def report(capsys, *args, status=0):
    code = cli.main(["doctor", *args, "--json"])
    output = capsys.readouterr()
    assert MARKER not in output.out + output.err
    document = json.loads(output.out)
    assert document["exit_code"] == code == status
    assert document["succeeded"] is (code == 0)
    return document["checks"]


@contextmanager
def provider(models=(200, MODEL_LIST), slots=(200, [{"n_ctx": 65536}])):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.command, self.path, self.headers.get("Authorization")))
            status, payload = {"/v1/models": models, "/slots": slots}.get(self.path, (405, {}))
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = do_GET

        def log_message(self, *_args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", seen
        finally:
            server.shutdown()
            thread.join()


def test_host_only(host, capsys, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a: pytest.fail("host-only network access"))
    checks = report(capsys)
    assert len(host) == 1
    assert checks["python"]["status"] == checks["sandbox"]["status"] == "ready"
    assert all(checks[name]["status"] == "skipped" for name in ("endpoint", "model", "slots"))
    assert cli.main(["doctor"]) == 0
    output = capsys.readouterr().out.lower()
    assert "sandbox" in output and "skipped" in output


def test_namespace_denied(host, capsys, monkeypatch, probe_module):
    def denied(*_args, **_kwargs):
        return SimpleNamespace(exit_code=1, output=b"bwrap: namespace denied",
                               timed_out=False, output_truncated=False)

    monkeypatch.setattr(probe_module, "run_probe", denied)
    check = report(capsys, status=2)["sandbox"]
    assert check["status"] == "failed"
    assert "namespace" in check["detail"].lower()


def test_sandbox_uses_system_binary_outside_path(host, capsys, monkeypatch, tmp_path):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", str(venv_bin))
    checks = report(capsys)
    assert checks["sandbox"]["status"] == "ready"
    assert len(host) == 1


def test_root_probe_recommends_non_root_user(host, capsys, monkeypatch, probe_module):
    def refused(*_args, **_kwargs):
        raise probe_module.ExecSessionError("Linux execution requires a non-root host UID")

    monkeypatch.setattr(probe_module, "run_probe", refused)
    check = report(capsys, status=2)["sandbox"]
    assert check["status"] == "failed"
    assert "non-root user" in check["detail"].lower()


@pytest.mark.parametrize("with_provider", [False, pytest.param(True, marks=pytest.mark.host_sandbox)])
def test_unsupported_host_cli_without_fcntl(tmp_path, monkeypatch, with_provider):
    monkeypatch.setenv("PYTHONPATH", str(Path(cli.__file__).resolve().parents[1]))
    monkeypatch.delenv("STATEFRAME_ENDPOINT", raising=False)
    monkeypatch.delenv("STATEFRAME_MODEL", raising=False)
    startup = """
import platform
import runpy
import sys

sys.modules["fcntl"] = None
platform.system = lambda: "Windows"
sys.argv = ["stateframe", "doctor", "--json", *sys.argv[1:]]
runpy.run_module("stateframe", run_name="__main__")
"""
    with provider() if with_provider else nullcontext((None, [])) as (url, seen):
        route = ["--endpoint", url, "--model", "local-model"] if with_provider else []
        result = subprocess.run([sys.executable, "-B", "-c", startup, *route], cwd=tmp_path,
                                capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert result.stderr == ""
    document = json.loads(result.stdout)
    assert document["exit_code"] == 2 and document["succeeded"] is False
    check = document["checks"]["sandbox"]
    assert check["status"] == "failed" and "Linux or macOS" in check["detail"]
    if with_provider:
        assert document["checks"]["model"]["status"] == "ready"
        assert document["checks"]["slots"]["status"] == "unknown"
        assert seen == [("GET", "/v1/models", None)]


def test_sandbox_probe_unavailable(host, capsys, monkeypatch, probe_module):
    def unavailable(*_args, **_kwargs):
        raise OSError(MARKER)

    monkeypatch.setattr(probe_module, "run_probe", unavailable)
    check = report(capsys, status=2)["sandbox"]
    assert check["status"] == "untested" and "permissions" in check["detail"]


@pytest.mark.parametrize("system,guidance", [
    ("Linux", "install"), ("Darwin", "sandbox-exec"), ("Other", "Linux"),
])
def test_sandbox_prerequisites(host, capsys, monkeypatch, system, guidance):
    from stateframe.engine import kernel_test_isolation

    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(kernel_test_isolation, "_SANDBOX_DIRS", ())
    check = report(capsys, status=2)["sandbox"]
    assert check["status"] == "failed" and guidance in check["detail"]
    assert not host


def test_excluded_python(host, capsys, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 14, 1))
    check = report(capsys, status=2)["python"]
    assert "3.14.1" in check["detail"] and check["status"] == "failed"


@pytest.mark.parametrize("args,guidance", [
    (["--model", "local-model"], "--endpoint"),
    (["--endpoint", "http://localhost"], "--model"),
    (["--api-key-env", "LOCAL_KEY"], "--endpoint"),
    (["--endpoint", "http://localhost", "--model", "local-model", "--api-key-env", MARKER], "--api-key-env"),
    (["--endpoint", f"http://localhost/v1?x={MARKER}", "--model", "local-model"], "--endpoint"),
    (["--endpoint", f"http://u:{MARKER}@localhost", "--model", "local-model"], "--endpoint"),
    (["--endpoint", "http://remote.invalid", "--model", "local-model"], "loopback"),
])
def test_configuration_failures(host, capsys, monkeypatch, args, guidance):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a: pytest.fail("invalid route was probed"))
    check = report(capsys, *args, status=2)["endpoint"]
    assert check["status"] == "failed" and guidance in check["detail"]


def test_remote_override_keeps_doctor_local(host, capsys, monkeypatch):
    monkeypatch.setenv("STATEFRAME_ALLOW_REMOTE_ENDPOINT", "1")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a: pytest.fail("doctor contacted a remote provider"))
    checks = report(capsys, "--endpoint", "http://remote.invalid", "--model", "local-model", status=2)
    assert checks["endpoint"]["status"] == "failed" and "loopback" in checks["endpoint"]["detail"]


@pytest.mark.parametrize("json_mode", [False, True])
def test_malformed_doctor_arguments_hide_unexpected_values(host, capsys, json_mode):
    flags = ["--json"] if json_mode else []
    code = cli.main(["doctor", "--api-key-env", "LOCAL_KEY", MARKER, *flags])
    captured = capsys.readouterr()
    assert code == 2 and MARKER not in captured.err + captured.out
    if json_mode:
        assert json.loads(captured.out)["exit_code"] == code
    assert not host


@pytest.mark.host_sandbox
def test_healthy_env_route(host, capsys, monkeypatch):
    with provider() as (url, seen):
        monkeypatch.setenv("STATEFRAME_ENDPOINT", url)
        monkeypatch.setenv("STATEFRAME_MODEL", "local-model")
        monkeypatch.setenv("LOCAL_KEY", MARKER)
        checks = report(capsys, "--api-key-env", "LOCAL_KEY")
    assert all(checks[name]["status"] == "ready" for name in ("endpoint", "model", "slots"))
    assert str(65536) in checks["slots"]["detail"].replace(",", "")
    assert checks["tool_calls"]["status"] == "skipped"
    assert seen == [("GET", path, f"Bearer {MARKER}") for path in ("/v1/models", "/slots")]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("listing,slot_data", [
    ((404, {}), (404, {})), ((200, {"data": []}), (200, [{"n_ctx": True}])),
])
def test_unknown_capabilities(host, capsys, listing, slot_data):
    with provider(listing, slot_data) as (url, seen):
        checks = report(capsys, "--endpoint", url, "--model", "local-model")
    assert checks["endpoint"]["status"] == "ready"
    assert checks["slots"]["status"] == "unknown"
    assert checks["model"]["status"] == "unknown"
    assert [path for _, path, _ in seen] == ["/v1/models", "/slots"]


@pytest.mark.host_sandbox
def test_wrong_model_and_smallest_slot(host, capsys, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CONTEXT_TOKENS", "32768")
    monkeypatch.setenv("STATEFRAME_MODEL", "local-model")
    with provider(slots=(200, [{"n_ctx": 65536}, {"n_ctx": 8192}])) as (url, _):
        checks = report(capsys, "--endpoint", url, "--model", MARKER, status=2)
    assert checks["model"]["status"] == "failed" and "--model" in checks["model"]["detail"]
    assert checks["slots"]["status"] == "failed"
    detail = checks["slots"]["detail"].replace(",", "")
    assert "8192" in detail and "32768" in detail and "context" in detail.lower()


@pytest.mark.host_sandbox
def test_invalid_context_setting(host, capsys, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CONTEXT_TOKENS", MARKER)
    with provider() as (url, _):
        checks = report(capsys, "--endpoint", url, "--model", "local-model", status=2)
    assert checks["slots"]["status"] == "failed" and "STATEFRAME_CONTEXT_TOKENS" in checks["slots"]["detail"]


@pytest.mark.host_sandbox
@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure(host, capsys, status):
    with provider((status, {"error": MARKER})) as (url, seen):
        checks = report(capsys, "--endpoint", url, "--model", "local-model", status=2)
    assert "--api-key-env" in checks["endpoint"]["detail"]
    assert checks["endpoint"]["status"] == "failed" and len(seen) == 1


@pytest.mark.host_sandbox
def test_connection_refused(host, capsys):
    # Reserve an ephemeral loopback port without listening on it.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        check = report(capsys, "--endpoint", url, "--model", "local-model", status=2)["endpoint"]
    assert check["status"] == "failed" and "server" in check["detail"]


def test_help(capsys):
    for args in (["--help"], ["doctor", "--help"]):
        with pytest.raises(SystemExit) as stop:
            cli.main(args)
        assert stop.value.code == 0
        output = capsys.readouterr().out
        assert "doctor" in output
        if args[0] == "doctor":
            assert all(flag in output for flag in ("--endpoint", "--model", "--api-key-env", "--json"))
