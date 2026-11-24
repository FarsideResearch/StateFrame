from __future__ import annotations

import json
import os
import sys
import signal
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread

import pytest

from _journal import measurement_events
from engine_run import run_monitored, start_monitored
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox

_TASK_TEXT = "Read app.py and report VALUE."
_SOURCE_TEXT = "VALUE = 6"
_PROVIDER_MESSAGE = f"Proxy echoed task: {_TASK_TEXT}; source: {_SOURCE_TEXT}."
_AUTH_KEY = "sk-provider-4xx-test-secret"

_STATUS_MESSAGES = {
    400: "provider returned HTTP 400 (request rejected)",
    401: "provider returned HTTP 401 (authentication rejected); check the API key",
    403: "provider returned HTTP 403 (access denied)",
    404: "provider returned HTTP 404 (model or endpoint not found)",
}


class ProviderEndpoint:
    """Local OpenAI-compatible endpoint with a deterministic status sequence."""

    def __init__(self, statuses: tuple[int | None, ...], *, dialect: str = "openai-strict",
                 error_bodies: tuple[dict[str, object] | None, ...] = (),
                 success_tool: str = "read", pause_at: int | None = None) -> None:
        self.statuses = statuses
        self.error_bodies = error_bodies
        self.dialect = dialect
        self.success_tool = success_tool
        self.pause_at = pause_at
        self.waiting = Event()
        self.release = Event()
        self.requests: list[bytes] = []
        parent = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: object) -> None:
                pass

            def _reply(self, status: int, payload: object) -> None:
                raw = json.dumps(payload, separators=(",", ":")).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/props" and parent.dialect == "llama.cpp":
                    self._reply(200, {"default_generation_settings": {"n_ctx": 65536}})
                elif self.path == "/slots" and parent.dialect == "llama.cpp":
                    self._reply(200, [{"id": 0, "n_ctx": 65536, "is_processing": False}])
                else:
                    self._reply(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if self.path.rstrip("/") != "/v1/chat/completions":
                    self._reply(404, {"error": "not found"})
                    return
                parent.requests.append(raw)
                request_number = len(parent.requests)
                if request_number == parent.pause_at:
                    parent.waiting.set()
                    parent.release.wait(90)
                if request_number <= len(parent.statuses):
                    status = parent.statuses[request_number - 1]
                    if status is not None:
                        error = (parent.error_bodies[request_number - 1]
                                 if request_number <= len(parent.error_bodies) else None)
                        if error is None:
                            error = {"error": {
                                "message": _PROVIDER_MESSAGE,
                                "type": "authentication_error",
                                "code": "model_not_allowed",
                                "debug": f"Authorization: Bearer {_AUTH_KEY}",
                            }}
                        self._reply(status, error)
                        return
                body = json.loads(raw)
                tool_names = {
                    tool.get("function", {}).get("name")
                    for tool in body.get("tools", ()) if isinstance(tool, dict)
                }
                if parent.success_tool == "edit" and "str_replace" in tool_names:
                    tool = "str_replace"
                    arguments = {"path": "app.py", "old_str": "VALUE = 6\n",
                                 "new_str": "VALUE = 7\n", "why": "Set the requested value."}
                elif parent.success_tool == "done" and "done" in tool_names:
                    tool = "done"
                    arguments = {"summary": "The task is complete."}
                elif "read" in tool_names:
                    tool = "read"
                    arguments = {"path": "app.py"}
                else:
                    tool = "done"
                    arguments = {"summary": "The task is complete."}
                self._reply(200, {
                    "id": "chatcmpl-provider-4xx-test",
                    "object": "chat.completion",
                    "model": "provider-4xx-test",
                    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                        "role": "assistant", "content": "", "tool_calls": [{
                            "id": f"call-{request_number}", "type": "function",
                            "function": {"name": tool, "arguments": json.dumps(arguments)},
                        }],
                    }}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                              "total_tokens": 110},
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> ProviderEndpoint:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"


def _new_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text(_SOURCE_TEXT + "\n", encoding="utf-8")
    return repo


def _environment(home: Path, *, source_root: Path = ROOT) -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("STATEFRAME_", "REACT_"))}
    environment.update({
        "HOME": str(home),
        "PYTHONPATH": str(source_root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": str(ROOT / ".tmp"),
        "SF_PROVIDER_TEST_KEY": _AUTH_KEY,
    })
    return environment


def _run_process(command: list[str], run_dir: Path, environment: dict[str, str]):
    return run_monitored(
        command, cwd=ROOT, env=environment, capture_output=True, text=True,
        expected_seconds=90, run_dir=run_dir, check=False,
    )


def _run_cli(tmp_path: Path, statuses: tuple[int | None, ...], *,
             dialect: str = "openai-strict",
             error_bodies: tuple[dict[str, object] | None, ...] = (),
             success_tool: str = "read",
             max_turns: int = 4):
    repo = _new_repo(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    run_dir = tmp_path / "run"
    environment = _environment(home)
    if dialect == "llama.cpp":
        environment["STATEFRAME_PROVIDER"] = "llama.cpp"
    with ProviderEndpoint(statuses, dialect=dialect, error_bodies=error_bodies,
                          success_tool=success_tool) as endpoint:
        process = _run_process(
            [sys.executable, "-B", "-m", "stateframe", "run",
             "--repo", str(repo), "--task", _TASK_TEXT,
             "--endpoint", endpoint.url, "--model", "provider-4xx-test",
             "--api-key-env", "SF_PROVIDER_TEST_KEY",
             "--run-dir", str(run_dir), "--max-turns", str(max_turns),
             "--test-python", sys.executable,
             "--shell", "off", "--no-project-hooks", "--no-instructions",
             "--timeout", "120", "--quiet", "--json"], run_dir, environment,
        )
        requests = tuple(endpoint.requests)
    output = process.stdout + process.stderr
    assert (run_dir / "summary.json").is_file(), output
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return process.returncode, summary, tuple(measurement_events(run_dir)), output, requests


def _run_chat(tmp_path: Path, statuses: tuple[int | None, ...], *, dialect: str,
              monkeypatch: pytest.MonkeyPatch):
    from stateframe import chat
    from stateframe.endpoint import Endpoint
    from stateframe.engine.kernel_tool_settings import ToolSettings
    from stateframe.runner import RunOptions

    repo = _new_repo(tmp_path)
    home = tmp_path / "chat-session"
    monkeypatch.setenv("SF_PROVIDER_TEST_KEY", _AUTH_KEY)
    if dialect == "llama.cpp":
        monkeypatch.setenv("STATEFRAME_PROVIDER", "llama.cpp")
    else:
        monkeypatch.delenv("STATEFRAME_PROVIDER", raising=False)
    lines: list[str] = []
    with ProviderEndpoint(statuses, dialect=dialect) as endpoint:
        options = RunOptions(
            repo=repo, task="", endpoint=Endpoint.from_args(
                endpoint.url, "provider-4xx-test", "SF_PROVIDER_TEST_KEY"),
            max_turns=4, python=sys.executable, test_python=sys.executable,
            timeout_s=120, quiet=True, no_project_hooks=True, instructions="off",
            tool_settings=ToolSettings(shell="off"),
        )
        session = chat.Session(repo=repo, home=home, base=options)
        session.start(printer=lines.append)
        record = session.run_task(_TASK_TEXT, printer=lines.append)
        run_dir = Path(record.run_dir)
        requests = tuple(endpoint.requests)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return summary, tuple(measurement_events(run_dir)), "\n".join(lines), requests, record


def _run_resume(tmp_path: Path, position: str, status: int, *, dialect: str):
    repo = _new_repo(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    run_dir = tmp_path / "run"
    environment = _environment(home)
    if dialect == "llama.cpp":
        environment["STATEFRAME_PROVIDER"] = "llama.cpp"
    with ProviderEndpoint((), dialect=dialect, success_tool="read", pause_at=2) as endpoint:
        launched_after = time.time()
        initial = start_monitored(
            [sys.executable, "-B", "-m", "stateframe", "run",
             "--repo", str(repo), "--task", _TASK_TEXT,
             "--endpoint", endpoint.url, "--model", "provider-4xx-test",
             "--api-key-env", "SF_PROVIDER_TEST_KEY", "--run-dir", str(run_dir),
             "--max-turns", "4", "--test-python", sys.executable,
             "--shell", "off", "--no-project-hooks", "--no-instructions",
             "--timeout", "120", "--quiet", "--json"], cwd=ROOT, env=environment,
            run_dir=run_dir, expected_seconds=90, step="provider resume fixture",
        )
        try:
            assert endpoint.waiting.wait(60), "initial run did not reach its held request"
            initial_request_count = len(endpoint.requests)
            assert initial_request_count == 2
            children = subprocess.check_output(
                ["pgrep", "-P", str(initial.pid)], text=True,
            ).split()
            assert len(children) == 1, children
            child = int(children[0])
            identity = record_identity(child, launched_after=launched_after,
                                       command_fragment="_child.py")
            signal_run_verified(
                run_dir, child, identity, signal.SIGKILL,
                deadline=time.monotonic() + 30, what="provider resume fixture",
                require_checkpoint_saved=False,
            )
            endpoint.statuses = ((None,) * initial_request_count + (status,)
                                 if position == "first" else
                                 (None,) * (initial_request_count + 1) + (status,))
            endpoint.release.set()
            initial.communicate(timeout=60)
        finally:
            endpoint.release.set()
            if initial.poll() is None:
                signal_run_owned(
                    run_dir, initial, signal.SIGKILL,
                    deadline=time.monotonic() + 10,
                    what="provider resume fixture cleanup",
                    require_checkpoint_saved=False,
                )
                initial.communicate(timeout=10)
            initial.close()
        resumed = _run_process(
            [sys.executable, "-B", "-m", "stateframe", "resume", str(run_dir),
             "--endpoint", endpoint.url, "--model", "provider-4xx-test",
             "--api-key-env", "SF_PROVIDER_TEST_KEY", "--max-turns", "4",
             "--quiet", "--json"], run_dir, environment,
        )
        requests = tuple(endpoint.requests)
    output = resumed.stdout + resumed.stderr
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return (resumed.returncode, summary, tuple(measurement_events(run_dir)), output,
            requests, initial_request_count)


def _assert_provider_failure(exit_code: int, summary: dict[str, object],
                             events: tuple[dict[str, object], ...], output: str,
                             requests: tuple[bytes, ...], *, status: int,
                             expected_requests: int) -> None:
    assert exit_code == 2, output
    assert summary["engine_result"] == "HARNESS_CONTRACT", summary
    assert summary["runtime_outcome"] == "engine_error", summary
    assert len(requests) == expected_requests, (len(requests), summary)
    assert "could not seal request" not in output + json.dumps(summary)
    terminal = summary.get("engine_terminal")
    assert isinstance(terminal, dict), summary
    assert terminal.get("error_type") == "ProviderHTTPError", terminal
    assert terminal.get("failure_origin") == "provider", terminal
    assert terminal.get("error_message") == _STATUS_MESSAGES[status], terminal
    assert summary.get("engine_error_reason") == f"provider error: {_STATUS_MESSAGES[status]}", summary
    assert _PROVIDER_MESSAGE not in output + json.dumps(summary)
    assert _TASK_TEXT not in str(terminal.get("error_message"))
    assert _SOURCE_TEXT not in str(terminal.get("error_message"))
    assert _AUTH_KEY not in output + json.dumps(summary)
    assert _TASK_TEXT not in output
    assert _SOURCE_TEXT not in output
    provider_rows = [row for row in events if row.get("kind") == "provider_error"]
    assert not provider_rows, events
    transport_rows = [row for row in events if row.get("kind") == "transport_error"]
    assert transport_rows and transport_rows[-1].get("http_status") == status, transport_rows
    assert transport_rows[-1].get("outcome") == "HARNESS_CONTRACT", transport_rows[-1]
    assert transport_rows[-1].get("failure_origin") == "harness", transport_rows[-1]
    assert transport_rows[-1].get("server_error", {}).get("message") == _PROVIDER_MESSAGE
    assert _PROVIDER_MESSAGE in str(transport_rows[-1].get("server_error_body"))
    assert _AUTH_KEY not in json.dumps(transport_rows[-1])
    worker_errors = [row for row in events if row.get("kind") == "worker_exception"]
    assert len(worker_errors) == 1, events
    assert worker_errors[0].get("failure_origin") == "provider", worker_errors[0]
    assert _PROVIDER_MESSAGE not in json.dumps(worker_errors[0])


@pytest.mark.parametrize(("status", "dialect", "position"), [
    (400, "llama.cpp", "first"), (401, "llama.cpp", "first"),
    (403, "llama.cpp", "first"), (404, "llama.cpp", "first"),
    (400, "openai-strict", "first"), (401, "openai-strict", "first"),
    (403, "openai-strict", "first"), (404, "openai-strict", "first"),
    (400, "openai-strict", "later"), (401, "openai-strict", "later"),
])
def test_cli_run_preserves_typed_provider_error(
        tmp_path: Path, status: int, dialect: str, position: str) -> None:
    statuses = (status,) if position == "first" else (None, status)
    exit_code, summary, events, output, requests = _run_cli(
        tmp_path, statuses, dialect=dialect,
    )
    _assert_provider_failure(
        exit_code, summary, events, output, requests,
        status=status, expected_requests=1 if position == "first" else 2,
    )
    assert f"HTTP {status}" in output


@pytest.mark.parametrize("status", [400, 401])
def test_chat_session_preserves_typed_provider_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    summary, events, output, requests, record = _run_chat(
        tmp_path, (status,), dialect="openai-strict", monkeypatch=monkeypatch,
    )
    _assert_provider_failure(
        summary["exit_code"], summary, events, output, requests,
        status=status, expected_requests=1,
    )
    assert record.result == "HARNESS_CONTRACT"
    assert record.error == f"provider error: {_STATUS_MESSAGES[status]}"
    assert _PROVIDER_MESSAGE not in output


@pytest.mark.parametrize("status", [400, 401])
def test_resume_preserves_typed_provider_error(
        tmp_path: Path, status: int) -> None:
    exit_code, summary, events, output, requests, initial_request_count = _run_resume(
        tmp_path, "first", status, dialect="openai-strict",
    )
    _assert_provider_failure(
        exit_code, summary, events, output, requests,
        status=status,
        expected_requests=initial_request_count + 1,
    )
    assert f"HTTP {status}" in output


@pytest.mark.parametrize(("status", "transport_class"), [
    (429, "capacity"), (503, "capacity"), (500, "upstream"),
])
def test_rate_limits_and_server_faults_keep_the_current_retry_schedule(
        tmp_path: Path, status: int, transport_class: str) -> None:
    exit_code, summary, events, output, requests = _run_cli(
        tmp_path, (status, None), success_tool="done",
    )
    errors = [row for row in events if row.get("kind") == "transport_error"]
    assert len(errors) == 1 and errors[0].get("http_status") == status, errors
    assert errors[0].get("transport_class") == transport_class, errors[0]
    retries = [row for row in events if row.get("kind") == "provider_transport_retry"]
    if status == 429:
        assert exit_code == 2 and summary["engine_result"] == "HARNESS_TRANSPORT", summary
        assert len(requests) == 1
        assert errors[0].get("retry_scheduled") is False, errors[0]
        assert not retries
    else:
        assert exit_code == 0 and summary["engine_result"] == "SUBMITTED", output
        assert len(requests) >= 2 and requests[0] == requests[1], summary
        assert errors[0].get("retry_scheduled") is True, errors[0]
        assert len(retries) == 1, retries
        from kernel_pilot import POLICY
        assert retries[0].get("retry_delay_seconds") == POLICY.transport_delay(1), retries[0]


@pytest.mark.parametrize("status", [408])
def test_other_contract_statuses_keep_the_harness_error_text(tmp_path: Path, status: int) -> None:
    exit_code, summary, events, output, requests = _run_cli(tmp_path, (status, None))
    assert exit_code == 2 and len(requests) == 1, output
    terminal = summary["engine_terminal"]
    assert summary["engine_result"] == "HARNESS_CONTRACT", summary
    assert terminal["error_type"] == "HarnessTransportError", terminal
    assert terminal["error_message"].startswith("transport failed after 1 attempt(s):"), terminal
    assert f"HTTP {status}" in terminal["error_message"]
    assert "provider returned" not in terminal["error_message"]
    assert not [row for row in events if row.get("kind") == "provider_error"]


@pytest.mark.parametrize(("body", "detail"), [
    ({"error": {"message": "Grammar validation failed", "type": "invalid_request_error",
                "code": "invalid_grammar"}}, "Grammar validation failed"),
    ({"error": {"message": "Tool schema rejected", "type": "invalid_request_error",
                "code": "invalid_tool_schema"}}, "Tool schema rejected"),
])
def test_grammar_and_tool_schema_400_use_fixed_provider_wording(
        tmp_path: Path, body: dict[str, object], detail: str) -> None:
    exit_code, summary, events, output, requests = _run_cli(
        tmp_path, (400,), error_bodies=(body,))
    expected = _STATUS_MESSAGES[400]
    assert exit_code == 2 and len(requests) == 1, output
    assert summary["engine_result"] == "HARNESS_CONTRACT", summary
    assert summary["engine_terminal"]["error_type"] == "ProviderHTTPError", summary
    assert summary["engine_terminal"]["error_message"] == expected, summary
    assert summary["engine_error_reason"] == f"provider error: {expected}", summary
    transport = [row for row in events if row.get("kind") == "transport_error"][-1]
    assert transport["server_error"]["message"] == detail
    assert not [row for row in events if row.get("kind") == "provider_error"]
    assert detail not in output + json.dumps(summary)


@pytest.mark.parametrize("relay_error", ["upstream_timeout", "relay_ttft_timeout"])
def test_relay_timeout_labels_keep_their_transport_error_class(
        tmp_path: Path, relay_error: str) -> None:
    body = {"error": {"label": relay_error, "message": "relay timed out"}}
    exit_code, summary, events, output, requests = _run_cli(
        tmp_path, (400,), error_bodies=(body,))
    assert exit_code == 2 and len(requests) == 1, output
    terminal = summary["engine_terminal"]
    assert terminal["error_type"] == "TransportTimeoutError", terminal
    assert terminal["error_message"].startswith("transport failed after 1 attempt(s):"), terminal
    transport = [row for row in events if row.get("kind") == "transport_error"][-1]
    assert transport["relay_error"] == relay_error
    assert not [row for row in events if row.get("kind") == "provider_error"]


def test_genuine_request_seal_violation_remains_harness_contract() -> None:
    from kernel_request_measurement import RequestMeasurement
    from kernel_request_seal import RequestSealError

    body = {"model": "provider-4xx-test", "messages": [{"role": "user", "content": "source"}],
            "tools": [], "max_tokens": 16}
    wire = json.dumps(body).encode()
    measured = RequestMeasurement.bind(
        body, tokenizer_identity="seal-control", request_id="r", logical_request_id="r",
        tokens=1, method="tokenize", template_overhead_tokens=0,
        max_prompt_tokens=33344, max_request_bytes=100000, wire=wire,
    )
    receipt = {"body_sha256": measured.body_sha256, "request_bytes": measured.request_bytes,
               **measured.receipt_fields()}
    changed = {**body, "messages": [{"role": "user", "content": "changed"}]}
    with pytest.raises(RequestSealError) as refused:
        measured.validate(
            changed, receipt, request_id="r", logical_request_id="r",
            tokenizer_identity="seal-control",
        )
    assert refused.value.outcome == "HARNESS_CONTRACT"
    assert refused.value.harness_contract is True
