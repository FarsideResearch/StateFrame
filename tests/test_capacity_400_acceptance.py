# DECLARED RE-FREEZE 11-05 13:21 (release manager): compare retry time and preserve an unfit vLLM 16k task.
"""Full CLI acceptance coverage for recovery from provider context-limit HTTP 400s."""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

import pytest

from _timeouts import scaled
from _journal import measurement_events
from engine_run import run_monitored
from stateframe.scripted_endpoint import Script, make_handler
from tools.compare_no_mcp_requests import mask_wire_ids


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox

CAPACITY_ERROR = {"error": {
    "message": "This model's maximum context length is 32768 tokens. "
               "However, you requested 41652 tokens for this request.",
    "type": "invalid_request_error",
    "code": "context_length_exceeded",
}}
CONTRACT_ERROR = {"error": {
    "message": "Unsupported parameter: 'temperature' is not supported with this model.",
    "type": "invalid_request_error",
    "code": "unsupported_parameter",
}}
# Match the existing fixed-width countdown mask in test_request_tokenizer_reprobe.
_COUNTDOWN = re.compile(
    rb"sandbox_seconds_remaining=\d+\.\d \(about (?:\d+|999\+) minutes left"
    rb"(?:, ~(?:\d+|999\+|----) turns at your pace)?\)"
)


def _stable_request(body: bytes) -> bytes:
    return _COUNTDOWN.sub(b"sandbox_seconds_remaining=<masked>", mask_wire_ids(body))


class CapacityEndpoint:
    """Use the shipped mock's discovery routes and return scripted chat errors."""

    def __init__(self, errors: tuple[dict[str, object], ...]) -> None:
        self.errors = errors
        self.requests: list[bytes] = []
        self.lock = Lock()
        script = Script({"model": "capacity-mock", "strict_openai": True,
                         "steps": [], "tokenize": {"mode": "ok"}})
        parent_handler = make_handler(script)
        endpoint = self

        class Handler(parent_handler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path.rstrip("/") != "/v1/chat/completions":
                    super().do_POST()
                    return
                wire = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body_bytes = (gzip.decompress(wire)
                              if self.headers.get("Content-Encoding") == "gzip" else wire)
                with endpoint.lock:
                    endpoint.requests.append(body_bytes)
                    request_number = len(endpoint.requests)
                if request_number <= len(endpoint.errors):
                    self._send(400, endpoint.errors[request_number - 1])
                    return
                body = json.loads(body_bytes)
                offered = {item.get("function", {}).get("name")
                           for item in body.get("tools", []) if isinstance(item, dict)}
                assert {"str_replace", "run_tests", "done"} <= offered, (
                    "the success reply needs edit, check, and done tools", offered)
                self._send(200, {
                    "id": f"chatcmpl-{request_number}",
                    "object": "chat.completion",
                    "model": script.model,
                    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                        "role": "assistant", "content": "",
                        "tool_calls": [
                            {"id": f"call_{request_number}_edit", "type": "function",
                             "function": {"name": "str_replace", "arguments": json.dumps({
                                 "path": "app.py", "old_str": "VALUE = 6",
                                 "new_str": "VALUE = 7"})}},
                            {"id": f"call_{request_number}_tests", "type": "function",
                             "function": {"name": "run_tests", "arguments": json.dumps({
                                 "target": "tests/test_app.py"})}},
                            {"id": f"call_{request_number}_done", "type": "function",
                             "function": {"name": "done", "arguments": json.dumps({
                                 "summary": "app.py already has VALUE set to seven."})}},
                        ],
                    }}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                              "total_tokens": 110},
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> CapacityEndpoint:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=scaled(2))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"


@dataclass(frozen=True)
class Case:
    exit_code: int
    summary: dict[str, object]
    requests: tuple[bytes, ...]
    events: tuple[dict[str, object], ...]
    output: str


def _run_case(
    work: Path,
    errors: tuple[dict[str, object], ...],
    *,
    task: str = "Fix app.py so VALUE is 7.",
) -> Case:
    repo = work / "repo"
    repo.mkdir(parents=True)
    (repo / "app.py").write_text("VALUE = 6\n", encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text(
        "from app import VALUE\n\ndef test_value():\n"
        "    assert VALUE == 7\n", encoding="utf-8")
    home = work / "home"
    home.mkdir()
    run = work / "run"
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("STATEFRAME_", "REACT_"))}
    environment.update({
        "HOME": str(home),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    })
    with CapacityEndpoint(errors) as endpoint:
        process = run_monitored(
            [sys.executable, "-B", "-m", "stateframe", "run",
             "--repo", str(repo), "--task", task,
             "--endpoint", endpoint.url, "--model", "capacity-mock",
             "--run-dir", str(run), "--max-turns", "1",
             "--test-cmd", "python3 -m pytest -q tests/test_app.py",
             "--runner-kind", "pytest",
             "--test-python", sys.executable, "--target", "app.py",
             "--edit", "app.py", "--shell", "off",
             "--no-project-hooks", "--no-instructions", "--timeout", "180",  # declared re-freeze (chief 13:04): at load ~100 on 32 CPUs the no-retry control took 106 s and the retry 117 s; 45 s failed the control too
             "--quiet", "--json"],
            cwd=ROOT, env=environment, capture_output=True, text=True,
            expected_seconds=55, run_dir=run, check=False,
        )
        requests = tuple(endpoint.requests)
    output = process.stdout + process.stderr
    traceback = run / "child-traceback.txt"
    if traceback.is_file():
        output += "\n" + traceback.read_text(encoding="utf-8")
    if ("could not start in the isolated startup probe" in output
            and "sandbox_apply: Operation not permitted" in output):
        pytest.skip("host sandbox is unavailable")
    assert (run / "summary.json").is_file(), output
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return Case(process.returncode, summary, requests,
                tuple(measurement_events(run)), output)


def _assert_base_terminal(case: Case, engine_result: str, runtime_outcome: str) -> None:
    """Observed on 67563ee before adding capacity recovery."""
    assert case.exit_code == 2, case.output
    assert case.summary["engine_result"] == engine_result, case.summary
    assert case.summary["runtime_outcome"] == runtime_outcome, case.summary
    assert case.summary["succeeded"] is False, case.summary


@pytest.fixture(scope="module")
def capacity_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Case, Case]:
    base = tmp_path_factory.mktemp("capacity-400-recovery")
    return (_run_case(base / "first", (CAPACITY_ERROR,)),
            _run_case(base / "second", (CAPACITY_ERROR,)))


def test_capacity_400_then_success_retries_with_smaller_request(
        capacity_runs: tuple[Case, Case]) -> None:
    case = capacity_runs[0]
    assert len(case.requests) == 2, (len(case.requests), case.summary, case.output)
    assert len(case.requests[1]) < len(case.requests[0])
    assert case.exit_code == 0, case.output
    assert case.summary["engine_result"] == "SUBMITTED", case.summary
    assert case.summary["succeeded"] is True, case.summary


def test_two_capacity_400s_stop_with_base_terminal_and_capacity_reason(tmp_path: Path) -> None:
    case = _run_case(tmp_path, (CAPACITY_ERROR, CAPACITY_ERROR))
    _assert_base_terminal(case, "HARNESS_TRANSPORT", "missing_result")
    assert 1 <= len(case.requests) <= 2
    transport = [event for event in case.events if event.get("kind") == "transport_error"]
    assert transport, case.events
    assert any(event.get("transport_class") == "capacity" for event in transport), transport
    assert "context_length_exceeded" in json.dumps(transport), transport


def test_non_capacity_400_does_not_retry(tmp_path: Path) -> None:
    case = _run_case(tmp_path, (CONTRACT_ERROR,))
    _assert_base_terminal(case, "HARNESS_CONTRACT", "engine_error")
    assert len(case.requests) == 1, (len(case.requests), case.summary)
    assert not any(event.get("kind") == "provider_transport_retry" for event in case.events)


def test_capacity_retry_wall_time_within_ten_seconds_of_success_control(
        tmp_path: Path) -> None:
    retry_start = time.perf_counter()
    retry_case = _run_case(tmp_path / "retry", (CAPACITY_ERROR,))
    retry_wall = time.perf_counter() - retry_start
    success_control_start = time.perf_counter()
    success_control_case = _run_case(tmp_path / "success-control", ())
    success_control_wall = time.perf_counter() - success_control_start

    assert retry_case.exit_code == 0, retry_case.output
    assert len(retry_case.requests) == 2, retry_case.output
    assert success_control_case.exit_code == 0, success_control_case.output
    assert len(success_control_case.requests) == 1, success_control_case.output
    assert success_control_case.summary["engine_result"] == "SUBMITTED", (
        success_control_case.summary)
    assert success_control_case.summary["succeeded"] is True, success_control_case.summary
    print(f"retry_wall={retry_wall:.3f}s success_control_wall={success_control_wall:.3f}s")
    # declared re-freeze 11-05: 180 s timeout accepted only with this relative bound
    assert retry_wall <= success_control_wall + 10.0, (
        f"capacity retry took {retry_wall:.3f}s; equivalent success control took "
        f"{success_control_wall:.3f}s (maximum allowed overhead: 10.0s)"
    )


def test_capacity_retry_request_is_deterministic_across_fresh_runs(
        capacity_runs: tuple[Case, Case]) -> None:
    first, second = capacity_runs
    assert len(first.requests) == len(second.requests) == 2, (
        len(first.requests), len(second.requests))
    assert first.exit_code == second.exit_code == 0
    assert _stable_request(first.requests[1]) == _stable_request(second.requests[1])
