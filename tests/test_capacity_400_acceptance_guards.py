# DECLARED RE-FREEZE 11-05 13:21 (release manager): preserve retry-scope and false-positive guards.
"""CLI acceptance guards for capacity retry scope and false positive HTTP 400s."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pytest

from _journal import measurement_events
from engine_run import run_monitored, sandbox_test_python
from test_capacity_400_acceptance import (
    CAPACITY_ERROR, ROOT, CapacityEndpoint, Case, _assert_base_terminal, _run_case,
)


pytestmark = pytest.mark.host_sandbox
NEGATIVE_FIXTURE = (
    Path(__file__).parent / "fixtures" / "capacity_400_negative"
    / "response_format_schema.json"
)


class DegenerationEndpoint(CapacityEndpoint):
    """Reuse the acceptance endpoint's discovery routes and primary success reply."""

    def __init__(self) -> None:
        super().__init__((CAPACITY_ERROR,))
        base_handler = self.server.RequestHandlerClass
        endpoint = self

        class Handler(base_handler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path.rstrip("/") != "/v1/chat/completions":
                    super().do_POST()
                    return
                with endpoint.lock:
                    next_request = len(endpoint.requests) + 1
                if next_request not in (2, 3):
                    super().do_POST()
                    return
                wire = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = (gzip.decompress(wire)
                        if self.headers.get("Content-Encoding") == "gzip" else wire)
                with endpoint.lock:
                    endpoint.requests.append(body)
                if next_request == 3:
                    self._send(400, CAPACITY_ERROR)
                    return
                self._send(200, {
                    "id": "chatcmpl-degenerate",
                    "object": "chat.completion",
                    "model": "capacity-mock",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant",
                        "content": "repeated provider output. " * 25,
                    }}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 100,
                              "total_tokens": 200},
                })

        self.server.RequestHandlerClass = Handler


def _run_degenerating_case(work: Path) -> Case:
    """Use the frozen CLI fixture's repository and command with the guard enabled."""
    repo = work / "repo"
    repo.mkdir(parents=True)
    (repo / "app.py").write_text("VALUE = 6\n", encoding="utf-8")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text(
        "from app import VALUE\n\ndef test_value():\n    assert VALUE == 7\n",
        encoding="utf-8",
    )
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
        "STATEFRAME_DEGENERATION_GUARD": "on",
    })
    with (DegenerationEndpoint() as endpoint,
          sandbox_test_python(sys.executable) as test_python):
        process = run_monitored(
            [sys.executable, "-B", "-m", "stateframe", "run",
             "--repo", str(repo), "--task", "Fix app.py so VALUE is 7.",
             "--endpoint", endpoint.url, "--model", "capacity-mock",
             "--run-dir", str(run), "--max-turns", "1",
             "--test-cmd", "python3 -m pytest -q tests/test_app.py",
             "--runner-kind", "pytest", "--test-python", test_python,
             "--target", "app.py", "--edit", "app.py", "--shell", "off",
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


def test_capacity_retry_is_not_rearmed_by_degeneration_resend(tmp_path: Path) -> None:
    case = _run_degenerating_case(tmp_path)
    degeneration = [event for event in case.events
                    if event.get("kind") == "degeneration_detected"]
    assert len(degeneration) == 1, (degeneration, case.summary, case.output)
    capacity_retries = [event for event in case.events
                        if event.get("kind") == "capacity_retry"]
    assert len(capacity_retries) == 1, (capacity_retries, case.summary, case.output)
    assert len(case.requests) == 3, (len(case.requests), case.summary, case.output)
    _assert_base_terminal(case, "HARNESS_TRANSPORT", "missing_result")


def test_response_format_schema_400_is_not_capacity(tmp_path: Path) -> None:
    fixture = json.loads(NEGATIVE_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(fixture["note"], str) and fixture["note"]
    body = fixture["body"]
    assert isinstance(body, dict)
    # If the classifier is wrong, keep the same provider rejection on the
    # extra send so the run terminates promptly and exposes the retry count.
    with sandbox_test_python(sys.executable) as test_python:
        case = _run_case(tmp_path, (body, body), test_python=test_python)
    assert len(case.requests) == 1, (len(case.requests), case.summary, case.output)
    assert not any(event.get("kind") in {"capacity_retry", "provider_transport_retry"}
                   for event in case.events), case.events
    _assert_base_terminal(case, "HARNESS_CONTRACT", "engine_error")
