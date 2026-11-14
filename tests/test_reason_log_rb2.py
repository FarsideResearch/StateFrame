"""RB2 sites (request budget, context capacity, request seal) log one reason line per cause per run."""
import io
import re
import time
from types import SimpleNamespace

import pytest

import kernel_context_capacity as capacity
import kernel_endpoint_size
import kernel_measurement
import kernel_reason_log as rl
import kernel_request_budget as budget
import kernel_request_seal as seal

LINE = re.compile(r"^\[stateframe\] (WARN|ERROR|INFO) (?P<component>[a-z_]+): .+", re.M)


@pytest.fixture(autouse=True)
def stream(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    monkeypatch.delenv("STATEFRAME_CONTEXT_PREFLIGHT", raising=False)
    rl.reset_for_tests()
    out = io.StringIO()
    rl.configure(level="INFO", stream=out)
    yield out
    rl.reset_for_tests()


def _lines(out, component):
    return [m.group(0) for m in LINE.finditer(out.getvalue()) if m.group("component") == component]


def test_context_overflow_refuses_with_error_line_once(monkeypatch, stream):
    monkeypatch.setenv("STATEFRAME_N_CTX", "1000")
    monkeypatch.setattr(kernel_endpoint_size, "ENDPOINT_SIZE",
                        SimpleNamespace(configured=False, context_tokens=None))
    monkeypatch.setattr(kernel_measurement, "_prompt_token_estimate", lambda body: 900)
    capture = SimpleNamespace(journal=None, request_id="r1", step=1, frame_route=None,
                              capacity_endpoint="http://127.0.0.1:1")
    for _ in range(2):
        body = {"max_tokens": 500, "messages": [{"role": "user", "content": "x"}]}
        with pytest.raises(capacity.ContextCapacityError):
            capacity.enforce_context_capacity(capture, body)
    lines = _lines(stream, "context_capacity")
    assert len(lines) == 1, stream.getvalue()
    assert lines[0].startswith("[stateframe] ERROR context_capacity: request refused: exceeds context")
    assert "measured 1418 vs limit 1000" in lines[0]


def test_diet_deferred_is_warn_with_measured_vs_limit_and_dedupes(stream):
    detail = {"reason": "request_diet_cap_infeasible", "request_bytes_after": 2048,
              "max_request_bytes": 1500}
    budget._log_diet_deferred(detail, stage="FREE")
    budget._log_diet_deferred(dict(detail), stage="FREE")
    lines = _lines(stream, "request_budget")
    assert len(lines) == 1, stream.getvalue()
    assert lines[0].startswith("[stateframe] WARN request_budget: request diet deferred at stage FREE")
    assert "measured 2048 vs limit 1500" in lines[0]
    assert rl.run_summary()["by_cause"]["refusal:diet_deferred:request_diet_cap_infeasible"]["count"] == 2


def test_lean_history_mutation_logs_invariant_error_once(stream):
    capture = SimpleNamespace(lean_prepared_history=[{"role": "user", "content": "a"}],
                              journal=None, request_id="r2")
    body = {"messages": [{"role": "user", "content": "b"}]}
    for _ in range(2):
        with pytest.raises(budget.LeanHistoryMutationError):
            budget._lean_append_only_receipt(capture, body, boundary="capture_budget_input")
    lines = _lines(stream, "request_budget")
    assert len(lines) == 1, stream.getvalue()
    assert lines[0].startswith("[stateframe] ERROR request_budget: LEAN prepared history invariant broken")
    assert "message index 0" in lines[0]


def _seal_self(**overrides):
    fields = dict(request_id="req-1", deadline=None)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_seal_identity_violation_is_error_once(stream):
    capture = SimpleNamespace(request_id="other")
    runtime = SimpleNamespace(current_request_id="req-1")
    for _ in range(2):
        with pytest.raises(seal.RequestSealError, match="qualified request identity changed"):
            seal.QualifiedRequest.validate(_seal_self(), capture, runtime)
    lines = _lines(stream, "request_seal")
    assert len(lines) == 1, stream.getvalue()
    assert lines[0].startswith("[stateframe] ERROR request_seal: qualified request seal violated")


def test_seal_deadline_expiry_is_timeout_warn(stream):
    capture = SimpleNamespace(request_id="req-1")
    runtime = SimpleNamespace(current_request_id="req-1")
    expired = _seal_self(deadline=time.monotonic() - 1.0)
    with pytest.raises(TimeoutError, match="deadline expired"):
        seal.QualifiedRequest.validate(expired, capture, runtime)
    lines = _lines(stream, "request_seal")
    assert len(lines) == 1, stream.getvalue()
    assert lines[0].startswith("[stateframe] WARN request_seal: qualified request")
    assert rl.run_summary()["by_cause"]["timeout:deadline"]["count"] == 1
