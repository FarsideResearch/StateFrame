"""Qualification work budgets on a large admissible synthetic repository."""

from __future__ import annotations

import json
import time
from collections import Counter
from types import SimpleNamespace

import pytest

import kernel_measurement as measurement
from kernel_final_wire_allocation import _recover_post_fit_growth
from kernel_pilot import ACTIVE_PARENT
from kernel_request_seal import RequestSealError


@pytest.fixture(autouse=True)
def isolated_parent_meter():
    """A standalone capture must not inherit another test's active run meter."""
    token = ACTIVE_PARENT.set(None)
    try:
        yield
    finally:
        ACTIVE_PARENT.reset(token)


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for directory in range(50):
        package = root / f"package_{directory:02}"
        package.mkdir()
        for index in range(100):
            (package / f"module_{index:02}.py").write_bytes(b"VALUE = 1\n")
    return root


def test_distinct_candidate_qualification_budget(repository, monkeypatch, record_property):
    """A fallback that already fits needs no second count of the same candidate."""
    monkeypatch.setenv("STATEFRAME_TURN_BOUNDED_QUALIFICATION", "on")
    monkeypatch.setattr("kernel_cache_layout_candidate.replay_stable_wire_decision", lambda: None)
    monkeypatch.setattr("kernel_context_capacity.enforce_context_capacity", lambda *a, **kw: None)
    history = [{"role": "user", "content": f"read result {step}"} for step in range(200)]
    fitted = {"messages": history, "tools": []}
    projected = {**fitted, "messages": [*history, {"role": "user", "content": "too large"}]}
    fitted_inputs = {"token_ceiling": 200, "byte_ceiling": 100_000}
    capture = SimpleNamespace(request_id="fit", _frame_candidate_inputs=fitted_inputs)
    counts = Counter()

    def count(body, _meter, _capture, **_kwargs):
        messages = len(body["messages"])
        counts[messages] += 1
        return messages, "tokenize"

    monkeypatch.setattr("kernel_frame_allocation_capture.final_request_tokens", count)
    started = time.perf_counter()
    result, recovered = _recover_post_fit_growth(
        capture, SimpleNamespace(repo=repository), projected, fitted, object(),
        fitted_inputs, {}, meter=SimpleNamespace(measure_method="tokenize"), endpoint="local", key="",
    )
    elapsed = time.perf_counter() - started
    assert recovered and json.dumps(result) == json.dumps(fitted)
    record_property("fit_seconds", elapsed)
    record_property("fit_counts", str(dict(counts)))
    assert elapsed < 120
    assert counts == {201: 1, 200: 1}, f"fit seconds={elapsed:.6f} counts={dict(counts)}"


def _send(tmp_path, repository, monkeypatch, *, mutate=None):
    monkeypatch.setenv("STATEFRAME_TURN_BOUNDED_QUALIFICATION", "on")
    monkeypatch.setattr("kernel_wire_truth_gate.enabled", lambda: False)
    journal = measurement.AuditJournal(tmp_path / "run", "qualification-budget", {})
    capture = measurement.RequestCapture(journal, 201, "snapshot")
    runtime = SimpleNamespace(route="prepared")
    capture._wire_truth_runtime = runtime
    capture._qualified_adapter = "budget"
    capture._qualified_request_gate = lambda *args: None
    capture._qualified_host_text_preparer = lambda *args: None
    capture._qualified_request_gate_passed = True
    monkeypatch.setattr("kernel_stateframe_framefit5.has_framefit5_state", lambda _: True)
    body = {"messages": [{"role": "user", "content": f"read result {step}"}
                         for step in range(200)], "tools": []}
    wire = json.dumps(body).encode()
    counts = Counter()
    submitted = []
    paths = tuple(repository.glob("*/*.py"))
    assert len(paths) == 5000

    def validate(current, current_runtime, candidate):
        assert current is capture and current_runtime is runtime
        assert json.dumps(candidate).encode() == wire
        if current_runtime.route != "prepared":
            raise RequestSealError("route changed")
        counts["heavy"] += 1
        for path in paths:
            counts["reads"] += 1
            assert path.read_bytes() == b"VALUE = 1\n"

    def provider(_capture, payload, *_args, **_kwargs):
        submitted.append(payload)
        return b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'

    capture.qualified_request = SimpleNamespace(wire=wire, validate=validate)
    monkeypatch.setattr(measurement, "_request_bytes_for_capture", provider)
    if mutate:
        emit = journal.emit

        def observe(kind, **fields):
            result = emit(kind, **fields)
            if kind == "request_submitted":
                mutate(runtime)
            return result

        monkeypatch.setattr(journal, "emit", observe)
    started = time.perf_counter()
    try:
        capture._send_transport_with_retries(
            body, wire, "http://127.0.0.1:1/v1/chat/completions", "",
            admission=None, socket_timeout=2, deadline=time.monotonic() + 120,
            wall_timeout=2, backend_identity={}, request_extensions={},
        )
    except RequestSealError:
        assert mutate is not None and not submitted
        raise
    elapsed = time.perf_counter() - started
    assert submitted == [wire]
    return counts, elapsed


def test_adjacent_retry_validation_budget(tmp_path, repository, monkeypatch, record_property):
    counts, elapsed = _send(tmp_path, repository, monkeypatch)
    record_property("send_seconds", elapsed)
    record_property("send_counts", str(dict(counts)))
    assert elapsed < 120
    # The unbound path still checks freshness after journal observers. The
    # first guard and the adjacent seal check share the same observation.
    assert counts == {"heavy": 2, "reads": 10_000}, (
        f"send seconds={elapsed:.6f} counts={dict(counts)}")


def test_submission_observer_drift_is_refused(tmp_path, repository, monkeypatch):
    def change_route(runtime):
        runtime.route = "changed"

    with pytest.raises(RequestSealError, match="route changed"):
        _send(tmp_path, repository, monkeypatch, mutate=change_route)
