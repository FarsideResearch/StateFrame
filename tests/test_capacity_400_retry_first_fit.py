"""The capacity retry should qualify its compact surface only once."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
sys.path.insert(0, str(ENGINE))

import kernel_final_wire_allocation  # noqa: E402
import kernel_measurement  # noqa: E402
from kernel_frame_allocation_capture import FrameFit5ImpossibleMinimum  # noqa: E402
from kernel_capacity_retry import RetryLimits, reseal_capacity_retry  # noqa: E402


def test_retry_compacts_before_first_qualification(monkeypatch) -> None:
    original = {
        "max_tokens": 100,
        "tools": [{"type": "function", "function": {
            "name": "edit", "description": "Long optional prose.",
            "parameters": {"type": "object", "description": "Optional prose.",
                           "properties": {"description": {"type": "string"}},
                           "required": ["description"]},
        }}],
    }
    previous = SimpleNamespace(wire=b"x" * 10_000,
                               measurement=SimpleNamespace(output_tokens=100))
    runtime = SimpleNamespace(_active_capture=None)
    capture = SimpleNamespace(qualified_request=previous,
                              completion_owner_runtime=runtime)
    calls = []

    def qualified(candidate, _runtime, body, **_kwargs):
        calls.append(body)
        assert candidate._capacity_retry_compact_tool_prose is True
        function = body["tools"][0]["function"]
        assert function["description"] == "Edit a file."
        assert "description" not in function["parameters"]
        assert "description" in function["parameters"]["properties"]
        wire = json.dumps(body).encode()
        record = SimpleNamespace(body=body, wire=wire,
                                 measurement=SimpleNamespace(prompt_tokens=50,
                                                             output_tokens=40))
        candidate._request_measurement = record.measurement
        return record

    monkeypatch.setattr(kernel_final_wire_allocation, "build_qualified_request", qualified)
    limits = RetryLimits(9_000, 60, None, 40)
    body, wire, actual = reseal_capacity_retry(
        capture, original, "http://127.0.0.1:1/v1/chat/completions", "", None, limits)

    assert len(calls) == 1
    assert (body, wire, actual) == (calls[0], json.dumps(calls[0]).encode(), limits)
    assert original["tools"][0]["function"]["description"] == "Long optional prose."


def test_impossible_compacted_candidate_is_not_qualified_twice(monkeypatch) -> None:
    calls = []

    def impossible(*_args, **_kwargs):
        calls.append(1)
        raise FrameFit5ImpossibleMinimum({})

    monkeypatch.setattr(kernel_final_wire_allocation, "build_qualified_request", impossible)
    capture = SimpleNamespace(
        qualified_request=SimpleNamespace(wire=b"x" * 10_000,
                                          measurement=SimpleNamespace(output_tokens=100)),
        completion_owner_runtime=SimpleNamespace(_active_capture=None),
    )
    with pytest.raises(FrameFit5ImpossibleMinimum):
        reseal_capacity_retry(capture, {"max_tokens": 100, "tools": []},
                              "http://127.0.0.1:1/v1/chat/completions", "", None,
                              RetryLimits(9_000, 60, None, 40))
    assert calls == [1]


def test_capacity_retry_does_not_call_sleep_for_zero_delay(monkeypatch) -> None:
    waits = []
    monkeypatch.setattr(kernel_measurement.time, "sleep", waits.append)
    kernel_measurement._wait_for_transport_retry(0.0)
    assert waits == []
    kernel_measurement._wait_for_transport_retry(0.25)
    assert waits == [0.25]
