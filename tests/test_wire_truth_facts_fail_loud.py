"""A failed shared done decision must stop wire fact construction."""
from __future__ import annotations

from types import SimpleNamespace

import kernel_done_readiness_authority
import kernel_wire_truth_facts as facts
import pytest


@pytest.mark.parametrize("failure", [ImportError, AttributeError, TypeError, ValueError])
@pytest.mark.parametrize("with_journal", [False, True])
def test_shared_done_decision_failure_stops_request_facts(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception], with_journal: bool,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    journal = SimpleNamespace(emit=lambda kind, **data: events.append((kind, data)))
    runtime = SimpleNamespace(binding=lambda: {"product_revision": "candidate"})
    if with_journal:
        runtime.journal = journal
    capture = SimpleNamespace(request_id="request-1")

    def unavailable(_runtime: object) -> None:
        raise failure("shared done decision unavailable")

    monkeypatch.setattr(kernel_done_readiness_authority, "cached_decision", unavailable)

    with pytest.raises(RuntimeError, match="DONE_READINESS_UNAVAILABLE") as caught:
        facts.capture_context(capture, runtime, {"tools": [{"name": "done"}]})

    assert isinstance(caught.value.__cause__, failure)
    assert not hasattr(capture, "wire_truth_fact_store")
    if with_journal:
        assert events == [("wire_truth_fact_evidence_failure", {
            "producer": "kernel_wire_truth_facts", "dispatch": False,
            "reason_code": "DONE_READINESS_UNAVAILABLE",
        })]
    else:
        assert events == []
