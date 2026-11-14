"""The parent sees a subframe reservation after an automatic check replaces the tool tail."""
from __future__ import annotations

from stateframe.engine.kernel_native_tools import NativeToolBatch


class Journal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, name: str, **fields: object) -> dict:
        self.events.append((name, fields))
        return {}


def _close(tool: str, observation: str) -> tuple[dict, str]:
    batch = NativeToolBatch()
    batch.journal = Journal()
    batch.inflight = {"id": "call_0", "function": {"name": tool, "arguments": "{}"}}
    batch.index = 1
    batch.action_closed({"step_record_id": "step-1", "result": "APPLIED",
                         "observation": observation})
    receipts = batch.take_model_receipts()
    assert len(receipts) == 1
    return receipts[0], batch.render_model_receipts(receipts)


def test_parallel_success_receipt_delivers_adaptive_request_cap(monkeypatch) -> None:
    monkeypatch.setenv("KERNEL_NATIVE_PREFIX_AUTHORITY", "1")
    observation = "Subframe results:\nsubframe_request_cap=7\nSubframe a: COMMITTED"
    receipt, message = _close("parallel", observation)
    assert receipt == {"call_id": "call_0", "batch_index": 0, "tool": "parallel",
                       "status": "APPLIED", "reason": "", "detail": "subframe_request_cap=7"}
    assert message == ("[NATIVE BATCH RECEIPTS]\n"
                       "Every queued call has an explicit host receipt.\n"
                       "call 0 parallel: APPLIED; subframe_request_cap=7")

    # A similarly worded observation from another tool cannot masquerade as
    # the subframe reservation in the parent's next request.
    other_receipt, other_message = _close("read", observation)
    assert "detail" not in other_receipt
    assert "subframe_request_cap=7" not in other_message
