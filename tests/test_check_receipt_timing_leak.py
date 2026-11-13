"""Wall-clock timing must never reach the model through a check receipt.

The runner tail is masked to ``in <duration>`` before the model sees it, but
the byte counts that travel next to it (``stdout_bytes``,
``stdout_omitted_bytes`` ...) used to be computed from the RAW stream.  A run
that took ``0.42s`` and a rerun that took ``12.345s`` then produced different
model-facing receipt lines, which broke same-seed determinism and the cached
prompt prefix.  Every model-facing size here must be derived from the same
normalised text the model is shown.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kernel_check_receipts import _public_receipt, record_check_output, retrieve_check_output
from kernel_refusal_compact import _collapse_to_canonical_receipts, render_evidence_pin


FAST = "1 failed, 1 passed in 0.42s"
SLOW = "1 failed, 1 passed in 12.345s"

# Model-facing receipt fields that may legitimately differ between two runs of
# the same check under a different clock.  Empty on purpose: the wire must be
# clock-independent.  Add nothing here without a determinism argument.
TIME_DEPENDENT_MODEL_FIELDS: frozenset[str] = frozenset()


def _stdout(tail: str) -> str:
    return (
        "============ test session starts ============\n"
        "collected 2 items\n\n"
        "tests/test_report.py F.                 [100%]\n\n"
        "FAILED tests/test_report.py::test_sections - AssertionError\n"
        f"============ {tail} ============\n"
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        stateframe_check_execution_binding_mode="off",
        _check_output_receipts={},
        host_check_receipts=[],
    )


def _record(runtime: SimpleNamespace, tail: str) -> dict[str, object]:
    return record_check_output(
        runtime, selector="tests/test_report.py", status="FAIL", outcome="BEHAVIOR_FAIL",
        execution_id="check-execution:one", revision="sha256:" + "ab" * 32,
        stdout=_stdout(tail), stderr=f"warning: slow test ran in {tail.rsplit(' ', 1)[-1]}\n",
        assertion_context=f"AssertionError after in {tail.rsplit(' ', 1)[-1]}",
        exit_code=1, collected_tests=2, counts={"passed": 1, "failed": 1},
    )


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _strip_allowed(row: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in row.items() if k not in TIME_DEPENDENT_MODEL_FIELDS}


def test_private_store_sizes_follow_the_masked_text() -> None:
    """Repro: the stored receipt's model-facing sizes must not count duration digits."""
    fast, slow = _runtime(), _runtime()
    _record(fast, FAST)
    _record(slow, SLOW)
    (fast_private,) = fast._check_output_receipts.values()
    (slow_private,) = slow._check_output_receipts.values()
    # The raw streams stay lossless (they really differ).
    assert fast_private["stdout"] != slow_private["stdout"]
    for field in ("stdout_bytes", "stderr_bytes", "assertion_context_bytes",
                  "stdout_omitted_bytes", "stderr_omitted_bytes",
                  "assertion_context_omitted_bytes", "runner_tail",
                  "assertion_context_preview", "summary"):
        assert fast_private[field] == slow_private[field], field


def test_public_receipt_sizes_match_stored_sizes() -> None:
    """The public projection and the stored receipt must agree on every size."""
    runtime = _runtime()
    _record(runtime, FAST)
    (private,) = runtime._check_output_receipts.values()
    public = _public_receipt(private, None)
    for field in ("stdout_bytes", "stderr_bytes", "assertion_context_bytes",
                  "stdout_omitted_bytes", "stderr_omitted_bytes",
                  "assertion_context_omitted_bytes", "runner_tail",
                  "assertion_context_preview"):
        assert public[field] == private[field], field
    assert "0.42" not in public["summary"]
    assert "0.42" not in private["summary"]


@pytest.mark.parametrize("renderer", [
    "returned_receipt", "host_check_receipts", "public_receipt",
    "canonical_rows", "evidence_pin", "retrieve_page",
])
def test_every_model_facing_check_render_is_clock_independent(renderer: str) -> None:
    """CLASS KILLER: each model-facing check render is byte-identical across durations."""
    rendered: list[str] = []
    for tail in (FAST, SLOW):
        runtime = _runtime()
        returned = _record(runtime, tail)
        (private,) = runtime._check_output_receipts.values()
        if renderer == "returned_receipt":
            value: object = _strip_allowed(dict(returned))
        elif renderer == "host_check_receipts":
            value = [_strip_allowed(dict(row)) for row in runtime.host_check_receipts]
        elif renderer == "public_receipt":
            value = _strip_allowed(_public_receipt(private, None))
        elif renderer == "canonical_rows":
            rows = [{"status": "FAIL", "text": "", "revision": private["revision"],
                     "ordinal": 1, "source": "journal",
                     "execution_id": private["execution_id"]}]
            value = [_strip_allowed(row) for row in _collapse_to_canonical_receipts(runtime, rows)]
        elif renderer == "evidence_pin":
            value = render_evidence_pin(runtime, (), include_framefit_journal=False)
        else:
            value = _strip_allowed(retrieve_check_output(
                runtime, private["receipt_id"], stream="combined", offset=0, limit=4000))
        text = _canonical(value)
        assert "0.42" not in text and "12.345" not in text, renderer
        rendered.append(text)
    assert rendered[0] == rendered[1], renderer


def test_time_dependent_allowlist_is_empty() -> None:
    assert TIME_DEPENDENT_MODEL_FIELDS == frozenset()
