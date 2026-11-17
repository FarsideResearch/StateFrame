"""RED tests for the progress classifier and tool-call loop breaker.

Each test documents a current defect in src/stateframe/engine/kernel_tool_call_loop_breaker.py
(or its caller in probes/react_repair.py). Controls marked "control" should already pass.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from kernel_tool_call_loop_breaker import (
    ToolCallLoopBreaker,
    UnboundedProgressWatch,
    terminal_preemption_allowed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REACT_REPAIR = REPO_ROOT / "src" / "stateframe" / "engine" / "probes" / "react_repair.py"


@pytest.fixture(autouse=True)
def _bookmarks_default(monkeypatch):
    # Run on the default (bookmarks unset) code path.
    monkeypatch.delenv("STATEFRAME_CODE_BOOKMARKS", raising=False)


def test_l1_fresh_ranges_same_revision_are_progress():
    """FAILS today: read identity ignores effective_range, so distinct fresh ranges look like repeats."""
    watch = UnboundedProgressWatch(threshold=32)
    binding = {"product_revision": "r1"}
    receipts: list[dict] = []
    for boundary, start in enumerate(range(1, 1 + 50 * 10, 50)):
        receipts.append({
            "kind": "read",
            "result_class": "OK",
            "target_path": "a.py",
            "delivery": {
                "path": "a.py",
                "revision": "r1",
                "effective_range": {"start_line": start, "end_line": start + 49},
            },
        })
        watch.observe(binding, receipts, boundary=boundary)
    assert watch.streak == 0


def test_l1_same_range_reread_still_charges():
    """Control: identical read of the same range repeatedly must keep charging the streak."""
    watch = UnboundedProgressWatch(threshold=32)
    binding = {"product_revision": "r1"}
    receipts: list[dict] = []
    receipt = {
        "kind": "read",
        "result_class": "OK",
        "target_path": "a.py",
        "delivery": {
            "path": "a.py",
            "revision": "r1",
            "effective_range": {"start_line": 1, "end_line": 50},
        },
    }
    for boundary in range(10):
        receipts.append(dict(receipt))
        watch.observe(binding, receipts, boundary=boundary)
    assert watch.streak >= 9


def test_l2_shell_noop_with_new_output_is_progress():
    """FAILS today: kind 'shell' is outside the semantic kinds, so new output never resets the streak."""
    watch = UnboundedProgressWatch(threshold=32)
    binding = {"product_revision": "r1"}
    receipts: list[dict] = []
    for boundary in range(10):
        receipts.append({
            "kind": "shell",
            "result_class": "NO_OP",
            "detail": f"output-{boundary}",
            "fingerprint": "f",
        })
        watch.observe(binding, receipts, boundary=boundary)
    assert watch.streak == 0


def test_l2_web_result_is_progress():
    """FAILS today: kind 'web_fetch' is not classified as evidence, so new results never count."""
    watch = UnboundedProgressWatch(threshold=32)
    binding = {"product_revision": "r1"}
    receipts: list[dict] = []
    for boundary in range(10):
        receipts.append({
            "kind": "web_fetch",
            "result_class": "OK",
            "detail": f"page-{boundary}",
            "fingerprint": "f",
        })
        watch.observe(binding, receipts, boundary=boundary)
    assert watch.streak == 0


def test_l2_host_observed_read_is_progress():
    """FAILS today: result_class 'HOST_OBSERVED' is not in the accepted result set, so it is dropped."""
    watch = UnboundedProgressWatch(threshold=32)
    binding = {"product_revision": "r1"}
    receipts: list[dict] = []
    for boundary in range(10):
        receipts.append({
            "kind": "read",
            "result_class": "HOST_OBSERVED",
            "target_path": f"h{boundary}",
            "detail": f"observed-{boundary}",
            "fingerprint": "f",
        })
        watch.observe(binding, receipts, boundary=boundary)
    assert watch.streak == 0


def test_l2_identical_shell_output_still_charges():
    """Control: identical shell NO_OP output repeated must still charge the streak."""
    watch = UnboundedProgressWatch(threshold=32)
    binding = {"product_revision": "r1"}
    receipts: list[dict] = []
    for boundary in range(10):
        receipts.append({
            "kind": "shell",
            "result_class": "NO_OP",
            "detail": "output-same",
            "fingerprint": "f",
        })
        watch.observe(binding, receipts, boundary=boundary)
    assert watch.streak >= 9


def test_l3_breaker_new_observation_never_trips():
    """FAILS today: action fingerprint ignores the observation, so varied results still trip the breaker."""
    breaker = ToolCallLoopBreaker(threshold=5)
    decisions = [breaker.observe("read", {"path": "a.py"}, f"obs-{i}") for i in range(20)]
    assert not any(decision.tripped for decision in decisions)
    assert breaker.tripped is False


def test_l3_breaker_identical_observation_warns():
    """Control: identical tool, args and observation repeated must trip with a notice."""
    breaker = ToolCallLoopBreaker(threshold=5)
    decision = None
    for _ in range(5):
        decision = breaker.observe("read", {"path": "a.py"}, "same-observation")
    assert decision is not None
    assert decision.tripped is True
    assert isinstance(decision.notice, str)


def test_l3_breaker_is_warn_only():
    """FAILS today: a ready breaker still preempts the terminal boundary; it must be warn-only."""
    runtime = types.SimpleNamespace(
        _tool_call_loop_breaker_ready=True,
        _green_then_exhaust_boundary_active=False,
    )
    assert terminal_preemption_allowed(runtime) is False


def test_l3_unbounded_honours_switch_off():
    """FAILS today: the breaker construction gate admits step_cap-is-None runs even with the switch off."""
    lines = REACT_REPAIR.read_text(encoding="utf-8").splitlines()
    call_indexes = [i for i, line in enumerate(lines) if "ToolCallLoopBreaker.from_environment()" in line]
    assert len(call_indexes) == 1, "expected exactly one ToolCallLoopBreaker.from_environment() call site"
    call = call_indexes[0]

    if_index = None
    for candidate in range(call - 1, max(call - 5, -1), -1):
        if lines[candidate].lstrip().startswith("if ("):
            if_index = candidate
            break
    assert if_index is not None, "no `if (` condition found within 4 lines above the breaker construction"

    end = next(k for k in range(if_index, call) if lines[k].rstrip().endswith(":"))
    condition = "\n".join(lines[if_index:end + 1])
    assert "step_cap is None" not in condition, condition
