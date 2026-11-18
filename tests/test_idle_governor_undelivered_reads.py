"""A read that delivered no requested bytes is never treated as a zero-novelty re-read.

Budget misses, size-limit withholds and empty pages used to reach the governor as a
path-less read, which counted toward the budget and was denied once governed. The
model could then never obtain those bytes. Undelivered reads now ride a distinct
IDLE_UNDELIVERED receipt that spends no budget. A span of a delivered file that was
not delivered before is novel, even when the digest is absent.
"""
from __future__ import annotations

import pytest

from kernel_gate_switches import IDLE_GOVERNOR_ENV, idle_governor_enabled
from kernel_idle_governor import GovernorDecision, IdleGovernor

N1, N2 = 3, 5
DELIVERED = GovernorDecision.ALLOW


def _read(path: str, turn: int, digest: str | None = "d-a") -> dict[str, object]:
    action: dict[str, object] = {"kind": "read", "path": path, "turn": turn}
    if digest is not None:
        action["content_digest"] = digest
    return action


def _undelivered(path: str, turn: int) -> dict[str, object]:
    """Action shape the executor produces for a read whose requested bytes never arrived."""
    return {"kind": "read", "path": path, "turn": turn, "undelivered": True}


def _governed() -> IdleGovernor:
    gov = IdleGovernor(n1=N1, n2=N2)
    for turn in range(N2 + 2):
        gov.should_allow(_read("a.py", turn))
    assert gov.governed
    return gov


def _governed_by_ranges() -> IdleGovernor:
    """Governed state from re-reads of one span with no digest."""
    gov = IdleGovernor(n1=N1, n2=N2)
    for turn in range(N2 + 2):
        gov.should_allow({"kind": "read", "path": "a.py@1-5", "turn": turn})
    assert gov.governed
    return gov


# (a) Undelivered reads never warn, deny or spend the budget.
def test_undelivered_reads_never_warn_deny_or_spend_budget():
    gov = IdleGovernor(n1=N1, n2=N2)
    for turn in range(N2 * 3):
        assert gov.should_allow(_undelivered("big.py@225-310", turn)) == DELIVERED
        assert gov.last_receipt.event == "IDLE_UNDELIVERED"
    assert gov.consecutive_non_product == 0
    assert gov.warnings == 0 and gov.denials == 0
    assert not gov.governed


# (b) Once governed by real re-reads, an undelivered read is still not refused.
def test_undelivered_read_is_not_denied_after_governing():
    gov = _governed()
    before = (gov.denials, gov.consecutive_non_product)
    assert gov.should_allow(_undelivered("big.py@225-310", 50)) == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_UNDELIVERED"
    assert (gov.denials, gov.consecutive_non_product) == before
    assert gov.governed, "an undelivered read must not reset governing either"


# (f) An undelivered read does not mark its path as delivered.
def test_undelivered_read_does_not_mark_path_delivered():
    gov = _governed()
    gov.should_allow(_undelivered("c.py", 40))
    assert gov.should_allow(_read("c.py", 41, digest="d-c")) == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_NOVELTY_ESCAPE"


def test_undelivered_receipt_keeps_its_path_visible():
    gov = IdleGovernor(n1=N1, n2=N2)
    gov.should_allow(_undelivered("big.py@225-310", 7))
    row = gov.trace[-1]
    assert row["event"] == "IDLE_UNDELIVERED"
    assert row["path"] == "big.py"
    assert gov.summary()["undelivered_reads"] == 1


# (c) A span of a delivered file not delivered before is novel, even without a digest.
def test_new_range_of_a_delivered_file_is_novel_without_digest():
    gov = _governed_by_ranges()
    assert gov.should_allow({"kind": "read", "path": "a.py@6-9", "turn": 20}) == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_NOVELTY_ESCAPE"


def test_identical_range_reread_without_digest_still_counts():
    gov = _governed_by_ranges()
    before = gov.denials
    assert gov.should_allow({"kind": "read", "path": "a.py@1-5", "turn": 20}) == GovernorDecision.DENY
    assert gov.denials == before + 1


def test_new_range_is_novel_with_digest_and_identical_range_digest_still_counts():
    gov = _governed()
    assert gov.should_allow(_read("a.py@6-9", 30, digest="d-new")) == GovernorDecision.ALLOW
    assert gov.should_allow(_read("a.py@6-9", 31, digest="d-new")) == GovernorDecision.DENY


# (d) Delivered identical re-reads still warn, then deny, exactly as before.
def test_delivered_identical_rereads_warn_then_deny_as_before():
    gov = IdleGovernor(n1=N1, n2=N2)
    decisions = [gov.should_allow(_read("a.py", turn)) for turn in range(N2)]
    assert decisions == [
        GovernorDecision.ALLOW, GovernorDecision.ALLOW, GovernorDecision.WARN,
        GovernorDecision.ALLOW, GovernorDecision.DENY,
    ]
    assert gov.warnings == 1 and gov.denials == 1 and gov.governed


# (e) The governor switch: off disables the whole organ, default on.
def test_idle_governor_switch_default_on_and_off_disables_it():
    assert idle_governor_enabled({}) is True
    assert idle_governor_enabled({IDLE_GOVERNOR_ENV: "off"}) is False
    with pytest.raises(ValueError):
        idle_governor_enabled({IDLE_GOVERNOR_ENV: "maybe"})


# Production mapping: the executor delivery receipt -> governor action builder used by react_repair.
def test_builder_maps_withheld_budget_miss_to_undelivered():
    from kernel_idle_governor import read_governor_action
    from kernel_read_exec import _record_requested_byte_delivery

    delivery: dict[str, object] = {}
    _record_requested_byte_delivery(delivery, status="WITHHELD")
    action = read_governor_action("big.py@225-310", turn=12, read_ok=True, delivery=delivery,
                                  deliver_requested=True, observation="")
    assert action == {"kind": "read", "path": "big.py@225-310", "turn": 12, "undelivered": True}


def test_builder_maps_empty_page_to_undelivered_and_proof_stub_stays_budgeted():
    from kernel_idle_governor import read_governor_action
    from kernel_read_exec import _record_requested_byte_delivery

    empty: dict[str, object] = {}
    _record_requested_byte_delivery(empty, status="DELIVERED", content="")
    stub: dict[str, object] = {}
    _record_requested_byte_delivery(stub, status="REUSED_PROVEN", content="")
    empty_action = read_governor_action("e.py", turn=1, read_ok=True, delivery=empty,
                                        deliver_requested=True, observation="")
    stub_action = read_governor_action("e.py", turn=1, read_ok=True, delivery=stub,
                                       deliver_requested=True, observation="")
    assert empty_action.get("undelivered") is True
    assert "undelivered" not in stub_action and stub_action["path"] == ""


def test_builder_keeps_digest_for_delivered_bytes_and_observation_digest_when_off():
    import hashlib

    from kernel_idle_governor import read_governor_action
    from kernel_read_exec import _record_requested_byte_delivery

    delivery: dict[str, object] = {}
    _record_requested_byte_delivery(delivery, status="DELIVERED", content="x = 1\n",
                                    metadata={"page_bytes_sha256": "sha256:abc"})
    delivered = read_governor_action("a.py@1-1", turn=2, read_ok=True, delivery=delivery,
                                     deliver_requested=True, observation="")
    assert delivered == {"kind": "read", "path": "a.py@1-1", "turn": 2,
                         "content_digest": "sha256:abc"}
    off = read_governor_action("a.py", turn=3, read_ok=True, delivery={},
                               deliver_requested=False, observation="obs")
    assert off["content_digest"] == hashlib.sha256(b"obs").hexdigest()[:32]
    assert "undelivered" not in off


def test_builder_output_drives_governor_without_spending_budget():
    from kernel_idle_governor import read_governor_action
    from kernel_read_exec import _record_requested_byte_delivery

    gov = _governed()
    denials, warnings = gov.denials, gov.warnings
    for turn in range(40):
        delivery: dict[str, object] = {}
        _record_requested_byte_delivery(delivery, status="WITHHELD")
        action = read_governor_action("big.py@225-310", turn=100 + turn, read_ok=True,
                                      delivery=delivery, deliver_requested=True, observation="")
        assert gov.should_allow(action) == GovernorDecision.ALLOW
    assert gov.denials == denials and gov.warnings == warnings
    assert gov.undelivered_reads == 40


# No-progress stop: the run-level detector (UnboundedProgressWatch, default threshold 32)
# is the only stop an endless read stream can hit now that the governor admits undelivered
# reads. Undelivered receipts must be no-progress evidence for it, not progress.
def _read_receipt(path: str, status: str, nbytes: int, start: int, end: int) -> dict[str, object]:
    return {"kind": "read", "result_class": "OK", "target_path": path,
            "delivery": {"delivery_status": status, "requested_bytes_delivered": nbytes,
                         "effective_range": {"start_line": start, "end_line": end}}}


def _run_until_stop(receipt_for_turn, limit: int = 300) -> int | None:
    from kernel_tool_call_loop_breaker import UnboundedProgressWatch

    watch = UnboundedProgressWatch()
    binding = {"product_revision": "rev-1"}
    receipts: list[dict[str, object]] = []
    for turn in range(limit):
        receipts.append(receipt_for_turn(turn))
        if watch.observe(binding, receipts, boundary=turn):
            return turn
    return None


def test_distinct_range_undelivered_stream_reaches_no_progress_stop():
    stop = _run_until_stop(lambda t: _read_receipt("big.py", "WITHHELD", 0, 1 + t * 10, 10 + t * 10))
    assert stop is not None and stop <= 40, "an endless undelivered stream must reach the no-progress stop"


def test_same_range_undelivered_stream_reaches_no_progress_stop():
    stop = _run_until_stop(lambda t: _read_receipt("big.py", "WITHHELD", 0, 225, 310))
    assert stop is not None and stop <= 40


def test_empty_page_stream_reaches_no_progress_stop():
    stop = _run_until_stop(lambda t: _read_receipt("e.py", "DELIVERED", 0, 1 + t, 1 + t))
    assert stop is not None and stop <= 40


def test_fresh_delivered_ranges_still_count_as_progress():
    stop = _run_until_stop(lambda t: _read_receipt("ok.py", "DELIVERED", 12, 1 + t * 5, 5 + t * 5))
    assert stop is None, "delivered new ranges are real progress and must not trip the stop"
