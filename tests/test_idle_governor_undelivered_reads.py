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
    decisions = [gov.should_allow(_read("a.py", turn)) for turn in range(N2 + 1)]
    assert decisions == [
        GovernorDecision.ALLOW, GovernorDecision.ALLOW, GovernorDecision.ALLOW,
        GovernorDecision.WARN, GovernorDecision.ALLOW, GovernorDecision.DENY,
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
    assert delivered["kind"] == "read" and delivered["path"] == "a.py@1-1"
    assert delivered["turn"] == 2 and delivered["content_digest"] == "sha256:abc"
    assert delivered["evidence_identity"]
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


@pytest.mark.parametrize("content", ["x", "\n", "\r\n", "﻿", "é"])
def test_builder_boundary_one_delivered_unit_counts_as_delivered(content):
    """Boundary: zero delivered bytes is undelivered; a single delivered byte/char is a real delivery."""
    from kernel_idle_governor import read_governor_action
    from kernel_read_exec import _record_requested_byte_delivery

    delivery: dict[str, object] = {}
    _record_requested_byte_delivery(delivery, status="DELIVERED", content=content)
    action = read_governor_action("b.py@1-1", turn=3, read_ok=True, delivery=delivery,
                                  deliver_requested=True, observation="")
    assert "undelivered" not in action
    assert action["path"] == "b.py@1-1" and action.get("content_digest")


def _grep_page(query: str, *, digest: str | None = None, offset: int = 0,
               hit: str | None = None):
    from types import SimpleNamespace

    result_digest = digest or f"set-{query}"
    return SimpleNamespace(
        text=f"[search page] query={query} hit={hit or query}",
        metadata={"result_set_digest": result_digest, "offset": offset,
                  "shown": 1, "total": 1, "complete": True},
    )


def _grep_action(query: str, turn: int, *, digest: str | None = None,
                 offset: int = 0, delivered: bool = True, hit: str | None = None):
    from kernel_idle_governor import grep_governor_action

    return grep_governor_action(
        query, turn=turn, page=_grep_page(query, digest=digest, offset=offset, hit=hit),
        delivered=delivered,
    )


def test_thirty_distinct_delivered_greps_are_evidence_not_idle_actions():
    gov = IdleGovernor()
    actions = [_grep_action(f"needle-{index}", index // 15) for index in range(30)]

    decisions = [gov.should_allow(action) for action in actions]

    assert decisions == [GovernorDecision.ALLOW] * 30
    assert gov.consecutive_non_product == 0
    assert gov.warnings == 0 and gov.denials == 0
    assert all(row["event"] == "IDLE_NOVELTY_ESCAPE" for row in gov.trace)
    assert all(row["evidence_identity"] for row in gov.trace)


def test_new_read_ranges_and_file_revisions_are_evidence_not_idle_actions():
    gov = IdleGovernor()
    actions = [
        {"kind": "read", "path": "src/module.py", "turn": 0, "revision": "rev-1",
         "byte_start": index * 4, "byte_end": index * 4 + 4}
        for index in range(20)
    ]
    actions.append({"kind": "read", "path": "src/module.py", "turn": 1,
                    "revision": "rev-2", "byte_start": 0, "byte_end": 4})

    assert [gov.should_allow(action) for action in actions] == [GovernorDecision.ALLOW] * 21
    assert gov.consecutive_non_product == 0
    assert gov.warnings == 0 and gov.denials == 0


def test_changed_content_at_the_same_unversioned_range_is_new_evidence():
    gov = IdleGovernor()
    before = {"kind": "read", "path": "src/module.py", "turn": 1,
              "byte_start": 4, "byte_end": 8, "content_digest": "sha256:before"}
    after = {**before, "turn": 2, "content_digest": "sha256:after"}

    assert gov.should_allow(before) == GovernorDecision.ALLOW
    assert gov.should_allow(after) == GovernorDecision.ALLOW
    assert [row["event"] for row in gov.trace] == [
        "IDLE_NOVELTY_ESCAPE", "IDLE_NOVELTY_ESCAPE",
    ]
    assert gov.consecutive_non_product == 0


def test_paged_search_results_are_new_evidence_and_repeats_are_steered():
    gov = IdleGovernor(n1=2, n2=3)
    first = _grep_action("symbol", 1, digest="set-1", offset=0)
    second = _grep_action("symbol", 2, digest="set-1", offset=12, hit="next-hit")
    changed_results = _grep_action("symbol", 3, digest="set-2", offset=0)

    assert gov.should_allow(first) == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_NOVELTY_ESCAPE"
    assert gov.should_allow(second) == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_NOVELTY_ESCAPE"
    assert gov.should_allow(changed_results) == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_NOVELTY_ESCAPE"
    assert gov.should_allow(_grep_action("symbol", 4, digest="set-1", offset=0)) == GovernorDecision.ALLOW
    assert gov.should_allow(_grep_action("symbol", 5, digest="set-1", offset=0)) == GovernorDecision.WARN
    assert gov.should_allow(_grep_action("symbol", 6, digest="set-1", offset=0)) == GovernorDecision.DENY
    assert gov.last_refusal == "read budget exhausted: edit, test, or abstain"


def test_changing_only_grep_page_offset_does_not_hide_a_repeat_loop(tmp_path):
    gov = IdleGovernor(n1=2, n2=4)
    pages = None
    action_identities = None
    try:
        from kernel_idle_governor import grep_governor_action
    except ImportError:
        # The bare tip predates the page builder. Its count-based policy must still
        # recognize this as a repeat-heavy loop, which distinguishes this regression
        # from the productive-read bug fixed by the reviewed change.
        actions = [
            {"kind": "grep", "query": "unmatched", "turn": offset,
             "result_set_digest": "same-empty-result", "page_offset": offset,
             "page_digest": f"display-header-{offset}"}
            for offset in range(400)
        ]
    else:
        from kernel_navigation_pages import search_repository

        first = search_repository(tmp_path, "unmatched")
        result_digest = first.metadata["result_set_digest"]
        pages = [
            search_repository(
                tmp_path, "unmatched", cursor=f"nav1:{result_digest}:{offset}")
            for offset in range(400)
        ]
        actions = [
            grep_governor_action(
                "unmatched", turn=offset, page=page, delivered=True,
            )
            for offset, page in enumerate(pages)
        ]
        action_identities = {action["evidence_identity"] for action in actions}
        # Preserve the page offset as receipt metadata so the governor must ignore
        # it when the returned evidence itself is unchanged.
        for offset, action in enumerate(actions):
            action.pop("evidence_identity", None)
            action["page_offset"] = offset

    decisions = [gov.should_allow(action) for action in actions]

    assert decisions[0] == GovernorDecision.ALLOW
    assert GovernorDecision.WARN in decisions
    assert GovernorDecision.DENY in decisions
    assert gov.warnings == 1 and gov.denials > 0
    if pages is not None:
        assert len({page.evidence_digest for page in pages}) == 1
        assert action_identities is not None and len(action_identities) == 1


def test_failed_and_refused_grep_calls_do_not_spend_or_remember_evidence():
    from kernel_idle_governor import grep_governor_action

    gov = IdleGovernor(n1=2, n2=4)
    page = _grep_page("same")
    for turn in range(400):
        action = (grep_governor_action("same", turn=turn, page=page, delivered=False)
                  if turn % 3 == 0 else
                  {"kind": "grep", "query": "same", "turn": turn,
                   "result_class": "FAILED" if turn % 3 == 1 else "REFUSED"})
        assert gov.should_allow(action) == GovernorDecision.ALLOW
    assert gov.consecutive_non_product == 0
    assert gov.warnings == 0 and gov.denials == 0

    successful = _grep_action("same", 401)
    assert gov.should_allow(successful) == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_NOVELTY_ESCAPE"


def test_four_hundred_repeated_grep_results_still_warn_and_deny():
    gov = IdleGovernor()
    decisions = [gov.should_allow(_grep_action("same", turn, digest="same-results"))
                 for turn in range(400)]

    assert decisions[0] == GovernorDecision.ALLOW
    assert decisions[8] == GovernorDecision.WARN
    assert decisions[14] == GovernorDecision.DENY
    assert decisions[15:] == [GovernorDecision.DENY] * 385
    assert gov.warnings == 1 and gov.denials == 386


def test_four_hundred_repeated_read_ranges_still_warn_and_deny():
    gov = IdleGovernor()
    action = {"kind": "read", "path": "src/module.py@1-20",
              "revision": "rev-1", "byte_start": 0, "byte_end": 200}
    decisions = [gov.should_allow({**action, "turn": turn}) for turn in range(400)]

    assert decisions[0] == GovernorDecision.ALLOW
    assert decisions[8] == GovernorDecision.WARN
    assert decisions[14] == GovernorDecision.DENY
    assert decisions[15:] == [GovernorDecision.DENY] * 385
    assert gov.warnings == 1 and gov.denials == 386


def test_same_turn_batch_and_replay_keep_grep_decisions_identical():
    from kernel_idle_governor import replay_receipts

    history = [_grep_action(f"batch-{index}", 7) for index in range(24)]
    parent = replay_receipts(history, n1=2, n2=4)
    resumed = replay_receipts(history, n1=2, n2=4)
    subframe = replay_receipts(history, n1=2, n2=4)
    duplicate = _grep_action("batch-0", 8)

    decisions = []
    for governor in (parent, resumed, subframe):
        governor.should_allow(duplicate)
        decisions.append(governor.should_allow({**duplicate, "turn": 9}))
    assert decisions == [GovernorDecision.WARN] * 3
    assert [governor.consecutive_non_product for governor in (parent, resumed, subframe)] == [2, 2, 2]


def test_replay_counts_same_fingerprint_calls_as_distinct_batched_actions():
    from kernel_idle_governor import replay_receipts

    base = _grep_action("same-query", 7, digest="same-result")
    receipts = [
        {**base, "fingerprint": "wire-pattern", "receipt_id": f"call-{index}",
         "result_class": "HITS"}
        for index in range(3)
    ]

    resumed = replay_receipts(receipts, n1=2, n2=4)

    assert [row["event"] for row in resumed.trace] == [
        "IDLE_NOVELTY_ESCAPE", "IDLE_ALLOWED", "IDLE_WARNING",
    ]
    assert resumed.consecutive_non_product == 2


def test_replayed_read_receipts_preserve_revision_and_range_identity():
    from kernel_idle_governor import read_governor_action, replay_receipts

    delivery = {"delivery_status": "DELIVERED", "requested_bytes_delivered": 4,
                "requested_bytes_sha256": "sha256:page", "revision": "rev-1",
                "byte_start": 0, "byte_end": 4,
                "effective_range": {"start_line": 1, "end_line": 1}}
    first = read_governor_action("src/module.py", turn=1, read_ok=True,
                                 delivery=delivery, deliver_requested=True, observation="")
    receipts = [{**first, "result_class": "OK", "delivery": delivery}]
    resumed = replay_receipts(receipts, n1=1, n2=2)
    same = {**first, "turn": 2, "delivery": delivery}
    other_revision_delivery = {**delivery, "revision": "rev-2"}
    changed = read_governor_action(
        "src/module.py", turn=2, read_ok=True, delivery=other_revision_delivery,
        deliver_requested=True, observation="")

    assert resumed.should_allow(same) == GovernorDecision.WARN
    assert resumed.trace[-1]["evidence_identity"]
    assert resumed.should_allow(changed) == GovernorDecision.ALLOW
    assert resumed.last_receipt.event == "IDLE_NOVELTY_ESCAPE"


def test_repeated_directory_pages_still_reach_the_idle_governor(tmp_path):
    from types import SimpleNamespace

    from kernel_idle_governor import read_governor_action
    from kernel_read_exec import execute_read

    (tmp_path / "known.py").write_text("value = 1\n", encoding="utf-8")
    gov = IdleGovernor(n1=2, n2=4)
    actions = []

    def read_page(turn):
        delivery = {}
        result, observation, _ = execute_read(
            tmp_path, tmp_path, ".", ".", 0, args={},
            measurement=SimpleNamespace(), step=turn, delivery=delivery,
        )
        assert result == "DIRECTORY"
        action = read_governor_action(
            ".", turn=turn, read_ok=False, delivery=delivery,
            deliver_requested=True, observation=observation,
        )
        actions.append(action)
        return gov.should_allow(action)

    decisions = [read_page(turn) for turn in range(8)]

    assert GovernorDecision.WARN in decisions
    assert GovernorDecision.DENY in decisions
    from kernel_idle_governor import replay_receipts
    replayed = replay_receipts(actions, n1=2, n2=4)
    assert [(row["event"], row["decision"]) for row in replayed.trace] == [
        (row["event"], row["decision"]) for row in gov.trace
    ]


def test_new_directory_entries_are_novel_evidence(tmp_path):
    from types import SimpleNamespace

    from kernel_idle_governor import read_governor_action
    from kernel_read_exec import execute_read

    (tmp_path / "first.py").write_text("value = 1\n", encoding="utf-8")
    gov = IdleGovernor(n1=1, n2=2)

    def read_page(turn):
        delivery = {}
        result, observation, _ = execute_read(
            tmp_path, tmp_path, ".", ".", 0, args={},
            measurement=SimpleNamespace(), step=turn, delivery=delivery,
        )
        assert result == "DIRECTORY"
        return gov.should_allow(read_governor_action(
            ".", turn=turn, read_ok=False, delivery=delivery,
            deliver_requested=True, observation=observation,
        ))

    assert read_page(1) == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_NOVELTY_ESCAPE"

    (tmp_path / "second.py").write_text("other = 2\n", encoding="utf-8")
    assert read_page(2) == GovernorDecision.ALLOW
    assert gov.last_receipt.event == "IDLE_NOVELTY_ESCAPE"
    assert gov.warnings == 0 and gov.denials == 0


def test_reused_tool_call_id_does_not_suppress_a_distinct_governor_receipt():
    from kernel_idle_governor import grep_governor_action

    gov = IdleGovernor(n1=2, n2=4)
    first = grep_governor_action(
        "first query", turn=10,
        page=_grep_page("first query", digest="first-result"), delivered=True)
    second = grep_governor_action(
        "second query", turn=11,
        page=_grep_page("second query", digest="second-result"), delivered=True)
    first["receipt_id"] = second["receipt_id"] = "provider-reused-id"

    gov.should_allow(first)
    gov.should_allow(second)

    assert len(gov.trace) == 2
    assert gov.trace[-1]["turn"] == 11
    assert gov.trace[-1]["evidence_identity"] == second["evidence_identity"]


def test_withheld_directory_page_cannot_become_visible_evidence():
    from kernel_idle_governor import read_governor_action

    action = read_governor_action(
        ".", turn=2, read_ok=False,
        delivery={"schema": "navigation-page/1", "result_set_digest": "set-1",
                  "page_evidence_digest": "page-1", "host_outcome": "REFUSED",
                  "delivery_status": "WITHHELD", "requested_bytes_delivered": 0},
        deliver_requested=True, observation="[read] REFUSED",
    )

    assert action.get("undelivered") is True
    assert "evidence_identity" not in action
