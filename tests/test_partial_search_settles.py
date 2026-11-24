"""A search that could not cover everything is a settled read, never an unresolved effect."""
import re
from pathlib import Path

import kernel_continuation_evidence as evidence

ENGINE = Path(evidence.__file__).resolve().parent


def test_partial_grep_is_settled():
    assert evidence.result_class_disposition("grep", "PARTIAL") == "settled_applied"


def test_every_grep_receipt_class_has_a_disposition():
    source = (ENGINE / "probes" / "react_repair.py").read_text(encoding="utf-8")
    block = re.search(r'bolt\.receipt\(\s*"grep", pat\[:80\], pat,(.*?)json\.dumps', source, re.S).group(1)
    classes = set(re.findall(r'"([A-Z_]+)"', block))
    assert classes == {"ERROR", "PARTIAL", "HITS", "NO_HITS"}
    assert {evidence.result_class_disposition("grep", item) for item in classes} <= {
        "settled_applied", "settled_not_applied"}


def test_partial_search_with_no_hits_is_never_proof_of_absence():
    import kernel_repeated_zero as repeated_zero
    import kernel_tool_reply as tool_reply

    for fields in ({}, {"hit_count": 0}, {"match_count": 0}):
        row = {"kind": "grep", "result_class": "PARTIAL", **fields}
        assert tool_reply.outcome_from_receipt(row)[0] is tool_reply.Outcome.ACCEPTED
    # A complete empty search still reports no hits.
    assert tool_reply.outcome_from_receipt(
        {"kind": "grep", "result_class": "NO_HITS"})[0] is tool_reply.Outcome.NO_HITS
    assert tool_reply.outcome_from_receipt(
        {"kind": "grep", "hit_count": 0})[0] is tool_reply.Outcome.NO_HITS

    class Runtime:
        binding = {"product_revision": "r"}

    runtime = Runtime()
    for _ in range(3):
        assert repeated_zero.observe(
            runtime, "grep", {"pattern": "x"}, "partial", zero=False) == "partial"

    source = (ENGINE / "probes" / "react_repair.py").read_text(encoding="utf-8")
    assert "zero=(_grep_total == 0 and _grep_complete))" in source
    assert re.search(r'"hit_count": \(_grep_total if _grep_total or _grep_complete\s+else None\)', source)
    assert '"\\n(no grep matches)\\n" if _grep_complete' in source
