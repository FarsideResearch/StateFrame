"""Public witness-filter regression for span-facts provider failures."""
from __future__ import annotations

import sys

import pytest

import witness_replay as wr


_SPAN = ["def get_name(record):", "    return record.name"]
_OLD_LINE = "    return record.name"
_UNBOUND_GUARD = (
    1,
    _OLD_LINE,
    "    if stale:\n        return record.name\n    return record.name",
)
_EXPECTED_DEFECT_REASON = (
    "unsupported: span facts defect: the witness cannot judge this candidate"
)


@pytest.mark.parametrize("fault", ("import", "runtime"))
def test_filter_candidates_marks_span_facts_fault_unsupported(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    if fault == "import":
        monkeypatch.setitem(sys.modules, "mutations", None)
    else:
        import mutations

        def fail(_span_lines: list[str]) -> None:
            raise RuntimeError("forced span-facts failure")

        monkeypatch.setattr(mutations, "span_facts", fail)

    survivors, evidence = wr.filter_candidates(
        "Add a guard before reading the record name.", _SPAN, [_UNBOUND_GUARD]
    )

    assert survivors == [_UNBOUND_GUARD]
    (verdict,) = evidence
    assert verdict["supported"] is False, verdict
    assert verdict["family_witness"] is None, verdict
    assert verdict["shape"] is None, verdict
    assert verdict["delta"] == "unknown", verdict
    assert verdict["survived"] is True, verdict
    assert verdict["reason"] == _EXPECTED_DEFECT_REASON, verdict


def test_filter_candidates_still_kills_unbound_guard_with_span_facts() -> None:
    survivors, evidence = wr.filter_candidates(
        "Add a guard before reading the record name.", _SPAN, [_UNBOUND_GUARD]
    )

    assert survivors == []
    (verdict,) = evidence
    assert verdict["supported"] is True, verdict
    assert verdict["family_witness"] is False, verdict
    assert verdict["shape"] == "contact_family", verdict
    assert verdict["survived"] is False, verdict
    assert "not bound above the boundary" in verdict["reason"], verdict


_THRESHOLD = (1, "    if count > limit:", "    if count < limit:")
_THRESHOLD_SPAN = ["def check(count, limit):", "    if count > limit:", "        return True"]


@pytest.mark.parametrize("task", (
    "Values equal to the limit are rejected; the boundary must be included.", "Fix it."))
def test_span_facts_fault_keeps_the_boolean_verdict(monkeypatch: pytest.MonkeyPatch, task: str) -> None:
    import mutations

    mirror = (1, "    if count > limit:", "    if limit < count:")
    healthy = wr.filter_candidates(task, _THRESHOLD_SPAN, [_THRESHOLD, mirror])

    def fail(_span_lines: list[str]) -> None:
        raise RuntimeError("forced span-facts failure")

    monkeypatch.setattr(mutations, "span_facts", fail)
    assert wr.filter_candidates(task, _THRESHOLD_SPAN, [_THRESHOLD, mirror]) == healthy
    assert mirror not in healthy[0]


def test_span_facts_fault_emits_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted = []
    monkeypatch.setattr(wr, "emit_reason", lambda *args, **fields: emitted.append((args, fields)))
    monkeypatch.setitem(sys.modules, "mutations", None)

    wr.filter_candidates("Add a guard.", _SPAN, [_UNBOUND_GUARD])

    assert emitted[0][0][:2] == ("degrade", "witness_replay:span_facts_unavailable")
    assert emitted[0][1]["phase"] == "import"
