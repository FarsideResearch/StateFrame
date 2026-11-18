"""Obligation seeds keep clauses intact and make every occurrence limit visible."""
import pytest

from stateframe.engine import kernel_obligation_seed as seed
from stateframe.engine import kernel_work_record as work


@pytest.fixture(autouse=True)
def _seed_on(monkeypatch):
    monkeypatch.setenv(seed.MODE_ENV, "on")


def _long_tail_clause() -> str:
    words = " ".join(f"field{i}" for i in range(90))
    return f"The service must keep every field {words} SECURITY_MUST_STAY."


def test_long_clause_keeps_tail_and_full_length():
    clause = _long_tail_clause()
    assert len(clause) > seed.MAX_OBLIGATION_TEXT
    obligations = seed.extract_obligations(clause)
    assert len(obligations) == 1
    text = obligations[0].text
    assert "SECURITY_MUST_STAY" in text
    assert "..." not in text
    assert len(text) >= len(clause) - 2  # trailing sentence punctuation may be stripped


def test_clauses_sharing_long_prefix_stay_separate():
    shared = "The service must keep the audit trail " + " ".join(
        f"segment{i}" for i in range(70))
    brief = f"{shared} ENDING_ALPHA.\n{shared} ENDING_BETA."
    assert len(shared) > 400
    texts = [item.text for item in seed.extract_obligations(brief)]
    assert len(texts) == 2
    assert any("ENDING_ALPHA" in text for text in texts)
    assert any("ENDING_BETA" in text for text in texts)
    assert len({item.id for item in seed.extract_obligations(brief)}) == 2


def test_260_distinct_items_are_all_extracted():
    brief = "\n".join(f"Add item_{i:03d}." for i in range(260))
    obligations = seed.extract_obligations(brief)
    assert len(obligations) == 260
    assert [item.id for item in obligations][-1] == "obl-259"


def test_exact_occurrence_limit_does_not_emit_overflow():
    assert seed.MAX_EXTRACTED_OCCURRENCES + 1 == work.MAX_STEPS
    brief = "\n".join("Add audit item." for _ in range(seed.MAX_EXTRACTED_OCCURRENCES))

    obligations = seed.extract_obligations(brief)

    assert len(obligations) == seed.MAX_EXTRACTED_OCCURRENCES
    assert all(not item.overflow for item in obligations)
    assert len(seed.host_seed_operations(obligations)) == seed.MAX_EXTRACTED_OCCURRENCES


def test_large_repeated_brief_has_bounded_unresolved_overflow_record():
    limit = seed.MAX_EXTRACTED_OCCURRENCES
    brief = "\n".join("Add audit item." for _ in range(2_000))

    obligations = seed.extract_obligations(brief)

    assert len(obligations) <= limit + 1, "extraction must cap retained occurrence records"
    overflow = [item for item in obligations if getattr(item, "overflow", False)]
    assert len(overflow) == 1
    assert sum(not getattr(item, "overflow", False) for item in obligations) == limit
    assert len({item.id for item in obligations}) == limit + 1
    assert "unresolved" in overflow[0].text.lower()
    assert str(limit) in overflow[0].text
    assert overflow[0].to_dict()["overflow"] is True

    operations = seed.host_seed_operations(obligations)
    assert len(operations) <= limit + 1
    assert operations[-1]["fields"]["overflow"] is True
    serialized_operations = seed.host_seed_operations([item.to_dict() for item in obligations])
    assert len(serialized_operations) == limit + 1
    assert serialized_operations[-1]["fields"]["overflow"] is True

    store = work.WorkRecordStore(brief)
    steps = store.snapshot()["steps"]
    assert len(steps) == limit + 1
    assert steps[-1]["fields"]["overflow"] is True
    assert "unresolved" in steps[-1]["text"].lower()
    assert overflow[0].id in seed.open_obligations(store.snapshot())

    manual = [seed.Obligation(f"obl-{index:03d}", "Add audit item", "behaviour")
              for index in range(limit + 20)]
    manual_operations = seed.host_seed_operations(manual)
    assert len(manual_operations) == limit + 1
    assert manual_operations[-1]["fields"]["overflow"] is True


def test_display_obligations_bounds_display_and_reports_hidden():
    brief = "\n".join(f"Add item_{i:03d}." for i in range(260))
    obligations = seed.extract_obligations(brief)
    display_obligations = getattr(seed, "display_obligations")
    shown, hidden = display_obligations(obligations)
    assert len(shown) == seed.MAX_OBLIGATIONS == 256
    assert hidden == 4
    assert shown[0].id == "obl-000"


def test_display_text_clips_to_limit_with_ellipsis():
    display_text = getattr(seed, "display_text")
    clipped = display_text("x" * 600)
    assert len(clipped) == seed.MAX_OBLIGATION_TEXT
    assert clipped.endswith("...")
    assert display_text("short") == "short"


def test_legacy_bounded_text_alias_matches_display_text():
    display_text = getattr(seed, "display_text")
    assert seed._bounded_text("y" * 600) == display_text("y" * 600)


def test_work_record_seed_with_long_clause_does_not_crash():
    # The work-record step cap is 400; the step must carry display text, not raise.
    store = work.WorkRecordStore(_long_tail_clause())
    steps = [row for row in store.snapshot()["steps"]]
    assert steps and all(len(row["text"]) <= seed.MAX_OBLIGATION_TEXT for row in steps)


def test_render_states_hidden_obligations_explicitly():
    brief = "\n".join(f"Add item_{i:03d}." for i in range(260))
    store = work.WorkRecordStore(brief)
    rendered = work.render_work_record(store.snapshot())
    assert "(228 more obligations not shown; all are tracked)" in rendered
