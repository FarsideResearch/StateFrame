"""The lossless seed lane must preserve shipped request content when unset."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from stateframe.engine import kernel_requirement_alignment as alignment
from stateframe.engine import kernel_obligation_seed as seed
from stateframe.engine import kernel_work_record as work


ROOT = Path(__file__).resolve().parents[1]
BASELINE_TIP = "85aaa134d7a07b397e6e553d1d6fbec8e0671544"
LOSSLESS_SWITCH = "STATEFRAME_OBLIGATION_SEED_LOSSLESS"


def _tickets() -> dict[str, str]:
    return {
        "dotted_names": (
            "The service must preserve api.client.send() and api.client.reply "
            "while updating src/api/client.py."
        ),
        "repeated_clauses": (
            "The service must retain repeated clause markers.\n"
            "The service must retain repeated clause markers."
        ),
        "metadata_labels": (
            "The service must preserve values after the Focus: label and continue normal processing.\n"
            "Title: Invoice repair overview."
        ),
        "long_brief": "\n".join(
            f"The service must retain item_{index:03d}." for index in range(512)
        ),
        "urls": (
            "The fetcher should keep https://example.test/api/v1/ response bytes "
            "and retry failed requests."
        ),
    }


@pytest.fixture(autouse=True)
def _shipped_profile_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(seed.MODE_ENV, "on")
    monkeypatch.setenv("STATEFRAME_WORK_RECORD", "on")
    monkeypatch.setenv("STATEFRAME_REQUIREMENT_ALIGNMENT", "on")
    monkeypatch.delenv(LOSSLESS_SWITCH, raising=False)


def test_seed_output_and_request_message_match_untouched_shipped_profile() -> None:
    baseline = json.loads(
        (ROOT / "tests/fixtures/obligation_seed_lossless_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline["baseline_tip"] == BASELINE_TIP
    assert baseline["profile"] == "r2hc3"
    assert baseline["profile_switches"] == {
        "STATEFRAME_OBLIGATION_SEED": "on",
        "STATEFRAME_REQUIREMENT_ALIGNMENT": "on",
        "STATEFRAME_WORK_RECORD": "on",
    }

    mismatches: list[str] = []
    for name, brief in _tickets().items():
        expected = baseline["cases"][name]
        assert hashlib.sha256(brief.encode("utf-8")).hexdigest() == expected["brief_sha256"]

        obligations = [item.to_dict() for item in seed.extract_obligations(brief)]
        canonical_seed = json.dumps(
            obligations, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if (len(obligations) != expected["seed_output_count"]
                or hashlib.sha256(canonical_seed).hexdigest() != expected["seed_output_sha256"]
                or obligations[:1] != expected["seed_output_first"]
                or obligations[-1:] != expected["seed_output_last"]):
            mismatches.append(f"{name}: obligation seed output changed")

        store = work.WorkRecordStore(brief)
        rendered = work.render_work_record(store.snapshot())
        request_message = json.dumps(
            {"role": "user", "content": rendered},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request_hash = hashlib.sha256(request_message).hexdigest()
        request_base64 = base64.b64encode(request_message).decode("ascii")
        if (request_hash != expected["request_message_sha256"]
                or request_base64 != expected["request_message_base64"]):
            mismatches.append(f"{name}: request message bytes changed")

    assert not mismatches, "; ".join(mismatches)


def test_alignment_sentence_spans_match_untouched_shipped_profile_when_lossless_is_off(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", "off")
    baseline = json.loads(
        (ROOT / "tests/fixtures/obligation_seed_lossless_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline["profile_switches"]["STATEFRAME_REQUIREMENT_ALIGNMENT"] == "on"

    mismatches: list[str] = []
    for name, brief in _tickets().items():
        expected = baseline["cases"][name]
        spans = [list(row) for row in alignment.extract_ticket_spans(brief)]
        canonical = json.dumps(spans, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
        if (len(spans) != expected["alignment_span_count"]
                or hashlib.sha256(canonical).hexdigest() != expected["alignment_span_sha256"]
                or spans[:1] != expected["alignment_span_first"]
                or spans[-1:] != expected["alignment_span_last"]):
            mismatches.append(f"{name}: shipped alignment sentence spans changed")

    assert not mismatches, "; ".join(mismatches)


def test_lossless_shapes_and_occurrence_cap_are_inactive_when_unset() -> None:
    span = seed.SourceSpan("brief", "a" * 64, 0, 1)
    item = seed.Obligation(
        "obl-000", "Keep one value.", "constraint", source_spans=(span,), overflow=True
    )

    assert "source_span_ids" not in item.to_dict()
    assert "overflow" not in item.to_dict()
    fields = seed.obligation_step_fields(item)
    assert "origin" not in fields
    assert "overflow" not in fields

    manual = [
        seed.Obligation(f"obl-{index:03d}", "Keep one value.", "constraint")
        for index in range(seed.MAX_EXTRACTED_OCCURRENCES + 1)
    ]
    operations = seed.host_seed_operations(manual)
    assert len(operations) == len(manual)
    assert all("overflow" not in operation["fields"] for operation in operations)
