"""Metadata is advisory only at a field boundary without mixed instruction text."""
import hashlib
import json

import pytest

import kernel_obligation_seed as seed
import kernel_requirements_ledger as ledger
import kernel_work_record as work_record


@pytest.fixture(params=[("off", "off"), ("on", "off"), ("on", "on")])
def parser_modes(request, monkeypatch):
    alignment, grounding = request.param
    monkeypatch.setenv("STATEFRAME_REQUIREMENT_ALIGNMENT", alignment)
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", grounding)
    monkeypatch.setenv(seed.MODE_ENV, "on")
    ledger.clear_cache()
    yield
    ledger.clear_cache()


def test_metadata_prefix_scans_are_linear_for_many_short_lines(monkeypatch):
    monkeypatch.setenv("STATEFRAME_REQUIREMENT_ALIGNMENT", "off")
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", "off")
    monkeypatch.setenv(seed.MODE_ENV, "on")
    brief = "\n".join(["Keep audit events."] * 5_000)
    original = ledger._task_metadata_label
    scanned = 0
    state_counts = {}

    def counted(text, *, source_prefix="", prefix_state=None):
        nonlocal scanned
        if prefix_state is None:
            scanned += len(source_prefix)
        else:
            current = prefix_state.characters_scanned
            scanned += current - state_counts.get(prefix_state, 0)
            state_counts[prefix_state] = current
        if prefix_state is None:
            return original(text, source_prefix=source_prefix)
        return original(text, source_prefix=source_prefix, prefix_state=prefix_state)

    monkeypatch.setattr(ledger, "_task_metadata_label", counted)
    ledger.clear_cache()
    rows = ledger.parse_ticket_once("linear-scan-ticket", brief, revision="linear-v1")
    obligations = seed.extract_obligations(brief)
    scanned += sum(state.characters_scanned - observed
                   for state, observed in state_counts.items())
    ledger.clear_cache()

    assert len(rows) == 5_000
    assert len(obligations) == seed.MAX_EXTRACTED_OCCURRENCES + 1
    assert obligations[-1].overflow is True
    assert scanned <= 4 * len(brief), f"metadata prefix scanning visited {scanned} characters"


@pytest.mark.parametrize(("text", "action"), [
    ("Delete entries with Title: expired.", "Delete entries"),
    ("Encrypt entries with Symptom: exposed.", "Encrypt entries"),
    ('Keep records matching "Title: expired".', "Keep records"),
    ('"Title: expired" is the value to delete.', "delete"),
    ("Title: Lease cleanup; Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("Title: Lease cleanup. Encrypt every stored token.", "Encrypt every stored token"),
    ("Title: Lease cleanup, Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("Title: Lease cleanup and Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("Title: Lease cleanup - Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("- Title: Lease cleanup; Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("1. Symptom: Leaks; Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("## Title: Cleanup; Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("Title: Lease cleanup\n  Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("Title: Lease cleanup\r\n  Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("- Title: Lease cleanup\n  Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("Focus: Reject stale lease tokens", "Reject stale lease tokens"),
    ("Focus: Obfuscate stored tokens.", "Obfuscate stored tokens"),
    ("Retain this prefix. Title: expired.", "Title: expired"),
    ("Delete café entries with\r\n  Title: expiré.", "Title: expiré"),
    ('Keep "this.\n  Title: expired." as the matching value.', "Title: expired"),
])
def test_instruction_labels_keep_requirement_work(parser_modes, text, action):
    built = ledger.build_ledger("metadata-ticket", text)
    spans = {row["id"]: row for row in built["spans"]}
    action_rows = [row for row in built["coverage_dispositions"]
                   if action in " ".join(spans[row["span_id"]]["quote"].split())]
    assert action_rows
    assert all(row["requirement_refs"] for row in action_rows), action_rows
    for span in spans.values():
        assert text.encode()[span["start_byte"]:span["end_byte"]].decode() == span["quote"]
        assert span["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert ledger.coverage_complete(built)
    obligations = seed.extract_obligations(text)
    work = [item for item in obligations if not item.unmappable]
    assert any(action in " ".join(item.text.split()) for item in work), obligations


@pytest.mark.parametrize("text", [
    "Title: Lease cleanup", "Title: Lease cleanup.", "- Title: Lease cleanup",
    "## Title: Lease cleanup", "Symptom: Expired leases", "Focus:",
])
def test_standalone_descriptive_fields_remain_advisory(parser_modes, text):
    built = ledger.build_ledger("metadata-ticket", text)
    assert not built["requirements"]
    assert all(row["disposition"] == "context_only" for row in built["coverage_dispositions"])
    items = seed.extract_obligations(text)
    assert items and all(item.unmappable for item in items)


@pytest.mark.parametrize("mode", ["off", "on"])
def test_seed_metadata_rule_is_shared_in_both_modes(monkeypatch, mode):
    monkeypatch.setenv(seed.MODE_ENV, mode)
    for text in ("Delete entries with Title: expired.",
                 "Symptom: Leaks; Obfuscate stored tokens.",
                 "Focus: Obfuscate stored tokens."):
        items = seed.extract_obligations(text)
        assert items and all(not item.unmappable for item in items), items


@pytest.mark.parametrize("text", [
    "Keep audit events.\n\nKeep audit events.",
    "Keep audit events. Keep audit events.",
    "Keep café events.\r\n\r\nKeep café events.",
    "Keep café events.\r\rKeep café events.",
])
def test_seed_occurrences_keep_exact_identity_through_projection(parser_modes, text):
    items = seed.extract_obligations(text)
    assert len(items) == 2
    origins = [item.to_dict()["source_spans"][0] for item in items]
    assert len({origin["id"] for origin in origins}) == 2
    assert len({origin["start_byte"] for origin in origins}) == 2
    for item, origin in zip(items, origins):
        assert item.to_dict()["source_span_ids"] == [origin["id"]]
        assert origin["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert text.encode()[origin["start_byte"]:origin["end_byte"]].decode().rstrip(".") == item.text
        restored = json.loads(json.dumps(item.to_dict()))
        operation = seed.host_seed_operations([restored])[0]
        projected = seed.source_span_from_fields(operation["fields"])
        assert projected.to_dict() == origin
        assert projected.id == origin["id"]
    store = work_record.WorkRecordStore(text)
    snapshot = json.loads(json.dumps(store.snapshot()))
    reopened = work_record.WorkRecordStore.from_state(snapshot)
    assert [seed.source_span_from_fields(step["fields"]).to_dict()
            for step in reopened.snapshot()["steps"]] == origins
    assert seed.extract_obligations(text) == items


def test_seed_identity_includes_explicit_source_id(monkeypatch):
    monkeypatch.setenv(seed.MODE_ENV, "on")
    text = "Keep audit events."
    first = seed.extract_obligations(text, source_id="ticket-a")[0].to_dict()
    second = seed.extract_obligations(text, source_id="ticket-b")[0].to_dict()
    assert first["source_spans"][0]["source_id"] == "ticket-a"
    assert second["source_spans"][0]["source_id"] == "ticket-b"
    assert first["source_span_ids"] != second["source_span_ids"]


def test_long_derived_seed_origins_survive_bounded_work_fields(monkeypatch):
    monkeypatch.setenv(seed.MODE_ENV, "on")
    prefix = "Keep audit events " + " ".join(f"segment{i}" for i in range(80))
    text = prefix + " ALPHA.\n" + prefix + " BETA."
    items = seed.extract_obligations(text)
    assert len(items) == 2 and all(len(item.text) > seed.MAX_OBLIGATION_TEXT for item in items)
    store = work_record.WorkRecordStore(text)
    for step, item in zip(store.snapshot()["steps"], items):
        origin = seed.source_span_from_fields(step["fields"]).to_dict()
        assert [origin["id"]] == item.to_dict()["source_span_ids"]
        assert [origin] == item.to_dict()["source_spans"]
        assert text.encode()[origin["start_byte"]:origin["end_byte"]].decode().rstrip(".") == item.text


def test_derived_surface_clauses_cite_the_exact_parent(monkeypatch):
    monkeypatch.setenv(seed.MODE_ENV, "on")
    text = "The model, parser, and API should preserve café labels."
    items = seed.extract_obligations(text)
    assert len(items) == 3
    for item in items:
        origin = item.source_spans[0]
        exact = text.encode()[origin.start_byte:origin.end_byte].decode()
        # The last subject/predicate is contiguous; the first two combine
        # separate subject and predicate positions and must cite their parent.
        assert exact == (item.text if item.text.startswith("API ") else text)
        assert seed.source_span_from_fields(seed.obligation_step_fields(item)) == origin


def test_compatibility_candidate_offsets_are_not_synthetic():
    text = "   Keep café events.\r\n Keep café events."
    spans = seed._candidate_spans(text)
    assert len(spans) == 2 and spans[0] != spans[1]
    assert [text[start:end] for start, end in spans] == ["Keep café events"] * 2


@pytest.mark.parametrize("mutation", ["span_id", "refs", "offset", "hash"])
def test_forged_serialized_origins_are_rejected(monkeypatch, mutation):
    monkeypatch.setenv(seed.MODE_ENV, "on")
    item = seed.extract_obligations("Keep audit events.")[0].to_dict()
    if mutation == "span_id":
        item["source_spans"][0]["id"] = "span-wrong"
    elif mutation == "refs":
        item["source_span_ids"] = ["span-wrong"]
    elif mutation == "offset":
        item["source_spans"][0]["start_byte"] = True
    else:
        item["source_spans"][0]["source_sha256"] = "not-a-hash"
    with pytest.raises(ValueError):
        seed.host_seed_operations([json.loads(json.dumps(item))])


@pytest.mark.parametrize("origin", [[], ["brief", "!", 0, 1], ["brief", "AAAA", 0, 1]])
def test_malformed_compact_origins_are_rejected(origin):
    with pytest.raises(ValueError):
        seed.source_span_from_fields({"origin": origin})


def test_legacy_manual_obligations_still_project_without_an_origin():
    item = seed.Obligation("obl-000", "Keep audit events.", "constraint")
    fields = seed.host_seed_operations([item.to_dict()])[0]["fields"]
    assert fields["source_text"] == item.text
    assert seed.source_span_from_fields(fields) is None
