"""Metadata is advisory only at a field boundary without mixed instruction text."""
import hashlib

import pytest

import kernel_obligation_seed as seed
import kernel_requirements_ledger as ledger


@pytest.fixture(params=[("off", "off"), ("on", "off"), ("on", "on")])
def parser_modes(request, monkeypatch):
    alignment, grounding = request.param
    monkeypatch.setenv("STATEFRAME_REQUIREMENT_ALIGNMENT", alignment)
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", grounding)
    monkeypatch.setenv(seed.MODE_ENV, "on")
    ledger.clear_cache()
    yield
    ledger.clear_cache()


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
