"""Coverage is the bidirectional, versioned ownership edge for a source span."""
import copy

import pytest

import kernel_requirements_ledger as ledger


@pytest.fixture
def built(monkeypatch):
    monkeypatch.setenv("STATEFRAME_REQUIREMENT_ALIGNMENT", "off")
    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", "off")
    ledger.clear_cache()
    yield ledger.build_ledger("ownership-ticket", "Sort results by timestamp.\nEncrypt stored tokens.")
    ledger.clear_cache()


def assert_invalid(built):
    assert ledger.coverage_complete(built) is False
    with pytest.raises(ledger.LedgerError):
        ledger.validate_ledger(built)


@pytest.mark.parametrize("mutation", [
    "wrong_version", "wrong_owner", "duplicate_key", "dangling_ref", "missing_ref",
    "missing_row", "unreferenced_source_span", "advisory_owned_span", "extra_wrong_owner",
    "wrong_version_on_advisory", "missing_span", "non_mapping_requirement",
    "non_mapping_coverage", "non_mapping_ref", "boolean_version", "boolean_ref_version",
])
def test_invalid_ownership_is_rejected_by_both_apis(built, mutation):
    requirements = built["requirements"]
    coverage = built["coverage_dispositions"]
    if mutation == "wrong_version":
        coverage[0]["requirement_refs"][0]["version"] = 99
    elif mutation == "wrong_owner":
        coverage[0]["requirement_refs"] = copy.deepcopy(coverage[1]["requirement_refs"])
    elif mutation == "duplicate_key":
        requirements.append(copy.deepcopy(requirements[0]))
    elif mutation == "dangling_ref":
        coverage[0]["requirement_refs"][0]["id"] = "R-MISSING"
    elif mutation == "missing_ref":
        coverage[0]["requirement_refs"] = []
    elif mutation == "missing_row":
        coverage.pop()
    elif mutation == "unreferenced_source_span":
        requirements[0]["source_span_ids"].append(built["spans"][1]["id"])
    elif mutation == "advisory_owned_span":
        coverage[0].update(disposition="context_only", requirement_refs=[])
    elif mutation == "extra_wrong_owner":
        coverage[0]["requirement_refs"].extend(copy.deepcopy(coverage[1]["requirement_refs"]))
    elif mutation == "wrong_version_on_advisory":
        coverage[0]["disposition"] = "context_only"
        coverage[0]["requirement_refs"][0]["version"] = 99
    elif mutation == "missing_span":
        requirements[0]["source_span_ids"] = ["S-MISSING"]
    elif mutation == "non_mapping_requirement":
        requirements.append(None)
    elif mutation == "non_mapping_coverage":
        coverage.append(None)
    elif mutation == "non_mapping_ref":
        coverage[0]["requirement_refs"] = [None]
    elif mutation == "boolean_version":
        requirements[0]["version"] = True
    elif mutation == "boolean_ref_version":
        coverage[0]["requirement_refs"][0]["version"] = True
    assert_invalid(built)


def test_valid_many_to_many_ownership(built):
    span_ids = [row["id"] for row in built["spans"]]
    refs = [{"id": req["id"], "version": req["version"]} for req in built["requirements"]]
    for req in built["requirements"]:
        req["source_span_ids"] = span_ids[:]
    for coverage in built["coverage_dispositions"]:
        coverage["requirement_refs"] = copy.deepcopy(refs)
    assert ledger.validate_ledger(built)
    assert ledger.coverage_complete(built)
    assert ledger.ledger_json(built)


def test_same_id_different_versions_have_separate_owners(built):
    first, second = built["requirements"]
    second.update(id=first["id"], version=first["version"] + 1)
    built["coverage_dispositions"][1]["requirement_refs"] = [
        {"id": second["id"], "version": second["version"]}]
    assert ledger.validate_ledger(built)
    assert ledger.coverage_complete(built)
    built["coverage_dispositions"][0]["requirement_refs"][0]["version"] = second["version"]
    assert_invalid(built)


def test_each_version_must_retain_its_reverse_edge(built):
    first = built["requirements"][0]
    later = copy.deepcopy(first)
    later["version"] += 1
    built["requirements"].append(later)
    assert_invalid(built)


@pytest.mark.parametrize("bad", [None, {}, {"requirements": None}, {"spans": 4}])
def test_boolean_helper_retains_malformed_input_behavior(bad):
    assert ledger.coverage_complete(bad) is False


def test_reference_validation_does_not_depend_on_optional_schema(built, tmp_path):
    built["coverage_dispositions"][0]["requirement_refs"][0]["version"] = 99
    with pytest.raises(ledger.LedgerError):
        ledger.validate_ledger(built, schema_path=tmp_path / "absent-schema.json")
