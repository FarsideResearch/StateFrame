"""MutationReceipt.from_mapping must accept the flat shape as_dict writes and the nested producer shape."""

from stateframe.engine.kernel_authored_mutation import MutationReceipt


def _receipt(variant_id: str = "var-abc") -> MutationReceipt:
    return MutationReceipt(
        check_identity="chk:1",
        variant_id=variant_id,
        result="EDIT_SENSITIVE",
        reason="candidate_pass_mutant_behavior_fail",
        authored_receipt_id="rcpt-1",
    )


def test_round_trip_preserves_variant_id() -> None:
    original = _receipt("var-abc")
    assert MutationReceipt.from_mapping(original.as_dict()) == original


def test_nested_producer_shape_still_parses() -> None:
    value = {
        "check_identity": "chk:1",
        "variant": {"variant_id": "nested-1", "path": "a.py"},
        "result": "SURVIVED",
        "reason": "mutant_survived_same_observation",
    }
    assert MutationReceipt.from_mapping(value).variant_id == "nested-1"


def test_flat_shape_wins_when_both_present() -> None:
    value = {"check_identity": "chk:1", "variant_id": "flat-1",
             "variant": {"variant_id": "nested-1"}, "result": "SURVIVED"}
    assert MutationReceipt.from_mapping(value).variant_id == "flat-1"


def test_empty_flat_falls_back_to_nested() -> None:
    value = {"check_identity": "chk:1", "variant_id": "",
             "variant": {"variant_id": "nested-1"}, "result": "SURVIVED"}
    assert MutationReceipt.from_mapping(value).variant_id == "nested-1"


def test_non_mapping_variant_yields_empty_without_raising() -> None:
    value = {"check_identity": "chk:1", "variant": "not-a-mapping", "result": "SURVIVED"}
    assert MutationReceipt.from_mapping(value).variant_id == ""


def test_non_str_variant_ids_yield_empty_without_raising() -> None:
    flat_int = {"check_identity": "chk:1", "variant_id": 123, "result": "SURVIVED"}
    nested_none = {"check_identity": "chk:1", "variant": {"variant_id": None}, "result": "SURVIVED"}
    assert MutationReceipt.from_mapping(flat_int).variant_id == ""
    assert MutationReceipt.from_mapping(nested_none).variant_id == ""


def test_missing_variant_yields_empty() -> None:
    assert MutationReceipt.from_mapping({"check_identity": "chk:1"}).variant_id == ""
