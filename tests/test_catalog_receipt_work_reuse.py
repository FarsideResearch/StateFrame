"""Count avoidable receipt copies and public-tree conversions in Catalog A."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

import kernel_catalog as catalog


_DEEPCOPY = copy.deepcopy
_ENTRY_COUNT = 20_000


class _CopyCountDict(dict):
    copied = 0

    def __deepcopy__(self, memo):
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        type(self).copied += 1
        clone = type(self)()
        memo[id(self)] = clone
        for key, value in self.items():
            clone[_DEEPCOPY(key, memo)] = _DEEPCOPY(value, memo)
        return clone


def _large_tree():
    return {
        f"entry-{index:05d}": {
            "sequence": index,
            "labels": [f"label-{index}", None, True],
        }
        for index in range(_ENTRY_COUNT)
    }


def _plain_as_public(value: object, copied: list[int]) -> object:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return value.as_dict()
    if isinstance(value, catalog.Mapping):
        return {str(key): _plain_as_public(nested, copied) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_as_public(item, copied) for item in value]
    if isinstance(value, catalog.StrEnum):
        return value.value
    copied[0] += 1
    return _DEEPCOPY(value)


def _plain_canonical_value(value: object) -> tuple[str, int]:
    copied = [0]
    result = json.dumps(
        _plain_as_public(value, copied),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return result, copied[0]


def _plain_coalesce(receipts):
    result = []
    by_execution = {}
    projection_rank = {}

    def rank(record):
        projection = str(record.get("projection", record.get("receipt_projection", ""))).lower()
        if projection in {"rich", "canonical", "adapted"}:
            return 2
        if projection in {"legacy", "raw"}:
            return 0
        return int(any(key in record for key in (
            "canonical_before_binding", "canonical_after_binding", "observation",
            "observed", "after_binding",
        )))

    for raw in receipts:
        if not isinstance(raw, catalog.Mapping):
            continue
        record = _DEEPCOPY(dict(raw))
        execution_id = catalog._host_execution_id(record)
        if not execution_id or execution_id not in by_execution:
            if execution_id:
                by_execution[execution_id] = len(result)
                projection_rank[execution_id] = rank(record)
                record.setdefault("host_execution_id", execution_id)
                source_ids = [str(record["receipt_id"])] if record.get("receipt_id") else []
                record.setdefault("source_receipt_ids", source_ids)
            result.append(record)
            continue
        index = by_execution[execution_id]
        merged = result[index]
        source_ids = list(merged.get("source_receipt_ids", ()))
        if record.get("receipt_id"):
            source_ids.append(str(record["receipt_id"]))
        if rank(record) > projection_rank.get(execution_id, 0):
            preferred, fallback = record, merged
            projection_rank[execution_id] = rank(record)
        else:
            preferred, fallback = merged, record
        merged = _DEEPCOPY(dict(preferred))
        for key, value in fallback.items():
            if key in {"receipt_id", "source_receipt_ids"}:
                continue
            if key not in merged or merged[key] in (None, "", (), [], {}):
                merged[key] = value
        merged["source_receipt_ids"] = list(dict.fromkeys(source_ids))
        merged["host_execution_id"] = execution_id
        result[index] = merged
    return tuple(result)


def _large_receipt(*, receipt_id="receipt-1", projection="legacy"):
    payload = {
        f"entry-{index:05d}": _CopyCountDict(value=index)
        for index in range(_ENTRY_COUNT)
    }
    return {
        "kind": "read",
        "receipt_id": receipt_id,
        "host_execution_id": "execution-1",
        "path": "large.json",
        "projection": projection,
        "opaque_payload": payload,
    }


def test_reducer_reuses_owned_snapshot_and_mutations_stay_fresh():
    receipt = _large_receipt()
    _CopyCountDict.copied = 0
    expected = catalog.reduce_receipt(catalog.empty_catalog(), _DEEPCOPY(dict(receipt)))
    old_pipeline_copies = _CopyCountDict.copied

    _CopyCountDict.copied = 0
    actual = catalog.reduce_receipt(catalog.empty_catalog(), receipt)
    optimized_pipeline_copies = _CopyCountDict.copied

    assert actual == expected
    assert old_pipeline_copies == 2 * _ENTRY_COUNT
    assert optimized_pipeline_copies == _ENTRY_COUNT
    normalized = catalog.normalize_receipt(receipt)
    normalized["opaque_payload"]["entry-00000"]["value"] = -1
    assert receipt["opaque_payload"]["entry-00000"]["value"] == 0

    receipt["opaque_payload"]["entry-00000"]["value"] = 42
    changed = catalog.normalize_receipt(receipt)
    assert changed == catalog.normalize_receipt(_DEEPCOPY(dict(receipt)))
    assert changed["opaque_payload"]["entry-00000"]["value"] == 42


def test_coalesce_removes_duplicate_full_copy_without_changing_result():
    receipts = (
        _large_receipt(receipt_id="legacy", projection="legacy"),
        _large_receipt(receipt_id="rich", projection="rich"),
    )
    _CopyCountDict.copied = 0
    expected = _plain_coalesce(receipts)
    old_pipeline_copies = _CopyCountDict.copied

    _CopyCountDict.copied = 0
    actual = catalog.coalesce_execution_receipts(receipts)
    optimized_pipeline_copies = _CopyCountDict.copied

    assert actual == expected
    assert old_pipeline_copies == 3 * _ENTRY_COUNT
    assert optimized_pipeline_copies == 2 * _ENTRY_COUNT
    actual[0]["opaque_payload"]["entry-00000"]["value"] = -1
    assert receipts[0]["opaque_payload"]["entry-00000"]["value"] == 0
    assert receipts[1]["opaque_payload"]["entry-00000"]["value"] == 0
    again = catalog.coalesce_execution_receipts(receipts)
    assert again[0]["opaque_payload"]["entry-00000"]["value"] == 0


def test_canonical_and_token_paths_skip_public_tree_copies_for_native_json(monkeypatch):
    value = _large_tree()
    expected, copied_nodes_before = _plain_canonical_value(value)
    original_deepcopy = catalog.copy.deepcopy
    copied_nodes_after = 0

    def count_deepcopy(item, memo=None):
        nonlocal copied_nodes_after
        copied_nodes_after += 1
        return original_deepcopy(item, memo)

    monkeypatch.setattr(catalog.copy, "deepcopy", count_deepcopy)
    actual = catalog._canonical_value(value)
    assert actual == expected
    assert copied_nodes_before == 4 * _ENTRY_COUNT
    assert copied_nodes_after == 0
    assert catalog._estimate_tokens(value) == max(1, math.ceil(len(expected.encode("utf-8")) / 4))
    assert copied_nodes_after == 0

    value["entry-00000"]["labels"][0] = "changed"
    changed = catalog._canonical_value(value)
    assert changed != expected
    assert changed == _plain_canonical_value(value)[0]


def test_non_native_canonical_values_keep_legacy_projection():
    value = {
        1: {"bytes": b"raw"},
        "state": catalog.AssertionState.CURRENT,
    }
    expected, _ = _plain_canonical_value(value)
    assert catalog._canonical_value(value) == expected


def test_native_json_fast_path_preserves_deep_catalog_inputs(monkeypatch):
    def nested(value, depth, *, dictionaries=False):
        for _ in range(depth):
            value = {"nested": value} if dictionaries else [value]
        return value

    def plain_native_json(value):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

    deep_list = nested({"leaf": "value"}, 493)
    expected_list = plain_native_json(deep_list)
    assert catalog._canonical_value(deep_list) == expected_list
    expected_list_tokens = max(1, math.ceil(len(expected_list.encode("utf-8")) / 4))
    assert catalog._estimate_tokens(deep_list) == expected_list_tokens

    deep_dict = nested(["value"], 493, dictionaries=True)
    expected_dict = plain_native_json(deep_dict)
    assert catalog._canonical_value(deep_dict) == expected_dict
    expected_dict_tokens = max(1, math.ceil(len(expected_dict.encode("utf-8")) / 4))
    assert catalog._estimate_tokens(deep_dict) == expected_dict_tokens

    shared = {"leaf": "shared"}
    shared_value = [shared, shared]
    assert catalog._canonical_value(shared_value) == plain_native_json(shared_value)

    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(RecursionError):
        catalog._canonical_value(cyclic)

    assertion = catalog.CatalogAssertion(
        assertion_id="deep-assertion",
        subject=catalog.EntityRef("deep-entity", "file"),
        predicate="contains",
        value=nested("leaf", 493),
        revision=None,
        dependencies=(),
        receipt_refs=(),
        source_span_refs=(),
        observed_at=None,
        admitted_seq=1,
        state=catalog.AssertionState.CURRENT,
        availability=catalog.Availability.PRESENT,
    )
    state = catalog.CatalogState(assertion_records=(assertion,))
    # Isolate canonical traversal from deepcopy's separate recursion limit.
    monkeypatch.setattr(catalog, "_copy", lambda value: value)
    payload = {
        "watermark": state.watermark,
        "assertions": [item.as_dict() for item in state.assertion_records],
        "unknowns": [],
        "changes": [],
        "revisions": [],
    }
    expected_basis = hashlib.sha256(plain_native_json(payload).encode("utf-8")).hexdigest()
    assert state.basis_digest() == expected_basis


def test_reduction_observes_input_mutation_and_new_journal_rows():
    receipt = _large_receipt()
    state = catalog.reduce_receipt(catalog.empty_catalog(), receipt)
    receipt["opaque_payload"]["entry-00000"]["value"] = 1

    with pytest.raises(catalog.CatalogIntegrityError):
        catalog.reduce_receipt(state, receipt)

    receipt["receipt_id"] = "receipt-2"
    receipt["sequence"] = 2
    updated = catalog.reduce_receipt(state, receipt)
    assert updated.watermark == 2
    assert len(updated.change_records) == 2
