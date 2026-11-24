"""Catalog receipt copy cost and ownership regression coverage."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from types import MethodType, SimpleNamespace
from typing import Callable, TypeVar

import pytest

import kernel_catalog
import kernel_unknowns_ledger
from kernel_catalog import coalesce_execution_receipts, normalize_receipt
from kernel_measurement import digest
from kernel_stateframe_runtime import StateFrameRuntime

_T = TypeVar("_T")
_NO_ASSUMPTIONS = object()


def _runtime(owner_rows: list[dict[str, object]], assumptions=_NO_ASSUMPTIONS):
    owner = SimpleNamespace(_receipts=owner_rows, task_row={})
    runtime = SimpleNamespace(owner=owner, test_receipts=(), host_check_receipts=(),
                              host_repair_check_receipts=(),
                              catalog_projection=SimpleNamespace(unknown_records=(object(),)),
                              _catalog_receipt_fingerprint=None, _catalog_projection_error=None)
    runtime._catalog_task_scope = MethodType(StateFrameRuntime._catalog_task_scope, runtime)
    runtime._catalog_supported_receipt = StateFrameRuntime._catalog_supported_receipt
    runtime._catalog_host_admitted = StateFrameRuntime._catalog_host_admitted
    rows = () if assumptions is _NO_ASSUMPTIONS else assumptions
    runtime._catalog_assumption_receipts = MethodType(lambda _self: tuple(rows), runtime)
    runtime._catalog_receipts = MethodType(StateFrameRuntime._catalog_receipts, runtime)
    runtime._catalog_sync = MethodType(StateFrameRuntime._catalog_sync, runtime)
    return runtime


def _count_deepcopy_calls(operation: Callable[[], _T]) -> tuple[int, _T]:
    original = copy.deepcopy
    calls = 0

    def counted(value: object, memo=None):
        nonlocal calls
        calls += 1
        return original(value, memo)

    copy.deepcopy = counted
    try:
        result = operation()
    finally:
        copy.deepcopy = original
    return calls, result


def _reference_catalog_sync(rows: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    """Plain pre-optimization path for host-owned read receipts used below."""
    collected: list[dict[str, object]] = []
    for offset, value in enumerate(rows):
        receipt = copy.deepcopy(dict(value))
        receipt.setdefault("receipt_id", f"stateframe-catalog:owner:{offset}")
        sequence = receipt.get("admitted_seq", receipt.get("sequence", receipt.get("seq")))
        if type(sequence) is not int or sequence < 0:
            receipt["seq"] = len(collected) + 1
        receipt["host_admitted"] = True
        receipt.setdefault("scope", "diagnostic")
        collected.append(receipt)

    expected: list[dict[str, object]] = []
    for receipt in coalesce_execution_receipts(collected):
        current = copy.deepcopy(dict(receipt))
        normalized = normalize_receipt(current)
        assert normalized.get("canonical_outcome") is None
        expected.append(current)
    digest(tuple(expected))
    return tuple(expected)


def _install_replay(monkeypatch: pytest.MonkeyPatch):
    replayed: list[tuple[dict[str, object], ...]] = []

    def replay(rows):
        replayed.append(tuple(rows))
        return SimpleNamespace(unknown_records=(object(),))

    monkeypatch.setattr(kernel_catalog, "replay", replay)
    monkeypatch.setattr(kernel_unknowns_ledger, "mode", lambda: "off")
    return replayed


def test_catalog_sync_halves_copy_traversals_for_twenty_thousand_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"kind": "read", "host_admitted": True,
         "payload": {"path": f"src/{index}.py", "facts": [index, {"clean": True}]}}
        for index in range(20_000)
    ]
    reference_count, expected = _count_deepcopy_calls(lambda: _reference_catalog_sync(rows))
    runtime = _runtime(rows)
    replayed = _install_replay(monkeypatch)

    actual_count, _ = _count_deepcopy_calls(runtime._catalog_sync)

    assert len(replayed) == 1
    assert replayed[0] == expected
    assert runtime._catalog_receipt_fingerprint == digest(expected)
    assert actual_count * 5 < reference_count * 3


def test_catalog_sync_rebuilds_after_input_mutation_and_appended_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"kind": "read", "host_admitted": True,
             "payload": {"facts": ["original"]}}]
    runtime = _runtime(rows)
    replayed = _install_replay(monkeypatch)

    runtime._catalog_sync()
    assert replayed[0] == _reference_catalog_sync([{
        "kind": "read", "host_admitted": True, "payload": {"facts": ["original"]},
    }])
    rows[0]["payload"]["facts"][0] = "changed"  # type: ignore[index]
    runtime._catalog_sync()
    assert replayed[1] == _reference_catalog_sync([rows[0]])
    rows.append({"kind": "read", "host_admitted": True,
                 "payload": {"facts": ["appended"]}})
    runtime._catalog_sync()
    assert replayed[2] == _reference_catalog_sync(rows)

    assert len(replayed) == 3
    assert replayed[0][0]["payload"]["facts"] == ["original"]  # type: ignore[index]
    assert replayed[1][0]["payload"]["facts"] == ["changed"]  # type: ignore[index]
    assert replayed[2][1]["payload"]["facts"] == ["appended"]  # type: ignore[index]


def test_catalog_receipt_results_remain_detached_from_inputs_and_other_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = [{"kind": "read", "host_admitted": True,
               "payload": {"facts": ["source"]}}]
    runtime = _runtime(source)
    monkeypatch.setattr(kernel_unknowns_ledger, "mode", lambda: "off")

    first = runtime._catalog_receipts()
    first[0]["payload"]["facts"].append("caller")  # type: ignore[index,union-attr]
    second = runtime._catalog_receipts()

    assert source[0]["payload"]["facts"] == ["source"]  # type: ignore[index]
    assert second[0]["payload"]["facts"] == ["source"]  # type: ignore[index,union-attr]


def test_catalog_receipts_snapshot_each_reused_sequence_item_before_advancing() -> None:
    shared_notes = ["first"]

    class ReusingReceipts(Sequence[Mapping[str, object]]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> Mapping[str, object]:
            if index == 0:
                shared_notes[:] = ["first"]
            elif index == 1:
                shared_notes[:] = ["second"]
            else:
                raise IndexError(index)
            return {"kind": "read", "host_admitted": True,
                    "payload": {"notes": shared_notes}}

    runtime = _runtime(ReusingReceipts())  # type: ignore[arg-type]

    receipts = runtime._catalog_receipts()

    assert [receipt["payload"]["notes"] for receipt in receipts] == [
        ["first"], ["second"],
    ]  # type: ignore[index]
    assert [digest(receipt) for receipt in receipts] == [
        digest({"kind": "read", "host_admitted": True,
                "payload": {"notes": ["first"]},
                "receipt_id": "stateframe-catalog:owner:0", "seq": 1,
                "scope": "diagnostic"}),
        digest({"kind": "read", "host_admitted": True,
                "payload": {"notes": ["second"]},
                "receipt_id": "stateframe-catalog:owner:1", "seq": 2,
                "scope": "diagnostic"}),
    ]


def test_catalog_diagnostic_unknown_notes_are_detached_from_payload_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_notes = ["source"]
    runtime = _runtime([{
        "kind": "read",
        "actor": "model",
        "payload": {"notes": shared_notes},
        "unknowns": [{"notes": shared_notes}],
    }])
    monkeypatch.setattr(kernel_unknowns_ledger, "mode", lambda: "off")

    receipt = runtime._catalog_receipts()[0]
    receipt["unknowns"][0]["notes"].append("caller")  # type: ignore[index,union-attr]

    assert receipt["payload"]["notes"] == ["source"]  # type: ignore[index]


@pytest.mark.parametrize("source_kind", ["assumption", "ledger"])
def test_catalog_appended_receipt_streams_are_detached(
    monkeypatch: pytest.MonkeyPatch, source_kind: str,
) -> None:
    cached = [{"kind": "read", "host_admitted": True,
               "payload": {"facts": [source_kind]}}]
    runtime = _runtime([], assumptions=cached if source_kind == "assumption" else ())
    if source_kind == "ledger":
        monkeypatch.setattr(kernel_unknowns_ledger, "mode", lambda: "on")
        monkeypatch.setattr(kernel_unknowns_ledger, "observe", lambda *_args, **_kwargs: cached)
    else:
        monkeypatch.setattr(kernel_unknowns_ledger, "mode", lambda: "off")

    receipts = runtime._catalog_receipts()
    receipts[0]["payload"]["facts"].append("caller")  # type: ignore[index,union-attr]

    assert cached[0]["payload"]["facts"] == [source_kind]  # type: ignore[index]
