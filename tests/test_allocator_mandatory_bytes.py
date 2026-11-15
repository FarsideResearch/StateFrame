"""The allocator mandatory-byte bound is one formula: the proof wire and the public entry agree.

`mandatory_bytes(snapshot)` must be the exact wire that prepare_candidate's
FLOOR_INFEASIBLE proof carries, so any caller pricing a snapshot uses the same bound.
"""
from __future__ import annotations

from kernel_working_set_prepare import mandatory_bytes, prepare_candidate
from kernel_working_set_stop import ReasonCode
from test_working_set_prepare import snapshot, source

_PATH = "a.py"
_TEXT = "".join(f"VALUE_{n:04d} = {n}\n" for n in range(2800))


def _owned_snapshot(byte_cap: int):
    return snapshot(source("a", _TEXT, state="DUE"), byte_cap=byte_cap)


def test_mandatory_bytes_is_the_floor_infeasible_proof_wire() -> None:
    expected = mandatory_bytes(_owned_snapshot(byte_cap=1))
    cap = len(expected) - 1
    result = prepare_candidate(_owned_snapshot(byte_cap=cap))
    assert result.reason is ReasonCode.FLOOR_INFEASIBLE
    assert result.proof.mandatory_wire == expected
    assert result.proof.byte_cap == cap


def test_mandatory_bytes_is_stable_across_cap_values() -> None:
    assert mandatory_bytes(_owned_snapshot(byte_cap=1)) == mandatory_bytes(
        _owned_snapshot(byte_cap=10_000_000))
