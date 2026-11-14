"""Allocator floor failures caused by a small slot use the existing typed stop."""
from types import SimpleNamespace

from kernel_endpoint_size import resolve_endpoint_size
from kernel_working_set_live import allocator_context_floor
from kernel_working_set_stop import ReasonCode


def test_floor_infeasible_from_context_maps_without_suspension_or_replay():
    size = resolve_endpoint_size({"STATEFRAME_CONTEXT_TOKENS": "16384"})
    snapshot = SimpleNamespace(context_limit=size.context_tokens)
    result = SimpleNamespace(reason=ReasonCode.FLOOR_INFEASIBLE, floor_bytes=60_000)

    stop = allocator_context_floor(snapshot, result, size=size)

    assert stop is not None
    fields = stop.terminal_fields()
    assert fields["result"] == "RUN_INCOMPLETE"
    assert fields["reason_code"] == "CONTEXT_TOO_SMALL"
    assert fields["configured_context_tokens"] == 16_384
    assert fields["minimum_context_tokens"] > 16_384
    assert "needs at least" in fields["message"]


def test_allocator_context_floor_does_not_translate_other_refusals():
    size = resolve_endpoint_size({"STATEFRAME_CONTEXT_TOKENS": "16384"})
    snapshot = SimpleNamespace(context_limit=size.context_tokens)
    result = SimpleNamespace(reason=ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED,
                             floor_bytes=60_000)

    assert allocator_context_floor(snapshot, result, size=size) is None


def test_allocator_floor_at_normal_context_remains_a_suspension():
    size = resolve_endpoint_size({})
    snapshot = SimpleNamespace(context_limit=size.context_tokens)
    result = SimpleNamespace(reason=ReasonCode.FLOOR_INFEASIBLE, floor_bytes=60_000)

    assert allocator_context_floor(snapshot, result, size=size) is None
