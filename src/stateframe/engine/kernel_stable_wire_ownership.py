"""Out-of-band ownership for stable-wire projection and its recorded semantic view.

The host owns panels between the task and the captured newest user suffix. Native
exchange rows remain opaque. No ownership keys are added to provider messages.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


ORDER_MARKER = "\n[Stable wire order/1]\n"


class CanonicalMessages(list[dict[str, object]]):
    """Host-materialized rows whose split-frame controls have already been consumed.

    This transient provenance survives deepcopy but adds no serialized keys. It
    prevents native names in the semantic view from masquerading as wire controls.
    """

    frame_index: int | None = None

    def __init__(self, rows: Iterable[dict[str, object]] = ()) -> None:
        super().__init__(rows)
        self.frame_index = getattr(rows, "frame_index", None)


def request_exchange_start(messages: Sequence[Mapping[str, object]]) -> int:
    """Locate the native exchange by call identities, without inspecting text."""
    from kernel_stateframe_f4 import latest_complete_tool_exchange

    exchange = latest_complete_tool_exchange(messages)
    if exchange is None:
        raise ValueError("stable wire requires a complete native exchange")
    return (next(i for i, row in enumerate(messages) if row is exchange[0])
            if exchange else len(messages) - 1)


def request_frame_index(messages: Sequence[Mapping[str, object]], *, capture: object = None,
                        exchange_start: int | None = None, newest_count: int = 0) -> int:
    """Locate a plain host frame within its owned interval, excluding captured text.

    Final-layout frames immediately precede the exchange. Earlier intermediate
    layouts can have host panels after the frame; only those need a bounded
    search. Native rows, the leading preface and captured users are never candidates.
    """
    if isinstance(messages, CanonicalMessages) and messages.frame_index is not None:
        return messages.frame_index
    end = request_exchange_start(messages) if exchange_start is None else exchange_start
    system_end = next((i for i, row in enumerate(messages) if row.get("role") != "system"),
                      len(messages))
    start = system_end + 1
    captured = getattr(capture, "_framefit5_newest_user_suffix", ())
    count = len(captured) if captured else newest_count
    # A captured suffix either ends before the final-layout frame or directly
    # before the native exchange in an intermediate layout.
    captured_at_end = bool(captured) and list(messages[end - count:end]) == list(captured)
    final = end - 1
    if (not captured_at_end and start <= final < len(messages)
            and messages[final].get("role") == "user"
            and str(messages[final].get("content", "")).startswith("[StateFrame]\n")):
        return final
    end -= count
    indexes = [i for i in range(start, end)
               if messages[i].get("role") == "user"
               and isinstance(messages[i].get("content"), str)
               and messages[i]["content"].startswith("[StateFrame]\n")]
    if len(indexes) != 1:
        raise ValueError("stable wire requires exactly one host-owned StateFrame")
    return indexes[0]


@dataclass(frozen=True)
class RequestOwnership:
    prefix_end: int
    frame_index: int
    exchange_start: int
    newest_start: int
    newest_end: int

    def host_owned(self, index: int) -> bool:
        return (self.prefix_end <= index < self.exchange_start
                and index != self.frame_index
                and not self.newest_start <= index < self.newest_end)


def request_ownership(messages: Sequence[Mapping[str, object]], frame_index: int,
                      *, capture: object = None, exchange_start: int | None = None,
                      newest_count: int | None = None) -> RequestOwnership:
    """Use the host's capture and native exchange identities, never notice spelling."""
    from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
    if exchange_start is None:
        exchange_start = request_exchange_start(messages)
    if not frame_index < exchange_start < len(messages):
        raise ValueError("stable wire exchange boundary is outside the host layout")
    system_end = next((i for i, row in enumerate(messages) if row.get("role") != "system"),
                      len(messages))
    prefix_end = system_end + 1  # leading system messages and the host's task row
    if prefix_end > frame_index:
        raise ValueError("stable wire requires a system/task preface")
    captured = getattr(capture, "_framefit5_newest_user_suffix", ())
    count = len(captured) if newest_count is None else newest_count
    newest_end = frame_index if frame_index == exchange_start - 1 else exchange_start
    newest_start = newest_end - count
    if (newest_start < prefix_end
            or (newest_count is None and list(messages[newest_start:newest_end]) != list(captured))):
        raise FrameAllocationReplanNeeded("newest user suffix changed before stable wire")
    return RequestOwnership(prefix_end, frame_index, exchange_start, newest_start, newest_end)


@dataclass(frozen=True)
class ProjectionOrder:
    prefix_end: int
    tail_count: int
    notices: tuple[int, ...]
    newest_count: int

    def encode(self) -> str:
        # Small positional provenance travels in the already volatile status text,
        # outside the append-only prefix and outside the frame reconstruction.
        return ORDER_MARKER + json.dumps(
            [self.prefix_end, self.tail_count, list(self.notices), self.newest_count],
            separators=(",", ":"))


def projection_order(messages: Sequence[object], frame_index: int) -> ProjectionOrder:
    """Read strictly validated ownership after a wire JSON round trip."""
    index = frame_index + 2
    row = messages[index] if index < len(messages) else None
    content = row.get("content") if isinstance(row, Mapping) else None
    if (not isinstance(content, str) or ORDER_MARKER not in content
            or row.get("name") != "stateframe_stable_wire_status"):
        raise ValueError("stable wire recorded row ownership is missing")
    try:
        value = json.loads(content.rsplit(ORDER_MARKER, 1)[1])
    except (TypeError, ValueError) as error:
        raise ValueError("stable wire recorded row ownership is malformed") from error
    if (not isinstance(value, list) or len(value) != 4
            or any(type(value[i]) is not int for i in (0, 1, 3))
            or not isinstance(value[2], list)):
        raise ValueError("stable wire recorded row ownership is malformed")
    prefix, tail, notices, newest = value
    if (not 1 <= prefix <= frame_index or not 0 <= newest <= tail
            or not 0 <= tail < len(messages) - frame_index - 3
            or any(type(i) is not int or not 0 <= i < tail - newest for i in notices)
            or notices != sorted(set(notices))):
        raise ValueError("stable wire recorded row ownership is inconsistent")
    return ProjectionOrder(prefix, tail, tuple(notices), newest)
