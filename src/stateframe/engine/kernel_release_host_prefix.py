"""Read-only binding of enclosing host contributions prepared before sealing."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time

from kernel_frame_optional_allocation import FrameAllocationReplanNeeded


def _freeze(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class ReleasePrefixBoundary:
    request_id: str
    allocation_digest: str | None
    messages: bytes
    tools: bytes


def bind_release_prefix_boundary(capture, messages, tools) -> None:
    """Record host-side authority immediately before calling the enclosing host."""
    allocation = getattr(capture, "frame_request_allocation", None)
    if getattr(capture, "_release_prefix_boundary", None) is not None:
        # A transport retry must not acquire a new re-capture allowance.
        return
    capture._release_prefix_boundary = ReleasePrefixBoundary(
        capture.request_id, allocation.digest if allocation is not None else None,
        _freeze(messages), _freeze(tools),
    )


def qualify_release_host_prefix(capture, runtime, body, *, deadline=None):
    """Strictly bind prepared wrapper inputs; never recapture or repair them."""
    boundary = getattr(capture, "_release_prefix_boundary", None)
    if boundary is None:
        return body
    if not isinstance(boundary, ReleasePrefixBoundary):
        raise FrameAllocationReplanNeeded("release prefix boundary is invalid")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("request deadline expired before release prefix binding")
    if (runtime is None or capture.request_id != boundary.request_id
            or runtime.current_request_id != boundary.request_id):
        raise FrameAllocationReplanNeeded("release prefix request authority changed")
    if (_freeze(body.get("messages")) != boundary.messages
            or _freeze(body.get("tools", [])) != boundary.tools):
        raise FrameAllocationReplanNeeded("release wrapper changed prepared inputs")
    record = getattr(capture, "qualified_request", None)
    if record is not None:
        record.validate(capture, runtime)
    return body
