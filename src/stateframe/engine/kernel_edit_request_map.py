"""Send-time placement of the [CURRENT SOURCE] handle map (STATEFRAME_EDIT_PROTOCOL=handles).

The transport calls ``apply`` once per provider body, right after the read-once final
message and before visibility, budget, capacity and seal, so the map bytes are fitted
and sealed with the rest of the wire. The map is added to the outgoing body only: the
run transcript never holds it, so request N+1 retains request N's prefix up to where
N's map began, and no source chunk or open-site row ever carries a handle.

Under ``legacy_safe`` (the default) ``apply`` returns the same list object untouched.
A failure to mint is fail-open: the body is sent unchanged, the run's handle state is
cleared, and the edit executor's STALE_HANDLE refusal hands the model a repair request
with fresh handles for the touched file.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kernel_edit_handles import MAX_MAP_FILE_BYTES, compose_request_map, current_turn_has_map
from kernel_edit_protocol_switch import handles_enabled

SCHEMA = "stateframe-edit-request-map/1"
# The edit protocol run state (kernel_edit_source_tool.EditProtocolRunState) the host
# binds on its runtime; the executors read the handle map this module attaches.
STATE_ATTR = "_edit_protocol_state"


@dataclass(frozen=True)
class RunBinding:
    """The run's edit-protocol state and workspace root, for sends that carry no runtime."""
    state: object
    root: Path


# The transport does not always hold the host runtime at the send seam (no active runtime in
# plain worker runs), and it sends from a worker thread that does not inherit context
# variables. An engine run owns its process (parallel workers are subprocesses), so the host
# binds the run's state + root here, process-wide, for the run's duration.
_BINDING_LOCK = threading.Lock()
_RUN_BINDING: "RunBinding | None" = None


def bind_run(state: object, root: object) -> None:
    """Bind (or with ``state=None`` clear) the run's state and workspace root."""
    global _RUN_BINDING
    with _BINDING_LOCK:
        _RUN_BINDING = None if state is None else RunBinding(state=state, root=Path(str(root)))


def current_binding() -> "RunBinding | None":
    with _BINDING_LOCK:
        return _RUN_BINDING


def _workspace_reader(root: Path) -> Callable[[str], "str | None"]:
    def read(relative: str) -> str | None:
        try:
            path = (root / relative).resolve(strict=True)
            path.relative_to(root)
            if not path.is_file() or path.stat().st_size > MAX_MAP_FILE_BYTES:
                return None
            return path.read_bytes().decode("utf-8")
        except (OSError, RuntimeError, ValueError, UnicodeError):
            return None
    return read


def _root(runtime: object) -> Path | None:
    value = getattr(getattr(runtime, "owner", None), "root", None)
    if value is None:
        binding = current_binding()
        value = binding.root if binding is not None else None
    if value is None:
        return None
    try:
        return Path(value).resolve(strict=True)
    except (OSError, RuntimeError, TypeError):
        return None


def _emit(journal: object, event: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(event, producer="kernel_edit_request_map", dispatch=False, **fields)


def apply(runtime: object, messages: object, *, request_ordinal: object,
          journal: object = None) -> object:
    """Return the provider message list with this request's map placed; bind the map."""
    if not handles_enabled():
        return messages
    state = getattr(runtime, STATE_ATTR, None)
    if state is None:
        binding = current_binding()
        state = binding.state if binding is not None else None
    set_request = getattr(state, "set_request", None)
    if not callable(set_request) or not isinstance(messages, list) or not messages:
        return messages
    if current_turn_has_map(messages):
        # A repair request already carries its own map and bound handles.
        return messages
    root = _root(runtime)
    if root is None or type(request_ordinal) is not int or request_ordinal < 0:
        set_request(None)
        return messages
    try:
        placed, handle_map = compose_request_map(
            messages, request_ordinal=request_ordinal, read_source=_workspace_reader(root))
    except (TypeError, ValueError) as error:
        set_request(None)
        _emit(journal, "edit_handle_map_failed", request_ordinal=request_ordinal,
              error_type=type(error).__name__)
        return messages
    set_request(handle_map)
    if handle_map is not None:
        _emit(journal, "edit_handle_map", schema=SCHEMA, request_ordinal=request_ordinal,
              handles=len(handle_map.records), files=len(handle_map.paths()))
    return placed


__all__ = ["SCHEMA", "STATE_ATTR", "RunBinding", "apply", "bind_run", "current_binding"]
