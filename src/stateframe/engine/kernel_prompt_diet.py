"""Control the size of repeated prompt sections through explicit feature switches.

Compaction changes presentation; source evidence and task-state ownership remain
with their producing components.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_PROMPT_DIET"
GATE_ATTR = "_stateframe_prompt_diet_gate"


def mode() -> str:
    import os
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def content_digest(value: object) -> str:
    """A stable digest over a JSON-safe value (or an already-rendered string)."""
    if isinstance(value, str):
        body = value
    else:
        body = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _active_emit(kind: str, **fields: object) -> None:
    """Journal the switch's own event on the active runtime (fired proof).  Never raises."""
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        journal = getattr(ACTIVE_RUNTIME.get(), "journal", None)
    except (ImportError, LookupError):
        return
    emit = getattr(journal, "emit", None)
    if callable(emit):
        try:
            emit(kind, producer="kernel_prompt_diet", dispatch=False, **fields)
        except Exception:  # noqa: BLE001 - telemetry never changes behaviour
            return


def unchanged_notice(block_name: str, digest: str, *, since_request_id: object = None) -> str:
    """The short stand-in emitted instead of a full block whose content has
    not changed since the last time it was rendered for this runtime."""
    ref = f" since_request_id={since_request_id}" if since_request_id is not None else ""
    _active_emit("prompt_diet_applied", stage="acted", block=str(block_name))
    return (f"[{block_name}_UNCHANGED | content identical to the last rendered "
            f"instance; digest={digest[:16]}{ref}]")


class DeltaGate:
    """Per-runtime-instance memory of the last digest rendered for each
    named block.  One gate per runtime; one digest slot per block name."""

    __slots__ = ("_digests", "_request_ids")

    def __init__(self) -> None:
        self._digests: dict[str, str] = {}
        self._request_ids: dict[str, object] = {}

    def should_render(self, block_name: str, digest: str) -> bool:
        """True on first render for this block, or whenever the digest changed."""
        return self._digests.get(block_name) != digest

    def commit(self, block_name: str, digest: str, *, request_id: object = None) -> None:
        self._digests[block_name] = digest
        self._request_ids[block_name] = request_id

    def last_request_id(self, block_name: str) -> object:
        return self._request_ids.get(block_name)


def get_gate(runtime: object) -> DeltaGate:
    """Fetch (or lazily create) the one DeltaGate for a runtime instance.

    Stored as a plain attribute on the runtime object so it rides along with
    the runtime's own lifecycle; nothing else reads or writes this attribute.
    """
    gate = getattr(runtime, GATE_ATTR, None)
    if not isinstance(gate, DeltaGate):
        gate = DeltaGate()
        setattr(runtime, GATE_ATTR, gate)
    return gate


def _tool_name(tool: object) -> str:
    if isinstance(tool, Mapping):
        function = tool.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str):
                return name.strip().lower()
        name = tool.get("name")
        if isinstance(name, str):
            return name.strip().lower()
    return ""


def filter_figure_it_out_tools(tools: Sequence[Mapping], *, suppress: bool) -> list:
    """Drop the figure_it_out operator tool schemas when ``suppress`` is True.

    ``suppress`` is the caller's own answer to "will the guidance block for
    these tools be rendered this turn" (e.g. the history-tail arm, which
    unconditionally denies ``append_operator_message``).  When it will not
    be, the ~420 prompt tokens of unused operator schema are worth nothing
    and are removed.  Never touches any other tool.
    """
    if not suppress or not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return list(tools) if isinstance(tools, Sequence) else tools
    try:
        from kernel_figure_it_out import OPERATOR_NAMES
    except ImportError:
        return list(tools)
    drop = {name.strip().lower() for name in OPERATOR_NAMES}
    kept = [tool for tool in tools if _tool_name(tool) not in drop]
    if len(kept) != len(tools):
        _active_emit("prompt_diet_applied", stage="acted", block="figure_it_out_tools",
                     dropped=len(tools) - len(kept))
    return kept
