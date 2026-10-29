"""Append-only epochs for the opt-in cache-stable F4 frame (STATEFRAME_FRAME_EPOCH).

Why: llama.cpp ``cache_prompt`` reuses the longest common token prefix of consecutive
requests.  A frame that rewrites early bytes every turn pays a cold prefill every turn.

Request layout inside one epoch::

    [stable block]      system, ticket, current source blocks   (structural)
    [epoch header]      epoch id + digests                       (immutable in epoch)
    [snapshot]          F4 state + clause table at epoch start   (immutable in epoch)
    [log]               ONE chronological append-only list of state deltas and
                        closed real tool exchanges, in arrival order
    [volatile tail]     host notices + decision handoff          (rewritten every turn)

Invariant: within an epoch, every byte before the volatile tail of request N is a byte
prefix of request N+1.  A new epoch (re-rendered snapshot) starts only when

* ``structural_change`` - any STRUCTURAL_CHANGE_FIELDS input changed,
* ``exchange_rewritten`` - an already-logged call id reappears with different bytes,
* ``delta_budget`` - the log would exceed ``STATEFRAME_FRAME_EPOCH_BUDGET`` tokens,
* ``headroom`` - the whole request would not fit the caller's ceiling; the fresh epoch is
  measured once more and only then does the caller report overflow.

Epoch state is a derived render cache, never decision state: it is rebuilt from the F4
projection + real-exchange ring and is deliberately NOT restored on a worker restart
(the runtime rejects this switch together with KERNEL_CONTINUATION_RESTART=1).
This module owns no authorization, tool construction or state projection.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence

from kernel_stateframe_f4 import CLAUSE_HEADER, HANDOFF_HEADER, STATE_HEADER, state_message


MODE_ENV = "STATEFRAME_FRAME_EPOCH"
BUDGET_ENV = "STATEFRAME_FRAME_EPOCH_BUDGET"
SCHEMA = "stateframe-frame-epoch/2"
EPOCH_HEADER = "[StateFrame epoch]\n"
DELTA_HEADER = "[StateFrame epoch delta]\n"
DEFAULT_BUDGET_TOKENS = 4096

# Declared, and exactly what render() compares.  Ordinary state/decision/clause changes
# and new exchanges are log appends, never rollovers.
STRUCTURAL_CHANGE_FIELDS = ("stable_messages", "tools")
ROLLOVER_REASONS = ("structural_change", "exchange_rewritten", "delta_budget", "headroom")
_VISIBLE_STATE_KEYS = ("schema", "revision", "state")
_NON_STABLE_PREFIXES = (
    STATE_HEADER, HANDOFF_HEADER, CLAUSE_HEADER,
    "[Working source:", "[Current source:", "CURRENT_SOURCE ",
    "[Observed source:", "[Retained source:", "[Source content:",
)


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def budget_tokens(environ: Mapping[str, str] | None = None) -> int:
    """Positive log budget (local token estimate; a rollover guard, not cache accounting)."""
    source = os.environ if environ is None else environ
    raw = source.get(BUDGET_ENV, str(DEFAULT_BUDGET_TOKENS))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{BUDGET_ENV} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{BUDGET_ENV} must be a positive integer")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _tokens(messages: Sequence[Mapping[str, object]]) -> int:
    from kernel_context import token_estimate
    return token_estimate({"messages": list(messages), "tools": []}) if messages else 0


def _copy_messages(messages: Sequence[Mapping[str, object]] | None) -> list[dict]:
    return [copy.deepcopy(dict(message)) for message in (messages or ())
            if isinstance(message, Mapping)]


def message_bytes(messages: Sequence[Mapping[str, object]]) -> bytes:
    """Per-message canonical serialization (same framing as kernel_stateframe_f4.frame_bytes)."""
    return b"".join(_canonical(dict(message)) + b"\n" for message in messages)


def common_prefix_bytes(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def stable_prefix_messages(prefix: Sequence[Mapping[str, object]],
                           source_messages: Sequence[Mapping[str, object]] = ()) -> list[dict]:
    """Structural block: non-source prefix first, current source blocks last.

    Sources are last so a source change (an edit) rolls the epoch while the system/ticket
    bytes before it stay a reusable common prefix.
    """
    result = [copy.deepcopy(dict(message)) for message in prefix
              if isinstance(message, Mapping)
              and not str(message.get("content") or "").startswith(_NON_STABLE_PREFIXES)]
    result.extend({"role": "user", "content": str(item.get("content", ""))}
                  for item in source_messages if isinstance(item, Mapping))
    return result


def _visible_payload(snapshot: Mapping[str, object]) -> dict[str, object]:
    return {key: copy.deepcopy(snapshot[key]) for key in _VISIBLE_STATE_KEYS if key in snapshot}


def valid_pair(pair: object) -> bool:
    """A closed exchange in the two-message form: one assistant carrying EXACTLY one call
    whose id equals the single tool result's id.  Anything else cannot be closed here."""
    if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray)):
        return False
    if len(pair) != 2 or not all(isinstance(item, Mapping) for item in pair):
        return False
    assistant, result = pair
    if assistant.get("role") != "assistant" or result.get("role") != "tool":
        return False
    call_id = result.get("tool_call_id")
    calls = assistant.get("tool_calls")
    if (not isinstance(call_id, str) or not call_id or not isinstance(calls, Sequence)
            or isinstance(calls, (str, bytes, bytearray)) or len(calls) != 1):
        return False
    call = calls[0]
    return isinstance(call, Mapping) and call.get("id") == call_id


def _diff(before: object, after: object, path: tuple[str, ...] = ()) -> list[dict[str, object]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        changes: list[dict[str, object]] = []
        for key in sorted(set(before) | set(after), key=str):
            child = path + (str(key),)
            if key not in after:
                changes.append({"op": "delete", "path": list(child)})
            elif key not in before:
                changes.append({"op": "set", "path": list(child), "value": copy.deepcopy(after[key])})
            else:
                changes.extend(_diff(before[key], after[key], child))
        return changes
    if before != after:
        return [{"op": "set", "path": list(path), "value": copy.deepcopy(after)}]
    return []


def apply_state_changes(snapshot: Mapping[str, object],
                        changes: Sequence[Mapping[str, object]]) -> dict:
    """Apply delta ops (as rendered) to one visible F4 payload."""
    result = copy.deepcopy(dict(snapshot))
    for change in changes:
        path = change.get("path") if isinstance(change, Mapping) else None
        if not isinstance(path, list) or not path:
            continue
        cursor: dict = result
        for key in path[:-1]:
            child = cursor.get(key)
            if not isinstance(child, dict):
                child = {}
                cursor[key] = child
            cursor = child
        if change.get("op") == "delete":
            cursor.pop(path[-1], None)
        elif change.get("op") == "set":
            cursor[path[-1]] = copy.deepcopy(change.get("value"))
    return result


def reconstruct_state(messages: Sequence[Mapping[str, object]]) -> dict:
    """Visible F4 state a reader derives from ONE rendered request (snapshot + deltas)."""
    snapshot: dict | None = None
    result: dict | None = None
    for message in messages:
        content = str(message.get("content") or "") if isinstance(message, Mapping) else ""
        if content.startswith(STATE_HEADER + "\n"):
            if snapshot is not None:
                raise ValueError("rendered epoch has more than one StateFrame snapshot")
            snapshot = json.loads(content.split("\n", 1)[1])
            result = copy.deepcopy(snapshot)
        elif content.startswith(DELTA_HEADER):
            if result is None:
                raise ValueError("epoch delta precedes its snapshot")
            payload = json.loads(content.split("\n", 1)[1])
            result = apply_state_changes(result, payload.get("state_changes", ()))
    if result is None:
        raise ValueError("rendered epoch has no StateFrame snapshot")
    return result


@dataclasses.dataclass(frozen=True)
class EpochRender:
    messages: tuple[dict, ...]
    persistent_count: int          # messages before the volatile tail
    epoch_id: int
    rolled_over: bool
    rollover_reason: str | None
    delta_tokens: int
    log_messages: int
    persistent_bytes: int
    total_bytes: int


@dataclasses.dataclass
class _Epoch:
    epoch_id: int
    stable: list[dict]
    stable_digest: str
    tools_digest: str
    header: dict
    snapshot_message: dict
    snapshot_blocks: list[dict]
    log: list[dict]
    logged_calls: dict[str, str]       # call id -> pair digest
    last_payload: dict
    last_blocks: list[dict]
    delta_tokens: int

    def persistent(self) -> list[dict]:
        return copy.deepcopy([*self.stable, self.header, self.snapshot_message,
                              *self.snapshot_blocks, *self.log])


class FrameEpoch:
    """Append-only renderer for one runtime episode (in memory only)."""

    def __init__(self, *, budget: int | None = None):
        self.budget_tokens = budget_tokens() if budget is None else budget
        if not isinstance(self.budget_tokens, int) or self.budget_tokens <= 0:
            raise ValueError("epoch budget must be a positive integer")
        self._epoch: _Epoch | None = None
        self._next_id = 1
        self.last_render: EpochRender | None = None

    @property
    def epoch_id(self) -> int:
        return self._epoch.epoch_id if self._epoch else 0

    def _fresh(self, stable: list[dict], tools_digest: str, payload: dict, message: dict,
               blocks: list[dict], pairs: list[tuple[dict, dict]]) -> _Epoch:
        epoch_id = self._next_id  # consumed only when this epoch is committed
        stable_digest = _digest(stable)
        header = {"role": "user", "content": EPOCH_HEADER + json.dumps({
            "schema": SCHEMA, "epoch_id": epoch_id, "stable_digest": stable_digest,
            "tools_digest": tools_digest,
            "reading": ("state below is the snapshot at epoch start; each later "
                        "[StateFrame epoch delta] updates it in order; the newest value wins"),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
        epoch = _Epoch(epoch_id, stable, stable_digest, tools_digest, header, message, blocks,
                       [], {}, copy.deepcopy(payload), copy.deepcopy(blocks), 0)
        # Exchanges already closed at epoch start belong to the new base, in ring order.
        for pair in pairs:
            epoch.log.extend(copy.deepcopy(list(pair)))
            epoch.logged_calls[pair[1]["tool_call_id"]] = _digest(list(pair))
        epoch.delta_tokens = _tokens(epoch.log)
        return epoch

    def _delta(self, epoch: _Epoch, changes: list[dict], blocks: list[dict] | None,
               previous: Mapping[str, object], current: Mapping[str, object]) -> dict:
        payload: dict[str, object] = {
            "epoch_id": epoch.epoch_id,
            "from_revision": previous.get("revision"),
            "to_revision": current.get("revision"),
            "state_changes": changes,
        }
        if blocks is not None:
            payload["clause_table"] = [str(item.get("content") or "") for item in blocks]
        return {"role": "user", "content": DELTA_HEADER + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}

    def _plan(self, stable: list[dict], tools_digest: str, payload: dict, message: dict,
              blocks: list[dict], pairs: list[tuple[dict, dict]]) -> tuple[_Epoch, str | None]:
        """Return the epoch to render (a mutated copy or a fresh one) + rollover reason."""
        current = self._epoch
        if current is None:
            return self._fresh(stable, tools_digest, payload, message, blocks, pairs), None
        if _digest(stable) != current.stable_digest or tools_digest != current.tools_digest:
            return self._fresh(stable, tools_digest, payload, message, blocks, pairs), "structural_change"
        epoch = copy.deepcopy(current)
        new_pairs: list[tuple[dict, dict]] = []
        for pair in pairs:
            call_id = pair[1]["tool_call_id"]
            seen = epoch.logged_calls.get(call_id)
            if seen is None:
                new_pairs.append(pair)
            elif seen != _digest(list(pair)):
                return (self._fresh(stable, tools_digest, payload, message, blocks, pairs),
                        "exchange_rewritten")
        # Chronology: the ring reports an exchange only once it is closed, and this turn's
        # projected state already reflects it, so newly closed exchanges are appended
        # first and the delta that brings the state current follows them.
        appended: list[dict] = []
        for pair in new_pairs:
            appended.extend(copy.deepcopy(list(pair)))
            epoch.logged_calls[pair[1]["tool_call_id"]] = _digest(list(pair))
        changes = _diff(epoch.last_payload, payload)
        blocks_changed = _canonical(blocks) != _canonical(epoch.last_blocks)
        if changes or blocks_changed:
            appended.append(self._delta(epoch, changes, blocks if blocks_changed else None,
                                        epoch.last_payload, payload))
        added = _tokens(appended)
        if appended and epoch.delta_tokens + added > self.budget_tokens:
            return self._fresh(stable, tools_digest, payload, message, blocks, pairs), "delta_budget"
        epoch.log.extend(appended)
        epoch.delta_tokens += added
        epoch.last_payload = copy.deepcopy(payload)
        epoch.last_blocks = copy.deepcopy(blocks)
        return epoch, None

    def render(self, *, stable_messages: Sequence[Mapping[str, object]],
               snapshot: Mapping[str, object],
               snapshot_blocks: Sequence[Mapping[str, object]] = (),
               exchanges: Sequence[Sequence[Mapping[str, object]]] = (),
               tail_messages: Sequence[Mapping[str, object]] = (),
               tools: object = None,
               fits: Callable[[Sequence[Mapping[str, object]]], bool] | None = None,
               ) -> EpochRender:
        """Plan, measure against ``fits`` (whole-request headroom), then commit."""
        stable = _copy_messages(stable_messages)
        payload = _visible_payload(snapshot)
        message = state_message(payload)
        blocks = _copy_messages(snapshot_blocks)
        pairs = [(copy.deepcopy(dict(pair[0])), copy.deepcopy(dict(pair[1])))
                 for pair in (exchanges or ()) if valid_pair(pair)]
        tools_digest = _digest(tools)
        tail = _copy_messages(tail_messages)
        epoch, reason = self._plan(stable, tools_digest, payload, message, blocks, pairs)
        messages = [*epoch.persistent(), *tail]
        # One measurement, at most one re-snapshot, no loop: roll only when a fresh epoch is
        # strictly smaller than the grown one; otherwise the caller reports the overflow.
        if fits is not None and reason is None and self._epoch is not None and not fits(messages):
            fresh = self._fresh(stable, tools_digest, payload, message, blocks, pairs)
            fresh_messages = [*fresh.persistent(), *tail]
            if len(message_bytes(fresh_messages)) < len(message_bytes(messages)):
                epoch, reason, messages = fresh, "headroom", fresh_messages
        self._epoch = epoch
        if epoch.epoch_id == self._next_id:
            self._next_id += 1
        persistent = messages[:len(messages) - len(tail)]
        result = EpochRender(
            messages=tuple(messages), persistent_count=len(persistent),
            epoch_id=epoch.epoch_id, rolled_over=reason is not None, rollover_reason=reason,
            delta_tokens=epoch.delta_tokens, log_messages=len(epoch.log),
            persistent_bytes=len(message_bytes(persistent)),
            total_bytes=len(message_bytes(messages)),
        )
        self.last_render = result
        return result


__all__ = [
    "BUDGET_ENV", "DEFAULT_BUDGET_TOKENS", "DELTA_HEADER", "EPOCH_HEADER", "EpochRender",
    "FrameEpoch", "MODE_ENV", "ROLLOVER_REASONS", "SCHEMA", "STRUCTURAL_CHANGE_FIELDS",
    "apply_state_changes", "budget_tokens", "common_prefix_bytes", "enabled", "message_bytes",
    "mode", "reconstruct_state", "stable_prefix_messages", "valid_pair",
]
