"""Stable, compact identities for values that can enter provider messages.

Journal run IDs remain storage/lease identities. Wire IDs use the persisted run
seed, request step, and the content being identified. Journal traffic is never
part of a provider-visible identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping


WIRE_ID_HEX_LENGTH = 20  # 80 bits; birthday bound at 10^6 IDs is < 4.2e-13.
_WORKSPACE_ID = re.compile(r"(?<=continuation-workspaces/)[0-9a-f]{32}(?=/)")
_ABSOLUTE_WORKSPACE = re.compile(
    r"(?<![\w])/(?:[^/\s\"'\\]+/)*continuation-workspaces/[0-9a-f]{32}/repo/"
)
_PYTEST_DURATION = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")
_OBJECT_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")


def detfix3_legacy() -> bool:
    """Restore pre-DETFIX3 volatile-text handling for controlled comparison."""
    return os.environ.get("SF_LAB_DETFIX3", "on").strip().lower() == "legacy"


_MODE_LINE_EMITTED = False


def mode_line() -> str:
    """One stderr line naming effective history, memory, wire, and eviction modes."""
    from kernel_history_spool import legacy as memgrow2_legacy
    from kernel_memfix_history import legacy as memfix_legacy
    from kernel_evictfix_mode import mode as evictfix_mode
    memgrow2 = "legacy" if memgrow2_legacy() else "on"
    memfix = "legacy" if memfix_legacy() else "on"
    detfix3 = "legacy" if detfix3_legacy() else "on"
    return (f"stateframe: MEMGROW2 mode={memgrow2} DETFIX3 mode={detfix3} "
            f"MEMFIX mode={memfix} EVICTFIX mode={evictfix_mode()}")


def emit_mode_line(stream=None) -> None:
    """Write the mode line once per process. Never touches the journal or the wire."""
    global _MODE_LINE_EMITTED
    if _MODE_LINE_EMITTED:
        return
    _MODE_LINE_EMITTED = True
    import sys
    target = stream if stream is not None else sys.stderr
    print(mode_line(), file=target, flush=True)


def stable_wire_id(run_seed: object, kind: str, *position: object) -> str:
    """Derive a domain-separated ID from the persisted run seed and content."""
    identity = [str(run_seed or "unseeded"), kind, *position]
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:WIRE_ID_HEX_LENGTH]


# Task identity of the journal this process opened (one task per engine child).
# Set from the journal on create and on resume, never from a random source.
_BOUND_TASK_IDENTITY = ""


def bind_wire_task_identity(task_id: object) -> None:
    """Record the open journal's task identity for callers that hold no journal."""
    global _BOUND_TASK_IDENTITY
    _BOUND_TASK_IDENTITY = str(task_id or "")


def wire_run_seed(journal: object = None) -> str:
    """Return the wire-identity root: the task identity, never a per-run random seed.

    The host's ``run_identity_seed`` is a fresh random token per run. Rooting
    wire IDs in it made two runs of the same task, and a resumed run against
    its uninterrupted control, send different bytes. The task identity is the
    same for all of them, so IDs depend only on task + step + content.
    """
    task_id = getattr(journal, "task_id", None) if journal is not None else None
    if isinstance(task_id, str) and task_id:
        return "task:" + task_id
    if _BOUND_TASK_IDENTITY:
        return "task:" + _BOUND_TASK_IDENTITY
    return "unseeded"


def _content_digest(content: object) -> str:
    """Hash JSON-shaped identity content with stable key and number handling."""
    encoded = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checked_ordinal(value: object) -> int:
    """Require a caller-supplied ordinal that can be reconstructed on replay."""
    if type(value) is not int or value < 1:
        raise ValueError("wire identity ordinal must be a positive integer")
    return value


def content_wire_id(journal: object, kind: str, step: object,
                    content: object, *, ordinal: object = 1) -> str:
    """Derive an ID from persisted run seed, step, content, and stable ordinal.

    The default ordinal is one for idempotent identities. Callers that need
    distinct IDs for duplicate actions must recompute the ordinal from durable
    action history and pass it explicitly; this helper keeps no process state.
    """
    content_digest = _content_digest(content)
    return stable_wire_id(
        wire_run_seed(journal), kind, step, content_digest, _checked_ordinal(ordinal),
    )


def seeded_content_wire_id(seed: object, kind: str, step: object,
                           content: object, *, ordinal: object = 1) -> str:
    """Derive an ID for a nested request that has no primary audit journal."""
    seed_text = str(seed or "unseeded")
    content_digest = _content_digest(content)
    return stable_wire_id(seed_text, kind, step, content_digest,
                          _checked_ordinal(ordinal))


_WIRE_IDENTITY_METADATA_KEYS = frozenset({
    "request_id", "receipt_id", "edit_receipt", "check_receipt_id",
    "execution_id", "worker_id", "branch_id", "journal_id",
    "id", "event_id", "source_event_id", "journal_event_id",
    "message_id", "ask_id", "session_id", "sequence",
    "event_sequence", "timestamp", "timestamp_ns", "recorded_at_ns",
    "tool_call_id", "attempt_id", "experiment_id", "stimulus_correlation_id",
    "correlation_id", "correlation", "action_id", "dispatch_ref", "receipt_ref",
    "provider_call_id", "call_id",
    "_wire_identity_key", "_wire_identity_step", "_wire_identity_ordinal",
    # Journal chain hashes cover timestamps, so they differ on every run.
    "previous_hash", "record_hash",
})


# "<32-hex journal run id>:<sequence>" is a journal position under any key name
# (continuation_dispatch_ref, ...). Positions shift whenever an observer adds
# events, so they never enter a wire identity.
_JOURNAL_POSITION = re.compile(r"[0-9a-f]{32}:[0-9]+")


def wire_identity_content(value: object) -> object:
    """Remove generated identity metadata before hashing nested request content."""
    if isinstance(value, str) and _JOURNAL_POSITION.fullmatch(value):
        return "<journal-position>"
    if isinstance(value, Mapping):
        return {
            str(key): wire_identity_content(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _WIRE_IDENTITY_METADATA_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [wire_identity_content(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [wire_identity_content(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(
            item, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("wire identity content must contain JSON values")


def request_wire_id(journal: object, step: object, content: object,
                    *, purpose: object = None, ordinal: object = 1) -> str:
    """Identify one provider request from its model-visible content and purpose."""
    return content_wire_id(
        journal, "request", step,
        {"purpose": str(purpose or ""), "content": content},
        ordinal=ordinal,
    )


def wire_run_id(journal: object) -> str:
    """Public alias for a run; never expose its random journal directory name."""
    return stable_wire_id(
        wire_run_seed(journal), "wire-run",
    )


def public_wire_text(value: str, journal: object) -> str:
    """Normalize volatile model text, leaving host journal and audit data intact."""
    if detfix3_legacy():
        # Legacy comparison retains its output text, but host-generated run and
        # workspace IDs are never allowed to escape into provider messages.
        return _public_identity_text(value, journal)
    value = _PYTEST_DURATION.sub("in <duration>", value)
    value = _OBJECT_ADDRESS.sub("0x<addr>", value)
    return _public_identity_text(value, journal)


def _public_identity_text(value: str, journal: object) -> str:
    """Alias host IDs in arbitrary frame fields without rewriting source text."""
    if journal is None:
        # DETFIX4b: callers holding only stored output (no journal) still must not leak the
        # random continuation-workspace directory; a fixed placeholder is replay-stable.
        value = _ABSOLUTE_WORKSPACE.sub("continuation-workspaces/<workspace>/repo/", value)
        return _WORKSPACE_ID.sub("<workspace>", value)
    workspace_id = stable_wire_id(wire_run_seed(journal), "continuation-workspace")
    value = _ABSOLUTE_WORKSPACE.sub(
        f"continuation-workspaces/{workspace_id}/repo/", value)
    value = _WORKSPACE_ID.sub(workspace_id, value)
    host_run_id = getattr(journal, "run_id", None)
    if isinstance(host_run_id, str) and host_run_id and host_run_id in value:
        host_token = re.compile(
            rf"(?<!sha256:)(?<![0-9a-f]){re.escape(host_run_id)}(?![0-9a-f])")
        value = host_token.sub(wire_run_id(journal), value)
    return value


def public_wire_value(value: object, journal: object, *,
                      normalize_output: bool = False) -> object:
    """Detach values; normalize output only when the caller owns output fields."""
    transform = public_wire_text if normalize_output else _public_identity_text
    if isinstance(value, str):
        return transform(value, journal)
    if isinstance(value, dict):
        return {_public_identity_text(key, journal) if isinstance(key, str) else key:
                public_wire_value(item, journal, normalize_output=normalize_output)
                for key, item in value.items()}
    if isinstance(value, list):
        return [public_wire_value(item, journal, normalize_output=normalize_output)
                for item in value]
    if isinstance(value, tuple):
        return tuple(public_wire_value(item, journal, normalize_output=normalize_output)
                     for item in value)
    return value


def next_wire_id(journal: object, kind: str, *content: object,
                 ordinal: object = 1) -> str:
    """Compatibility wrapper; derive from step/content, never journal sequence."""
    step = content[0] if content and type(content[0]) is int else 0
    identity_content = content[1:] if content and type(content[0]) is int else content
    return content_wire_id(journal, kind, step, identity_content, ordinal=ordinal)


def action_receipt_ordinal(receipts: object, *, request_id: object,
                           step: object, action_kind: str,
                           target_path: object = None) -> int:
    """Recompute a duplicate-action ordinal from the durable receipt history."""
    journal_ordinal = getattr(receipts, "action_receipt_ordinal", None)
    if callable(journal_ordinal):
        return journal_ordinal(
            request_id=request_id, step=step, action_kind=action_kind,
            target_path=target_path,
        )
    if isinstance(receipts, Mapping) or isinstance(receipts, (str, bytes, bytearray)):
        return 1
    try:
        iterator = reversed(receipts)  # receipt history is append-only and indexed on disk
    except TypeError:
        return 1
    previous = 0
    for receipt in iterator:
        if not isinstance(receipt, Mapping):
            continue
        if receipt.get("turn") != step:
            # Receipt history is chronological. Earlier steps cannot contribute.
            break
        if (receipt.get("request_id") == request_id
                and receipt.get("kind") == action_kind
                and receipt.get("target_path") == target_path):
            previous += 1
    return previous + 1


def action_receipt_wire_id(
        journal: object, *, request_id: object, step: object, tool_call_id: object,
        action_kind: str, fingerprint: str, target_path: object = None,
        ordinal: object = 1) -> str:
    """Bind action receipt IDs to the request/action that produced them.

    The global journal sequence also counts host telemetry emitted after a send.
    That sequence can shift without changing the model action, so it is not a
    stable position for an action receipt. The seeded request ID, model step,
    and canonical receipt content remain stable through replay. The ordinal is
    recomputed from persisted receipt history. Provider tool-call IDs are
    excluded because they are transport metadata and may vary independently.
    """
    return content_wire_id(
        journal, "action-receipt", step, {
            "request_id": request_id if isinstance(request_id, str) else "",
            "action_kind": action_kind,
            "target_path": target_path if isinstance(target_path, str) else "",
            "fingerprint": fingerprint,
        }, ordinal=ordinal,
    )


def completion_root_budget_wire_id(journal: object) -> str:
    """Derive the completion fuel identity from the seeded journal run.

    The root budget is rendered into the opt-in completion frame. Do not use
    its standalone time-based default for a live request-path budget.
    """
    return stable_wire_id(
        wire_run_seed(journal), "completion-root-budget",
        0, "root-fuel-budget",
    )
