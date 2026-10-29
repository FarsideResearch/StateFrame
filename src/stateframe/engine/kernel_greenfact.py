"""GREENFACT.

Measured: b1420 benchmark cut 4 (green + hidden-correct): done within 2 calls T 8/8 (done FIRST call 8/8) vs C 2/8 vs
P 3/8; premature benchmark cuts b1423 c11-c13: 0 dones in every arm (no harm), edits continued.
Lever = exact TEXT + exact POSITION: appended to the END of the LAST user message of the request
(splice op insert_after / role user / anchor_mode message_end). Rewording or moving it is a new arm and needs
its own screen. Rendered once per request ID and green revision (legacy callers without an ID deduplicate per revision):
when public tests are green on the CURRENT binding (owner's strict check), i.e. the candidate is green and unchanged since that
green.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable

GREENFACT_TEXT = ("\n\n[host] All tests pass on the current candidate. Calling done now will submit it. "
                  "Further edits will require re-running tests.")
GREENFACT_SHA256 = "d2c2c045587d7567948ba51f4b77fabe6580e406ebc353c7e89146119f99a6bc"
MODE_ENV = "STATEFRAME_GREENFACT"
EVENT = "done_fact_rendered"


def enabled() -> bool:
    return os.environ.get(MODE_ENV, "off").strip().lower() == "on"  # default OFF in the flagged release; FIVE enables via overlay


def _green_revision(runtime: object | None) -> tuple[str | None, str | None]:
    if runtime is None:
        return None, "no_runtime"
    # The turn-fact memo key can omit product_files; verify the live receipt uncached.
    from kernel_completion_owner import _green_current_uncached
    if not _green_current_uncached(runtime):
        return None, None
    from kernel_check_receipts import current_wire_revision
    revision = current_wire_revision(runtime)
    if not revision or revision == "UNKNOWN":
        return None, "no_revision"
    return revision, None


def _revision_set(runtime: object | None, journal: object | None, name: str) -> set[str] | None:
    for target in (runtime, journal):
        if target is None:
            continue
        value = getattr(target, name, None)
        if isinstance(value, set):
            return value
        try:
            value = set()
            setattr(target, name, value)
            return value
        except (AttributeError, TypeError):
            continue
    return None


def _emit(journal: object | None, kind: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_greenfact", dispatch=False, **fields)


def _record_skipped(
        runtime: object | None, journal: object | None, *, revision: str | None,
        reason: str, turn: object) -> None:
    seen = _revision_set(runtime, journal, "_greenfact_skipped_revisions")
    key = revision if revision is not None else f"{reason}:UNKNOWN"
    if seen is not None and key in seen:
        return
    if seen is not None:
        seen.add(key)
    fields: dict[str, object] = {"reason": reason, "turn": turn}
    if revision is not None:
        fields["revision"] = revision[:80]
    _emit(journal, "greenfact_skipped", **fields)


def maybe_append_greenfact(
        messages: object, runtime: object | None, *, journal: object | None = None,
        turn: object = None, validate: Callable[[], bool] | None = None,
        request_id: str | None = None) -> bool:
    """Append once per request/revision; calls without an ID deduplicate by revision."""
    if not enabled():
        return False
    active_journal = journal if journal is not None else getattr(runtime, "journal", None)
    if turn is None:
        turn = getattr(getattr(runtime, "owner", None), "current_step", None)
    revision, reason = _green_revision(runtime)
    if reason is not None:
        _record_skipped(
            runtime, active_journal, revision=None, reason=reason, turn=turn)
        return False
    if revision is None:
        return False

    event_fields: dict[str, object] = {"revision": revision[:80], "turn": turn}
    if isinstance(request_id, str) and request_id:
        event_fields["request_id"] = request_id
    _emit(active_journal, "greenfact_green_now", **event_fields)
    rendered_name = (
        "_greenfact_rendered_request_revisions"
        if isinstance(request_id, str) and request_id else
        "_greenfact_rendered_revisions"
    )
    rendered = _revision_set(runtime, active_journal, rendered_name)
    if rendered is None:
        _record_skipped(
            runtime, active_journal, revision=revision,
            reason="render_state_unavailable", turn=turn)
        return False
    render_key = (
        f"{revision}\0{request_id}"
        if rendered_name == "_greenfact_rendered_request_revisions"
        else revision
    )
    if render_key in rendered:
        return False
    if not isinstance(messages, list):
        _record_skipped(
            runtime, active_journal, revision=revision,
            reason="no_user_message", turn=turn)
        return False
    user_indexes = [
        index for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not user_indexes:
        _record_skipped(
            runtime, active_journal, revision=revision,
            reason="no_user_message", turn=turn)
        return False
    last = messages[user_indexes[-1]]
    content = last.get("content")
    if not isinstance(content, str):
        _record_skipped(
            runtime, active_journal, revision=revision,
            reason="non_string_content", turn=turn)
        return False

    already_present = content.endswith(GREENFACT_TEXT)
    if not already_present:
        last["content"] = content + GREENFACT_TEXT
        if validate is not None:
            try:
                accepted = validate()
            except Exception:
                last["content"] = content
                raise
            if not accepted:
                last["content"] = content
                _record_skipped(
                    runtime, active_journal, revision=revision,
                    reason="context_capacity", turn=turn)
                return False
    rendered.add(render_key)
    rendered_fields = dict(event_fields)
    rendered_fields["already_present"] = already_present
    _emit(active_journal, EVENT, **rendered_fields)
    return not already_present


assert hashlib.sha256(GREENFACT_TEXT.encode("utf-8")).hexdigest() == GREENFACT_SHA256

__all__ = ["GREENFACT_TEXT", "GREENFACT_SHA256", "EVENT", "MODE_ENV", "enabled", "maybe_append_greenfact"]
