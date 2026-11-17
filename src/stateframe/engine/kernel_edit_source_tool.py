"""Host executor for edit_source: typed validation, one admitted publish, replay-stable.

The model addresses source through request-scoped handles minted for the
current-source map. This module turns one ``edit_source`` call into either a
closed set of typed refusals (every defect at once, nothing written) or one
published candidate generation that is then written to the workspace.

Determinism: no clock, no randomness. A re-delivered call (same request
ordinal, same arguments) returns the stored outcome as ``REPLAYED`` without
reading or writing anything, so a retried transport cannot apply an edit twice.

File bytes are never decoded and re-encoded for the write: the validator and
``EditAuthority`` compose on bytes, so BOMs and CRLF line endings survive.
"""
from __future__ import annotations

import ast
import codecs
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from kernel_destructive_edit_guard import edit_admission, guard_write
from kernel_edit_handles import HandleMap, HandleRecord, SpanKind
from kernel_edit_refusal_codes import Refusal, RefusalCode
from kernel_edit_source_batch import (
    UNRESOLVED_PATH, BatchInputError, BatchValidation, CoveredSpan, validate_edit_source_batch)
from kernel_edit_tool_names import EDIT_SOURCE_TOOL, REWRITE_SOURCE_TOOL, is_destructive_edit_tool
from kernel_paged_source import source_revision
from kernel_revision_bound_edit import CandidateTransaction, EditAuthority, EditPlanError, validate_source

AUTHORITY_REF = "edit-protocol/1"
SOURCE_RECEIPT = "edit-protocol:current-source"

# rewrite_source replaces a whole region; the shared classification must keep
# it destructive so stale-repair and guard policies treat it as such.
if not is_destructive_edit_tool(REWRITE_SOURCE_TOOL):  # pragma: no cover - import-time guard
    raise ImportError(f"{REWRITE_SOURCE_TOOL} must be classified destructive")

APPLIED = "APPLIED"
REJECTED = "REJECTED"
REPLAYED = "REPLAYED"

ReadBytes = Callable[[Path], bytes]
# Writes the published candidate; returns (applied, withheld_observation).
WriteBytes = Callable[[Path, bytes], "tuple[bool, str | None]"]
NotEditable = Callable[[str], "str | None"]


@dataclass(frozen=True)
class EditOutcome:
    """The single observation an edit protocol call produces."""

    tool: str
    status: str                      # APPLIED | REJECTED | REPLAYED
    path: str | None
    refusals: tuple[Refusal, ...]
    changed_paths: tuple[str, ...]
    generation_id: str | None
    detail: str                      # prose reason when no typed code fits ("" otherwise)

    def codes(self) -> tuple[str, ...]:
        return tuple(refusal.code.value for refusal in self.refusals)

    def as_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "status": self.status,
            "path": self.path,
            "refusals": [refusal.as_dict() for refusal in self.refusals],
            "changed_paths": list(self.changed_paths),
            "detail": self.detail,
        }

    def observation(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)

    def replayed(self) -> "EditOutcome":
        return EditOutcome(self.tool, REPLAYED, self.path, self.refusals, self.changed_paths,
                           self.generation_id, self.detail)


class EditProtocolRunState:
    """Per-run seam: the current request's handle map and the replay store.

    The current-source map wiring attaches the map per request; the executors
    only read it. Outcomes are keyed by request ordinal plus argument digest.
    """

    def __init__(self) -> None:
        self.handle_map: HandleMap | None = None
        self.outcomes: dict[str, EditOutcome] = {}

    def set_request(self, handle_map: HandleMap | None) -> None:
        if handle_map is not None and not isinstance(handle_map, HandleMap):
            raise TypeError("handle_map must be a HandleMap or None")
        self.handle_map = handle_map

    def lookup(self, handle: object) -> HandleRecord | None:
        if self.handle_map is None or not isinstance(handle, str):
            return None
        return self.handle_map.lookup(handle)

    def idempotent_key(self, tool: str, args: Mapping[str, object]) -> str:
        ordinal = self.handle_map.request_ordinal if self.handle_map is not None else -1
        payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=repr)
        digest = hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()
        return f"{tool}:{ordinal}:{digest}"

    def remember(self, key: str, outcome: EditOutcome) -> EditOutcome:
        self.outcomes[key] = outcome
        return outcome


def covered_spans(handle_map: HandleMap | None) -> tuple[CoveredSpan, ...]:
    """One validator span per minted handle record."""
    if handle_map is None:
        return ()
    return tuple(
        CoveredSpan(handle=record.handle, path=record.path, revision=record.revision,
                    start=record.span.byte_start, end=record.span.byte_end,
                    kind=record.kind.value)
        for record in handle_map.records)


def python_parse_check(path: str) -> Callable[[bytes], str | None]:
    """Parse check for composed Python bytes: an error description or None.

    A leading UTF-8 BOM is legal in a Python file (the interpreter strips it
    when it decodes the file), so it is stripped before parsing.
    """
    def check(composed: bytes) -> str | None:
        if composed.startswith(codecs.BOM_UTF8):
            composed = composed[len(codecs.BOM_UTF8):]
        result = validate_source(path, composed)
        if result.syntax_status == "FAIL":
            return "; ".join(result.diagnostics) or "syntax error"
        if result.syntax_status != "PASS":
            # Defensive: fall back to the stdlib parser if the adapter is unavailable.
            try:
                ast.parse(composed.decode("utf-8", errors="replace"), filename=path)
            except SyntaxError as error:
                return str(error)
        return None
    return check


def default_write_bytes(path: Path, data: bytes) -> tuple[bool, str | None]:
    """Write through the scope-widen guard (identity-checked for widened files)."""
    from kernel_scope_widen_guard import write as guarded_write
    guarded_write(path, data)
    return True, None


def _rejected(tool: str, *, path: str | None = None, refusals: tuple[Refusal, ...] = (),
              detail: str = "", generation_id: str | None = None) -> EditOutcome:
    return EditOutcome(tool, REJECTED, path, refusals, (), generation_id, detail)


def _each_edit(edits: object) -> list[Mapping[str, object]]:
    return [edit for edit in edits if isinstance(edit, Mapping)] if isinstance(edits, list) else []


def _old_text(edit: Mapping[str, object]) -> str:
    value = edit.get("old_text")
    return value if isinstance(value, str) else ""


def _parse_check_for(relative: str) -> Callable[[bytes], str | None] | None:
    return python_parse_check(relative) if relative.endswith(".py") else None


def _validate(edits: list[object], state: EditProtocolRunState, raw: bytes,
              relative: str, current: str) -> BatchValidation:
    ordinal = state.handle_map.request_ordinal if state.handle_map is not None else -1
    return validate_edit_source_batch(
        edits, covered_spans(state.handle_map), raw, current_revision=current,
        authority_ref=AUTHORITY_REF, inventory_ref=f"edit-protocol:q{ordinal}",
        source_receipt=SOURCE_RECEIPT, parse_check=_parse_check_for(relative))


def _publish(tool: str, root: Path, relative: str, raw: bytes, current: str,
             validation: BatchValidation, *, read_bytes: ReadBytes,
             write_bytes: WriteBytes | None, journal: object) -> EditOutcome:
    """EditAuthority -> prepare -> publish -> one workspace write."""
    abs_path = root / relative
    authority = EditAuthority(root=root, sources={relative: raw}, base_revision=current,
                              authority_ref=AUTHORITY_REF, allowed_paths=[relative],
                              journal=journal)
    transaction = CandidateTransaction(authority)
    generation = transaction.prepare(validation.plan)
    if generation.diagnostics:
        detail = "; ".join(generation.diagnostics)
        stale = any(item.startswith("STALE_") for item in generation.diagnostics)
        refusals = (Refusal.build(RefusalCode.SOURCE_CHANGED, edit_index=0, path=relative,
                                  old_text=""),) if stale else ()
        return _rejected(tool, path=relative, refusals=refusals, detail=detail)
    try:
        current_bytes = read_bytes(abs_path)
    except OSError:
        return _rejected(tool, path=relative, detail="file is missing", refusals=(
            Refusal.build(RefusalCode.SOURCE_CHANGED, edit_index=0, path=relative, old_text=""),))
    receipt = transaction.publish(current_sources={relative: current_bytes})
    if not receipt.published:
        return _rejected(tool, path=relative, generation_id=receipt.generation_id,
                         detail="; ".join(receipt.diagnostics))
    candidate = generation.sources[relative]
    applied, withheld = (write_bytes or default_write_bytes)(abs_path, candidate)
    if not applied:
        return _rejected(tool, path=relative, generation_id=receipt.generation_id,
                         detail=withheld or f"{tool} effect withheld")
    return EditOutcome(tool, APPLIED, relative, (), (relative,), receipt.generation_id, "")


def _read_current(root: Path, relative: str, read_bytes: ReadBytes,
                  edits: list[Mapping[str, object]], tool: str
                  ) -> tuple[bytes, str] | EditOutcome:
    try:
        raw = read_bytes(root / relative)
    except OSError:
        refusals = tuple(Refusal.build(RefusalCode.SOURCE_CHANGED, edit_index=index, path=relative,
                                       old_text=_old_text(edit))
                         for index, edit in enumerate(edits or [{}]))
        return _rejected(tool, path=relative, refusals=refusals, detail="file is missing")
    return raw, source_revision(raw.decode("utf-8", errors="surrogateescape"))


def execute_edit_source(root: Path, args: Mapping[str, object], *, state: EditProtocolRunState,
                        read_bytes: ReadBytes, journal: object = None,
                        not_editable: NotEditable | None = None,
                        write_bytes: WriteBytes | None = None) -> EditOutcome:
    """Validate and apply one edit_source batch; at most one publish."""
    tool = EDIT_SOURCE_TOOL
    key = state.idempotent_key(tool, args)
    stored = state.outcomes.get(key)
    if stored is not None:
        return stored.replayed()
    try:
        edits = args.get("edits") if isinstance(args, Mapping) else None
        if not isinstance(edits, list):
            raise BatchInputError("edits must be a list of {handle, old_text, new_text}")
        if not edits:
            raise BatchInputError("a batch needs at least one edit")
        mappings = _each_edit(edits)

        primary = next((record for record in (state.lookup(edit.get("handle")) for edit in mappings)
                        if record is not None), None)
        if primary is None:
            refusals = tuple(Refusal.build(RefusalCode.STALE_HANDLE, edit_index=index,
                                           path=UNRESOLVED_PATH,
                                           old_text=_old_text(edit) if isinstance(edit, Mapping) else "")
                             for index, edit in enumerate(edits))
            return state.remember(key, _rejected(tool, refusals=refusals))

        relative = primary.path
        reason = not_editable(relative) if not_editable is not None else None
        if reason:
            return state.remember(key, _rejected(tool, path=relative, detail=reason, refusals=(
                Refusal.build(RefusalCode.NOT_EDITABLE, edit_index=0, path=relative, old_text=""),)))

        current = _read_current(root, relative, read_bytes, mappings, tool)
        if isinstance(current, EditOutcome):
            return state.remember(key, current)
        raw, revision = current

        validation = _validate(edits, state, raw, relative, revision)
        if validation.refusals:
            return state.remember(key, _rejected(tool, path=relative, refusals=validation.refusals))
        return state.remember(key, _publish(tool, root, relative, raw, revision, validation,
                                            read_bytes=read_bytes, write_bytes=write_bytes,
                                            journal=journal))
    except BatchInputError as error:
        return state.remember(key, _rejected(tool, detail=f"{tool} REJECTED: {error}"))
    except EditPlanError as error:
        return state.remember(key, _rejected(tool, detail=str(error)))


def execute_rewrite_source(root: Path, args: Mapping[str, object], *, state: EditProtocolRunState,
                           read_bytes: ReadBytes, journal: object = None,
                           not_editable: NotEditable | None = None,
                           public_brief: object = None,
                           write_bytes: WriteBytes | None = None) -> EditOutcome:
    """Replace the whole region of one REWRITE handle; destructive-guarded, one publish."""
    tool = REWRITE_SOURCE_TOOL
    key = state.idempotent_key(tool, args)
    stored = state.outcomes.get(key)
    if stored is not None:
        return stored.replayed()
    handle = args.get("handle") if isinstance(args, Mapping) else None
    new_text = args.get("new_text") if isinstance(args, Mapping) else None
    if not isinstance(handle, str) or not isinstance(new_text, str):
        return state.remember(key, _rejected(
            tool, detail=f"{tool} REJECTED: handle and new_text must be strings"))

    record = state.lookup(handle)
    if record is None:
        return state.remember(key, _rejected(tool, refusals=(
            Refusal.build(RefusalCode.STALE_HANDLE, edit_index=0, path=UNRESOLVED_PATH,
                          old_text=""),)))
    relative = record.path
    if record.kind is not SpanKind.REWRITE:
        return state.remember(key, _rejected(
            tool, path=relative, detail=f"{tool} requires a REWRITE handle", refusals=(
                Refusal.build(RefusalCode.NOT_EDITABLE, edit_index=0, path=relative,
                              old_text=""),)))
    reason = not_editable(relative) if not_editable is not None else None
    if reason:
        return state.remember(key, _rejected(tool, path=relative, detail=reason, refusals=(
            Refusal.build(RefusalCode.NOT_EDITABLE, edit_index=0, path=relative, old_text=""),)))

    try:
        current = _read_current(root, relative, read_bytes, [{"old_text": ""}], tool)
        if isinstance(current, EditOutcome):
            return state.remember(key, current)
        raw, revision = current
        if revision != record.revision:
            # The handle's bytes are gone; record.text() would raise.
            return state.remember(key, _rejected(tool, path=relative, refusals=(
                Refusal.build(RefusalCode.SOURCE_CHANGED, edit_index=0, path=relative,
                              old_text=""),)))
        text = raw.decode("utf-8", errors="surrogateescape")
        old_text = record.text(text)
        edits: list[object] = [{"handle": handle, "old_text": old_text, "new_text": new_text}]
        validation = _validate(edits, state, raw, relative, revision)
        if validation.refusals:
            return state.remember(key, _rejected(tool, path=relative, refusals=validation.refusals))

        # Destructive guard before any authority call. The shrink guard's own
        # switch decides whether it fires, exactly as for the legacy writers.
        after = raw[:record.span.byte_start] + new_text.encode("utf-8") + raw[record.span.byte_end:]
        admission = edit_admission(raw, after)
        shrink = guard_write(root / relative, after.decode("utf-8", errors="surrogateescape"),
                             root=root, public_brief=public_brief)
        if shrink is not None:
            return state.remember(key, _rejected(tool, path=relative, detail=shrink))

        outcome = _publish(tool, root, relative, raw, revision, validation,
                           read_bytes=read_bytes, write_bytes=write_bytes, journal=journal)
        if outcome.status == APPLIED:
            outcome = EditOutcome(outcome.tool, outcome.status, outcome.path, outcome.refusals,
                                  outcome.changed_paths, outcome.generation_id,
                                  f"admission={admission}")
        return state.remember(key, outcome)
    except BatchInputError as error:
        return state.remember(key, _rejected(tool, path=relative, detail=f"{tool} REJECTED: {error}"))
    except (EditPlanError, ValueError) as error:
        return state.remember(key, _rejected(tool, path=relative, detail=str(error)))


EXECUTORS: dict[str, Callable[..., EditOutcome]] = {
    EDIT_SOURCE_TOOL: execute_edit_source,
    REWRITE_SOURCE_TOOL: execute_rewrite_source,
}


__all__ = [
    "APPLIED", "AUTHORITY_REF", "EXECUTORS", "REJECTED", "REPLAYED", "SOURCE_RECEIPT",
    "EditOutcome", "EditProtocolRunState", "covered_spans", "default_write_bytes",
    "execute_edit_source", "execute_rewrite_source", "python_parse_check",
]
