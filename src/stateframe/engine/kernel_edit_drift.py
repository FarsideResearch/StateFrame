"""COLLAB-DRIFT: an edit whose file moved since delivery is merged, not refused.

Switches (off|on, default off, anything else raises):
  STATEFRAME_COLLAB_DRIFT  when on, an edit delivered against older bytes is re-located or three-way merged
                           onto the current file (kernel_edit_merge). Off keeps today's SOURCE_CHANGED refusal.
  STATEFRAME_STRICT_DRIFT  when on, forces today's refusal even if COLLAB_DRIFT is on.

Pure: no I/O. The executors in kernel_edit_source_tool own reads, guards and the single publish; this module decides
the merged text, the typed refusals and the receipt fields.

Ambiguity is never merged. A base span that occurs more than once, or a base that was not retained, stays a
SOURCE_CHANGED refusal, exactly as with the switch off.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_edit_handles import HandleRecord
from kernel_edit_merge import (
    MergeAmbiguous, MergeApplied, MergeConflict, ConflictHunk, merge_span, merge_texts)
from kernel_edit_refusal_codes import Refusal, RefusalCode
from kernel_edit_source_batch import (
    MAX_EDITS, NODE_KIND, UNRESOLVED_PATH, BatchEdit, CoveredSpan,
    validate_edit_source_batch)
from kernel_revision_bound_edit import EditOperation, EditPlan, bind_site

COLLAB_DRIFT_ENV = "STATEFRAME_COLLAB_DRIFT"
STRICT_DRIFT_ENV = "STATEFRAME_STRICT_DRIFT"
# Largest side of a conflict hunk shown to the model, in lines. Longer hunks are cut and flagged.
MAX_CONFLICT_LINES = 60


def _switch(name: str, environ: Mapping[str, str] | None) -> bool:
    source = os.environ if environ is None else environ
    value = str(source.get(name, "")).strip().lower() or "off"  # empty means unset, as for the protocol switch
    if value not in {"on", "off"}:
        raise ValueError(f"{name} must be on or off, got {value!r}")
    return value == "on"


def collab_drift_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _switch(COLLAB_DRIFT_ENV, environ)


def strict_drift_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _switch(STRICT_DRIFT_ENV, environ)


def drift_merge_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """True when stale handles are merged: COLLAB on and STRICT off."""
    return collab_drift_enabled(environ) and not strict_drift_enabled(environ)


@dataclass(frozen=True)
class DriftResult:
    """Merged file text, or the typed refusals (and conflict details) that stop it."""

    text: str | None
    refusals: tuple[Refusal, ...]
    conflicts: tuple[Mapping[str, object], ...]
    modes: tuple[str, ...]
    base_revisions: tuple[str, ...]
    hunks_digest: str

    @property
    def ok(self) -> bool:
        return self.text is not None and not self.refusals


def _lines_text(lines: list[str] | str, limit: int = MAX_CONFLICT_LINES) -> tuple[str, bool]:
    text = "".join(lines) if isinstance(lines, list) else lines
    parts = text.splitlines(keepends=True)
    if len(parts) <= limit:
        return text, False
    return "".join(parts[:limit]), True


def conflict_payload(hunk: ConflictHunk) -> Mapping[str, object]:
    """Bounded, JSON-ready view of one conflict hunk. Line ranges are 1-based inclusive."""
    def span(lo: int, hi: int) -> list[int]:
        return [lo + 1, hi] if hi > lo else [lo + 1, lo]

    base, base_cut = _lines_text(hunk.base_text)
    ours, ours_cut = _lines_text(hunk.ours_text)
    theirs, theirs_cut = _lines_text(hunk.theirs_text)
    return {
        "base_lines": span(*hunk.base_lines),
        "ours_lines": span(*hunk.ours_lines),
        "theirs_lines": span(*hunk.theirs_lines),
        "base_text": base,
        "ours_text": ours,
        "theirs_text": theirs,
        "truncated": base_cut or ours_cut or theirs_cut,
    }


def _digest(parts: Sequence[object]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def _refuse(code: RefusalCode, index: int, path: str, old_text: str) -> Refusal:
    return Refusal.build(code, edit_index=index, path=path, old_text=old_text)


def resolve_edit_batch(records: Sequence[HandleRecord | None], edits: Sequence[BatchEdit],
                       current_text: str, *, path: str, authority_ref: str) -> DriftResult:
    """Apply each edit against the bytes its handle was shown, onto the current text, in order.

    Each edit is validated against its own delivered base with the production validator (old_text must occur
    exactly once there). Its location in the current text is then proven by merge_span: unique base bytes
    re-anchor; otherwise a line three-way merge applies non-overlapping changes and reports overlapping ones as
    EDIT_CONFLICT. Edits merge sequentially, so a second edit sees the first one's result.
    """
    refusals: list[Refusal] = []
    conflicts: list[Mapping[str, object]] = []
    modes: list[str] = []
    bases: list[str] = []
    applied_hunks: list[object] = []
    if len(edits) > MAX_EDITS:
        refusals.extend(_refuse(RefusalCode.TOO_MANY_EDITS, index, path, edit.old_text)
                        for index, edit in enumerate(edits) if index >= MAX_EDITS)
    cur = current_text
    for index, edit in enumerate(edits[:MAX_EDITS]):
        record = records[index] if index < len(records) else None
        if record is None:
            refusals.append(_refuse(RefusalCode.STALE_HANDLE, index, UNRESOLVED_PATH, edit.old_text))
            continue
        if record.path != path:
            refusals.append(_refuse(RefusalCode.MIXED_FILE_BATCH, index, record.path, edit.old_text))
            continue
        if record.base_text is None:
            refusals.append(_refuse(RefusalCode.SOURCE_CHANGED, index, path, edit.old_text))
            continue
        base = record.base_text
        base_raw = base.encode("utf-8", errors="surrogateescape")
        check = validate_edit_source_batch(
            [edit], [CoveredSpan(handle=record.handle, path=path, revision=record.revision,
                                 start=0, end=len(base_raw))],
            base_raw, current_revision=record.revision, authority_ref=authority_ref,
            inventory_ref="edit-protocol:drift", source_receipt="edit-protocol:drift")
        if check.refusals:
            refusals.extend(_refuse(item.code, index, path, edit.old_text) for item in check.refusals)
            continue
        offset = base.find(edit.old_text)
        ours = base[:offset] + edit.new_text + base[offset + len(edit.old_text):]
        # merge_span counts bytes with surrogatepass; use the same encoding for the base range.
        outcome = merge_span(base, (0, len(base.encode("utf-8", errors="surrogatepass"))), ours, cur)
        bases.append(record.revision)
        if isinstance(outcome, MergeApplied):
            cur = outcome.text
            modes.append(outcome.mode)
            applied_hunks.append([[hunk.side, *hunk.base_lines] for hunk in outcome.hunks])
        elif isinstance(outcome, MergeConflict):
            conflicts.extend(conflict_payload(hunk) for hunk in outcome.hunks)
            refusals.append(_refuse(RefusalCode.EDIT_CONFLICT, index, path, edit.old_text))
        else:
            assert isinstance(outcome, MergeAmbiguous)
            refusals.append(_refuse(RefusalCode.SOURCE_CHANGED, index, path, edit.old_text))
    if refusals:
        return DriftResult(None, tuple(refusals), tuple(conflicts), tuple(modes), tuple(bases), "")
    return DriftResult(cur, (), (), tuple(modes), tuple(bases), _digest(applied_hunks))


def resolve_rewrite(record: HandleRecord, new_text: str, current_text: str, *, path: str) -> DriftResult:
    """Merge a whole-file rewrite (base = the file as delivered) onto the current text."""
    if record.base_text is None:
        return DriftResult(None, (_refuse(RefusalCode.SOURCE_CHANGED, 0, path, ""),), (), (),
                           (record.revision,), "")
    outcome = merge_texts(record.base_text, new_text, current_text)
    if isinstance(outcome, MergeApplied):
        return DriftResult(outcome.text, (), (), (outcome.mode,), (record.revision,),
                           _digest([[hunk.side, *hunk.base_lines] for hunk in outcome.hunks]))
    if isinstance(outcome, MergeConflict):
        conflicts = tuple(conflict_payload(hunk) for hunk in outcome.hunks)
        return DriftResult(None, (_refuse(RefusalCode.EDIT_CONFLICT, 0, path, ""),), conflicts, (),
                           (record.revision,), "")
    return DriftResult(None, (_refuse(RefusalCode.SOURCE_CHANGED, 0, path, ""),), (), (),
                       (record.revision,), "")


def plan_replacing(raw: bytes, final_text: str, *, path: str, revision: str, ordinal: int,
                   authority_ref: str, source_receipt: str) -> EditPlan | None:
    """One replace operation covering exactly the bytes that differ between raw and final_text.

    Returns None when there is nothing to replace or the file is empty (no span to bind). Pure insertions widen
    by one character so the bound span is never empty.
    """
    current = raw.decode("utf-8", errors="surrogateescape")
    if current == final_text:
        return None
    if not current:
        return None
    limit = min(len(current), len(final_text))
    pre = 0
    while pre < limit and current[pre] == final_text[pre]:
        pre += 1
    suf = 0
    while suf < limit - pre and current[len(current) - 1 - suf] == final_text[len(final_text) - 1 - suf]:
        suf += 1
    start, end = pre, len(current) - suf
    new = final_text[pre:len(final_text) - suf]
    if start == end:
        if start > 0:
            start -= 1
            new = current[start] + new
        else:
            end = 1
            new = new + current[0]
    start_b = len(current[:start].encode("utf-8", errors="surrogateescape"))
    end_b = start_b + len(current[start:end].encode("utf-8", errors="surrogateescape"))
    site = bind_site(path=path, file_revision=revision, source=raw, start=start_b, end=end_b,
                     node_kind=NODE_KIND, source_receipt=source_receipt,
                     permitted_operations=("replace",), authority_ref=authority_ref)
    return EditPlan(base_revision=revision, authority_ref=authority_ref,
                    inventory_ref=f"edit-protocol:q{ordinal}",
                    edits=(EditOperation("replace", site_ref=site, new_text=new, order=0),))


def receipt_detail(result: DriftResult, current_revision: str) -> str:
    """Receipt fields for an applied merge: mode, base and current revisions, hunk digest."""
    modes = "+".join(sorted(set(result.modes))) or "none"
    bases = ",".join(sorted({rev[7:19] for rev in result.base_revisions}))
    return (f"drift={modes} base={bases} current={current_revision[7:19]} "
            f"hunks={result.hunks_digest[7:19] if result.hunks_digest else 'none'}")


__all__ = [
    "COLLAB_DRIFT_ENV", "STRICT_DRIFT_ENV", "DriftResult",
    "collab_drift_enabled", "conflict_payload", "drift_merge_enabled", "plan_replacing",
    "receipt_detail", "resolve_edit_batch", "resolve_rewrite", "strict_drift_enabled",
]
