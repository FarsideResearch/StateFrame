"""Pure batch validator for the ``edit_source`` protocol.

Input: one to four ``(handle, old_text, new_text)`` edits against the spans a request
delivered, plus the current bytes of the one file those spans cover. Output: either every
defect at once as a list of typed refusals (never just the first one), or a clean
``EditPlan`` of ``replace`` operations bound to host-owned site references.

Pure: no I/O, no environment reads, no clock. Byte offsets are offsets into the file
bytes; text is matched as UTF-8 so BOM, CRLF and non-ASCII content survive untouched.
Parse checking is an injected callable so the validator stays language-neutral.

Code coverage of the catalog from this module:
  batch shape   TOO_MANY_EDITS, MIXED_FILE_BATCH
  handle        STALE_HANDLE (unknown to this request), SOURCE_CHANGED (revision or
                digest drift), NOT_EDITABLE (read-only span)
  old_text      EMPTY_OLD_TEXT, DISPLAY_MARKER, OLD_NOT_FOUND, UNCOVERED_SOURCE,
                AMBIGUOUS_MATCH
  batch bytes   OVERLAPPING_EDITS
  result        PARSE_BREAK (via ``parse_check``)
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from kernel_edit_guard import contains_source_omission
from kernel_edit_refusal_codes import Refusal, RefusalCode
from kernel_revision_bound_edit import EditOperation, EditPlan, bind_site

MAX_EDITS = 4
SPAN_KINDS = frozenset({"SPAN", "REWRITE"})
NODE_KIND = "edit_source_span"

# Prefixes our own renderers put around delivered source. They are never file content
# unless the file itself happens to contain them, which the validator checks.
RENDERER_MARKERS: tuple[str, ...] = (
    "[CURRENT SOURCE]",
    "[WORKING SOURCE]",
    "[Working source: ",
    "[Current source: ",
    "[Paged source:",
    "[Fresh current-file context:",
    "[StateFrame]",
    "CURRENT_SOURCE ",
    "Coverage: COMPLETE FILE;",
)

# Label used in fingerprints for refusals that cannot be tied to one resolved path
# (for example an unknown handle). Fixed so identical retries fingerprint identically.
UNRESOLVED_PATH = "(unresolved)"


class BatchInputError(ValueError):
    """The caller handed the validator something that is not a batch at all."""


@dataclass(frozen=True)
class CoveredSpan:
    """One span of source a request delivered under a handle.

    ``start``/``end`` are byte offsets into the file; ``revision`` is the file revision
    the span was rendered from; ``source_digest`` (optional) pins the exact bytes.
    """

    handle: str
    path: str
    revision: str
    start: int
    end: int
    kind: str = "SPAN"
    source_digest: str | None = None
    editable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.handle, str) or not self.handle:
            raise BatchInputError("span handle must be a non-empty string")
        if not isinstance(self.path, str) or not self.path:
            raise BatchInputError("span path must be a non-empty string")
        if not isinstance(self.revision, str) or not self.revision:
            raise BatchInputError("span revision must be a non-empty string")
        if type(self.start) is not int or type(self.end) is not int:
            raise BatchInputError("span offsets must be integers")
        if self.start < 0 or self.end < self.start:
            raise BatchInputError("span offsets must be non-negative and ordered")
        if self.kind not in SPAN_KINDS:
            raise BatchInputError(f"span kind must be one of {sorted(SPAN_KINDS)}")


@dataclass(frozen=True)
class BatchEdit:
    """One proposed edit: replace ``old_text`` inside the span ``handle`` with ``new_text``."""

    handle: str
    old_text: str
    new_text: str

    @classmethod
    def coerce(cls, value: object, index: int) -> "BatchEdit":
        if isinstance(value, BatchEdit):
            return value
        if isinstance(value, Mapping):
            handle, old, new = value.get("handle"), value.get("old_text"), value.get("new_text")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
            handle, old, new = value
        else:
            raise BatchInputError(f"edit {index} must be a mapping or (handle, old_text, new_text)")
        if not isinstance(handle, str) or not handle:
            raise BatchInputError(f"edit {index} handle must be a non-empty string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise BatchInputError(f"edit {index} old_text and new_text must be strings")
        return cls(handle, old, new)


@dataclass(frozen=True)
class _Match:
    """A resolved edit: its byte range in the file and the replacement bytes."""

    index: int
    start: int
    end: int
    new: bytes
    old_text: str


@dataclass(frozen=True)
class BatchValidation:
    """Every refusal found, or (when there are none) the plan to execute."""

    refusals: tuple[Refusal, ...]
    plan: EditPlan | None
    path: str | None

    @property
    def ok(self) -> bool:
        return not self.refusals and self.plan is not None

    @property
    def codes(self) -> tuple[RefusalCode, ...]:
        return tuple(refusal.code for refusal in self.refusals)

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "path": self.path,
            "refusals": [refusal.as_dict() for refusal in self.refusals],
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }


def _as_bytes(value: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise BatchInputError("source must be bytes or text")


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def display_marker_in(old_text: str, source: bytes) -> bool:
    """True when ``old_text`` carries renderer or omission marker text that is not in the file."""
    # Validates a proposed tool argument against file bytes; no request row is classified,
    # so the authorship ledger has nothing to answer here (listed in the sniff baseline).
    for marker in RENDERER_MARKERS:
        if marker in old_text and marker.encode("utf-8") not in source:
            return True
    if contains_source_omission(old_text):
        # Keep a genuine literal: the file really contains that text.
        return old_text.encode("utf-8") not in source
    return False


def compose(source: bytes, matches: Iterable[_Match]) -> bytes:
    """Apply non-overlapping byte replacements to ``source``; order-independent."""
    out = bytearray()
    cursor = 0
    for match in sorted(matches, key=lambda item: item.start):
        out += source[cursor:match.start]
        out += match.new
        cursor = match.end
    out += source[cursor:]
    return bytes(out)


def validate_edit_source_batch(
    edits: Sequence[object],
    spans: Iterable[CoveredSpan],
    source: str | bytes | bytearray | memoryview,
    *,
    current_revision: str,
    authority_ref: str,
    inventory_ref: str,
    source_receipt: str,
    parse_check: Callable[[bytes], str | None] | None = None,
) -> BatchValidation:
    """Validate a batch against the spans a request delivered and the file's current bytes.

    ``parse_check`` receives the composed file bytes and returns an error description or
    None. Every defect is reported; the plan is built only when nothing is refused.
    """
    if isinstance(edits, (str, bytes)) or not isinstance(edits, Sequence):
        raise BatchInputError("edits must be a sequence")
    if not edits:
        raise BatchInputError("a batch needs at least one edit")
    raw = _as_bytes(source)
    by_handle: dict[str, CoveredSpan] = {}
    for span in spans:
        if not isinstance(span, CoveredSpan):
            raise BatchInputError("spans must be CoveredSpan records")
        if span.handle in by_handle:
            raise BatchInputError(f"duplicate span handle in request: {span.handle}")
        by_handle[span.handle] = span

    batch = [BatchEdit.coerce(item, index) for index, item in enumerate(edits)]
    refusals: list[Refusal] = []
    matches: list[_Match] = []

    # The file the batch edits is the one the first resolvable handle names.
    primary_path: str | None = None
    for edit in batch:
        span = by_handle.get(edit.handle)
        if span is not None:
            primary_path = span.path
            break

    def refuse(code: RefusalCode, index: int, path: str | None, old_text: str) -> None:
        refusals.append(Refusal.build(code, edit_index=index,
                                      path=path or primary_path or UNRESOLVED_PATH,
                                      old_text=old_text))

    for index, edit in enumerate(batch):
        span = by_handle.get(edit.handle)
        path = span.path if span is not None else None

        if index >= MAX_EDITS:
            refuse(RefusalCode.TOO_MANY_EDITS, index, path, edit.old_text)
            # Excess edits are not resolved; the model must resend them later anyway.
            continue
        if span is None:
            refuse(RefusalCode.STALE_HANDLE, index, None, edit.old_text)
            continue
        if primary_path is not None and span.path != primary_path:
            refuse(RefusalCode.MIXED_FILE_BATCH, index, path, edit.old_text)
            continue
        if span.revision != current_revision or (
                span.source_digest is not None and span.source_digest != _digest(raw)):
            refuse(RefusalCode.SOURCE_CHANGED, index, path, edit.old_text)
            continue
        if not span.editable:
            refuse(RefusalCode.NOT_EDITABLE, index, path, edit.old_text)
            continue
        if span.end > len(raw):
            # The delivered span no longer fits the file: the file shrank underneath it.
            refuse(RefusalCode.SOURCE_CHANGED, index, path, edit.old_text)
            continue
        if edit.old_text == "":
            refuse(RefusalCode.EMPTY_OLD_TEXT, index, path, edit.old_text)
            continue
        if display_marker_in(edit.old_text, raw):
            refuse(RefusalCode.DISPLAY_MARKER, index, path, edit.old_text)
            continue

        needle = edit.old_text.encode("utf-8")
        window = raw[span.start:span.end]
        count = window.count(needle)
        if count == 0:
            code = RefusalCode.UNCOVERED_SOURCE if needle in raw else RefusalCode.OLD_NOT_FOUND
            refuse(code, index, path, edit.old_text)
            continue
        if count > 1:
            refuse(RefusalCode.AMBIGUOUS_MATCH, index, path, edit.old_text)
            continue
        offset = span.start + window.index(needle)
        matches.append(_Match(index, offset, offset + len(needle),
                              edit.new_text.encode("utf-8"), edit.old_text))

    # Overlap is a property of resolved byte ranges; adjacent ranges do not overlap.
    overlapping: set[int] = set()
    for i, left in enumerate(matches):
        for right in matches[i + 1:]:
            if left.start < right.end and right.start < left.end:
                overlapping.update((left.index, right.index))
    for match in matches:
        if match.index in overlapping:
            refuse(RefusalCode.OVERLAPPING_EDITS, match.index, primary_path, match.old_text)

    if not refusals and parse_check is not None:
        composed = compose(raw, matches)
        if parse_check(composed) is not None:
            # Attribute to the edits that break the file on their own; when only the
            # combination breaks it, every edit in the batch is reported.
            culprits = [m for m in matches if parse_check(compose(raw, (m,))) is not None]
            for match in culprits or matches:
                refuse(RefusalCode.PARSE_BREAK, match.index, primary_path, match.old_text)

    if refusals:
        refusals.sort(key=lambda item: (item.edit_index, item.code.value))
        return BatchValidation(tuple(refusals), None, primary_path)

    assert primary_path is not None  # every edit resolved to a span of this path
    operations = tuple(
        EditOperation(
            "replace",
            site_ref=bind_site(
                path=primary_path,
                file_revision=current_revision,
                source=raw,
                start=match.start,
                end=match.end,
                node_kind=NODE_KIND,
                source_receipt=source_receipt,
                permitted_operations=("replace",),
                authority_ref=authority_ref,
            ),
            new_text=match.new.decode("utf-8", errors="surrogateescape"),
            order=match.index,
        )
        for match in matches
    )
    plan = EditPlan(
        base_revision=current_revision,
        authority_ref=authority_ref,
        inventory_ref=inventory_ref,
        edits=operations,
    )
    return BatchValidation((), plan, primary_path)
