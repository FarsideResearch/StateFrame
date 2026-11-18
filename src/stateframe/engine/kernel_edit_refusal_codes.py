"""Closed catalog of edit refusal codes with handle-free public text and stable fingerprints.

Pure module: no I/O, no environment reads, no engine imports. It only names the twelve
reasons an edit proposal can be refused and gives each a fixed public sentence. The
legacy prose observations ("str_replace REJECTED: old_str not found", "matches N
places", "would not parse afterwards", ...) keep their bytes; `classify_legacy_refusal`
maps that prose to a code so later units can report the same refusal in typed form.

A refusal fingerprint is sha256(code|path_digest|old_text). It never includes the
request-scoped handle, so re-minting handles between requests (q42h1 -> q43h1) leaves
the fingerprint unchanged and an identical retry is recognisable as identical.
"""
from __future__ import annotations

import enum
import hashlib
import re
from dataclasses import dataclass


class RefusalCode(str, enum.Enum):
    """The closed set of edit refusal reasons."""

    STALE_HANDLE = "STALE_HANDLE"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    UNCOVERED_SOURCE = "UNCOVERED_SOURCE"
    OLD_NOT_FOUND = "OLD_NOT_FOUND"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    DISPLAY_MARKER = "DISPLAY_MARKER"
    OVERLAPPING_EDITS = "OVERLAPPING_EDITS"
    MIXED_FILE_BATCH = "MIXED_FILE_BATCH"
    EMPTY_OLD_TEXT = "EMPTY_OLD_TEXT"
    TOO_MANY_EDITS = "TOO_MANY_EDITS"
    PARSE_BREAK = "PARSE_BREAK"
    NOT_EDITABLE = "NOT_EDITABLE"
    EDIT_CONFLICT = "EDIT_CONFLICT"


REFUSAL_CODES: tuple[RefusalCode, ...] = tuple(RefusalCode)
CODE_COUNT = 13

# Fixed public sentences. They describe the defect and the repair without naming any
# handle, path, or request ordinal; those belong to the surrounding request, not here.
PUBLIC_TEXT: dict[RefusalCode, str] = {
    RefusalCode.STALE_HANDLE: (
        "The handle belongs to an earlier request and is no longer valid. Use a handle "
        "from the current source map."),
    RefusalCode.SOURCE_CHANGED: (
        "The file changed after this source was delivered. Use the current source map "
        "and re-issue the edit against it."),
    RefusalCode.UNCOVERED_SOURCE: (
        "old_text lies outside the span this handle covers. Pick the handle whose span "
        "contains the text, or edit only inside the covered span."),
    RefusalCode.OLD_NOT_FOUND: (
        "old_text does not occur in the covered span. Copy the exact current text, "
        "including whitespace."),
    RefusalCode.AMBIGUOUS_MATCH: (
        "old_text occurs more than once in the covered span. Include more surrounding "
        "lines so it matches exactly once."),
    RefusalCode.DISPLAY_MARKER: (
        "old_text contains display marker text that is not file content. Anchor on real "
        "source lines only."),
    RefusalCode.OVERLAPPING_EDITS: (
        "Two edits in the batch touch overlapping byte ranges. Merge them into one edit "
        "or separate their anchors."),
    RefusalCode.MIXED_FILE_BATCH: (
        "A batch may edit one file only. Split edits for other files into separate calls."),
    RefusalCode.EMPTY_OLD_TEXT: (
        "old_text is required and must be non-empty. Copy the exact current text to "
        "replace."),
    RefusalCode.TOO_MANY_EDITS: (
        "A batch holds at most four edits. Send the remaining edits in a later call."),
    RefusalCode.PARSE_BREAK: (
        "The file would not parse after this edit. Nothing was changed; fix the "
        "replacement text so the file stays valid."),
    RefusalCode.NOT_EDITABLE: (
        "This target is not editable: protected path, directory, or outside the "
        "workspace. Nothing was changed."),
    RefusalCode.EDIT_CONFLICT: (
        "The file and this edit both changed the same lines since the source was "
        "delivered, so neither version was applied. Nothing was changed. The conflict "
        "details name the lines and give the base, your edit, and the current text; "
        "re-issue the edit against the current text."),
}

# Any spelling of a request-scoped handle. Public text and fingerprints never carry one.
HANDLE_PATTERN = re.compile(r"\bq\d+h\d+\b")


def public_text(code: RefusalCode) -> str:
    """Fixed public sentence for a code; raises on anything outside the catalog."""
    if not isinstance(code, RefusalCode):
        raise ValueError(f"unknown refusal code: {code!r}")
    return PUBLIC_TEXT[code]


def path_digest(path: str) -> str:
    """Stable digest of a workspace-relative path so fingerprints never embed the path."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


def fingerprint(code: RefusalCode, path: str, old_text: str) -> str:
    """sha256(code|path_digest|old_text); independent of handles and request ordinals."""
    if not isinstance(code, RefusalCode):
        raise ValueError(f"unknown refusal code: {code!r}")
    if not isinstance(old_text, str):
        raise ValueError("old_text must be a string")
    material = f"{code.value}|{path_digest(path)}|".encode("utf-8") + old_text.encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class Refusal:
    """One refused edit: code, which edit in the batch, its fingerprint, and public text."""

    code: RefusalCode
    edit_index: int
    fingerprint: str
    public_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, RefusalCode):
            raise ValueError(f"unknown refusal code: {self.code!r}")
        if type(self.edit_index) is not int or self.edit_index < 0:
            raise ValueError("edit_index must be a non-negative integer")
        if not re.fullmatch(r"[0-9a-f]{64}", self.fingerprint or ""):
            raise ValueError("fingerprint must be a sha256 hex digest")
        if HANDLE_PATTERN.search(self.public_text):
            raise ValueError("public_text must not spell a handle")

    @classmethod
    def build(cls, code: RefusalCode, *, edit_index: int, path: str,
              old_text: str) -> "Refusal":
        return cls(
            code=code,
            edit_index=edit_index,
            fingerprint=fingerprint(code, path, old_text),
            public_text=public_text(code),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "edit_index": self.edit_index,
            "fingerprint": self.fingerprint,
            "public_text": self.public_text,
        }


# Legacy prose -> code. Order matters: the first matching needle wins. Needles are the
# exact fragments the current executors emit (see str_replace_edit and the revision
# bound EditAuthority diagnostics); they are matched case-sensitively on purpose so a
# wording change surfaces as a failing test rather than a silent reclassification.
_LEGACY_NEEDLES: tuple[tuple[str, RefusalCode], ...] = (
    ("old_str is required and must be non-empty", RefusalCode.EMPTY_OLD_TEXT),
    ("the marker is not file content", RefusalCode.DISPLAY_MARKER),
    ("would not parse afterwards", RefusalCode.PARSE_BREAK),
    ("syntax error at line", RefusalCode.PARSE_BREAK),
    ("old_str not found", RefusalCode.OLD_NOT_FOUND),
    ("OLD_TEXT_MISMATCH", RefusalCode.OLD_NOT_FOUND),
    ("protected public test paths are immutable", RefusalCode.NOT_EDITABLE),
    ("is a directory, not a file", RefusalCode.NOT_EDITABLE),
    ("STALE_SOURCE", RefusalCode.SOURCE_CHANGED),
    ("STALE_REVISION", RefusalCode.SOURCE_CHANGED),
    ("STALE_SPAN", RefusalCode.UNCOVERED_SOURCE),
    ("AMBIGUOUS", RefusalCode.AMBIGUOUS_MATCH),
)
_LEGACY_MATCHES_N_PLACES = re.compile(r"old_str matches \d+ places")


def classify_legacy_refusal(observation: str) -> RefusalCode | None:
    """Map one of today's prose refusals to its catalog code, or None when it is not a
    refusal this catalog describes. Reads the prose; never rewrites it."""
    if not isinstance(observation, str) or not observation:
        return None
    for needle, code in _LEGACY_NEEDLES:
        if needle in observation:
            return code
    if _LEGACY_MATCHES_N_PLACES.search(observation):
        return RefusalCode.AMBIGUOUS_MATCH
    return None
