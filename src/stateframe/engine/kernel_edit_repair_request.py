"""Repair-ready refusal: turn a refused ``edit_source`` batch into the NEXT request.

Pure module: no I/O, no environment reads, no clock, no engine wiring. Given the typed
refusals of one batch, the current bytes of the file it touched, a handle minter and the
rejected proposal, ``build`` produces everything the next request needs so the model can
correct the edit without a read tool call:

  * the current source of the touched file, verbatim;
  * fresh request-scoped handles over that source (minted by the injected minter, so the
    handle module stays unimported from the engine side until the host wires it);
  * the rejected proposal, verbatim, so the model edits its own words rather than
    reconstructing them;
  * the code list with the fixed public text of each refusal.

The observation that goes back into the transcript carries only the code list and the
refusal fingerprints; the payload itself is placed in the next user message by the host.

Escalation ladder (per refusal fingerprint, which ignores handle spelling):
  1st refusal -> RETRY: correct and resend.
  2nd identical -> NARROW: resend exactly one edit.
  3rd identical -> BLOCKED: the ``[EDIT BLOCKED]`` marker is emitted for the host's
  loop-breaker path; nothing here ends the run.

Determinism: the payload is a pure function of its arguments. Handles come from
``minter(request_ordinal, path, source)``; replay and resume re-mint identically from the
same request ordinal. The ledger is plain data so it can be journaled and restored.
"""
from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from kernel_edit_refusal_codes import HANDLE_PATTERN, Refusal, RefusalCode

SCHEMA = "stateframe-edit-repair-request/1"
REFUSED_MARKER = "[EDIT REFUSED]"
BLOCKED_MARKER = "[EDIT BLOCKED]"
# Same spelling the batch validator lists as a renderer marker, so a model that copies
# this header into old_text is refused with DISPLAY_MARKER rather than silently matched.
SOURCE_HEADER_PREFIX = "[Current source: "
NARROW_AT = 2
BLOCKED_AT = 3


class EscalationLevel(str, enum.Enum):
    RETRY = "RETRY"
    NARROW = "NARROW"
    BLOCKED = "BLOCKED"


def level_for(repeat_count: int) -> EscalationLevel:
    """Ladder step for a fingerprint seen ``repeat_count`` times, this refusal included."""
    if type(repeat_count) is not int or repeat_count < 1:
        raise ValueError("repeat_count must be a positive integer")
    if repeat_count >= BLOCKED_AT:
        return EscalationLevel.BLOCKED
    if repeat_count >= NARROW_AT:
        return EscalationLevel.NARROW
    return EscalationLevel.RETRY


_INSTRUCTIONS: dict[EscalationLevel, str] = {
    EscalationLevel.RETRY: (
        "Correct the refused edits against the current source below and resend the batch "
        "using handles from the current source map. No read is needed; the full current "
        "text of the file is included here."),
    EscalationLevel.NARROW: (
        "The same refusal has now happened twice. Narrow to one edit: resend exactly one "
        "edit, the smallest change that makes progress, with old_text copied exactly from "
        "the current source below and a handle from the current source map."),
    EscalationLevel.BLOCKED: (
        "The same refusal has happened three times. This edit is blocked; do not resend "
        "it. Choose a different approach or report that the edit cannot be made."),
}


@runtime_checkable
class HandleMapLike(Protocol):
    """What ``build`` needs from a minted handle map (``kernel_edit_handles.HandleMap``)."""

    request_ordinal: int
    records: Sequence[object]

    def render(self) -> str: ...

    def as_dict(self) -> dict[str, object]: ...


Minter = Callable[[int, str, str], HandleMapLike]
"""``minter(request_ordinal, path, source_text) -> HandleMapLike`` over the whole file."""


def source_revision(source: str) -> str:
    """Same digest spelling as the paged-source descriptors: ``sha256:`` + hex of UTF-8."""
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def _check_ordinal(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("request_ordinal must be a non-negative integer")


def _source_text(source: object) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, (bytes, bytearray, memoryview)):
        try:
            return bytes(source).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("current source must be valid UTF-8") from error
    raise ValueError("current source must be text or bytes")


@dataclass(frozen=True)
class ProposedEdit:
    """One edit of the rejected proposal, kept verbatim (no stripping, no normalising)."""

    handle: str
    old_text: str
    new_text: str

    @classmethod
    def coerce(cls, value: object, index: int) -> "ProposedEdit":
        if isinstance(value, ProposedEdit):
            return value
        if isinstance(value, Mapping):
            handle, old, new = value.get("handle"), value.get("old_text"), value.get("new_text")
        elif (isinstance(value, Sequence) and not isinstance(value, (str, bytes))
              and len(value) == 3):
            handle, old, new = value
        elif all(hasattr(value, name) for name in ("handle", "old_text", "new_text")):
            handle, old, new = value.handle, value.old_text, value.new_text  # type: ignore[attr-defined]
        else:
            raise ValueError(f"proposal edit {index} must be a mapping or (handle, old_text, new_text)")
        if not isinstance(handle, str) or not isinstance(old, str) or not isinstance(new, str):
            raise ValueError(f"proposal edit {index} fields must be strings")
        return cls(handle, old, new)

    def as_dict(self) -> dict[str, str]:
        return {"handle": self.handle, "old_text": self.old_text, "new_text": self.new_text}


@dataclass(frozen=True)
class FingerprintLedger:
    """How many times each refusal fingerprint has been refused so far. Plain data."""

    counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for item in self.counts:
            if (not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
                    or type(item[1]) is not int or item[1] < 1):
                raise ValueError("ledger entries must be (fingerprint, positive count)")
            if item[0] in seen:
                raise ValueError("ledger fingerprints must be unique")
            seen.add(item[0])
        if tuple(sorted(self.counts)) != self.counts:
            raise ValueError("ledger entries must be sorted")

    @classmethod
    def from_counts(cls, counts: Mapping[str, int]) -> "FingerprintLedger":
        return cls(tuple(sorted((str(k), int(v)) for k, v in counts.items())))

    @classmethod
    def from_dict(cls, data: object) -> "FingerprintLedger":
        if not isinstance(data, Mapping) or not isinstance(data.get("counts"), Mapping):
            raise ValueError("ledger data needs a counts mapping")
        return cls.from_counts(data["counts"])

    def count(self, fingerprint: str) -> int:
        return dict(self.counts).get(fingerprint, 0)

    def record(self, fingerprints: Sequence[str]) -> "FingerprintLedger":
        """Ledger after one refusal carrying ``fingerprints`` (each counted once per refusal)."""
        counts = dict(self.counts)
        for fingerprint in sorted(set(fingerprints)):
            counts[fingerprint] = counts.get(fingerprint, 0) + 1
        return FingerprintLedger.from_counts(counts)

    def as_dict(self) -> dict[str, object]:
        return {"counts": dict(self.counts)}


@dataclass(frozen=True)
class RepairRequest:
    """The next-request payload for one refused batch."""

    request_ordinal: int
    path: str
    current_source: str
    current_revision: str
    refusals: tuple[Refusal, ...]
    handle_map: HandleMapLike | None
    rejected_proposal: tuple[ProposedEdit, ...]
    level: EscalationLevel
    repeat_count: int
    ledger: FingerprintLedger
    schema: str = field(default=SCHEMA)

    @property
    def codes(self) -> tuple[RefusalCode, ...]:
        return tuple(refusal.code for refusal in self.refusals)

    @property
    def fingerprints(self) -> tuple[str, ...]:
        return tuple(refusal.fingerprint for refusal in self.refusals)

    @property
    def blocked(self) -> bool:
        return self.level is EscalationLevel.BLOCKED

    @property
    def handles(self) -> tuple[str, ...]:
        if self.handle_map is None:
            return ()
        return tuple(str(getattr(record, "handle")) for record in self.handle_map.records)

    def instruction(self) -> str:
        return _INSTRUCTIONS[self.level]

    def observation(self) -> str:
        """Transcript observation: code list + fingerprints only. No handles, no bytes."""
        count = len(self.refusals)
        codes = ", ".join(f"{r.code.value} (edit {r.edit_index})" for r in self.refusals)
        head = BLOCKED_MARKER if self.blocked else REFUSED_MARKER
        text = (f"{head} edit_source refused {count} edit{'s' if count != 1 else ''}; "
                f"nothing was changed. codes: {codes}. fingerprints: "
                f"{', '.join(self.fingerprints)}. ")
        if self.blocked:
            text += "This proposal is blocked after three identical refusals."
        else:
            text += f"The repair request q{self.request_ordinal} carries the current source."
        assert HANDLE_PATTERN.search(text) is None
        return text

    def render_proposal(self) -> str:
        """The rejected proposal verbatim, as JSON (field order fixed, text untouched)."""
        return json.dumps([edit.as_dict() for edit in self.rejected_proposal],
                          ensure_ascii=False, indent=2)

    def render_source(self) -> str:
        lines = self.current_source.splitlines(keepends=True)
        header = (f"{SOURCE_HEADER_PREFIX}{self.path} revision {self.current_revision} "
                  f"({len(lines)} lines, complete file)]")
        if not self.current_source:
            return header + "\nThe file is empty; there is nothing to anchor old_text on."
        body = self.current_source
        return header + "\n" + body + ("" if body.endswith("\n") else "\n")

    def render(self) -> str:
        """Text the host places in the next user message."""
        parts = [f"{BLOCKED_MARKER if self.blocked else REFUSED_MARKER} request "
                 f"q{self.request_ordinal}: {len(self.refusals)} refused edit"
                 f"{'s' if len(self.refusals) != 1 else ''} in {self.path}; nothing was changed."]
        for refusal in self.refusals:
            parts.append(f"edit {refusal.edit_index}: {refusal.code.value}: {refusal.public_text}")
        parts.append(self.instruction())
        parts.append("Rejected proposal, verbatim:\n" + self.render_proposal())
        parts.append(self.render_source().rstrip("\n"))
        if self.handle_map is not None:
            parts.append(self.handle_map.render())
        return "\n\n".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "request_ordinal": self.request_ordinal,
            "path": self.path,
            "current_revision": self.current_revision,
            "current_source": self.current_source,
            "codes": [code.value for code in self.codes],
            "refusals": [refusal.as_dict() for refusal in self.refusals],
            "handles": list(self.handles),
            "handle_map": None if self.handle_map is None else self.handle_map.as_dict(),
            "rejected_proposal": [edit.as_dict() for edit in self.rejected_proposal],
            "escalation": {"level": self.level.value, "repeat_count": self.repeat_count,
                           "blocked": self.blocked, "instruction": self.instruction()},
            "ledger": self.ledger.as_dict(),
        }

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
                                         ensure_ascii=False).encode("utf-8")).hexdigest()


def build(
    refusals: Sequence[Refusal],
    current_bytes: str | bytes | bytearray | memoryview,
    minter: Minter,
    rejected_proposal: Sequence[object],
    *,
    path: str,
    request_ordinal: int,
    ledger: FingerprintLedger | None = None,
) -> RepairRequest:
    """Build the next request for a refused batch.

    ``refusals`` are the typed refusals of the batch; ``current_bytes`` the touched file's
    current content; ``minter`` mints fresh handles for ``request_ordinal`` (the ordinal
    of the request being built); ``rejected_proposal`` is the batch as the model sent it;
    ``ledger`` is the fingerprint ledger before this refusal (None = first refusal).
    """
    if isinstance(refusals, (str, bytes)) or not isinstance(refusals, Sequence) or not refusals:
        raise ValueError("refusals must be a non-empty sequence")
    if not all(isinstance(refusal, Refusal) for refusal in refusals):
        raise ValueError("refusals must be Refusal records")
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    _check_ordinal(request_ordinal)
    if not callable(minter):
        raise ValueError("minter must be callable")
    if isinstance(rejected_proposal, (str, bytes)) or not isinstance(rejected_proposal, Sequence):
        raise ValueError("rejected_proposal must be a sequence of edits")
    proposal = tuple(ProposedEdit.coerce(item, index) for index, item in enumerate(rejected_proposal))
    if not proposal:
        raise ValueError("rejected_proposal must hold at least one edit")

    source = _source_text(current_bytes)
    ordered = tuple(sorted(refusals, key=lambda r: (r.edit_index, r.code.value)))
    before = ledger if ledger is not None else FingerprintLedger()
    after = before.record([refusal.fingerprint for refusal in ordered])
    repeat_count = max(after.count(refusal.fingerprint) for refusal in ordered)

    handle_map: HandleMapLike | None = None
    if source:
        minted = minter(request_ordinal, path, source)
        if not isinstance(minted, HandleMapLike):
            raise ValueError("minter must return a handle map (records, render, as_dict)")
        if minted.request_ordinal != request_ordinal:
            raise ValueError("minter returned handles for a different request ordinal")
        handle_map = minted

    return RepairRequest(
        request_ordinal=request_ordinal,
        path=path,
        current_source=source,
        current_revision=source_revision(source),
        refusals=ordered,
        handle_map=handle_map,
        rejected_proposal=proposal,
        level=level_for(repeat_count),
        repeat_count=repeat_count,
        ledger=after,
    )
