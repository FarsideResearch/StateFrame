"""legacy_safe str_replace through the shared batch validator and EditAuthority.

One path-addressed ``str_replace`` becomes a single-edit batch over a synthetic
whole-file span, so the legacy front end and the handle protocol share one
validator and one admission authority. The model-visible prose is today's
``str_replace`` prose, byte for byte (tests/fixtures/legacy_safe_observations.json).
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from kernel_edit_refusal_codes import RefusalCode, public_text
from kernel_edit_source_batch import CoveredSpan, validate_edit_source_batch
from kernel_revision_bound_edit import EditAuthority, EditPlanError, compose_candidate

SYNTHETIC_HANDLE = "q0h1"            # never reaches a model; whole-file span for one path-addressed edit
AUTHORITY_PREFIX = "legacy_safe:"

_NOT_FOUND = ("str_replace REJECTED: old_str not found. Copy the exact current text "
              "(check whitespace) — `read` the region first.")
_EMPTY = ("str_replace REJECTED: old_str is required and must be non-empty: copy the "
          "exact current text to replace. Nothing was changed.")


@dataclass(frozen=True)
class LegacyEditOutcome:
    observation: str | None          # exact legacy prose when refused, else None
    candidate: bytes | None          # composed file bytes when admitted, else None
    codes: tuple[RefusalCode, ...]


def python_parse_check(raw: bytes) -> str | None:
    """Parse the composed bytes as Python (a UTF-8 BOM is legal); the legacy detail or None."""
    try:
        ast.parse(raw)
    except SyntaxError as error:
        return f"{error.msg} line {error.lineno}"
    except ValueError as error:  # e.g. NUL bytes; the legacy reader refuses those earlier
        return str(error)
    return None


def legacy_prose(code: RefusalCode, *, old_text: str, source: str, parse_detail: str | None) -> str:
    """Today's str_replace sentence for a validator refusal code."""
    if code in (RefusalCode.OLD_NOT_FOUND, RefusalCode.DISPLAY_MARKER, RefusalCode.UNCOVERED_SOURCE):
        # A marker that is not in the file means old_text is not in the file either.
        return _NOT_FOUND
    if code is RefusalCode.AMBIGUOUS_MATCH:
        return (f"str_replace REJECTED: old_str matches {source.count(old_text)} places. Include more "
                f"surrounding lines so it is unique.")
    if code is RefusalCode.PARSE_BREAK:
        return (f"str_replace REJECTED: the file would not parse afterwards "
                f"({parse_detail}). Nothing was changed.")
    if code is RefusalCode.EMPTY_OLD_TEXT:
        return _EMPTY
    return f"str_replace REJECTED: {public_text(code)} Nothing was changed."


def single_edit(*, relative_path: str, source: bytes, old_text: str, new_text: str,
                root: str | Path, parse_check=python_parse_check) -> LegacyEditOutcome:
    """Validate and compose one exact replacement over the whole file.

    Raises EditPlanError when EditAuthority refuses a plan the validator accepted
    (the two disagree; never silently written).
    """
    revision = "sha256:" + hashlib.sha256(source).hexdigest()
    authority_ref = AUTHORITY_PREFIX + revision
    span = CoveredSpan(SYNTHETIC_HANDLE, relative_path, revision, 0, len(source))
    validation = validate_edit_source_batch(
        [{"handle": SYNTHETIC_HANDLE, "old_text": old_text, "new_text": new_text}], [span], source,
        current_revision=revision, authority_ref=authority_ref,
        inventory_ref="inventory:" + revision, source_receipt="receipt:" + revision,
        parse_check=parse_check)
    if validation.refusals:
        codes = tuple(refusal.code for refusal in validation.refusals)
        detail = None
        if codes[0] is RefusalCode.PARSE_BREAK:
            text = source.decode("utf-8", errors="surrogateescape")
            composed = text.replace(old_text, new_text, 1).encode("utf-8", errors="surrogateescape")
            detail = parse_check(composed)
        prose = legacy_prose(codes[0], old_text=old_text,
                             source=source.decode("utf-8", errors="surrogateescape"), parse_detail=detail)
        return LegacyEditOutcome(prose, None, codes)
    authority = EditAuthority(root=root, sources={relative_path: source}, base_revision=revision,
                              authority_ref=authority_ref)
    admission = authority.admit(validation.plan)
    if not admission.accepted:
        raise EditPlanError("ADMIT", "validator-accepted edit refused by the edit authority",
                            diagnostics=admission.diagnostics)
    return LegacyEditOutcome(None, bytes(compose_candidate(authority, validation.plan)[relative_path]), ())


__all__ = ["AUTHORITY_PREFIX", "LegacyEditOutcome", "SYNTHETIC_HANDLE", "legacy_prose",
           "python_parse_check", "single_edit"]
