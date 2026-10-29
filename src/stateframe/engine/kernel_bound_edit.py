"""Optional exact replacement targets for the structured repair repair arm.

The switch is deliberately independent from the repair packet and edit
affordance switches. That keeps the existing executor available as an A/B
control while allowing a bound-target arm to reject stale or ambiguous
replacements before they reach the writer.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping


MODE_ENV = "STATEFRAME_BOUND_EDIT"
SCHEMA = "stateframe-bound-edit/1"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _digest(source: str) -> str:
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()


def bind_target(*, path: str, symbol: str | None, revision: object,
                source: str, start: int | None = None, end: int | None = None,
                boundary: str | None = None, node_kind: str = "legacy-source",
                source_receipt: str = "legacy-source", permitted_operations: tuple[str, ...] = ("replace",),
                authority_ref: str = "legacy-authority") -> dict[str, object]:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("bound target requires a path")
    if not isinstance(source, str):
        raise TypeError("bound target source must be text")
    if start is not None or end is not None or boundary is not None:
        from kernel_revision_bound_edit import bind_site
        finish = start if end is None else end
        return bind_site(
            path=path, file_revision=revision, source=source, start=0 if start is None else start,
            end=finish, boundary=boundary, node_kind=node_kind,
            source_receipt=source_receipt, permitted_operations=permitted_operations,
            authority_ref=authority_ref,
        ).to_dict()
    return {
        "schema": SCHEMA,
        "path": path.replace("\\", "/").lstrip("./"),
        "symbol": str(symbol or ""),
        "revision": str(revision),
        "source_digest": _digest(source),
        "old_bytes": source,
    }


def validate_target(target: object, *, path: str, revision: object,
                    source: str) -> str | None:
    """Return a typed refusal, or ``None`` when the exact target is current."""
    if not enabled():
        return None
    if not isinstance(target, Mapping):
        return "bound edit target required: supply target_ref from the repair packet"
    if target.get("schema") == "revision-bound-edit/1":
        from kernel_revision_bound_edit import SiteReference
        try:
            ref = SiteReference.from_mapping(target)
        except ValueError as error:
            return f"bound edit target rejected: {error}"
        normalized = str(path).replace("\\", "/").lstrip("./")
        if ref.path != normalized:
            return f"bound edit target rejected: path changed from {ref.path} to {normalized}"
        if ref.file_revision != str(revision) or ref.source_digest != _digest(source):
            return "bound edit target rejected: stale file revision and source bytes"
        raw = source.encode("utf-8")
        if ref.end > len(raw) or ref.start < 0 or ref.end < ref.start:
            return "bound edit target rejected: span is outside current source"
        preimage = (raw[ref.start:ref.end] if ref.boundary is None else
                    raw[max(0, ref.start - 32):min(len(raw), ref.start + 32)])
        if "sha256:" + hashlib.sha256(preimage).hexdigest() != ref.preimage_digest:
            return "bound edit target rejected: stale span preimage"
        return None
    if target.get("schema") != SCHEMA:
        return "bound edit target rejected: wrong target schema"
    normalized = str(path).replace("\\", "/").lstrip("./")
    if target.get("path") != normalized:
        return f"bound edit target rejected: path changed from {target.get('path')} to {normalized}"
    if str(target.get("revision")) != str(revision):
        return "bound edit target rejected: stale revision; rebind the current source"
    if target.get("source_digest") != _digest(source):
        return "bound edit target rejected: stale source bytes; rebind the current source"
    if target.get("old_bytes") != source:
        return "bound edit target rejected: old bytes do not match the current source"
    return None


def target_property() -> dict[str, object]:
    legacy = {
        "type": "object",
        "description": "Exact host-bound target from the current repair packet.",
        "properties": {
            "schema": {"const": SCHEMA},
            "path": {"type": "string", "maxLength": 240},
            "symbol": {"type": "string", "maxLength": 160},
            "revision": {"type": "string", "maxLength": 160},
            "source_digest": {"type": "string", "maxLength": 160},
            # This is an exact host-bound snapshot, not a frame excerpt.  Keep the
            # schema able to carry a paged file's full validation source.
            "old_bytes": {"type": "string", "maxLength": 4_000_000},
        },
        "required": ["schema", "path", "symbol", "revision", "source_digest", "old_bytes"],
        "additionalProperties": False,
    }
    from kernel_revision_bound_edit import site_ref_property
    typed = site_ref_property()
    # Keep legacy target payloads valid while making the typed site reference
    # the advertised batch/edit contract.  Both are objects, never scalars.
    return {"oneOf": [typed, legacy]}


def describe() -> str:
    return ("Bound-target arm is ON: edit and str_replace require target_ref with the "
            "current path, revision, digest, and exact old bytes.") if enabled() else ""


def bind_site(**kwargs: object) -> dict[str, object]:
    from kernel_revision_bound_edit import bind_site as _bind_site
    return _bind_site(**kwargs).to_dict()


def validate_site(target: object, *, path: str, source: str, revision: object) -> str | None:
    return validate_target(target, path=path, revision=revision, source=source)


__all__ = ["MODE_ENV", "SCHEMA", "bind_site", "bind_target", "describe", "enabled", "mode",
           "target_property", "validate_site", "validate_target"]
