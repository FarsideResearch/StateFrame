"""Bounded current-file context for rejected exact str_replace operations."""
from __future__ import annotations

import os
from pathlib import Path


FLAG = "STATEFRAME_EDIT_REJECTION_FRESH_TEXT"
_MARKER = "[Fresh current-file context:"


def enabled() -> bool:
    value = os.environ.get(FLAG, "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{FLAG} must be on or off")
    return value == "on"


def annotate_rejection(observation: str, *, root: str | Path,
                       path: str | Path, old_text: str) -> str:
    """Attach exact bounded bytes from the current revision to a no-match result."""
    if (not enabled() or not isinstance(observation, str)
            or "old_str not found" not in observation
            or not isinstance(old_text, str) or not old_text
            or _MARKER in observation):
        return observation

    root_path = Path(root).resolve()
    candidate = Path(path)
    candidate = (candidate if candidate.is_absolute() else root_path / candidate).resolve()
    try:
        relative_path = candidate.relative_to(root_path).as_posix()
        source = candidate.read_bytes()
    except (OSError, ValueError) as error:
        return observation.rstrip() + (
            "\n\n[Fresh current-file context unavailable: "
            f"{type(error).__name__}; source was not changed.]"
        )

    from kernel_edit_anchor_rebase import rejection_payload

    payload = rejection_payload(
        path=relative_path,
        source=source,
        old_text=old_text,
        code="OLD_TEXT_MISMATCH",
        context_radius=12,
    )
    suggestion = payload.get("anchor_suggestion")
    anchors = suggestion.get("anchors") if isinstance(suggestion, dict) else None
    anchor = anchors[0] if isinstance(anchors, list) and anchors else {}
    if not isinstance(anchor, dict):
        anchor = {}

    line_range = anchor.get("line_range")
    if (isinstance(line_range, list) and len(line_range) == 2
            and all(isinstance(value, int) for value in line_range)):
        line_label = f"{line_range[0]}-{line_range[1]}"
    else:
        line_label = "unknown"
    revision = str(payload.get("current_revision") or "unavailable")
    basis = suggestion.get("selection_basis") if isinstance(suggestion, dict) else None
    spans = payload.get("closest_current_spans")
    first_span = spans[0] if isinstance(spans, list) and spans else {}
    target_range = first_span.get("line_range") if isinstance(first_span, dict) else None
    if (basis != "module_head_fallback" and isinstance(target_range, list)
            and len(target_range) == 2
            and all(isinstance(value, int) for value in target_range)):
        target_label = f"{target_range[0]}-{target_range[1]}"
    else:
        target_label = "none"
    if basis == "module_head_fallback":
        context_label = "No close line match found; showing bounded file-head context."
    elif basis == "closest_line_similarity_context":
        context_label = (
            "Closest generic line-level match by similarity; retry only with an exact current anchor."
        )
    else:
        context_label = "Current exact source context; retry only with an exact current anchor."

    text = anchor.get("old_str")
    if isinstance(text, str):
        excerpt = text
    else:
        encoded = anchor.get("bytes_base64")
        if not isinstance(encoded, str):
            excerpt = "[Current source excerpt unavailable within the bounded payload.]"
        else:
            excerpt = "[Exact current source bytes, base64]\n" + encoded

    note = (
        f"{_MARKER} path={relative_path}; revision={revision}; "
        f"context_lines={line_label}; match_lines={target_label}]\n"
        f"{context_label}\n{excerpt}"
    )
    return observation.rstrip() + "\n\n" + note


__all__ = ["FLAG", "annotate_rejection", "enabled"]
