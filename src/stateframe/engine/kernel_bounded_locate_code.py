"""Thin coding-pack translations for :mod:`kernel_bounded_locate`.

The kernel does not know what a ``read`` or ``str_replace`` tool looks like.
This adapter only extracts the stable path/span/evidence fields from coding
tool arguments; host verification and phase authority remain in the kernel.
"""
from __future__ import annotations

from collections.abc import Mapping

from kernel_bounded_locate import EditSiteHypothesis, EditSpan


# Source navigation is a host-registered read effect.  Completion holds may
# restrict certification or writes, but they must not turn an admissible S2
# lookup into a misleading "done refused" observation.
READ_TOOLS = frozenset({
    "read", "grep", "list", "ls", "list_files", "where_defined", "who_calls",
    "references", "recall_evidence", "recall", "recall_generated_artifact", "read_ticket",
})
OBSERVATION_TOOLS = frozenset({*READ_TOOLS, "probe"})
EDIT_TOOLS = frozenset({
    "edit", "str_replace", "file_create", "file_delete", "file_move", "replace_module", "write", "replace",
})


def is_read_tool(name: object) -> bool:
    return str(name or "").strip().lower() in READ_TOOLS


def is_edit_tool(name: object) -> bool:
    return str(name or "").strip().lower() in EDIT_TOOLS


def target_path(arguments: Mapping[str, object] | None) -> str:
    if not isinstance(arguments, Mapping):
        return ""
    return str(arguments.get("path") or arguments.get("target_path") or "").strip()


def hypothesis_from_edit_arguments(
    arguments: Mapping[str, object], *, default_span: object = None
) -> EditSiteHypothesis:
    """Build a typed proposal from a coding tool's optional receipt fields.

    The fields are intentionally optional at the adapter boundary so legacy
    tools can remain byte-compatible.  The host session still rejects a
    missing span/evidence set before producing an edit-site receipt.
    """

    if not isinstance(arguments, Mapping):
        raise TypeError("edit arguments must be a mapping")
    nested = arguments.get("edit_site") or arguments.get("site_hypothesis")
    if isinstance(nested, Mapping):
        path = nested.get("path") or arguments.get("path")
        span = nested.get("span", default_span)
        refs = nested.get("evidence_refs", arguments.get("evidence_refs", ()))
    else:
        path = arguments.get("path")
        span = arguments.get("span", default_span)
        if span is None and arguments.get("start_line") is not None:
            span = {
                "start_line": arguments.get("start_line"),
                "end_line": arguments.get("end_line", arguments.get("start_line")),
                "start_column": arguments.get("start_column", 0),
                "end_column": arguments.get("end_column"),
            }
        refs = arguments.get("evidence_refs", ())
    if path is None or span is None:
        raise ValueError("coding edit arguments require path and span")
    # Constructing EditSpan here gives callers a useful typed error before the
    # kernel's host verification step.
    return EditSiteHypothesis(
        path=str(path), span=EditSpan.coerce(span), evidence_refs=refs,
    )


__all__ = [
    "READ_TOOLS", "OBSERVATION_TOOLS", "EDIT_TOOLS", "is_read_tool", "is_edit_tool", "target_path",
    "hypothesis_from_edit_arguments",
]
