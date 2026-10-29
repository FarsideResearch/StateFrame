"""Optional readable source layout; never changes stored state or evidence authority."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import kernel_workspace_truth as _workspace_truth

from kernel_read_fixes import plain_source_labels_enabled, source_header


_CALLSITE_CENSUS_HEADER = (
    "[host source relations; static call sites of constructors used in the shown source]"
)
_CALLSITE_CENSUS_END = "[/host source relations]"


def strip_callsite_census(content: str) -> str:
    """Remove the bounded presentation-only header while preserving source bytes."""
    header_end = content.find("\n")
    if header_end < 0:
        return content
    prefix = content[:header_end + 1]
    body = content[header_end + 1:]
    if not body.startswith(_CALLSITE_CENSUS_HEADER + "\n"):
        return content
    end_marker = "\n" + _CALLSITE_CENSUS_END + "\n"
    end = body.find(end_marker)
    if end < 0:
        return content
    return prefix + body[end + len(end_marker):]


def configuration() -> str:
    mode = os.environ.get('STATEFRAME_SOURCE_PRESENTATION', 'json')
    if mode not in {'json', 'blocks', 'current_blocks'}:
        raise ValueError('STATEFRAME_SOURCE_PRESENTATION must be json, blocks or current_blocks')
    return mode


def separate_frame_sources(frame: dict, *, current_only: bool = False) -> tuple[dict, list[dict]]:
    """Move only source text out of the operational JSON; retain all metadata."""
    projected = copy.deepcopy(frame)
    sources = []
    for record in projected.get('earlier_other_file_reads', []):
        if 'content' in record:
            if current_only and record.get('revision_relevance') == 'historical_file_revision':
                record['content_withheld'] = 'historical_file_revision'
            else:
                sources.append(copy.deepcopy(record))
            record.pop('content')
    return projected, sources


def source_block(evidence: dict, *, label: str,
                 metadata_keys: tuple[str, ...] | None = None) -> str:
    content = evidence.get('content')
    path = evidence.get('path')
    if not isinstance(content, str) or not isinstance(path, str) or not path:
        raise ValueError('source presentation requires path and exact text content')
    metadata = ({key: evidence[key] for key in metadata_keys if key in evidence}
                if metadata_keys is not None else
                {key: value for key, value in evidence.items() if key != 'content'})
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    if plain_source_labels_enabled():
        role = {
            "Working source": "working",
            "Retained source": "retained",
            "Observed source": "observed",
            "Repair source": "repair",
        }.get(label, "current")
        header = source_header(path, role=role)
    else:
        header = f'[{label}: {path}]'
    return (f'{header}\n[Source metadata]\n{encoded}\n'
            f'[Source content: {path}]\n{content}\n[End source content: {path}]')


def scope_labels_enabled() -> bool:
    value = os.environ.get("STATEFRAME_WORKING_SOURCE_SCOPE", "0")
    if value not in {"0", "1"}:
        raise ValueError("STATEFRAME_WORKING_SOURCE_SCOPE must be 0 or 1")
    return value == "1"


def working_source_block(path: str, content: str, *, exists: bool,
                         workspace_root: str | Path | None = None,
                         revision: str | None = None, journal: object = None) -> str:
    """Add one coverage line without changing the existing source text or header."""
    if not isinstance(path, str) or not path or not isinstance(content, str):
        raise ValueError("working source requires path and exact text")
    if type(exists) is not bool or (not exists and content):
        raise ValueError("working source existence must agree with committed content")
    lines = len(content.splitlines())
    if not exists and _workspace_truth.enabled():
        fact = _workspace_truth.exists(
            workspace_root, path, revision=revision, journal=journal,
        )
        coverage = f"Coverage: {_workspace_truth.render_fact(fact)}; no source bytes were supplied."
    else:
        coverage = (f"Coverage: COMPLETE FILE; lines {1 if lines else 0}-{lines} of {lines}; nothing omitted."
                    if exists else "Coverage: FILE MISSING; no source available.")
    header = (source_header(path, role="working")
              if plain_source_labels_enabled() else f"[Working source: {path}]")
    return f"{header}\n{coverage}\n{content}"


__all__ = [
    "strip_callsite_census",
    "configuration",
    "scope_labels_enabled",
    "separate_frame_sources",
    "source_block",
    "working_source_block",
]
