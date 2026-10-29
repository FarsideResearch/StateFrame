"""Switch-gated validation and telemetry for non-Python edit payloads.

The historical edit paths are intentionally left in place when the switch is
off.  This module is only consulted by v21 callers after the switch has been
enabled, so the baseline remains byte-identical at the dispatch boundary.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
from html.parser import HTMLParser


SWITCH_ENV = "STATEFRAME_TOKENLAB_NONPY_EDITS"
EVENT_NAME = "tokenlab_v21_nonpy_edit"
EVENT_SCHEMA = "stateframe-v21-nonpy-edits/1"

_HTML_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})
_HTML_CRITICAL_TAGS = frozenset({
    "article", "aside", "body", "div", "footer", "form", "head", "header",
    "html", "li", "main", "nav", "ol", "p", "script", "section", "style",
    "table", "tbody", "td", "tfoot", "th", "thead", "title", "tr", "ul",
})


class NonPythonValidationError(ValueError):
    """Raised when an enabled file-type gate rejects a candidate."""


class _CriticalTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._open: list[str] = []
        self._unclosed: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in _HTML_CRITICAL_TAGS and lowered not in _HTML_VOID_TAGS:
            self._open.append(lowered)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return None

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered not in _HTML_CRITICAL_TAGS:
            return
        for index in range(len(self._open) - 1, -1, -1):
            if self._open[index] == lowered:
                self._unclosed.extend(self._open[index + 1:])
                del self._open[index:]
                return

    @property
    def unclosed(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self._unclosed, *self._open)))


def enabled(environment: dict[str, str] | None = None) -> bool:
    values = os.environ if environment is None else environment
    return values.get(SWITCH_ENV, "off").strip().lower() == "on"


def file_type(path: pathlib.Path | str) -> str:
    suffix = pathlib.PurePosixPath(str(path)).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".json":
        return "json"
    if suffix in {".html", ".htm"}:
        return "html"
    return "opaque"


def applies(path: pathlib.Path | str, environment: dict[str, str] | None = None) -> bool:
    """Whether the v21 policy replaces the historical Python-only gate."""
    return enabled(environment) and file_type(path) != "python"


def _html_warning(source: str) -> str | None:
    parser = _CriticalTagParser()
    try:
        parser.feed(source)
        parser.close()
    except Exception:  # HTMLParser is advisory here and must never reject an edit.
        return "WARNING: HTML well-formedness could not be fully checked"
    if not parser.unclosed:
        return None
    tags = ", ".join(f"<{tag}>" for tag in parser.unclosed)
    return f"WARNING: unclosed critical HTML tags: {tags}"


def validate(source: str, path: pathlib.Path | str) -> str | None:
    """Validate an enabled non-Python candidate and return an advisory warning."""
    kind = file_type(path)
    if kind == "json":
        try:
            json.loads(source)
        except json.JSONDecodeError as exc:
            raise NonPythonValidationError(
                f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
            ) from exc
        return None
    if kind == "html":
        return _html_warning(source)
    return None


def validate_python(source: str, path: pathlib.Path | str) -> None:
    """Keep the explicit Python syntax contract available to shared callers."""
    ast.parse(source, filename=str(path))


def emit(
    path: pathlib.Path | str,
    *,
    operation: str,
    outcome: str,
    warning: str | None = None,
    reason: str | None = None,
) -> None:
    """Emit v21 telemetry without passing journal-reserved fields."""
    if not enabled():
        return
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        runtime = None
    journal = getattr(runtime, "journal", None)
    emit_fn = getattr(journal, "emit", None)
    if not callable(emit_fn):
        return
    fields: dict[str, object] = {
        "producer": "kernel_tokenlab_nonpy_edits",
        "event_schema": EVENT_SCHEMA,
        "path": str(path),
        "file_type": file_type(path),
        "operation": operation,
        "outcome": outcome,
        "dispatch": False,
    }
    if warning is not None:
        fields["warning"] = warning
    if reason is not None:
        fields["reason"] = reason
    emit_fn(EVENT_NAME, **fields)


__all__ = [
    "EVENT_NAME", "EVENT_SCHEMA", "NonPythonValidationError", "SWITCH_ENV",
    "applies", "emit", "enabled", "file_type", "validate", "validate_python",
]
