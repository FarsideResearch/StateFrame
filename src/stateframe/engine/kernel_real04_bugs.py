"""Opt-in r44 repairs for the real04 frame failures.

Each repair has an independent switch.  The helpers are deliberately small and
side-effect free so callers can import them only after their corresponding arm
is enabled; an OFF run therefore keeps the old object identity and wire bytes.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

from kernel_context import SelectionPriority, token_estimate

WATCHDOG_ENV = "STATEFRAME_REAL04_WATCHDOG_SPAN"
UNKNOWN_TEST_ENV = "KERNEL_REAL04_UNKNOWN_TEST_REPAIR"
TARGET_METADATA_ENV = "KERNEL_REAL04_TARGET_METADATA"

BOUNDED_SOURCE_HEADER = "[Real04 bounded source window]"
TARGET_METADATA_HEADER = "[Real04 target metadata]"


def switch_enabled(name: str) -> bool:
    value = os.environ.get(name, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{name} must be off or on")
    return value == "on"


def _source_body(content: str) -> str:
    """Remove only presentation wrappers; the stored source is never changed."""
    from kernel_source_presentation import strip_callsite_census
    content = strip_callsite_census(content)
    lines = content.splitlines()
    if lines and lines[0].startswith(("[Working source:", "[Current source:")):
        lines = lines[1:]
    if lines and lines[0].startswith("Coverage:"):
        lines = lines[1:]
    if lines and lines[0].startswith(BOUNDED_SOURCE_HEADER):
        lines = lines[1:]
    if lines and lines[0].startswith("[Source content:"):
        lines = lines[1:]
    if lines and lines[-1].startswith("[End source content:"):
        lines = lines[:-1]
    return "\n".join(lines)


def _target_span(source: str, target_fn: str) -> tuple[int, int]:
    lines = source.splitlines()
    if not lines:
        return 1, 1
    name = str(target_fn or "").rsplit(".", 1)[-1]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 1, len(lines)
    candidates = [node for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and (not name or node.name == name)]
    if not candidates:
        return 1, len(lines)
    node = min(candidates, key=lambda value: (getattr(value, "lineno", 1),
                                               getattr(value, "col_offset", 0)))
    start = max(1, int(getattr(node, "lineno", 1) or 1))
    end = max(start, int(getattr(node, "end_lineno", start) or start))
    return min(start, len(lines)), min(end, len(lines))


OMISSION_MARKER = "... [Real04 source window omitted middle lines; use reference for the complete file] ..."

_MARKER_PATTERN = re.compile(
    r"path=(?P<path>'(?:[^'\\]|\\.)*') lines=(?P<start>\d+)-(?P<end>\d+) "
    r"reference=(?P<reference>'(?:[^'\\]|\\.)*')"
)


def _window_body(source: str, start: int, end: int) -> str:
    """The untruncated (before/target/after, 3-line context) window body for a
    declared ``[start, end]`` span, from CURRENT workspace bytes.  Shared by the
    producer (``_window_text``) and the preflight verifier so both derive the
    same rendered text from the same span."""
    lines = source.splitlines()
    before = lines[max(0, start - 3):start - 1]
    target = lines[start - 1:end]
    after = lines[end:min(len(lines), end + 3)]
    return "\n".join([*before, *target, *after])


def verify_bounded_window(text: str, *, path: str, source: str) -> bool:
    """Confirm a rendered bounded-source marker's BODY derives from CURRENT
    workspace bytes at its own declared span.  The prior preflight check only
    looked for the marker header, a path substring, and the mere presence of a
    ``reference=`` token -- any working block containing those three substrings
    passed, including one whose body was replaced with stale or arbitrary text
    (fix round, rv_real04_bugs_REPORT.md finding 2).  This reconstructs the
    exact expected body via the same ``_window_body`` the producer uses and
    requires the message body to equal it exactly, or to be a genuine
    prefix+omission-marker+suffix truncation of it (the only shrink pattern
    the producer emits) -- never an unrelated or edited body.
    """
    if BOUNDED_SOURCE_HEADER not in text:
        return False
    marker_line = next(
        (line for line in text.splitlines() if line.startswith(BOUNDED_SOURCE_HEADER)), None)
    if marker_line is None:
        return False
    match = _MARKER_PATTERN.search(marker_line)
    if match is None:
        return False
    try:
        marker_path = ast.literal_eval(match.group("path"))
        reference = ast.literal_eval(match.group("reference"))
    except (ValueError, SyntaxError):
        return False
    if marker_path != path or not isinstance(reference, str) or not reference.strip():
        return False
    start, end = int(match.group("start")), int(match.group("end"))
    lines = source.splitlines()
    if start < 1 or end < start or end > max(1, len(lines)):
        return False
    marker_at = text.index(marker_line)
    body = text[marker_at + len(marker_line):]
    if body.startswith("\n"):
        body = body[1:]
    expected = _window_body(source, start, end)
    if body == expected:
        return True
    omission_wrapped = "\n" + OMISSION_MARKER + "\n"
    if omission_wrapped not in body:
        return False
    prefix, _, suffix = body.partition(omission_wrapped)
    return (bool(prefix or suffix)
            and len(prefix) + len(suffix) < len(expected)
            and expected.startswith(prefix) and expected.endswith(suffix))


def _window_text(path: str, source: str, target_fn: str, reference: str,
                 max_tokens: int) -> str:
    from kernel_read_fixes import source_header

    start, end = _target_span(source, target_fn)
    header = source_header(path, role="working")
    marker = (f"{BOUNDED_SOURCE_HEADER} path={path!r} lines={start}-{end} "
              f"reference={reference!r}")
    body = _window_body(source, start, end)
    omission = OMISSION_MARKER

    def render(value: str) -> str:
        return f"{header}\n{marker}\n{value}"

    content = render(body)
    # Keep both the declaration and the tail of a large target while shrinking
    # deterministically against the same estimator used by StateFrame.  The
    # fixed marker has a nonzero cost, so use a bounded loop and never retry an
    # unchanged body when a caller supplies an impossibly small budget.
    for _ in range(24):
        if token_estimate({"role": "user", "content": content}) <= max(1, max_tokens):
            return content
        if len(body) <= 32:
            next_body = body[:max(0, len(body) // 2)]
        else:
            keep = max(4, (len(body) - len(omission) - 2) // 4)
            next_body = body[:keep] + "\n" + omission + "\n" + body[-keep:]
        if next_body == body:
            next_body = body[:max(0, len(body) // 2)]
        body = next_body
        content = render(body)
    return content


def bound_working_source_message(message: Mapping[str, object], *, path: str,
                                 target_fn: str, reference: str,
                                 max_tokens: int) -> dict:
    if not switch_enabled(WATCHDOG_ENV):
        return message  # type: ignore[return-value]
    content = message.get("content")
    if not isinstance(content, str):
        return message  # type: ignore[return-value]
    if token_estimate(message) <= max_tokens:
        return message  # type: ignore[return-value]
    bounded = copy.deepcopy(dict(message))
    bounded["content"] = _window_text(
        path, _source_body(content), target_fn, reference, max_tokens)
    return bounded


def bound_required_source_items(items: Sequence[object], messages: Mapping[str, dict], *,
                                path: str, target_fn: str,
                                available_tokens: int) -> tuple[tuple[object, ...], dict[str, dict]]:
    """Bound the required working-source item while retaining its read reference."""
    if not switch_enabled(WATCHDOG_ENV):
        return items, messages  # type: ignore[return-value]
    source_id = "source:" + str(path)
    source_item = next((item for item in items
                        if getattr(item, "item_id", None) == source_id), None)
    source_message = messages.get(source_id)
    if source_item is None or not isinstance(source_message, Mapping):
        return items, messages  # type: ignore[return-value]
    # Fix round (rv_real04_bugs_REPORT.md finding 3): budget the source window
    # against the MANDATORY dependency closure only (required control context
    # or an item whose action_required flag makes it non-droppable).  Summing
    # every candidate -- including droppable optional/retained evidence that
    # the selector can and does omit -- exhausted the budget on paper even
    # when a bounded source plus every mandatory item actually fit, skipping
    # bounding entirely and leaving the oversized source unchanged.
    def _is_mandatory(item: object) -> bool:
        return (getattr(item, "priority", None) == SelectionPriority.REQUIRED_CONTROL_CONTEXT
                or bool(getattr(item, "action_required", False)))

    mandatory_tokens = sum(int(getattr(item, "token_estimate", 0))
                           for item in items
                           if item is not source_item and _is_mandatory(item))
    budget = int(available_tokens) - mandatory_tokens
    if budget <= 0:
        return items, messages  # type: ignore[return-value]
    reference = str(getattr(source_item, "retrievable_ref", "") or ("source:" + str(path)))
    bounded = bound_working_source_message(
        source_message, path=str(path), target_fn=str(target_fn or ""),
        reference=reference, max_tokens=budget)
    if bounded is source_message:
        return items, messages  # type: ignore[return-value]
    rendered = copy.deepcopy(dict(messages))
    rendered[source_id] = bounded
    from kernel_measurement import canonical_bytes
    rendered_item = replace(
        source_item,
        text=canonical_bytes(bounded).decode(),
        token_estimate=token_estimate(bounded),
    )
    result_items = tuple(rendered_item if item is source_item else item for item in items)
    return result_items, rendered


def classify_real04_run_test_result(observation: object, exit_code: object, *,
                                    default: str) -> str:
    """Preserve r43's classification for proven collection/import failures.

    Fix round (rv_real04_bugs_REPORT.md finding 1): the original body reclassified
    a proven exit-2 import/collection failure as NO_VERDICT (retryable_unknown).
    r43 already settles that exit code as HARNESS_ERROR (settled_not_applied) via
    RUN_TEST_RESULT_CLASSES_BY_EXIT; overriding it to a retryable class fenced the
    continuation runtime's unknown-effect retry (kernel_continuation_runtime.py)
    down to only the exact run_tests repair, so a persistently missing dependency
    could never repair itself and the run livelocked. There is no known-good
    reclassification for this case, so this function now always returns r43's
    own classification unchanged; ``observation``/``exit_code`` are accepted for
    API compatibility with existing call sites but are intentionally unused.
    """
    if not switch_enabled(UNKNOWN_TEST_ENV):
        return default
    del observation, exit_code
    return default


def _safe_target_metadata(row: Mapping[str, object]) -> tuple[dict, dict]:
    changed: dict[str, list[str]] = {}
    raw_changed = row.get("changed_functions")
    if isinstance(raw_changed, Mapping):
        for raw_path, raw_names in raw_changed.items():
            path = str(raw_path).replace("\\", "/").lstrip("./")
            if not path:
                continue
            names = [str(raw_names)] if isinstance(raw_names, str) else (
                [str(name) for name in raw_names
                 if str(name).strip()]
                if isinstance(raw_names, Sequence) else [])
            if names:
                changed[path] = names
    safe_entry: dict[str, str] = {}
    entry = row.get("entry")
    if isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
        safe_entry["path"] = str(entry["path"]).replace("\\", "/").lstrip("./")
        if isinstance(entry.get("fn"), str) and entry.get("fn"):
            safe_entry["fn"] = str(entry["fn"])
    return changed, safe_entry


def preserve_target_metadata(projected: dict, original: Mapping[str, object]) -> dict:
    """Restore only public target identity after a strict projection seam."""
    if not switch_enabled(TARGET_METADATA_ENV):
        return projected
    changed, entry = _safe_target_metadata(original)
    result = dict(projected)
    if changed:
        existing = result.get("changed_functions")
        merged = (copy.deepcopy(dict(existing))
                  if isinstance(existing, Mapping) else {})
        for path, names in changed.items():
            if path not in merged:
                merged[path] = list(names)
        result["changed_functions"] = merged
    if entry:
        existing = result.get("entry")
        merged_entry = {}
        if isinstance(existing, Mapping):
            if isinstance(existing.get("path"), str):
                merged_entry["path"] = str(existing["path"])
            if isinstance(existing.get("fn"), str) and existing.get("fn"):
                merged_entry["fn"] = str(existing["fn"])
        for key in ("path", "fn"):
            if key in entry and key not in merged_entry:
                merged_entry[key] = entry[key]
        result["entry"] = merged_entry
    return result


def target_metadata_block(row: Mapping[str, object]) -> str:
    if not switch_enabled(TARGET_METADATA_ENV):
        return ""
    changed, entry = _safe_target_metadata(row)
    if not changed and not entry:
        return ""
    return (f"{TARGET_METADATA_HEADER}\n"
            f"entry={json.dumps(entry, sort_keys=True, ensure_ascii=False)}\n"
            f"changed_functions={json.dumps(changed, sort_keys=True, ensure_ascii=False)}")


__all__ = [
    "BOUNDED_SOURCE_HEADER", "TARGET_METADATA_HEADER", "WATCHDOG_ENV",
    "UNKNOWN_TEST_ENV", "TARGET_METADATA_ENV", "bound_required_source_items",
    "bound_working_source_message", "classify_real04_run_test_result",
    "preserve_target_metadata", "switch_enabled", "target_metadata_block",
    "verify_bounded_window",
]
