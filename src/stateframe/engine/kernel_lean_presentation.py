"""Opt-in lean-history presentation for the v46 request boundary.

This module owns only model-facing presentation.  The runtime supplies the
already-final tool contract, and the existing completion/stall/premature
owners remain responsible for deciding that contract and its truth.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence

from kernel_context import token_estimate
from kernel_source_delivery import DEFAULT_MAX_CHARS


MODE_ENV = "STATEFRAME_LEAN_PRESENTATION"
EXCHANGES_ENV = "STATEFRAME_LEAN_PRESENTATION_K"
TOKENS_ENV = "STATEFRAME_LEAN_PRESENTATION_T"
TOKENS_ALIAS_ENV = "STATEFRAME_LEAN_PRESENTATION_TOKENS"
TRUTH_TAIL_ENV = "STATEFRAME_LEAN_PRESENTATION_TRUTH_TAIL"
SOURCE_TOKENS_ENV = "STATEFRAME_LEAN_SOURCE_TOKENS"
DEFAULT_EXCHANGES = 3
DEFAULT_TOKENS = 2000
# The ordinary source reader caps one delivered page at 12,000 characters.  A
# separate lean source budget therefore defaults to its conservative byte/4
# proxy, while callers can lower it when the model context is tighter.
DEFAULT_SOURCE_TOKENS = max(1, DEFAULT_MAX_CHARS // 4)
SCHEMA = "stateframe-lean-presentation/1"
WORKING_SOURCE_SECTION = "[WORKING SOURCE]"
_TERMINAL_OWNER_MARKER = "Final step: call done now to submit this revision"
_TRUTHY = frozenset({"1", "on", "true", "yes"})
_GREEN = frozenset({"PASS", "PASSED", "SATISFIED", "GREEN", "SUCCESS", "OK"})
_RED = frozenset({"FAIL", "FAILED", "ERROR", "RED", "VIOLATED", "UNSATISFIED"})
_SOURCE_HEADER = re.compile(
    r"^\[(?:Current|Working|Observed|Retained|Repair) source: ([^\]\n]+)\]\n"
)
_LEGACY_SOURCE_HEADER = re.compile(r"^CURRENT_SOURCE ([^\s]+)(?: version=([^\s]+))?\n")
_EVIDENCE_START = "[EVIDENCEPIN]"
_EVIDENCE_END = "[/EVIDENCEPIN]"
_LIMITS_START = "[CONTINUATION LIMITS |"
_COMPACT_REFUSAL_START = "[STATEFRAME compact refusal]"
_COMPACT_REFUSAL_END = "[/STATEFRAME compact refusal]"


class LeanPresentationError(ValueError):
    """Raised when a presentation switch or bound is malformed."""


@dataclasses.dataclass(frozen=True)
class Exchange:
    """One provider-complete assistant/tool exchange, copied verbatim."""

    messages: tuple[dict[str, object], ...]
    tokens: int


@dataclasses.dataclass(frozen=True)
class Projection:
    """Final model-facing request plus audit metadata."""

    messages: tuple[dict[str, object], ...]
    tools: tuple[dict[str, object], ...]
    metadata: dict[str, object]


@dataclasses.dataclass(frozen=True)
class _WorkingSource:
    path: str
    revision: str
    content: str
    rendered: str
    position: int


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise LeanPresentationError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_lean_presentation_mode", None)
    if configured is not None:
        value = str(configured).strip().lower()
        if value not in _TRUTHY | {"off"}:
            raise LeanPresentationError(f"{MODE_ENV} must be off or on")
        return value in _TRUTHY
    return mode() == "on"


def _bound(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise LeanPresentationError(f"{name} must be a nonnegative integer") from exc
    if value < 0:
        raise LeanPresentationError(f"{name} must be a nonnegative integer")
    return value


def exchange_limit(environ: Mapping[str, str] | None = None) -> int:
    source = os.environ if environ is None else environ
    return _bound(source, EXCHANGES_ENV, DEFAULT_EXCHANGES)


def token_limit(environ: Mapping[str, str] | None = None) -> int:
    source = os.environ if environ is None else environ
    primary = source.get(TOKENS_ENV)
    alias = source.get(TOKENS_ALIAS_ENV)
    if primary not in (None, "") and alias not in (None, ""):
        primary_value = _bound(source, TOKENS_ENV, DEFAULT_TOKENS)
        alias_value = _bound(source, TOKENS_ALIAS_ENV, DEFAULT_TOKENS)
        if primary_value != alias_value:
            raise LeanPresentationError(
                f"{TOKENS_ENV} and {TOKENS_ALIAS_ENV} must agree when both are set")
        return primary_value
    return _bound(source, TOKENS_ENV if primary not in (None, "") else TOKENS_ALIAS_ENV,
                  DEFAULT_TOKENS)


def source_token_limit(environ: Mapping[str, str] | None = None) -> int:
    source = os.environ if environ is None else environ
    return _bound(source, SOURCE_TOKENS_ENV, DEFAULT_SOURCE_TOKENS)


def truth_tail_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    value = str(source.get(TRUTH_TAIL_ENV, "on") or "on").strip().lower()
    if value not in _TRUTHY | {"off"}:
        raise LeanPresentationError(f"{TRUTH_TAIL_ENV} must be on or off")
    return value in _TRUTHY


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _source_path(value: object) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def _source_revision(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _same_revision(left: object, right: object) -> bool:
    if left in (None, "") or right in (None, ""):
        return False
    return str(left).removeprefix("sha256:") == str(right).removeprefix("sha256:")


def _frame_source_revisions(frame: object | None) -> dict[str, str]:
    row = _frame_mapping(frame)
    revisions: dict[str, str] = {}

    def remember(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        for raw_path, raw_revision in value.items():
            path = _source_path(raw_path)
            if path and raw_revision not in (None, ""):
                revisions.setdefault(path, str(raw_revision))

    remember(row.get("artifact_revisions"))
    for key in (
            "current_sources", "earlier_other_file_reads", "retained_evidence",
            "observed_sources", "current_source_spans", "source_spans", "recent_reads"):
        values = row.get(key)
        if isinstance(values, Mapping):
            values = (values,)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            path = _source_path(value.get("path", value.get("requested_path")))
            revision = value.get(
                "file_revision", value.get("revision", value.get("source_revision",
                                                                  value.get("version"))))
            if path and revision not in (None, ""):
                revisions.setdefault(path, str(revision))
    return revisions


def _source_message_parts(content: object) -> tuple[str, str, str] | None:
    if not isinstance(content, str):
        return None
    from kernel_source_presentation import strip_callsite_census
    content = strip_callsite_census(content)
    match = _SOURCE_HEADER.match(content)
    metadata: dict[str, object] = {}
    if match:
        path = _source_path(match.group(1))
        body = content[match.end():]
    else:
        legacy = _LEGACY_SOURCE_HEADER.match(content)
        if legacy is None:
            return None
        path = _source_path(legacy.group(1))
        body = content[legacy.end():]
        if legacy.group(2):
            metadata["version"] = legacy.group(2)
    if not path:
        return None

    if body.startswith("[Source metadata]\n"):
        marker = f"\n[Source content: {path}]\n"
        end_marker = f"\n[End source content: {path}]"
        if marker not in body or not body.endswith(end_marker):
            return None
        raw_metadata, body = body[len("[Source metadata]\n"):].split(marker, 1)
        try:
            parsed = json.loads(raw_metadata)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, Mapping):
            return None
        metadata.update(parsed)
        body = body[:-len(end_marker)]
    elif body.startswith("Coverage: "):
        coverage, separator, body = body.partition("\n")
        if coverage.startswith("Coverage: FILE MISSING"):
            return None
        if not separator:
            return None

    freshness = metadata.get("freshness")
    relevance = str(metadata.get("revision_relevance", "")).lower()
    if (freshness not in (None, "current")
            or relevance in {"historical_file_revision", "stale", "historical"}):
        return None
    revision = metadata.get(
        "file_revision", metadata.get("revision", metadata.get("source_revision",
                                                                  metadata.get("version"))))
    return path, str(revision) if revision not in (None, "") else _source_revision(body), body


def _frame_source_record(value: Mapping[str, object], position: int) -> _WorkingSource | None:
    path = _source_path(value.get("path", value.get("requested_path")))
    content = value.get("content", value.get("source"))
    if not path or not isinstance(content, str):
        return None
    if value.get("freshness") not in (None, "current"):
        return None
    if str(value.get("revision_relevance", "")).lower() in {
            "historical_file_revision", "stale", "historical"}:
        return None
    from kernel_source_presentation import source_block
    rendered = source_block(dict(value), label="Working source")
    revision = value.get("file_revision", value.get("revision", value.get("source_revision")))
    return _WorkingSource(path, str(revision or _source_revision(content)), content,
                          rendered, position)


def _working_sources(messages: Sequence[Mapping[str, object]],
                     frame: object | None) -> tuple[_WorkingSource, ...]:
    expected = _frame_source_revisions(frame)
    latest: dict[str, _WorkingSource] = {}

    def accept(candidate: _WorkingSource) -> None:
        bound = expected.get(candidate.path)
        if bound is not None and not _same_revision(candidate.revision, bound):
            return
        latest[candidate.path] = candidate

    for position, message in enumerate(messages):
        parts = _source_message_parts(message.get("content"))
        if parts is None:
            continue
        path, revision, content = parts
        accept(_WorkingSource(path, revision, content, str(message.get("content")), position))

    row = _frame_mapping(frame)
    frame_position = -1
    for key in (
            "current_sources", "earlier_other_file_reads", "retained_evidence",
            "observed_sources", "current_source_spans", "source_spans", "recent_reads"):
        values = row.get(key)
        if isinstance(values, Mapping):
            values = (values,)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            continue
        for value in values:
            if isinstance(value, Mapping):
                candidate = _frame_source_record(value, frame_position)
                if candidate is not None and candidate.path not in latest:
                    accept(candidate)
                frame_position -= 1

    return tuple(sorted(latest.values(), key=lambda item: item.position, reverse=True))


def _source_section_text(sources: Sequence[_WorkingSource]) -> str:
    lines = [WORKING_SOURCE_SECTION,
             "Current source carried into this lean request; revisions are host-bound:"]
    lines.extend(f"- {source.path} revision={source.revision}" for source in sources)
    return "\n".join(lines)


def _source_messages(sources: Sequence[_WorkingSource], limit: int) -> tuple[
        tuple[dict[str, object], ...], tuple[_WorkingSource, ...], tuple[_WorkingSource, ...], int]:
    selected: list[_WorkingSource] = []
    for source in sources:
        candidate = [*selected, source]
        candidate_messages = [
            {"role": "user", "content": _source_section_text(candidate)},
            *({"role": "user", "content": item.rendered} for item in candidate),
        ]
        if token_estimate({"messages": candidate_messages}) <= limit:
            selected.append(source)
    omitted = tuple(source for source in sources if source not in selected)
    if not selected:
        return (), (), omitted, 0
    messages = (
        {"role": "user", "content": _source_section_text(selected)},
        *({"role": "user", "content": source.rendered} for source in selected),
    )
    return messages, tuple(selected), omitted, token_estimate({"messages": messages})


def _prefix_count(messages: Sequence[Mapping[str, object]]) -> int:
    """Keep the stable system/developer prefix and the initial user request."""
    count = 0
    saw_user = False
    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            count += 1
            continue
        if role == "user" and not saw_user:
            count += 1
            saw_user = True
            break
        break
    return count


def _tool_call_ids(message: Mapping[str, object]) -> tuple[str, ...]:
    calls = message.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
        return ()
    result: list[str] = []
    for call in calls:
        row = _mapping(call)
        identifier = row.get("id") if row is not None else None
        if not isinstance(identifier, str) or not identifier or identifier in result:
            return ()
        result.append(identifier)
    return tuple(result)


def _closed_exchanges(messages: Sequence[Mapping[str, object]], prefix_count: int) -> tuple[Exchange, ...]:
    exchanges: list[Exchange] = []
    index = prefix_count
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            index += 1
            continue
        calls = _tool_call_ids(message)
        end = index + 1
        if calls:
            results: list[Mapping[str, object]] = []
            # Consume exactly this assistant message's receipts.  An unrelated
            # orphan receipt after a complete exchange must not invalidate the
            # newest valid exchange or shift the bounded window backward.
            while (end < len(messages)
                   and messages[end].get("role") == "tool"
                   and len(results) < len(calls)):
                results.append(messages[end])
                end += 1
            result_ids = tuple(
                str(result.get("tool_call_id"))
                for result in results
                if isinstance(result.get("tool_call_id"), str)
            )
            if (len(result_ids) != len(calls)
                    or len(set(result_ids)) != len(result_ids)
                    or set(result_ids) != set(calls)):
                index = end if end > index + 1 else index + 1
                continue
            group = (message, *results)
        else:
            group = (message,)
        copied = tuple(copy.deepcopy(dict(item)) for item in group)
        exchanges.append(Exchange(copied, sum(token_estimate(item) for item in copied)))
        index = end
    return tuple(exchanges)


def select_recent_exchanges(
        messages: Sequence[Mapping[str, object]], *,
        max_exchanges: int | None = None,
        max_tokens: int | None = None,
) -> tuple[Exchange, ...]:
    """Select newest complete exchanges without splitting a tool-call pair."""
    prefix_count = _prefix_count(messages)
    exchange_limit_value = DEFAULT_EXCHANGES if max_exchanges is None else max_exchanges
    token_limit_value = DEFAULT_TOKENS if max_tokens is None else max_tokens
    if exchange_limit_value < 0 or token_limit_value < 0:
        raise LeanPresentationError("presentation bounds must be nonnegative")
    candidates = _closed_exchanges(messages, prefix_count)
    selected: list[Exchange] = []
    used = 0
    recent_candidates = candidates[-exchange_limit_value:] if exchange_limit_value else ()
    for exchange in reversed(recent_candidates):
        if used + exchange.tokens > token_limit_value:
            break
        selected.append(exchange)
        used += exchange.tokens
    selected.reverse()
    return tuple(selected)


def _frame_mapping(frame: object | None) -> Mapping[str, object]:
    row = _mapping(frame)
    return row if row is not None else {}


def _field(value: object, *names: str) -> object | None:
    row = _mapping(value)
    if row is not None:
        for name in names:
            if name in row:
                return row[name]
    for name in names:
        result = getattr(value, name, None)
        if result is not None:
            return result
    return None


def _status_value(value: object) -> str | None:
    raw = _field(value, "state", "status", "verdict", "result", "result_class")
    if raw is None:
        return None
    normalized = str(raw).strip().upper()
    if normalized in _GREEN:
        return "PASSED"
    if normalized in _RED:
        return "FAILED"
    if normalized:
        return normalized
    return None


def public_check_status(runtime: object | None, frame: object | None) -> str:
    """Read public status from the existing completion authority, then frame facts."""
    try:
        from kernel_stateframe_f4 import public_checks_were_not_run
        if public_checks_were_not_run(_frame_mapping(frame)):
            return "NOT_RUN"
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    if runtime is not None:
        try:
            from kernel_completion_owner import _green_current
            if _green_current(runtime):
                return "PASSED"
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        verdict_fn = getattr(runtime, "public_test_verdict", None)
        if callable(verdict_fn):
            try:
                status = _status_value(verdict_fn())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                status = None
            if status:
                return status
    frame_row = _frame_mapping(frame)
    for value in (
            frame_row.get("verification"),
            frame_row.get("completion_verdict"),
            frame_row.get("completion_contract"),
            frame_row.get("public_checks"),
            frame_row.get("world"),
    ):
        nested = _mapping(value)
        if nested is not None:
            nested_status = _status_value(nested.get("current_public_checks"))
            if nested_status:
                return nested_status
        status = _status_value(value)
        if status:
            return status
    return "UNKNOWN"


def _open_clause_rows(runtime: object | None, frame: object | None) -> tuple[Mapping[str, object], ...]:
    if runtime is not None:
        try:
            from kernel_premature_rule import open_clause_rows
            return tuple(open_clause_rows(runtime))
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    frame_row = _frame_mapping(frame)
    raw = frame_row.get("open_clause_rows", frame_row.get("clauses", ()))
    if isinstance(raw, Mapping):
        raw = (raw,)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    rows: list[Mapping[str, object]] = []
    for index, item in enumerate(raw):
        row = _mapping(item)
        if row is None:
            row = {"id": f"clause-{index + 1}", "clause": str(item)}
        closed = str(_field(row, "status", "state", "result", "verdict") or "").upper()
        if closed not in _GREEN:
            rows.append(row)
    return tuple(rows)


def _tool_name(tool: Mapping[str, object]) -> str:
    direct = tool.get("name")
    if direct:
        return str(direct)
    function = _mapping(tool.get("function"))
    return str(function.get("name")) if function and function.get("name") else ""


def _host_line(runtime: object | None, tools: Sequence[Mapping[str, object]], status: str,
               sources: Sequence[_WorkingSource] = ()) -> str:
    def bounded(value: object) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= 1200 else text[:1199] + "…"

    base = ""
    if runtime is not None:
        try:
            from kernel_completion_owner import stall_open_work_frame
            owner_frame = stall_open_work_frame(runtime, tools)
            if isinstance(owner_frame, Mapping):
                lines = owner_frame.get("lines")
                if isinstance(lines, Sequence) and lines and isinstance(lines[0], str):
                    base = bounded(lines[0])
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        if not base:
            try:
                from kernel_completion_owner import _stall_rule_excerpt
                excerpt = _stall_rule_excerpt(runtime, None)
                if excerpt and excerpt != "no host assertion/observation excerpt recorded":
                    base = f'Last host assertion excerpt (verbatim): "{bounded(excerpt)}"'
            except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
    if not base:
        base = "public checks passed on the current revision" if status == "PASSED" else "host assertion unavailable"
    if sources:
        entries = ", ".join(
            f"{source.path}@{source.revision.removeprefix('sha256:')[:16]}"
            for source in sources
        )
        base += f"; files already read and unchanged since this revision: {entries}"
    return bounded(base)


def _remove_empty_wrapper(content: str) -> tuple[str, int]:
    """Remove only a wrapper with no body; retain every non-empty refusal."""
    removed = 0
    result = content
    while _COMPACT_REFUSAL_START in result:
        start = result.find(_COMPACT_REFUSAL_START)
        end = result.find(_COMPACT_REFUSAL_END, start + len(_COMPACT_REFUSAL_START))
        if end < 0:
            break
        body = result[start + len(_COMPACT_REFUSAL_START):end]
        if body.strip():
            break
        result = (result[:start].rstrip() + "\n" +
                  result[end + len(_COMPACT_REFUSAL_END):].lstrip()).strip()
        removed += 1
    return result, removed


def _extract_owned_blocks(content: str) -> tuple[str, str, str, str, int]:
    """Extract latest evidence/budget/terminal blocks for one final tail."""
    evidence = ""
    limits = ""
    terminal = ""
    result = content
    while _EVIDENCE_START in result:
        start = result.find(_EVIDENCE_START)
        end = result.find(_EVIDENCE_END, start + len(_EVIDENCE_START))
        if end < 0:
            break
        evidence = result[start:end + len(_EVIDENCE_END)]
        result = (result[:start].rstrip() + "\n" +
                  result[end + len(_EVIDENCE_END):].lstrip()).strip()
    remaining_lines: list[str] = []
    for line in result.splitlines():
        if line.startswith(_LIMITS_START):
            limits = line.strip()
            continue
        if _TERMINAL_OWNER_MARKER in line:
            terminal = line.strip()
            continue
        remaining_lines.append(line)
    result = "\n".join(remaining_lines).strip()
    return result, evidence, limits, terminal, 0


def _normalize_final_messages(
        messages: Sequence[Mapping[str, object]],
        *,
        collect_owned: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Normalize duplicate contracts and collect one authoritative late tail."""
    normalized: list[dict[str, object]] = []
    seen_system: set[str] = set()
    blocks = {"evidence_pin": "", "continuation_limits": "", "terminal_line": ""}
    removed_empty = 0
    removed_duplicate_system = 0
    for original in messages:
        message = copy.deepcopy(dict(original))
        role = str(message.get("role", ""))
        content = message.get("content")
        if isinstance(content, str):
            # Source messages are byte-bound evidence.  Their trailing
            # newline participates in the revision digest and must never be
            # normalized by presentation cleanup.
            is_source = bool(
                _SOURCE_HEADER.match(content)
                or _LEGACY_SOURCE_HEADER.match(content)
                or content.startswith("CURRENT_SOURCE ")
            )
            if not is_source:
                content, empty_count = _remove_empty_wrapper(content)
                removed_empty += empty_count
                if collect_owned and role not in {"system", "developer"}:
                    content, evidence, limits, terminal, _ = _extract_owned_blocks(content)
                    for key, value in (
                            ("evidence_pin", evidence),
                            ("continuation_limits", limits),
                            ("terminal_line", terminal)):
                        if value:
                            blocks[key] = value
            message["content"] = content
        if role in {"system", "developer"}:
            identity = hashlib.sha256(
                json.dumps(message, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()
            if identity in seen_system:
                removed_duplicate_system += 1
                continue
            seen_system.add(identity)
        if role == "user" and isinstance(message.get("content"), str):
            if not str(message["content"]).strip() and not message.get("tool_calls"):
                removed_empty += 1
                continue
        normalized.append(message)
    return normalized, {
        **blocks,
        "removed_empty_wrappers": removed_empty,
        "removed_duplicate_system_contracts": removed_duplicate_system,
    }


def deduplicate_final_messages(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Apply rank-1 cuts without taking ownership of late evidence blocks."""
    normalized, _ = _normalize_final_messages(messages, collect_owned=False)
    return normalized


def _runtime_terminal_line(runtime: object | None) -> str:
    if runtime is None:
        return ""
    try:
        from kernel_completion_owner import _contract_terminal_line
        value = _contract_terminal_line(runtime)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        value = ""
    return str(value or "").strip()


def _runtime_continuation_limits(runtime: object | None) -> str:
    if runtime is None:
        return ""
    limits = getattr(runtime, "continuation_limits", None)
    if limits is None:
        limits = getattr(runtime, "decision_budget", None)
    render = getattr(limits, "render_continuation_limits", None)
    if not callable(render):
        return ""
    try:
        value = str(render()).strip()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return ""
    return value if value.startswith(_LIMITS_START) else ""


def _runtime_evidence_pin(runtime: object | None) -> str:
    if runtime is None:
        return ""
    try:
        from kernel_completion_owner import _owner_block_value
        value = _owner_block_value(runtime, "evidence_pin")
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        value = ""
    return value if isinstance(value, str) and value else ""


def render_truth_tail(
        *, status: str, tools: Sequence[Mapping[str, object]],
        open_clauses: Sequence[Mapping[str, object]] = (), host_line: str = "",
        terminal_owner_line: str = "", grounded_intake_line: str = "",
        evidence_pin: str = "", continuation_limits: str = "", include_menu: bool = True,
) -> str:
    """Render public status, open clauses, host line, menu, and terminal handoff."""
    menu = ",".join(name for name in (_tool_name(tool) for tool in tools) if name)
    head = f"[LEAN TRUTH TAIL] check_status={status}"
    if include_menu:
        head += f"; menu={menu}"
    lines = [head]
    for row in open_clauses:
        clause = str(row.get("clause", row.get("text", row.get("requirement", ""))))
        if clause:
            lines.append(f"Open requirement clause (verbatim): {json.dumps(clause, ensure_ascii=False)}")
    lines.append(f"Host line (verbatim): {json.dumps(host_line, ensure_ascii=False)}")
    if status == "NO_HOST_CHECKS":
        lines.append(
            'Host limitation (verbatim): "NO HOST CHECKS: public host checks were not run."'
        )
    if grounded_intake_line:
        lines.append(f"Grounded intake line (verbatim): {json.dumps(grounded_intake_line, ensure_ascii=False)}")
    if evidence_pin:
        lines.append(evidence_pin)
    if continuation_limits:
        # Keep the canonical marker at column zero so the owner can verify the
        # exact serialized budget line after every presentation projection.
        lines.append(continuation_limits)
    if terminal_owner_line:
        lines.append(
            f"Terminal owner line (verbatim): {json.dumps(terminal_owner_line, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def project(
        messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]], *,
        runtime: object | None = None, frame: object | None = None,
        environ: Mapping[str, str] | None = None,
        ceiling_tokens: int | None = None,
) -> Projection:
    """Build the bounded D-style presentation from the already-final tool list."""
    configured_mode = getattr(runtime, "stateframe_lean_presentation_mode", None)
    is_enabled = enabled(runtime) if configured_mode is not None else mode(environ) == "on"
    if not is_enabled:
        return Projection(
            tuple(copy.deepcopy(dict(message)) for message in messages),
            tuple(copy.deepcopy(dict(tool)) for tool in tools),
            {"schema": SCHEMA, "enabled": False},
        )
    if not truth_tail_enabled(environ):
        raise LeanPresentationError(f"{TRUTH_TAIL_ENV} must be on for lean presentation")
    normalized, owned_blocks = _normalize_final_messages(
        tuple(message for message in messages if isinstance(message, Mapping))
    )
    source = tuple(normalized)
    prefix_count = _prefix_count(source)
    prefix = [copy.deepcopy(dict(message)) for message in source[:prefix_count]]
    selected = select_recent_exchanges(
        source,
        max_exchanges=exchange_limit(environ),
        max_tokens=token_limit(environ),
    )
    tool_source = tools if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)) else ()
    final_tools = tuple(copy.deepcopy(dict(tool)) for tool in tool_source if isinstance(tool, Mapping))
    status = public_check_status(runtime, frame)
    clauses = _open_clause_rows(runtime, frame)
    source_records = _working_sources(source, frame)
    source_messages, carried_sources, omitted_sources, source_tokens = _source_messages(
        source_records, source_token_limit(environ))
    host_line = _host_line(runtime, final_tools, status, carried_sources)
    grounded_intake_line = ""
    if runtime is not None:
        try:
            from kernel_grounded_intake import presentation_line
            grounded_intake_line = presentation_line(runtime, frame, source)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            grounded_intake_line = ""
    terminal_owner_line = str(owned_blocks["terminal_line"] or _runtime_terminal_line(runtime))
    evidence_pin = str(owned_blocks["evidence_pin"] or _runtime_evidence_pin(runtime))
    continuation_limits = str(
        owned_blocks["continuation_limits"] or _runtime_continuation_limits(runtime)
    )
    tail = render_truth_tail(
        status=status,
        tools=final_tools,
        open_clauses=clauses,
        host_line=host_line,
        terminal_owner_line=terminal_owner_line,
        grounded_intake_line=grounded_intake_line,
        evidence_pin=evidence_pin,
        continuation_limits=continuation_limits,
    )
    if prefix and prefix[-1].get("role") == "user" and isinstance(prefix[-1].get("content"), str):
        prefix[-1]["content"] = f"{prefix[-1]['content']}\n\n{tail}"
    else:
        prefix.append({"role": "user", "content": tail})
    def build(selected_exchanges: Sequence[Exchange]) -> tuple[dict[str, object], ...]:
        return tuple(prefix + [dict(message) for message in source_messages]
                     + [message for exchange in selected_exchanges for message in exchange.messages])

    projected = build(selected)
    fit_tokens = token_estimate({"messages": projected, "tools": final_tools})
    # Late owned blocks are mandatory.  Spend available capacity on optional
    # history first, never on the EVIDENCEPIN, limits, terminal line, or menu.
    while (ceiling_tokens is not None and fit_tokens > ceiling_tokens and selected):
        selected = selected[1:]
        projected = build(selected)
        fit_tokens = token_estimate({"messages": projected, "tools": final_tools})
    actual_fit = ceiling_tokens is None or fit_tokens <= ceiling_tokens
    window_tokens = sum(exchange.tokens for exchange in selected)
    request_digest = hashlib.sha256(
        json.dumps({"messages": projected, "tools": final_tools},
                   sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    metadata: dict[str, object] = {
        "schema": SCHEMA,
        "enabled": True,
        "exchanges_limit": exchange_limit(environ),
        "tokens_limit": token_limit(environ),
        "source_tokens_limit": source_token_limit(environ),
        "selected_exchanges": len(selected),
        "selected_tokens": window_tokens,
        "working_source_section_present": bool(source_messages),
        "working_source_files": tuple(source.path for source in carried_sources),
        "working_source_revisions": tuple(
            (source.path, source.revision) for source in carried_sources),
        "working_source_tokens": source_tokens,
        "working_source_omitted_files": tuple(source.path for source in omitted_sources),
        "prefix_messages": prefix_count,
        "open_clause_count": len(clauses),
        "terminal_owner_line_preserved": bool(terminal_owner_line),
        "grounded_intake_line_present": bool(grounded_intake_line),
        "check_status": status,
        "menu": tuple(name for name in (_tool_name(tool) for tool in final_tools) if name),
        "truth_tail": tail,
        "evidence_pin_present": bool(evidence_pin),
        "continuation_limits_present": bool(continuation_limits),
        "required_evidence_present": bool(evidence_pin and continuation_limits and terminal_owner_line),
        "fit_ceiling_tokens": ceiling_tokens,
        "fit_tokens": fit_tokens,
        "actual_fit": actual_fit,
        "removed_empty_wrappers": owned_blocks["removed_empty_wrappers"],
        "removed_duplicate_system_contracts": owned_blocks["removed_duplicate_system_contracts"],
        "request_digest": request_digest,
    }
    return Projection(projected, final_tools, metadata)


project_request = project


__all__ = [
    "DEFAULT_EXCHANGES", "DEFAULT_SOURCE_TOKENS", "DEFAULT_TOKENS", "EXCHANGES_ENV",
    "MODE_ENV", "Projection", "SCHEMA", "SOURCE_TOKENS_ENV", "TOKENS_ALIAS_ENV",
    "TOKENS_ENV", "TRUTH_TAIL_ENV", "WORKING_SOURCE_SECTION", "enabled",
    "exchange_limit", "mode", "source_token_limit", "truth_tail_enabled",
    "deduplicate_final_messages", "project", "project_request", "public_check_status", "render_truth_tail",
    "select_recent_exchanges", "token_limit",
]
