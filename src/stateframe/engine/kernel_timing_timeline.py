"""Bounded, source-only timing facts for one explicitly selected Python owner.

This is a candidate projection, not runtime, lease, or delivery evidence.  It
ships disabled and reports source order without predicting successful effects.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

ENV = "STATEFRAME_TIMING_TIMELINE"
HEADER = "[SOURCE TIMELINE"
MAX_SOURCE_BYTES = 256 * 1024
MAX_EVENTS = 32
_REVISION = re.compile(r"^sha256:[0-9a-f]{64}$")
_UNKNOWN = {"", "unknown", "none", "null", "unset", "n/a"}


@dataclass(frozen=True)
class TimelineEvent:
    """One literal call or control marker; columns are zero-based UTF-8 file columns."""

    kind: str
    operation: str
    receiver: str
    expression: str
    unit: str
    line: int
    end_line: int
    start_column: int
    end_column: int


@dataclass(frozen=True)
class TimelineFacts:
    """Static facts with source locators; no field asserts observed behavior."""

    status: str
    reason: str
    path: str
    owner: str
    source_revision: str
    owner_start_line: int
    owner_end_line: int
    operation: str
    default_status: str
    pre_loop_actions: tuple[TimelineEvent, ...] = ()
    loop_events: tuple[TimelineEvent, ...] = ()
    loop_exit_status: str = "UNKNOWN"


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "off") or "off").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value


def _noop(reason: str, *, path: str = "", owner: str = "",
          revision: str = "UNKNOWN") -> TimelineFacts:
    return TimelineFacts("NOOP", reason, path, owner, revision, 0, 0, "UNKNOWN", "UNKNOWN")


def _valid_relative_path(path: object) -> bool:
    if not isinstance(path, str) or not path or len(path) > 512 or "\\" in path or "\x00" in path:
        return False
    pure = PurePosixPath(path)
    return (not pure.is_absolute() and bool(pure.parts)
            and all(part not in {"", ".", ".."} for part in pure.parts)
            and pure.as_posix() == path)


def _function_rows(body: list[ast.stmt], scopes: tuple[str, ...] = ()) -> list[tuple[str, ast.AST]]:
    rows: list[tuple[str, ast.AST]] = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            rows.extend(_function_rows(node.body, (*scopes, node.name)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified = ".".join((*scopes, node.name)) if scopes else node.name
            rows.append((qualified, node))
            rows.extend(_function_rows(node.body, (*scopes, node.name)))
    return rows


def _descendants(node: ast.AST) -> list[ast.AST]:
    result: list[ast.AST] = []
    pending = list(ast.iter_child_nodes(node))
    while pending:
        current = pending.pop()
        result.append(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(current))
    return result


def _marker(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Attribute) and function.attr in {"wait", "sleep", "extend_visibility"}:
        return function.attr
    if isinstance(function, ast.Name) and function.id in {"wait", "sleep", "extend_visibility"}:
        return function.id
    return None


def _calls(node: ast.AST) -> list[ast.Call]:
    nodes = ([node] if isinstance(node, ast.Call) else []) + _descendants(node)
    return sorted((item for item in nodes
                   if isinstance(item, ast.Call) and _marker(item)),
                  key=lambda item: (item.lineno, item.col_offset, item.end_lineno or item.lineno,
                                    item.end_col_offset or item.col_offset))


def _simple_wait_call(expression: ast.AST) -> ast.Call | None:
    if isinstance(expression, ast.Call) and _marker(expression) == "wait":
        return expression
    if (isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not)
            and isinstance(expression.operand, ast.Call)
            and _marker(expression.operand) == "wait"):
        return expression.operand
    return None


def _numeric_expression(value: ast.AST, source: str) -> str | None:
    if not isinstance(value, ast.Constant) or isinstance(value.value, bool):
        return None
    if not isinstance(value.value, (int, float)):
        return None
    if isinstance(value.value, float) and not math.isfinite(value.value):
        return None
    if value.value <= 0 or abs(value.value) > 10**12:
        return None
    text = ast.get_source_segment(source, value)
    return text if isinstance(text, str) and text else None


def _event(call: ast.Call, source: str, line_offset: int,
           column_offset: int) -> tuple[TimelineEvent | None, str | None]:
    marker = _marker(call)
    if marker is None:
        return None, "not a timing marker"
    function = call.func
    if marker == "sleep":
        if (not isinstance(function, ast.Attribute) or function.attr != "sleep"
                or not isinstance(function.value, ast.Name) or function.value.id != "time"
                or len(call.args) != 1 or call.keywords):
            return None, "sleep unit or receiver is not explicit"
        argument = call.args[0]
        unit = "seconds (time.sleep call argument)"
        operation = "time.sleep"
        receiver = "time"
        kind = "wait"
    elif marker == "wait":
        if not isinstance(function, ast.Attribute) or not isinstance(function.value, ast.Name):
            return None, "wait receiver is not a simple source name"
        seconds = [item.value for item in call.keywords if item.arg == "seconds"]
        if (len(seconds) != 1 or len(call.args) != 0
                or len(call.keywords) != 1 or seconds[0] is None):
            return None, "wait duration unit is not explicit seconds"
        argument = seconds[0]
        unit = "seconds (explicit seconds keyword)"
        operation = function.attr
        receiver = function.value.id
        kind = "wait"
    else:
        if not isinstance(function, ast.Attribute) or function.attr != "extend_visibility":
            return None, "visibility operation is not receiver-bound"
        if not isinstance(function.value, ast.Name):
            return None, "visibility operation receiver is not a simple source name"
        seconds = [item.value for item in call.keywords if item.arg == "seconds"]
        if (len(seconds) != 1 or call.args or len(call.keywords) != 1 or seconds[0] is None):
            return None, "visibility grant duration unit is not explicit seconds"
        argument = seconds[0]
        unit = "seconds (explicit seconds keyword)"
        operation = "extend_visibility"
        receiver = function.value.id
        kind = "action"
    expression = _numeric_expression(argument, source)
    call_text = ast.get_source_segment(source, call)
    if expression is None or not isinstance(call_text, str) or not call_text:
        return None, "duration expression is not a bounded positive literal"
    return TimelineEvent(
        kind=kind,
        operation=operation,
        receiver=receiver,
        expression=call_text,
        unit=unit,
        line=line_offset + call.lineno,
        end_line=line_offset + int(call.end_lineno or call.lineno),
        start_column=column_offset + int(call.col_offset),
        end_column=column_offset + int(call.end_col_offset or call.col_offset),
    ), None


def _has_unsupported_control_scope(node: ast.AST) -> bool:
    control_types = (ast.Break, ast.Return, ast.Raise, ast.Continue)
    scope_types: tuple[type[ast.AST], ...] = (
        ast.For, ast.While, ast.AsyncFor, ast.Try, ast.If, ast.With, ast.AsyncWith,
    )
    nested_root_types: tuple[type[ast.AST], ...] = (
        ast.For, ast.While, ast.AsyncFor, ast.Try, ast.With, ast.AsyncWith,
    )
    try_star = getattr(ast, "TryStar", None)
    if isinstance(try_star, type):
        scope_types += (try_star,)
        nested_root_types += (try_star,)
    match_type = getattr(ast, "Match", None)
    if isinstance(match_type, type):
        scope_types += (match_type,)
        nested_root_types += (match_type,)
    pending = [(node, isinstance(node, nested_root_types))]
    while pending:
        current, nested = pending.pop()
        if nested and isinstance(current, control_types):
            return True
        if current is not node and isinstance(
                current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        child_nested = nested or (current is not node and isinstance(current, scope_types))
        pending.extend((child, child_nested) for child in ast.iter_child_nodes(current))
    return False


def _has_opaque_timing_or_control(node: ast.AST) -> bool:
    opaque_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    control_types = (ast.Break, ast.Return, ast.Raise, ast.Continue)
    for candidate in _descendants(node):
        if not isinstance(candidate, opaque_types):
            continue
        for nested in ast.walk(candidate):
            if isinstance(nested, control_types):
                return True
            if isinstance(nested, ast.Call) and _marker(nested) is not None:
                return True
    return False


def _control_events(node: ast.AST, source: str, line_offset: int,
                    column_offset: int, *, condition: str = "") -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    control_types = (ast.Break, ast.Return, ast.Raise, ast.Continue)
    items = ([node] if isinstance(node, control_types) else []) + _descendants(node)
    for item in items:
        if not isinstance(item, control_types):
            continue
        kind = ({ast.Break: "loop_exit", ast.Return: "function_exit", ast.Raise: "raise",
                 ast.Continue: "loop_continue"})[type(item)]
        text = ast.get_source_segment(source, item) or kind
        if condition:
            text = f"if {condition}: {text}"
        events.append(TimelineEvent(kind, kind, "", text, "not applicable",
                                    line_offset + item.lineno,
                                    line_offset + int(item.end_lineno or item.lineno),
                                    column_offset + int(item.col_offset),
                                    column_offset + int(item.end_col_offset or item.col_offset)))
    return events


def _is_direct_call(statement: ast.stmt, call: ast.Call) -> bool:
    return isinstance(statement, ast.Expr) and statement.value is call


def _selected_function(source: str, owner: str, owner_span: tuple[int, int] | None) -> ast.AST | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    rows = _function_rows(tree.body)
    exact = [node for name, node in rows if name == owner]
    if len(exact) == 1:
        selected = exact[0]
    elif not exact and owner_span is not None:
        leaf = owner.rsplit(".", 1)[-1]
        same_leaf = [node for name, node in rows if name == leaf]
        if len(same_leaf) != 1:
            return None
        selected = same_leaf[0]
    else:
        return None
    if owner_span is not None:
        start, end = owner_span
        if (type(start) is not int or type(end) is not int or start < 1 or end < start
                or (selected.lineno, int(selected.end_lineno or selected.lineno)) != (1, end - start + 1)):
            return None
    return selected


def analyse(source: object, *, owner: str, path: str, start_line: int = 1,
            source_revision: str | None = None,
            owner_span: tuple[int, int] | None = None,
            column_offset: int = 0) -> TimelineFacts:
    """Extract a narrow ordered fact from exactly one supplied function span.

    ``source`` should be the selected owner slice, not a whole repository dump.
    For a host-selected slice, ``owner_span`` is its absolute file line range and
    ``source_revision`` is the digest of the complete bounded source file;
    ``column_offset`` restores indentation removed from the selected slice.
    """
    if (not isinstance(source, str) or len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES
            or not isinstance(owner, str) or not owner or len(owner) > 256
            or not _valid_relative_path(path) or type(start_line) is not int or start_line < 1
            or type(column_offset) is not int or not 0 <= column_offset <= 4096):
        return _noop("invalid_source_or_locator", path=path if isinstance(path, str) else "",
                     owner=owner if isinstance(owner, str) else "")
    try:
        source_bytes = source.encode("utf-8")
    except UnicodeError:
        return _noop("invalid_source_encoding", path=path, owner=owner)
    revision = ("sha256:" + hashlib.sha256(source_bytes).hexdigest()
                if source_revision is None else source_revision)
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        return _noop("invalid_source_revision", path=path, owner=owner)
    if owner_span is not None and (not isinstance(owner_span, tuple) or len(owner_span) != 2):
        return _noop("invalid_owner_span", path=path, owner=owner, revision=revision)
    selected = _selected_function(source, owner, owner_span)
    if selected is None:
        return _noop("owner_span_not_unique_or_current", path=path, owner=owner, revision=revision)
    function_start = start_line + selected.lineno - 1
    function_end = start_line + int(selected.end_lineno or selected.lineno) - 1
    if owner_span is not None and (function_start, function_end) != owner_span:
        return _noop("owner_span_mismatch", path=path, owner=owner, revision=revision)
    if _has_opaque_timing_or_control(selected):
        return _noop("opaque_subtree_timing_or_control_unmodeled",
                     path=path, owner=owner, revision=revision)

    nodes = _descendants(selected)
    loops = [node for node in nodes if isinstance(node, (ast.While, ast.For, ast.AsyncFor))]
    candidates = [node for node in loops if _calls(node)]
    if len(candidates) != 1:
        return _noop("no_unique_relevant_loop", path=path, owner=owner, revision=revision)
    loop = candidates[0]
    if not isinstance(loop, ast.While) or loop not in selected.body:
        return _noop("unsupported_loop_control_scope", path=path, owner=owner, revision=revision)
    if loop.orelse:
        return _noop("loop_else_control_unsupported", path=path, owner=owner, revision=revision)
    if len(_calls(selected)) > MAX_EVENTS:
        return _noop("too_many_timing_markers", path=path, owner=owner, revision=revision)

    all_markers = _calls(selected)
    loop_markers = set(_calls(loop))
    pre_actions: list[TimelineEvent] = []
    loop_events: list[TimelineEvent] = []
    loop_start = loop.lineno
    for statement in selected.body:
        if statement is loop:
            break
        markers = [call for call in _calls(statement) if call not in loop_markers]
        if _has_unsupported_control_scope(statement):
            return _noop("unsupported_nested_control_scope", path=path, owner=owner, revision=revision)
        if _control_events(statement, source, start_line - 1, column_offset):
            return _noop("loop_preceded_by_exit", path=path, owner=owner, revision=revision)
        if not markers:
            continue
        for call in markers:
            event, reason = _event(call, source, start_line - 1, column_offset)
            if event is None or event.kind != "action" or not _is_direct_call(statement, call):
                return _noop(reason or "conditional_pre_loop_operation", path=path,
                             owner=owner, revision=revision)
            pre_actions.append(event)

    def add_statement_events(statement: ast.stmt) -> str | None:
        if _has_unsupported_control_scope(statement):
            return "unsupported_nested_control_scope"
        if isinstance(statement, ast.If):
            test_calls = _calls(statement.test)
            if test_calls:
                simple_wait = _simple_wait_call(statement.test)
                if len(test_calls) != 1 or simple_wait is not test_calls[0]:
                    return "unsupported_conditional_wait_expression"
                for call in test_calls:
                    event, reason = _event(call, source, start_line - 1, column_offset)
                    if event is None or event.kind != "wait":
                        return reason or "conditional_timing_order"
                    loop_events.append(event)
            if any(_control_events(branch, source, start_line - 1, column_offset)
                   for branch in statement.orelse):
                return "conditional_else_control_unsupported"
            branch_calls = [call for branch in (*statement.body, *statement.orelse)
                            for call in _calls(branch)]
            if branch_calls:
                return "timing_operation_inside_divergent_branch"
            condition = ast.get_source_segment(source, statement.test) or "condition"
            loop_events.extend(_control_events(statement, source, start_line - 1,
                                               column_offset, condition=condition))
            return None
        markers = _calls(statement)
        if markers:
            if not isinstance(statement, ast.Expr) or len(markers) != 1 or not _is_direct_call(statement, markers[0]):
                return "timing_operation_in_unsupported_statement"
            event, reason = _event(markers[0], source, start_line - 1, column_offset)
            if event is None:
                return reason or "unsupported_timing_operation"
            loop_events.append(event)
            return None
        loop_events.extend(_control_events(statement, source, start_line - 1, column_offset))
        return None

    # A wait in the loop condition is evaluated on every condition check.
    condition_calls = _calls(loop.test)
    if condition_calls:
        simple_wait = _simple_wait_call(loop.test)
        if len(condition_calls) != 1 or simple_wait is not condition_calls[0]:
            return _noop("unsupported_loop_condition", path=path, owner=owner, revision=revision)
    elif not (isinstance(loop.test, ast.Constant) and loop.test.value is True):
        return _noop("unsupported_loop_condition", path=path, owner=owner, revision=revision)
    for call in condition_calls:
        event, reason = _event(call, source, start_line - 1, column_offset)
        if event is None or event.kind != "wait":
            return _noop(reason or "unsupported_loop_condition", path=path, owner=owner, revision=revision)
        loop_events.append(event)
    for statement in loop.body:
        reason = add_statement_events(statement)
        if reason:
            return _noop(reason, path=path, owner=owner, revision=revision)
    loop_events.sort(key=lambda event: (event.line, event.start_column, event.end_line, event.end_column))
    waits = [event for event in loop_events if event.kind == "wait"]
    grants = [event for event in loop_events if event.kind == "action"]
    if len(waits) != 1 or len(grants) != 1:
        return _noop("loop_must_have_one_explicit_wait_and_grant", path=path,
                     owner=owner, revision=revision)
    required_end = max((event.line, event.start_column) for event in (*waits, *grants))
    if any((event.line, event.start_column) < required_end
           for event in loop_events
           if event.kind in {"loop_exit", "loop_continue", "function_exit", "raise"}):
        return _noop("control_exit_precedes_complete_timing_order", path=path,
                     owner=owner, revision=revision)
    receiver = grants[0].receiver
    if any(event.receiver != receiver for event in (*pre_actions, *grants)):
        return _noop("grant_receiver_mismatch", path=path, owner=owner, revision=revision)
    if not all_markers or any(call not in loop_markers
                              and not any(call is marker for statement in selected.body
                                          if statement.lineno < loop_start for marker in _calls(statement))
                              for call in all_markers):
        return _noop("timing_marker_outside_supported_order", path=path, owner=owner, revision=revision)
    exits = [event for event in loop_events if event.kind in {"loop_exit", "function_exit", "raise"}]
    exit_status = "EXPLICIT_SOURCE_EXIT" if exits else "NO_EXPLICIT_EXIT_LOCATED"
    return TimelineFacts(
        status="FACTS", reason="source_order_only", path=path, owner=owner,
        source_revision=revision, owner_start_line=function_start, owner_end_line=function_end,
        operation=f"{receiver}.extend_visibility", default_status="UNKNOWN",
        pre_loop_actions=tuple(pre_actions), loop_events=tuple(loop_events),
        loop_exit_status=exit_status,
    )


def _event_line(event: TimelineEvent) -> str:
    return (f"L{event.line}:{event.start_column}-{event.end_line}:{event.end_column} "
            f"{event.kind} {event.expression} [unit={event.unit}]")


def render(facts: object, *, max_bytes: int = 4096) -> str | None:
    """Render bounded source order; never infer a runtime interval or guarantee."""
    if not isinstance(facts, TimelineFacts) or facts.status != "FACTS":
        return None
    if type(max_bytes) is not int or max_bytes < 256 or max_bytes > 16 * 1024:
        return None
    lines = [
        f"{HEADER} | status=SOURCE_FACTS_ONLY]",
        f"source={facts.path}@{facts.source_revision} owner={facts.owner} "
        f"span={facts.owner_start_line}-{facts.owner_end_line}",
        f"operation={facts.operation}; default_binding={facts.default_status}",
    ]
    if facts.pre_loop_actions:
        lines.append("Before selected loop:")
        lines.extend("  " + _event_line(event) for event in facts.pre_loop_actions)
    else:
        lines.append("Before selected loop: no explicit matching grant call located.")
    lines.append("Selected loop source order:")
    lines.extend("  " + _event_line(event) for event in facts.loop_events)
    lines.append(f"Loop exit source facts: {facts.loop_exit_status}.")
    lines.append("These source facts do not establish execution, elapsed time, delivery, or effect.")
    rendered = "\n".join(lines)
    try:
        if len(rendered.encode("utf-8")) > max_bytes:
            return None
    except UnicodeError:
        return None
    return rendered


__all__ = ["ENV", "HEADER", "TimelineEvent", "TimelineFacts", "analyse", "mode", "render"]
