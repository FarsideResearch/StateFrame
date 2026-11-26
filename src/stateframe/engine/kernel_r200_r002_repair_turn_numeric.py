"""Revision-bound arithmetic advice for a directly observed numeric assertion."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import operator
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction

PREFIX = "[Arithmetic units]"
MAX_LINE_BYTES = 720
_MAX_SOURCE_BYTES = 16_384
_REPLAY_WITHOUT_ADVISORY = object()
_NUMBER = r"[+-]?(?:\d{1,24}(?:\.\d{0,24})?|\.\d{1,24})(?:[eE][+-]?\d{1,2})?"
_NUMERIC_DIFF = re.compile(rf"^E +assert ({_NUMBER}) == ({_NUMBER})$", re.M)
_LOCATION = re.compile(r"^([\w./-]{1,160}\.py):([1-9]\d{0,5}): AssertionError$", re.M | re.ASCII)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_CITATION_END = re.compile(r" and check check-receipt:[0-9a-f]{24}@sha256:[0-9a-f]{64}\.\Z")


def is_advisory(text: str) -> bool:
    return (text.startswith(PREFIX + " ") and "\n" not in text
            and " Evidence from " in text and _CITATION_END.search(text) is not None)

# Each scale is the amount of the canonical unit in one stored increment.
_UNITS = {
    "usd": ("USD", 1), "dollars": ("USD", 1), "cents": ("USD", Fraction(1, 100)),
    "s": ("s", 1), "seconds": ("s", 1), "ms": ("s", Fraction(1, 1000)),
    "milliseconds": ("s", Fraction(1, 1000)), "minutes": ("s", 60),
    "m": ("m", 1), "meters": ("m", 1), "km": ("m", 1000),
    "ratio": ("", 1), "percent": ("", Fraction(1, 100)),
    "basis points": ("", Fraction(1, 10_000)),
}


@dataclass(frozen=True)
class Unit:
    dimensions: tuple[tuple[str, int], ...]
    scale: Fraction

    def combine(self, other: Unit, direction: int) -> Unit:
        powers = Counter(dict(self.dimensions))
        for name, power in other.dimensions:
            powers[name] += direction * power
        return Unit(tuple(sorted((name, power) for name, power in powers.items() if power)),
                    self.scale * other.scale ** direction)

    def render(self) -> str:
        dimensions = "*".join(name if power == 1 else f"{name}^{power}"
                              for name, power in self.dimensions) or "dimensionless"
        return f"{self.scale} {dimensions}/stored unit"


def _unit(annotation: ast.expr | None) -> Unit:
    match annotation:
        case ast.Subscript(value=ast.Name(id="Annotated"),
                           slice=ast.Tuple(elts=[ast.Name(id=("int" | "float")),
                                                  ast.Constant(value=str() as label)])):
            numerator, divider, denominator = label.lower().partition("/")
            name, scale = _UNITS[numerator]
            unit = Unit(((name, 1),) if name else (), Fraction(scale))
            if not divider:
                return unit
            name, scale = _UNITS[denominator]
            return unit.combine(Unit(((name, 1),) if name else (), Fraction(scale)), -1)
        case _:
            raise ValueError("unknown unit declaration")


def _number(node: ast.expr) -> Fraction:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _number(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        if abs(node.value) <= 10**24 and math.isfinite(node.value):
            return Fraction(str(node.value))
    raise ValueError("unsupported numeric literal")


def _scalar_unit(value: Fraction) -> Unit:
    """Numeric literals have no repository-grounded unit or conversion scale."""
    return Unit((), Fraction(1))


def _propagate(expression: ast.expr, declared: Mapping[str, Unit],
               arguments: Mapping[str, Fraction]) -> tuple[Unit, Fraction, str]:
    if sum(1 for _ in ast.walk(expression)) > 32:
        raise ValueError("arithmetic expression is too large")
    operands: dict[str, str] = {}

    def visit(node: ast.expr) -> tuple[Unit | None, Fraction]:
        if isinstance(node, ast.Name):
            unit = declared[node.id]
            operands[node.id] = unit.render()
            return unit, arguments[node.id]
        if not isinstance(node, ast.BinOp):
            value = _number(node)
            operands[str(value)] = "scalar"
            return None, value
        (left, a), (right, b) = visit(node.left), visit(node.right)
        if isinstance(node.op, (ast.Mult, ast.Div)):
            direction = -1 if isinstance(node.op, ast.Div) else 1
            value = a / b if direction == -1 else a * b
            if left is None and right is None:
                return None, value
            return (left or _scalar_unit(a)).combine(
                right or _scalar_unit(b), direction), value
        if isinstance(node.op, (ast.Add, ast.Sub)) and left == right:
            return left, (operator.add if isinstance(node.op, ast.Add) else operator.sub)(a, b)
        raise ValueError("unsupported arithmetic operator or incompatible operands")

    result, value = visit(expression)
    if result is None:
        raise ValueError("expression has no declared unit")
    return result, value, ", ".join(f"{name}={unit}" for name, unit in operands.items())


def _source(root, path, pages):
    from kernel_request_diet import _current_source_segments, _read_page_source_segment
    from kernel_stateframe_action_ledger import _read_open_site_file_bytes
    from kernel_stateframe_framefit5 import _safe_relative_path

    if _safe_relative_path(path) != path:
        raise ValueError("invalid source location")
    raw = _read_open_site_file_bytes(root, path)
    if len(raw) > _MAX_SOURCE_BYTES:
        raise ValueError("source is too large")
    source = raw.decode("utf-8")
    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    lines = raw.splitlines(keepends=True)
    segments = _current_source_segments(pages)
    segments.extend(segment for row in pages if (segment := _read_page_source_segment(row)))
    visible = set()
    for segment in segments:
        if (segment["path"], segment["revision"]) != (path, revision):
            continue
        start, end = segment["start_line"], segment["end_line"]
        body = segment["bytes"] if "bytes" in segment else segment["text"].encode()
        if end <= len(lines) and body == b"".join(lines[start - 1:end]):
            visible.update(range(start, end + 1))
    return source, revision, ast.parse(source), visible


def advice(root, pages, verdict, receipt, tool_text):
    """Resolve one pytest equality and one straight-line Python return."""
    revision = verdict.get("tested_revision")
    receipt_id = verdict.get("check_receipt_id")
    if (verdict.get("status") != "FAIL" or verdict.get("changed_since_check") is not False
            or revision != verdict.get("current_revision") or receipt.get("revision") != revision
            or not isinstance(revision, str) or not _DIGEST.fullmatch(revision)
            or receipt.get("receipt_id") != receipt_id or not isinstance(receipt_id, str)
            or not re.fullmatch(r"check-receipt:[0-9a-f]{24}", receipt_id)
            or receipt.get("outcome") != "BEHAVIOR_FAIL"
            or receipt.get("reached_observation") is not True
            or receipt.get("execution_complete") is not True):
        return None
    context = receipt.get("assertion_context_preview")
    if (not isinstance(context, str) or len(context) > 1200
            or receipt.get("assertion_context_omitted_bytes") != 0
            or not isinstance(tool_text, str) or repr(context) not in tool_text
            or f"full_output_handle={receipt_id}" not in tool_text):
        return None
    try:
        ((test_path, raw_line),) = _LOCATION.findall(context)
        ((observed, expected),) = _NUMERIC_DIFF.findall(context)
        if Fraction(observed) == Fraction(expected):
            return None
        selector = receipt.get("selector")
        if not isinstance(selector, str):
            return None
        test_source, test_revision, test_tree, test_visible = _source(root, test_path, pages)
        line = int(raw_line)
        if line not in test_visible:
            return None
        assertion_text = test_source.splitlines()[line - 1].strip()
        if not any(row.startswith("> ") and row[2:].strip() == assertion_text
                   for row in context.splitlines()):
            return None
        (check,) = (node for node in ast.walk(test_tree)
                    if isinstance(node, ast.Assert) and node.lineno == node.end_lineno == line)
        selector_path = selector.split("::", 1)[0]
        if selector_path != test_path:
            bare_named_test = (
                "::" not in selector
                and any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == selector
                        and any(child is check for child in ast.walk(node))
                        for node in test_tree.body)
            )
            if not bare_named_test:
                return None
        match check.test:
            case ast.Compare(left=ast.Call(func=ast.Name(id=callee), args=arguments, keywords=[]),
                             ops=[ast.Eq()], comparators=[right]):
                if _number(right) != Fraction(expected):
                    return None
            case _:
                return None
        # Only a unique direct import and literal arguments have a local,
        # deterministic mapping from this assertion to repository code.
        ((import_node, symbol),) = (
            (node, alias.name)
            for node in test_tree.body if isinstance(node, ast.ImportFrom) and node.level == 0
            for alias in node.names if (alias.asname or alias.name) == callee
        )
        if (not set(range(import_node.lineno, import_node.end_lineno + 1)) <= test_visible
                or any(isinstance(node, ast.Name) and node.id == callee
                       and not isinstance(node.ctx, ast.Load) for node in ast.walk(test_tree))):
            return None
        path = import_node.module.replace(".", "/") + ".py"
        source, source_revision, tree, visible = _source(root, path, pages)
        (typing_import,) = (node for node in tree.body if isinstance(node, ast.ImportFrom)
                            and node.module == "typing" and node.level == 0
                            and any(alias.name == "Annotated" and alias.asname is None
                                    for alias in node.names))
        (function,) = (node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == symbol)
        (return_statement,) = function.body
        if (not isinstance(return_statement, ast.Return) or return_statement.value is None
                or function.decorator_list or function.args.defaults or function.args.vararg
                or function.args.kwarg or function.args.kwonlyargs or function.args.posonlyargs
                or len(arguments) != len(function.args.args)
                or not set(range(typing_import.lineno, typing_import.end_lineno + 1)) <= visible
                or not set(range(function.lineno, return_statement.lineno + 1)) <= visible
                or return_statement.lineno != return_statement.end_lineno):
            return None
        values = {arg.arg: _number(value) for arg, value in zip(function.args.args, arguments)}
        declared = {arg.arg: _unit(arg.annotation) for arg in function.args.args}
        required = _unit(function.returns)
        result, value, operands = _propagate(return_statement.value, declared, values)
        if result == required or value != Fraction(observed):
            return None
        expression = ast.get_source_segment(source, return_statement.value)
        if not expression or len(expression) > 120:
            return None
        text = (f"{PREFIX} {path}:{return_statement.lineno} `{expression}` ({operands}). "
                f"Implied scale and dimensions {result.render()}. Required {required.render()}. "
                f"Evidence from {path}:{typing_import.lineno}-{typing_import.end_lineno},"
                f"{function.lineno}-{return_statement.lineno}@{source_revision}, "
                f"{test_path}:{import_node.lineno}-{import_node.end_lineno},{line}@{test_revision} "
                f"and check {receipt_id}@{revision}.")
        if len(text.encode()) <= MAX_LINE_BYTES:
            return text
    except (KeyError, ValueError, TypeError, AttributeError, SyntaxError, ArithmeticError, RecursionError, RuntimeError):
        return None
    return None


def admit(messages, runtime, verdict, receipt, tool_text, source_plan, allocator_on, render,
          replay_advisory=None):
    """Price the advisory against the rich envelope before giving it to a fitter."""
    from kernel_cache_layout_candidate import compose_stable_guidance, final_layout_mode
    from kernel_context import token_estimate
    from kernel_launch_identity_cost import reasoning_profile
    from kernel_provider_dialect import RUN_CAPABILITY_CACHE, origin_of
    from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET, HARD_MAX_REQUEST_BYTES
    from kernel_request_diet import RequestTokenMeter
    from kernel_stateframe_framefit5 import _workspace_root

    if replay_advisory is _REPLAY_WITHOUT_ADVISORY:
        return None

    capture = getattr(runtime, "_active_capture", None)
    surface = getattr(capture, "final_tool_surface", None)
    tools = surface.get("tools") if isinstance(surface, Mapping) else None
    draft = getattr(capture, "_capacity_retry_original_body", None)
    meter = getattr(capture, "_request_token_meter", None) or getattr(runtime, "_request_token_meter", None)
    frame_limit = getattr(runtime, "frame_ceiling", None)
    if (not isinstance(tools, list) or not isinstance(draft, Mapping)
            or meter is None or type(frame_limit) is not int):
        return None
    if receipt.get("capture_complete") is not True or receipt.get("output_truncated") is not False:
        return None
    context_preview = receipt.get("assertion_context_preview")
    if (not isinstance(context_preview, str) or len(context_preview) > 1200
            or _NUMERIC_DIFF.search(context_preview) is None):
        return None
    byte_limit, prompt_limit = HARD_MAX_REQUEST_BYTES, ENDPOINT_PROMPT_TOKEN_BUDGET
    limits = getattr(capture, "_capacity_retry_limits", None)
    if limits is not None:
        byte_limit = min(byte_limit, limits.max_request_bytes)
        prompt_limit = min(prompt_limit, limits.max_prompt_tokens)

    body = {**draft, "messages": messages, "tools": tools}
    try:
        if allocator_on:
            from kernel_working_set_live import _source_plan_with_deferred_omissions
            from kernel_working_set_render import domain, render_assignment
            from kernel_working_set_shadow import freeze_shadow, SnapshotFailure

            context = copy.copy(runtime)
            context._request_source_closure = _source_plan_with_deferred_omissions(
                copy.deepcopy(source_plan))
            route = getattr(capture, "frame_route", None)
            settings = getattr(route, "settings", {})
            endpoint = settings.get("endpoint") or getattr(capture, "_tokenize_endpoint", None)
            capabilities = RUN_CAPABILITY_CACHE.get(origin_of(endpoint)) if endpoint else None
            profile = reasoning_profile(manifest=getattr(capture.journal, "manifest", None))
            dialect = capabilities.dialect if capabilities is not None else (
                "openai-strict" if "max_completion_tokens" in body else "llama.cpp")
            snapshot = freeze_shadow(body, context, request_id=capture.request_id,
                                     dialect=dialect, reasoning_mode=profile.mode)
            if snapshot is None or isinstance(snapshot, SnapshotFailure):
                return None
            layouts = ("plain", "final_cache", "stable_first") if final_layout_mode() == "on" else ("plain",)
            snapshot = replace(snapshot, layouts=layouts, byte_cap=byte_limit,
                               prompt_limit=prompt_limit, frame_limit=frame_limit)
            assignment = {slot.key: max(domain(snapshot, slot), key=lambda mode: mode.rank)
                          for slot in snapshot.sources}
            pages = json.loads(render_assignment(snapshot, assignment).wire)["messages"]
        else:
            pages = messages
        line = advice(_workspace_root(runtime), pages, verdict, receipt, tool_text)
        if line is None:
            return None
        content = render(line)
        fixed = json.loads(snapshot.fixed_body) if allocator_on else body
        rows = fixed["messages"]
        trial = {**fixed, "messages": [*rows[:-1], {**rows[-1], "content": content}]}
        trial = compose_stable_guidance(capture, trial)
        if isinstance(replay_advisory, str) and replay_advisory == line:
            _record_advice(runtime, capture, receipt, line)
            return content
        if isinstance(meter, RequestTokenMeter):
            # A failed advisory price must not alter the live meter's fallback
            # or retry state for the unmodified request.
            probe = RequestTokenMeter(meter._tokenizer, endpoint=meter.endpoint,
                template_overhead_tokens=meter.template_overhead_tokens,
                route_allow_remote=meter.route_allow_remote, disabled=meter._disabled,
                auth_key=meter._tokenizer_auth_key, generation_endpoint=meter._generation_endpoint)
            probe.provider_identity = meter.provider_identity
            probe.bind_request(capture.request_id, meter._request_id or capture.request_id,
                               deadline=meter._deadline)
        else:
            probe = meter
        if allocator_on:
            snapshot = replace(snapshot, fixed_body=json.dumps(trial).encode())
            for layout in layouts:
                try:
                    projected = render_assignment(snapshot, assignment, layout)
                except ValueError:
                    if layout == "plain":
                        return None
                    continue  # The allocator can only admit the remaining layouts.
                projected_body = json.loads(projected.wire)
                tokens, method = probe.measure(projected_body)
                from kernel_working_set_prepare import Measurement, _violations

                if (method != "tokenize"
                        or _violations(snapshot, projected,
                                       Measurement(tokens, token_estimate(projected_body), method))):
                    return None
            _record_advice(runtime, capture, receipt, line)
        else:
            tokens, method = probe.measure(trial)
            if (method != "tokenize" or tokens > prompt_limit
                    or token_estimate(trial) > frame_limit
                    or len(json.dumps(trial).encode()) > byte_limit):
                return None
            _record_advice(runtime, capture, receipt, line)
        return content
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return None


def replay_advisory():
    """Return an advisory already present in the next recorded replay request.

    The replay hook is installed by the child as a closure over its Replayer. Reading
    its request path does not advance the tape, so assembly can retain a delivered
    advisory without probing a tokenizer that may no longer be available. A recorded
    request without one is distinct from a fresh request: replay must not add advice
    that was absent from the original send.
    """
    try:
        from kernel_cache_layout_candidate import _REPLAY_LAYOUT_DECISION_PROVIDER
    except ImportError:
        return None
    provider = _REPLAY_LAYOUT_DECISION_PROVIDER
    closure = getattr(provider, "__closure__", None)
    if not callable(provider):
        return None
    if closure is None:
        return _REPLAY_WITHOUT_ADVISORY
    replayer = next((cell.cell_contents for cell in closure
                     if callable(getattr(cell.cell_contents,
                                         "request_for_layout_decision", None))), None)
    if replayer is None:
        return None
    next_path = getattr(replayer, "next_request_path", None)
    if not callable(next_path):
        return _REPLAY_WITHOUT_ADVISORY
    try:
        if next_path() is None:
            return None
        raw = replayer.request_for_layout_decision()
        if not isinstance(raw, bytes) or len(raw) > 16 * 1024 * 1024:
            return _REPLAY_WITHOUT_ADVISORY
        body = json.loads(raw)
        messages = body.get("messages") if isinstance(body, Mapping) else None
        if not isinstance(messages, list) or not messages:
            return _REPLAY_WITHOUT_ADVISORY
        found = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            parts = [content] if isinstance(content, str) else (
                [part.get("text") for part in content
                 if isinstance(part, Mapping) and part.get("type") == "text"
                 and isinstance(part.get("text"), str)]
                if isinstance(content, list) else [])
            found.extend(line for part in parts for line in part.splitlines()
                         if is_advisory(line))
        return found[0] if len(found) == 1 else _REPLAY_WITHOUT_ADVISORY
    except (AttributeError, ImportError, IndexError, KeyError, OSError,
            TypeError, ValueError, RecursionError):
        return _REPLAY_WITHOUT_ADVISORY


def _record_advice(runtime, capture, receipt, line):
    # The detached candidate owns draft bytes. The caller's capture owns the
    # advisory audit and survives until those bytes reach transport.
    live_runtime = getattr(runtime.owner, "measurement", None)
    active = getattr(live_runtime, "_active_capture", None)
    if active is None or active.request_id != getattr(capture, "request_id", None):
        return
    # The existing transport receipt binds the candidate to the submitted
    # body hash and tests exact presence there, including after requalification.
    row = {"component": "arithmetic_units", "receipt_id": receipt["receipt_id"],
           "check_revision": receipt["revision"], "text": line}
    if row not in active.advice_candidates:
        active.advice_candidates.append(row)
