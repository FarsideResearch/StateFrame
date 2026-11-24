"""Opt-in, backend-aware selection of independent file/check subframes.

This module decides whether fan-out is useful; the parallel tool still owns all
admission, slot leases, and execution. No source body is hidden unless the host
can name disjoint Python file/check pairs and recommends parallel execution.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SWITCH = "STATEFRAME_PARALLEL_SUGGEST"
_host_spec: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ParallelUnit:
    source: str
    test: str
    source_bytes: int
    test_bytes: int
    source_lines: int
    test_lines: int


@dataclass(frozen=True, slots=True)
class SuggestionConfig:
    min_units: int = 2
    input_share_pct: int = 50
    edit_fraction_pct: int = 25
    tokens_per_edit_line: int = 12

    def __post_init__(self) -> None:
        if self.min_units < 2 or not 1 <= self.input_share_pct <= 100:
            raise ValueError("invalid parallel suggestion unit/input threshold")
        if not 1 <= self.edit_fraction_pct <= 100 or self.tokens_per_edit_line < 1:
            raise ValueError("invalid parallel suggestion edit estimate")


@dataclass(frozen=True, slots=True)
class SuggestionDecision:
    recommend: bool
    backend_kind: str
    axis: str
    reason: str
    input_tokens: int
    edit_tokens: int
    free_slots: int


def decide_parallel_suggestion(
    units: Sequence[ParallelUnit], *, frame_budget_tokens: int,
    max_output_tokens: int, configured_slots: int, backend_kind: str,
    discovered_slots: int | None, config: SuggestionConfig = SuggestionConfig(),
) -> SuggestionDecision:
    """Select on input OR output pressure; hosted capacity needs no pressure.

    Input uses the existing request fallback of 2.46 serialized bytes/token.
    Output estimates changed lines as max(12, 25% of source lines) plus
    max(8, 25% of test lines) per pair, at 12 emitted tokens/line. These three
    parameters are configured through the engine switches documented in the
    subframe guide. The strict thresholds avoid fan-out at the exact boundary.
    """
    if backend_kind not in {"hosted", "local", "unknown"}:
        raise ValueError("unknown parallel suggestion backend")
    if (frame_budget_tokens < 1 or max_output_tokens < 1
            or configured_slots < 0 or discovered_slots is not None and discovered_slots < 0):
        raise ValueError("parallel suggestion budgets and slots must be non-negative")
    backend = "unknown" if backend_kind == "local" and discovered_slots is None else backend_kind
    free = min(configured_slots, discovered_slots) if backend == "local" else configured_slots
    # ceil(bytes / 2.46), matching kernel_request_budget's fallback calibration.
    input_tokens = math.ceil(sum(unit.source_bytes + unit.test_bytes for unit in units) / 2.46)
    edit_lines = sum(
        max(12, math.ceil(unit.source_lines * config.edit_fraction_pct / 100))
        + max(8, math.ceil(unit.test_lines * config.edit_fraction_pct / 100))
        for unit in units
    )
    edit_tokens = edit_lines * config.tokens_per_edit_line
    input_fires = input_tokens * 100 > frame_budget_tokens * config.input_share_pct
    output_fires = edit_tokens > 2 * max_output_tokens
    axis = ("both" if input_fires and output_fires else "input" if input_fires
            else "output" if output_fires else "none")
    enough_units = len(units) >= config.min_units
    recommend = enough_units and free >= 2 and (backend == "hosted" or axis != "none")
    reason = (f"{backend}:axis={axis}:units={len(units)}/{config.min_units}:"
              f"free_slots={free}:input={input_tokens}/{frame_budget_tokens}:"
              f"edit={edit_tokens}/{2 * max_output_tokens}")
    return SuggestionDecision(recommend, backend, axis, reason,
                              input_tokens, edit_tokens, free)


def independent_pairs(
    sources: Mapping[str, str], editable_paths: Sequence[str],
    check_paths: Sequence[str],
) -> tuple[ParallelUnit, ...]:
    """Return only a complete one-to-one Python import graph, without guesses."""
    editable = tuple(sorted(set(editable_paths)))
    checks = tuple(sorted(set(check_paths)))
    if len(editable) < 2 or len(checks) < 2:
        return ()
    if any(not path.endswith(".py") or path not in sources for path in (*editable, *checks)):
        return ()
    modules = {path: ".".join(Path(path).with_suffix("").parts).removeprefix("src.")
               for path in editable}
    parsed: dict[str, ast.AST] = {}
    for path in (*editable, *checks):
        try:
            parsed[path] = ast.parse(sources[path], filename=path)
        except SyntaxError:
            return ()

    def imports(path: str) -> set[str]:
        tree = parsed[path]
        found = {node.module or "" for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
        found.update(alias.name for node in ast.walk(tree)
                     if isinstance(node, ast.Import) for alias in node.names)
        return found

    def refers_to(names: set[str], module: str) -> bool:
        return any(name == module or name.endswith("." + module) for name in names)

    assignments: dict[str, str] = {}
    for check in checks:
        named = imports(check)
        matched = [path for path, module in modules.items()
                   if refers_to(named, module)]
        if not matched:
            continue
        if len(matched) != 1 or matched[0] in assignments:
            return ()
        assignments[matched[0]] = check
    if len(assignments) < 2:
        return ()
    for source in assignments:
        if any(refers_to(imports(source), modules[other])
               for other in assignments if other != source):
            return ()
    return tuple(ParallelUnit(path, assignments[path],
                              len(sources[path].encode("utf-8")),
                              len(sources[assignments[path]].encode("utf-8")),
                              len(sources[path].splitlines()),
                              len(sources[assignments[path]].splitlines()))
                 for path in sorted(assignments))


def render_suggestion(units: Sequence[ParallelUnit], *, backend_kind: str,
                      axis: str) -> tuple[str, str]:
    pairs = "; ".join(f"{unit.source}<->{unit.test}" for unit in units)
    fact = f"[Host fact] Independent pairs: {pairs}; `parallel` can do them at once."
    examples = json.dumps([
        {"id": f"unit_{index + 1}",
         "goal": f"Implement the requested change in {unit.source} and pass {unit.test}"[:300],
         "owns": [unit.source], "check": {"tests": [unit.test]}}
        for index, unit in enumerate(units)
    ], ensure_ascii=False, separators=(",", ":"))
    guidance = (f"Parallel selection: {backend_kind}, {axis} axis. Call `parallel` now with "
                f"independent subframes {examples}. Each subframe reads its own files "
                "and runs its named test. Keep shared edits in the parent.")
    return fact, guidance


def config_from_environment() -> SuggestionConfig:
    def number(raw: str | None, default: int) -> int:
        return int(raw) if raw is not None else default

    return SuggestionConfig(
        min_units=number(os.environ.get("STATEFRAME_PARALLEL_SUGGEST_MIN_UNITS"), 2),
        input_share_pct=number(os.environ.get("STATEFRAME_PARALLEL_SUGGEST_INPUT_SHARE_PCT"), 50),
        edit_fraction_pct=number(os.environ.get("STATEFRAME_PARALLEL_SUGGEST_EDIT_FRACTION_PCT"), 25),
        tokens_per_edit_line=number(
            os.environ.get("STATEFRAME_PARALLEL_SUGGEST_TOKENS_PER_EDIT_LINE"), 12),
    )


def install_host_context(spec: Mapping[str, object]) -> None:
    """Bind the parent run's capacity and provider proof; subframes never call this."""
    global _host_spec
    _host_spec = spec


def enabled() -> bool:
    value = os.environ.get(SWITCH, "off")
    if value not in {"on", "off"}:
        raise ValueError(f"{SWITCH} must be on or off")
    return value == "on" and _host_spec is not None


def _backend(spec: Mapping[str, object]) -> str:
    requested = os.environ.get("STATEFRAME_PARALLEL_SUGGEST_BACKEND", "auto")
    if requested not in {"auto", "hosted", "local", "unknown"}:
        raise ValueError("STATEFRAME_PARALLEL_SUGGEST_BACKEND must be auto, hosted, local, or unknown")
    if requested != "auto":
        return requested
    chat_url = spec.get("chat_url")
    records = spec.get("provider_capabilities")
    if not isinstance(chat_url, str) or not isinstance(records, Mapping):
        return "unknown"
    parsed = urlsplit(chat_url)
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    record = records.get(origin)
    return ("local" if isinstance(record, Mapping)
            and record.get("dialect") == "llama.cpp" else "unknown")


def _capacity(spec: Mapping[str, object], backend: str) -> tuple[int, int | None]:
    limits = spec.get("parallel_limits")
    cap = limits.get("max_concurrency", 0) if isinstance(limits, Mapping) else 0
    if type(cap) is not int or cap < 0:
        return 0, None
    mode = spec.get("parallel_mode")
    if isinstance(mode, str) and mode.isdecimal():
        cap = min(cap, int(mode))
    pool = spec.get("parallel_slots")
    if isinstance(pool, list) and all(type(slot) is int and slot >= 0 for slot in pool):
        cap = min(cap, len(pool))
    if backend != "local":
        return cap, None
    endpoint = spec.get("chat_url")
    if not isinstance(endpoint, str):
        return cap, None
    from stateframe.parallel_resources import idle_slots

    discovered = idle_slots(endpoint.removesuffix("/chat/completions"))
    if discovered is None:
        return cap, None
    parent = spec.get("prompt_cache_slot")
    parent_slot = parent.get("slot") if isinstance(parent, Mapping) else None
    allowed = set(pool) if isinstance(pool, list) else None
    free = [slot for slot in discovered if slot != parent_slot
            and (allowed is None or slot in allowed)]
    return cap, len(free)


def evaluate_host(
    sources: Mapping[str, str], editable_paths: Sequence[str],
    check_paths: Sequence[str], *, frame_budget_tokens: int,
    max_output_tokens: int,
) -> tuple[tuple[ParallelUnit, ...], SuggestionDecision] | None:
    """Return a logged decision for real pair evidence on an enabled parent."""
    if not enabled():
        return None
    assert _host_spec is not None
    units = independent_pairs(sources, editable_paths, check_paths)
    backend = _backend(_host_spec)
    cap, discovered = _capacity(_host_spec, backend)
    decision = decide_parallel_suggestion(
        units, frame_budget_tokens=frame_budget_tokens,
        max_output_tokens=max_output_tokens, configured_slots=cap,
        backend_kind=backend, discovered_slots=discovered,
        config=config_from_environment(),
    )
    return units, decision


def frame_suggestion(runtime: object, plan: Mapping[str, object],
                     tools: Sequence[Mapping[str, object]], *,
                     frame_budget_tokens: int,
                     previously_read_paths: Sequence[str] = ()) -> tuple[
                         tuple[ParallelUnit, ...], SuggestionDecision, dict[str, str]] | None:
    """Measure authenticated current files at the allocation boundary."""
    if not enabled():
        return None
    offered = {tool.get("function", {}).get("name") for tool in tools
               if isinstance(tool.get("function"), Mapping)}
    if not {"parallel", "read"}.issubset(offered):
        return None
    from kernel_measurement import output_token_limits
    from kernel_request_source_window import _safe_framefit_path
    from kernel_stateframe_framefit5 import _workspace_root
    from kernel_stateframe_headcarry import _read_source

    from stateframe.workcopy import frame_source_path_allowed

    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", None)
    row = row if isinstance(row, Mapping) else {}
    planned = {unit.get("path") for unit in plan.get("units", ())
               if isinstance(unit, Mapping) and isinstance(unit.get("path"), str)}
    declared = row.get("editable_paths")
    product_files = row.get("files")
    if row.get("scope_source") == "discovery":
        from kernel_edit_affordance import WIDENED_PATHS_KEY
        widened = row.get(WIDENED_PATHS_KEY, ())
        editable = ((widened,) if isinstance(widened, str) else
                    widened if isinstance(widened, (list, tuple)) else ())
    elif isinstance(declared, (list, tuple)) and declared:
        editable = declared
    elif isinstance(product_files, Mapping):
        product_paths = tuple(path for path in product_files
                              if isinstance(path, str)
                              and not path.startswith(("tests/", "test/"))
                              and not Path(path).name.startswith("test_"))
        ticket = row.get("msg") if isinstance(row.get("msg"), str) else ""
        editable = (product_paths if len(product_paths) <= 24 else
                    tuple(path for path in product_paths
                          if path in planned or path in ticket))
    else:
        editable = tuple(path for path in planned if not str(path).startswith("tests/"))
    declared_checks = row.get("public_check_paths")
    checks = declared_checks if isinstance(declared_checks, (list, tuple)) and declared_checks else (
        path for path in planned if str(path).startswith("tests/"))
    editable_paths = tuple(sorted(set(path for path in editable if isinstance(path, str))))
    check_paths = tuple(sorted(set(path.split("::", 1)[0] for path in checks
                                   if isinstance(path, str))))
    paths = (*editable_paths, *check_paths)
    if len(paths) > 24:
        return None
    sources: dict[str, str] = {}
    root = _workspace_root(runtime)
    for path in paths:
        if (_safe_framefit_path(path) != path or not frame_source_path_allowed(path)):
            continue
        raw = _read_source(root, path)
        if not isinstance(raw, bytes) or len(raw) > 512_000:
            continue
        try:
            sources[path] = raw.decode("utf-8")
        except UnicodeError:
            continue
    evaluated = evaluate_host(
        sources, editable_paths, check_paths,
        frame_budget_tokens=frame_budget_tokens,
        max_output_tokens=output_token_limits()[1],
    )
    if evaluated is None:
        return None
    units, decision = evaluated
    if decision.recommend and any(path in previously_read_paths
                                  for unit in units for path in (unit.source, unit.test)):
        decision = replace(decision, recommend=False,
                           reason=decision.reason + ":prior_read")
    print(f"stateframe: parallel suggestion {decision.reason}; "
          f"recommendation={'parallel' if decision.recommend else 'serial'}")
    journal = getattr(runtime, "journal", None)
    if callable(getattr(journal, "emit", None)):
        journal.emit("parallel_suggestion_decision", producer="kernel_parallel_suggest",
                     recommendation="parallel" if decision.recommend else "serial",
                     backend_kind=decision.backend_kind, axis=decision.axis,
                     reason=decision.reason, free_slots=decision.free_slots,
                     pairs=[f"{unit.source}<->{unit.test}" for unit in units],
                     dispatch=False)
    return units, decision, sources


def apply_suggestion(messages: list[dict[str, object]], plan: dict[str, object],
                     sources: Mapping[str, str], units: Sequence[ParallelUnit],
                     decision: SuggestionDecision, *,
                     protected_messages: Sequence[Mapping[str, object]] = ()) -> set[str]:
    """Swap generated pair bodies for revision-bound outlines before final fit."""
    if not decision.recommend:
        return set()
    from kernel_paged_source import parse_descriptor, render_descriptor
    from kernel_request_diet import source_window_path
    from kernel_request_source_window import parse_model_source_window

    pairs = {path for unit in units for path in (unit.source, unit.test)}
    planned = {unit.get("path") for unit in plan.get("units", ())
               if isinstance(unit, Mapping)}
    existing = plan.get("deferred_paths", ())
    plan["deferred_paths"] = list(dict.fromkeys((
        *(existing if isinstance(existing, (list, tuple)) else ()),
        *(path for path in sorted(pairs) if path in planned),
    )))
    kept = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content.startswith("[Host fact] Independent pairs:"):
            continue
        descriptor = parse_descriptor(content) if isinstance(content, str) else None
        if isinstance(content, str) and content.startswith("[Parallel outline: "):
            try:
                descriptor = json.loads(content.split("\n", 1)[1])
            except (IndexError, ValueError):
                descriptor = None
        if isinstance(descriptor, Mapping) and descriptor.get("path") in pairs:
            continue
        metadata = parse_model_source_window(content)
        path = metadata.get("path") if isinstance(metadata, Mapping) else (
            source_window_path(content) if isinstance(content, str) else None)
        if path is None and isinstance(content, str):
            header = content.split("\n", 1)[0]
            match = re.fullmatch(r"\[(?:Current|Working) source: ([^\]\n]+)\]", header)
            path = match.group(1) if match else None
        if (message.get("role") == "user" and message not in protected_messages
                and path in pairs):
            continue
        kept.append(message)
    messages[:] = kept
    from kernel_request_diet import _source_delivery_ranges
    from kernel_request_source_window import (
        DEFERRED_SOURCE_NOTICE_HEADER,
        _render_truthful_deferred_source_notice,
    )
    delivered = [(path, revision, start, end)
                 for (path, revision), ranges in sorted(_source_delivery_ranges(messages).items())
                 for start, end in ranges]
    attached = ("Current source bytes attached to this request: "
                + "; ".join(f"{path} lines {start}-{end} revision {revision}"
                            for path, revision, start, end in delivered)
                if delivered else
                "No revision-bound source windows are attached outside the StateFrame.")
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith("[Host fact]\n"):
            lines = content.splitlines()
            message["content"] = "\n".join(
                attached if line.startswith("Current source bytes attached to this request:")
                else line for line in lines)
        elif content.startswith(DEFERRED_SOURCE_NOTICE_HEADER):
            message["content"] = _render_truthful_deferred_source_notice(
                plan["deferred_paths"], delivered_source_ranges=delivered)
    fact, guidance = render_suggestion(units, backend_kind=decision.backend_kind,
                                       axis=decision.axis)
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, str):
            messages[0] = {**messages[0], "content": guidance + "\n" + content}
    # The generic delivery scanner recognizes [Paged source:] as delivered
    # bytes. These are index-only descriptors, so give them a distinct marker.
    outlines = [{"role": "user", "content": render_descriptor(path, sources[path]).replace(
        "[Paged source:", "[Parallel outline:", 1)} for path in sorted(pairs)]
    # FrameFit5 pins task then typed StateFrame immediately after system.
    from kernel_stateframe_framefit5 import STATE_HEADER
    state_index = next((index for index, message in enumerate(messages)
                        if str(message.get("content") or "").startswith(STATE_HEADER + "\n")), None)
    position = state_index + 1 if state_index is not None else 2
    messages[position:position] = [{"role": "user", "content": fact}, *outlines]
    return pairs
