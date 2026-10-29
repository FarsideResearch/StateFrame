"""Language-neutral contracts for the benchmark honest-loop evidence path.

This module is deliberately a reducer/adapter boundary.  It does not inspect a
task, invent a check, execute a process, select an edit, or decide completion.
Existing runtime owners supply source, observations, and terminal authority;
these pure contracts only preserve their identity and classify what is known.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath


SCHEMA = "stateframe-aha10-honest-loop/1"
FREE_MODE = "FREE"
STAGED_MODE = "STAGED"
UNKNOWN = "UNKNOWN"
UNRUNNABLE = "UNRUNNABLE"
UNCOVERED = "UNCOVERED"
PARTIAL = "PARTIAL"
ENVIRONMENT = "ENVIRONMENT"
REPAIR = "REPAIR"
PRESERVE = "PRESERVE"
UNCERTIFIED_SUBMIT = "UNCERTIFIED_SUBMIT"

_OPTIONAL_SECTIONS = frozenset({
    "optional", "advice", "playbook", "history", "history_ledger", "diagnostic", "scratch",
})
_LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".go": "go", ".rs": "rust", ".ts": "typescript",
    ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
    ".java": "java", ".php": "php", ".vue": "vue",
}
_ADAPTERS = frozenset({"python", "go", "rust", "typescript", "javascript", "java", "php", "vue"})
_TOKEN = re.compile(
    r"(?P<string>(?:'[^'\n]*'|\"[^\"\n]*\"))|"
    r"(?P<bool>\b(?:True|False|true|false)\b)|"
    r"(?P<number>\b-?(?:\d+(?:\.\d*)?|\.\d+)\b)|"
    r"(?P<enum>\b[A-Z][A-Z0-9_]{2,}\b)"
)
# Whitespace before parentheses is prose in the ticket (``build (the client)``),
# not a declared call.  Source binding may still admit a genuine ``build(...)``.
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\(")
_PROSE_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s+\(")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else str(value or "").strip()


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _tool_name(tool: object) -> str:
    value = _mapping(tool)
    function = _mapping(value.get("function"))
    return _text(function.get("name") or value.get("name"))


_CAPABILITY_ALIASES = {
    "source": ("read", "grep", "list_files", "references", "where_defined", "who_calls"),
    "write": ("edit", "str_replace", "file_create", "apply_patch", "write_file", "create_file", "patch"),
    "witness": ("typed_test_writer", "run_tests", "probe", "check", "execute", "authored_check"),
    "settlement": ("done", "submit", "complete", "question_gate", "recover"),
}


def capability_routes(tools: Sequence[object]) -> dict[str, str | None]:
    """Select one currently dispatchable route for each irreducible capability."""

    names = {_tool_name(tool) for tool in tools if _tool_name(tool)}
    return {
        capability: next((name for name in aliases if name in names), None)
        for capability, aliases in _CAPABILITY_ALIASES.items()
    }


def _tool_names(tools: Sequence[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_tool_name(tool) for tool in tools if _tool_name(tool)))


def request_boundary_receipt(
        original_tools: Sequence[object], provisional_tools: Sequence[object],
        final_tools: Sequence[object], *, protected_surface: Sequence[object] = (),
        removal_reasons: Mapping[str, object] | None = None,
        hard_limit: Mapping[str, object] | None = None,
        offered_actions: Sequence[object] = (),
) -> dict[str, object]:
    """Reconcile every tool removal at one final-wire boundary.

    The provisional pass is part of the accounting surface.  A final receipt
    must not imply that no tools were removed merely because the second reducer
    saw the already-trimmed list.
    """

    original = _tool_names(original_tools)
    provisional = _tool_names(provisional_tools)
    final = _tool_names(final_tools)
    protected = tuple(dict.fromkeys(
        _text(item) if isinstance(item, str) else _tool_name(item)
        for item in protected_surface
        if (_text(item) if isinstance(item, str) else _tool_name(item))
    ))
    reasons = removal_reasons if isinstance(removal_reasons, Mapping) else {}
    removals: list[dict[str, object]] = []
    for name in original:
        if name in final:
            continue
        phase = "provisional" if name not in provisional else "final"
        configured = reasons.get(name)
        reason = _text(configured) or (
            "provisional_tool_or_schema_limit" if phase == "provisional"
            else "final_request_or_schema_limit"
        )
        removals.append({"tool": name, "phase": phase, "reason": reason})
    offered = tuple(dict.fromkeys(_text(item) for item in offered_actions if _text(item)))
    missing = tuple(item for item in offered if item not in final)
    protected_missing = tuple(item for item in protected if item not in final)
    hard = dict(hard_limit) if isinstance(hard_limit, Mapping) else None
    hard_blocked = bool(hard and hard.get("blocked", hard.get("status") in {"BLOCKED", "HOST_BLOCKED"}))
    status = "HOST_BLOCKED" if protected_missing or hard_blocked else "READY"
    return {
        "schema": SCHEMA,
        "original_tools": list(original),
        "provisional_tools": list(provisional),
        "final_tools": list(final),
        "removed_tools": removals,
        "dropped_tools": [row["tool"] for row in removals],
        "protected_surface": list(protected),
        "protected_missing": list(protected_missing),
        "offered_actions": list(offered),
        "missing_offered_actions": list(missing),
        "menu_admission_equal": not missing,
        "hard_limit": hard,
        "status": status,
    }


@dataclass(frozen=True)
class RequestContext:
    """Effective request mode; FREE has no stage exits or stage section filter."""

    mode: str
    stage: str | None
    stage_exits: frozenset[str]
    allowed_sections: frozenset[str] | None
    max_tools: int
    max_schema_bytes: int
    max_request_bytes: int

    def __post_init__(self) -> None:
        if self.mode not in {FREE_MODE, STAGED_MODE}:
            raise ValueError("request context mode must be FREE or STAGED")
        if self.mode == FREE_MODE and (self.stage is not None or self.stage_exits or self.allowed_sections is not None):
            raise ValueError("free request context cannot carry stage exits or section filtering")
        if min(self.max_tools, self.max_schema_bytes, self.max_request_bytes) <= 0:
            raise ValueError("request ceilings must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "mode": self.mode,
            "stage": self.stage,
            "stage_exits": sorted(self.stage_exits),
            "allowed_sections": sorted(self.allowed_sections) if self.allowed_sections is not None else None,
            "max_tools": self.max_tools,
            "max_schema_bytes": self.max_schema_bytes,
            "max_request_bytes": self.max_request_bytes,
        }


def request_context(
        *, idealpath_owner: object | None, captured_stage: object = None,
        stage_exits: Sequence[object] = (), allowed_sections: Sequence[object] | None = None,
        max_tools: int = 5, max_schema_bytes: int = 24 * 1024,
        max_request_bytes: int = 32 * 1024,
) -> RequestContext:
    """Resolve ownerless requests to FREE even when a stale capture says S1."""
    if idealpath_owner is None:
        return RequestContext(FREE_MODE, None, frozenset(), None,
                              max_tools, max_schema_bytes, max_request_bytes)
    stage = _text(getattr(idealpath_owner, "stage", None) or captured_stage or "S1")
    if not stage:
        stage = "S1"
    sections = None if allowed_sections is None else frozenset(
        _text(item) for item in allowed_sections if _text(item)
    )
    return RequestContext(STAGED_MODE, stage, frozenset(
        _text(item) for item in stage_exits if _text(item)
    ), sections, max_tools, max_schema_bytes, max_request_bytes)


def _wire_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _message_section(message: Mapping[str, object]) -> str:
    explicit = _text(message.get("_request_budget_section"))
    if explicit:
        return explicit.casefold()
    content = _text(message.get("content"))
    marker = re.match(r"^\[([^\]]+)\]", content)
    return marker.group(1).casefold() if marker else "required"


def _message_is_actionable(message: Mapping[str, object]) -> bool:
    content = _text(message.get("content")).casefold()
    return bool(
        message.get("source_unit_handle")
        or message.get("current_action")
        or message.get("action_result")
        or "source window" in content
        or "current action" in content
        or "action result" in content
    )


def free_diet(body: Mapping[str, object], context: RequestContext) -> dict[str, object]:
    """Trim optional material while keeping action/result/source and a continuation."""
    if context.mode != FREE_MODE:
        raise ValueError("free_diet requires a FREE request context")
    messages = [dict(item) for item in _sequence(body.get("messages")) if isinstance(item, Mapping)]
    tools = [dict(item) for item in _sequence(body.get("tools")) if isinstance(item, Mapping)]
    result: dict[str, object] = {"messages": messages, "tools": tools}
    dropped: list[str] = []
    dropped_tools: list[str] = []
    protected_tools = {
        (_tool_name(item) if isinstance(item, Mapping) else _text(item))
        for item in _sequence(body.get("protected_tools"))
        if (_tool_name(item) if isinstance(item, Mapping) else _text(item))
    }
    hard_limit = False
    while len(tools) > context.max_tools:
        candidates = [index for index, tool in enumerate(tools)
                      if _tool_name(tool) not in protected_tools]
        if not candidates:
            hard_limit = True
            break
        index = candidates[-1]
        dropped_tools.append(_tool_name(tools[index]))
        tools.pop(index)
    while tools and _wire_bytes(tools) > context.max_schema_bytes:
        candidates = [index for index, tool in enumerate(tools)
                      if _tool_name(tool) not in protected_tools]
        if not candidates:
            hard_limit = True
            break
        index = candidates[-1]
        dropped_tools.append(_tool_name(tools[index]))
        tools.pop(index)
    result["tools"] = tools
    while _wire_bytes(result) > context.max_request_bytes:
        optional = [
            (index, _wire_bytes(message)) for index, message in enumerate(messages)
            if _message_section(message) in _OPTIONAL_SECTIONS and not _message_is_actionable(message)
        ]
        if not optional:
            break
        index, _size = max(optional, key=lambda item: (item[1], -item[0]))
        dropped.append(_message_section(messages[index]))
        messages.pop(index)
        result["messages"] = messages
    actionable = [message for message in messages if _message_is_actionable(message)]
    complete = _wire_bytes(result) <= context.max_request_bytes
    continuation = None
    if not complete:
        continuation = {
            "schema": SCHEMA,
            "status": "CONTINUATION_AVAILABLE",
            "source_unit_handles": [
                _text(message.get("source_unit_handle")) for message in actionable
                if _text(message.get("source_unit_handle"))
            ],
            "dropped_sections": list(dict.fromkeys(dropped)),
            "dropped_tools": list(dict.fromkeys(item for item in dropped_tools if item)),
            "reason": "actionable_surface_exceeds_free_request_ceiling",
        }
    result["messages"] = messages
    result["dropped_optional_sections"] = list(dict.fromkeys(dropped))
    result["dropped_tools"] = list(dict.fromkeys(item for item in dropped_tools if item))
    result["actionable_messages_retained"] = len(actionable)
    result["continuation"] = continuation
    result["protected_tools"] = sorted(protected_tools)
    result["hard_limit"] = {
        "blocked": hard_limit,
        "reason": "protected_tools_exceed_free_ceiling" if hard_limit else None,
    }
    result["dispatchable_tools"] = list(_tool_names(tools))
    result["boundary_receipt"] = request_boundary_receipt(
        _sequence(body.get("tools")), _sequence(body.get("tools")), tools,
        protected_surface=sorted(protected_tools),
        hard_limit=result["hard_limit"],
    )
    return result


def bounded_source_delivery(
        units: Sequence[Mapping[str, object]], *, max_chars: int,
        required_unit_handles: Sequence[object] = (),
        required_paths: Sequence[object] = (),
) -> dict[str, object]:
    """Deliver complete actionable units before optional dependency expansion."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    grouped: dict[str, list[Mapping[str, object]]] = {}
    candidates: list[Mapping[str, object]] = []
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        path = _text(unit.get("path")) or "<opaque>"
        grouped.setdefault(path, []).append(unit)
        candidates.append(unit)
    retained: list[Mapping[str, object]] = []
    omitted: list[Mapping[str, object]] = []
    remaining = max_chars
    required_handles = {_text(item) for item in required_unit_handles if _text(item)}
    required_path_set = {_text(item) for item in required_paths if _text(item)}
    required = [unit for unit in candidates if (
        _text(unit.get("unit_handle", unit.get("handle", unit.get("id")))) in required_handles
        or _text(unit.get("path")) in required_path_set
    )]
    optional = [unit for unit in candidates if unit not in required]

    # Round-robin across required paths so one large file cannot consume the
    # entire source floor before another declared consumer gets a unit.
    def interleave(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
        selected = {id(row) for row in rows}
        queues = {
            path: [item for item in items if id(item) in selected]
            for path, items in grouped.items()
            if any(id(item) in selected for item in items)
        }
        result: list[Mapping[str, object]] = []
        while queues:
            next_queues: dict[str, list[Mapping[str, object]]] = {}
            for path, queue in queues.items():
                row = queue.pop(0)
                result.append(row)
                if queue:
                    next_queues[path] = queue
            queues = next_queues
        return result

    ordered = [*interleave(required), *interleave(optional)]
    for unit in ordered:
        raw = unit.get("text", unit.get("source", unit.get("content", "")))
        size = len(_text(raw).encode("utf-8"))
        if unit.get("complete", True) is True and size <= remaining:
            retained.append(unit)
            remaining -= size
        else:
            omitted.append(unit)
    retained_paths = {_text(unit.get("path")) for unit in retained}
    omitted_paths = {_text(unit.get("path")) for unit in omitted}
    explicit_counts = {
        key: sum(unit.get(key) is True for unit in candidates)
        for key in ("resolved", "edited", "observed")
    }
    coverage: dict[str, object] = {
        "lexical": len(candidates),
        "acquired": sum(unit.get("acquired", True) is True for unit in candidates),
        "delivered": len(retained),
        "resolved": explicit_counts["resolved"] if any("resolved" in unit for unit in candidates) else UNKNOWN,
        "edited": explicit_counts["edited"] if any("edited" in unit for unit in candidates) else UNKNOWN,
        "observed": explicit_counts["observed"] if any("observed" in unit for unit in candidates) else UNCOVERED,
    }
    status = "COMPLETE" if candidates and not omitted else PARTIAL if candidates else UNKNOWN
    revisions = sorted({
        _text(unit.get("revision")) for unit in candidates if _text(unit.get("revision"))
    })
    return {
        "schema": SCHEMA,
        "retained_units": list(retained),
        "omitted_units": list(omitted),
        "complete_units": status == "COMPLETE",
        "status": status,
        "continuation": bool(omitted),
        "continuation_action": (
            {"tool": "read", "source_paths": sorted(omitted_paths)} if omitted else None
        ),
        "source_paths": sorted(path for path in retained_paths | omitted_paths if path),
        "revisions": revisions,
        "coverage": coverage,
    }


@dataclass(frozen=True)
class RequirementResolution:
    exact_clause: str
    status: str
    candidates: tuple[str, ...]
    bindings: tuple[str, ...]
    observables: tuple[str, ...]
    quantifier: str
    owner: str
    stage_obligation: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        shared_scope = self.quantifier != "unspecified" or any(
            _whole_word(self.exact_clause, term) for term in ("shared", "universal", "all consumers")
        )
        return {
            "exact_clause": self.exact_clause,
            "status": self.status,
            "unresolved": self.status != "RESOLVED",
            "candidates": list(self.candidates),
            "bindings": list(self.bindings),
            "observables": list(self.observables),
            "quantifier": self.quantifier,
            "owner": self.owner,
            "shared_scope": shared_scope,
            "universal_scope": shared_scope,
            "stage_obligation": self.stage_obligation,
            "reason": self.reason,
        }


def _whole_word(text: str, term: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text, re.IGNORECASE) is not None


def resolve_requirement_clause(
        clause: str, *, authorized_bindings: Mapping[str, object] | Sequence[str] = (),
        observable_bindings: Mapping[str, object] | Sequence[str] = (),
) -> dict[str, object]:
    """Resolve only source-bound semantics; prose tokens remain unresolved."""
    exact = clause if isinstance(clause, str) else str(clause or "")
    binding_map = authorized_bindings if isinstance(authorized_bindings, Mapping) else {
        _text(item): {"source": _text(item)} for item in _sequence(authorized_bindings) if _text(item)
    }
    observable_map = observable_bindings if isinstance(observable_bindings, Mapping) else {
        _text(item): {"observable": _text(item)} for item in _sequence(observable_bindings) if _text(item)
    }
    # Whitespace-parenthesized prose is intentionally retained only as a
    # diagnostic hint.  It is not a callable candidate unless the clause uses
    # an actual call token and an authorized source binding.
    candidates = tuple(dict.fromkeys(match.group(1) for match in _CALL.finditer(exact)))
    prose_candidates = tuple(dict.fromkeys(match.group(1) for match in _PROSE_CALL.finditer(exact)))
    bound = tuple(item for item in candidates if item in binding_map)
    observable_names = tuple(_text(item) for item in observable_map)
    contract_trigger = any(_whole_word(exact, term) for term in (
        "must", "should", "preserve", "keep", "forward", "carry", "return", "valid", "allowed",
    ))
    bound_observables = tuple(
        _text(binding_map[name].get("observable"))
        for name in bound if isinstance(binding_map.get(name), Mapping)
        and _text(binding_map[name].get("observable"))
    )
    clause_observables = tuple(
        name for name in observable_names if _whole_word(exact, name)
    )
    has_observable = bool(bound_observables or clause_observables)
    quantifier = next((term for term in ("every", "each", "all", "any") if _whole_word(exact, term)), "unspecified")
    owner = next((term for term in ("caller", "host", "solver", "model", "worker", "runtime") if _whole_word(exact, term)), "unresolved")
    valid_bound = tuple(
        item for item in bound
        if isinstance(binding_map.get(item), Mapping)
        and (_text(binding_map[item].get("source")) or _text(binding_map[item].get("locator")))
    )
    resolved = bool(valid_bound and contract_trigger and has_observable)
    reason = "source_binding_and_observable_bound" if resolved else (
        "prose_call_candidate_requires_source_binding_and_observable"
        if prose_candidates and not candidates else "no_supported_source_bound_semantics"
    )
    stage_obligation = any(_whole_word(exact, term) for term in ("stage", "staged", "required"))
    return RequirementResolution(
        exact, "RESOLVED" if resolved else "UNRESOLVED", candidates, valid_bound,
        tuple(dict.fromkeys((*bound_observables, *clause_observables))),
        quantifier, owner, stage_obligation, reason,
    ).as_dict()


@dataclass(frozen=True)
class SourceUnit:
    handle: str
    path: str
    start_line: int
    end_line: int
    revision: str
    language: str
    kind: str
    name: str
    capabilities: tuple[str, ...]
    status: str = "READY"

    def as_dict(self) -> dict[str, object]:
        return {
            "handle": self.handle, "path": self.path, "span": [self.start_line, self.end_line],
            "revision": self.revision, "language": self.language, "kind": self.kind,
            "name": self.name, "capabilities": list(self.capabilities), "status": self.status,
        }


def _source_handle(path: str, start: int, end: int, revision: str, kind: str, name: str) -> str:
    return "unit:" + hashlib.sha256(json.dumps(
        {"path": path, "start": start, "end": end, "revision": revision, "kind": kind, "name": name},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()[:32]


def source_units(path: str, source: str, *, revision: str, language: str | None = None) -> tuple[SourceUnit, ...]:
    """Return opaque handles for module init, private/nested call sites, or file units."""
    relative = PurePosixPath(str(path).replace("\\", "/")).as_posix()
    lang = (language or _LANGUAGE_BY_SUFFIX.get(PurePosixPath(relative).suffix.lower(), "unknown")).casefold()
    if lang == "python":
        try:
            tree = ast.parse(source, filename=relative)
        except (SyntaxError, ValueError):
            return ()
        total = max(1, len(source.splitlines()))
        rows: list[SourceUnit] = []
        rows.append(SourceUnit(
            _source_handle(relative, 1, total, revision, "file", relative),
            relative, 1, total, revision, lang, "file", relative,
            ("source_read", "file_reachability"),
        ))
        rows.append(SourceUnit(
            _source_handle(relative, 1, total, revision, "module_initialization", "<module>"),
            relative, 1, total, revision, lang, "module_initialization", "<module>",
            ("source_read", "module_initialization", "reachability"),
        ))
        nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in sorted(nodes, key=lambda item: (int(getattr(item, "lineno", 0)), int(getattr(item, "end_lineno", 0)))):
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", start) or start)
            rows.append(SourceUnit(
                _source_handle(relative, start, end, revision, "callable", node.name),
                relative, start, end, revision, lang, "callable", node.name,
                ("source_read", "call", "reachability", "observation"),
            ))
        call_sites = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        for node in sorted(call_sites, key=lambda item: (int(getattr(item, "lineno", 0)), int(getattr(item, "col_offset", 0)))):
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", start) or start)
            function = getattr(node, "func", None)
            name = ast.unparse(function) if function is not None else "<call>"
            rows.append(SourceUnit(
                _source_handle(relative, start, end, revision, "call_site", name),
                relative, start, end, revision, lang, "call_site", name,
                ("source_read", "call", "reachability", "observation"),
            ))
        return tuple(rows)
    status = "READY" if lang in _ADAPTERS else UNRUNNABLE
    total = max(1, len(source.splitlines()))
    file_capabilities = ("source_read", "file_reachability", f"adapter:{lang}") if status == "READY" else ()
    rows = [SourceUnit(
        _source_handle(relative, 1, total, revision, "file", relative), relative, 1, total,
        revision, lang, "file", relative, file_capabilities, status,
    )]
    if status != "READY":
        return tuple(rows)
    rows.append(SourceUnit(
        _source_handle(relative, 1, total, revision, "module_initialization", "<module>"),
        relative, 1, total, revision, lang, "module_initialization", "<module>",
        ("source_read", "module_initialization", "reachability", f"adapter:{lang}"),
    ))
    declaration = re.compile(r"\b(?:func|fn|function)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for match in declaration.finditer(source):
        name = match.group(1)
        line = source.count("\n", 0, match.start()) + 1
        rows.append(SourceUnit(
            _source_handle(relative, line, line, revision, "callable", name),
            relative, line, line, revision, lang, "callable", name,
            ("source_read", "call", "reachability", "observation", f"adapter:{lang}"),
        ))
    calls = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    keywords = {"if", "for", "while", "switch", "catch", "func", "fn", "function"}
    for match in calls.finditer(source):
        name = match.group(1)
        declaration_prefix = source[max(0, match.start() - 16):match.start()]
        if name in keywords or re.search(r"\b(?:func|fn|function)\s+$", declaration_prefix):
            continue
        line = source.count("\n", 0, match.start()) + 1
        rows.append(SourceUnit(
            _source_handle(relative, line, line, revision, "call_site", name),
            relative, line, line, revision, lang, "call_site", name,
            ("source_read", "call", "reachability", "observation", f"adapter:{lang}"),
        ))
    return tuple(rows)


def source_unit_fixtures() -> tuple[dict[str, object], ...]:
    """Metadata fixtures proving adapter labels are not Python-name policy."""
    return tuple({"language": language, "path": f"fixture.{suffix}", "status": "UNRUNNABLE" if language == "unknown" else "READY"}
                 for language, suffix in (
                     ("python", "py"), ("go", "go"), ("rust", "rs"), ("typescript", "ts"),
                     ("javascript", "js"), ("java", "java"), ("php", "php"), ("vue", "vue"),
                 ))


@dataclass(frozen=True)
class InventoryMatch:
    """One lexical match with syntax role and independent evidence flags."""

    query: str
    path: str
    line: int
    role: str
    language: str
    revision: str
    delivered: bool = False
    resolved: bool = False
    observed: bool = False


def inventory_matches(path: str, source: str, query: str, *,
                      revision: object = None, language: str | None = None) -> tuple[InventoryMatch, ...]:
    """Group lexical hits without promoting them to resolved behavior."""
    normalized = path.replace("\\", "/")
    lang = (language or _LANGUAGE_BY_SUFFIX.get(PurePosixPath(normalized).suffix.lower(), "unknown"))
    revision_value = _text(revision) or _digest(source)
    docstring_lines: set[int] = set()
    if lang == "python":
        try:
            tree = ast.parse(source, filename=normalized)
            scopes = [tree, *[node for node in ast.walk(tree)
                              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]]
            for scope in scopes:
                first = getattr(scope, "body", [None])[0]
                value = getattr(getattr(first, "value", None), "value", None)
                if isinstance(first, ast.Expr) and isinstance(value, str):
                    docstring_lines.update(range(
                        int(getattr(first, "lineno", 1) or 1),
                        int(getattr(first, "end_lineno", getattr(first, "lineno", 1)) or 1) + 1,
                    ))
        except (SyntaxError, ValueError):
            pass
    rows: list[InventoryMatch] = []
    for line_number, line in enumerate(source.splitlines(), 1):
        if query not in line:
            continue
        if line_number in docstring_lines:
            rows.append(InventoryMatch(query, normalized, line_number, "docstring", lang, revision_value))
            continue
        comment_start = len(line)
        for marker in ("#", "//", "/*", "<!--"):
            position = line.find(marker)
            if position >= 0:
                comment_start = min(comment_start, position)
        if line[:comment_start].find(query) >= 0:
            rows.append(InventoryMatch(query, normalized, line_number, "executable", lang, revision_value))
        if comment_start < len(line) and query in line[comment_start:]:
            rows.append(InventoryMatch(query, normalized, line_number, "comment", lang, revision_value))
    return tuple(rows)


def inventory_coverage(matches: Sequence[InventoryMatch]) -> dict[str, int]:
    """Return lexical/delivered/resolved/observed counts as separate facts."""
    return {
        "lexical": len(matches),
        "delivered": sum(item.delivered for item in matches),
        "resolved": sum(item.resolved for item in matches),
        "observed": sum(item.observed for item in matches),
        "executable": sum(item.role == "executable" for item in matches),
        "comments": sum(item.role == "comment" for item in matches),
        "docstrings": sum(item.role == "docstring" for item in matches),
    }


def _host_execution_record(
        receipt: Mapping[str, object], execution_store: object,
) -> Mapping[str, object] | None:
    """Resolve a receipt only through the host execution store, never its own claims."""
    if isinstance(execution_store, Mapping):
        direct = execution_store.get(_text(receipt.get("receipt_id")))
        if isinstance(direct, Mapping):
            return direct
        rows = execution_store.get("records", ())
    else:
        rows = execution_store
    for row in _sequence(rows):
        if (isinstance(row, Mapping)
                and _text(row.get("receipt_id")) == _text(receipt.get("receipt_id"))):
            return row
    return None


def validate_solver_check_receipt(
        receipt: Mapping[str, object], *, source_unit_handle: str, revision: str,
        input_value: object, parent_identity: str, candidate_identity: str, clause_id: str,
        execution_store: object = None, consumers: Sequence[str] = (),
) -> dict[str, object]:
    """Validate an already executed receipt without authoring its check/oracle."""
    host_execution = _host_execution_record(receipt, execution_store)
    if host_execution is None:
        return {
            "schema": SCHEMA, "status": UNKNOWN, "receipt_id": _text(receipt.get("receipt_id")),
            "reason": "host_execution_store_record_required",
        }
    host_identity_fields = (
        "receipt_id", "check_identity", "oracle_identity", "source_unit_handle",
        "revision", "parent_identity", "candidate_identity", "clause_id",
    )
    if any(
            not _text(host_execution.get(key))
            or host_execution.get(key) != receipt.get(key)
            for key in host_identity_fields
    ):
        return {
            "schema": SCHEMA, "status": UNKNOWN, "receipt_id": _text(receipt.get("receipt_id")),
            "reason": "receipt_does_not_match_host_execution_identity",
        }
    host_has_payload = any(key in host_execution for key in (
        "input", "observation", "integrity_digest", "integrity",
    ))
    if host_has_payload and dict(host_execution) != dict(receipt):
        return {
            "schema": SCHEMA, "status": UNKNOWN, "receipt_id": _text(receipt.get("receipt_id")),
            "reason": "receipt_does_not_match_host_execution_record",
        }
    provenance = receipt.get("provenance")
    if receipt.get("host_authored") is True or (
            isinstance(provenance, Mapping)
            and _text(provenance.get("kind")).upper() in {"HOST_CHECK", "HOST_AUTHORED", "MODEL_CLAIM"}
    ):
        return {
            "schema": SCHEMA, "status": UNKNOWN, "receipt_id": _text(receipt.get("receipt_id")),
            "reason": "host_must_not_author_check_or_oracle",
        }
    required = {
        "receipt_id": _text(receipt.get("receipt_id")),
        "source_unit_handle": _text(receipt.get("source_unit_handle")),
        "revision": _text(receipt.get("revision")),
        "parent_identity": _text(receipt.get("parent_identity")),
        "candidate_identity": _text(receipt.get("candidate_identity")),
        "clause_id": _text(receipt.get("clause_id")),
        "check_identity": _text(receipt.get("check_identity", receipt.get("check_id"))),
        "oracle_identity": _text(receipt.get("oracle_identity", receipt.get("oracle_hash"))),
    }
    missing = [name for name, value in required.items() if not value]
    missing.extend(name for name in ("input", "observation") if name not in receipt)
    if "reachability" not in receipt and "reachable" not in receipt:
        missing.append("reachability")
    if "integrity" not in receipt and "integrity_digest" not in receipt:
        missing.append("integrity")
    reached_consumers = {
        _text(item) for item in _sequence(
            receipt.get("reached_consumers", receipt.get("reached_source_units", ()))
        ) if _text(item)
    }
    declared_consumers = {_text(item) for item in consumers if _text(item)}
    if declared_consumers and not declared_consumers.issubset(reached_consumers):
        missing.append("reached_consumers")
    reached = receipt.get("reachability", receipt.get("reachable"))
    source_revision = receipt.get("source_revision")
    source_revision_matches = source_revision in (None, "", revision)
    matches = (
        not missing and required["source_unit_handle"] == source_unit_handle
        and required["revision"] == revision and required["parent_identity"] == parent_identity
        and required["candidate_identity"] == candidate_identity and required["clause_id"] == clause_id
        and receipt.get("input") == input_value and reached is True
        and source_revision_matches
        and receipt.get("immutable") is True and receipt.get("solver_authored") is True
    )
    host_status = _text(host_execution.get("status", host_execution.get("outcome"))).upper()
    host_matches = host_status in {"EXECUTED", "PASS", "PASSED", "FAIL", "FAILED"}
    matches = matches and host_matches
    payload = {key: value for key, value in receipt.items()
               if key not in {"integrity_digest", "integrity"}}
    expected_digest = _digest(payload)
    integrity = (receipt.get("integrity_digest") == expected_digest
                 or receipt.get("integrity") == expected_digest)
    status = "VALID" if matches and integrity else UNKNOWN
    return {
        "schema": SCHEMA, "status": status, "receipt_id": required["receipt_id"],
        "reason": "validated_executed_receipt" if status == "VALID" else "receipt_identity_or_integrity_incomplete",
        "source_unit_handle": required["source_unit_handle"], "revision": required["revision"],
        "observation": receipt.get("observation"), "integrity": integrity,
    }


def _receipt_for(receipts: Sequence[object], key: str) -> Mapping[str, object] | None:
    for receipt in receipts:
        if isinstance(receipt, Mapping) and _text(receipt.get("receipt_id")) == key:
            return receipt
    return None


def _settlement_receipt_valid(receipt: Mapping[str, object], *, clause_id: str,
                              revision: str, execution_store: object = None,
                              consumers: Sequence[str] = ()) -> bool:
    """Require the same immutable identity contract used by witness binding."""
    source_unit = _text(receipt.get("source_unit_handle"))
    parent = _text(receipt.get("parent_identity"))
    candidate = _text(receipt.get("candidate_identity"))
    if not source_unit or not parent or not candidate:
        return False
    return validate_solver_check_receipt(
        receipt, source_unit_handle=source_unit, revision=revision,
        input_value=receipt.get("input"), parent_identity=parent,
        candidate_identity=candidate, clause_id=clause_id,
        execution_store=execution_store,
        consumers=consumers,
    ).get("status") == "VALID"


def settle_obligations(
        obligations: Mapping[str, Mapping[str, object]], receipts: Sequence[object], *,
        revision: str, parent_identity: str = "", candidate_identity: str = "",
        execution_store: object = None,
) -> dict[str, object]:
    """Classify each clause/site without converting missing evidence to success."""
    rows: dict[str, dict[str, object]] = {}
    for obligation_id, obligation in obligations.items():
        role = _text(obligation.get("role", REPAIR)).upper()
        reachability_declared = any(key in obligation for key in ("source_unit_handles", "consumers"))
        unit_handles = tuple(_text(item) for item in _sequence(obligation.get("source_unit_handles", obligation.get("consumers", ()))) if _text(item))
        expected_check = _text(obligation.get("check_identity", obligation.get("check_id")))
        expected_oracle = _text(obligation.get("oracle_identity", obligation.get("oracle_hash")))
        relevant = [receipt for receipt in receipts if isinstance(receipt, Mapping)
                    and _text(receipt.get("clause_id")) == str(obligation_id)]
        current: list[Mapping[str, object]] = []
        for item in relevant:
            if _text(item.get("revision")) != revision:
                continue
            if not _settlement_receipt_valid(
                    item, clause_id=str(obligation_id), revision=revision,
                    execution_store=execution_store, consumers=unit_handles):
                # ENVIRONMENT and UNRUNNABLE are honest host outcomes, but they
                # are not executed evidence and can never certify a clause.
                continue
            if expected_check and _text(item.get("check_identity", item.get("check_id"))) != expected_check:
                continue
            if expected_oracle and _text(item.get("oracle_identity", item.get("oracle_hash"))) != expected_oracle:
                continue
            current.append(item)
        if not current:
            rows[str(obligation_id)] = {"outcome": UNRUNNABLE if obligation.get("unsupported") else UNCOVERED,
                                        "verified": False, "reason": "current_revision_evidence_missing"}
            continue
        parent = [item for item in current if _text(item.get("candidate_identity")) == parent_identity]
        candidate = [item for item in current if _text(item.get("candidate_identity")) == candidate_identity]
        parent_reached = {_text(item.get("source_unit_handle")) for item in parent
                          if item.get("reachability", item.get("reachable")) is True}
        candidate_reached = {_text(item.get("source_unit_handle")) for item in candidate
                            if item.get("reachability", item.get("reachable")) is True}
        reached = parent_reached | candidate_reached
        missed = sorted(set(unit_handles) - parent_reached) + sorted(set(unit_handles) - candidate_reached)
        if reachability_declared and not unit_handles:
            missed = ["<declared-but-empty-reachability>"]
        parent_pass = any(_text(item.get("outcome")).upper() in {"PASS", "PASSED"} for item in parent)
        candidate_pass = any(_text(item.get("outcome")).upper() in {"PASS", "PASSED"} for item in candidate)
        parent_fail = any(_text(item.get("outcome")).upper() in {"FAIL", "FAILED", "COUNTEREXAMPLE"} for item in parent)
        if missed:
            outcome, reason = UNCOVERED, "relevant_consumer_not_reached"
        elif role == REPAIR and parent_fail and candidate_pass:
            outcome, reason = REPAIR, "parent_failed_candidate_passed"
        elif role == PRESERVE and parent_pass and candidate_pass:
            outcome, reason = PRESERVE, "parent_and_candidate_preserved"
        elif any(_text(item.get("outcome")).upper() == ENVIRONMENT for item in current):
            outcome, reason = ENVIRONMENT, "environment_observation"
        elif any(_text(item.get("outcome")).upper() == UNRUNNABLE for item in current):
            outcome, reason = UNRUNNABLE, "adapter_unavailable"
        else:
            outcome, reason = UNKNOWN, "required_pair_not_observed"
        rows[str(obligation_id)] = {"outcome": outcome, "verified": outcome in {REPAIR, PRESERVE}, "reason": reason, "reached": sorted(reached)}
    verified = bool(rows) and all(row.get("verified") is True for row in rows.values())
    return {
        "schema": SCHEMA, "outcomes": rows, "behavior_complete": verified,
        "regression_green": all(row.get("outcome") in {REPAIR, PRESERVE}
                                 for row in rows.values()),
        "artifact": "STORED" if candidate_identity else "NOT_STORED",
        "verification": "VERIFIED" if verified else "UNVERIFIED",
        "acceptance": "ACCEPTED" if verified else "UNACCEPTED",
        "certification": "CERTIFIED" if verified else "UNCERTIFIED",
    }


def terminal_projection(settlement: Mapping[str, object]) -> dict[str, object]:
    """Keep artifact, verification, acceptance, and certification orthogonal."""
    acceptance = _text(settlement.get("acceptance")) or "UNACCEPTED"
    certification = _text(settlement.get("certification")) or "UNCERTIFIED"
    artifact = _text(settlement.get("artifact")) or "NOT_STORED"
    verified = _text(settlement.get("verification")) == "VERIFIED"
    certified = certification == "CERTIFIED"
    disposition = _text(settlement.get("disposition"))
    if not disposition:
        disposition = "PASS" if certified else (
            UNCERTIFIED_SUBMIT if artifact == "STORED" and acceptance == "ACCEPTED"
            else "OPEN"
        )
    return {
        "artifact": artifact,
        "verification": _text(settlement.get("verification")) or "UNVERIFIED",
        "acceptance": acceptance,
        "certification": certification,
        "accepted": acceptance == "ACCEPTED",
        "verified": verified,
        "certified": certified,
        "disposition": disposition,
    }


class AdvisoryEpisodeLedger:
    """Ownerless DECIDE2 episode dedupe; it never executes a suggestion."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def open(self, *, predicate: object, evidence: object, capability: object,
             revision: object, options: Sequence[object]) -> dict[str, object]:
        offered = tuple(dict.fromkeys(_text(item) for item in options if _text(item)))
        key = _digest({"predicate": predicate, "evidence": evidence, "capability": capability, "revision": revision})
        if len(offered) <= 1:
            return {"status": "SKIPPED", "reason": "deterministic_or_single_route", "key": key, "options": list(offered)}
        if key in self._seen:
            return {"status": "DEDUPED", "reason": "unchanged_episode", "key": key, "options": list(offered)}
        self._seen.add(key)
        return {"status": "OPEN", "reason": "consequential_ownerless_choice", "key": key, "options": list(offered), "execute": False}


@dataclass(frozen=True)
class RecoveryRoute:
    key: str
    error: str
    route: str
    replayed: bool

    def as_dict(self) -> dict[str, object]:
        return {"key": self.key, "error": self.error, "route": self.route, "replayed": self.replayed}


class HostFailureRecovery:
    def __init__(self) -> None:
        self._failures: dict[str, RecoveryRoute] = {}

    def observe(self, *, request_id: object, stage: object, revision: object, error: object) -> RecoveryRoute:
        # Error text is diagnostic payload, not identity. One request/stage/
        # revision gets one executable recovery route and one accounting event.
        key = _digest({"request_id": request_id, "stage": stage, "revision": revision})
        existing = self._failures.get(key)
        if existing is not None:
            return RecoveryRoute(key, existing.error, existing.route, True)
        route = RecoveryRoute(key, _text(error)[:240], "read", False)
        self._failures[key] = route
        return route


class AssumptionCache:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def put(self, *, query: object, source_revision: object, index_revision: object, value: object) -> None:
        self._values[_digest({"query": query, "source_revision": source_revision, "index_revision": index_revision})] = value

    def get(self, *, query: object, source_revision: object, index_revision: object) -> object | None:
        return self._values.get(_digest({"query": query, "source_revision": source_revision, "index_revision": index_revision}))


HOST_COST_PROFILE_SECONDS = 312.0


class HostCostMemo:
    """Bounded offline host-cost check keyed by every identity that can invalidate it."""

    def __init__(self, *, budget_seconds: float = HOST_COST_PROFILE_SECONDS) -> None:
        self.budget_seconds = float(budget_seconds)
        self._receipts: dict[str, dict[str, object]] = {}

    def check(self, *, revision: object, content_identity: object,
              environment_identity: object, elapsed_seconds: object) -> dict[str, object]:
        key = _digest({"revision": revision, "content": content_identity,
                       "environment": environment_identity})
        cached = self._receipts.get(key)
        if cached is not None:
            return {**cached, "cached": True}
        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError):
            elapsed = self.budget_seconds + 1.0
        status = "PASS" if 0.0 <= elapsed <= self.budget_seconds else "HOST_COST_EXCEEDED"
        receipt = {
            "schema": "stateframe-host-cost/1", "key": key,
            "revision": str(revision), "content_identity": str(content_identity),
            "environment_identity": str(environment_identity),
            "elapsed_seconds": elapsed, "budget_seconds": self.budget_seconds,
            "status": status, "cached": False,
        }
        self._receipts[key] = receipt
        return dict(receipt)


def provider_boundary_receipt(clients: Sequence[Mapping[str, object]], *, reasoning_off: bool) -> dict[str, object]:
    """Prove all shared-policy client constructions use one effective boundary."""
    rows = tuple(item for item in clients if isinstance(item, Mapping))
    policies = {_text(item.get("transport_policy")) for item in rows}
    effective_reasoning = all(item.get("reasoning") in {False, "off", "OFF"} for item in rows)
    return {
        "schema": "stateframe-provider-boundary/1", "client_count": len(rows),
        "shared_transport_policy": len(policies) == 1 and "" not in policies,
        "transport_policies": sorted(policies),
        "reasoning_off": bool(reasoning_off) and effective_reasoning,
        "status": "PASS" if len(rows) == 3 and len(policies) == 1 and "" not in policies
        and bool(reasoning_off) and effective_reasoning else "UNKNOWN",
    }


def _typed_literal(value: str) -> tuple[str, object]:
    if value.casefold() in {"true", "false"}:
        return "boolean", value.casefold() == "true"
    if value[:1] in {"'", '"'}:
        return "string", value[1:-1]
    try:
        return "number", float(value) if "." in value else int(value)
    except ValueError:
        return "enum", value


def _code_prefix(line: str) -> str:
    """Exclude comment text while leaving comment markers inside strings intact."""
    quote = ""
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char == "#" or line.startswith(("//", "/*", "<!--"), index):
            return line[:index]
        index += 1
    return line


def typed_literal_changes(before: str, after: str) -> list[dict[str, object]]:
    """Find one typed value delta per otherwise unchanged source line."""
    changes: list[dict[str, object]] = []
    for line_number, (old, new) in enumerate(zip(before.splitlines(), after.splitlines()), 1):
        if old.lstrip().startswith(("#", "//", "/*", "*", "<!--")) \
                and new.lstrip().startswith(("#", "//", "/*", "*", "<!--")):
            continue
        old_code, new_code = _code_prefix(old), _code_prefix(new)
        old_tokens, new_tokens = list(_TOKEN.finditer(old_code)), list(_TOKEN.finditer(new_code))
        if len(old_tokens) != len(new_tokens) or old_code == new_code:
            continue
        old_shape = _TOKEN.sub("<value>", old_code).strip()
        new_shape = _TOKEN.sub("<value>", new_code).strip()
        if old_shape != new_shape:
            continue
        for old_match, new_match in zip(old_tokens, new_tokens):
            old_raw, new_raw = old_match.group(0), new_match.group(0)
            if old_raw == new_raw:
                continue
            old_type, old_value = _typed_literal(old_raw)
            new_type, new_value = _typed_literal(new_raw)
            names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", new_code[:new_match.start()])
            changes.append({"line": line_number, "name": names[-1] if names else "",
                            "old": old_value, "new": new_value, "type": old_type,
                            "old_type": old_type, "new_type": new_type,
                            "old_raw": old_raw, "new_raw": new_raw,
                            "old_line": old, "new_line": new})
    return changes


def authorize_typed_change(change: Mapping[str, object], *, receipt: Mapping[str, object] | None = None,
                           host_checked_value: Mapping[str, object] | None = None) -> bool:
    """Require a receipt or host check for the proposed new typed value."""
    new = change.get("new")
    name = _text(change.get("name")).casefold()
    value_type = _text(change.get("type"))
    for source in (receipt, host_checked_value):
        if not isinstance(source, Mapping):
            continue
        if source.get("integrity") is not True and not source.get("receipt_id"):
            continue
        if source.get("authorized") is False:
            continue
        allowed = source.get("new_value", source.get("value"))
        subject = _text(source.get("subject", source.get("name"))).casefold()
        subject_matches = not name or name in {subject, subject.rsplit(".", 1)[-1]}
        type_matches = not source.get("type") or _text(source.get("type")) == value_type
        new_type_matches = not source.get("new_type") or _text(source.get("new_type")) == _text(change.get("new_type", value_type))
        if subject_matches and type_matches and new_type_matches and (allowed == new or str(allowed) == str(new)):
            return True
    return False


def smoke_identity(*, parent_identity: object, checked_dependencies: Sequence[object],
                   source_identity: object, environment_identity: object, current_digest: object) -> str:
    return _digest({"parent": parent_identity, "dependencies": list(checked_dependencies),
                    "source": source_identity, "environment": environment_identity, "current": current_digest})


def classify_smoke(*, failure: object, baseline_failure: object | None, identity: str,
                   baseline_identity: str | None) -> str:
    if baseline_identity is None or baseline_identity != identity:
        return "STALE_BASELINE"
    if not failure:
        return "PASS"
    if baseline_failure and _text(failure) == _text(baseline_failure):
        return ENVIRONMENT
    return "FAIL"


def negative_search_status(*, complete: bool, found: bool) -> str:
    """An incomplete negative index cannot prove absence."""
    if found:
        return "RESOLVED"
    return "ABSENT" if complete else UNKNOWN


def compose_free_loop_capture(
        *, context: RequestContext, original_tools: Sequence[object],
        provisional_tools: Sequence[object], final_tools: Sequence[object],
        source_delivery: Mapping[str, object],
        requirement_resolutions: Sequence[Mapping[str, object]] = (),
        obligations: Mapping[str, Mapping[str, object]] | None = None,
        receipts: Sequence[object] = (),
        revision: str = "",
        parent_identity: str = "",
        candidate_identity: str = "",
        execution_store: object = None,
        ticket_receipt: Mapping[str, object] | None = None,
        provider_boundary: Mapping[str, object] | None = None,
        host_cost: Mapping[str, object] | None = None,
        source_config_hash: str = "",
        offered_actions: Sequence[object] = (),
        reasoning_off: bool = True,
        model_calls: int = 0,
        hard_limit: Mapping[str, object] | None = None,
        recovery: Mapping[str, object] | None = None,
        scope: Mapping[str, object] | None = None,
        smoke: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compose one zero-model/free-loop proof without becoming an authority.

    Every input is an owner-produced receipt or an already frozen wire surface;
    this reducer only exposes mismatches and a truthful terminal disposition.
    """

    boundary = request_boundary_receipt(
        original_tools, provisional_tools, final_tools,
        protected_surface=tuple(capability_routes(final_tools).values()),
        hard_limit=hard_limit,
        offered_actions=offered_actions,
    )
    final_names = set(boundary["final_tools"])
    source_status = _text(
        source_delivery.get("final_delivery_status", source_delivery.get("status", "UNKNOWN"))
    ).upper()
    source_complete = source_status in {"COMPLETE", "DELIVERED", "VERIFIED"}
    if not source_complete:
        try:
            from kernel_request_source_window import source_delivery_satisfied
            source_complete = source_delivery_satisfied(source_delivery)
        except (ImportError, TypeError, ValueError):
            source_complete = False
    ticket = ticket_receipt if isinstance(ticket_receipt, Mapping) else {}
    ticket_provenance = bool(
        _text(ticket.get("source_id") or ticket.get("ticket_source_id") or ticket.get("id"))
        and _text(ticket.get("revision") or ticket.get("source_revision"))
        and _text(ticket.get("exact_clause") or ticket.get("quote") or ticket.get("span"))
    )
    provider_ready = (
        isinstance(provider_boundary, Mapping)
        and _text(provider_boundary.get("status")).upper() in {"PASS", "VALID"}
    )
    cost_ready = (
        host_cost is None
        or _text(host_cost.get("status")).upper() in {"PASS", "CACHED", "MEASURED"}
        or (isinstance(host_cost.get("value"), Mapping)
            and _text(host_cost["value"].get("status")).upper() in {"PASS", "CACHED"})
    ) if isinstance(host_cost, Mapping) else host_cost is None
    unresolved = [row for row in requirement_resolutions
                  if isinstance(row, Mapping) and _text(row.get("status")).upper() not in {"RESOLVED", ""}]
    settlement = settle_obligations(
        obligations if isinstance(obligations, Mapping) else {}, receipts,
        revision=revision, parent_identity=parent_identity, candidate_identity=candidate_identity,
        execution_store=execution_store,
    ) if obligations else {
        "behavior_complete": False, "regression_green": False,
        "artifact": "NOT_STORED", "verification": "UNVERIFIED",
        "acceptance": "UNACCEPTED", "certification": "UNCERTIFIED",
    }
    terminal = terminal_projection(settlement)
    organs = {
        "request_budget": boundary["status"] == "READY",
        "source_closure": source_complete,
        "requirements": bool(requirement_resolutions) and not unresolved,
        "witness": bool(obligations) and bool(settlement.get("behavior_complete")),
        "completion": terminal["disposition"] == "PASS",
        "recovery": isinstance(recovery, Mapping) and bool(recovery.get("route")),
        "scope": isinstance(scope, Mapping)
        and _text(scope.get("status", "SUPPORTED")).upper() not in {"SCOPE_UNSUPPORTED", "BLOCKED"},
        "smoke": isinstance(smoke, Mapping)
        and _text(smoke.get("status", "UNKNOWN")).upper() in {"PASS", ENVIRONMENT},
        "ticket_provenance": ticket_provenance,
        "provider_boundary": provider_ready,
        "host_cost": cost_ready,
    }
    failures: list[str] = []
    if context.mode != FREE_MODE:
        failures.append("effective_mode_not_free")
    if model_calls != 0:
        failures.append("model_call_observed")
    if not reasoning_off:
        failures.append("reasoning_not_off")
    if not ticket_provenance:
        failures.append("ticket_provenance_missing")
    if not provider_ready:
        failures.append("provider_boundary_missing")
    if host_cost is not None and not cost_ready:
        failures.append("host_cost_missing_or_exceeded")
    if not boundary["menu_admission_equal"]:
        failures.append("advisory_menu_not_dispatchable")
    if not final_names:
        failures.append("empty_final_menu")
    if boundary["status"] != "READY":
        failures.append("request_boundary_blocked")
    if unresolved:
        failures.append("requirement_open")
    if not source_complete:
        failures.append("source_delivery_incomplete")
    if not all(organs.values()):
        failures.extend(name for name, ready in organs.items() if not ready and name not in failures)
    disposition = "PASS" if not failures else (
        UNCERTIFIED_SUBMIT if terminal["artifact"] == "STORED" else "OPEN"
    )
    removal_telemetry = {
        "original_to_final": [row for row in boundary.get("removed_tools", [])
                              if isinstance(row, Mapping)],
        "removed_count": len(boundary.get("removed_tools", ())),
    }
    release_proof = None
    if host_cost is not None or source_config_hash:
        from kernel_aha10_r3_contract import compose_release_proof
        release_proof = compose_release_proof(
            budget={"status": boundary["status"]},
            witness={"status": "VALID" if settlement.get("behavior_complete") else "UNKNOWN"},
            authored_check={"status": "VALID" if settlement.get("behavior_complete") else "UNKNOWN"},
            delivery={"status": "COMPLETE" if source_complete else source_status},
            settlement=settlement,
            smoke=smoke if isinstance(smoke, Mapping) else {"status": "UNKNOWN"},
            provider=provider_boundary if isinstance(provider_boundary, Mapping) else {"status": "UNKNOWN"},
            cost=host_cost if isinstance(host_cost, Mapping) else {"status": "UNKNOWN"},
            scope=scope if isinstance(scope, Mapping) else {"status": "UNKNOWN"},
            advisory={"status": "VALID"},
            source_config_hash=source_config_hash,
        )
        if _text(release_proof.get("status")).upper() != "VALID":
            failures.append("release_proof_incomplete")
            disposition = UNCERTIFIED_SUBMIT if terminal["artifact"] == "STORED" else "OPEN"
    return {
        "schema": "stateframe-aha10-composed-free-loop/1",
        "effective_mode": context.mode,
        "captured_stage": context.stage,
        "ticket": dict(ticket_receipt) if isinstance(ticket_receipt, Mapping) else None,
        "reasoning_off": reasoning_off,
        "model_calls": model_calls,
        "final_tools": list(boundary["final_tools"]),
        "final_admission": list(boundary["final_tools"]),
        "boundary": boundary,
        "removal_telemetry": removal_telemetry,
        "source_delivery": dict(source_delivery),
        "requirement_resolutions": [dict(row) for row in requirement_resolutions if isinstance(row, Mapping)],
        "settlement": settlement,
        "terminal": terminal,
        "organs": organs,
        "recovery": dict(recovery) if isinstance(recovery, Mapping) else None,
        "scope": dict(scope) if isinstance(scope, Mapping) else None,
        "smoke": dict(smoke) if isinstance(smoke, Mapping) else None,
        "host_cost": dict(host_cost) if isinstance(host_cost, Mapping) else None,
        "source_config_hash": source_config_hash,
        "release_proof": release_proof,
        "failures": list(dict.fromkeys(failures)),
        "disposition": disposition,
        "status": "PASS" if disposition == "PASS" else "UNCERTIFIED_SUBMIT" if disposition == UNCERTIFIED_SUBMIT else "OPEN",
    }


__all__ = [
    "AdvisoryEpisodeLedger", "AssumptionCache", "ENVIRONMENT", "FREE_MODE", "HostCostMemo",
    "HOST_COST_PROFILE_SECONDS", "HostFailureRecovery",
    "PRESERVE", "REPAIR", "RecoveryRoute", "RequestContext", "SCHEMA", "SourceUnit", "STAGED_MODE",
    "InventoryMatch", "PARTIAL", "UNCERTIFIED_SUBMIT", "capability_routes", "compose_free_loop_capture",
    "UNCOVERED", "UNKNOWN", "UNRUNNABLE", "authorize_typed_change", "bounded_source_delivery",
    "classify_smoke", "free_diet",
    "negative_search_status",
    "provider_boundary_receipt", "request_boundary_receipt", "request_context", "resolve_requirement_clause",
    "settle_obligations", "smoke_identity",
    "source_unit_fixtures", "source_units", "inventory_matches", "inventory_coverage",
    "terminal_projection", "typed_literal_changes",
    "validate_solver_check_receipt",
]
