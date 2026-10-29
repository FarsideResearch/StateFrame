"""Shadow-mode pattern-card diagnostics.

The pattern-card layer is deliberately a compiler, not a completion gate.  It
turns deterministic facts about one source revision into a bounded record of
what a host *would* surface.  It never changes a work record, a completion
decision, a question answer, or a source file.

The starter cards live in :mod:`src/pattern_cards` as JSON.  This module
owns the typed representation, the small three-valued predicate interpreter,
the AST change extractor, and the shadow router.  Existing host receipts and
question records are imported rather than reimplemented; a diagnostic receipt
keeps the original ``CheckReceipt`` as its host witness and a question route
keeps the existing ``QuestionSpec``/``Question`` pair.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

try:  # The repository runs its kernels with ``src`` on PYTHONPATH.
    from kernel_obligation_assess import (
        COUNTEREXAMPLE,
        CheckDefinition,
        CheckReceipt,
        ObligationAssessment,
        ObligationSpec,
        PASS,
        SATISFIED,
        UNKNOWN as OBLIGATION_UNKNOWN,
        UNSATISFIED,
        digest as host_digest,
        run_check,
    )
    from kernel_questions import (
        AnswerKind,
        MODEL_TYPED,
        Question,
        QuestionSpec,
    )
    from kernel_post_edit_checks import _source_tree
except ImportError:  # pragma: no cover - package-style import fallback
    from .kernel_obligation_assess import (
        COUNTEREXAMPLE,
        CheckDefinition,
        CheckReceipt,
        ObligationAssessment,
        ObligationSpec,
        PASS,
        SATISFIED,
        UNKNOWN as OBLIGATION_UNKNOWN,
        UNSATISFIED,
        digest as host_digest,
        run_check,
    )
    from .kernel_questions import AnswerKind, MODEL_TYPED, Question, QuestionSpec
    from .kernel_post_edit_checks import _source_tree


__all__ = [
    "ABSTAIN",
    "AnswerKind",
    "BoundPatternInstance",
    "BudgetSpec",
    "CANDIDATE",
    "ChangeEvent",
    "DiagnosticReceipt",
    "EMIT_OBSERVATION",
    "EMIT_QUESTION",
    "ENABLED",
    "EventKind",
    "FRESH",
    "FALSE",
    "InterventionRecord",
    "Lifecycle",
    "MATERIAL_EVIDENCE",
    "PatternCard",
    "PatternCompiler",
    "PredicateAST",
    "PUBLIC_TESTS_BECAME_GREEN",
    "DEPENDENCY_TRACEBACK",
    "RevisionFacts",
    "Route",
    "SCOPED_OBSERVATION",
    "SHADOW",
    "STALE",
    "SUBMIT_EVIDENCE",
    "SourcePair",
    "TriState",
    "TRUE",
    "TYPED_QUESTION",
    "UNKNOWN",
    "VALIDATION_PATH_CHANGED",
    "INPUT_ALIAS_WRITE_ADDED",
    "OPTIONAL_FIELD_INTRODUCED",
    "PUBLIC_EXPORT_REMOVED",
    "SIGNATURE_CHANGED",
    "RETIRED",
    "bind_card",
    "build_revision_facts",
    "compile_pattern_cards",
    "default_pattern_cards",
    "extract_change_events",
    "load_pattern_cards",
    "run_bound_diagnostics",
    "replay_journal",
    "shadow_evaluate",
    "PATTERN_CARDS_ENV",
    "pattern_cards_mode",
    "pattern_cards_live",
    "render_live_state",
]


class Lifecycle(StrEnum):
    CANDIDATE = "CANDIDATE"
    SHADOW = "SHADOW"
    ENABLED = "ENABLED"
    RETIRED = "RETIRED"


CANDIDATE = Lifecycle.CANDIDATE
SHADOW = Lifecycle.SHADOW
ENABLED = Lifecycle.ENABLED
RETIRED = Lifecycle.RETIRED


class TriState(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_value(cls, value: object) -> "TriState":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.TRUE if value else cls.FALSE
        text = str(value or "").strip().upper()
        if text in {"TRUE", "YES", "PASS", "1"}:
            return cls.TRUE
        if text in {"FALSE", "NO", "FAIL", "0"}:
            return cls.FALSE
        return cls.UNKNOWN


TRUE = TriState.TRUE
FALSE = TriState.FALSE
UNKNOWN = TriState.UNKNOWN


class EventKind(StrEnum):
    SIGNATURE_CHANGED = "SIGNATURE_CHANGED"
    PUBLIC_EXPORT_REMOVED = "PUBLIC_EXPORT_REMOVED"
    INPUT_ALIAS_WRITE_ADDED = "INPUT_ALIAS_WRITE_ADDED"
    VALIDATION_PATH_CHANGED = "VALIDATION_PATH_CHANGED"
    OPTIONAL_FIELD_INTRODUCED = "OPTIONAL_FIELD_INTRODUCED"
    PUBLIC_TESTS_BECAME_GREEN = "PUBLIC_TESTS_BECAME_GREEN"
    DEPENDENCY_TRACEBACK = "DEPENDENCY_TRACEBACK"


SIGNATURE_CHANGED = EventKind.SIGNATURE_CHANGED
PUBLIC_EXPORT_REMOVED = EventKind.PUBLIC_EXPORT_REMOVED
INPUT_ALIAS_WRITE_ADDED = EventKind.INPUT_ALIAS_WRITE_ADDED
VALIDATION_PATH_CHANGED = EventKind.VALIDATION_PATH_CHANGED
OPTIONAL_FIELD_INTRODUCED = EventKind.OPTIONAL_FIELD_INTRODUCED
PUBLIC_TESTS_BECAME_GREEN = EventKind.PUBLIC_TESTS_BECAME_GREEN
DEPENDENCY_TRACEBACK = EventKind.DEPENDENCY_TRACEBACK


class Effect(StrEnum):
    EMIT_QUESTION = "EMIT_QUESTION"
    EMIT_OBSERVATION = "EMIT_OBSERVATION"
    SUBMIT_EVIDENCE = "SUBMIT_EVIDENCE"


EMIT_QUESTION = Effect.EMIT_QUESTION
EMIT_OBSERVATION = Effect.EMIT_OBSERVATION
SUBMIT_EVIDENCE = Effect.SUBMIT_EVIDENCE


class OracleRequirement(StrEnum):
    NONE = "NONE"
    EXISTING_OBLIGATION = "EXISTING_OBLIGATION"
    ACCEPTED_PROJECT_INVARIANT = "ACCEPTED_PROJECT_INVARIANT"


class Route(StrEnum):
    DIAGNOSTIC_RECEIPT = "diagnostic_receipt"
    SCOPED_OBSERVATION = "scoped_observation"
    TYPED_QUESTION = "typed_question"
    ABSTAIN = "abstain"


DIAGNOSTIC_RECEIPT = Route.DIAGNOSTIC_RECEIPT
SCOPED_OBSERVATION = Route.SCOPED_OBSERVATION
TYPED_QUESTION = Route.TYPED_QUESTION
ABSTAIN = Route.ABSTAIN


class Freshness(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


FRESH = Freshness.FRESH
STALE = Freshness.STALE


MATERIAL_EVIDENCE = "MATERIAL_EVIDENCE"

PATTERN_CARDS_ENV = "STATEFRAME_PATTERN_CARDS"


def pattern_cards_mode() -> str:
    """Resolve the pattern-card presentation mode.

    Existing callers remain shadow-only when the variable is absent.  ``off``
    is an explicit no-op arm; ``live`` is the only mode that changes a runtime
    frame.  ``shadow`` keeps the existing offline compiler behavior.
    """
    value = os.environ.get(PATTERN_CARDS_ENV, "shadow").strip().lower()
    if value in {"0", "off", "false"}:
        return "off"
    if value in {"shadow", "observe"}:
        return "shadow"
    if value in {"1", "on", "live"}:
        return "live"
    raise ValueError(f"{PATTERN_CARDS_ENV} must be off, shadow, or live")


def pattern_cards_live() -> bool:
    return pattern_cards_mode() == "live"


def _jsonable(value: object) -> object:
    """Make typed local records stable without invoking arbitrary code."""

    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: str(item))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:  # pragma: no cover - foreign record defensive path
            pass
    if hasattr(value, "as_dict") and callable(value.as_dict):
        try:
            return _jsonable(value.as_dict())
        except Exception:  # pragma: no cover - foreign record defensive path
            pass
    if hasattr(value, "__dict__"):
        try:
            return _jsonable(vars(value))
        except TypeError:  # pragma: no cover - slots-only opaque value
            pass
    return str(value)


def _canonical(value: object) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def stable_digest(value: object) -> str:
    """Return the same ``sha256:`` shape used by the host receipt seam."""

    try:
        return str(host_digest(_jsonable(value)))
    except Exception:  # pragma: no cover - import seam compatibility
        return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _identity(value: object) -> str:
    if isinstance(value, str) and value.startswith("sha256:"):
        return value
    return stable_digest(value)


def _text(value: object, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def _values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _record_values(value: object) -> tuple[object, ...]:
    """Normalize either a record sequence or an id -> record mapping."""

    if isinstance(value, Mapping):
        record_keys = {
            "id", "obligation_id", "clause", "text", "source_text", "check_id",
            "status", "verdict", "pattern_card",
        }
        if record_keys.intersection(value):
            return (value,)
        return tuple(value.values())
    return _values(value)


def _strings(value: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_text(item, 300) for item in _values(value) if _text(item)))


def _event_kind(value: object) -> EventKind | None:
    if isinstance(value, EventKind):
        return value
    try:
        return EventKind(str(value).strip().upper())
    except ValueError:
        return None


def _effect(value: object) -> Effect:
    if isinstance(value, Effect):
        return value
    text = str(value or "").strip().upper()
    try:
        return Effect(text)
    except ValueError as error:
        raise ValueError(f"unknown pattern-card effect: {value!r}") from error


def _oracle(value: object) -> OracleRequirement:
    if isinstance(value, OracleRequirement):
        return value
    text = str(value or "NONE").strip().upper()
    try:
        return OracleRequirement(text)
    except ValueError as error:
        raise ValueError(f"unknown oracle requirement: {value!r}") from error


@dataclass(frozen=True, slots=True)
class SourcePair:
    """One named before/after source unit."""

    path: str
    before: str | None
    after: str | None

    @property
    def changed(self) -> bool:
        return self.before != self.after

    @property
    def revision(self) -> str:
        return stable_digest({"path": self.path, "after": self.after})

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "before": self.before, "after": self.after}


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    """A typed fact extracted from one source revision transition."""

    kind: EventKind | str
    path: str = ""
    subject: str = "*"
    symbol: str = ""
    certainty: TriState | str = TRUE
    before: object = None
    after: object = None
    details: Mapping[str, object] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = _event_kind(self.kind)
        if kind is None:
            raise ValueError(f"unknown change-event kind: {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "certainty", TriState.from_value(self.certainty))
        object.__setattr__(self, "path", _text(self.path, 300))
        subject = _text(self.subject or self.symbol or self.path or "*", 300)
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "symbol", _text(self.symbol, 240))
        object.__setattr__(self, "details", copy.deepcopy(dict(self.details)))
        object.__setattr__(self, "evidence_refs", _strings(self.evidence_refs))

    @property
    def status(self) -> TriState:
        return self.certainty

    @property
    def truth(self) -> TriState:
        return self.certainty

    @property
    def predicate(self) -> str:
        return _text(self.details.get("predicate", self.kind.value), 200)

    @property
    def input_partition(self) -> str:
        return _text(self.details.get("input_partition", "all"), 120) or "all"

    @property
    def boundary(self) -> str:
        return _text(self.details.get("boundary", self.path or "public"), 160) or "public"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ChangeEvent":
        return cls(
            value.get("kind", ""),
            str(value.get("path", "")),
            str(value.get("subject", value.get("symbol", "*"))),
            str(value.get("symbol", "")),
            value.get("certainty", value.get("status", TRUE)),
            value.get("before"), value.get("after"),
            value.get("details", {}), value.get("evidence_refs", ()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "subject": self.subject,
            "symbol": self.symbol,
            "certainty": self.certainty.value,
            "before": _jsonable(self.before),
            "after": _jsonable(self.after),
            "details": _jsonable(self.details),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class RevisionFacts:
    """Immutable facts supplied to predicates and the router."""

    sources: tuple[SourcePair, ...] = ()
    events: tuple[ChangeEvent, ...] = ()
    obligations: tuple[object, ...] = ()
    receipts: object = ()
    before_receipts: object = ()
    after_receipts: object = ()
    revision: str = ""
    before_revision: str = ""
    fixture: object = None
    environment: object = None
    step: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "obligations", tuple(self.obligations))
        revision = self.revision or stable_digest({
            "sources": [source.to_dict() for source in self.sources],
            "events": [event.to_dict() for event in self.events],
        })
        object.__setattr__(self, "revision", _identity(revision))
        if self.before_revision:
            object.__setattr__(self, "before_revision", _identity(self.before_revision))
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))
        object.__setattr__(self, "step", max(0, int(self.step)))

    @property
    def changed(self) -> bool:
        return any(source.changed for source in self.sources)

    @property
    def material_digest(self) -> str:
        return stable_digest({
            "revision": self.revision,
            "events": [event.to_dict() for event in self.events],
            "receipts": _jsonable(self.receipts),
            "before_receipts": _jsonable(self.before_receipts),
            "after_receipts": _jsonable(self.after_receipts),
        })

    def event(self, kind: EventKind | str) -> tuple[ChangeEvent, ...]:
        target = _event_kind(kind)
        return tuple(event for event in self.events if event.kind == target)

    def to_dict(self) -> dict[str, object]:
        return {
            "sources": [source.to_dict() for source in self.sources],
            "events": [event.to_dict() for event in self.events],
            "obligations": [_jsonable(item) for item in self.obligations],
            "receipts": _jsonable(self.receipts),
            "revision": self.revision,
            "before_revision": self.before_revision,
            "fixture": _jsonable(self.fixture),
            "environment": _jsonable(self.environment),
            "step": self.step,
            "metadata": _jsonable(self.metadata),
        }


def _source_content(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("content", "source", "text", "bytes"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return None


def _source_mapping(value: object) -> dict[str, str | None] | None:
    if not isinstance(value, Mapping):
        return None
    # A single source record is accepted in addition to path -> content.
    if any(key in value for key in ("content", "source", "text", "bytes")):
        path = str(value.get("path", "<source>"))
        return {path: _source_content(value)}
    result: dict[str, str | None] = {}
    for key, item in value.items():
        content = _source_content(item)
        if content is None and isinstance(item, str):
            content = item
        result[str(key)] = content
    return result


def _source_pairs(before: object, after: object) -> tuple[SourcePair, ...]:
    before_map = _source_mapping(before)
    after_map = _source_mapping(after)
    if before_map is not None or after_map is not None:
        left = before_map or {}
        right = after_map or {}
        return tuple(
            SourcePair(path, left.get(path), right.get(path))
            for path in sorted(set(left) | set(right))
        )
    left = _source_content(before)
    right = _source_content(after)
    return (SourcePair("<source>", left, right),)


def _parse(source: str | None) -> ast.Module | None:
    if not isinstance(source, str):
        return None
    # ``_source_tree`` is the repository's deterministic parser helper.  The
    # fallback keeps this module usable in a package import during isolated
    # probe collection.
    try:
        return _source_tree(source)
    except (SyntaxError, TypeError, ValueError):
        return None


def _definition_nodes(tree: ast.Module | None) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    if tree is None:
        return {}
    result: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = {}

    def visit(body: Iterable[ast.stmt], prefix: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = ".".join((*prefix, node.name))
                result[name] = node
                visit(node.body, (*prefix, node.name))

    visit(tree.body)
    return result


def _function_nodes(tree: ast.Module | None) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        name: node for name, node in _definition_nodes(tree).items()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _dump(node: ast.AST | None) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False) if node is not None else ""


def _argument_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[object, ...]:
    args = node.args
    positional = (*args.posonlyargs, *args.args)
    defaults_start = len(positional) - len(args.defaults)
    positional_rows = tuple(
        (item.arg, index >= defaults_start, _dump(args.defaults[index - defaults_start]) if index >= defaults_start else "")
        for index, item in enumerate(positional)
    )
    keyword_rows = tuple(
        (item.arg, default is not None, _dump(default))
        for item, default in zip(args.kwonlyargs, args.kw_defaults)
    )
    return (
        positional_rows,
        keyword_rows,
        args.vararg.arg if args.vararg is not None else None,
        args.kwarg.arg if args.kwarg is not None else None,
    )


def _public_exports(tree: ast.Module | None) -> set[str]:
    if tree is None:
        return set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            names = {
                item.value for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            return names
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            name = next((target.id for target in targets if isinstance(target, ast.Name)), "")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            name = ""
            if isinstance(node, ast.Import):
                name = node.names[0].asname or node.names[0].name.split(".", 1)[0]
            elif node.names and node.names[0].name != "*":
                name = node.names[0].asname or node.names[0].name
        else:
            name = ""
        if name and not name.startswith("_"):
            names.add(name)
    return names


def _base_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


def _parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    return {
        item.arg for item in (*args.posonlyargs, *args.args, *args.kwonlyargs)
    } | ({args.vararg.arg} if args.vararg is not None else set()) | ({args.kwarg.arg} if args.kwarg is not None else set())


def _alias_writes(tree: ast.Module | None) -> dict[str, set[tuple[str, str, str]]]:
    """Find writes through a function parameter or a direct local alias."""

    result: dict[str, set[tuple[str, str, str]]] = {}
    for qualified, function in _function_nodes(tree).items():
        aliases = set(_parameter_names(function))
        writes: set[tuple[str, str, str]] = set()

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if node is not function:
                    return
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                if node is not function:
                    return
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                if isinstance(node.value, ast.Name) and node.value.id in aliases:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            aliases.add(target.id)
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and _base_name(target.value) in aliases:
                        writes.add((_base_name(target.value) or "*", "setitem", _dump(target.slice)))
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                if isinstance(node.value, ast.Name) and node.value.id in aliases and isinstance(node.target, ast.Name):
                    aliases.add(node.target.id)
                if isinstance(node.target, ast.Subscript) and _base_name(node.target.value) in aliases:
                    writes.add((_base_name(node.target.value) or "*", "setitem", _dump(node.target.slice)))
                self.generic_visit(node)

            def visit_AugAssign(self, node: ast.AugAssign) -> None:
                if isinstance(node.target, ast.Subscript) and _base_name(node.target.value) in aliases:
                    writes.add((_base_name(node.target.value) or "*", "setitem", _dump(node.target.slice)))
                self.generic_visit(node)

            def visit_Delete(self, node: ast.Delete) -> None:
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and _base_name(target.value) in aliases:
                        writes.add((_base_name(target.value) or "*", "del", _dump(target.slice)))
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Attribute) and _base_name(node.func.value) in aliases:
                    if node.func.attr in {"pop", "append"}:
                        argument = _dump(node.args[0]) if node.args else ""
                        writes.add((_base_name(node.func.value) or "*", node.func.attr, argument))
                self.generic_visit(node)

        Visitor().visit(function)
        if writes:
            result[qualified] = writes
    return result


_VALIDATION_WORDS = re.compile(r"(?:valid|check|ensure|assert|require|guard|parse|sanitize|schema)", re.I)


def _validation_paths(tree: ast.Module | None) -> dict[str, tuple[str, ...]]:
    if tree is None:
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for qualified, function in _function_nodes(tree).items():
        calls: list[str] = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            name = _base_name(node.func) or ""
            if _VALIDATION_WORDS.search(name):
                calls.append(f"{name}({_dump(node.args[0]) if node.args else ''})")
        if calls:
            result[qualified] = tuple(calls)
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names if _VALIDATION_WORDS.search(alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _VALIDATION_WORDS.search(module):
                imports.append(module)
    if imports:
        result["<imports>"] = tuple(imports)
    return result


def _optional_fields(tree: ast.Module | None) -> set[tuple[str, str, str]]:
    if tree is None:
        return set()
    result: set[tuple[str, str, str]] = set()
    for qualified, function in _function_nodes(tree).items():
        args = function.args
        positional = (*args.posonlyargs, *args.args)
        defaults_start = len(positional) - len(args.defaults)
        for index, item in enumerate(positional):
            if index >= defaults_start:
                result.add((qualified, "parameter", item.arg))
        for item, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                result.add((qualified, "parameter", item.arg))
        for node in ast.walk(function):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "get":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                continue
            field_name = node.args[0].value
            default = _dump(node.args[1]) if len(node.args) > 1 else "<none>"
            result.add((qualified, "field", f"{field_name}:{default}"))
    return result


def _unknown_event(kind: EventKind, pair: SourcePair, reason: str) -> ChangeEvent:
    return ChangeEvent(
        kind=kind,
        path=pair.path,
        subject=pair.path,
        certainty=UNKNOWN,
        before=pair.before,
        after=pair.after,
        details={"reason": reason, "predicate": kind.value},
    )


def _test_status(value: object) -> str:
    if isinstance(value, CheckReceipt):
        if value.status == PASS:
            return "GREEN"
        if value.status == COUNTEREXAMPLE:
            return "RED"
        return str(value.status).upper()
    if isinstance(value, ObligationAssessment):
        if value.verdict == SATISFIED:
            return "GREEN"
        if value.verdict == UNSATISFIED:
            return "RED"
        return "UNKNOWN"
    if isinstance(value, Mapping):
        for key in ("result_class", "status", "verdict", "result", "state"):
            if key in value:
                return _test_status(value[key])
    text = str(value or "").upper()
    if text in {"PASS", "PASSED", "GREEN", "SATISFIED", "OK", "0"}:
        return "GREEN"
    if text in {"FAIL", "FAILED", "RED", "UNSATISFIED", "COUNTEREXAMPLE", "1"}:
        return "RED"
    return "UNKNOWN"


def _receipt_values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (CheckReceipt, ObligationAssessment, DiagnosticReceipt)):
        return (value,)
    if isinstance(value, Mapping):
        if any(key in value for key in ("before", "after", "before_receipts", "after_receipts")):
            rows: list[object] = []
            for key in ("before", "after", "before_receipts", "after_receipts"):
                rows.extend(_receipt_values(value.get(key)))
            return tuple(rows)
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        rows: list[object] = []
        for item in value:
            rows.extend(_receipt_values(item))
        return tuple(rows)
    return (value,)


def _test_transition(before: object, after: object) -> tuple[TriState, str]:
    before_values = _receipt_values(before)
    after_values = _receipt_values(after)
    before_status = [_test_status(item) for item in before_values]
    after_status = [_test_status(item) for item in after_values]
    if not before_status or not after_status:
        return UNKNOWN, "missing before/after test result"
    if any(status == "GREEN" for status in after_status):
        if any(status == "GREEN" for status in before_status):
            return TRUE, "test already green"
        if any(status == "RED" for status in before_status):
            return TRUE, "red-to-green"
        return UNKNOWN, "candidate green; starter result unresolved"
    if any(status == "RED" for status in after_status):
        if any(status == "RED" for status in before_status):
            return TRUE, "persistent-fail"
        if any(status == "GREEN" for status in before_status):
            return TRUE, "candidate regression"
    if any(status == "UNKNOWN" for status in (*before_status, *after_status)):
        return UNKNOWN, "unresolved test result"
    return FALSE, "no discriminating test result"


def _extract_test_events(
    obligations: Sequence[object], receipts: object, before_receipts: object, after_receipts: object,
) -> list[ChangeEvent]:
    before = before_receipts
    after = after_receipts
    if isinstance(receipts, Mapping):
        before = receipts.get("before", receipts.get("before_receipts", before))
        after = receipts.get("after", receipts.get("after_receipts", after))
    if after is None or after == ():
        # A status transition can be carried directly on a ticket row in a
        # replay journal adapter.
        rows = [item for item in obligations if isinstance(item, Mapping) and ("before_status" in item or "after_status" in item)]
        if rows:
            before = tuple(row.get("before_status") for row in rows)
            after = tuple(row.get("after_status") for row in rows)
    status, reason = _test_transition(before, after)
    if status is UNKNOWN and not before and not after:
        return []
    return [ChangeEvent(
        kind=PUBLIC_TESTS_BECAME_GREEN,
        subject="public_tests",
        certainty=status,
        before=tuple(_test_status(item) for item in _receipt_values(before)),
        after=tuple(_test_status(item) for item in _receipt_values(after)),
        details={"reason": reason, "predicate": "public test discrimination", "boundary": "public_tests"},
        evidence_refs=tuple(
            getattr(item, "evidence_ref", "") for item in _receipt_values(after)
            if getattr(item, "evidence_ref", "")
        ),
    )]


_SITE_PACKAGES_RE = re.compile(
    r"(?:site-packages|dist-packages)[/\\]"
    r"([A-Za-z0-9_.-]+)(?:[/\\]([^\s:'\"]+))?",
    re.IGNORECASE,
)


def _dependency_traceback_events(*values: object) -> list[ChangeEvent]:
    """Characterize traceback crossings without treating them as API claims."""
    rows: list[tuple[str, tuple[str, ...], str]] = []

    def visit(value: object, key: str = "", evidence: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            refs = tuple(dict.fromkeys((*evidence, *(
                str(value[name]) for name in ("receipt_id", "event_id", "record_id", "evidence_ref")
                if value.get(name)
            ))))
            for name, child in value.items():
                visit(child, str(name), refs)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child, key, evidence)
            return
        if not isinstance(value, str):
            return
        lowered = key.lower()
        if not ("trace" in lowered or "failure" in lowered or "stderr" in lowered
                or "error" in lowered or "output" in lowered or "file " in value.lower()):
            return
        match = _SITE_PACKAGES_RE.search(value)
        if match is not None:
            rows.append((match.group(1), evidence, match.group(2) or ""))

    for value in values:
        visit(value)
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    events: list[ChangeEvent] = []
    for package, refs, module_path in rows:
        identity = (package, refs, module_path)
        if identity in seen:
            continue
        seen.add(identity)
        module = f"{package}/{module_path}" if module_path else package
        events.append(ChangeEvent(
            kind=DEPENDENCY_TRACEBACK,
            path="<runtime>",
            subject=package,
            symbol=module,
            before=None,
            after={"package": package, "module": module},
            details={
                "predicate": "dependency behavior characterization",
                "package": package,
                "module": module,
                "traceback_crossed_site_packages": True,
                "boundary": "third-party dependency",
                "input_partition": "runtime",
            },
            evidence_refs=refs or ("runtime-trace:" + stable_digest(identity),),
        ))
    return events


def extract_change_events(
    before: object,
    after: object,
    obligations: object = (),
    receipts: object = (),
    *,
    ticket_obligations: object = None,
    before_receipts: object = (),
    after_receipts: object = (),
    metadata: Mapping[str, object] | None = None,
) -> tuple[ChangeEvent, ...]:
    """Extract deterministic typed events from source and ticket facts.

    The source input may be a string, a ``path -> source`` mapping, or a
    mapping with ``path``/``content``.  A parse failure is represented as an
    ``UNKNOWN`` event for the affected detector; it is never converted into a
    positive AST claim.
    """

    if ticket_obligations is not None:
        obligations = ticket_obligations
    obligation_rows = tuple(_record_values(obligations))
    events: list[ChangeEvent] = []
    for pair in _source_pairs(before, after):
        if not pair.changed:
            continue
        before_tree = _parse(pair.before)
        after_tree = _parse(pair.after)
        if before_tree is None or after_tree is None:
            reason = "before source unavailable" if before_tree is None else "after source is not parseable"
            events.extend(_unknown_event(kind, pair, reason) for kind in EventKind
                          if kind is not DEPENDENCY_TRACEBACK)
            continue

        before_functions = _function_nodes(before_tree)
        after_functions = _function_nodes(after_tree)
        for name in sorted(set(before_functions) & set(after_functions)):
            left = _argument_signature(before_functions[name])
            right = _argument_signature(after_functions[name])
            if left != right:
                events.append(ChangeEvent(
                    SIGNATURE_CHANGED, pair.path, name, name,
                    before=left, after=right,
                    details={"predicate": "signature changed", "boundary": "callers", "function": name},
                ))
        before_exports = _public_exports(before_tree)
        after_exports = _public_exports(after_tree)
        for name in sorted(before_exports - after_exports):
            events.append(ChangeEvent(
                PUBLIC_EXPORT_REMOVED, pair.path, name, name,
                before=True, after=False,
                details={"predicate": "public export removed", "boundary": "public API", "export": name},
            ))

        before_writes = _alias_writes(before_tree)
        after_writes = _alias_writes(after_tree)
        for function in sorted(set(after_writes) | set(before_writes)):
            added = after_writes.get(function, set()) - before_writes.get(function, set())
            for alias, operation, argument in sorted(added):
                events.append(ChangeEvent(
                    INPUT_ALIAS_WRITE_ADDED, pair.path, function, function,
                    before=None, after={"alias": alias, "operation": operation, "argument": argument},
                    details={
                        "predicate": "caller-owned input mutation",
                        "operation": operation,
                        "alias": alias,
                        "argument": argument,
                        "boundary": "caller input",
                    },
                ))

        before_validation = _validation_paths(before_tree)
        after_validation = _validation_paths(after_tree)
        if before_validation != after_validation:
            events.append(ChangeEvent(
                VALIDATION_PATH_CHANGED, pair.path, pair.path, "",
                before=before_validation, after=after_validation,
                details={"predicate": "validation path changed", "boundary": "validation"},
            ))

        before_optional = _optional_fields(before_tree)
        after_optional = _optional_fields(after_tree)
        for function, field_kind, field_name in sorted(after_optional - before_optional):
            partition = "omitted"
            if field_kind == "field":
                field_name = field_name.split(":", 1)[0]
            events.append(ChangeEvent(
                OPTIONAL_FIELD_INTRODUCED, pair.path, function, function,
                before=None, after={"field": field_name, "kind": field_kind},
                details={
                    "predicate": "optional field introduced",
                    "field": field_name,
                    "input_partition": partition,
                    "boundary": "input contract",
                },
            ))

    events.extend(_extract_test_events(obligation_rows, receipts, before_receipts, after_receipts))
    events.extend(_dependency_traceback_events(receipts, before_receipts, after_receipts, metadata or {}))
    return tuple(sorted(events, key=lambda event: (
        event.path, event.kind.value, event.subject, event.symbol, _canonical(event.details),
    )))


def build_revision_facts(
    before: object,
    after: object,
    obligations: object = (),
    receipts: object = (),
    *,
    ticket_obligations: object = None,
    before_receipts: object = (),
    after_receipts: object = (),
    revision: object = None,
    fixture: object = None,
    environment: object = None,
    step: int = 0,
    metadata: Mapping[str, object] | None = None,
) -> RevisionFacts:
    if ticket_obligations is not None:
        obligations = ticket_obligations
    sources = _source_pairs(before, after)
    events = extract_change_events(
        before, after, obligations, receipts,
        before_receipts=before_receipts, after_receipts=after_receipts,
        metadata=metadata,
    )
    before_revision = stable_digest({"sources": [(item.path, item.before) for item in sources]})
    current_revision = _identity(revision) if revision is not None else stable_digest({
        "sources": [(item.path, item.after) for item in sources],
    })
    return RevisionFacts(
        sources=sources, events=events, obligations=tuple(_record_values(obligations)),
        receipts=receipts, before_receipts=before_receipts, after_receipts=after_receipts,
        revision=current_revision, before_revision=before_revision, fixture=fixture,
        environment=environment, step=step, metadata=metadata or {},
    )


@dataclass(frozen=True, slots=True)
class PredicateAST:
    """Small JSON-shaped three-valued predicate language."""

    op: str
    value: object = None
    args: tuple["PredicateAST", ...] = ()
    field: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "op", str(self.op or "true").strip().lower().replace("-", "_"))
        raw_args = self.args
        if not raw_args and self.op in {"all", "and", "any", "or"} and isinstance(self.value, (list, tuple)):
            raw_args = tuple(self.value)
        converted: list[PredicateAST] = []
        for item in raw_args:
            converted.append(item if isinstance(item, PredicateAST) else PredicateAST.from_mapping(item))
        object.__setattr__(self, "args", tuple(converted))
        if self.field:
            object.__setattr__(self, "field", str(self.field))

    @classmethod
    def from_mapping(cls, value: object) -> "PredicateAST":
        if isinstance(value, cls):
            return value
        if value is None or value is True:
            return cls("true")
        if value is False:
            return cls("false")
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise TypeError("predicate AST must be an object, string, or PredicateAST")
        op = value.get("op", value.get("type", value.get("predicate", "true")))
        args = value.get("args", value.get("children", ()))
        if op in {"event", "event_present", "event_absent"}:
            event = value.get("kind", value.get("event", value.get("value")))
            return cls(str(op), event, tuple(args) if isinstance(args, Sequence) else ())
        if op in {"all", "and", "any", "or"}:
            return cls(str(op), tuple(args) if isinstance(args, Sequence) else (), ())
        predicate_value = value.get("value", value.get("values", value.get("keywords")))
        return cls(str(op), predicate_value, tuple(args) if isinstance(args, Sequence) else (), str(value.get("field", "")))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"op": self.op}
        if self.value is not None and self.op not in {"all", "and", "any", "or"}:
            result["value"] = _jsonable(self.value)
        if self.field:
            result["field"] = self.field
        if self.args:
            result["args"] = [item.to_dict() for item in self.args]
        return result

    def evaluate(self, facts: RevisionFacts | Mapping[str, object]) -> TriState:
        return _evaluate_predicate(self, facts)


def _fact_value(facts: RevisionFacts | Mapping[str, object], field_name: str) -> object:
    if isinstance(facts, RevisionFacts):
        current: object = facts
        if field_name in {"revision", "before_revision", "fixture", "environment", "step", "events", "obligations", "receipts", "metadata"}:
            return getattr(facts, field_name)
        if field_name == "changed":
            return facts.changed
        if field_name == "material_digest":
            return facts.material_digest
        if str(field_name).startswith("metadata."):
            current = facts.metadata
            field_name = str(field_name)[len("metadata."):]
        else:
            current = facts.metadata
    else:
        current = facts
    for part in str(field_name).split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _event_match(event: ChangeEvent, target: object) -> bool:
    kind = _event_kind(target)
    if kind is not None:
        return event.kind == kind
    return str(target or "").strip().lower() in {event.kind.value.lower(), event.predicate.lower()}


def _evaluate_predicate(predicate: PredicateAST, facts: RevisionFacts | Mapping[str, object]) -> TriState:
    op = predicate.op
    if op in {"true", "always", "yes"}:
        return TRUE
    if op in {"false", "never", "no"}:
        return FALSE
    if op in {"unknown", "unresolved"}:
        return UNKNOWN
    if op in {"all", "and"}:
        values = tuple(item.evaluate(facts) for item in predicate.args)
        if any(item is FALSE for item in values):
            return FALSE
        return UNKNOWN if any(item is UNKNOWN for item in values) else TRUE
    if op in {"any", "or"}:
        values = tuple(item.evaluate(facts) for item in predicate.args)
        if any(item is TRUE for item in values):
            return TRUE
        return UNKNOWN if any(item is UNKNOWN for item in values) else FALSE
    if op == "not":
        child = predicate.args[0] if predicate.args else PredicateAST.from_mapping(predicate.value)
        value = child.evaluate(facts)
        return {TRUE: FALSE, FALSE: TRUE, UNKNOWN: UNKNOWN}[value]
    if op in {"event", "event_present", "has_event"}:
        events = _fact_value(facts, "events") or ()
        matches = [event for event in events if isinstance(event, ChangeEvent) and _event_match(event, predicate.value)]
        if any(event.certainty is TRUE for event in matches):
            return TRUE
        if any(event.certainty is UNKNOWN for event in matches):
            return UNKNOWN
        return FALSE
    if op == "event_absent":
        value = PredicateAST("event_present", predicate.value).evaluate(facts)
        return {TRUE: FALSE, FALSE: TRUE, UNKNOWN: UNKNOWN}[value]
    if op in {"field_present", "present"}:
        value = _fact_value(facts, predicate.field or str(predicate.value or ""))
        return UNKNOWN if value is None else TRUE
    if op in {"field_equals", "equals", "eq"}:
        current = _fact_value(facts, predicate.field)
        if current is None:
            return UNKNOWN
        return TRUE if _jsonable(current) == _jsonable(predicate.value) else FALSE
    if op in {"field_contains", "contains"}:
        current = _fact_value(facts, predicate.field)
        if current is None:
            return UNKNOWN
        if isinstance(current, str):
            return TRUE if str(predicate.value).lower() in current.lower() else FALSE
        try:
            return TRUE if predicate.value in current else FALSE
        except TypeError:
            return FALSE
    if op in {"obligation_bound", "bound_obligation"}:
        found, known = _find_obligation_bindings(
            _fact_value(facts, "obligations") or (), predicate.value,
        )
        if found:
            return TRUE
        return FALSE if known else UNKNOWN
    if op in {"obligation_clause_contains", "obligation_keyword"}:
        rows = _fact_value(facts, "obligations") or ()
        texts = [_obligation_text(item).lower() for item in rows]
        if not texts:
            return UNKNOWN
        return TRUE if any(str(predicate.value or "").lower() in text for text in texts) else FALSE
    if op in {"obligation_clause_contains_any", "obligation_keywords"}:
        values = predicate.value if isinstance(predicate.value, (list, tuple, set)) else (predicate.value,)
        rows = _fact_value(facts, "obligations") or ()
        texts = [_obligation_text(item).lower() for item in rows]
        if not texts:
            return UNKNOWN
        return TRUE if any(str(value or "").lower() in text for text in texts for value in values) else FALSE
    if op in {"receipt_status", "has_receipt"}:
        statuses = [_test_status(item) for item in _receipt_values(_fact_value(facts, "receipts"))]
        if not statuses:
            return UNKNOWN
        wanted = str(predicate.value or "").upper()
        return TRUE if wanted in statuses else FALSE
    if op in {"test_transition", "public_tests_transition"}:
        before = _fact_value(facts, "before_receipts")
        after = _fact_value(facts, "after_receipts")
        status, _ = _test_transition(before, after)
        return status if predicate.value in (None, "", "TRUE", "true") else (TRUE if status.value == str(predicate.value).upper() else FALSE)
    if op in {"callable", "python"} and callable(predicate.value):
        try:
            return TriState.from_value(predicate.value(facts))
        except Exception:
            return UNKNOWN
    return UNKNOWN


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    max_new_items_per_step: int = 1
    max_tokens_per_item: int = 120
    max_optional_items: int = 4

    def __post_init__(self) -> None:
        if int(self.max_new_items_per_step) < 0 or int(self.max_tokens_per_item) < 0:
            raise ValueError("pattern-card budgets must be non-negative")
        object.__setattr__(self, "max_new_items_per_step", int(self.max_new_items_per_step))
        object.__setattr__(self, "max_tokens_per_item", int(self.max_tokens_per_item))
        object.__setattr__(self, "max_optional_items", max(0, int(self.max_optional_items)))

    @classmethod
    def from_mapping(cls, value: object) -> "BudgetSpec":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls()
        return cls(
            value.get("max_new_items_per_step", value.get("new_items_per_step", 1)),
            value.get("max_tokens_per_item", value.get("tokens", 120)),
            value.get("max_optional_items", value.get("optional_items", 4)),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_new_items_per_step": self.max_new_items_per_step,
            "max_tokens_per_item": self.max_tokens_per_item,
            "max_optional_items": self.max_optional_items,
        }


@dataclass(frozen=True, slots=True)
class PatternCard:
    """Versioned, lifecycle-controlled pattern definition."""

    card_id: str
    version: str
    lifecycle: Lifecycle | str
    trigger: PredicateAST | Mapping[str, object]
    applicability: PredicateAST | Mapping[str, object] = field(default_factory=lambda: PredicateAST("true"))
    contraindications: tuple[PredicateAST | Mapping[str, object], ...] = ()
    operation_template: Mapping[str, object] | str = field(default_factory=dict)
    oracle_requirement: OracleRequirement | str = OracleRequirement.NONE
    allowed_effects: frozenset[Effect | str] = frozenset()
    dedup_key: tuple[str, ...] | str = ("subject", "predicate", "input_partition", "boundary")
    budgets: BudgetSpec | Mapping[str, object] = field(default_factory=BudgetSpec)
    name: str = ""
    false_positive_cost: str = ""

    def __post_init__(self) -> None:
        ident = _text(self.card_id, 80)
        if not ident:
            raise ValueError("pattern card requires card_id")
        object.__setattr__(self, "card_id", ident)
        object.__setattr__(self, "version", _text(self.version or "1", 40))
        try:
            object.__setattr__(self, "lifecycle", Lifecycle(str(self.lifecycle).upper()))
        except ValueError as error:
            raise ValueError(f"unknown pattern-card lifecycle: {self.lifecycle!r}") from error
        object.__setattr__(self, "trigger", PredicateAST.from_mapping(self.trigger))
        object.__setattr__(self, "applicability", PredicateAST.from_mapping(self.applicability))
        object.__setattr__(self, "contraindications", tuple(PredicateAST.from_mapping(item) for item in _values(self.contraindications)))
        operation = {"kind": self.operation_template} if isinstance(self.operation_template, str) else dict(self.operation_template)
        if not operation.get("kind"):
            operation["kind"] = self.card_id
        object.__setattr__(self, "operation_template", copy.deepcopy(operation))
        object.__setattr__(self, "oracle_requirement", _oracle(self.oracle_requirement))
        effects = frozenset(_effect(item) for item in _values(self.allowed_effects))
        if not effects:
            raise ValueError(f"pattern card {ident} has no allowed effects")
        object.__setattr__(self, "allowed_effects", effects)
        if isinstance(self.dedup_key, str):
            raw_key = tuple(item.strip() for item in self.dedup_key.replace("|", ",").split(",") if item.strip())
        else:
            raw_key = tuple(str(item) for item in _values(self.dedup_key))
        object.__setattr__(self, "dedup_key", raw_key or ("subject", "predicate", "input_partition", "boundary"))
        object.__setattr__(self, "budgets", BudgetSpec.from_mapping(self.budgets))
        object.__setattr__(self, "name", _text(self.name or ident, 160))
        object.__setattr__(self, "false_positive_cost", _text(self.false_positive_cost, 160))

    @property
    def id(self) -> str:
        return self.card_id

    @property
    def card_version(self) -> str:
        return self.version

    @property
    def operation(self) -> Mapping[str, object]:
        return self.operation_template

    @property
    def active(self) -> bool:
        return self.lifecycle in {Lifecycle.SHADOW, Lifecycle.ENABLED}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PatternCard":
        return cls(
            str(value.get("card_id", value.get("id", ""))),
            str(value.get("version", "1")),
            str(value.get("lifecycle", "CANDIDATE")),
            value.get("trigger", {"op": "true"}),
            value.get("applicability", {"op": "true"}),
            value.get("contraindications", ()),
            value.get("operation_template", value.get("operation", {})),
            value.get("oracle_requirement", "NONE"),
            value.get("allowed_effects", ()),
            value.get("dedup_key", ("subject", "predicate", "input_partition", "boundary")),
            value.get("budgets", {}),
            str(value.get("name", "")),
            str(value.get("false_positive_cost", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "pattern-card/1",
            "card_id": self.card_id,
            "version": self.version,
            "lifecycle": self.lifecycle.value,
            "name": self.name,
            "trigger": self.trigger.to_dict(),
            "applicability": self.applicability.to_dict(),
            "contraindications": [item.to_dict() for item in self.contraindications],
            "operation_template": _jsonable(self.operation_template),
            "oracle_requirement": self.oracle_requirement.value,
            "allowed_effects": sorted(item.value for item in self.allowed_effects),
            "dedup_key": list(self.dedup_key),
            "budgets": self.budgets.to_dict(),
            "false_positive_cost": self.false_positive_cost,
        }


def _obligation_text(value: object) -> str:
    if isinstance(value, ObligationSpec):
        return " ".join((value.clause, " ".join(value.targets), " ".join(value.acceptance_policy)))
    if isinstance(value, ObligationAssessment):
        return " ".join((value.clause, " ".join(value.check_ids), value.obligation_id))
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key in ("id", "obligation_id", "clause", "text", "source_text", "kind", "template", "acceptance_policy", "checks", "operation", "pattern_card"):
            if key in value:
                parts.append(str(value[key]))
        return " ".join(parts)
    return str(value or "")


def _obligation_id(value: object) -> str:
    if isinstance(value, (ObligationSpec, ObligationAssessment)):
        return str(value.obligation_id)
    if isinstance(value, Mapping):
        return str(value.get("obligation_id", value.get("id", "")))
    return ""


def _obligation_checks(value: object) -> tuple[str, ...]:
    if isinstance(value, ObligationSpec):
        return tuple(value.acceptance_policy)
    if isinstance(value, ObligationAssessment):
        return tuple(value.check_ids)
    if isinstance(value, Mapping):
        return _strings(value.get("acceptance_policy", value.get("checks", value.get("check_ids", ()))))
    return ()


def _operation_match(text: str, value: object) -> bool:
    lower = text.lower()
    if isinstance(value, Mapping):
        terms = [value.get("kind", ""), *(_values(value.get("check_templates", ())))]
    elif isinstance(value, (list, tuple, set, frozenset)):
        terms = list(value)
    else:
        terms = [value]
    for term in terms:
        term_text = str(term or "").lower()
        if not term_text:
            continue
        if term_text in lower or term_text.replace("_", " ") in lower:
            return True
    return False


def _card_keywords(card: PatternCard) -> tuple[str, ...]:
    operation = card.operation_template
    keywords = [card.card_id, card.name, str(operation.get("kind", "")), str(operation.get("predicate", ""))]
    keywords.extend(str(item) for item in _values(operation.get("keywords", ())))
    return tuple(item.lower() for item in keywords if item)


def _find_obligation_bindings(obligations: Iterable[object], match: object = None, card: PatternCard | None = None) -> tuple[bool, bool]:
    rows = tuple(obligations)
    if not rows:
        return False, False
    if isinstance(match, Mapping):
        terms = tuple(str(item).lower() for item in _values(match.get("keywords", match.get("kind", ()))))
    elif match is None and card is not None:
        terms = _card_keywords(card)
    else:
        terms = (str(match or "").lower(),)
    known = False
    for row in rows:
        text = _obligation_text(row).lower()
        explicit = row.get("pattern_card") if isinstance(row, Mapping) else None
        if card is not None and explicit is not None and str(explicit) == card.card_id:
            return True, True
        if card is not None and explicit is not None and str(explicit) != card.card_id:
            known = True
            continue
        checks = " ".join(item.lower() for item in _obligation_checks(row))
        if any(term and (term in text or term in checks) for term in terms):
            return True, True
        known = True
    return False, known


@dataclass(frozen=True, slots=True)
class BoundPatternInstance:
    card: PatternCard
    applicability: TriState
    triggered: TriState = FALSE
    oracle_binding: TriState = UNKNOWN
    events: tuple[ChangeEvent, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    subject: str = "*"
    predicate: str = ""
    input_partition: str = "all"
    boundary: str = "public"
    instance_id: str = ""
    reason: str = ""
    contraindicated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "applicability", TriState.from_value(self.applicability))
        object.__setattr__(self, "triggered", TriState.from_value(self.triggered))
        object.__setattr__(self, "oracle_binding", TriState.from_value(self.oracle_binding))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "obligation_ids", _strings(self.obligation_ids))
        object.__setattr__(self, "subject", _text(self.subject or "*", 300))
        object.__setattr__(self, "predicate", _text(self.predicate or self.card.operation.get("kind", ""), 200))
        object.__setattr__(self, "input_partition", _text(self.input_partition or "all", 120))
        object.__setattr__(self, "boundary", _text(self.boundary or "public", 160))
        if not self.instance_id:
            object.__setattr__(self, "instance_id", stable_digest({
                "card": self.card.card_id, "version": self.card.version,
                "subject": self.subject, "predicate": self.predicate,
                "input_partition": self.input_partition, "boundary": self.boundary,
            }))

    @property
    def bound(self) -> bool:
        return self.applicability is TRUE and self.oracle_binding is TRUE

    @property
    def dedup_key(self) -> tuple[str, str, str, str]:
        return (self.subject, self.predicate, self.input_partition, self.boundary)

    def to_dict(self) -> dict[str, object]:
        return {
            "card_id": self.card.card_id,
            "card_version": self.card.version,
            "applicability": self.applicability.value,
            "triggered": self.triggered.value,
            "oracle_binding": self.oracle_binding.value,
            "events": [event.to_dict() for event in self.events],
            "obligation_ids": list(self.obligation_ids),
            "subject": self.subject,
            "predicate": self.predicate,
            "input_partition": self.input_partition,
            "boundary": self.boundary,
            "instance_id": self.instance_id,
            "reason": self.reason,
            "contraindicated": self.contraindicated,
            "dedup_key": list(self.dedup_key),
        }


def bind_card(
    card: PatternCard,
    facts: RevisionFacts | Mapping[str, object],
    obligations: object = None,
    receipts: object = None,
) -> BoundPatternInstance:
    """Bind one card while retaining UNKNOWN instead of guessing."""

    if not isinstance(card, PatternCard):
        card = PatternCard.from_mapping(card)  # type: ignore[arg-type]
    if not isinstance(facts, RevisionFacts):
        raw_events = facts.get("events", ()) if isinstance(facts, Mapping) else ()
        events = tuple(
            item if isinstance(item, ChangeEvent) else ChangeEvent.from_mapping(item)
            for item in _values(raw_events)
            if isinstance(item, (ChangeEvent, Mapping))
        )
        facts = RevisionFacts(
            events=events,
            obligations=tuple(_record_values(facts.get("obligations", ()))) if isinstance(facts, Mapping) else (),
            receipts=facts.get("receipts", ()) if isinstance(facts, Mapping) else (),
            revision=str(facts.get("revision", "")) if isinstance(facts, Mapping) else "",
            metadata=facts if isinstance(facts, Mapping) else {},
        )
    if obligations is not None:
        facts = replace(facts, obligations=tuple(_record_values(obligations)))
    if receipts is not None:
        facts = replace(facts, receipts=receipts)
    triggered = card.trigger.evaluate(facts)
    applicability = card.applicability.evaluate(facts)
    contraindicated = False
    for predicate in card.contraindications:
        value = predicate.evaluate(facts)
        if value is TRUE:
            contraindicated = True
            applicability = FALSE
            break
    if triggered is FALSE:
        applicability = FALSE
    elif triggered is UNKNOWN and applicability is not FALSE:
        applicability = UNKNOWN
    matched_rows = _matching_obligations(card, facts)
    matched = bool(matched_rows)
    known = bool(facts.obligations)
    if card.oracle_requirement is OracleRequirement.NONE:
        oracle_binding = TRUE
    elif card.oracle_requirement is OracleRequirement.ACCEPTED_PROJECT_INVARIANT:
        invariant = facts.metadata.get("accepted_invariants", ()) if isinstance(facts.metadata, Mapping) else ()
        oracle_binding = TRUE if invariant else UNKNOWN
    else:
        oracle_binding = TRUE if matched else (FALSE if known else UNKNOWN)
    events = tuple(event for event in facts.events if _event_match(event, card.trigger.value))
    if not events and triggered is TRUE:
        events = facts.events
    event = events[0] if events else None
    subject = event.subject if event else (facts.metadata.get("subject", "*") if isinstance(facts.metadata, Mapping) else "*")
    predicate = event.predicate if event else str(card.operation.get("predicate", card.operation.get("kind", card.card_id)))
    partition = event.input_partition if event else "all"
    boundary = event.boundary if event else str(card.operation.get("boundary", "public"))
    reason = "triggered" if triggered is TRUE else "trigger unresolved" if triggered is UNKNOWN else "not triggered"
    if contraindicated:
        reason = "contraindicated by ticket facts"
    elif oracle_binding is UNKNOWN:
        reason = "oracle binding is unresolved"
    elif oracle_binding is FALSE:
        reason = "no matching ticket obligation"
    return BoundPatternInstance(
        card=card, applicability=applicability, triggered=triggered,
        oracle_binding=oracle_binding, events=events,
        obligation_ids=tuple(_obligation_id(row) for row in facts.obligations if _obligation_id(row)),
        subject=str(subject), predicate=predicate, input_partition=partition,
        boundary=boundary, reason=reason, contraindicated=contraindicated,
    )


@dataclass(frozen=True, slots=True, init=False)
class DiagnosticReceipt:
    """A pattern diagnostic bound to revision, fixture, and environment.

    ``host_receipt`` is the existing ``CheckReceipt`` when a check has run.
    A receipt with ``status=WOULD_EXECUTE`` is only a shadow plan; it is not a
    verdict and carries no evidence reference.
    """

    card_id: str
    instance_id: str
    status: str
    revision_digest: str
    fixture_digest: str
    environment_digest: str
    freshness: Freshness | str = Freshness.UNKNOWN
    observed: object = None
    expected: object = None
    counterexample: object = None
    evidence_refs: tuple[str, ...] = ()
    host_receipt: CheckReceipt | None = None
    reason: str = ""
    would_execute: bool = False

    def __init__(
        self,
        card_id: str = "",
        instance_id: str = "",
        status: str = "WOULD_EXECUTE",
        revision_digest: object = None,
        fixture_digest: object = None,
        environment_digest: object = None,
        freshness: Freshness | str = Freshness.UNKNOWN,
        observed: object = None,
        expected: object = None,
        counterexample: object = None,
        evidence_refs: object = (),
        host_receipt: CheckReceipt | None = None,
        reason: str = "",
        would_execute: bool = False,
        *,
        card: str | None = None,
        instance: str | None = None,
        result: str | None = None,
        revision: object = None,
        fixture: object = None,
        environment: object = None,
        receipt: CheckReceipt | None = None,
    ) -> None:
        if card is not None and not card_id:
            card_id = card
        if instance is not None and not instance_id:
            instance_id = instance
        if result is not None and status == "WOULD_EXECUTE":
            status = result
        if revision_digest is None:
            revision_digest = revision
        if fixture_digest is None:
            fixture_digest = fixture
        if environment_digest is None:
            environment_digest = environment
        if host_receipt is None and receipt is not None:
            host_receipt = receipt
        if (
            str(freshness).upper() in {"UNKNOWN", "FRESHNESS.UNKNOWN"}
            and not would_execute
            and any(value is not None for value in (revision_digest, fixture_digest, environment_digest))
        ):
            freshness = Freshness.FRESH
        object.__setattr__(self, "card_id", card_id)
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "revision_digest", revision_digest)
        object.__setattr__(self, "fixture_digest", fixture_digest)
        object.__setattr__(self, "environment_digest", environment_digest)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "counterexample", counterexample)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "host_receipt", host_receipt)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "would_execute", would_execute)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(self, "card_id", _text(self.card_id, 80))
        object.__setattr__(self, "instance_id", _text(self.instance_id, 240))
        object.__setattr__(self, "status", _text(self.status or "WOULD_EXECUTE", 60).upper())
        object.__setattr__(self, "revision_digest", _identity(self.revision_digest))
        object.__setattr__(self, "fixture_digest", _identity(self.fixture_digest))
        object.__setattr__(self, "environment_digest", _identity(self.environment_digest))
        try:
            freshness = Freshness(str(self.freshness).upper())
        except ValueError:
            freshness = Freshness.UNKNOWN
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "evidence_refs", _strings(self.evidence_refs))
        object.__setattr__(self, "reason", _text(self.reason, 500))

    @property
    def revision(self) -> str:
        return self.revision_digest

    @property
    def fixture(self) -> str:
        return self.fixture_digest

    @property
    def environment(self) -> str:
        return self.environment_digest

    @property
    def fresh(self) -> bool:
        return self.freshness is Freshness.FRESH

    @property
    def stale(self) -> bool:
        return self.freshness is Freshness.STALE

    @classmethod
    def from_check_receipt(
        cls,
        card_id: str,
        instance_id: str,
        receipt: CheckReceipt,
        *,
        current_revision: object = None,
        current_fixture: object = None,
        current_environment: object = None,
    ) -> "DiagnosticReceipt":
        if not isinstance(receipt, CheckReceipt):
            receipt = CheckReceipt.from_mapping(receipt)  # type: ignore[arg-type]
        freshness = _receipt_freshness(
            receipt,
            current_revision=current_revision,
            current_fixture=current_fixture,
            current_environment=current_environment,
        )
        return cls(
            card_id=card_id,
            instance_id=instance_id,
            status=receipt.display_status,
            revision_digest=receipt.snapshot_digest,
            fixture_digest=receipt.fixture_digest,
            environment_digest=receipt.environment_digest,
            freshness=freshness,
            observed=receipt.observed,
            expected=receipt.expected,
            counterexample=receipt.counterexample,
            evidence_refs=(receipt.evidence_ref,),
            host_receipt=receipt,
            reason=receipt.reason,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DiagnosticReceipt":
        return cls(
            str(value.get("card_id", value.get("card", ""))),
            str(value.get("instance_id", value.get("instance", ""))),
            str(value.get("status", value.get("result", "WOULD_EXECUTE"))),
            value.get("revision_digest", value.get("revision", "")),
            value.get("fixture_digest", value.get("fixture", "")),
            value.get("environment_digest", value.get("environment", "")),
            value.get("freshness", Freshness.UNKNOWN),
            value.get("observed"), value.get("expected"), value.get("counterexample"),
            value.get("evidence_refs", value.get("evidence", ())),
            None, str(value.get("reason", "")), bool(value.get("would_execute", False)),
        )

    def is_fresh(
        self,
        revision: object = None,
        fixture: object = None,
        environment: object = None,
    ) -> bool:
        if self.freshness is Freshness.STALE:
            return False
        checks = (
            (revision, self.revision_digest),
            (fixture, self.fixture_digest),
            (environment, self.environment_digest),
        )
        for current, bound in checks:
            if current is None:
                continue
            if _identity(current) != bound:
                return False
        return self.freshness is Freshness.FRESH or all(current is None for current, _ in checks)

    def for_revision(self, revision: object, fixture: object = None, environment: object = None) -> "DiagnosticReceipt":
        fresh = self.is_fresh(revision, fixture, environment)
        return DiagnosticReceipt(
            self.card_id, self.instance_id, self.status, self.revision_digest,
            self.fixture_digest, self.environment_digest,
            Freshness.FRESH if fresh else Freshness.STALE,
            self.observed, self.expected, self.counterexample, self.evidence_refs,
            self.host_receipt, self.reason, self.would_execute,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "pattern-diagnostic-receipt/1",
            "card_id": self.card_id,
            "instance_id": self.instance_id,
            "status": self.status,
            "revision_digest": self.revision_digest,
            "fixture_digest": self.fixture_digest,
            "environment_digest": self.environment_digest,
            "freshness": self.freshness.value,
            "observed": _jsonable(self.observed),
            "expected": _jsonable(self.expected),
            "counterexample": _jsonable(self.counterexample),
            "evidence_refs": list(self.evidence_refs),
            "host_receipt": self.host_receipt.to_dict() if self.host_receipt is not None else None,
            "reason": self.reason,
            "would_execute": self.would_execute,
        }


def _receipt_freshness(
    receipt: CheckReceipt,
    *,
    current_revision: object = None,
    current_fixture: object = None,
    current_environment: object = None,
) -> Freshness:
    checks = (
        (current_revision, receipt.snapshot_digest),
        (current_fixture, receipt.fixture_digest),
        (current_environment, receipt.environment_digest),
    )
    if any(current is not None and _identity(current) != bound for current, bound in checks):
        return Freshness.STALE
    return Freshness.FRESH if any(current is not None for current, _ in checks) else Freshness.UNKNOWN


def _lexical_tokens(value: object) -> int:
    text = _text(value, 4000)
    return max(1, int(math.ceil(len(text) / 4))) if text else 0


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    """The bounded shadow record returned by the compiler."""

    route: Route | str
    card_id: str
    card_version: str
    instance_id: str
    step: int
    subject: str
    predicate: str
    input_partition: str
    boundary: str
    priority: int
    message: str
    applicability: TriState | str
    effect: Effect | str | None = None
    card_ids: tuple[str, ...] = ()
    allowed_effects: frozenset[Effect | str] = frozenset()
    dedup_key: tuple[str, str, str, str] = ("*", "", "all", "public")
    diagnostic_receipt: DiagnosticReceipt | None = None
    observation: Mapping[str, object] | None = None
    question_spec: QuestionSpec | None = None
    question: Question | None = None
    reason: str = ""
    material_evidence: str = ""
    token_estimate: int = 0
    merged: bool = False
    would_surface: bool = True

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "route", Route(str(self.route)))
        except ValueError as error:
            raise ValueError(f"unknown intervention route: {self.route!r}") from error
        if self.effect is not None:
            object.__setattr__(self, "effect", _effect(self.effect))
        object.__setattr__(self, "applicability", TriState.from_value(self.applicability))
        object.__setattr__(self, "card_id", _text(self.card_id, 80))
        object.__setattr__(self, "card_version", _text(self.card_version, 40))
        object.__setattr__(self, "card_ids", _strings(self.card_ids) or (self.card_id,))
        object.__setattr__(self, "allowed_effects", frozenset(_effect(item) for item in _values(self.allowed_effects)))
        object.__setattr__(self, "subject", _text(self.subject or "*", 300))
        object.__setattr__(self, "predicate", _text(self.predicate, 200))
        object.__setattr__(self, "input_partition", _text(self.input_partition or "all", 120))
        object.__setattr__(self, "boundary", _text(self.boundary or "public", 160))
        object.__setattr__(self, "message", _text(self.message, 480))
        object.__setattr__(self, "reason", _text(self.reason, 300))
        object.__setattr__(self, "material_evidence", _text(self.material_evidence, 240))
        object.__setattr__(self, "step", max(0, int(self.step)))
        estimated = self.token_estimate or _lexical_tokens(self.message)
        object.__setattr__(self, "token_estimate", max(0, int(estimated)))

    @property
    def effect_name(self) -> str | None:
        return self.effect.value if isinstance(self.effect, Effect) else None

    @property
    def cards(self) -> tuple[str, ...]:
        return self.card_ids

    @property
    def receipt(self) -> DiagnosticReceipt | None:
        return self.diagnostic_receipt

    @property
    def dedup(self) -> tuple[str, str, str, str]:
        return self.dedup_key

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "pattern-intervention/1",
            "route": self.route.value,
            "card_id": self.card_id,
            "card_ids": list(self.card_ids),
            "card_version": self.card_version,
            "instance_id": self.instance_id,
            "step": self.step,
            "subject": self.subject,
            "predicate": self.predicate,
            "input_partition": self.input_partition,
            "boundary": self.boundary,
            "priority": self.priority,
            "message": self.message,
            "applicability": self.applicability.value,
            "effect": self.effect.value if isinstance(self.effect, Effect) else None,
            "allowed_effects": sorted(item.value for item in self.allowed_effects),
            "dedup_key": list(self.dedup_key),
            "diagnostic_receipt": self.diagnostic_receipt.to_dict() if self.diagnostic_receipt else None,
            "observation": _jsonable(self.observation),
            "question": self.question.to_dict() if self.question else None,
            "question_spec": {
                "qid": self.question_spec.qid,
                "answer_kind": str(self.question_spec.answer_kind.value),
                "answer_schema": _jsonable(self.question_spec.answer_schema),
                "priority": self.question_spec.priority,
            } if self.question_spec else None,
            "reason": self.reason,
            "material_evidence": self.material_evidence,
            "token_estimate": self.token_estimate,
            "merged": self.merged,
            "would_surface": self.would_surface,
        }

    as_dict = to_dict

    # Lightweight mapping compatibility makes the record convenient in
    # journal/report adapters without giving it mutable dictionary semantics.
    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def get(self, key: str, default: object = None) -> object:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()


@dataclass(frozen=True, slots=True)
class _Candidate:
    record: InterventionRecord
    card: PatternCard
    severity: int
    evidence_key: str


def _matching_obligations(card: PatternCard, facts: RevisionFacts) -> tuple[object, ...]:
    """Match a card to explicit ticket facts, including host-check ids."""

    operation = card.operation
    terms = [
        str(operation.get("kind", "")),
        *[str(item) for item in _values(operation.get("check_templates", ()))],
        *[str(item) for item in _values(operation.get("keywords", ()))],
        card.name,
    ]
    # The short starter-card names are intentionally expanded here.  The
    # ticket may say "payload remains usable" while the frozen check says
    # PRESERVE_INPUT; both are the same binding, not two independent guesses.
    expansions = {
        "01": ("call shape", "compat", "legacy", "preserve_call_shape", "old call"),
        "02": ("entry", "export", "public", "propagat", "api", "import"),
        "07": ("validation", "preserve_existing_validation", "validator"),
        "08": ("caller", "payload", "input", "preserve_input", "mutation", "retry"),
        "11": ("optional", "omitted", "null", "empty", "default", "partition"),
        "28": ("test", "regression", "public", "green", "witness"),
    }
    terms.extend(expansions.get(card.card_id, ()))
    rows: list[object] = []
    for row in facts.obligations:
        text = _obligation_text(row).lower()
        checks = " ".join(item.lower() for item in _obligation_checks(row))
        explicit = row.get("pattern_card") if isinstance(row, Mapping) else None
        if explicit is not None:
            if str(explicit) == card.card_id:
                rows.append(row)
            continue
        if any(term.strip().lower() and (term.strip().lower() in text or term.strip().lower() in checks) for term in terms):
            rows.append(row)
    return tuple(rows)


def _matching_receipts(
    card: PatternCard,
    facts: RevisionFacts,
    obligations: Sequence[object],
) -> tuple[object, ...]:
    check_ids: set[str] = set()
    for row in obligations:
        check_ids.update(_obligation_checks(row))
    values = _receipt_values(facts.receipts)
    values += _receipt_values(facts.after_receipts)
    result: list[object] = []
    for item in values:
        if isinstance(item, ObligationAssessment):
            result.extend(item.receipts)
            continue
        if isinstance(item, CheckReceipt):
            if not check_ids or item.check_id in check_ids:
                result.append(item)
            continue
        if isinstance(item, DiagnosticReceipt):
            result.append(item)
            continue
        if isinstance(item, Mapping):
            item_card = item.get("card_id", item.get("card"))
            item_check = str(item.get("check_id", item.get("check", "")))
            item_kind = str(item.get("kind", "")).lower()
            if item_card is not None and str(item_card) != card.card_id:
                continue
            if item_check and check_ids and item_check not in check_ids:
                continue
            if item_card is not None or item_check or item_kind in {"test", "public_test", "check"} or "status" in item or "result_class" in item:
                result.append(item)
    unique: list[object] = []
    seen_ids: set[int] = set()
    for item in result:
        marker = id(item)
        if marker in seen_ids:
            continue
        seen_ids.add(marker)
        unique.append(item)
    return tuple(unique)


def _receipt_status(value: object) -> str:
    if isinstance(value, DiagnosticReceipt):
        return value.status.upper()
    return _test_status(value)


def _receipt_has_violation(value: object) -> bool:
    if isinstance(value, ObligationAssessment):
        return value.verdict == UNSATISFIED or value.reason == COUNTEREXAMPLE
    return _receipt_status(value) in {"RED", str(COUNTEREXAMPLE).upper(), "COUNTEREXAMPLE", "UNSATISFIED"}


def _receipt_has_bound_pass(value: object) -> bool:
    if isinstance(value, ObligationAssessment):
        return value.verdict == SATISFIED
    return _receipt_status(value) in {"GREEN", str(PASS).upper(), "PASS", "SATISFIED"}


def _make_diagnostic_receipt(
    card: PatternCard,
    instance: BoundPatternInstance,
    facts: RevisionFacts,
    receipts: Sequence[object],
) -> DiagnosticReceipt:
    current_fixture = facts.fixture
    current_environment = facts.environment
    for value in receipts:
        if isinstance(value, DiagnosticReceipt):
            return value.for_revision(facts.revision, current_fixture, current_environment)
        if isinstance(value, CheckReceipt):
            return DiagnosticReceipt.from_check_receipt(
                card.card_id, instance.instance_id, value,
                current_revision=facts.revision,
                current_fixture=current_fixture,
                current_environment=current_environment,
            )
        if isinstance(value, ObligationAssessment):
            for receipt in value.receipts:
                return DiagnosticReceipt.from_check_receipt(
                    card.card_id, instance.instance_id, receipt,
                    current_revision=facts.revision,
                    current_fixture=current_fixture,
                    current_environment=current_environment,
                )
        if isinstance(value, Mapping):
            status = str(value.get("status", value.get("result_class", value.get("result", "WOULD_EXECUTE"))))
            snapshot = value.get("snapshot_digest", value.get("snapshot", facts.revision))
            fixture = value.get("fixture_digest", value.get("fixture", current_fixture))
            environment = value.get("environment_digest", value.get("environment", current_environment))
            fresh = Freshness.STALE if (
                facts.revision and snapshot and _identity(snapshot) != _identity(facts.revision)
            ) else Freshness.FRESH
            return DiagnosticReceipt(
                card.card_id, instance.instance_id, status,
                snapshot, fixture, environment, fresh,
                value.get("observed"), value.get("expected"), value.get("counterexample"),
                value.get("evidence_refs", value.get("evidence", ())),
                None, str(value.get("reason", "")), False,
            )
    return DiagnosticReceipt(
        card_id=card.card_id,
        instance_id=instance.instance_id,
        status="WOULD_EXECUTE",
        revision_digest=facts.revision,
        fixture_digest=_identity(current_fixture),
        environment_digest=_identity(current_environment),
        freshness=Freshness.UNKNOWN,
        reason="bound operation would run through the existing host check executor",
        would_execute=True,
    )


def _question_for(
    card: PatternCard,
    instance: BoundPatternInstance,
    facts: RevisionFacts,
) -> tuple[QuestionSpec, Question, str]:
    operation = card.operation
    qid = f"pattern.{card.card_id}.{instance.instance_id[-16:]}"
    subject = instance.subject if instance.subject != "*" else card.card_id
    question_text = str(operation.get("question", "Bind the public contract semantics before surfacing this pattern."))
    prompt = question_text.replace("{subject}", subject)
    spec = QuestionSpec(
        qid=qid,
        trigger=lambda _snapshot: True,
        answer_kind=MODEL_TYPED,
        answer_schema=Mapping,
        subject_refs=(subject,),
        priority=10,
    )
    question = Question(
        qid=qid,
        spec_qid=qid,
        basis_revision=facts.revision,
        answer_kind=MODEL_TYPED,
        answer_schema=Mapping,
        subject_refs=(subject,),
    )
    return spec, question, prompt


def _observation_for(
    card: PatternCard,
    instance: BoundPatternInstance,
    facts: RevisionFacts,
) -> tuple[dict[str, object], str]:
    event = instance.events[0] if instance.events else None
    if card.card_id == "15":
        package = str(event.details.get("package", event.subject)) if event else "dependency"
        module = str(event.details.get("module", "")) if event else ""
        target = module.rsplit("/", 1)[-1] if module else "the failing target"
        message = (
            f"traceback crosses site-packages for {package}; characterize the installed "
            f"dependency behavior at {target} before assuming an API contract"
        )
        return ({
            "kind": "dependency_fact_suggestion",
            "classification": "characterization_only",
            "package": package,
            "module": module,
            "target": target,
            "traceback_crossed_site_packages": True,
            "authority": "GUIDANCE_SCAFFOLD",
            "normative": False,
            "revision": facts.revision,
            "evidence_refs": list(event.evidence_refs) if event else [],
        }, message)
    if card.card_id == "08":
        operation = event.details.get("operation", "write") if event else "write"
        alias = event.details.get("alias", "parameter alias") if event else "parameter alias"
        message = (
            f"possible input mutation: {operation} on {alias}; "
            "AST evidence does not establish dynamic evidence"
        )
        return ({
            "kind": "possible_input_mutation",
            "claim": "possible input mutation",
            "operation": operation,
            "alias": alias,
            "dynamic_evidence": False,
            "source": "AST",
            "revision": facts.revision,
        }, message)
    if card.card_id == "28":
        reason = str(event.details.get("reason", "test result unresolved")) if event else "test result unresolved"
        labels = {
            "red-to-green": ("change witness", "candidate test result changed from RED to GREEN"),
            "test already green": ("regression guard", "the same public test was already GREEN on the starter"),
            "persistent-fail": ("persistent fail", "the same public test remains RED after the candidate change"),
        }
        label, message = labels.get(reason, ("test discrimination", reason))
        return ({
            "kind": label.replace(" ", "_"),
            "classification": label,
            "before": event.before if event else (),
            "after": event.after if event else (),
            "revision": facts.revision,
            "evidence_refs": list(event.evidence_refs) if event else [],
        }, message)
    operation = str(card.operation.get("kind", card.name))
    message = str(card.operation.get("observation", f"pattern {card.card_id} observed: {operation}"))
    return ({
        "kind": "pattern_observation",
        "pattern": card.card_id,
        "operation": operation,
        "revision": facts.revision,
    }, message)


def _priority_and_reason(
    card: PatternCard,
    instance: BoundPatternInstance,
    matching_receipts: Sequence[object],
    route: Route,
) -> tuple[int, str]:
    if any(_receipt_has_violation(item) for item in matching_receipts):
        return 300, "witnessed violation"
    if route is Route.DIAGNOSTIC_RECEIPT:
        return 200, "contract-critical diagnostic"
    if route is Route.TYPED_QUESTION:
        return 100, "binding question"
    if route is Route.SCOPED_OBSERVATION:
        return 10, "optional scoped observation"
    return 0, "abstain"


def _effect_for_route(route: Route) -> Effect | None:
    return {
        Route.DIAGNOSTIC_RECEIPT: SUBMIT_EVIDENCE,
        Route.SCOPED_OBSERVATION: EMIT_OBSERVATION,
        Route.TYPED_QUESTION: EMIT_QUESTION,
        Route.ABSTAIN: None,
    }[route]


def _make_candidate(
    card: PatternCard,
    instance: BoundPatternInstance,
    facts: RevisionFacts,
) -> _Candidate | None:
    if not card.active:
        return None
    if instance.triggered is FALSE:
        return None
    event_unknown = any(event.certainty is UNKNOWN for event in instance.events)
    obligations = _matching_obligations(card, facts)
    matching_receipts = _matching_receipts(card, facts, obligations)

    route = Route.ABSTAIN
    effect: Effect | None = None
    diagnostic: DiagnosticReceipt | None = None
    observation: Mapping[str, object] | None = None
    question_spec: QuestionSpec | None = None
    question: Question | None = None
    message = ""
    reason = instance.reason

    if instance.contraindicated or instance.applicability is FALSE:
        route = Route.ABSTAIN
        reason = "contraindicated or not applicable"
        message = "pattern card abstained: applicability is FALSE"
    elif instance.applicability is UNKNOWN or event_unknown:
        route = Route.ABSTAIN
        reason = "unresolved AST or applicability"
        message = "pattern card abstained: deterministic binding is UNKNOWN"
    elif card.card_id == "08" and not obligations:
        # This is the key ownership boundary: the AST may report a possible
        # write, but it cannot promote that fact to a preservation verdict.
        route = Route.SCOPED_OBSERVATION
        observation, message = _observation_for(card, instance, facts)
        reason = "AST-only mutation observation; preservation obligation is not bound"
    elif card.card_id == "28":
        route = Route.SCOPED_OBSERVATION
        observation, message = _observation_for(card, instance, facts)
        reason = "test discrimination is an observation until a host verdict is bound"
    elif instance.oracle_binding is TRUE and card.oracle_requirement is not OracleRequirement.NONE:
        route = Route.DIAGNOSTIC_RECEIPT
        diagnostic = _make_diagnostic_receipt(card, instance, facts, matching_receipts)
        if diagnostic.stale:
            route = Route.ABSTAIN
            reason = "existing receipt is STALE for this revision"
            message = "pattern card abstained: host receipt is stale"
        else:
            message = str(card.operation.get("diagnostic", card.operation.get("message", "run the bound host diagnostic")))
            reason = "bound existing obligation; host execution owns the verdict"
    elif instance.oracle_binding is TRUE:
        route = Route.SCOPED_OBSERVATION
        observation, message = _observation_for(card, instance, facts)
        reason = "pattern fact has no oracle requirement; scoped observation only"
    else:
        # Missing/ambiguous contract binding is a question, except for the
        # mutation card handled above.  The question asks exactly one action;
        # it does not authorize an edit or a completion decision.
        route = Route.TYPED_QUESTION
        question_spec, question, message = _question_for(card, instance, facts)
        reason = "ticket obligation or project invariant is not bound"

    effect = _effect_for_route(route)
    if effect is not None and effect not in card.allowed_effects:
        route = Route.ABSTAIN
        effect = None
        question_spec = None
        question = None
        diagnostic = None
        observation = None
        reason = "requested route is outside card allowed_effects"
        message = "pattern card abstained: effect is not allowed"
    dedup_key = instance.dedup_key
    material = stable_digest({
        "facts": facts.material_digest,
        "event": [event.to_dict() for event in instance.events],
        "receipts": [_jsonable(item) for item in matching_receipts],
    })
    priority, priority_reason = _priority_and_reason(card, instance, matching_receipts, route)
    if priority_reason and priority_reason not in reason:
        reason = f"{priority_reason}; {reason}"
    max_tokens = min(120, card.budgets.max_tokens_per_item)
    message = _text(message, max_tokens * 4)
    record = InterventionRecord(
        route=route,
        card_id=card.card_id,
        card_version=card.version,
        instance_id=instance.instance_id,
        step=facts.step,
        subject=instance.subject,
        predicate=instance.predicate,
        input_partition=instance.input_partition,
        boundary=instance.boundary,
        priority=priority,
        message=message,
        applicability=instance.applicability,
        effect=effect,
        card_ids=(card.card_id,),
        allowed_effects=card.allowed_effects,
        dedup_key=dedup_key,
        diagnostic_receipt=diagnostic,
        observation=observation,
        question_spec=question_spec,
        question=question,
        reason=reason,
        material_evidence=material,
        token_estimate=min(max_tokens, _lexical_tokens(message)),
        would_surface=route is not Route.ABSTAIN,
    )
    return _Candidate(record, card, priority, material)


def _merge_candidates(candidates: Sequence[_Candidate]) -> _Candidate:
    ordered = sorted(candidates, key=lambda item: (-item.severity, item.card.card_id, item.record.message))
    primary = ordered[0]
    common_effects = set(primary.card.allowed_effects)
    for item in ordered[1:]:
        common_effects.intersection_update(item.card.allowed_effects)
    chosen = primary.record
    if chosen.effect not in common_effects:
        alternate = next((item.record for item in ordered if item.record.effect in common_effects), chosen)
        chosen = alternate
    merged_ids = tuple(sorted({card_id for item in ordered for card_id in (item.record.card_ids or (item.card.card_id,))}))
    allowed = frozenset(common_effects or set().union(*(set(item.card.allowed_effects) for item in ordered)))
    merged_message = chosen.message
    if len(merged_ids) > 1:
        merged_message = _text(f"{chosen.message} (merged cards: {', '.join(merged_ids)})", 480)
    merged_record = replace(
        chosen,
        card_ids=merged_ids,
        allowed_effects=allowed,
        message=merged_message,
        token_estimate=min(chosen.token_estimate, 120),
        merged=len(merged_ids) > 1,
    )
    return _Candidate(merged_record, primary.card, max(item.severity for item in ordered), primary.evidence_key)


class PatternCompiler:
    """Compile source/ticket facts into bounded shadow intervention records."""

    def __init__(
        self,
        cards: Iterable[PatternCard | Mapping[str, object]] | None = None,
        *,
        budget: BudgetSpec | Mapping[str, object] | None = None,
    ) -> None:
        self.cards = tuple(cards) if cards is not None else default_pattern_cards()
        self.cards = tuple(item if isinstance(item, PatternCard) else PatternCard.from_mapping(item) for item in self.cards)
        self.budget = BudgetSpec.from_mapping(budget or {})
        self._seen: dict[tuple[str, str, str, str], str] = {}

    def reset(self) -> None:
        self._seen.clear()

    def candidates(self, facts: RevisionFacts) -> tuple[InterventionRecord, ...]:
        rows: list[InterventionRecord] = []
        for card in sorted(self.cards, key=lambda item: item.card_id):
            candidate = _make_candidate(card, bind_card(card, facts), facts)
            if candidate is not None:
                rows.append(candidate.record)
        return tuple(rows)

    def compile(
        self,
        facts: RevisionFacts,
        *,
        previous: Iterable[InterventionRecord | Mapping[str, object]] = (),
        max_items: int = 1,
    ) -> list[InterventionRecord]:
        """Return at most one new surfaced item for this revision step."""

        if not isinstance(facts, RevisionFacts):
            raise TypeError("PatternCompiler.compile expects RevisionFacts")
        limit = min(1, max(0, int(max_items)), self.budget.max_new_items_per_step)
        if limit == 0 or not facts.events:
            return []
        for item in previous:
            if isinstance(item, InterventionRecord):
                self._seen.setdefault(item.dedup_key, item.material_evidence)
            elif isinstance(item, Mapping):
                key = tuple(item.get("dedup_key", ()))
                if len(key) == 4:
                    self._seen.setdefault(key, str(item.get("material_evidence", "")))

        by_key: dict[tuple[str, str, str, str], list[_Candidate]] = {}
        for card in sorted(self.cards, key=lambda item: item.card_id):
            instance = bind_card(card, facts)
            candidate = _make_candidate(card, instance, facts)
            if candidate is None:
                continue
            key = candidate.record.dedup_key
            if self._seen.get(key) == candidate.evidence_key:
                continue
            by_key.setdefault(key, []).append(candidate)
        if not by_key:
            return []
        merged = [_merge_candidates(rows) for rows in by_key.values()]
        merged.sort(key=lambda item: (-item.severity, -item.record.priority, item.record.dedup_key, item.record.card_id))
        selected = merged[:limit]
        for item in selected:
            self._seen[item.record.dedup_key] = item.evidence_key
        return [item.record for item in selected]

    evaluate = compile

    def shadow_evaluate(
        self,
        before: object,
        after: object,
        obligations: object = (),
        receipts: object = (),
        **kwargs: object,
    ) -> list[InterventionRecord]:
        previous = kwargs.pop("previous", ())
        facts = build_revision_facts(before, after, obligations, receipts, **kwargs)
        return self.compile(facts, previous=previous)


def load_pattern_cards(directory: str | Path | None = None) -> tuple[PatternCard, ...]:
    path = Path(directory) if directory is not None else Path(__file__).resolve().parent / "pattern_cards"
    if not path.is_dir():
        return ()
    cards: list[PatternCard] = []
    for filename in sorted(path.glob("*.json")):
        try:
            value = json.loads(filename.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid pattern-card file {filename.name}: {error}") from error
        if isinstance(value, Mapping) and value.get("schema", "pattern-card/1") == "pattern-card/1":
            cards.append(PatternCard.from_mapping(value))
    return tuple(sorted(cards, key=lambda card: card.card_id))


def default_pattern_cards() -> tuple[PatternCard, ...]:
    return load_pattern_cards()


def run_bound_diagnostics(
    root: str | Path,
    checks: Sequence[CheckDefinition | Mapping[str, object]],
    facts: RevisionFacts,
    *,
    cache: dict[str, CheckReceipt] | None = None,
    timeout: float = 8.0,
) -> tuple[DiagnosticReceipt, ...]:
    """Run already-registered host checks for the bound current revision.

    This helper is intentionally explicit.  ``shadow_evaluate`` never starts
    a subprocess; callers that have a host checkout and frozen check
    definitions may use this function, then feed the returned receipts back
    into the shadow compiler.
    """

    result: list[DiagnosticReceipt] = []
    for raw in checks:
        check = raw if isinstance(raw, CheckDefinition) else CheckDefinition.from_mapping(raw)
        receipt = run_check(
            root, check, snapshot=facts.revision, environment=facts.environment,
            cache=cache, timeout=timeout,
        )
        result.append(DiagnosticReceipt.from_check_receipt(
            check.obligation_id or check.check_id,
            check.instance_key(facts.revision, facts.environment),
            receipt,
            current_revision=facts.revision,
            current_fixture=check.fixture,
            current_environment=facts.environment,
        ))
    return tuple(result)


def _journal_step(record: Mapping[str, object]) -> int:
    candidates: list[object] = [record.get("step")]
    for key in ("record", "receipt", "observation"):
        nested = record.get(key)
        if isinstance(nested, Mapping):
            candidates.extend((nested.get("step"), nested.get("turn"), nested.get("workspace_revision")))
    candidates.append(record.get("sequence", 0))
    for value in candidates:
        if isinstance(value, bool):
            continue
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _journal_source(record: Mapping[str, object]) -> tuple[str, str] | None:
    source = record.get("source")
    if isinstance(source, Mapping):
        path = source.get("path")
        content = source.get("content")
        if isinstance(path, str) and isinstance(content, str):
            return path, content
    observation = record.get("observation")
    if isinstance(observation, Mapping):
        evidence = observation.get("read_evidence")
        if isinstance(evidence, Mapping):
            path = evidence.get("path")
            content = evidence.get("content")
            if isinstance(path, str) and isinstance(content, str):
                return path, content
        action = str(observation.get("action", ""))
        raw_content = observation.get("observation")
        if action.startswith("read ") and isinstance(raw_content, str) and raw_content:
            return action[5:].strip(), raw_content
    record_value = record.get("record")
    if isinstance(record_value, Mapping):
        nested = _journal_source(record_value)
        if nested is not None:
            return nested
    return None


def _journal_test_result(record: Mapping[str, object]) -> tuple[str, str] | None:
    receipt = record.get("receipt")
    if isinstance(receipt, Mapping) and str(receipt.get("kind", "")).lower() in {"test", "public_test"}:
        result = str(receipt.get("result_class", receipt.get("status", ""))).upper()
        if result:
            return str(receipt.get("intent", receipt.get("selector", "public_tests"))), result
    nested = record.get("record")
    if isinstance(nested, Mapping):
        action = nested.get("action_parameters")
        result = str(nested.get("result", "")).upper()
        observation = str(nested.get("observation", ""))
        target = action.get("target", "public_tests") if isinstance(action, Mapping) else "public_tests"
        if "run_tests" in str(nested.get("tool", "")) or "pytest" in observation or result in {"GREEN", "RED"}:
            if result in {"GREEN", "RED"}:
                return str(target), result
            if "ALL TESTS PASSED" in observation or "exit 0" in observation:
                return str(target), "GREEN"
            if "TESTS FAILED" in observation or "exit 1" in observation:
                return str(target), "RED"
    return None


def _journal_obligations(records: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for record in records:
        if record.get("kind") != "work_record_seeded":
            continue
        state = record.get("state")
        if not isinstance(state, Mapping):
            continue
        steps = state.get("steps", ())
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            continue
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            fields = step.get("fields", {})
            fields = fields if isinstance(fields, Mapping) else {}
            rows.append({
                "obligation_id": fields.get("obligation_id", step.get("id", "")),
                "id": fields.get("obligation_id", step.get("id", "")),
                "clause": fields.get("source_text", step.get("text", "")),
                "text": step.get("text", ""),
                "kind": fields.get("kind", ""),
                "targets": fields.get("mentioned_files", ()),
            })
    return tuple(rows)


def replay_journal(
    journal: str | Path,
    *,
    obligations: object = None,
    compiler: PatternCompiler | None = None,
) -> tuple[dict[str, object], ...]:
    """Replay allowed JSONL journal facts through the shadow compiler.

    This adapter deliberately reads only the named JSONL file.  It rejects the
    repository's excluded evidence directories and never writes, edits, or
    executes a journal action.  Each returned row records the journal
    sequence/step and the bounded intervention that would have surfaced.
    """

    path = Path(journal)
    forbidden = {"private", "hidden_tests", "near_miss", "gold", "oracle"}
    if any(part in forbidden for part in path.parts):
        raise ValueError("replay_journal refuses excluded evidence paths")
    try:
        raw_records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid journal {path}: {error}") from error
    records = tuple(item for item in raw_records if isinstance(item, Mapping))
    ticket_obligations = tuple(_record_values(obligations)) if obligations is not None else _journal_obligations(records)
    engine = compiler or PatternCompiler()
    current_sources: dict[str, str] = {}
    current_tests: dict[str, str] = {}
    surfaces: list[dict[str, object]] = []
    for record in records:
        step = _journal_step(record)
        source = _journal_source(record)
        if source is not None:
            source_path, content = source
            prior = current_sources.get(source_path)
            current_sources[source_path] = content
            if prior is not None and prior != content:
                rows = engine.shadow_evaluate(
                    {source_path: prior}, {source_path: content}, ticket_obligations,
                    {"after": tuple({"result_class": status, "check_id": name} for name, status in current_tests.items())},
                    step=step,
                )
                for row in rows:
                    surfaces.append({
                        "sequence": int(record.get("sequence", 0) or 0),
                        "step": step,
                        "path": source_path,
                        "card_ids": list(row.card_ids),
                        "route": row.route.value,
                        "effect": row.effect.value if row.effect is not None else None,
                        "message": row.message,
                        "record": row.to_dict(),
                    })
        test = _journal_test_result(record)
        if test is not None:
            name, status = test
            prior_status = current_tests.get(name)
            current_tests[name] = status
            if prior_status is not None:
                same_source = dict(current_sources)
                rows = engine.shadow_evaluate(
                    same_source, same_source, ticket_obligations,
                    {"before": ({"result_class": prior_status, "check_id": name},),
                     "after": ({"result_class": status, "check_id": name},)},
                    step=step,
                )
                for row in rows:
                    surfaces.append({
                        "sequence": int(record.get("sequence", 0) or 0),
                        "step": step,
                        "path": None,
                        "card_ids": list(row.card_ids),
                        "route": row.route.value,
                        "effect": row.effect.value if row.effect is not None else None,
                        "message": row.message,
                        "record": row.to_dict(),
                    })
    return tuple(surfaces)


def shadow_evaluate(
    before: object,
    after: object,
    obligations: object,
    receipts: object,
    **kwargs: object,
) -> list[InterventionRecord]:
    """Public shadow API: log what would be surfaced, without runtime wiring."""

    return PatternCompiler().shadow_evaluate(before, after, obligations, receipts, **kwargs)


def compile_pattern_cards(
    facts_or_events: RevisionFacts | Iterable[ChangeEvent | Mapping[str, object]],
    obligations: object = (),
    receipts: object = (),
    *,
    cards: Iterable[PatternCard | Mapping[str, object]] | None = None,
    step: int = 0,
    revision: object = None,
    metadata: Mapping[str, object] | None = None,
) -> list[InterventionRecord]:
    """Functional compiler entry point for callers that already have events."""

    if isinstance(facts_or_events, RevisionFacts):
        facts = facts_or_events
    else:
        events = tuple(
            item if isinstance(item, ChangeEvent) else ChangeEvent.from_mapping(item)
            for item in facts_or_events
            if isinstance(item, (ChangeEvent, Mapping))
        )
        facts = RevisionFacts(
            events=events,
            obligations=tuple(_record_values(obligations)),
            receipts=receipts,
            revision=_identity(revision) if revision is not None else stable_digest([event.to_dict() for event in events]),
            step=step,
            metadata=metadata or {},
        )
    return PatternCompiler(cards).compile(facts)


def _emit_pattern_card_event(journal: object, kind: str, **fields: object) -> None:
    """Send live attribution to the host journal without requiring a type import."""
    if journal is None:
        return
    emitter = journal if callable(journal) else getattr(journal, "emit", None)
    if callable(emitter):
        emitter(kind, **fields)


def render_live_state(
    before: object,
    after: object,
    obligations: object = (),
    receipts: object = (),
    *,
    step: int = 0,
    revision: object = None,
    metadata: Mapping[str, object] | None = None,
    previous: Iterable[InterventionRecord | Mapping[str, object]] = (),
    journal: object = None,
    cards: Iterable[PatternCard | Mapping[str, object]] | None = None,
) -> dict[str, object] | None:
    """Compile one bounded typed live projection for a StateFrame.

    The live arm is presentation-only: it records card facts and interventions
    in the frame, while host receipts remain the authority for diagnostics and
    completion.  Each surfaced intervention gets a ``pattern_card_fired``
    journal event so a run can attribute the materialized card to a revision.
    """
    if pattern_cards_mode() != "live":
        return None
    facts = build_revision_facts(
        before, after, obligations, receipts, revision=revision, step=step,
        metadata=metadata,
    )
    if not facts.events:
        return None
    compiler = PatternCompiler(cards)
    rows = compiler.compile(facts, previous=previous, max_items=1)
    surfaced = [row for row in rows if row.would_surface]
    card_ids = {card_id for row in rows for card_id in row.card_ids}
    card_rows = []
    for card in compiler.cards:
        if card.card_id not in card_ids:
            continue
        card_rows.append({
            "card_id": card.card_id,
            "version": card.version,
            "lifecycle": card.lifecycle.value,
            "oracle_requirement": card.oracle_requirement.value,
            "allowed_effects": sorted(effect.value for effect in card.allowed_effects),
            "budgets": card.budgets.to_dict(),
        })
    for row in surfaced:
        _emit_pattern_card_event(
            journal,
            "pattern_card_fired",
            producer="kernel_pattern_cards",
            card_id=row.card_id,
            card_ids=list(row.card_ids),
            card_version=row.card_version,
            instance_id=row.instance_id,
            route=row.route.value,
            effect=row.effect.value if row.effect is not None else None,
            revision=facts.revision,
            step=facts.step,
            dedup_key=list(row.dedup_key),
            dispatch=False,
        )
    state = {
        "schema": "stateframe-pattern-cards/1",
        "mode": "live",
        "revision": facts.revision,
        "before_revision": facts.before_revision,
        "step": facts.step,
        "events": [event.to_dict() for event in facts.events],
        "cards": card_rows,
        "interventions": [row.to_dict() for row in rows],
        "surfaced_count": len(surfaced),
        "budget": compiler.budget.to_dict(),
    }
    _emit_pattern_card_event(
        journal,
        "pattern_cards_rendered",
        producer="kernel_pattern_cards",
        revision=facts.revision,
        step=facts.step,
        event_count=len(facts.events),
        surfaced_count=len(surfaced),
        dispatch=False,
    )
    return state


DEFAULT_PATTERN_CARDS = default_pattern_cards()
STARTER_CARDS = DEFAULT_PATTERN_CARDS

# Compatibility spellings used by small offline probes and audit notes.
Applicability = TriState
TruthValue = TriState
ChangeEventType = EventKind
AllowedEffect = Effect
PatternCardLifecycle = Lifecycle
DiagnosticRoute = Route
load_cards = load_pattern_cards
extract_events = extract_change_events
bind_pattern_instance = bind_card

__all__.extend([
    "DEFAULT_PATTERN_CARDS", "STARTER_CARDS", "Effect", "Freshness", "OracleRequirement",
    "DIAGNOSTIC_RECEIPT", "Applicability", "TruthValue", "ChangeEventType",
    "AllowedEffect", "PatternCardLifecycle", "DiagnosticRoute", "load_cards",
    "extract_events", "bind_pattern_instance",
])
