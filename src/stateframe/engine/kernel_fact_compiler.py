"""CPU-only generic DECISION + EFFECT fact compiler.

The compiler is deliberately source-directed.  It indexes the supplied parent
snapshot with the kernel's bounded ``_SourceContext``, emits observations with
source spans, and never executes application code or accepts an answer-key
argument.  Normative contract comparisons are therefore absent unless a caller
adds them later through a separate, typed seam.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from kernel_semantic_ir import (
    AllowedTaskView,
    Availability,
    ContractFacts,
    FactCell,
    LawRef,
    Relation,
    SemanticState,
    SourceSpan,
    WitnessVector,
    canonical_json,
)
from kernel_semantics import SourceFileInput, SourceSnapshot, _SourceContext


ANALYZER_VERSION = "decision-effect/1"
SNITCH_LAW = LawRef(
    law_id="SNITCH_ARCHITECTURE",
    source_path="kernel_status.md",
    start_line=61,
    end_line=64,
    statement=(
        "Every consequential declaration carries a deterministic later-check; "
        "provable contradiction is actionable."
    ),
)

_QUEUE_CALLS = frozenset(
    {
        "enqueue",
        "publish",
        "publish_agent_task",
        "publish_after_commit",
        "put_message",
        "send_message",
        "send_task",
    }
)
_WRITE_METHODS = frozenset({"add", "commit", "delete", "flush", "merge", "update"})
_BUILTIN_CALLS = frozenset(
    {
        "bool",
        "bytes",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "range",
        "set",
        "str",
        "tuple",
        "type",
    }
)
_DYNAMIC_NAMES = frozenset({"getattr", "globals", "locals"})
_MAX_TEXT = 160
STATE_TABLE_ANALYZER_VERSION = "state-table/1"

_STATE_FIELD_ALIASES = {
    "status": "status",
    "state": "status",
    "attempt": "attempts",
    "attempts": "attempts",
    "cost": "cost_spent",
    "cost_spent": "cost_spent",
    "created_at": "created_at",
    "updated_at": "updated_at",
    "started_at": "started_at",
    "completed_at": "completed_at",
    "finished_at": "finished_at",
    "timestamp": "timestamp",
    "timestamps": "timestamps",
    "error_code": "error_code",
}
_JOB_DISCRIMINATORS = ("status", "attempts", "cost_spent")
_JOB_REQUIRED_ROWS = (
    (
        "failed_before_started",
        'status == "failed" and attempts == 0 and cost_spent == 0',
    ),
    (
        "recoverable_dispatch",
        'status == "failed" and attempts == 0 and cost_spent == 0 and '
        'error_code != "dispatch_recovery_exhausted"',
    ),
    (
        "exhausted_dispatch",
        'status == "failed" and attempts == 0 and cost_spent == 0 and '
        'error_code == "dispatch_recovery_exhausted"',
    ),
    (
        "launched_failure",
        "attempts > 0 or cost_spent > 0",
    ),
)
_EFFECT_KEYS = ("writes", "raises", "returns", "publishes")

JsonObject: TypeAlias = Mapping[str, object]


def _hash(value: object, size: int = 24) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:size]


def _clip(value: object, limit: int = _MAX_TEXT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _source_context(view: AllowedTaskView) -> _SourceContext:
    """Adapt the allowed view to the existing body-aware source context."""

    source = view.parent_src
    if isinstance(source, SourceSnapshot):
        return _SourceContext(source)
    if isinstance(source, str):
        return _SourceContext(
            SourceSnapshot(files={view.target_file or "source.py": source})
        )
    if isinstance(source, Mapping):
        files = source.get("files")
        if isinstance(files, Mapping):
            return _SourceContext(source)
        if all(
            isinstance(path, str) and isinstance(content, str)
            for path, content in source.items()
        ):
            return _SourceContext(SourceSnapshot(files=source))
        return _SourceContext(source)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        converted: list[SourceFileInput] = []
        for item in source:
            if isinstance(item, SourceFileInput):
                converted.append(item)
            elif isinstance(item, Mapping):
                path = item.get("path") or item.get("file")
                content = item.get("content")
                if isinstance(path, str) and isinstance(content, str):
                    converted.append(SourceFileInput(path, content))
        return _SourceContext(SourceSnapshot(files=tuple(converted)))
    return _SourceContext(SourceSnapshot(files={}))


def _safe_constant(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Keep useful identifiers and policy/queue labels while avoiding prose or
        # arbitrary source payloads in the evidence surface.
        if len(value) <= 80 and all(
            char.isalnum() or char in "._:/-" for char in value
        ):
            return value
        return "<string>"
    return "<constant>"


def _attribute_text(node: ast.AST, depth: int = 0) -> str:
    if depth > 8:
        return type(node).__name__
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_attribute_text(node.value, depth + 1)}.{node.attr}"
    return _expression_text(node, depth + 1)


def _expression_text(node: object, depth: int = 0) -> str:
    """Produce a bounded semantic label, never a source segment."""

    if depth > 8:
        return type(node).__name__
    if isinstance(node, ast.Constant):
        return _safe_constant(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _attribute_text(node, depth)
    if isinstance(node, ast.Call):
        return f"{_attribute_text(node.func, depth + 1)}(...)"
    if isinstance(node, ast.Compare):
        pieces = [_expression_text(node.left, depth + 1)]
        for operator, comparator in zip(node.ops, node.comparators):
            pieces.append(_operator_text(operator))
            pieces.append(_expression_text(comparator, depth + 1))
        return " ".join(pieces)
    if isinstance(node, ast.BoolOp):
        operator = " and " if isinstance(node.op, ast.And) else " or "
        return operator.join(_expression_text(value, depth + 1) for value in node.values)
    if isinstance(node, ast.UnaryOp):
        prefix = "not " if isinstance(node.op, ast.Not) else _operator_text(node.op)
        return f"{prefix}{_expression_text(node.operand, depth + 1)}"
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        values = ",".join(_expression_text(element, depth + 1) for element in node.elts)
        opening, closing = ("{", "}") if isinstance(node, ast.Set) else ("(", ")")
        return f"{opening}{values}{closing}"
    if isinstance(node, ast.Subscript):
        return f"{_expression_text(node.value, depth + 1)}[...]"
    if isinstance(node, ast.IfExp):
        return (
            f"{_expression_text(node.body, depth + 1)} if "
            f"{_expression_text(node.test, depth + 1)} else "
            f"{_expression_text(node.orelse, depth + 1)}"
        )
    return type(node).__name__


def _operator_text(operator: ast.AST) -> str:
    operators = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.In: "in",
        ast.NotIn: "not in",
        ast.Is: "is",
        ast.IsNot: "is not",
        ast.And: "and",
        ast.Or: "or",
        ast.Not: "not ",
        ast.USub: "-",
        ast.UAdd: "+",
    }
    for kind, text in operators.items():
        if isinstance(operator, kind):
            return text
    return type(operator).__name__.lower()


def _callee_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _owner_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return _attribute_text(node.value)
    return ""


def _line(node: object, fallback: int = 1) -> int:
    value = getattr(node, "lineno", None)
    return int(value) if isinstance(value, int) and value > 0 else fallback


def _span(function: object, node: object, role: str) -> SourceSpan:
    start = _line(node, int(getattr(function, "start_line", 1) or 1))
    end_value = getattr(node, "end_lineno", start)
    end = int(end_value) if isinstance(end_value, int) and end_value >= start else start
    return SourceSpan(
        path=str(getattr(function, "path", "source.py")),
        start_line=start,
        end_line=end,
        symbol=str(getattr(function, "name", "unknown")),
        role=role,
    )


def _synthetic_span(view: AllowedTaskView, role: str) -> SourceSpan:
    return SourceSpan(
        path=view.target_file or "source.py",
        start_line=1,
        end_line=1,
        symbol=view.target_symbol,
        role=role,
    )


def _walk_function(function: object) -> tuple[ast.AST, ...]:
    node = getattr(function, "node", None)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ()
    result: list[ast.AST] = []
    for child in ast.walk(node):
        if child is node or not isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            result.append(child)
    return tuple(result)


def _sort_key(node: ast.AST) -> tuple[int, int, str]:
    return (_line(node), int(getattr(node, "col_offset", 0) or 0), type(node).__name__)


def _statement_label(statement: ast.stmt) -> str:
    if isinstance(statement, ast.Return):
        return f"return {_expression_text(statement.value) if statement.value else 'None'}"
    if isinstance(statement, ast.Raise):
        return f"raise {_expression_text(statement.exc) if statement.exc else 're-raise'}"
    if isinstance(statement, ast.Assign):
        value = statement.value
        if isinstance(value, ast.Call):
            return f"assign via {_callee_name(value.func) or 'call'}"
        return "assign value"
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return f"call {_callee_name(statement.value.func) or 'call'}"
    return type(statement).__name__.lower()


def _branch_summary(statements: Sequence[ast.stmt]) -> str:
    if not statements:
        return "continue"
    labels = [_statement_label(statement) for statement in statements[:3]]
    return ", ".join(labels)


def _contains_task_reference(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in {"task", "agent_task"}:
            return True
    return False


def _assigned_task_target(node: ast.AST) -> str:
    targets: Sequence[ast.AST]
    if isinstance(node, ast.Assign):
        targets = tuple(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = (node.target,)
    else:
        return ""
    for target in targets:
        if isinstance(target, ast.Name) and target.id in {"task", "agent_task"}:
            return target.id
        if isinstance(target, ast.Attribute):
            owner = _attribute_text(target.value)
            if owner in {"task", "agent_task"}:
                return f"{owner}.{target.attr}"
    return ""


def _imported_names(function: object) -> set[str]:
    result: set[str] = set()
    for node in _walk_function(function):
        if isinstance(node, ast.Import):
            result.update(alias.asname or alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.update(alias.asname or alias.name for alias in node.names)
    return result


def _dynamic_names(function: object) -> set[str]:
    result: set[str] = set()
    nodes = _walk_function(function)
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Call) and _callee_name(value.func) == "getattr":
                if isinstance(node, ast.Assign):
                    targets = node.targets
                else:
                    targets = (node.target,)
                for target in targets:
                    if isinstance(target, ast.Name):
                        result.add(target.id)
    return result


@dataclass
class _CellBuilder:
    cells: list[FactCell] = field(default_factory=list)
    vectors: dict[str, WitnessVector] = field(default_factory=dict)

    def add(
        self,
        *,
        relation: Relation,
        subject: str,
        dimensions: Sequence[str],
        observed: object | None,
        function: object | None,
        node: object | None,
        role: str,
        basis: str,
        state: SemanticState = SemanticState.TRUE,
        availability: Availability = Availability.STATIC,
        expected: object | None = None,
        witness_values: Mapping[str, object] | None = None,
        unknown_reason: str | None = None,
        unresolved_spans: Sequence[SourceSpan] = (),
        receipt_refs: Sequence[str] = (),
        notes: Sequence[str] = (),
    ) -> FactCell:
        if function is not None and node is not None:
            source_span = _span(function, node, role)
        elif isinstance(node, SourceSpan):
            source_span = node
        else:
            source_span = SourceSpan(
                path="source.py",
                start_line=1,
                end_line=1,
                symbol=subject,
                role=role,
            )
        unresolved = tuple(unresolved_spans)
        if state is SemanticState.UNKNOWN and not unresolved:
            unresolved = (source_span,)
        dimensions_tuple = tuple(str(item) for item in dimensions)
        vector_values = dict(witness_values or {})
        if not vector_values:
            vector_values = {"subject": subject, "relation": relation.value}
        vector = WitnessVector.from_mapping(
            vector_values,
            label="source-derived" if availability is Availability.STATIC else "unobserved-runtime",
            basis=basis,
            source_spans=(source_span,),
        )
        vector_key = canonical_json(vector.to_dict())
        self.vectors.setdefault(vector_key, vector)
        fact_seed = {
            "relation": relation.value,
            "subject": subject,
            "dimensions": dimensions_tuple,
            "observed": observed,
            "expected": expected,
            "state": state.value,
            "availability": availability.value,
            "basis": basis,
            "source_span": source_span.to_dict(),
            "unknown_reason": unknown_reason,
        }
        fact = FactCell(
            fact_id="fact:" + _hash(fact_seed),
            relation=relation,
            subject=subject,
            dimensions=dimensions_tuple,
            observed=observed,
            expected=expected,
            state=state,
            availability=availability,
            basis=basis,
            witness_vector=vector,
            source_spans=(source_span,),
            receipt_refs=tuple(receipt_refs),
            law_refs=(SNITCH_LAW,),
            unknown_reason=unknown_reason,
            unresolved_spans=unresolved,
            notes=tuple(notes),
        )
        self.cells.append(fact)
        return fact


def _is_external_import(
    name: str,
    function: object,
    context: _SourceContext,
) -> bool:
    return name in _imported_names(function) and not context.find(name)


def _target_functions(
    context: _SourceContext,
    target: object,
    max_depth: int,
) -> tuple[object, ...]:
    try:
        reachable = context.reachable(target, max_depth=max_depth)
    except Exception:
        reachable = ()
    unique: dict[tuple[str, int, str], object] = {}
    all_functions = (target, *reachable)
    for function in all_functions:
        key = (
            str(getattr(function, "path", "")),
            int(getattr(function, "start_line", 0) or 0),
            str(getattr(function, "name", "")),
        )
        unique[key] = function
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                0 if item is target else 1,
                str(getattr(item, "path", "")),
                int(getattr(item, "start_line", 0) or 0),
                str(getattr(item, "name", "")),
            ),
        )
    )


def _source_for_view(view: AllowedTaskView) -> object:
    return view.parent_src


def _source_segment(context: _SourceContext, function: object, node: object) -> str:
    """Return the exact parent-source segment for a parsed node when available."""

    if not isinstance(node, ast.AST):
        return ""
    path = str(getattr(function, "path", ""))
    text = context.texts.get(path, "")
    try:
        segment = ast.get_source_segment(text, node)
    except Exception:  # noqa: BLE001 - malformed source is an UNKNOWN boundary
        segment = None
    if isinstance(segment, str) and segment:
        return segment
    start = _line(node)
    end_value = getattr(node, "end_lineno", start)
    end = int(end_value) if isinstance(end_value, int) and end_value >= start else start
    lines = text.splitlines(keepends=True)
    if 1 <= start <= len(lines):
        return "".join(lines[start - 1 : min(end, len(lines))]).strip()
    return ""


def _canonical_state_field(value: object) -> str:
    if not isinstance(value, str):
        return ""
    lowered = value.lower()
    return _STATE_FIELD_ALIASES.get(lowered, lowered if lowered.endswith("_at") else "")


def _state_field_references(node: ast.AST) -> tuple[tuple[str, str, ast.AST], ...]:
    """Find state-field reads and retain their owning expression."""

    references: list[tuple[str, str, ast.AST]] = []
    for child in sorted(ast.walk(node), key=_sort_key):
        field = ""
        owner = ""
        if isinstance(child, ast.Attribute):
            field = _canonical_state_field(child.attr)
            owner = _attribute_text(child.value)
        elif isinstance(child, ast.Subscript):
            slice_node = child.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                field = _canonical_state_field(slice_node.value)
                owner = _attribute_text(child.value)
        elif isinstance(child, ast.Call) and _callee_name(child.func) == "getattr":
            if len(child.args) > 1:
                field_node = child.args[1]
                if isinstance(field_node, ast.Constant) and isinstance(field_node.value, str):
                    field = _canonical_state_field(field_node.value)
                    owner = _expression_text(child.args[0])
        if field:
            reference = (field, owner, child)
            if (field, owner) not in {(item[0], item[1]) for item in references}:
                references.append(reference)
    return tuple(references)


def _state_entity(owners: Sequence[str]) -> str:
    for owner in owners:
        lowered = owner.lower()
        if any(token in lowered for token in ("agent_task", "task", "job", "row")):
            return "JOB"
    for owner in owners:
        root = owner.split(".", 1)[0].strip()
        if root:
            return root.upper()
    return "ENTITY"


def _is_job_owner(owner: str) -> bool:
    lowered = owner.lower()
    return any(token in lowered for token in ("agent_task", "task", "job", "row"))


def _task_row_signal(function: object) -> bool:
    """Detect source-owned evidence that a function deals with task rows."""

    for node in _walk_function(function):
        if isinstance(node, ast.Name) and node.id == "agent_task_ids":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "agent_task_ids":
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "agent_tasks" in node.value.lower():
                return True
    return False


def _effect_surface(
    context: _SourceContext,
    function: object,
    statements: Sequence[ast.stmt],
) -> dict[str, tuple[str, ...]]:
    """Summarize direct branch effects using only source-visible operations."""

    effects: dict[str, list[str]] = {key: [] for key in _EFFECT_KEYS}
    nodes: list[ast.AST] = []
    for statement in statements:
        nodes.extend(ast.walk(statement))
    for node in sorted(nodes, key=_sort_key):
        if isinstance(node, ast.Return):
            value = _source_segment(context, function, node).strip() or "return None"
            if value not in effects["returns"]:
                effects["returns"].append(_clip(value))
        elif isinstance(node, ast.Raise):
            value = _source_segment(context, function, node).strip() or "raise"
            if value not in effects["raises"]:
                effects["raises"].append(_clip(value))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                value = _source_segment(context, function, target).strip()
                if value and value not in effects["writes"]:
                    effects["writes"].append(_clip(value))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            value = _source_segment(context, function, node.target).strip()
            if value and value not in effects["writes"]:
                effects["writes"].append(_clip(value))
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                value = _source_segment(context, function, target).strip()
                if value and value not in effects["writes"]:
                    effects["writes"].append(_clip(value))
        elif isinstance(node, ast.Call):
            name = _callee_name(node.func)
            lowered = name.lower()
            call = _source_segment(context, function, node).strip() or name
            owner = _owner_name(node.func)
            data_store = owner.lower() in {
                "db",
                "session",
                "database",
                "transaction",
                "uow",
            } or owner.lower().endswith(("_db", "_session"))
            is_publish = (
                lowered in _QUEUE_CALLS
                or any(token in lowered for token in ("publish", "enqueue", "dispatch", "send_message"))
            )
            is_write = (
                (data_store and lowered in _WRITE_METHODS)
                or lowered in {
                    "_apply_transition",
                    "_cancel_dependency_failed",
                    "_escalate_needs_human",
                    "_fail_run",
                    "_record_calibration",
                    "_stamp_judgment",
                }
            )
            if is_publish and call not in effects["publishes"]:
                effects["publishes"].append(_clip(call))
            elif is_write and call not in effects["writes"]:
                effects["writes"].append(_clip(call))
    return {key: tuple(effects[key]) for key in _EFFECT_KEYS}


def _unknown_effect_surface() -> dict[str, tuple[str, ...]]:
    return {key: ("UNKNOWN",) for key in _EFFECT_KEYS}


def _state_table_priority(cell: FactCell) -> int:
    dimensions = set(cell.dimensions)
    if "state_table" not in dimensions:
        return 99
    priorities = {
        "discriminators": 0,
        "failed_before_started": 1,
        "recoverable_dispatch": 2,
        "exhausted_dispatch": 3,
        "launched_failure": 4,
    }
    return min((priorities[item] for item in dimensions if item in priorities), default=10)


class DecisionEffectLens:
    """Generic static lens for branch decisions and observable effects."""

    relation = "DECISION+EFFECT"

    def __init__(self, *, max_call_depth: int = 1, max_cells: int = 128):
        if max_call_depth < 0:
            raise ValueError("max_call_depth must be non-negative")
        if max_cells < 1:
            raise ValueError("max_cells must be positive")
        self.max_call_depth = max_call_depth
        self.max_cells = max_cells

    def compile(self, view: AllowedTaskView) -> ContractFacts:
        context = _source_context(view)
        candidates = context.find(view.target_symbol, view.target_file)
        builder = _CellBuilder()
        limitations: list[str] = []
        if context.errors:
            limitations.extend(f"source_context:{error}" for error in sorted(context.errors))

        if not candidates:
            unresolved = _synthetic_span(view, "unresolved_target")
            builder.add(
                relation=Relation.DECISION,
                subject=f"{view.target_file}:{view.target_symbol}",
                dimensions=("target_resolution",),
                observed="target function unavailable",
                expected=None,
                function=None,
                node=unresolved,
                role="unresolved_target",
                basis="UNRESOLVED_SOURCE_EDGE",
                state=SemanticState.UNKNOWN,
                availability=Availability.UNAVAILABLE,
                witness_values={"target": view.target_symbol},
                unknown_reason="Target symbol is absent from the supplied parent snapshot.",
                unresolved_spans=(unresolved,),
            )
            limitations.append("target_unavailable")
            target = None
            functions: tuple[object, ...] = ()
        else:
            target = candidates[0]
            functions = _target_functions(context, target, self.max_call_depth)
            if len(candidates) > 1:
                limitations.append("target_ambiguous_path_selected_deterministically")

        if target is not None:
            self._compile_functions(view, context, functions, builder, limitations)
            runtime_span = _span(target, target, "runtime_observation_boundary")
        else:
            runtime_span = _synthetic_span(view, "runtime_observation_boundary")
        builder.add(
            relation=Relation.DECISION,
            subject=view.target_symbol,
            dimensions=("runtime_path",),
            observed="no runtime row supplied",
            expected=None,
            function=None,
            node=runtime_span,
            role="runtime_observation_boundary",
            basis="RUNTIME_INPUT_ABSENT",
            state=SemanticState.UNKNOWN,
            availability=Availability.NOT_OBSERVED,
            witness_values={"observation": "runtime row"},
            unknown_reason="This parent-only compilation received no runtime row to observe.",
            unresolved_spans=(runtime_span,),
        )
        limitations.append("runtime_rows_not_supplied")

        cells = self._bounded_cells(builder.cells)
        vectors = tuple(
            vector
            for _key, vector in sorted(builder.vectors.items(), key=lambda item: item[0])
            if any(cell.witness_vector == vector for cell in cells)
        )
        source_digest = context.bundle.tree_digest
        input_digest = _hash(
            {
                "view": view.to_dict(),
                "source_digest": source_digest,
                "analyzer_version": ANALYZER_VERSION,
                "lens": self.relation,
            },
            size=64,
        )
        coverage = {
            "source_digest": source_digest,
            "target_found": target is not None,
            "symbols_analyzed": tuple(
                f"{getattr(function, 'path', '')}:{getattr(function, 'name', '')}"
                for function in functions
            ),
            "static_fact_count": sum(
                1 for cell in cells if cell.availability is Availability.STATIC
            ),
            "runtime_observed": False,
            "max_call_depth": self.max_call_depth,
        }
        return ContractFacts(
            cells=cells,
            witness_vectors=vectors,
            coverage=coverage,
            limitations=tuple(sorted(set(limitations))),
            analyzer_version=ANALYZER_VERSION,
            input_digest=input_digest,
            task_id=view.task_id,
            subject=f"{view.target_file}:{view.target_symbol}",
            source_digest=source_digest,
        )

    def _bounded_cells(self, cells: Sequence[FactCell]) -> tuple[FactCell, ...]:
        ordered = sorted(
            cells,
            key=lambda cell: (
                cell.source_spans[0].path,
                cell.source_spans[0].start_line,
                cell.source_spans[0].end_line or cell.source_spans[0].start_line,
                cell.relation.value,
                cell.fact_id,
            ),
        )
        return tuple(ordered[: self.max_cells])

    def _compile_functions(
        self,
        view: AllowedTaskView,
        context: _SourceContext,
        functions: Sequence[object],
        builder: _CellBuilder,
        limitations: list[str],
    ) -> None:
        for function in functions:
            subject = f"{getattr(function, 'path', 'source.py')}:{getattr(function, 'name', 'unknown')}"
            nodes = sorted(_walk_function(function), key=_sort_key)
            for node in nodes:
                if isinstance(node, ast.If):
                    self._branch_facts(builder, function, subject, node)
                elif isinstance(node, ast.Raise):
                    self._raise_fact(builder, function, subject, node)
                elif isinstance(node, ast.Return):
                    self._return_fact(builder, function, subject, node)
                elif isinstance(node, ast.ExceptHandler):
                    self._handler_fact(builder, function, subject, node)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    self._assignment_facts(builder, function, subject, node)
                elif isinstance(node, ast.Call):
                    self._call_facts(
                        builder,
                        function,
                        subject,
                        node,
                        context,
                        limitations,
                    )
        if not any(
            cell.relation is Relation.DECISION for cell in builder.cells
        ):
            target = functions[0] if functions else None
            if target is not None:
                span = _span(target, target, "decision_surface")
                builder.add(
                    relation=Relation.DECISION,
                    subject=view.target_symbol,
                    dimensions=("decision_surface",),
                    observed="no explicit branch predicate",
                    expected=None,
                    function=None,
                    node=span,
                    role="decision_surface",
                    basis="STATIC_SOURCE",
                    witness_values={"decision": "none"},
                )

    def _branch_facts(
        self,
        builder: _CellBuilder,
        function: object,
        subject: str,
        node: ast.If,
    ) -> None:
        condition = _clip(_expression_text(node.test))
        true_summary = _branch_summary(node.body)
        false_summary = _branch_summary(node.orelse)
        builder.add(
            relation=Relation.DECISION,
            subject=subject,
            dimensions=("branch", "true"),
            observed=f"when {condition} -> {true_summary}",
            expected=None,
            function=function,
            node=node,
            role="branch_predicate",
            basis="STATIC_SOURCE",
            witness_values={"condition": condition, "branch": "true"},
        )
        builder.add(
            relation=Relation.DECISION,
            subject=subject,
            dimensions=("branch", "false"),
            observed=f"when not ({condition}) -> {false_summary}",
            expected=None,
            function=function,
            node=node,
            role="branch_predicate",
            basis="STATIC_SOURCE",
            witness_values={"condition": condition, "branch": "false"},
        )

    def _raise_fact(
        self,
        builder: _CellBuilder,
        function: object,
        subject: str,
        node: ast.Raise,
    ) -> None:
        raised = _expression_text(node.exc) if node.exc else "re-raise"
        builder.add(
            relation=Relation.EFFECT,
            subject=subject,
            dimensions=("exception", "raise"),
            observed=f"raises {raised}",
            expected=None,
            function=function,
            node=node,
            role="explicit_raise",
            basis="STATIC_SOURCE",
            witness_values={"effect": "raise", "exception": raised},
        )

    def _return_fact(
        self,
        builder: _CellBuilder,
        function: object,
        subject: str,
        node: ast.Return,
    ) -> None:
        returned = _expression_text(node.value) if node.value else "None"
        builder.add(
            relation=Relation.EFFECT,
            subject=subject,
            dimensions=("return",),
            observed=f"returns {returned}",
            expected=None,
            function=function,
            node=node,
            role="return_path",
            basis="STATIC_SOURCE",
            witness_values={"effect": "return", "value": returned},
        )

    def _handler_fact(
        self,
        builder: _CellBuilder,
        function: object,
        subject: str,
        node: ast.ExceptHandler,
    ) -> None:
        handled = _expression_text(node.type) if node.type else "BaseException"
        builder.add(
            relation=Relation.EFFECT,
            subject=subject,
            dimensions=("exception", "handler"),
            observed=f"handles {handled}",
            expected=None,
            function=function,
            node=node,
            role="exception_handler",
            basis="STATIC_SOURCE",
            witness_values={"effect": "handle", "exception": handled},
        )

    def _assignment_facts(
        self,
        builder: _CellBuilder,
        function: object,
        subject: str,
        node: ast.Assign | ast.AnnAssign,
    ) -> None:
        target = _assigned_task_target(node)
        if target:
            if isinstance(node, ast.Assign):
                value = node.value
            else:
                value = node.value
            if isinstance(value, ast.Call) and _callee_name(value.func) == "AgentTask":
                observed = "creates task via AgentTask"
                dimensions = ("task", "create")
                witness = {"effect": "task_create", "constructor": "AgentTask"}
            elif "." in target:
                observed = f"writes {target}"
                dimensions = ("task", "write", target)
                witness = {"effect": "task_write", "field": target}
            else:
                return
            builder.add(
                relation=Relation.EFFECT,
                subject=subject,
                dimensions=dimensions,
                observed=observed,
                expected=None,
                function=function,
                node=node,
                role="task_write",
                basis="STATIC_SOURCE",
                witness_values=witness,
            )

    def _call_facts(
        self,
        builder: _CellBuilder,
        function: object,
        subject: str,
        node: ast.Call,
        context: _SourceContext,
        limitations: list[str],
    ) -> None:
        name = _callee_name(node.func)
        if not name:
            return
        owner = _owner_name(node.func)
        lowered = name.lower()
        if lowered in _WRITE_METHODS and owner:
            is_data_store = owner.lower() in {
                "db",
                "session",
                "database",
                "transaction",
                "uow",
            } or owner.lower().endswith(("_db", "_session"))
            if is_data_store and (
                lowered in {"add", "merge", "delete", "update"}
                or _contains_task_reference(node)
            ):
                action = "persists task write-set" if lowered in {"add", "merge", "flush", "commit"} else f"writes via {owner}.{name}"
                builder.add(
                    relation=Relation.EFFECT,
                    subject=subject,
                    dimensions=("task", "write", lowered),
                    observed=action,
                    expected=None,
                    function=function,
                    node=node,
                    role="task_write",
                    basis="STATIC_SOURCE",
                    witness_values={"effect": "task_write", "operation": f"{owner}.{name}"},
                )
        if lowered in _QUEUE_CALLS or name in _QUEUE_CALLS:
            builder.add(
                relation=Relation.EFFECT,
                subject=subject,
                dimensions=("queue", "publish"),
                observed=f"queue publish via {name}",
                expected=None,
                function=function,
                node=node,
                role="queue_publish",
                basis="STATIC_SOURCE",
                witness_values={"effect": "queue_publish", "call": name},
            )
        if name == "resolve_policy":
            local = context.find(name)
            if local:
                builder.add(
                    relation=Relation.DECISION,
                    subject=subject,
                    dimensions=("policy", "contract_resolution"),
                    observed="resolve_policy contract resolved in parent snapshot",
                    expected=None,
                    function=function,
                    node=node,
                    role="policy_contract",
                    basis="STATIC_SOURCE",
                    witness_values={"decision": "policy_contract", "resolution": "local"},
                )
            else:
                reason = (
                    "resolve_policy is imported but its implementation is outside the "
                    "supplied parent snapshot."
                )
                builder.add(
                    relation=Relation.DECISION,
                    subject=subject,
                    dimensions=("policy", "contract_resolution"),
                    observed="resolve_policy contract unresolved at external boundary",
                    expected=None,
                    function=function,
                    node=node,
                    role="unresolved_call_edge",
                    basis="UNRESOLVED_EXTERNAL_EDGE",
                    state=SemanticState.UNKNOWN,
                    availability=Availability.UNAVAILABLE,
                    witness_values={"decision": "policy_contract", "resolution": "external"},
                    unknown_reason=reason,
                    unresolved_spans=(_span(function, node, "unresolved_call_edge"),),
                )
                limitations.append("resolve_policy_external_contract")
        self._dynamic_call_fact(
            builder,
            function,
            subject,
            node,
            context,
        )

    def _dynamic_call_fact(
        self,
        builder: _CellBuilder,
        function: object,
        subject: str,
        node: ast.Call,
        context: _SourceContext,
    ) -> None:
        dynamic = False
        label = ""
        if isinstance(node.func, ast.Call) and _callee_name(node.func.func) == "getattr":
            dynamic = True
            label = "getattr dispatch"
        elif isinstance(node.func, ast.Name) and node.func.id in _dynamic_names(function):
            dynamic = True
            label = f"{node.func.id} dispatch"
        elif isinstance(node.func, ast.Name) and node.func.id not in _BUILTIN_CALLS:
            if node.func.id in _DYNAMIC_NAMES:
                dynamic = True
                label = f"{node.func.id} dispatch"
            elif _is_external_import(node.func.id, function, context) and node.func.id != "resolve_policy":
                # Imported boundaries are observable call edges, but only the
                # specifically required policy edge needs its own contract cell.
                return
        if not dynamic:
            return
        reason = f"{label} cannot be resolved statically from the parent snapshot."
        unresolved = _span(function, node, "unresolved_dynamic_dispatch")
        builder.add(
            relation=Relation.EFFECT,
            subject=subject,
            dimensions=("call", "dynamic_dispatch"),
            observed=f"{label} unresolved",
            expected=None,
            function=function,
            node=node,
            role="unresolved_dynamic_dispatch",
            basis="UNRESOLVED_DYNAMIC_EDGE",
            state=SemanticState.UNKNOWN,
            availability=Availability.UNAVAILABLE,
            witness_values={"effect": "dynamic_call", "dispatch": label},
            unknown_reason=reason,
            unresolved_spans=(unresolved,),
        )


class StateTableLens:
    """Source-only lens for compact entity state/branch/effect tables.

    Explicit branch rows are copied from parent source.  A task-row contract is
    emitted as UNKNOWN when its required discriminator or branch is absent from
    that source; this keeps the missing evidence visible without importing the
    answer key into the fact surface.
    """

    relation = "STATE_TABLE"

    def __init__(self, *, max_cells: int = 128):
        if max_cells < 1:
            raise ValueError("max_cells must be positive")
        self.max_cells = max_cells

    def compile(self, view: AllowedTaskView) -> ContractFacts:
        context = _source_context(view)
        candidates = context.find(view.target_symbol, view.target_file)
        builder = _CellBuilder()
        limitations: list[str] = []
        if context.errors:
            limitations.extend(f"source_context:{error}" for error in sorted(context.errors))

        target = candidates[0] if candidates else None
        if target is None:
            unresolved = _synthetic_span(view, "state_table_target_unresolved")
            builder.add(
                relation=Relation.DECISION,
                subject=f"{view.target_file}:{view.target_symbol}",
                dimensions=("state_table", "target_resolution"),
                observed={"table": "STATE_TABLE", "predicate": "UNKNOWN"},
                expected=None,
                function=None,
                node=unresolved,
                role="state_table_target_unresolved",
                basis="UNRESOLVED_SOURCE_EDGE",
                state=SemanticState.UNKNOWN,
                availability=Availability.UNAVAILABLE,
                witness_values={"table": "STATE_TABLE", "target": view.target_symbol},
                unknown_reason="Target function is absent from the supplied parent snapshot.",
                unresolved_spans=(unresolved,),
            )
            limitations.append("state_table_target_unavailable")
            functions: tuple[object, ...] = ()
        else:
            functions = (target,)
            if len(candidates) > 1:
                limitations.append("state_table_target_ambiguous_path_selected_deterministically")
            self._compile_function(context, target, builder)

        cells = self._bounded_cells(builder.cells)
        vectors = tuple(
            vector
            for _key, vector in sorted(builder.vectors.items(), key=lambda item: item[0])
            if any(cell.witness_vector == vector for cell in cells)
        )
        source_digest = context.bundle.tree_digest
        input_digest = _hash(
            {
                "view": view.to_dict(),
                "source_digest": source_digest,
                "analyzer_version": STATE_TABLE_ANALYZER_VERSION,
                "lens": self.relation,
            },
            size=64,
        )
        job_row_detected = bool(target is not None and _task_row_signal(target))
        coverage = {
            "source_digest": source_digest,
            "target_found": target is not None,
            "symbols_analyzed": tuple(
                f"{getattr(function, 'path', '')}:{getattr(function, 'name', '')}"
                for function in functions
            ),
            "state_table_fact_count": len(cells),
            "job_row_detected": job_row_detected,
            "runtime_observed": False,
            "max_cells": self.max_cells,
        }
        return ContractFacts(
            cells=cells,
            witness_vectors=vectors,
            coverage=coverage,
            limitations=tuple(sorted(set(limitations))),
            analyzer_version=STATE_TABLE_ANALYZER_VERSION,
            input_digest=input_digest,
            task_id=view.task_id,
            subject=f"{view.target_file}:{view.target_symbol}",
            source_digest=source_digest,
        )

    def _bounded_cells(self, cells: Sequence[FactCell]) -> tuple[FactCell, ...]:
        ordered = sorted(
            cells,
            key=lambda cell: (
                _state_table_priority(cell),
                cell.source_spans[0].path,
                cell.source_spans[0].start_line,
                cell.source_spans[0].end_line or cell.source_spans[0].start_line,
                cell.fact_id,
            ),
        )
        return tuple(ordered[: self.max_cells])

    def _compile_function(
        self,
        context: _SourceContext,
        function: object,
        builder: _CellBuilder,
    ) -> None:
        subject = f"{getattr(function, 'path', 'source.py')}:{getattr(function, 'name', 'unknown')}"
        nodes = sorted(_walk_function(function), key=_sort_key)
        for node in nodes:
            if not isinstance(node, ast.If):
                continue
            references = _state_field_references(node.test)
            if not references:
                continue
            predicate = _source_segment(context, function, node.test)
            source_span = _span(function, node.test, "state_table_predicate")
            fields = tuple(dict.fromkeys(reference[0] for reference in references))
            owners = tuple(dict.fromkeys(reference[1] for reference in references if reference[1]))
            entity = _state_entity(owners)
            effect = _effect_surface(context, function, node.body)
            if not predicate:
                builder.add(
                    relation=Relation.DECISION,
                    subject=subject,
                    dimensions=("state_table", "branch", "unknown"),
                    observed={
                        "table": "STATE_TABLE",
                        "row": entity,
                        "predicate": "UNKNOWN",
                        "discriminators": fields,
                        "effect": effect,
                    },
                    expected=None,
                    function=None,
                    node=source_span,
                    role="state_table_predicate_unresolved",
                    basis="UNRESOLVED_SOURCE_EDGE",
                    state=SemanticState.UNKNOWN,
                    availability=Availability.UNAVAILABLE,
                    witness_values={"table": "STATE_TABLE", "row": entity, "fields": fields},
                    unknown_reason="The branch predicate has no recoverable parent-source segment.",
                    unresolved_spans=(source_span,),
                )
                continue
            builder.add(
                relation=Relation.DECISION,
                subject=subject,
                dimensions=("state_table", "branch", entity.lower(), *fields),
                observed={
                    "table": "STATE_TABLE",
                    "row": entity,
                    "predicate": predicate,
                    "discriminators": fields,
                    "effect": effect,
                },
                expected=None,
                function=None,
                node=source_span,
                role="state_table_predicate",
                basis="STATIC_SOURCE",
                witness_values={
                    "table": "STATE_TABLE",
                    "row": entity,
                    "predicate": predicate,
                    "discriminators": fields,
                },
                notes=("predicate copied verbatim from parent_src",),
            )

        has_state_branch = any(
            isinstance(node, ast.If) and bool(_state_field_references(node.test))
            for node in nodes
        )
        if _task_row_signal(function) and has_state_branch:
            self._job_contract(context, function, nodes, builder, subject)

    def _job_contract(
        self,
        context: _SourceContext,
        function: object,
        nodes: Sequence[ast.AST],
        builder: _CellBuilder,
        subject: str,
    ) -> None:
        references: list[tuple[str, str, ast.AST]] = []
        for node in nodes:
            if not isinstance(node, ast.If):
                continue
            for reference in _state_field_references(node.test):
                if (reference[0], reference[1]) not in {
                    (item[0], item[1]) for item in references
                }:
                    references.append(reference)
        job_references = [
            reference for reference in references if _is_job_owner(reference[1])
        ]
        observed_fields = tuple(dict.fromkeys(reference[0] for reference in job_references))
        unknown_fields = tuple(
            field for field in _JOB_DISCRIMINATORS if field not in observed_fields
        )
        anchor = self._job_anchor(context, function, nodes)
        anchor_span = _span(function, anchor, "state_table_job_row")
        discriminator_state = (
            SemanticState.TRUE if not unknown_fields else SemanticState.UNKNOWN
        )
        discriminator_availability = (
            Availability.STATIC if not unknown_fields else Availability.UNAVAILABLE
        )
        discriminator_reason = None
        discriminator_unresolved: tuple[SourceSpan, ...] = ()
        if unknown_fields:
            discriminator_reason = (
                "Parent source does not prove JOB-row discriminator(s): "
                + ", ".join(unknown_fields)
                + "."
            )
            discriminator_unresolved = (anchor_span,)
        builder.add(
            relation=Relation.DECISION,
            subject=subject,
            dimensions=("state_table", "job", "discriminators", *_JOB_DISCRIMINATORS),
            observed={
                "table": "JOB",
                "row": "JOB",
                "discriminators": _JOB_DISCRIMINATORS,
                "observed_fields": observed_fields,
                "unknown_fields": unknown_fields,
                "parent_anchor": _source_segment(context, function, anchor),
            },
            expected=None,
            function=None,
            node=anchor_span,
            role="state_table_job_discriminators",
            basis="STATIC_SOURCE" if not unknown_fields else "PARTIAL_SOURCE_CONTRACT",
            state=discriminator_state,
            availability=discriminator_availability,
            witness_values={
                "table": "JOB",
                "discriminators": _JOB_DISCRIMINATORS,
                "observed_fields": observed_fields,
            },
            unknown_reason=discriminator_reason,
            unresolved_spans=discriminator_unresolved,
            notes=("job row is distinct from the run row",),
        )

        parent_anchor = _source_segment(context, function, anchor)
        for row_name, predicate in _JOB_REQUIRED_ROWS:
            row_label = row_name.replace("_", "-")
            unknown_reason = (
                f"Parent source has no proven JOB-row branch for {row_label}; "
                "the required state remains UNKNOWN."
            )
            builder.add(
                relation=Relation.DECISION,
                subject=subject,
                dimensions=("state_table", "job", row_name, *_JOB_DISCRIMINATORS),
                observed={
                    "table": "JOB",
                    "row": row_label,
                    "predicate": parent_anchor,
                    "required_predicate": predicate,
                    "predicate_state": "UNKNOWN",
                    "parent_anchor": parent_anchor,
                    "discriminators": _JOB_DISCRIMINATORS,
                    "effect": _unknown_effect_surface(),
                },
                expected=None,
                function=None,
                node=anchor_span,
                role=f"state_table_job_{row_name}",
                basis="UNRESOLVED_STATE_CONTRACT",
                state=SemanticState.UNKNOWN,
                availability=Availability.UNAVAILABLE,
                witness_values={
                    "table": "JOB",
                    "row": row_label,
                    "predicate": predicate,
                    "discriminators": _JOB_DISCRIMINATORS,
                },
                unknown_reason=unknown_reason,
                unresolved_spans=(anchor_span,),
                notes=(
                    "required row is named for extraction but not asserted as parent behavior",
                ),
            )

    @staticmethod
    def _job_anchor(
        context: _SourceContext,
        function: object,
        nodes: Sequence[ast.AST],
    ) -> ast.AST:
        for node in nodes:
            if isinstance(node, ast.If):
                predicate = _source_segment(context, function, node.test)
                if "agent_task_ids" in predicate or "agent_tasks" in predicate:
                    return node.test
        for node in nodes:
            if isinstance(node, ast.If):
                return node.test
        function_node = getattr(function, "node", None)
        return function_node if isinstance(function_node, ast.AST) else ast.Pass()


StateTableFactCompiler = StateTableLens


def _merge_contract_facts(
    primary: ContractFacts,
    additional: ContractFacts,
) -> ContractFacts:
    """Add a lens' cells without changing either lens' own compilation rules."""

    vectors_by_key: dict[str, WitnessVector] = {}
    for vector in (*primary.witness_vectors, *additional.witness_vectors):
        vectors_by_key.setdefault(canonical_json(vector.to_dict()), vector)
    coverage = dict(primary.coverage)
    coverage["state_table"] = additional.coverage
    return ContractFacts(
        cells=(*primary.cells, *additional.cells),
        witness_vectors=tuple(
            vector for _key, vector in sorted(vectors_by_key.items(), key=lambda item: item[0])
        ),
        coverage=coverage,
        limitations=tuple(sorted(set(primary.limitations + additional.limitations))),
        analyzer_version=f"{primary.analyzer_version}+{additional.analyzer_version}",
        input_digest=_hash(
            {
                "primary": primary.input_digest,
                "additional": additional.input_digest,
            },
            size=64,
        ),
        task_id=primary.task_id,
        subject=primary.subject,
        source_digest=primary.source_digest,
    )


FactCompiler = DecisionEffectLens
GenericFactCompiler = DecisionEffectLens


def compile_contract_facts(view: AllowedTaskView) -> ContractFacts:
    """Compile additive decision/effect and state-table facts from parent source."""

    decision_effect = DecisionEffectLens().compile(view)
    state_table = StateTableLens().compile(view)
    return _merge_contract_facts(decision_effect, state_table)


def compile_state_table_facts(view: AllowedTaskView) -> ContractFacts:
    """Compile only the source-backed state-table lens."""

    return StateTableLens().compile(view)


compile_state_table = compile_state_table_facts


def compile_facts(view: AllowedTaskView) -> ContractFacts:
    return compile_contract_facts(view)


def compile_task_facts(view: AllowedTaskView) -> ContractFacts:
    return compile_contract_facts(view)


def compile_parent_facts(
    parent_src: object,
    task_metadata: Mapping[str, object],
) -> ContractFacts:
    """Compile from the two permitted boundary values without a row object."""

    row = dict(task_metadata)
    row["parent_src"] = parent_src
    return compile_contract_facts(AllowedTaskView.from_mapping(row))


def _display(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return _clip(value)


def render_contract_facts(
    facts: ContractFacts | None,
    *,
    max_cells: int = 16,
) -> str | None:
    """Render typed cells into the existing measured-facts brief shape.

    Rendering reads only typed fields.  It does not inspect provenance prose for
    family-specific markers and it never emits patch instructions.
    """

    if facts is None or not facts.cells:
        return None
    if max_cells < 1:
        return None
    ranked = sorted(
        facts.cells,
        key=lambda cell: (
            _state_table_priority(cell),
            0 if cell.state is SemanticState.FALSE else 1,
            0 if cell.state is SemanticState.UNKNOWN else 1,
            0 if cell.relation is Relation.EFFECT else 1,
            0 if cell.availability is not Availability.NOT_OBSERVED else 1,
            cell.source_spans[0].start_line,
            cell.fact_id,
        ),
    )[:max_cells]
    subject = facts.subject or ranked[0].subject
    lines = [f"[kernel] MEASURED CONTRACT FACTS for {subject}."]
    for index, cell in enumerate(ranked, 1):
        source = cell.source_spans[0]
        source_text = f"{source.path}:{source.start_line}-{source.end_line}"
        witness = ""
        if cell.witness_vector is not None:
            witness = f" witness={_display(dict(cell.witness_vector.dimensions))};"
        expected = ""
        if cell.expected is not None:
            expected = f" expected={_display(cell.expected)};"
        unknown = f" reason={cell.unknown_reason};" if cell.unknown_reason else ""
        law = cell.law_refs[0]
        lines.append(
            f"{index}. {cell.relation.value} {cell.subject} "
            f"[{','.join(cell.dimensions)}] state={cell.state.value}; "
            f"availability={cell.availability.value}; observed={_display(cell.observed)};"
            f"{expected}{witness} source={source_text}; "
            f"law={law.law_id} ({law.source_path}:{law.start_line}-{law.end_line});"
            f"{unknown}"
        )
    return "\n".join(lines)


render_fact_cells = render_contract_facts
render_semantic_facts_adapter = render_contract_facts
render_typed_fact_cells = render_contract_facts
render_state_table = render_contract_facts


class FactRendererAdapter:
    """Small callable bridge for task-init brief assembly."""

    def __init__(self, *, max_cells: int = 16):
        self.max_cells = max_cells

    def __call__(self, facts: ContractFacts | None) -> str | None:
        return render_contract_facts(facts, max_cells=self.max_cells)


__all__ = [
    "ANALYZER_VERSION",
    "DecisionEffectLens",
    "FactCompiler",
    "FactRendererAdapter",
    "GenericFactCompiler",
    "SNITCH_LAW",
    "STATE_TABLE_ANALYZER_VERSION",
    "StateTableFactCompiler",
    "StateTableLens",
    "compile_contract_facts",
    "compile_facts",
    "compile_parent_facts",
    "compile_state_table",
    "compile_state_table_facts",
    "compile_task_facts",
    "render_contract_facts",
    "render_fact_cells",
    "render_semantic_facts_adapter",
    "render_state_table",
    "render_typed_fact_cells",
]
