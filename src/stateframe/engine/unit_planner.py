"""Deterministic unit sequencing for single-file build tasks.

The planner is deliberately an adapter, not an execution engine.  It reads the task's
specification and test source, uses the test AST as the only dependency evidence, and
returns an immutable plan that a dispatcher can carry across context trims.

Cross-unit tests belong to every unit they exercise.  That overlap is intentional: a
handoff gate must prove the downstream unit against the same upstream contract that its
tests consume, rather than hiding a cross-class dependency in one arbitrary bucket.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

SCHEMA: Final = "unit-plan/1"
DEFAULT_PLAN_MAX_CHARS: Final = 4_000
DEFAULT_TRIM_MAX_CHARS: Final = 1_200
_IDENTIFIER: Final = re.compile(r"\b[A-Za-z_]\w*\b")
_CLASS_OR_FUNCTION: Final = re.compile(r"\b(class|def)\s+([A-Za-z_]\w*)")
LEGACY_HANDOFF: Final = "Unit N+1 starts only after all tests in unit N's gate pass."
JOINT_HANDOFF: Final = ("Shared tests are joint acceptance checks, not prerequisites for starting "
    "the other units they require. Implement dependencies together when needed; "
    "all required checks must pass for task completion.")


def joint_handoffs_enabled() -> bool:
    value = os.environ.get('UNIT_PLAN_JOINT_HANDOFFS', '0')
    if value not in {'0', '1'}:
        raise ValueError('UNIT_PLAN_JOINT_HANDOFFS must be 0 or 1')
    return value == '1'


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"unit planner requires non-empty {field}")
    return value


def _task_id(row: Mapping[str, object]) -> str:
    for key in ("sha", "provenance_sha", "task_id", "fn"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _digest({"spec": row.get("msg", row.get("spec", "")), "test_src": row.get("test_src", "")})


def _source_path(row: Mapping[str, object]) -> str:
    for key in ("path", "rel"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "<unknown>"


def _imported_symbols(tree: ast.Module, spec: str, target: str) -> tuple[tuple[str, str], ...]:
    """Return ``(local_name, source_name)`` imports named by the build spec."""
    spec_names = set(_IDENTIFIER.findall(spec))
    imported: list[tuple[str, str]] = []
    seen: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            source_name = alias.name
            local_name = alias.asname or source_name
            if source_name not in spec_names and local_name not in spec_names and source_name != target:
                continue
            if source_name not in seen:
                imported.append((local_name, source_name))
                seen.add(source_name)
    return tuple(imported)


def _local_functions(tree: ast.Module) -> Mapping[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node
    return MappingProxyType(result)


def _called_local_names(node: ast.AST, local_functions: Mapping[str, ast.AST]) -> tuple[str, ...]:
    return tuple(sorted({
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in local_functions
    }))


def _referenced_symbols(
    node: ast.AST,
    local_functions: Mapping[str, ast.AST],
    aliases: Mapping[str, str],
) -> frozenset[str]:
    """Collect imported symbols from a test and its local helper-call closure."""
    symbols: set[str] = set()
    pending: list[ast.AST] = [node]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        for child in ast.walk(current):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                source_name = aliases.get(child.id)
                if source_name is not None:
                    symbols.add(source_name)
        for helper_name in _called_local_names(current, local_functions):
            if helper_name not in visited:
                visited.add(helper_name)
                pending.append(local_functions[helper_name])
    return frozenset(symbols)


def _test_functions(tree: ast.Module) -> tuple[ast.AST, ...]:
    return tuple(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    )


def _symbol_kinds(spec: str) -> Mapping[str, str]:
    return MappingProxyType({name: kind for kind, name in _CLASS_OR_FUNCTION.findall(spec)})


def _topological_order(
    symbols: tuple[str, ...],
    references: Mapping[str, frozenset[str]],
    coverage: Mapping[str, int],
) -> tuple[str, ...]:
    """Order providers before consumers, with coverage resolving shared-test cycles."""
    position = {symbol: index for index, symbol in enumerate(symbols)}
    dependencies = {
        symbol: {other for other in references[symbol] if other in position and other != symbol}
        for symbol in symbols
    }
    ordered: list[str] = []
    remaining = set(symbols)
    while remaining:
        ready = sorted(
            (symbol for symbol in remaining if not dependencies[symbol] & remaining),
            key=lambda symbol: (position[symbol], symbol),
        )
        if not ready:
            # Shared cross-class tests can produce mutual evidence (for example, both
            # Job and JobQueue are used by every lifecycle test). The more widely
            # referenced symbol is the stable provider; source order breaks a tie.
            ready = sorted(
                remaining,
                key=lambda symbol: (-coverage[symbol], position[symbol], symbol),
            )
        ordered.extend(ready)
        remaining.difference_update(ready)
    return tuple(ordered)


def _bounded(text: str, max_chars: int, marker: str) -> str:
    if max_chars < len(marker) + 1:
        raise ValueError("max_chars is too small for bounded planner output")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(marker)].rstrip() + marker


@dataclass(frozen=True)
class PlannedUnit:
    """One admitted unit and its exact AST-derived test gate."""

    unit_id: str
    symbol: str
    kind: str
    gate_tests: tuple[str, ...]
    depends_on: tuple[str, ...]
    subcontract: str
    handoff_rule: str

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "symbol": self.symbol,
            "kind": self.kind,
            "gate_tests": list(self.gate_tests),
            "depends_on": list(self.depends_on),
            "subcontract": self.subcontract,
            "handoff_rule": self.handoff_rule,
        }


@dataclass(frozen=True)
class UnitPlan:
    """Immutable deterministic plan suitable for durable brief/trim transport."""

    task_id: str
    source_path: str
    units: tuple[PlannedUnit, ...]
    handoff_rule: str
    plan_digest: str
    handoff_policy: str = 'legacy'
    public_gate_paths: tuple[str, ...] = ()

    @property
    def is_single_unit(self) -> bool:
        return len(self.units) == 1

    def current(self, index: int) -> PlannedUnit:
        if not 0 <= index < len(self.units):
            raise IndexError(f"current unit index {index} outside plan of {len(self.units)} units")
        return self.units[index]

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema": SCHEMA,
            "task_id": self.task_id,
            "source_path": self.source_path,
            "units": [unit.to_dict() for unit in self.units],
            "handoff_rule": self.handoff_rule,
            "plan_digest": self.plan_digest,
        }
        if self.handoff_policy != 'legacy':
            result['handoff_policy'] = self.handoff_policy
        if self.public_gate_paths:
            result['public_gate_paths'] = list(self.public_gate_paths)
        return result


def derive_unit_plan(row: Mapping[str, object], *, joint_handoffs: bool = False) -> UnitPlan:
    """Derive a stable dependency-ordered plan from one build-task row."""
    from kernel_public_evidence import declared_public_check_paths

    public_gate_paths = tuple(declared_public_check_paths(dict(row)) or ())
    target = str(row.get("fn") or "").strip()
    if not target:
        task_id = _task_id(row)
        source_path = _source_path(row)
        plan_basis = {
            "schema": SCHEMA,
            "task_id": task_id,
            "source_path": source_path,
            "units": [],
            "handoff_rule": "Localize a product source before deriving unit sequencing.",
            "public_gate_paths": list(public_gate_paths),
            "target_bound": False,
        }
        return UnitPlan(
            task_id=task_id,
            source_path=source_path,
            units=(),
            handoff_rule="Localize a product source before deriving unit sequencing.",
            plan_digest=_digest(plan_basis),
            handoff_policy="legacy",
            public_gate_paths=public_gate_paths,
        )
    spec = _text(row.get("msg", row.get("spec", "")), "spec")
    test_src = _text(row.get("test_src", ""), "test_src")
    try:
        tree = ast.parse(test_src, filename="<unit-planner-test-src>")
    except SyntaxError as exc:
        raise ValueError(f"unit planner cannot parse test_src: {exc.msg}") from exc

    imported = _imported_symbols(tree, spec, target)
    if not imported:
        raise ValueError("unit planner found no spec-named imported class/function")
    aliases = dict(imported)
    local_functions = _local_functions(tree)
    tests = _test_functions(tree)
    if not tests:
        raise ValueError("unit planner found no test functions")

    test_refs = {
        node.name: _referenced_symbols(node, local_functions, aliases)
        for node in tests
    }
    source_order = tuple(source_name for _, source_name in imported)
    exercised = tuple(
        source_name for source_name in source_order
        if any(source_name in refs for refs in test_refs.values())
    )
    if not exercised:
        raise ValueError("unit planner found no exercised spec-named class/function")
    references = {
        symbol: frozenset(
            ref for test_name, refs in test_refs.items()
            if symbol in refs
            for ref in refs
        )
        for symbol in exercised
    }
    coverage = {
        symbol: sum(symbol in refs for refs in test_refs.values())
        for symbol in exercised
    }
    ordered = _topological_order(exercised, references, coverage)
    kinds = _symbol_kinds(spec)
    task_id = _task_id(row)
    units: list[PlannedUnit] = []
    for index, symbol in enumerate(ordered):
        gate_tests = tuple(
            test_name for test_name, refs in test_refs.items() if symbol in refs
        )
        depends_on = tuple(
            candidate for candidate in ordered[:index] if candidate in references[symbol]
        )
        unit_id = f"unit:{symbol}:g0"
        gate = ", ".join(public_gate_paths or gate_tests)
        subcontract = f"Implement {symbol} ({kinds.get(symbol, 'symbol')}); gate = [{gate}]."
        if index + 1 < len(ordered):
            next_symbol = ordered[index + 1]
            handoff = (
                f"Handoff: do not start {next_symbol} until every test in {symbol}'s gate passes."
            )
        else:
            handoff = f"Handoff: {symbol} is the final unit; its gate must pass for plan completion."
        if joint_handoffs:
            shared = [candidate for candidate in ordered if candidate != symbol
                      and candidate in references[symbol]]
            if shared:
                subcontract += f" Joint checks also exercise: {', '.join(shared)}."
            handoff = JOINT_HANDOFF
        units.append(PlannedUnit(unit_id, symbol, kinds.get(symbol, "symbol"), gate_tests,
                                 depends_on, subcontract, handoff))

    plan_basis = {
        "schema": SCHEMA,
        "task_id": task_id,
        "source_path": _source_path(row),
        "units": [unit.to_dict() for unit in units],
        "handoff_rule": JOINT_HANDOFF if joint_handoffs else LEGACY_HANDOFF,
    }
    if public_gate_paths:
        plan_basis['public_gate_paths'] = list(public_gate_paths)
    if joint_handoffs:
        plan_basis['handoff_policy'] = 'joint'
    return UnitPlan(
        task_id=task_id,
        source_path=_source_path(row),
        units=tuple(units),
        handoff_rule=JOINT_HANDOFF if joint_handoffs else LEGACY_HANDOFF,
        plan_digest=_digest(plan_basis),
        handoff_policy='joint' if joint_handoffs else 'legacy',
        public_gate_paths=public_gate_paths,
    )


def render_plan_block(plan: UnitPlan, *, max_chars: int = DEFAULT_PLAN_MAX_CHARS) -> str:
    """Render the bounded multi-unit PLAN block; single units add no plan noise."""
    if not plan.units:
        return _bounded(
            "[PLAN unit-plan/1]\n"
            f"task={plan.task_id} source={plan.source_path} digest={plan.plan_digest}\n"
            "target not localized; inspect the allowed scope and bind a product source "
            "before unit sequencing.\n[/PLAN]",
            max_chars,
            "\n…[PLAN TRUNCATED]",
        )
    if plan.is_single_unit and not plan.public_gate_paths:
        return ""
    lines = [
        "[PLAN unit-plan/1]",
        f"task={plan.task_id} source={plan.source_path} digest={plan.plan_digest}",
        f"handoff={plan.handoff_rule}",
    ]
    if plan.public_gate_paths:
        lines.insert(1, "public gate (run full files): " + ", ".join(plan.public_gate_paths))
    for index, unit in enumerate(plan.units, start=1):
        lines.extend((
            f"{index}. {unit.symbol} [{unit.unit_id}]",
            (f"   coverage test names (metadata): {', '.join(unit.gate_tests)}"
             if plan.public_gate_paths else f"   gate: {', '.join(unit.gate_tests)}"),
            f"   contract: {unit.subcontract}",
        ))
    lines.append("[/PLAN]")
    return _bounded("\n".join(lines), max_chars, "\n…[PLAN TRUNCATED]")


def render_brief(brief: str, plan: UnitPlan, *, max_chars: int = DEFAULT_PLAN_MAX_CHARS) -> str:
    """Attach the plan once to a brief, omitting it for a one-unit task (E394)."""
    brief = _text(brief, "brief")
    block = render_plan_block(plan, max_chars=max_chars)
    return brief if not block else f"{brief}\n\n{block}"


def render_current_unit_line(plan: UnitPlan, index: int = 0) -> str:
    """Render only current-unit state for repeated turn/trim delivery."""
    if not plan.units:
        return "[planner] target not localized; inspect the allowed scope and bind a product source before unit sequencing."
    unit = plan.current(index)
    public_gate = ("public gate (run full files): " + ", ".join(plan.public_gate_paths) + ". "
                   if plan.public_gate_paths else "")
    if plan.handoff_policy == 'joint':
        return ('[planner] ' + public_gate + 'Static decomposition: ' + ', '.join(unit.symbol for unit in plan.units)
                + '. Active work is not tracked by this static plan. ' + plan.handoff_rule)
    return (
        f"[planner] {public_gate}current unit {index + 1}/{len(plan.units)}: {unit.symbol} "
        f"({unit.unit_id}); gate={','.join(plan.public_gate_paths or unit.gate_tests)}"
    )


def render_trim_message(
    plan: UnitPlan,
    index: int = 0,
    *,
    world_cache_digest: str = "",
    dropped: int = 0,
    progress: str = "",
    max_chars: int = DEFAULT_TRIM_MAX_CHARS,
) -> str:
    """Compose trim-safe current-unit state with the E407 world-cache digest."""
    if dropped < 0:
        raise ValueError("dropped must be non-negative")
    parts = [
        f"[kernel] ({dropped} earlier exchanges trimmed to fit context. {progress or 'unit plan retained'})",
        render_current_unit_line(plan, index),
    ]
    if world_cache_digest.strip():
        parts.append(world_cache_digest.strip())
    return _bounded("\n".join(parts), max_chars, "\n…[TRIM DIGEST TRUNCATED]")


# Short aliases make the adapter easy to discover at the integration seam.
build_unit_plan = derive_unit_plan
render_plan = render_plan_block


__all__ = [
    "DEFAULT_PLAN_MAX_CHARS", "DEFAULT_TRIM_MAX_CHARS", "SCHEMA", "PlannedUnit", "UnitPlan",
    "build_unit_plan", "derive_unit_plan", "render_brief", "render_current_unit_line",
    "render_plan", "render_plan_block", "render_trim_message",
]
