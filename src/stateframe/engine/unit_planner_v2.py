"""Unit planner v2: index-ordered, multi-file, cycle-coalescing, gated per unit.

Chosen at the existing planner call sites only when ``STATEFRAME_UNIT_PLAN_V2=on`` and a
CrossFileIndex can be built for the task's files; otherwise the v1 planner
(``unit_planner.py``) answers unchanged.  v1 orders units by test co-occurrence, which is
symmetric, so a consumer covered by more tests lands before its provider.  v2 orders by
provider->consumer edges the index resolves (``refs(...).target``/``receiver``,
``users()``, ``Def.bases``).  A dependency cycle is coalesced into one joint unit (Tarjan
SCC), never deadlocked and never silently ordered.

Shared tests still belong to every unit they touch (joint acceptance), the plan is
immutable and digest-pinned, and v1's render functions work on it unchanged.

``unit_gate`` is the mechanism: a unit passes only when its gate tests pass and
``chains.seam_violations`` reports no error-kind violation in the unit's files.  The
"current unit" pointer advances only on a passing gate.

Index surface used (crossfile_index_api): defs(path=None), refs(path), users(qualname),
build(root, paths).  Everything is deterministic (sorted, no hash-seed dependence).
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

from unit_planner import (
    PlannedUnit, UnitPlan, _bounded, _digest, _local_functions, _source_path,
    _symbol_kinds, _task_id, _test_functions, _text,
)

MODE_ENV: Final = "STATEFRAME_UNIT_PLAN_V2"
SCHEMA_V2: Final = "unit-plan/2"
MAX_FILES: Final = 400
MAX_UNITS: Final = 150
GATED_HANDOFF: Final = ("Unit N+1 starts only after unit N's gate passes: its tests and the "
                        "cross-file seam check. Mutually dependent symbols form one joint unit.")
_SKIP_DIRS: Final = frozenset({"__pycache__", "node_modules", "venv", "site-packages", "build", "dist"})
_FAILED_LINE: Final = re.compile(r"^(?:FAILED|ERROR)\s+\S*?::(?:\w+::)?(\w+)", re.M)

SymKey = tuple[str, str]          # (path, top-level name)


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    return "on" if str(source.get(MODE_ENV, "off")).strip().lower() == "on" else "off"


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


# ---------------------------------------------------------------- plan types
@dataclass(frozen=True)
class PlannedUnitV2(PlannedUnit):
    """A v1 unit plus its files and, for a coalesced cycle, every member symbol."""

    files: tuple[str, ...] = ()
    members: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        out = super().to_dict()
        out["files"] = list(self.files)
        out["members"] = list(self.members)
        return out


@dataclass(frozen=True)
class UnitPlanV2(UnitPlan):
    index_revision: str = ""
    index_paths: tuple[str, ...] = ()

    def _basis(self) -> dict[str, object]:
        basis: dict[str, object] = {
            "schema": SCHEMA_V2,
            "task_id": self.task_id,
            "source_path": self.source_path,
            "units": [unit.to_dict() for unit in self.units],
            "handoff_rule": self.handoff_rule,
            "index_revision": self.index_revision,
            "index_paths": list(self.index_paths),
        }
        if self.public_gate_paths:
            basis["public_gate_paths"] = list(self.public_gate_paths)
        return basis

    def to_dict(self) -> dict[str, object]:
        return {**self._basis(), "plan_digest": self.plan_digest}


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    failing_tests: tuple[str, ...] = ()
    seam_violations: tuple[Mapping[str, object], ...] = ()
    seam_checked: bool = True


# ---------------------------------------------------------------- test AST
def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return (name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
            or any(p in ("tests", "test") for p in parts[:-1]))


def _module_path(module: str, paths: frozenset[str]) -> str | None:
    stem = "/".join(p for p in module.split(".") if p)
    if not stem:
        return None
    for cand in (f"{stem}.py", f"{stem}/__init__.py"):
        if cand in paths:
            return cand
    return None


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _import_aliases(tree: ast.Module, paths: frozenset[str], symbols: Mapping[SymKey, object]):
    """Return ``(symbol_alias, module_alias)`` for project imports in the test source.

    symbol_alias: local name -> (path, name); module_alias: dotted local name -> path.
    Handles ``from mod import Sym``, ``from pkg import mod`` and ``import pkg.mod [as m]``.
    """
    sym_alias: dict[str, SymKey] = {}
    mod_alias: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and not node.level and node.module:
            base = _module_path(node.module, paths)
            for alias in node.names:
                local = alias.asname or alias.name
                if base is not None and (base, alias.name) in symbols:
                    sym_alias[local] = (base, alias.name)
                    continue
                sub = _module_path(f"{node.module}.{alias.name}", paths)
                if sub is not None:
                    mod_alias[local] = sub
        elif isinstance(node, ast.Import):
            for alias in node.names:
                target = _module_path(alias.name, paths)
                if target is not None:
                    mod_alias[alias.asname or alias.name] = target
    return sym_alias, mod_alias


def _test_refs(node: ast.AST, helpers: Mapping[str, ast.AST], sym_alias: Mapping[str, SymKey],
               mod_alias: Mapping[str, str], symbols: Mapping[SymKey, object]) -> frozenset[SymKey]:
    """Project symbols a test touches, following local helper calls (like v1)."""
    found: set[SymKey] = set()
    pending, visited = [node], set()
    while pending:
        current = pending.pop()
        for child in ast.walk(current):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in sym_alias:
                    found.add(sym_alias[child.id])
            elif isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
                dotted = _dotted(child)
                for key, path in mod_alias.items():
                    if dotted and dotted.startswith(key + "."):
                        name = dotted[len(key) + 1:].split(".")[0]
                        if (path, name) in symbols:
                            found.add((path, name))
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                    and child.func.id in helpers and child.func.id not in visited:
                visited.add(child.func.id)
                pending.append(helpers[child.func.id])
    return frozenset(found)


# ---------------------------------------------------------------- index graph
def _top_level_symbols(index, paths: Sequence[str]) -> dict[SymKey, object]:
    out: dict[SymKey, object] = {}
    for path in paths:
        for d in index.defs(path):
            if d.kind in ("def", "class") and "." not in d.qualname:
                out.setdefault((d.path, d.qualname), d)
    return out


def _owner(spans: Mapping[str, list], path: str, lineno: int) -> SymKey | None:
    for start, end, key in spans.get(path, ()):
        if start <= lineno <= (end or start):
            return key
    return None


def _target_key(qualified: str | None, symbols: Mapping[SymKey, object]) -> SymKey | None:
    if not qualified or "::" not in qualified:
        return None
    path, _, qual = qualified.partition("::")
    key = (path, qual.split(".")[0])
    return key if key in symbols else None


def _edges(index, paths: Sequence[str], symbols: Mapping[SymKey, object]) -> set[tuple[SymKey, SymKey]]:
    """``(provider, consumer)`` pairs from refs/users/bases, restricted to ``symbols``."""
    spans: dict[str, list] = {}
    for key, d in symbols.items():
        spans.setdefault(key[0], []).append((d.lineno, d.end_lineno, key))
    edges: set[tuple[SymKey, SymKey]] = set()

    def add(provider: SymKey | None, consumer: SymKey | None) -> None:
        if provider and consumer and provider != consumer:
            edges.add((provider, consumer))

    for path in paths:
        for ref in index.refs(path):
            qualified = ref.target or (ref.receiver if ref.kind in ("attr", "key") else None)
            add(_target_key(qualified, symbols), _owner(spans, ref.path, ref.lineno))
    for key in sorted(symbols):
        for ref in index.users(f"{key[0]}::{key[1]}"):
            add(key, _owner(spans, ref.path, ref.lineno))
    by_name: dict[str, list[SymKey]] = {}
    for key in sorted(symbols):
        by_name.setdefault(key[1], []).append(key)
    for key, d in sorted(symbols.items()):
        for base in getattr(d, "bases", None) or ():
            hits = by_name.get(str(base).split("::")[-1].split(".")[-1], [])
            if len(hits) == 1:
                add(hits[0], key)
    return edges


def _sort_key(symbols: Mapping[SymKey, object]) -> Callable[[SymKey], tuple]:
    return lambda key: (key[0], symbols[key].lineno, key[1])


def _tarjan(nodes: Sequence[SymKey], deps: Mapping[SymKey, Sequence[SymKey]]) -> list[list[SymKey]]:
    """Strongly connected components over ``deps`` (node -> its providers)."""
    counter, stack, on_stack = [0], [], set()
    index_of: dict[SymKey, int] = {}
    low: dict[SymKey, int] = {}
    comps: list[list[SymKey]] = []

    def visit(node: SymKey) -> None:
        index_of[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in deps.get(node, ()):
            if nxt not in index_of:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index_of[nxt])
        if low[node] == index_of[node]:
            comp = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                comp.append(member)
                if member == node:
                    break
            comps.append(comp)

    for node in nodes:
        if node not in index_of:
            visit(node)
    return comps


def _order_components(comps: list[list[SymKey]], edges: set[tuple[SymKey, SymKey]],
                      sort_key: Callable[[SymKey], tuple]) -> list[list[SymKey]]:
    comp_of = {m: i for i, comp in enumerate(comps) for m in comp}
    needs: dict[int, set[int]] = {i: set() for i in range(len(comps))}
    for provider, consumer in edges:
        if provider in comp_of and consumer in comp_of and comp_of[provider] != comp_of[consumer]:
            needs[comp_of[consumer]].add(comp_of[provider])
    first = {i: min(sort_key(m) for m in comp) for i, comp in enumerate(comps)}
    ordered: list[list[SymKey]] = []
    remaining = set(needs)
    while remaining:
        ready = [i for i in remaining if not needs[i] & remaining]
        pick = min(ready, key=lambda i: first[i])          # SCC condensation is a DAG
        ordered.append(sorted(comps[pick], key=sort_key))
        remaining.discard(pick)
    return ordered


# ---------------------------------------------------------------- derive
def derive_unit_plan_v2(row: Mapping[str, object], index, paths: Sequence[str]) -> UnitPlanV2:
    """Derive the index-ordered plan; ValueError when there is nothing to plan."""
    spec = _text(row.get("msg", row.get("spec", "")), "spec")
    test_src = _text(row.get("test_src", ""), "test_src")
    try:
        tree = ast.parse(test_src, filename="<unit-planner-test-src>")
    except SyntaxError as exc:
        raise ValueError(f"unit planner cannot parse test_src: {exc.msg}") from exc
    paths = tuple(sorted(p for p in paths if not _is_test_path(p)))
    pathset = frozenset(paths)
    symbols = _top_level_symbols(index, paths)
    sym_alias, mod_alias = _import_aliases(tree, pathset, symbols)
    helpers = _local_functions(tree)
    tests = _test_functions(tree)
    if not tests:
        raise ValueError("unit planner found no test functions")
    test_refs = {t.name: _test_refs(t, helpers, sym_alias, mod_alias, symbols) for t in tests}
    touched = {key for refs in test_refs.values() for key in refs}
    if not touched:
        raise ValueError("unit planner found no tested project symbol")

    spec_kinds = _symbol_kinds(spec)
    wanted = set(touched) | {key for key in symbols if key[1] in spec_kinds}
    edges = _edges(index, paths, symbols)
    # providers of anything wanted are wanted too (transitively), so order is complete
    providers_of: dict[SymKey, set[SymKey]] = {}
    for provider, consumer in edges:
        providers_of.setdefault(consumer, set()).add(provider)
    queue = sorted(wanted)
    while queue:
        for provider in sorted(providers_of.get(queue.pop(), ())):
            if provider not in wanted:
                wanted.add(provider)
                queue.append(provider)
    if len(wanted) > MAX_UNITS:
        raise ValueError(f"unit planner refuses {len(wanted)} units (max {MAX_UNITS})")
    sort_key = _sort_key(symbols)
    nodes = sorted(wanted, key=sort_key)
    unit_edges = {(p, c) for p, c in edges if p in wanted and c in wanted}
    deps = {n: sorted((p for p, c in unit_edges if c == n), key=sort_key) for n in nodes}
    ordered = _order_components(_tarjan(nodes, deps), unit_edges, sort_key)

    names = [key[1] for comp in ordered for key in comp]
    label = {key: (key[1] if names.count(key[1]) == 1 else f"{key[0]}::{key[1]}")
             for comp in ordered for key in comp}
    try:
        from kernel_public_evidence import declared_public_check_paths
        public_gate_paths = tuple(declared_public_check_paths(dict(row)) or ())
    except ImportError:
        public_gate_paths = ()

    unit_ids: list[str] = []
    drafts = []
    for comp in ordered:
        symbol = "+".join(label[k] for k in comp)
        unit_ids.append(f"unit:{symbol}:g0")
        gate_tests = tuple(name for name, refs in test_refs.items() if refs & set(comp))
        drafts.append((comp, symbol, gate_tests))
    unit_of = {k: unit_ids[i] for i, (comp, _, _) in enumerate(drafts) for k in comp}
    units: list[PlannedUnitV2] = []
    for i, (comp, symbol, gate_tests) in enumerate(drafts):
        depends = tuple(sorted({unit_of[p] for p, c in unit_edges if c in comp and p not in comp},
                               key=unit_ids.index))
        kind = symbols[comp[0]].kind if len(comp) == 1 else "joint"
        files = tuple(sorted({k[0] for k in comp}))
        gate = ", ".join(public_gate_paths or gate_tests)
        if len(comp) == 1:
            subcontract = f"Implement {symbol} ({kind}) in {', '.join(files)}; gate = [{gate}]."
        else:
            subcontract = (f"Implement {', '.join(label[k] for k in comp)} together in "
                           f"{', '.join(files)} (they depend on each other); gate = [{gate}].")
        nxt = drafts[i + 1][1] if i + 1 < len(drafts) else None
        handoff = (f"Handoff: do not start {nxt} until {symbol}'s gate tests and seam check pass."
                   if nxt else f"Handoff: {symbol} is the final unit; its gate must pass for plan completion.")
        units.append(PlannedUnitV2(unit_ids[i], symbol, kind, gate_tests, tuple(depends),
                                   subcontract, handoff, files, tuple(label[k] for k in comp)))
    plan = UnitPlanV2(
        task_id=_task_id(row), source_path=_source_path(row), units=tuple(units),
        handoff_rule=GATED_HANDOFF, plan_digest="", handoff_policy="gated",
        public_gate_paths=public_gate_paths,
        index_revision=str(getattr(index, "revision", "") or ""), index_paths=paths)
    return UnitPlanV2(**{**plan.__dict__, "plan_digest": _digest(plan._basis())})


# ---------------------------------------------------------------- gate
def _failing_names(plan_unit: PlannedUnit, test_results: Mapping[str, object]) -> tuple[str, ...]:
    by_name = {str(k).split("::")[-1]: bool(v) for k, v in test_results.items()}
    return tuple(t for t in plan_unit.gate_tests if not by_name.get(t, False))


def unit_gate(plan: UnitPlanV2, unit_id: str, index, test_results: Mapping[str, object],
              files: Mapping[str, str] | None = None) -> GateVerdict:
    """Pass only if the unit's gate tests pass and its files have no error seam violation.

    ``test_results`` maps test name (or node id) -> passed; a gate test missing from it
    counts as failing.  ``files`` is path -> source for the seam check; without it the
    seam half is skipped (``seam_checked=False``, fail-open like the seam checker).
    """
    unit = next((u for u in plan.units if u.unit_id == unit_id), None)
    if unit is None:
        raise KeyError(f"unknown unit {unit_id!r}")
    failing = _failing_names(unit, test_results)
    seams: tuple[Mapping[str, object], ...] = ()
    if files is not None:
        import chains
        scoped = set(getattr(unit, "files", ()))
        found = chains.seam_violations(chains.seam_manifest(dict(files)), dict(files), index=index)
        seams = tuple(v for v in found
                      if v.get("severity", "error") != "info" and v.get("file") in scoped)
    return GateVerdict(not failing and not seams, failing, seams, files is not None)


def results_from_run(plan: UnitPlan, exit_code: int | None, output: str) -> dict[str, bool] | None:
    """Per-test pass map from a pytest run, or None when the run proves nothing."""
    names = {t for u in plan.units for t in u.gate_tests}
    if exit_code == 0:
        return {t: True for t in names}
    if exit_code == 1:
        failed = set(_FAILED_LINE.findall(output or ""))
        return {t: t not in failed for t in names}
    return None                                    # collection error / usage error / no tests


# ---------------------------------------------------------------- wiring
def _index_class():
    try:
        from kernel_crossfile_index import CrossFileIndex
    except ImportError:
        return None
    return CrossFileIndex


def project_paths(root: Path) -> tuple[str, ...]:
    out: list[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(p.startswith(".") or p in _SKIP_DIRS for p in parts[:-1]) or _is_test_path(rel):
            continue
        out.append(rel)
        if len(out) >= MAX_FILES:
            break
    return tuple(out)


def build_index(root: Path, paths: Sequence[str]):
    cls = _index_class()
    return None if cls is None or not paths else cls.build(Path(root), list(paths))


def derive_with_index(row: Mapping[str, object], root: Path) -> UnitPlanV2 | None:
    """v2 plan, or None (caller keeps v1) when off/unindexable/unplannable."""
    if not enabled():
        return None
    try:
        paths = project_paths(root)
        index = build_index(root, paths)
        return None if index is None else derive_unit_plan_v2(row, index, paths)
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return None


def gate_and_advance(bolt, root: Path, output: str):
    """Run the current unit's gate after a test run; advance bolt.unit_index only on pass."""
    plan = getattr(bolt, "unit_plan", None)
    if not isinstance(plan, UnitPlanV2) or not plan.units:
        return None
    pointer = getattr(bolt, "unit_index", 0)
    results = results_from_run(plan, getattr(bolt, "last_test_exit", None), output)
    if results is None:
        return None
    unit = plan.current(pointer)
    try:
        index = build_index(root, plan.index_paths)
        files = {p: (Path(root) / p).read_text(encoding="utf-8", errors="replace")
                 for p in plan.index_paths if (Path(root) / p).is_file()}
        verdict = unit_gate(plan, unit.unit_id, index, results, files=files if index else None)
    except (OSError, ValueError, SyntaxError) as error:
        # The pointer stays put: an unverifiable gate never advances the unit.
        bolt._rec("unit_plan", "unit_gate_error", _bounded(
            f"{unit.unit_id} {type(error).__name__}: {error}", 240, "…"))
        return None
    if verdict.passed:
        if pointer + 1 < len(plan.units):
            bolt.unit_index = pointer + 1
        bolt._rec("unit_plan", "unit_gate_pass", f"{unit.unit_id} next={getattr(bolt, 'unit_index', pointer)}")
    else:
        bolt._rec("unit_plan", "unit_gate_fail",
                  _bounded(f"{unit.unit_id} tests={list(verdict.failing_tests)} "
                           f"seams={[(v['kind'], v['file'], v['name']) for v in verdict.seam_violations]}",
                           240, "…"))
    return verdict


__all__ = [
    "GATED_HANDOFF", "GateVerdict", "MODE_ENV", "PlannedUnitV2", "SCHEMA_V2", "UnitPlanV2",
    "build_index", "derive_unit_plan_v2", "derive_with_index", "enabled", "gate_and_advance",
    "mode", "project_paths", "results_from_run", "unit_gate",
]
