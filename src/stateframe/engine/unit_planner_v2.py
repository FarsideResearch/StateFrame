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
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Final

from unit_coalesce_contract import GateObservation, GateRun
from unit_coalesce_contract import enabled as _coalesce_enabled
from unit_planner import (
    PlannedUnit,
    UnitPlan,
    _bounded,
    _digest,
    _local_functions,
    _source_path,
    _symbol_kinds,
    _task_id,
    _test_functions,
    _text,
)

MODE_ENV: Final = "STATEFRAME_UNIT_PLAN_V2"
SCHEMA_V2: Final = "unit-plan/2"
MAX_FILES: Final = 400
MAX_UNITS: Final = 150
GATED_HANDOFF: Final = ("Unit N+1 starts only after unit N's gate passes: its tests and the "
                        "cross-file seam check. Mutually dependent symbols form one joint unit.")
_SKIP_DIRS: Final = frozenset({"__pycache__", "node_modules", "venv", "site-packages", "build", "dist"})
_FAILED_LINE: Final = re.compile(r"^(?:FAILED|ERROR)\s+\S*?::(?:\w+::)?(\w+)", re.MULTILINE)

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
    # unit_coalesce (STATEFRAME_UNIT_COALESCE) fields; defaults keep v2 digests byte-identical
    coalesced_from: tuple[str, ...] = ()
    gate_attempts: int = 0            # 0 = unset (DEFAULT_GATE_ATTEMPTS applies)

    def to_dict(self) -> dict[str, object]:
        out = super().to_dict()
        out["files"] = list(self.files)
        out["members"] = list(self.members)
        if self.coalesced_from:
            out["coalesced_from"] = list(self.coalesced_from)
        if self.gate_attempts:
            out["gate_attempts"] = self.gate_attempts
        return out


@dataclass(frozen=True)
class UnitPlanV2(UnitPlan):
    index_revision: str = ""
    index_paths: tuple[str, ...] = ()
    lineage: tuple[str, ...] = ()     # previous plan digests, oldest first (unit_coalesce)
    coalesce_events: int = 0

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
        if self.lineage:
            basis["lineage"] = list(self.lineage)
        if self.coalesce_events:
            basis["coalesce_events"] = self.coalesce_events
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


def gate_and_advance(bolt, root: Path, output: str, *, evidence: str | None = None):
    """Run the current unit's gate after a test run; advance bolt.unit_index only on pass."""
    plan = getattr(bolt, "unit_plan", None)
    if not isinstance(plan, UnitPlanV2) or not plan.units:
        return None
    evidence = output if evidence is None else evidence
    _revoke_coalesce_passes(bolt, plan, root, evidence)
    pointer = getattr(bolt, "unit_index", 0)
    if pointer >= len(plan.units) and hasattr(bolt, "unit_gate_passed"):
        return None
    results = results_from_run(plan, getattr(bolt, "last_test_exit", None), output)
    unit = plan.current(pointer)
    results, coalesce_on = _coalesce_results(bolt, unit, results, evidence, root=root)
    partial = results is None and bool(getattr(bolt, "unit_gate_partial_results", None)) and coalesce_on
    if results is None:
        if not partial:
            return None
        results = bolt.unit_gate_partial_results
    try:
        if partial and not _partial_repair_changed(bolt, plan, unit, root):
            return None
        index = build_index(root, plan.index_paths)
        files = {p: (Path(root) / p).read_text(encoding="utf-8", errors="replace")
                 for p in plan.index_paths if (Path(root) / p).is_file()}
        verdict = unit_gate(plan, unit.unit_id, index, results, files=files if index else None)
        if verdict.passed and coalesce_on:
            from unit_gate_evidence import pass_snapshot, shared_source_identity
            pass_sources = pass_snapshot(unit, root, evidence, shared_source_identity(
                plan, root, getattr(bolt, "measurement", None)))
    except (OSError, ValueError, SyntaxError) as error:
        # The pointer stays put: an unverifiable gate never advances the unit.
        bolt._rec("unit_plan", "unit_gate_error", _bounded(
            f"{unit.unit_id} {type(error).__name__}: {error}", 240, "…"))
        return None
    if partial:
        # A confirmed red can spend an edited repair even when -x left later
        # cases unrun. It cannot establish a pass or masquerade as a full gate.
        # unit_gate's ordinary missing-test rule must not attribute an unrun
        # case to a producer. Unknown coverage still prevents advancement.
        confirmed = tuple(name for name in unit.gate_tests if results.get(name) is False)
        verdict = GateVerdict(False, confirmed, verdict.seam_violations, verdict.seam_checked)
        if confirmed or verdict.seam_violations:
            _coalesce_gate_hook(bolt, plan, unit, verdict, files, root, pointer, coalesce_on)
        return None
    if verdict.passed:
        next_pointer = pointer + 1
        if coalesce_on:
            # Completion belongs to the gate, even when optional advice fails.
            bolt.unit_gate_passed = set(getattr(bolt, "unit_gate_passed", ())) | {unit.unit_id}
            snapshots = dict(getattr(bolt, "unit_gate_pass_sources", {}))
            snapshots[unit.unit_id] = pass_sources
            bolt.unit_gate_pass_sources = snapshots
            rechecks = set(getattr(bolt, "unit_gate_rechecks", ())) - {unit.unit_id}
            bolt.unit_gate_rechecks = rechecks
            next_pointer = min([next_pointer] + [i for i, u in enumerate(plan.units) if u.unit_id in rechecks])
            while (next_pointer < len(plan.units)
                   and plan.units[next_pointer].unit_id in bolt.unit_gate_passed
                   and plan.units[next_pointer].unit_id not in rechecks):
                next_pointer += 1
        if next_pointer < len(plan.units) or coalesce_on:
            bolt.unit_index = next_pointer
        _coalesce_gate_hook(bolt, plan, unit, verdict, files, root, pointer, coalesce_on)
        bolt._rec("unit_plan", "unit_gate_pass", f"{unit.unit_id} next={getattr(bolt, 'unit_index', pointer)}")
    else:
        bolt._rec("unit_plan", "unit_gate_fail",
                  _bounded(f"{unit.unit_id} tests={list(verdict.failing_tests)} "
                           f"seams={[(v['kind'], v['file'], v['name']) for v in verdict.seam_violations]}",
                           240, "…"))
        _coalesce_gate_hook(bolt, plan, unit, verdict, files, root, pointer, coalesce_on)
    return verdict


def _coalesce_error(bolt, unit, error: Exception) -> None:
    import logging
    reason = _bounded(f"{getattr(unit, 'unit_id', 'host')} {type(error).__name__}: {error}", 240, "…")
    logging.getLogger(__name__).warning("unit_coalesce_error: %s", reason)
    if bolt is not None:
        try:
            bolt._rec("unit_plan", "unit_coalesce_error", reason)
        except Exception as record_error:  # noqa: BLE001 - journal failure still has a log
            logging.getLogger(__name__).warning("unit_coalesce_error journal: %s", record_error)


def _partial_repair_changed(bolt, plan, unit, root) -> bool:
    """Avoid indexing a pre-edit partial run; only an actual repair spends budget."""
    try:
        from unit_coalesce import observe_tree, tree_digest
        files = {p: (Path(root) / p).read_text(encoding="utf-8", errors="replace")
                 for p in plan.index_paths if (Path(root) / p).is_file()}
        observe_tree(bolt, plan, files)
        return bolt.unit_gate_snapshots.get(unit.unit_id) != tree_digest(plan, files)
    except Exception as error:  # noqa: BLE001 - partial evidence must never abort the task
        _coalesce_error(bolt, unit, error)
        return False


def coalesce_pytest_command(command: Sequence[str]) -> list[str]:
    """Retain fail-fast and request named evidence only under both switches."""
    args = list(command)
    try:
        if not enabled() or not _coalesce_enabled():
            return args
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        _coalesce_error(None, None, error)
        return args
    if args[1:5] != ["-m", "pytest", "-x", "-q"]:
        return args
    # Replace only the known host prefix, preserving selector/config argument values verbatim.
    return args[:4] + ["--verbosity=1"] + args[5:]


def coalesce_observation(text: str, stdout: str, exit_code: int | None,
                         *, complete: bool = True, collected: tuple[str, ...] | None = None,
                         source_digest: str = "") -> str:
    """Keep raw stdout before receipt compaction, without changing the wire text."""
    try:
        if enabled() and _coalesce_enabled():
            return GateObservation(text, (GateRun(stdout, exit_code, complete, collected, source_digest),))
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        _coalesce_error(None, None, error)
    return text


def capture_incomplete_gate_observation(run: Callable[..., str]) -> Callable[..., str]:
    """Cover every runner early return; OFF preserves the exact original object."""
    @wraps(run)
    def captured(*args, **kwargs) -> str:
        result = run(*args, **kwargs)
        if isinstance(result, GateObservation):
            return result
        return coalesce_observation(result, "", None, complete=False)
    return captured


def coalesce_combined_observation(text: str, observations: Sequence[str]) -> str:
    """Retain each selector's evidence; refused/timeout children cannot borrow a pass."""
    try:
        if enabled() and _coalesce_enabled():
            runs = tuple(run for observation in observations
                         for run in getattr(observation, "unit_gate_runs", (GateRun("", None, False),)))
            return GateObservation(text, runs)
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        _coalesce_error(None, None, error)
    return text


def coalesce_collection(root: Path, command, targets, runtime):
    """Prepare a sandboxed collection only in the opt-in evidence lane."""
    try:
        if not enabled() or not _coalesce_enabled():
            return None
        from unit_gate_evidence import collection_command
        return collection_command(command, targets, root, getattr(getattr(runtime, "owner", None), "unit_plan", None),
                                  runtime=runtime)
    except Exception as error:  # noqa: BLE001 - missing collection cannot certify a pass
        _coalesce_error(getattr(runtime, "owner", None), None, error)
        return None


def _coalesce_results(bolt, unit, results, output, *, root=None):
    """Only PASSED satisfies a gate. Nonpass outcomes are observed failures.

    Full collected-node coverage is required to pass. A confirmed partial red is
    queued separately so only an edited repair can spend the failure allowance.
    """
    try:
        if not _coalesce_enabled():
            return results, False
        bolt.unit_gate_partial_results = None
        from unit_gate_evidence import run_evidence, statuses, test_name
        runs = getattr(output, "unit_gate_runs", None)
        complete = bool(unit.gate_tests)  # an unreceipted seam-only run proves nothing
        nodes = statuses(output or "")
        collected: set[str] = set()
        identity = ""
        if runs is not None:
            nodes, collected, complete, identity = run_evidence(runs)
            output = "\n".join(run.stdout for run in runs)
        # A truncated/timeout/collection-incomplete red still contradicts an
        # earlier positive for that exact case, including in the current unit
        # which has never passed. Such a capture cannot contribute new passes.
        previous_key, previous = getattr(bolt, "unit_gate_node_evidence", (None, {}))
        negative = {node: False for node, passed in nodes.items() if not passed}
        if negative:
            bolt.unit_gate_node_evidence = (previous_key, {**previous, **negative})
        if (runs is not None and complete and root is not None and collected
                and all(run.collected is not None for run in runs)):
            plan = bolt.unit_plan
            key = _digest([identity, [(p, (Path(root) / p).read_text(encoding="utf-8", errors="replace")
                                       if (Path(root) / p).is_file() else None) for p in plan.index_paths],
                           sorted(collected)])
            previous_key, previous = getattr(bolt, "unit_gate_node_evidence", (None, {}))
            if previous_key == key:
                # Each later execution supersedes that exact case's old result.
                nodes = {**previous, **nodes}
            bolt.unit_gate_node_evidence = (key, dict(nodes))
        by_name = {name: all(passed for node, passed in nodes.items() if test_name(node) == name)
                   for name in {test_name(node) for node in nodes}}
        missing = sorted(name for name in unit.gate_tests
                         if name not in by_name or (collected and (
                             not any(test_name(node) == name for node in collected)
                             or any(test_name(node) == name and node not in nodes for node in collected))))
        if runs is not None and any(run.collected is None for run in runs):
            missing = sorted(set(missing) | set(unit.gate_tests))
            if not unit.gate_tests:
                complete = False
        if not unit.gate_tests and not nodes:
            complete = False
        # Named cases without a collection inventory cannot prove all parameter
        # cases ran. Keep the plain-text compatibility lane conservative too.
        if not collected and any("[" in node for node in nodes):
            missing = sorted(set(missing) | set(unit.gate_tests))
        if missing or not complete or results is None:
            reason = "incomplete_capture" if not complete else "deselected" if "deselected" in output else "partial_run"
            bolt._rec("unit_plan", "unit_gate_unobserved", _bounded(
                f"{unit.unit_id} reason={reason} tests={missing}", 240, "…"))
            if complete and results is not None and any(by_name.get(name) is False for name in unit.gate_tests):
                bolt.unit_gate_partial_results = by_name
            return None, True
        return by_name, True
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        _coalesce_error(bolt, unit, error)
        return None, False


def _revoke_coalesce_passes(bolt, plan, root, evidence) -> None:
    """Fresh red evidence and changed providers revoke cached dependent passes."""
    active = False
    try:
        if not hasattr(bolt, "unit_gate_passed") or not _coalesce_enabled():
            return
        active = True
        from unit_gate_evidence import (
            pass_snapshot,
            run_evidence,
            shared_source_identity,
            statuses,
            test_name,
        )
        runs = getattr(evidence, "unit_gate_runs", None)
        nodes = run_evidence(runs)[0] if runs is not None else statuses(evidence or "")
        failed = {test_name(node) for node, passed in nodes.items() if not passed}
        snapshots = getattr(bolt, "unit_gate_pass_sources", {})
        invalid = ({u.unit_id for u in plan.units if set(u.gate_tests) & failed}
                   | set(getattr(bolt, "unit_gate_rechecks", ())))
        test_sources = shared_source_identity(plan, root, getattr(bolt, "measurement", None))
        for unit in plan.units:
            if unit.unit_id not in snapshots:
                if unit.unit_id in bolt.unit_gate_passed:
                    invalid.add(unit.unit_id)
                continue
            current = pass_snapshot(unit, root, evidence, test_sources)
            previous = snapshots[unit.unit_id]
            # Old checkpoints only recorded product bytes. They must establish
            # a fresh pass before borrowing test/collection authority.
            if (not isinstance(previous, tuple) or len(previous) != 3
                    or current[:2] != previous[:2]
                    or current[2] is not None and current[2] != previous[2]):
                invalid.add(unit.unit_id)
        invalid = invalidate_dependents(plan, invalid)
        revoked = set(bolt.unit_gate_passed) & invalid
        if revoked:
            bolt.unit_gate_passed = set(bolt.unit_gate_passed) - invalid
            bolt.unit_gate_rechecks = set(getattr(bolt, "unit_gate_rechecks", ())) | revoked
            # A recheck needs fresh evidence, including when a checkpoint has
            # kept the same tree/collection key. Expire the revoked names once;
            # later selectors may still accumulate while that recheck is pending.
            key, cached = getattr(bolt, "unit_gate_node_evidence", (None, {}))
            expired = {name for unit in plan.units if unit.unit_id in invalid for name in unit.gate_tests}
            bolt.unit_gate_node_evidence = (key, {node: value for node, value in cached.items()
                                                  if test_name(node) not in expired})
            bolt._rec("unit_plan", "unit_gate_pass_revoked", _bounded(str(sorted(revoked)), 240, "…"))
        outstanding = set(getattr(bolt, "unit_gate_rechecks", ())) & {u.unit_id for u in plan.units}
        if outstanding and getattr(bolt, "unit_index", 0) >= len(plan.units):
            bolt.unit_index = min(i for i, u in enumerate(plan.units) if u.unit_id in outstanding)
    except Exception as error:  # noqa: BLE001 - advisory invalidation failures remain visible
        _coalesce_error(bolt, None, error)
        if active:
            # Failure to read a changed test/harness cannot retain completion
            # authority from an earlier tree. Keep every affected gate pending.
            passed = set(getattr(bolt, "unit_gate_passed", ()))
            bolt.unit_gate_passed = set()
            bolt.unit_gate_rechecks = set(getattr(bolt, "unit_gate_rechecks", ())) | passed
            bolt.unit_gate_node_evidence = (None, {})
            positions = [i for i, unit in enumerate(plan.units) if unit.unit_id in bolt.unit_gate_rechecks]
            if positions:
                bolt.unit_index = min(getattr(bolt, "unit_index", 0), min(positions))


def invalidate_dependents(plan, unit_ids: set[str]) -> set[str]:
    invalid = set(unit_ids)
    for unit in plan.units:  # plan is topologically ordered
        if invalid.intersection(unit.depends_on):
            invalid.add(unit.unit_id)
    return invalid


def _coalesce_gate_hook(bolt, plan, unit, verdict, files, root, pointer, active) -> None:
    """Isolate coalescer failures without changing existing handler ordinals."""
    try:
        if not active:
            return
        from unit_coalesce import (
            advance_passed,
            observe_tree,
            on_gate_failure,
            tree_digest,
        )
        if verdict.passed:
            observe_tree(bolt, plan, files)
            advance_passed(bolt, plan, pointer)
            if bolt.unit_index < len(plan.units):
                bolt.unit_gate_snapshots[plan.units[bolt.unit_index].unit_id] = tree_digest(plan, files)
        else:
            on_gate_failure(bolt, plan, unit, verdict, files, root)
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        if not verdict.passed and bolt.unit_plan is plan:
            bolt.unit_index = pointer
        _coalesce_error(bolt, unit, error)


def seed_coalesce(bolt, root: Path) -> None:
    """Capture the initial tree at plan installation, before the model can repair it."""
    plan = getattr(bolt, "unit_plan", None)
    if not isinstance(plan, UnitPlanV2) or not plan.units:
        return
    try:
        if not _coalesce_enabled():
            return
        from unit_coalesce import observe_tree
        files = {p: (Path(root) / p).read_text(encoding="utf-8", errors="replace")
                 for p in plan.index_paths if (Path(root) / p).is_file()}
        observe_tree(bolt, plan, files)
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        _coalesce_error(bolt, plan.units[0], error)


def refresh_coalesced_plan(bolt, messages) -> None:
    """Append approved merge/escalation advice without modifying the cached prefix."""
    pending = getattr(bolt, "unit_plan_refresh", None)
    if not pending:
        return
    try:
        from kernel_stateframe_framefit5 import latest_complete_tool_exchange
        if latest_complete_tool_exchange(messages) is None:
            # A pending native batch must remain contiguous. Its test result
            # consumes the notice at the actual delivery boundary instead.
            return
        messages.append({"role": "user", "content": pending})
        bolt.unit_plan_refresh = None
        bolt.unit_plan_refresh_from = None
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        _coalesce_error(bolt, None, error)


def coalesce_notice_observation(bolt, observation: str) -> str:
    """Deliver deferred advice inside its native result, never between call/results."""
    pending = getattr(bolt, "unit_plan_refresh", None)
    if not pending:
        return observation
    result = observation + "\n" + pending
    bolt.unit_plan_refresh = None
    bolt.unit_plan_refresh_from = None
    return result


def unit_plan_state(bolt) -> dict[str, object] | None:
    """Regenerate authoritative plan facts on every request, including after resume."""
    try:
        plan = getattr(bolt, "unit_plan", None)
        if not isinstance(plan, UnitPlanV2) or not enabled() or not _coalesce_enabled():
            return None
        state = {**plan.to_dict(), "lineage": list(plan.lineage),
                 "unit_index": getattr(bolt, "unit_index", 0),
                 "passed_units": sorted(getattr(bolt, "unit_gate_passed", ())),
                 "escalated_unit": getattr(bolt, "unit_escalate", None),
                 "authority": "current plan supersedes the initial task PLAN"}
        # A shared integration gate repeats its test list in every unit. Keep
        # ordinary small-plan projections intact, but never spend the product
        # frame's byte allowance on that quadratic expansion. Full authority
        # stays in bolt.unit_plan and survives checkpoints unchanged.
        if len(json.dumps(state).encode("utf-8")) <= 12288:
            return state
        return _bounded_plan_state(bolt, plan)
    except Exception as error:  # noqa: BLE001 - optional projection errors must be visible
        _coalesce_error(bolt, None, error)
        return None


def _bounded_plan_state(bolt, plan: UnitPlanV2) -> dict[str, object]:
    """Bound current-unit facts independently of total units/shared gate size."""
    def label(value: str) -> str:
        # ASCII JSON may expand one character to twelve bytes. Bound encoded
        # labels too; display prefixes are never complete identities.
        shown = _bounded(value, 160, "…")
        while len(json.dumps(shown).encode("utf-8")) > 256:
            shown = _bounded(shown, len(shown) // 2, "…")
        return shown

    def sample(values: Sequence[str]) -> dict[str, object]:
        return {"sample": [label(value) for value in values[:4]], "count": len(values),
                "digest": _digest(list(values)), "omitted": max(0, len(values) - 4)}

    pointer = getattr(bolt, "unit_index", 0)
    passed = sorted(getattr(bolt, "unit_gate_passed", ()))
    rechecks = sorted(getattr(bolt, "unit_gate_rechecks", ()))
    escalated = getattr(bolt, "unit_escalate", None)
    state: dict[str, object] = {
        "schema": "unit-plan-progress/1", "plan_schema": SCHEMA_V2, "plan_digest": plan.plan_digest,
        "task_id": label(plan.task_id), "unit_index": pointer,
        "unit_count": len(plan.units), "coalesce_events": plan.coalesce_events,
        "lineage": list(plan.lineage[-4:]), "lineage_count": len(plan.lineage),
        "passed_units": sample(passed), "recheck_units": sample(rechecks),
        "escalated_unit": label(escalated) if escalated else None,
        "escalated_unit_digest": _digest(escalated) if escalated else None,
        "authority": "current plan supersedes the initial task PLAN",
        "projection": "current unit and progress; full plan retained by host",
        "units": [],
    }
    if 0 <= pointer < len(plan.units):
        unit = plan.units[pointer]
        state["units"] = [{
            "unit_id": label(unit.unit_id), "unit_digest": _digest(unit.to_dict()),
            "symbol": label(unit.symbol), "kind": unit.kind,
            "files": sample(unit.files), "members": sample(unit.members),
            "gate_tests": sample(unit.gate_tests), "depends_on": sample(unit.depends_on),
            "gate_attempts": unit.gate_attempts,
        }]
    return state


def consume_coalesce_escalation(bolt) -> bool:
    """Stop further model/tool retries after the current native result is closed."""
    if not getattr(bolt, "unit_escalate", None):
        return False
    try:
        if not enabled() or not _coalesce_enabled():
            return False
        runtime = getattr(bolt, "measurement", None)
        if runtime is None:
            raise ValueError("unit escalation has no runtime")
        runtime.system_outcome = getattr(runtime, "system_outcome", None) or "unit_coalesce_escalated"
        if not getattr(bolt, "unit_escalation_consumed", False):
            bolt._rec("unit_plan", "unit_coalesce_stopped", str(bolt.unit_escalate))
            bolt.unit_escalation_consumed = True
        return True
    except Exception as error:  # noqa: BLE001 - an unavailable stop consumer must be reported
        _coalesce_error(bolt, None, error)
        return False


def install_unit_plan(bolt, proposed, root: Path):
    """Keep a restored coalesced plan instead of reinstalling its stale ancestor."""
    try:
        restored = getattr(bolt, "unit_plan", None)
        if (isinstance(restored, UnitPlanV2) and restored.lineage
                and restored.task_id == getattr(proposed, "task_id", None)
                and _coalesce_enabled()):
            # Re-deriving against edited files may change the digest or fall back
            # to v1. Neither invalidates this task's restored coalesced plan.
            return restored
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        _coalesce_error(bolt, None, error)
    bolt.unit_plan = proposed
    seed_coalesce(bolt, root)
    return proposed


def weld_initial_plan(bolt, messages) -> None:
    """Guard opt-in installation advice; retain the legacy OFF error contract."""
    try:
        active = isinstance(bolt.unit_plan, UnitPlanV2) and _coalesce_enabled()
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        _coalesce_error(bolt, None, error)
        return
    if not active:
        _weld_initial_plan(bolt, messages, coalescing=False)
        return
    try:
        _weld_initial_plan(bolt, messages, coalescing=True)
    except Exception as error:  # noqa: BLE001 - advisory coalescing must log and fail open
        _coalesce_error(bolt, None, error)


def _weld_initial_plan(bolt, messages, *, coalescing: bool) -> None:
    """A resumed transcript already owns its plan history, including tail notices."""
    from unit_planner import render_plan_block
    plan = bolt.unit_plan
    if coalescing and any("[PLAN " in str(message.get("content", ""))
                          and any(f"digest={digest}" in str(message.get("content", ""))
                                  for digest in (*plan.lineage, plan.plan_digest))
                          for message in messages):
        bolt._rec("unit_plan", "unit_plan_weld_skipped", f"already delivered digest={plan.plan_digest}")
        return
    if len(getattr(plan, "units", ())) <= 1:
        return
    block = bolt.organ_out("unit_plan", render_plan_block(plan))
    if block:
        if len(messages) > 1 and messages[1].get("role") == "user":
            messages[1]["content"] = str(messages[1]["content"]) + "\n\n" + block
        else:
            messages.append({"role": "user", "content": block})
        bolt._rec("unit_plan", "plan_welded", f"units={len(plan.units)}")


def render_unit_progress(bolt) -> str:
    """Keep the legacy line off-switch; show completion after opt-in passed-unit skipping."""
    from unit_planner import render_current_unit_line
    plan = bolt.unit_plan
    pointer = getattr(bolt, "unit_index", 0)
    if hasattr(bolt, "unit_gate_passed") and pointer >= len(plan.units):
        return f"[planner] all unit gates passed; digest={plan.plan_digest}"
    return render_current_unit_line(plan, pointer)


__all__ = [
    "GATED_HANDOFF",
    "MODE_ENV",
    "SCHEMA_V2",
    "GateVerdict",
    "PlannedUnitV2",
    "UnitPlanV2",
    "build_index",
    "capture_incomplete_gate_observation",
    "coalesce_collection",
    "coalesce_combined_observation",
    "coalesce_notice_observation",
    "coalesce_observation",
    "coalesce_pytest_command",
    "consume_coalesce_escalation",
    "derive_unit_plan_v2",
    "derive_with_index",
    "enabled",
    "gate_and_advance",
    "install_unit_plan",
    "mode",
    "project_paths",
    "refresh_coalesced_plan",
    "render_unit_progress",
    "results_from_run",
    "seed_coalesce",
    "unit_gate",
    "unit_plan_state",
    "weld_initial_plan",
]
