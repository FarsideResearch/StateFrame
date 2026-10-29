"""Self-contained Mission Control repository intelligence for StateFrame.

The extraction and graph algorithms are vendored from the coding executor;
this module is the release-local adapter.  It owns only bounded, read-only
orientation and navigation queries.  No grader, hidden, gold, private, or
backup path is admitted to the public graph universe.
"""
from __future__ import annotations

import ast
import copy
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from mc_repo_intel_vendor import MC_COMMIT, SOURCE_PATH, VENDOR_SCHEMA
from mc_repo_intel_vendor.call_graph import build_graph
from mc_repo_intel_vendor import repo_map as _repo_map_module
from mc_repo_intel_vendor.repo_map import build_repo_map, get_current_commit_sha
from mc_repo_intel_vendor.ts_dispatch import outline_file as outline
from kernel_repo_catalog import StructuralIndexError, build_structural_index

logger = logging.getLogger(__name__)

SCHEMA = "stateframe-mc-repo-intel/1"
MAX_MAP_CHARS = 2400
MAX_QUERY_CHARS = 240
MAX_ROWS = 24
FORBIDDEN_PARTS = frozenset({
    "grader", "graders", "gold", "private", "hidden", "hidden_tests", "backups",
})


def _tool(name: str, description: str, properties: Mapping[str, object],
          required: Sequence[str]) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


def tool_schemas() -> list[dict[str, object]]:
    """Return the bounded public repo-intel graph tool schemas."""
    return [
        _tool(
            "where_defined",
            "Find the exact repository file and line defining one symbol; read-only and bounded.",
            {"symbol": {"type": "string", "description": "qualified or short symbol name"}},
            ("symbol",),
        ),
        _tool(
            "who_calls",
            "Find direct or bounded-transitive callers of one symbol in the checked-out repository.",
            {
                "symbol": {"type": "string", "description": "qualified or short symbol name"},
                "transitive": {"type": "boolean", "description": "follow caller hops; default false"},
            },
            ("symbol",),
        ),
        _tool(
            "references",
            "Return the definition plus caller and callee references for one symbol.",
            {"symbol": {"type": "string", "description": "qualified or short symbol name"}},
            ("symbol",),
        ),
    ]


def _root(runtime: object) -> Path:
    owner = getattr(runtime, "owner", None)
    candidate = getattr(owner, "root", None)
    if candidate is None:
        candidate = getattr(runtime, "repo_root", None)
    if candidate is None:
        raise ValueError("repo-intel workspace root is unavailable")
    resolved = Path(candidate).resolve()
    if not resolved.is_dir():
        raise ValueError("repo-intel workspace root is not a directory")
    return resolved


def _public_graph_root(root: Path) -> Path:
    """Fail closed if the workspace itself is an explicitly private artifact."""
    # macOS temporary directories commonly live below an OS-level ``private``
    # path; only the checkout's own final component is a meaningful artifact
    # scope here.  Repository-relative forbidden directories are excluded by
    # the vendored walkers themselves.
    if root.name.casefold() in FORBIDDEN_PARTS:
        raise ValueError("repo-intel refuses a forbidden workspace scope")
    return root


def _bounded_symbol(value: object) -> str:
    symbol = str(value or "").strip()
    if not symbol:
        raise ValueError("symbol is required")
    if len(symbol) > MAX_QUERY_CHARS:
        raise ValueError("symbol exceeds the bounded query length")
    if any(part.casefold() in FORBIDDEN_PARTS for part in symbol.replace("\\", "/").split("/")):
        raise ValueError("repo-intel refuses a forbidden symbol scope")
    return symbol


def _bounded_outline_path(root: Path, value: object) -> Path:
    relative = str(value or "").strip().replace("\\", "/")
    if not relative or len(relative) > MAX_QUERY_CHARS or relative.startswith("/"):
        raise ValueError("outline path is invalid")
    if any(part.casefold() in FORBIDDEN_PARTS for part in relative.split("/")):
        raise ValueError("repo-intel refuses a forbidden outline path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("outline path escapes workspace") from error
    if not candidate.is_file():
        raise ValueError("outline path is not a file")
    return candidate


def _ref(value: object) -> dict[str, object]:
    if hasattr(value, "to_dict"):
        result = value.to_dict()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        result = {"value": str(value)}
    return {str(key): copy.deepcopy(item) for key, item in result.items()}


def _bounded_refs(values: Sequence[object]) -> list[dict[str, object]]:
    return [_ref(value) for value in list(values)[:MAX_ROWS]]


def _graph(runtime: object):
    root = _public_graph_root(_root(runtime))
    graph = build_graph(str(root), use_cache=True, use_memo=True)
    return root, graph


def _not_found(action: str, symbol: str, graph: object) -> dict[str, object]:
    suggestions = []
    search_names = getattr(graph, "search_names", None)
    if callable(search_names):
        try:
            suggestions = _bounded_refs(search_names(symbol, limit=5))
        except Exception:  # noqa: BLE001 - suggestions are advisory
            suggestions = []
    return {
        "schema": SCHEMA,
        "status": "NOT_FOUND",
        "action": action,
        "symbol": symbol,
        "results": [],
        "suggestions": suggestions,
        "coverage": getattr(graph, "coverage_summary", lambda: {})(),
    }


def _ambiguous(action: str, symbol: str, candidates: Sequence[object], graph: object) -> dict[str, object]:
    """Return candidates instead of silently selecting the first short-name match."""
    bounded = _bounded_refs(candidates)
    return {
        "schema": SCHEMA,
        "status": "AMBIGUOUS",
        "action": action,
        "symbol": symbol,
        "results": bounded,
        "candidates": bounded,
        "coverage": getattr(graph, "coverage_summary", lambda: {})(),
    }


def _definition_candidates(graph: object, symbol: str) -> list[object]:
    resolver = getattr(graph, "definition_candidates", None)
    if callable(resolver):
        return list(resolver(symbol))
    found = getattr(graph, "find_definition", lambda _symbol: None)(symbol)
    return [found] if found is not None else []


def _bounded_transitive_callers(graph: object, symbol: str) -> tuple[list[object], dict[str, object]]:
    resolver = getattr(graph, "find_callers_bounded", None)
    if callable(resolver):
        return resolver(symbol)
    refs = graph.find_callers(symbol, transitive=True)
    return list(refs), {
        "visited": len(refs), "examined_edges": None,
        "max_depth": None, "truncated": False, "frontier": [],
    }


def operation(runtime: object, name: str, arguments: Mapping[str, object] | None) -> str | None:
    """Execute one admitted repo-intel query, returning a bounded JSON result."""
    if name not in {"where_defined", "who_calls", "references", "outline"}:
        return None
    args = arguments if isinstance(arguments, Mapping) else {}
    try:
        if name == "outline":
            root = _public_graph_root(_root(runtime))
            target = _bounded_outline_path(root, args.get("path"))
            relative = str(target.relative_to(root)).replace("\\", "/")
            result = build_structural_index(
                root,
                focus_paths=(relative,),
                include_call_graph=False,
            ).file_view(relative)
            result["workspace"] = root.name
            result["vendor"] = {"schema": VENDOR_SCHEMA, "mc_commit": MC_COMMIT}
            from kernel_owner_first import coordinates_enabled
            if coordinates_enabled():
                from kernel_request_source_window import owner_coordinates_for_runtime
                coordinates = owner_coordinates_for_runtime(runtime, relative)
                if coordinates is not None:
                    result["owner_coordinates"] = coordinates
            _emit(runtime, "repo_outline", path=str(target.relative_to(root)), degraded=result.get("degraded"))
            return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        symbol = _bounded_symbol(args.get("symbol"))
        root, graph = _graph(runtime)
        if name == "where_defined":
            candidates = _definition_candidates(graph, symbol)
            result = (_not_found(name, symbol, graph) if not candidates else
                      _ambiguous(name, symbol, candidates, graph) if len(candidates) > 1 else {
                "schema": SCHEMA,
                "status": "FOUND",
                "action": name,
                "symbol": symbol,
                "definition": _ref(candidates[0]),
                "coverage": graph.coverage_summary(),
            })
        elif name == "who_calls":
            transitive = bool(args.get("transitive", False))
            candidates = _definition_candidates(graph, symbol)
            if not candidates:
                result = _not_found(name, symbol, graph)
                refs = []
                work = None
            elif len(candidates) > 1:
                result = _ambiguous(name, symbol, candidates, graph)
                refs = []
                work = None
            elif transitive:
                refs, work = _bounded_transitive_callers(graph, candidates[0].symbol)
                result = None
            else:
                refs = graph.find_callers(candidates[0].symbol, transitive=False)
                work = None
                result = None
            if result is not None:
                pass
            else:
                result = {
                    "schema": SCHEMA,
                    "status": "PARTIAL" if isinstance(work, Mapping)
                    and work.get("truncated") else ("FOUND" if refs else "EMPTY"),
                    "action": name,
                    "symbol": symbol,
                    "transitive": transitive,
                    "count": len(refs),
                    "total_relation": "lower_bound" if isinstance(work, Mapping)
                    and work.get("truncated") else "exact",
                    "results": _bounded_refs(refs),
                    "coverage": graph.coverage_summary(),
                }
                if isinstance(work, Mapping):
                    result["work"] = dict(work)
        else:
            candidates = _definition_candidates(graph, symbol)
            if not candidates:
                result = _not_found(name, symbol, graph)
            elif len(candidates) > 1:
                result = _ambiguous(name, symbol, candidates, graph)
            else:
                found = candidates[0]
                callers = graph.find_callers(found.symbol)
                callees = graph.find_callees(found.symbol)
                result = {
                    "schema": SCHEMA,
                    "status": "FOUND",
                    "action": name,
                    "symbol": symbol,
                    "definition": _ref(found),
                    "callers": _bounded_refs(callers),
                    "callees": _bounded_refs(callees),
                    "counts": {"callers": len(callers), "callees": len(callees)},
                    "coverage": graph.coverage_summary(),
                }
        result["workspace"] = root.name
        result["vendor"] = {"schema": VENDOR_SCHEMA, "mc_commit": MC_COMMIT}
        _emit(runtime, "repo_intel_query", action=name, symbol=symbol,
              status=result.get("status"), result_count=_result_count(result))
        return json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (OSError, StructuralIndexError, TypeError, ValueError, RuntimeError) as error:
        _emit(runtime, "repo_intel_query_failed", action=name,
              error_type=type(error).__name__)
        return json.dumps({
            "schema": SCHEMA, "status": "REFUSED", "action": name,
            "reason": type(error).__name__, "results": [],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _result_count(result: Mapping[str, object]) -> int:
    if isinstance(result.get("results"), list):
        return len(result["results"])
    return sum(len(result.get(key, ())) for key in ("callers", "callees")
               if isinstance(result.get(key), list))


def _emit(runtime: object, event: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(event, producer="kernel_repo_intel", dispatch=False, **fields)


def host_where_defined(root: Path, symbol: str) -> dict[str, object]:
    """Host-side ``where_defined`` (P3/M3): the tool's own query, run at intake.

    Returns the same payload shapes the tool returns: ``status`` FOUND with a
    ``definition`` ref, or the shared :func:`_not_found` payload.  A graph that
    cannot be built yields ``status`` UNAVAILABLE (never a fabricated verdict).
    """
    try:
        resolved = _public_graph_root(Path(root).resolve())
        name = _bounded_symbol(symbol)
        graph = build_graph(str(resolved), use_cache=True, use_memo=True)
        found = graph.find_definition(name)
    except (OSError, TypeError, ValueError, RuntimeError, RecursionError) as error:
        return {"schema": SCHEMA, "status": "UNAVAILABLE", "action": "where_defined",
                "symbol": str(symbol), "reason": type(error).__name__}
    if found is None:
        return _not_found("where_defined", name, graph)
    return {"schema": SCHEMA, "status": "FOUND", "action": "where_defined", "symbol": name,
            "definition": _ref(found), "coverage": graph.coverage_summary()}


# ----------------------------------------------------------------------- import graph fact (P3)

_IMPORT_FACT_MAX_FILES = 400


def _module_level_nodes(tree: ast.Module) -> list[ast.stmt]:
    """Statements that run at import time, in source order (if/try/with bodies included)."""
    ordered: list[ast.stmt] = []

    def visit(nodes: Sequence[ast.stmt]) -> None:
        for node in nodes:
            ordered.append(node)
            if isinstance(node, (ast.If, ast.Try, ast.With)):
                visit(getattr(node, "body", ()))
                visit(getattr(node, "orelse", ()))
                visit(getattr(node, "finalbody", ()))
                for handler in getattr(node, "handlers", ()) or ():
                    visit(handler.body)
    visit(tree.body)
    return ordered


def _module_files(dotted: str, roots: Sequence[Path]) -> list[Path]:
    """Files executed by importing ``dotted`` (package ``__init__`` chain, then module)."""
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        return []
    for base in roots:
        if not ((base / parts[0]).is_dir() or (base / f"{parts[0]}.py").is_file()):
            continue
        files: list[Path] = []
        for index in range(1, len(parts) + 1):
            stem = base.joinpath(*parts[:index])
            if (stem / "__init__.py").is_file():
                files.append(stem / "__init__.py")
            elif stem.with_suffix(".py").is_file():
                files.append(stem.with_suffix(".py"))
                break
            elif not stem.is_dir():
                break
        if files:
            return files
    return []


def _imported_modules(node: ast.stmt, file: Path, roots: Sequence[Path]) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level:
        package = file.parent
        for _ in range(node.level - 1):
            package = package.parent
        base = next((root for root in roots if _inside(package, root)), None)
        if base is None:
            return []
        prefix = ".".join(package.relative_to(base).parts)
        module = ".".join(part for part in (prefix, node.module or "") if part)
    else:
        module = node.module or ""
    if not module:
        return []
    # ``from pkg import sub`` may import a submodule file as well as names.
    return [module, *(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _construction_site(tree: ast.Module, names: frozenset[str]) -> tuple[int, str] | None:
    """First import-time call ``<name>(`` or ``<name> = <call>`` among ``names``."""
    for statement in _module_level_nodes(tree):
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                  ast.If, ast.Try, ast.With)):
            continue                    # bodies were flattened; defs do not run at import
        if isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(
                getattr(statement, "value", None), ast.Call):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in names:
                    return statement.lineno, target.id
        stack: list[ast.AST] = [statement]
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Call):
                func = node.func
                short = func.attr if isinstance(func, ast.Attribute) else (
                    func.id if isinstance(func, ast.Name) else "")
                if short in names:
                    return statement.lineno, short
            stack.extend(ast.iter_child_nodes(node))
    return None


def import_construction_fact(root: Path, path: str, names: Sequence[str], *,
                             roots: Sequence[str] | None = None) -> dict[str, object] | None:
    """First import statement of ``path`` whose transitive import closure constructs one of ``names``.

    Python only (the import system is the graph).  Imports resolve against the
    smoke's ``source_roots`` -- the vendored call graph's import table resolves
    from the checkout root only, so ``app.*`` under ``backend/`` would be
    invisible to it.  Bounded (``_IMPORT_FACT_MAX_FILES`` parsed files); a
    construction inside the target file itself is not an import edge.
    """
    wanted = frozenset(str(name) for name in names if str(name).isidentifier())
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not wanted or target.suffix != ".py" or not target.is_file() or not _inside(target, root):
        return None
    if roots is None:
        from kernel_ship_tree_smoke import source_roots
        roots = source_roots(root, [path])
    search_roots = [Path(item).resolve() for item in roots]
    try:
        target_tree = ast.parse(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return None
    lines = target.read_text(encoding="utf-8").splitlines()
    parsed: dict[Path, ast.Module | None] = {}

    def tree_of(file: Path) -> ast.Module | None:
        if file not in parsed:
            if len(parsed) >= _IMPORT_FACT_MAX_FILES:
                return None
            try:
                parsed[file] = ast.parse(file.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, SyntaxError, ValueError):
                parsed[file] = None
        return parsed[file]

    visited: set[Path] = {target}
    for statement in _module_level_nodes(target_tree):
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        queue: list[tuple[Path, tuple[str, ...]]] = []
        for module in _imported_modules(statement, target, search_roots):
            for file in _module_files(module, search_roots):
                queue.append((file, (module,)))
        while queue:
            file, chain = queue.pop(0)
            if file in visited:
                continue
            visited.add(file)
            tree = tree_of(file)
            if tree is None:
                continue
            site = _construction_site(tree, wanted)
            if site is not None:
                site_line, name = site
                site_text = file.read_text(encoding="utf-8").splitlines()[site_line - 1].strip()
                return {
                    "path": target.relative_to(root).as_posix(),
                    "line": statement.lineno,
                    "statement": lines[statement.lineno - 1].strip(),
                    "name": name,
                    "constructor": {"path": file.relative_to(root).as_posix() if _inside(file, root)
                                    else file.name, "line": site_line, "text": site_text},
                    "chain": list(chain),
                }
            for inner in _module_level_nodes(tree):
                for module in _imported_modules(inner, file, search_roots):
                    for nested in _module_files(module, search_roots):
                        if nested not in visited:
                            queue.append((nested, (*chain, module)))
    return None


def _map_blocks(repo_map: str) -> list[tuple[str, str]]:
    lines = str(repo_map or "").splitlines()
    header = lines[0] if lines else ""
    blocks: list[tuple[str, str]] = []
    current_path = ""
    current: list[str] = []
    for line in lines[1:]:
        if line and not line.startswith("  ") and line.endswith(":"):
            if current_path:
                blocks.append((current_path, "\n".join(current)))
            current_path = line[:-1]
            current = [line]
        elif current_path:
            current.append(line)
    if current_path:
        blocks.append((current_path, "\n".join(current)))
    if header:
        return [("__header__", header), *blocks]
    return blocks


def bound_repo_map(repo_map: str, *, target_path: str = "", symbol: str = "",
                   max_chars: int = MAX_MAP_CHARS) -> str:
    """Rank executor map entries by task locality, then cap bytes deterministically."""
    blocks = _map_blocks(repo_map)
    if not blocks:
        return "[MC repo map unavailable; use where_defined or list_files.]"
    target = str(target_path or "").replace("\\", "/").strip().casefold()
    wanted = str(symbol or "").strip().casefold()
    target_parent = str(Path(target).parent).casefold() if target else ""
    cap = max(64, int(max_chars))

    def score(item: tuple[str, str]) -> tuple[int, str]:
        path, text = item
        folded_path = path.casefold()
        folded_text = text.casefold()
        value = 0
        if path == "__header__":
            return (-10000, path)
        elif target and folded_path == target:
            value += 1000
        elif target_parent and folded_path.startswith(target_parent + "/"):
            value += 180
        if wanted and wanted in folded_text:
            value += 120
        if target and Path(target).name and Path(target).name in folded_path:
            value += 40
        return (-value, path)

    ranked = sorted(blocks, key=score)
    rendered: list[str] = []
    used = 0
    omitted = 0
    for path, block in ranked:
        addition = ("\n" if rendered else "") + block
        if used + len(addition) > cap:
            omitted += 1
            continue
        rendered.append(addition)
        used += len(addition)
    if omitted:
        marker = f"[repo map trimmed: {omitted} lower-ranked entries omitted]"
        while rendered and len("\n".join(rendered) + "\n" + marker) > cap:
            rendered.pop()
        rendered.append(marker)
    return "\n".join(rendered)[:cap]


def _ensure_focus_entry(repo_root: Path, repo_map: str, target_path: str) -> str:
    """Add the requested file when the upstream global file cap omitted it."""
    normalized = str(target_path or "").strip().replace("\\", "/").lstrip("./")
    if not normalized or normalized in repo_map:
        return repo_map
    candidate = repo_root / normalized
    try:
        candidate.relative_to(repo_root)
    except ValueError:
        return repo_map
    if not candidate.is_file():
        return repo_map
    try:
        entry = _repo_map_module._file_entry(candidate, repo_root)
    except Exception:  # noqa: BLE001 - focused orientation remains fail-open
        entry = None
    if not entry:
        return repo_map
    return f"{repo_map}\n{entry}" if repo_map else entry


def intake_orientation(root: Path, *, target_path: str = "", symbol: str = "",
                       selectors: Sequence[object] = ()) -> str:
    """Produce map text for the single existing NAVBASICS orientation owner."""
    commit = get_current_commit_sha(str(root))
    repo_map = build_repo_map(str(root), commit_sha=commit)
    repo_map = _ensure_focus_entry(root, repo_map, target_path)
    bounded = bound_repo_map(repo_map, target_path=target_path, symbol=symbol)
    from kernel_nav_basics import render_intake_orientation
    orientation = render_intake_orientation(root, selectors, repo_map=bounded)
    _emit_for_root(root, "repo_intel_orientation", commit=commit,
                   map_chars=len(bounded), target_path=str(target_path), symbol=str(symbol))
    return orientation


def _emit_for_root(root: Path, event: str, **fields: object) -> None:
    # Orientation has no runtime journal at this seam; callers get a stable
    # text block and query events remain journaled by operation().
    logger.debug("%s root=%s fields=%s", event, root.name, fields)


__all__ = [
    "FORBIDDEN_PARTS", "MC_COMMIT", "MAX_MAP_CHARS", "SCHEMA", "SOURCE_PATH",
    "bound_repo_map", "host_where_defined", "import_construction_fact", "intake_orientation",
    "operation", "outline", "tool_schemas",
]
