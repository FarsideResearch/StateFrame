"""Deterministic, opt-in grounding for repository paths and Python symbols.

This module deliberately does not interpret prose.  It only extracts path-like
and code-like tokens, resolves paths against the live workspace, and resolves
symbols from a small AST index rebuilt at each inspection boundary.
"""
from __future__ import annotations

import ast
import dataclasses
import difflib
import json
import os
import re
import sys
import threading
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path

import kernel_parse_cache as _parse_cache
import kernel_turn0_persistence as _turn0
import kernel_workspace_truth as _workspace_truth


MODE_ENV = "STATEFRAME_CLAIM_GROUNDING"
FAST_MODE_ENV = "STATEFRAME_CLAIM_GROUNDING_FAST"
SCHEMA = "stateframe-claim-grounding/1"
_SKIP_DIRS = frozenset({
    ".git", ".venv", "__pycache__", "node_modules", "site-packages", "backup", "backups",
})
_MUTATING_TOOLS = frozenset({"edit", "str_replace", "replace_module", "write", "replace"})
_PATH_FIELDS = frozenset({
    "path", "target_path", "file", "filename", "source_path", "test_path",
    "working_path", "reference", "target",
})
_SYMBOL_FIELDS = frozenset({"symbol", "target_ref", "function", "qualified_name", "method"})
_PATH_EXTENSIONS = frozenset({
    ".c", ".cc", ".cpp", ".css", ".go", ".h", ".html", ".java", ".js", ".jsx",
    ".json", ".md", ".py", ".rb", ".rs", ".sh", ".sql", ".toml", ".ts", ".tsx",
    ".txt", ".vue", ".xml", ".yaml", ".yml",
})

# The expressions are intentionally narrow.  They are not a prose classifier:
# a path needs a path shape/extension, while a symbol needs a qualified form,
# an uppercase name, or an underscore-bearing identifier.
# Line locators the host itself teaches (diet notices say "path:start-end"): accept
# :N, :N:M (line:column), :N-M (range) and #LN / #LN-LM. Live benchmark 1036 seq 2960:
# the model's correct first edit cited target_ref="backend/app/utils/ai_utils.py:14-16",
# the :N-M form did not parse, and the edit was rejected UNGROUNDED_CLAIM.
_LOCATOR = r"(?::[0-9]+(?:[:-][0-9]+)?|#L[0-9]+(?:-L?[0-9]+)?)"
_PATH_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:\.{0,2}/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+"
    rf"{_LOCATOR}?(?:::[A-Za-z_][A-Za-z0-9_.]*)?"
    r"|(?<![A-Za-z0-9_])[A-Za-z0-9_.-]+\.(?:c|cc|cpp|css|go|h|html|java|js|jsx|json|md|py|rb|rs|sh|sql|toml|ts|tsx|txt|vue|xml|yaml|yml)"
    rf"{_LOCATOR}?(?:::[A-Za-z_][A-Za-z0-9_.]*)?"
)
_CODE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|::)[A-Za-z_][A-Za-z0-9_]*)+"
    r"|[A-Z][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*)"
    r"(?![A-Za-z0-9_])"
)
_BACKTICK = re.compile(r"`([^`]+)`")

_AST_INDEX_CACHE_LIMIT = 4
_AST_PERSIST_SCHEMA = "claim-ast-symbols/1"
_AST_INDEX_CACHE_LOCK = threading.RLock()
_AST_INDEX_CACHE: OrderedDict[
    tuple[object, ...],
    tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]],
] = OrderedDict()
_INVENTORY_CACHE_LIMIT = 4
_INVENTORY_CACHE_LOCK = threading.RLock()
_InventoryResult = tuple[
    tuple[tuple[str, Path], ...], tuple[str, ...], tuple[str, ...], tuple[object, ...],
]
_INVENTORY_CACHE: OrderedDict[tuple[str, int, str], _InventoryResult] = OrderedDict()


def claim_grounding_mode() -> str:
    """Return the strict two-state switch used by the runtime."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def mode() -> str:
    return claim_grounding_mode()


def enabled() -> bool:
    return claim_grounding_mode() == "on"


def claim_grounding_fast_enabled() -> bool:
    """Use the fast path only when explicitly enabled; malformed values stay legacy."""
    try:
        return os.environ.get(FAST_MODE_ENV, "off").strip().lower() == "on"
    except Exception:
        return False


def _inventory_cache_key(
        root: Path, product_revision: str | None) -> tuple[str, int, str] | None:
    if (not claim_grounding_fast_enabled() or not isinstance(product_revision, str)
            or not product_revision):
        return None
    try:
        return root.as_posix(), int(root.stat().st_mtime_ns), product_revision
    except Exception:
        return None


@dataclasses.dataclass(frozen=True)
class ClaimCheck:
    """The host-owned result for one cited path or symbol."""

    cited: str
    kind: str
    exists: bool
    resolved_path: str | None
    closest: tuple[str, ...] = ()
    existence_known: bool = True
    oracle_fact: object | None = dataclasses.field(default=None, compare=False, repr=False)

    def to_dict(self) -> dict:
        result = {
            "cited": self.cited,
            "kind": self.kind,
            "exists": self.exists,
            "resolved_path": self.resolved_path,
            "closest": list(self.closest),
        }
        if self.oracle_fact is not None:
            result["existence_known"] = self.existence_known
            result["probe"] = getattr(self.oracle_fact, "probe", "unknown")
            result["probe_status"] = getattr(self.oracle_fact, "status", "unknown")
            result["revision"] = getattr(self.oracle_fact, "revision", None)
        return result

    as_dict = to_dict


@dataclasses.dataclass(frozen=True)
class _Citation:
    cited: str
    kind: str
    role: str
    check: ClaimCheck


class _AstIndex:
    """Small live AST index; definition locations are relative to ``root``."""

    def __init__(
        self,
        root: Path,
        product_revision: str | None = None,
        *,
        task_id: str = "",
        journal: object | None = None,
    ):
        self.root = root
        self.product_revision = product_revision
        self._closest_by_name: dict[str, tuple[str, ...]] = {}
        self._python_candidates, self.paths, self.files, signature = self._inventory()
        self.symbols: dict[str, tuple[str, ...]] = {}
        cache_key = (self.root.as_posix(), signature)
        with _AST_INDEX_CACHE_LOCK:
            cached = _AST_INDEX_CACHE.get(cache_key)
            if cached is not None:
                _AST_INDEX_CACHE.move_to_end(cache_key)
                self.paths, self.files = cached[0], cached[1]
                self.symbols = dict(cached[2])
                return

        try:
            persist_context, symbols = _load_persisted_ast_symbols(
                self, task_id=task_id, journal=journal, signature=signature)
        except Exception:
            # A cache outage is advisory; continue with the established AST build.
            persist_context, symbols = None, None
            _turn0.emit_cache_bypass(
                journal, name="claim_ast", reason="load_failed")
        if symbols is not None:
            self.symbols = symbols
            _remember_ast_index(cache_key, self.paths, self.files, self.symbols)
            return

        with _parse_cache.gc_paused():  # L112: bulk acyclic index build
            self._build()
        _remember_ast_index(cache_key, self.paths, self.files, self.symbols)
        if persist_context is not None:
            try:
                _save_persisted_ast_symbols(
                    persist_context, self, signature=signature, journal=journal)
            except Exception:
                _turn0.emit_cache_bypass(
                    journal, name="claim_ast", reason="write_failed")

    def _inventory(self) -> _InventoryResult:
        try:
            cache_key = _inventory_cache_key(self.root, self.product_revision)
        except Exception:
            cache_key = None
        if cache_key is not None:
            try:
                with _INVENTORY_CACHE_LOCK:
                    cached = _INVENTORY_CACHE.get(cache_key)
                    if cached is not None:
                        _INVENTORY_CACHE.move_to_end(cache_key)
                        return cached
            except Exception:
                pass

        inventory, complete = self._inventory_uncached()
        if cache_key is not None and complete:
            try:
                if _inventory_cache_key(self.root, self.product_revision) == cache_key:
                    with _INVENTORY_CACHE_LOCK:
                        for previous_key in tuple(_INVENTORY_CACHE):
                            if previous_key[0] == cache_key[0] and previous_key != cache_key:
                                del _INVENTORY_CACHE[previous_key]
                        _INVENTORY_CACHE[cache_key] = inventory
                        _INVENTORY_CACHE.move_to_end(cache_key)
                        while len(_INVENTORY_CACHE) > _INVENTORY_CACHE_LIMIT:
                            _INVENTORY_CACHE.popitem(last=False)
            except Exception:
                pass
        return inventory

    def _inventory_uncached(self) -> tuple[_InventoryResult, bool]:
        paths: set[str] = set()
        files: set[str] = set()
        python_candidates: list[tuple[str, Path]] = []
        signature: list[tuple[object, ...]] = []
        pending: list[tuple[Path, tuple[str, ...]]] = [(self.root, ())]
        complete = True
        while pending:
            directory, parent_parts = pending.pop()
            try:
                entries = os.scandir(directory)
            except OSError:
                complete = False
                continue
            with entries:
                for entry in entries:
                    parts = (*parent_parts, entry.name)
                    if any(part in _SKIP_DIRS for part in parts):
                        continue
                    try:
                        is_file = entry.is_file(follow_symlinks=True)
                        is_directory = entry.is_dir(follow_symlinks=True)
                        recurse = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        complete = False
                        continue
                    relative = "/".join(parts)
                    if not (is_file or is_directory):
                        continue
                    candidate = Path(entry.path)
                    if recurse:
                        pending.append((candidate, parts))
                    paths.add(relative)
                    if is_file:
                        files.add(relative)
                        if candidate.suffix == ".py":
                            python_candidates.append((relative, candidate))
                    try:
                        current = entry.stat(follow_symlinks=True)
                        signature.append((
                            relative, int(current.st_dev), int(current.st_ino),
                            int(current.st_mode), int(current.st_size),
                            int(current.st_mtime_ns), int(current.st_ctime_ns),
                        ))
                    except OSError:
                        complete = False
                        signature.append((relative, None))
        python_candidates.sort(key=lambda item: item[0])
        signature.sort(key=lambda item: str(item[0]))
        return ((tuple(python_candidates), tuple(sorted(paths)), tuple(sorted(files)),
                 tuple(signature)), complete)

    def _build(self) -> None:
        symbols: dict[str, set[str]] = {}
        for relative, candidate in self._python_candidates:
            module = relative[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[:-9]
            try:
                source = candidate.read_text(encoding="utf-8")
                # L112: one parse + walk per (relative path, content); any edit is a new key.
                # Keyed by the RELATIVE path so scratch copies of the root share entries.
                names = _parse_cache.derived(
                    ("claim_grounding.symbols", relative, module), source, relative,
                    lambda tree, relative=relative, module=module: self._symbol_names(
                        tree, relative, module))
            except (OSError, UnicodeError, SyntaxError):
                continue
            if module:
                symbols.setdefault(module, set()).add(relative)
            for name in names:
                symbols.setdefault(name, set()).add(relative)
        self.symbols = {
            name: tuple(sorted(files)) for name, files in sorted(symbols.items())
        }

    @staticmethod
    def _target_names(target: ast.AST) -> tuple[str, ...]:
        if isinstance(target, ast.Name):
            return (target.id,)
        if isinstance(target, (ast.Tuple, ast.List)):
            names = []
            for item in target.elts:
                names.extend(_AstIndex._target_names(item))
            return tuple(names)
        return ()

    def _symbol_names(self, tree: ast.AST, relative: str, module: str) -> frozenset[str]:
        """Every name ``_walk`` records for one file (cacheable: pure in the tree)."""
        local: dict[str, set[str]] = {}
        self._walk(tree, relative, (), local, module)
        return frozenset(local)

    def _add(self, name: str, relative: str, symbols: dict[str, set[str]]) -> None:
        if not name:
            return
        symbols.setdefault(name, set()).add(relative)

    def _walk(self, node: ast.AST, relative: str, prefix: tuple[str, ...],
              symbols: dict[str, set[str]], module: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = child.name
                qualified = ".".join((*prefix, name))
                self._add(name, relative, symbols)
                self._add(qualified, relative, symbols)
                if module:
                    self._add(f"{module}.{qualified}", relative, symbols)
                self._walk(child, relative, (*prefix, name), symbols, module)
                continue
            if isinstance(child, ast.Assign):
                targets = child.targets
            elif isinstance(child, ast.AnnAssign):
                targets = (child.target,)
            else:
                targets = ()
            for target in targets:
                for name in self._target_names(target):
                    qualified = ".".join((*prefix, name))
                    self._add(name, relative, symbols)
                    self._add(qualified, relative, symbols)
                    if module:
                        self._add(f"{module}.{qualified}", relative, symbols)
            self._walk(child, relative, prefix, symbols, module)

    def resolve_symbol(self, cited: str, *, include_closest: bool = True,
                       memoize_closest: bool = False) -> ClaimCheck:
        name = cited.strip().replace("::", ".")
        files = self.symbols.get(name, ())
        if files:
            return ClaimCheck(cited=cited, kind="symbol", exists=True,
                              resolved_path=files[0], closest=())
        if not include_closest:
            choices = ()
        elif memoize_closest:
            cached = self._closest_by_name.get(name)
            if cached is None:
                cached = tuple(difflib.get_close_matches(
                    name, self.symbols, n=3, cutoff=0.35))
                self._closest_by_name[name] = cached
            choices = cached
        else:
            choices = tuple(difflib.get_close_matches(
                name, self.symbols, n=3, cutoff=0.35))
        return ClaimCheck(cited=cited, kind="symbol", exists=False,
                          resolved_path=None, closest=choices)

    def resolve_path(self, cited: str, root: Path, *,
                     include_closest: bool = True) -> ClaimCheck:
        raw = _path_without_locator(cited)
        candidate = Path(raw)
        root_resolved = root.resolve()
        if candidate.is_absolute():
            candidate_path = candidate
        else:
            candidate_path = root / candidate
        try:
            resolved = candidate_path.resolve()
            relative = resolved.relative_to(root_resolved).as_posix()
            exists = resolved.exists()
        except (OSError, RuntimeError, ValueError):
            relative = None
            exists = False
        if exists and relative:
            return ClaimCheck(cited=_display_path(raw), kind="path", exists=True,
                              resolved_path=relative, closest=())
        choices = (_closest_paths(
            _display_path(raw), self.files if Path(raw).suffix else self.paths
        ) if include_closest else [])
        return ClaimCheck(cited=_display_path(raw), kind="path", exists=False,
                          resolved_path=None, closest=tuple(choices))


def _remember_ast_index(
    cache_key: tuple[object, ...],
    paths: tuple[str, ...],
    files: tuple[str, ...],
    symbols: Mapping[str, tuple[str, ...]],
) -> None:
    with _AST_INDEX_CACHE_LOCK:
        for previous_key in tuple(_AST_INDEX_CACHE):
            if previous_key[0] == cache_key[0] and previous_key != cache_key:
                del _AST_INDEX_CACHE[previous_key]
        _AST_INDEX_CACHE[cache_key] = (paths, files, dict(symbols))
        _AST_INDEX_CACHE.move_to_end(cache_key)
        while len(_AST_INDEX_CACHE) > _AST_INDEX_CACHE_LIMIT:
            _AST_INDEX_CACHE.popitem(last=False)


def _ast_symbols_from_payload(
    value: object,
    *,
    files: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != {"schema", "symbols"}:
        raise ValueError("AST index payload is malformed")
    if value.get("schema") != _AST_PERSIST_SCHEMA:
        raise ValueError("AST index schema is unsupported")
    raw_symbols = value.get("symbols")
    if not isinstance(raw_symbols, Mapping):
        raise ValueError("AST symbol mapping is malformed")
    file_set = set(files)
    symbols: dict[str, tuple[str, ...]] = {}
    for symbol, raw_paths in raw_symbols.items():
        if (not isinstance(symbol, str) or not symbol
                or not isinstance(raw_paths, list)
                or any(not isinstance(path, str) or path not in file_set for path in raw_paths)):
            raise ValueError("AST symbol row is malformed")
        paths = tuple(raw_paths)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("AST symbol paths are not canonical")
        symbols[symbol] = paths
    return symbols


def _load_persisted_ast_symbols(
    index: _AstIndex,
    *,
    task_id: str,
    journal: object | None,
    signature: tuple[object, ...],
) -> tuple[_turn0.CacheContext | None, dict[str, tuple[str, ...]] | None]:
    if not _turn0.enabled():
        return None, None
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    module_version = _turn0.implementation_version(
        f"{_AST_PERSIST_SCHEMA}:python-{python_version}", Path(__file__))
    context, reason = _turn0.make_context(
        index.root,
        task_id=task_id,
        name="claim_ast",
        module_version=module_version or "",
        walked_paths=index.files,
    )
    if context is None:
        _turn0.emit_cache_outcome(journal, name="claim_ast", outcome="miss")
        _turn0.emit_cache_bypass(
            journal, name="claim_ast", reason=reason or "context_unavailable")
        return None, None
    payload, reason = _turn0.read_payload(context)
    if reason not in {None, "missing"}:
        _turn0.emit_cache_bypass(journal, name="claim_ast", reason=reason)
    if payload is None or reason is not None:
        _turn0.emit_cache_outcome(journal, name="claim_ast", outcome="miss")
        return context, None
    try:
        symbols = _ast_symbols_from_payload(payload, files=index.files)
    except (TypeError, ValueError, KeyError):
        _turn0.emit_cache_outcome(journal, name="claim_ast", outcome="miss")
        _turn0.emit_cache_bypass(journal, name="claim_ast", reason="payload_invalid")
        return context, None
    _candidates, paths, files, current_signature = index._inventory()
    if (paths != index.paths or files != index.files or current_signature != signature
            or not _turn0.sources_unchanged(context, files)):
        _turn0.emit_cache_outcome(journal, name="claim_ast", outcome="miss")
        _turn0.emit_cache_bypass(journal, name="claim_ast", reason="source_changed")
        return context, None
    _turn0.emit_cache_outcome(journal, name="claim_ast", outcome="hit")
    return context, symbols


def _save_persisted_ast_symbols(
    context: _turn0.CacheContext,
    index: _AstIndex,
    *,
    signature: tuple[object, ...],
    journal: object | None,
) -> None:
    _candidates, paths, files, current_signature = index._inventory()
    if (paths != index.paths or files != index.files or current_signature != signature
            or not _turn0.sources_unchanged(context, files)):
        _turn0.emit_cache_bypass(journal, name="claim_ast", reason="source_changed")
        return
    payload = {
        "schema": _AST_PERSIST_SCHEMA,
        "symbols": {key: list(value) for key, value in sorted(index.symbols.items())},
    }
    reason = _turn0.write_payload(context, payload)
    if reason:
        _turn0.emit_cache_bypass(journal, name="claim_ast", reason=reason)


def _display_path(value: str) -> str:
    value = value.replace("\\", "/").strip().strip("`\"'")
    while value.startswith("./"):
        value = value[2:]
    return value


def _path_without_locator(value: str) -> str:
    value = _display_path(value)
    if "::" in value:
        value = value.split("::", 1)[0]
    value = re.sub(_LOCATOR + "$", "", value)
    return value


def _path_shape(value: str) -> bool:
    token = value.strip().strip("`\"'")
    if "::" in token:
        token = token.split("::", 1)[0]
    token = re.sub(_LOCATOR + "$", "", token)
    return (
        "/" in token or token.startswith(("./", "../"))
        or Path(token).suffix.lower() in _PATH_EXTENSIONS
    )


def _code_shaped_token(value: str) -> bool:
    normalized = value.replace("::", ".")
    return ("." in normalized or "_" in normalized
            or sum(character.isupper() for character in normalized) >= 2)


def _prose_role(role: str | None) -> bool:
    return (role in {"summary", "claim", "basis"}
            or (isinstance(role, str) and role.startswith("evidence")))


def _closest_paths(requested: str, candidates: Sequence[str], n: int = 3) -> list[str]:
    requested = _display_path(requested)
    requested_path = Path(requested)
    requested_parent = requested_path.parent.as_posix()
    requested_stem = requested_path.stem.lower()

    def rank(candidate: str) -> tuple[float, str]:
        candidate_path = Path(candidate)
        score = difflib.SequenceMatcher(None, requested.lower(), candidate.lower()).ratio()
        if candidate_path.parent.as_posix() == requested_parent:
            score += 0.28
        if candidate_path.suffix.lower() == requested_path.suffix.lower():
            score += 0.08
        score += 0.12 * difflib.SequenceMatcher(
            None, requested_stem, candidate_path.stem.lower()).ratio()
        return (-score, candidate)

    return [candidate for _, candidate in sorted((rank(item) for item in candidates))[:n]]


def _clean_text(value: object) -> str:
    return str(value).strip().strip("`\"'").rstrip(".,;:)")


def _split_path_symbol(value: str) -> tuple[str, str | None]:
    value = _clean_text(value)
    if "::" not in value:
        return _path_without_locator(value), None
    path, symbol = value.split("::", 1)
    return _path_without_locator(path), symbol.strip().replace("::", ".") or None


def _fact(check: ClaimCheck) -> str:
    if check.oracle_fact is not None:
        return _workspace_truth.render_fact(check.oracle_fact)
    return _workspace_truth.render_legacy_claim_fact(check.cited, check.closest)


class ClaimGrounding:
    """Inspect tool arguments and retain bounded ungrounded-citation facts."""

    def __init__(
        self, root: str | Path, journal=None, planned_ledger=None, *, task_id: str = "",
    ):
        self.root = Path(root).resolve()
        self.journal = journal
        self.task_id = task_id if isinstance(task_id, str) else ""
        # The ledger is an optional readiness view.  Claim grounding remains
        # the authority for observed paths; this seam only reclassifies an
        # admitted, not-yet-materialized producer as pending(planned).
        self.planned_ledger = planned_ledger
        self._ungrounded: deque[_Citation] = deque(maxlen=32)
        self.last_result: dict | None = None

    def _resolve(
            self, cited: str, kind: str, index: _AstIndex | None,
            role: str | None = None, *,
            workspace_root: str | Path | None = None,
        product_revision: str | None = None) -> ClaimCheck:
        if _workspace_truth.enabled():
            root = workspace_root
            probe = "index" if kind == "symbol" else "stat"
            fact = _workspace_truth.probe_live_workspace(
                root, probe, cited, revision=product_revision, journal=self.journal,
            )
            known = fact.exists is not None
            resolved_path = (
                fact.subject if probe == "stat" and fact.status in {"exists", "missing"}
                else None
            )
            return ClaimCheck(
                cited=cited, kind=kind,
                exists=bool(fact.exists) if known else True,
                resolved_path=resolved_path, closest=fact.closest,
                existence_known=known, oracle_fact=fact,
            )
        if claim_grounding_fast_enabled():
            try:
                if _prose_role(role):
                    if kind == "symbol":
                        return index.resolve_symbol(cited, include_closest=False)
                    return index.resolve_path(cited, self.root, include_closest=False)
                if kind == "symbol":
                    code_shaped = _code_shaped_token(cited)
                    return index.resolve_symbol(
                        cited, include_closest=code_shaped,
                        memoize_closest=code_shaped,
                    )
                return index.resolve_path(cited, self.root)
            except Exception:
                # Optimization errors must preserve the legacy grounding result.
                pass
        if kind == "symbol":
            return index.resolve_symbol(cited)
        return index.resolve_path(cited, self.root)

    def resolve(self, cited: str, kind: str | None = None, *,
                product_revision: str | None = None,
                workspace_root: str | Path | None = None) -> ClaimCheck:
        """Resolve one explicit citation against a fresh workspace index."""
        active_root = (
            workspace_root if workspace_root is not None
            else None if _workspace_truth.enabled() else self.root
        )
        index = None if _workspace_truth.enabled() else _AstIndex(
            self.root, product_revision=product_revision,
            task_id=self.task_id, journal=self.journal,
        )
        value = _clean_text(cited)
        inferred = kind or ("path" if _path_shape(value) else "symbol")
        if inferred == "path_symbol":
            path, symbol = _split_path_symbol(value)
            path_check = self._resolve(
                path, "path", index, workspace_root=active_root,
                product_revision=product_revision,
            )
            if symbol is None:
                return path_check
            symbol_check = self._resolve(
                symbol, "symbol", index, workspace_root=active_root,
                product_revision=product_revision,
            )
            return ClaimCheck(
                cited=value,
                kind="path_symbol",
                exists=path_check.exists and symbol_check.exists,
                resolved_path=path_check.resolved_path,
                closest=tuple(dict.fromkeys((*path_check.closest, *symbol_check.closest)))[:3],
                existence_known=path_check.existence_known and symbol_check.existence_known,
                oracle_fact=(
                    path_check.oracle_fact if not path_check.exists
                    else symbol_check.oracle_fact
                ),
            )
        return self._resolve(
            value, inferred, index, workspace_root=active_root,
            product_revision=product_revision,
        )

    def reconcile_created_path(
            self, cited: object, *, workspace_root: str | Path | None = None,
            product_revision: str | None = None) -> dict | None:
        """Replace pre-dispatch missing-path evidence after a file is created.

        The fresh index resolves the receipt against the live workspace. A path
        is reconciled only after it exists inside that workspace; a stale or
        outside path leaves the earlier evidence untouched.
        """
        if not isinstance(cited, str) or not cited.strip():
            return None
        active_root = (
            workspace_root if workspace_root is not None
            else None if _workspace_truth.enabled() else self.root
        )
        check = self.resolve(
            cited, kind="path", workspace_root=active_root,
            product_revision=product_revision,
        )
        if not check.exists or not check.resolved_path:
            return None

        def relative_path(value: str) -> str | None:
            raw = _path_without_locator(_clean_text(value))
            if not raw:
                return None
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = Path(active_root) / candidate
            try:
                return candidate.resolve(strict=False).relative_to(Path(active_root).resolve()).as_posix()
            except (OSError, RuntimeError, ValueError):
                return None

        created_relative = check.resolved_path
        remaining = [
            item for item in self._ungrounded
            if not (
                item.check.kind == "path"
                and relative_path(item.check.cited) == created_relative
            )
        ]
        self._ungrounded = deque(remaining, maxlen=self._ungrounded.maxlen)

        citation = _Citation(cited=check.cited, kind="path", role="path", check=check)
        self._journal(citation, "file_create", "HOST_OBSERVED")
        row = check.to_dict()
        row["citation_role"] = "path"
        result = {
            "schema": SCHEMA,
            "tool": "file_create",
            "checks": [row],
            "ungrounded": [],
            "pending_planned": [],
            "unplanned_ungrounded": [],
            "authority": "HOST_OBSERVED",
            "rejection": None,
        }
        self.last_result = result
        return result

    @staticmethod
    def _is_path_value(value: str) -> bool:
        return _path_shape(value)

    def _add(
            self, citations: list[_Citation], seen: set[tuple[str, str, str]],
            value: object, kind: str, role: str, index: _AstIndex | None, *,
            workspace_root: str | Path | None = None,
            product_revision: str | None = None) -> None:
        if not isinstance(value, str):
            return
        cited = _clean_text(value)
        if not cited:
            return
        if kind == "path" and "::" in cited:
            path, symbol = _split_path_symbol(cited)
            self._add(
                citations, seen, path, "path", role, index,
                workspace_root=workspace_root, product_revision=product_revision,
            )
            if symbol:
                self._add(
                    citations, seen, symbol, "symbol", role, index,
                    workspace_root=workspace_root, product_revision=product_revision,
                )
            return
        cited = _display_path(cited) if kind == "path" else cited
        key = (cited, kind, role)
        if key in seen:
            return
        seen.add(key)
        check = self._resolve(
            cited, kind, index, role=role, workspace_root=workspace_root,
            product_revision=product_revision,
        )
        citation = _Citation(cited=cited, kind=kind, role=role, check=check)
        citations.append(citation)
        if not check.exists:
            self._ungrounded.append(citation)

    def _text(
            self, citations: list[_Citation], seen: set[tuple[str, str, str]],
            value: object, role: str, index: _AstIndex | None, *,
            workspace_root: str | Path | None = None,
            product_revision: str | None = None) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        for match in _BACKTICK.finditer(value):
            token = _clean_text(match.group(1))
            if _path_shape(token):
                self._add(
                    citations, seen, token, "path", role, index,
                    workspace_root=workspace_root, product_revision=product_revision,
                )
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:]*", token):
                self._add(
                    citations, seen, token, "symbol", role, index,
                    workspace_root=workspace_root, product_revision=product_revision,
                )
        for match in _PATH_TOKEN.finditer(value):
            token = match.group(0)
            if _path_shape(token):
                self._add(
                    citations, seen, token, "path", role, index,
                    workspace_root=workspace_root, product_revision=product_revision,
                )
        for match in _CODE_TOKEN.finditer(value):
            token = match.group(0)
            if _path_shape(token):
                continue
            self._add(
                citations, seen, token, "symbol", role, index,
                workspace_root=workspace_root, product_revision=product_revision,
            )

    def _extract(
            self, tool: str, arguments: Mapping[str, object], index: _AstIndex | None, *,
            workspace_root: str | Path | None = None,
            product_revision: str | None = None) -> list[_Citation]:
        citations: list[_Citation] = []
        seen: set[tuple[str, str, str]] = set()
        for key, value in arguments.items():
            if key in _PATH_FIELDS:
                if isinstance(value, str) and value.strip():
                    self._add(citations, seen, value, "path", key, index,
                              workspace_root=workspace_root, product_revision=product_revision)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    for item in value:
                        self._add(citations, seen, item, "path", key, index,
                                  workspace_root=workspace_root, product_revision=product_revision)
            elif key in _SYMBOL_FIELDS:
                if isinstance(value, str) and value.strip():
                    if _path_shape(value):
                        self._add(citations, seen, value, "path", key, index,
                                  workspace_root=workspace_root, product_revision=product_revision)
                    else:
                        self._add(citations, seen, value, "symbol", key, index,
                                  workspace_root=workspace_root, product_revision=product_revision)
            elif key in {"evidence", "evidence_refs", "basis"}:
                values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else (value,)
                for item in values:
                    self._text(citations, seen, item, key, index,
                               workspace_root=workspace_root, product_revision=product_revision)
        if tool == "done":
            for key in ("summary", "claim"):
                self._text(citations, seen, arguments.get(key), key, index,
                           workspace_root=workspace_root, product_revision=product_revision)
        if tool == "dispute_check":
            self._text(citations, seen, arguments.get("claim"), "claim", index,
                       workspace_root=workspace_root, product_revision=product_revision)
            basis = arguments.get("basis")
            values = basis if isinstance(basis, Sequence) and not isinstance(basis, (str, bytes)) else (basis,)
            for item in values:
                self._text(citations, seen, item, "basis", index,
                           workspace_root=workspace_root, product_revision=product_revision)
        return citations

    def _journal(self, citation: _Citation, tool: str, authority: str) -> None:
        if self.journal is None or not hasattr(self.journal, "emit"):
            return
        check = citation.check.to_dict()
        # Keep ClaimCheck.kind nested and use claim_kind at the journal boundary:
        # ``kind`` is a continuation-journal reserved field.
        self.journal.emit(
            "claim_check",
            producer="stateframe_claim_grounding",
            claim_check=check,
            cited=citation.check.cited,
            claim_kind=citation.check.kind,
            exists=citation.check.exists,
            resolved_path=citation.check.resolved_path,
            citation_role=citation.role,
            action=tool,
            authority=authority,
            dispatch=False,
        )

    def _advisory_locator_misses(
            self, tool: str, args: Mapping[str, object], missing: list[_Citation], *,
            workspace_root: str | Path | None = None,
            product_revision: str | None = None) -> list[_Citation]:
        """Return only claims supported by the available host evidence.

A target reference alone does not establish execution or satisfy a requirement.
"""
        if tool not in _MUTATING_TOOLS or not missing:
            return missing
        advisory = [item for item in missing if item.role == "target_ref"]
        if not advisory or len(advisory) != len(missing):
            return missing
        path_value, old_str = args.get("path"), args.get("old_str")
        if not isinstance(path_value, str) or not isinstance(old_str, str) or not old_str:
            return missing
        active_root = (
            workspace_root if workspace_root is not None
            else None if _workspace_truth.enabled() else self.root
        )
        check = self.resolve(
            path_value, kind="path", workspace_root=active_root,
            product_revision=product_revision,
        )
        if not check.exists or not check.resolved_path:
            return missing
        try:
            text = (Path(active_root) / check.resolved_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return missing
        if text.count(old_str) != 1:
            return missing
        # Advisory only for an unparsed LOCATOR on the same file; a target_ref that
        # names a different (missing) file is a real ungrounded claim.
        own = _display_path(check.resolved_path)
        if any(not _display_path(item.check.cited).startswith(own) for item in advisory):
            return missing
        emit = getattr(self.journal, "emit", None)
        if callable(emit):
            emit("claim_locator_unparsed", tool=tool, path=check.resolved_path,
                 target_ref=[item.check.cited for item in advisory][:3], dispatch=False)
        return []

    def inspect_tool(
            self, tool: str, arguments: Mapping[str, object] | None, *,
            product_revision: str | None = None,
            workspace_root: str | Path | None = None) -> dict:
        """Return grounding evidence; only existing-target mutations are rejected."""
        name = str(tool or "").strip().lower()
        args = arguments if isinstance(arguments, Mapping) else {}
        active_root = (
            workspace_root if workspace_root is not None
            else None if _workspace_truth.enabled() else self.root
        )
        index = None if _workspace_truth.enabled() else _AstIndex(
            self.root, product_revision=product_revision,
            task_id=self.task_id, journal=self.journal,
        )
        citations = self._extract(
            name, args, index, workspace_root=active_root,
            product_revision=product_revision,
        )
        authority = "NON_AUTHORITATIVE" if name == "done" or any(
            not item.check.existence_known or not item.check.exists for item in citations
        ) else "HOST_OBSERVED"
        for citation in citations:
            self._journal(citation, name, authority)
        checks = []
        for citation in citations:
            row = citation.check.to_dict()
            row["citation_role"] = citation.role
            if (not citation.check.exists or citation.check.oracle_fact is not None):
                row["fact"] = _fact(citation.check)
            checks.append(row)
        missing = [item for item in citations if not item.check.exists]
        pending_planned: dict[str, object] = {}
        if self.planned_ledger is not None:
            try:
                from kernel_planned_artifacts import enabled as planned_enabled
                use_planned_view = planned_enabled()
            except (ImportError, ValueError):
                use_planned_view = False
            if use_planned_view:
                for index, citation in enumerate(missing):
                    try:
                        record = self.planned_ledger.planned_for_path(
                            citation.check.resolved_path or citation.check.cited)
                    except (AttributeError, TypeError, ValueError):
                        record = None
                    predicates = getattr(record, "predicates", None)
                    if (record is None or predicates is None
                            or getattr(record, "deletion_break", False)
                            or getattr(predicates, "near_duplicates", ())):
                        continue
                    if getattr(predicates, "exists", None) is not False:
                        continue
                    pending_planned[citation.check.cited] = {
                        "artifact_id": getattr(record, "id", ""),
                        "path": getattr(record, "target_path", citation.check.cited),
                        "readiness": "WAITING_ON_PLANNED_INPUT",
                    }
                    checks[index]["classification"] = "pending(planned)"
                    checks[index]["readiness"] = "WAITING_ON_PLANNED_INPUT"
                    checks[index]["planned_artifact_id"] = getattr(record, "id", "")
        unplanned_missing = [item for item in missing
                             if item.check.cited not in pending_planned]
        unplanned_missing = self._advisory_locator_misses(
            name, args, unplanned_missing, workspace_root=active_root,
            product_revision=product_revision,
        )
        rejection = None
        receipt = None
        if unplanned_missing and name in _MUTATING_TOOLS:
            first = unplanned_missing[0].check
            receipt = {
                "schema": SCHEMA,
                "status": "REJECTED",
                "code": "UNGROUNDED_CLAIM",
                "tool": name,
                "claim": first.cited,
                "path": first.cited,
                "alternatives": list(first.closest[:3]),
                "checks": checks,
                "authority": "HOST_OBSERVED",
            }
            rejection = json.dumps(receipt, ensure_ascii=True, sort_keys=True,
                                   separators=(",", ":"))
        result = {
            "schema": SCHEMA,
            "tool": name,
            "checks": checks,
            "ungrounded": [row for row in checks if not row["exists"]],
            "pending_planned": list(pending_planned.values()),
            "unplanned_ungrounded": [row for row in checks
                                     if not row["exists"]
                                     and row.get("classification") != "pending(planned)"],
            "authority": authority,
            "rejection": rejection,
        }
        if receipt is not None:
            result["receipt"] = receipt
        self.last_result = result
        return result

    def before_dispatch(self, tool: str, arguments: Mapping[str, object] | None) -> str | None:
        return self.inspect_tool(tool, arguments).get("rejection")

    check_tool = inspect_tool

    def recent_ungrounded(
            self, limit: int = 3, *, workspace_root: str | Path | None = None,
            product_revision: str | None = None) -> list[dict]:
        all_rows = list(self._ungrounded)
        if _workspace_truth.enabled():
            active_root = (
                workspace_root if workspace_root is not None else None
            )
            refreshed: list[_Citation] = []
            for citation in all_rows:
                check = self.resolve(
                    citation.check.cited, kind=citation.check.kind,
                    workspace_root=active_root, product_revision=product_revision,
                )
                if check.exists or not check.existence_known:
                    continue
                refreshed.append(_Citation(
                    cited=citation.cited, kind=citation.kind,
                    role=citation.role, check=check,
                ))
            self._ungrounded = deque(refreshed, maxlen=self._ungrounded.maxlen)
            all_rows = refreshed
        rows = all_rows[-max(0, int(limit)):]
        result = []
        for citation in rows:
            row = citation.check.to_dict()
            row.update({"citation_role": citation.role, "fact": _fact(citation.check),
                        "authority": "NON_AUTHORITATIVE"})
            result.append(row)
        return result

    def frame_projection(
            self, *, workspace_root: str | Path | None = None,
            product_revision: str | None = None) -> dict:
        recent = self.recent_ungrounded(
            3, workspace_root=workspace_root, product_revision=product_revision,
        )
        pending = []
        if isinstance(self.last_result, Mapping):
            raw_pending = self.last_result.get("pending_planned", ())
            if isinstance(raw_pending, Sequence) and not isinstance(raw_pending, (str, bytes)):
                pending = [dict(item) for item in raw_pending if isinstance(item, Mapping)]
        return {
            "schema": SCHEMA,
            "mode": "on",
            "recent_ungrounded": recent,
            "facts": [row["fact"] for row in recent],
            "pending_planned": pending[:8],
            "done_claim_authority": "NON_AUTHORITATIVE",
        }

    def model_facts(
            self, *, workspace_root: str | Path | None = None,
            product_revision: str | None = None) -> str:
        facts = self.frame_projection(
            workspace_root=workspace_root, product_revision=product_revision,
        )["facts"]
        if not facts:
            return ""
        return "[Claim grounding]\n" + "\n".join(
            f"FACT: {fact}" for fact in facts
        ) + "\n[/Claim grounding]"


ClaimGrounder = ClaimGrounding


def render_ungrounded_fact(
        check: ClaimCheck | Mapping[str, object], *,
        workspace_root: str | Path | None = None,
        product_revision: str | None = None, journal: object = None) -> str:
    if _workspace_truth.enabled():
        if isinstance(check, ClaimCheck):
            cited, kind = check.cited, check.kind
        else:
            cited = str(check.get("cited", ""))
            kind = str(check.get("kind", ""))
        probe = {"symbol": "index", "path": "stat"}.get(kind, "unsupported")
        fact = _workspace_truth.probe_live_workspace(
            workspace_root, probe, cited, revision=product_revision, journal=journal,
        )
        return _workspace_truth.render_fact(fact)
    if isinstance(check, ClaimCheck):
        return _fact(check)
    rendered = check.get("fact")
    if isinstance(rendered, str) and rendered:
        return rendered
    cited = str(check.get("cited", ""))
    closest = check.get("closest", ())
    if not isinstance(closest, Sequence) or isinstance(closest, (str, bytes)):
        closest = ()
    return _workspace_truth.render_legacy_claim_fact(
        cited, tuple(str(item) for item in closest),
    )


def resolve_claim(root: str | Path, cited: str, kind: str | None = None) -> ClaimCheck:
    return ClaimGrounding(root).resolve(cited, kind=kind)


def ground_tool_arguments(root: str | Path, tool: str,
                          arguments: Mapping[str, object], journal=None,
                          planned_ledger=None) -> dict:
    return ClaimGrounding(root, journal, planned_ledger).inspect_tool(tool, arguments)


__all__ = [
    "ClaimCheck", "ClaimGrounder", "ClaimGrounding", "SCHEMA", "claim_grounding_mode",
    "enabled", "ground_tool_arguments", "mode", "render_ungrounded_fact", "resolve_claim",
]
