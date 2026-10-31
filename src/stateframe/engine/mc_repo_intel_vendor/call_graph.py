"""Code graph index — function-level call graph over a Python repo.

Sibling to blast_radius.py (which is file-import fan-out). This module goes one
level deeper: it records symbol DEFINITIONS (functions/methods/classes) and the
CALL EDGES between them, so consumers can ask precise structural questions that
grep cannot answer reliably:

  - find_definition(symbol)          — precise go-to-def
  - find_callers(symbol, transitive) — "who calls this" (the reachability query)
  - find_callees(symbol, transitive) — "what does this call"
  - snapshot() + diff()              — impact analysis for a PR/edit
  - pagerank() / orphans()           — repo-map ranking + dead-code

Design rules (from project rules #158/#153 on interaction bugs):
  * BIAS TO OVER-REPORT. An unresolved call is linked to every def sharing its
    short name rather than dropped. A too-broad caller set is safe to review; a
    missed caller is exactly the silent break we are trying to catch.
  * NEVER RAISE from a public method. Parse/IO errors degrade to empty results
    and a debug log, mirroring blast_radius.py.
  * ZERO required deps for extraction (stdlib `ast`). networkx is optional and
    only powers transitive traversal / pagerank; without it we fall back to a
    hand-rolled BFS so the core still works.

Symbol id format:  "<relpath>::<QualName>"
  module fn   -> "app/grading.py::grade"
  method      -> "app/grading.py::Grader.grade"
  nested fn   -> "app/grading.py::grade.<locals>.helper"   (rare; kept qualified)

Python-first by design; JS/TS/Vue call-site extraction is a planned extension
(tree-sitter defs already exist in repo_intel/signatures.py).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import logging
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in networkx-less envs
    _NX_AVAILABLE = False
    logger.warning(
        "call_graph: networkx not installed — transitive traversal falls back to "
        "a built-in BFS and pagerank is disabled. Add networkx to requirements.txt."
    )

# Directories that are never part of the source graph (mirrors blast_radius)
_SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".tox", "migrations", ".next", ".nuxt",
    "coverage", ".pytest_cache", ".mypy_cache", ".eggs", ".mission",
    "grader", "graders", "gold", "private", "hidden", "hidden_tests", "backups",
})

_CACHE_FILENAME = ".mc_codegraph_cache.json"


REPO_CACHE_MAX_BYTES_ENV = "STATEFRAME_REPO_INTEL_CACHE_MAX_BYTES"
_REPO_CACHE_DEFAULT_MAX_BYTES = 1 << 30           # 1 GiB 
_REPO_CACHE_RECENT_SECONDS = 600                  # entries used this recently are treated as in use
_REPO_CACHE_EVICT_INTERVAL_SECONDS = 60
_REPO_CACHE_LAST_EVICT: dict[str, float] = {}


def _entry_bytes(entry: Path) -> int:
    total = 0
    stack = [entry]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as items:
                for item in items:
                    try:
                        if item.is_symlink():
                            continue
                        if item.is_dir(follow_symlinks=False):
                            stack.append(Path(item.path))
                        elif item.is_file(follow_symlinks=False):
                            total += item.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def enforce_repo_cache_cap(base: Path, keep: Path | None = None, *, now: float | None = None,
                           force: bool = False) -> list[str]:
    """Bounded LRU eviction of per-repo cache entries under ``base``.

    The cache had no eviction and grew to 13.7 GB / 7,141 entries (one per worktree/run),
    blocking deploys. Entries are the immediate child directories of ``base``; least
    recently used (mtime, refreshed on every use) are removed until the total is under
    the cap. Never removed: ``keep``, entries used within the last 10 minutes (treated as
    held by a live process), symlinks, or anything outside ``base``. Best effort and
    concurrency-tolerant; at most one scan per base per minute unless ``force``.
    Returns the removed entry names.
    """
    try:
        cap = int(os.getenv(REPO_CACHE_MAX_BYTES_ENV) or _REPO_CACHE_DEFAULT_MAX_BYTES)
    except ValueError:
        cap = _REPO_CACHE_DEFAULT_MAX_BYTES
    now = time.time() if now is None else now
    try:
        root = Path(base).resolve()
    except OSError:
        return []
    marker = str(root)
    if not force and now - _REPO_CACHE_LAST_EVICT.get(marker, 0.0) < _REPO_CACHE_EVICT_INTERVAL_SECONDS:
        return []
    _REPO_CACHE_LAST_EVICT[marker] = now
    keep_resolved = Path(keep).resolve() if keep is not None else None
    entries: list[tuple[float, int, Path]] = []
    try:
        with os.scandir(root) as items:
            for item in items:
                try:
                    if item.is_symlink() or not item.is_dir(follow_symlinks=False):
                        continue
                    path = Path(item.path)
                    mtime = item.stat(follow_symlinks=False).st_mtime
                except OSError:
                    continue
                entries.append((mtime, _entry_bytes(path), path))
    except OSError:
        return []
    total = sum(size for _mtime, size, _path in entries)
    removed: list[str] = []
    for mtime, size, path in sorted(entries):
        if total <= cap:
            break
        if keep_resolved is not None and path == keep_resolved:
            continue
        if now - mtime < _REPO_CACHE_RECENT_SECONDS:
            continue
        if path.parent != root:
            continue
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            total -= size
            removed.append(path.name)
    return removed


def repo_cache_file(repo_path, filename: str = _CACHE_FILENAME) -> Path:
    """Cache location for a repo, OUTSIDE the repo (StateFrame vendored copy, 10-05).

    MC writes the cache into a fresh clone; in StateFrame the repo is a sealed task snapshot, and a
    cache file inside it broke exact-membership verification (sample task parent gained this file).
    STATEFRAME_REPO_CACHE_DIR overrides the root; the default is ~/.cache/stateframe/repo-intel.
    """
    repo = Path(repo_path).resolve()
    requested_name = Path(filename)
    if requested_name.is_absolute() or requested_name.name != filename:
        raise ValueError("repo cache filename must be a single relative filename")
    requested = Path(os.getenv("STATEFRAME_REPO_CACHE_DIR") or Path.home() / ".cache" / "stateframe" / "repo-intel").expanduser().resolve()
    try:
        requested.relative_to(repo)
    except ValueError:
        base = requested
    else:
        # A caller-provided cache root must not be allowed to mutate the
        # analysed checkout.  Prefer the normal user cache, then the OS temp
        # directory if the checkout itself lives below the user's home cache.
        candidates = (
            (Path.home() / ".cache" / "stateframe" / "repo-intel").resolve(),
            (Path(tempfile.gettempdir()) / "stateframe" / "repo-intel").resolve(),
            Path("/private/tmp/stateframe/repo-intel"),
        )
        base = next((candidate for candidate in candidates if not candidate.is_relative_to(repo)), candidates[-1])
    key = hashlib.sha256(str(repo).encode("utf-8")).hexdigest()[:16]
    target = base / key
    target.mkdir(parents=True, exist_ok=True)
    try:
        os.utime(target, None)          # mark this entry recently used (LRU)
    except OSError:
        pass
    enforce_repo_cache_cap(base, keep=target)
    return target / requested_name.name
# v7 requires complete, extractor-versioned per-file entries; malformed rows miss.
_CACHE_VERSION = 7
_EXTRACTOR_VERSION = "call-graph-extract/1"
# v6: _FileExtract carries `variables` (module-level assignments) — see
#     _FileVisitor.visit_Assign/visit_AnnAssign. A v5 cache entry lacks the key,
#     so it MUST miss rather than deserialize into a constant-less extract.
#   v3: multi-language extraction (TS/JS/Vue/Go/Rust/Java via tree-sitter) —
#       the cache now holds non-Python file extracts too, so a v2 cache (Python-
#       only) is silently discarded and rebuilt on first warm read.
#   v4: per-file COVERAGE status (indexed/parse_partial/skipped) + per-symbol
#       static complexity PROPERTIES. Both are produced during the single
#       existing parse and persisted, so a warm build surfaces them for free —
#       a v3 cache is discarded and rebuilt once.
#   v5: CallSite now carries the callee NAME token position (name_line/name_col)
#       so the type-resolution layer (app/lsp_resolve.py) can point a real
#       goto-definition at it. A v4 cache lacks those fields; rather than let
#       every cached Python call site resolve at column 0 (which would silently
#       degrade LSP resolution to incorrect answers), the whole cache is
#       discarded and rebuilt once.

# A file larger than this is not parsed — an AST/tree-sitter walk over a
# multi-MB generated bundle costs more than the edges are worth. Recorded as
# coverage status "skipped" (reason "oversized") so its absence is never silent.
_MAX_PARSE_BYTES = 1_500_000

# ── Coverage taxonomy ────────────────────────────────────────────────────────
# Every candidate source file lands in exactly one bucket. The point is ABSENCE
# HONESTY (rule #158): a symbol the graph cannot see must be explained, not
# silently missing, or "no callers" reads as proof when it is merely ignorance.
COVERAGE_INDEXED = "indexed"            # parsed cleanly, defs/calls extracted
COVERAGE_PARSE_PARTIAL = "parse_partial"  # parsed WITH errors — extract is incomplete
COVERAGE_SKIPPED = "skipped"            # parse failed / unreadable / oversized
COVERAGE_EXCLUDED = "excluded"          # deliberately not a graph candidate
COVERAGE_UNKNOWN = "unknown"            # path was outside the indexed candidate universe
COVERAGE_STATUSES = (
    COVERAGE_INDEXED, COVERAGE_PARSE_PARTIAL, COVERAGE_SKIPPED, COVERAGE_EXCLUDED,
    COVERAGE_UNKNOWN,
)
# Statuses that mean "the graph's view of this file is incomplete or absent".
COVERAGE_INCOMPLETE = frozenset({COVERAGE_PARSE_PARTIAL, COVERAGE_SKIPPED, COVERAGE_UNKNOWN})

# Query work must be bounded before the adapter truncates its presentation.  The
# limits are deliberately conservative defaults for an interactive orientation
# request; callers that need a larger traversal must add an explicit owner-level
# continuation rather than silently turning a read into an unbounded scan.
DEFAULT_TRAVERSAL_MAX_DEPTH = 2
DEFAULT_TRAVERSAL_MAX_NODES = 200
DEFAULT_TRAVERSAL_MAX_EDGES = 1_000

# ── Non-callable symbol kinds (module-level assignments) ─────────────────────
# A module-level `_PLAN_LOCK_NAMESPACE = 726586002` is not a callable, so it has
# no place in the CALL graph — but it is very often the exact thing an agent is
# looking for, and before this it was invisible to every action (measured: the
# 2026-08 code-tool A/B lost a whole question to it). These are indexed in a
# SEPARATE namespace from `defs` so they are findable by name WITHOUT polluting
# call resolution, centrality, dead-code or complexity ranking — a constant with
# no callers is not "dead code", and a constant sharing a short name with a
# function must never dilute that function's edge confidence.
KIND_CONSTANT = "constant"   # module-level ALL_CAPS name
KIND_VARIABLE = "variable"   # any other module-level simple name
VALUE_KINDS = frozenset({KIND_CONSTANT, KIND_VARIABLE})

# Decorator short-names that mark a function as an externally-invoked entrypoint
# (so it is not flagged as a dead-code orphan just because nothing local calls it).
_ENTRYPOINT_DECORATORS = frozenset({
    "route", "get", "post", "put", "patch", "delete", "websocket",
    "task", "job", "command", "cli", "fixture", "hookimpl",
    "on_event", "middleware", "listener", "step",
})

# Confidence tiers for a resolved call edge, keyed to the branch in
# _resolve_callee() that produced it. Confidence is ALWAYS an annotation, never
# a filter — even an "unresolved" call stays a reported edge (rule #158).
_CONF_SELF_METHOD = 1.0    # self.method(), matched via receiver's own class
_CONF_LSP = 0.95           # real type resolution (app/lsp_resolve.py) — see below
_CONF_MODULE_FUNC = 0.9    # module.func(), matched via the import table
_CONF_BARE_SINGLE = 0.7    # bare name, exactly one candidate repo-wide
_CONF_BARE_MULTI = 0.4     # bare name, multiple same-named candidates (over-report)
_CONF_UNRESOLVED = 0.0     # no candidate found — synthetic "unresolved:<short>" node

# WHY _CONF_LSP SITS BELOW _CONF_SELF_METHOD AND ABOVE _CONF_MODULE_FUNC:
# `self.method()` matched against the receiver's own class is a syntactic
# certainty — nothing a type engine says can improve on it, so it keeps 1.0 and
# is never re-asked. Everything below it is a NAME match that a real inference
# engine can beat, so a definitive LSP answer outranks it. It is not 1.0 because
# type inference on dynamic Python is very good, not infallible.
#
# The layer is consulted ONLY for edges the name matcher already resolved at a
# LOW tier (bare-name single/multi, unresolved). High-confidence import-table
# hits are never second-guessed: they are cheap and already right, and re-asking
# would spend the whole budget re-confirming answers.
_LSP_CONSULT_BELOW = _CONF_MODULE_FUNC  # consult when best name-match conf < 0.9

# Edge resolution strategy labels (CallSite.strategy / CodeGraph.edge_strategy).
STRATEGY_NAME_MATCH = "name_match"  # import table + short-name matching
STRATEGY_LSP = "lsp"                # real type resolution overrode the name match


def _decode_import_entry(entry: str) -> tuple[str, Optional[str]]:
    """Split an import-table value into (target_file, original_name_or_None).

    An import-table entry is either a bare target relpath (module/namespace
    alias — ``mod.func()`` resolution keys on ``func``, the file is enough) or
    ``"<relpath>::<original_name>"`` (a RENAMED symbol import — the def in the
    target file is indexed under its own declared name, not the local alias
    the call site uses, so the original name must travel with the file).
    """
    if "::" in entry:
        file_part, _, name_part = entry.partition("::")
        return file_part, name_part
    return entry, None


def _confidence_label(score: float) -> str:
    """Bucket a numeric confidence into a label an LLM can act on consistently."""
    if score <= 0.0:
        return "unresolved"
    if score >= 0.9:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


# ── Signature ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Signature:
    """A callable's parameter shape — enough to detect breaking changes."""
    posonly: tuple[str, ...] = ()
    args: tuple[str, ...] = ()          # normal positional-or-keyword params
    defaults: int = 0                   # count of trailing args w/ defaults
    kwonly: tuple[str, ...] = ()
    kwonly_defaults: tuple[str, ...] = ()  # kwonly names that HAVE a default
    has_star_args: bool = False
    has_star_kwargs: bool = False

    @property
    def min_positional(self) -> int:
        """Minimum positional args a caller MUST supply (excluding self/cls)."""
        return len(self.posonly) + len(self.args) - self.defaults

    @property
    def max_positional(self) -> Optional[int]:
        """Max positional args accepted, or None if *args makes it unbounded."""
        if self.has_star_args:
            return None
        return len(self.posonly) + len(self.args)

    def to_dict(self) -> dict:
        return {
            "posonly": list(self.posonly),
            "args": list(self.args),
            "defaults": self.defaults,
            "kwonly": list(self.kwonly),
            "kwonly_defaults": list(self.kwonly_defaults),
            "has_star_args": self.has_star_args,
            "has_star_kwargs": self.has_star_kwargs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Signature":
        return cls(
            posonly=tuple(d.get("posonly", [])),
            args=tuple(d.get("args", [])),
            defaults=int(d.get("defaults", 0)),
            kwonly=tuple(d.get("kwonly", [])),
            kwonly_defaults=tuple(d.get("kwonly_defaults", [])),
            has_star_args=bool(d.get("has_star_args", False)),
            has_star_kwargs=bool(d.get("has_star_kwargs", False)),
        )


def _signature_from_node(node: ast.AST, *, is_method: bool) -> Signature:
    """Build a Signature from a FunctionDef/AsyncFunctionDef, dropping self/cls."""
    a = getattr(node, "args", None)
    if a is None:
        return Signature()

    posonly = [p.arg for p in getattr(a, "posonlyargs", [])]
    args = [p.arg for p in a.args]
    # Drop the implicit receiver for instance/class methods so caller arg-counts line up.
    if is_method:
        if posonly:
            posonly = posonly[1:]
        elif args:
            args = args[1:]

    kwonly = [p.arg for p in a.kwonlyargs]
    # a.kw_defaults is positionally aligned with kwonlyargs; None => required.
    kwonly_defaults = [
        p.arg for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None
    ]
    return Signature(
        posonly=tuple(posonly),
        args=tuple(args),
        defaults=len(a.defaults),
        kwonly=tuple(kwonly),
        kwonly_defaults=tuple(kwonly_defaults),
        has_star_args=a.vararg is not None,
        has_star_kwargs=a.kwarg is not None,
    )


# ── Static complexity properties ─────────────────────────────────────────────
#
# Computed during the SAME parse that produces defs/calls — no extra pass, no
# extra IO, and persisted in the per-file cache so a warm build gets them free.
# These are HINTS for ranking and review, never gates: an accurate-enough
# cyclomatic count and loop-nesting depth tell an agent which symbol is
# genuinely risky to touch, which pagerank centrality alone cannot.

_PROPERTY_KEYS = (
    "cyclomatic", "max_loop_depth", "calls_in_loops", "length",
    "recursive", "unguarded_recursion", "linear_scan_in_loop",
)

_PY_LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)
# A branch point adds 1 to cyclomatic complexity.
_PY_BRANCH_NODES = (
    ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While,
    ast.ExceptHandler, ast.Assert, ast.comprehension,
)
# Nodes that make an enclosed recursive call CONDITIONAL (i.e. it has a base case).
_PY_GUARD_NODES = (ast.If, ast.IfExp, ast.While, ast.Try, ast.ExceptHandler)
# Nested scopes own their own metrics — never fold a nested def into its parent.
_PY_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_PY_MATCH_CASE = getattr(ast, "match_case", None)  # py3.10+


def _call_short_name(func: ast.AST) -> Optional[str]:
    """Short callee name of a call target (`f()` -> f, `self.f()`/`a.b.f()` -> f)."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _empty_properties(length: int = 0) -> dict:
    return {
        "cyclomatic": 1, "max_loop_depth": 0, "calls_in_loops": 0,
        "length": length, "recursive": False, "unguarded_recursion": False,
        "linear_scan_in_loop": False,
    }


def _complexity_properties(node: ast.AST, short_name: str) -> dict:
    """Static metrics for one Python function/method body. Never raises.

    * cyclomatic         — 1 + branch points (if/loop/except/assert/comprehension,
                           plus each extra operand of a boolean operator).
    * max_loop_depth     — deepest nesting of for/while INSIDE this def.
    * calls_in_loops     — call sites that execute once per iteration.
    * recursive          — a direct self-call (`f()` or `self.f()` inside `f`).
    * unguarded_recursion— recursion with NO visible base case (a real hang risk).
                           A base case counts either way it is normally written:
                           the self-call sits under a conditional, OR the body
                           has a conditional early exit (`if …: return/raise`)
                           before it. Missing the second form would flag every
                           textbook `factorial` — a guard that cries wolf on
                           correct code is a guard people learn to ignore.
    * linear_scan_in_loop— hint: work compounds per iteration (a call inside a
                           loop, or nested iteration). Deliberately broad — the
                           `hotspots` deep-loops list uses the stricter depth>=2.
    """
    length = max(
        1, int(getattr(node, "end_lineno", getattr(node, "lineno", 1)) or 1)
        - int(getattr(node, "lineno", 1) or 1) + 1
    )
    props = _empty_properties(length)
    # Set when the body contains a conditional exit (`if …: return/raise`) — the
    # early-return spelling of a base case.
    conditional_exit = False
    bare_self_call = False

    def _walk(n: ast.AST, loop_depth: int, guard_depth: int) -> None:
        nonlocal conditional_exit, bare_self_call
        for child in ast.iter_child_nodes(n):
            if isinstance(child, _PY_NESTED_SCOPES):
                continue  # the nested def gets its own SymbolDef + properties
            if isinstance(child, _PY_BRANCH_NODES) or (
                _PY_MATCH_CASE is not None and isinstance(child, _PY_MATCH_CASE)
            ):
                props["cyclomatic"] += 1
            if isinstance(child, ast.BoolOp):
                props["cyclomatic"] += max(0, len(child.values) - 1)

            is_loop = isinstance(child, _PY_LOOP_NODES)
            child_loop_depth = loop_depth + 1 if is_loop else loop_depth
            if is_loop and child_loop_depth > props["max_loop_depth"]:
                props["max_loop_depth"] = child_loop_depth

            if isinstance(child, (ast.Return, ast.Raise)) and guard_depth > 0:
                conditional_exit = True

            if isinstance(child, ast.Call):
                if loop_depth >= 1:
                    props["calls_in_loops"] += 1
                if _call_short_name(child.func) == short_name:
                    props["recursive"] = True
                    if guard_depth == 0:
                        bare_self_call = True

            _walk(
                child, child_loop_depth,
                guard_depth + (1 if isinstance(child, _PY_GUARD_NODES) else 0),
            )

    try:
        _walk(node, 0, 0)
    except RecursionError:  # pragma: no cover - pathologically deep source
        logger.debug("call_graph: recursion limit computing properties for %s", short_name)
    props["unguarded_recursion"] = bool(bare_self_call and not conditional_exit)
    props["linear_scan_in_loop"] = (
        props["calls_in_loops"] > 0 or props["max_loop_depth"] >= 2
    )
    return props


# ── Extraction records ───────────────────────────────────────────────────────

@dataclass
class SymbolDef:
    symbol: str            # fully-qualified id ("<relpath>::<QualName>")
    file: str              # relpath
    qualname: str          # QualName within the file
    short_name: str        # last dotted component (for over-report matching)
    lineno: int
    end_lineno: int
    kind: str              # "function" | "method" | "class"
    is_entrypoint: bool
    signature: Signature = field(default_factory=Signature)
    # Static complexity metrics (see _complexity_properties). Empty for classes
    # and for any language whose extractor does not compute them.
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["signature"] = self.signature.to_dict()
        d["properties"] = dict(self.properties or {})
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolDef":
        d = dict(d)
        d["signature"] = Signature.from_dict(d.get("signature", {}))
        d["properties"] = dict(d.get("properties") or {})
        return cls(**d)


@dataclass
class CallSite:
    caller: str            # fully-qualified id of the enclosing def (or "<module>" id)
    callee_short: str      # short name of the invoked callable
    n_positional: int      # positional args supplied at the call site
    has_star: bool         # caller used *args / **kwargs spread (arg-count unknowable)
    keywords: tuple[str, ...]  # explicit keyword arg names supplied
    hint_self: bool        # call was `self.<name>(...)`
    hint_module: Optional[str]  # `mod.<name>(...)` — the `mod` token, else None
    lineno: int
    # Position of the CALLEE NAME TOKEN (1-based line, 0-based column, pointing
    # just past the token's last character — what a goto-definition wants). For
    # `obj.method()` this is the `method` token, NOT the start of the call, and
    # for a multi-line receiver chain the line differs from `lineno`. 0/0 means
    # "not captured" (every non-Python extractor, which has no LSP backend
    # anyway) and is treated as "do not ask the type engine".
    name_line: int = 0
    name_col: int = 0
    confidence: float = 0.0
    # Which resolver produced `confidence` — STRATEGY_NAME_MATCH or STRATEGY_LSP.
    # Like `confidence`, only meaningful post-_finalize (see the note below).
    strategy: str = STRATEGY_NAME_MATCH
    # NOTE: the value above is a placeholder at extraction time — real resolution
    # confidence needs repo-global state (_by_short across ALL files) that isn't
    # available during the single-file AST walk, so it can only be computed in
    # CodeGraph._finalize(). build_graph() always repopulates CodeGraph.calls
    # fresh from _FileExtract on every call (cache hit or miss), so the value
    # persisted in the per-file cache JSON is never meaningful — only
    # CodeGraph.calls[i].confidence, set post-finalize, matters to any consumer.

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["keywords"] = list(self.keywords)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CallSite":
        d = dict(d)
        d["keywords"] = tuple(d.get("keywords", []))
        return cls(**d)


@dataclass
class _FileExtract:
    """Per-file extraction result — the unit that gets content-hash cached."""
    defs: list[SymbolDef]
    calls: list[CallSite]
    # local alias -> relpath of imported module file, OR "relpath::original_name"
    # when the alias RENAMES an imported symbol (see _decode_import_entry).
    import_table: dict[str, str]
    # Module-level assignments (VALUE_KINDS). Separate from `defs` so they are
    # findable by name without ever entering call resolution or graph ranking.
    variables: list[SymbolDef] = field(default_factory=list)
    # Coverage verdict for the file this extract came from (see COVERAGE_*).
    # Carried on the extract (not computed later) so it is cached alongside the
    # defs/calls and a warm build reports coverage without re-parsing anything.
    status: str = COVERAGE_INDEXED
    status_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "defs": [d.to_dict() for d in self.defs],
            "calls": [c.to_dict() for c in self.calls],
            "variables": [v.to_dict() for v in self.variables],
            "import_table": self.import_table,
            "status": self.status,
            "status_reason": self.status_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_FileExtract":
        return cls(
            defs=[SymbolDef.from_dict(x) for x in d.get("defs", [])],
            calls=[CallSite.from_dict(x) for x in d.get("calls", [])],
            variables=[SymbolDef.from_dict(x) for x in d.get("variables", [])],
            import_table=dict(d.get("import_table", {})),
            status=str(d.get("status") or COVERAGE_INDEXED),
            status_reason=str(d.get("status_reason") or ""),
        )


_EXTRACT_FIELDS = frozenset({"defs", "calls", "variables", "import_table", "status", "status_reason"})
_SYMBOL_FIELDS = frozenset({
    "symbol", "file", "qualname", "short_name", "lineno", "end_lineno", "kind",
    "is_entrypoint", "signature", "properties",
})
_SIGNATURE_FIELDS = frozenset({
    "posonly", "args", "defaults", "kwonly", "kwonly_defaults",
    "has_star_args", "has_star_kwargs",
})
_CALL_FIELDS = frozenset({
    "caller", "callee_short", "n_positional", "has_star", "keywords", "hint_self",
    "hint_module", "lineno", "name_line", "name_col", "confidence", "strategy",
})


def _string_list(value: object) -> bool:
    return isinstance(value, list) and all(type(item) is str for item in value)


def _finite_number(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _valid_cached_symbol(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _SYMBOL_FIELDS:
        return False
    properties = value.get("properties")
    if any(type(value.get(key)) is not str for key in (
            "symbol", "file", "qualname", "short_name", "kind")):
        return False
    if (type(value.get("lineno")) is not int or type(value.get("end_lineno")) is not int
            or type(value.get("is_entrypoint")) is not bool
            or value["lineno"] < 0 or value["end_lineno"] < 0
            or not isinstance(properties, dict)
            or any(type(key) is not str or key not in _PROPERTY_KEYS
                   or type(item) not in {int, bool} for key, item in properties.items())):
        return False
    signature = value.get("signature")
    return (
        isinstance(signature, dict)
        and set(signature) == _SIGNATURE_FIELDS
        and _string_list(signature.get("posonly"))
        and _string_list(signature.get("args"))
        and type(signature.get("defaults")) is int
        and signature["defaults"] >= 0
        and _string_list(signature.get("kwonly"))
        and _string_list(signature.get("kwonly_defaults"))
        and type(signature.get("has_star_args")) is bool
        and type(signature.get("has_star_kwargs")) is bool
    )


def _valid_cached_call(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _CALL_FIELDS
        and all(type(value.get(key)) is str for key in (
            "caller", "callee_short", "strategy"))
        and type(value.get("n_positional")) is int
        and type(value.get("has_star")) is bool
        and _string_list(value.get("keywords"))
        and type(value.get("hint_self")) is bool
        and (value.get("hint_module") is None or type(value.get("hint_module")) is str)
        and all(type(value.get(key)) is int for key in ("lineno", "name_line", "name_col"))
        and all(value.get(key) >= 0 for key in ("lineno", "name_line", "name_col"))
        and _finite_number(value.get("confidence"))
        and 0.0 <= value["confidence"] <= 1.0
        and value.get("strategy") in {STRATEGY_NAME_MATCH, STRATEGY_LSP}
    )


def _valid_cached_extract(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _EXTRACT_FIELDS:
        return False
    definitions = value.get("defs")
    variables = value.get("variables")
    calls = value.get("calls")
    import_table = value.get("import_table")
    return (
        isinstance(definitions, list)
        and all(_valid_cached_symbol(item) for item in definitions)
        and isinstance(variables, list)
        and all(_valid_cached_symbol(item) for item in variables)
        and isinstance(calls, list)
        and all(_valid_cached_call(item) for item in calls)
        and isinstance(import_table, dict)
        and all(type(key) is str and type(item) is str for key, item in import_table.items())
        and type(value.get("status")) is str
        and value["status"] in COVERAGE_STATUSES
        and type(value.get("status_reason")) is str
    )


def _cached_extract(entry: object, file_hash: str) -> Optional[_FileExtract]:
    """Decode a complete entry only when its content and extractor are current."""
    if not isinstance(entry, dict):
        return None
    cached_hash = entry.get("hash")
    extract = entry.get("extract")
    if (type(cached_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", cached_hash) is None
            or cached_hash != file_hash
            or entry.get("extractor_version") != _EXTRACTOR_VERSION
            or not _valid_cached_extract(extract)):
        return None
    try:
        return _FileExtract.from_dict(extract)
    except (AttributeError, IndexError, KeyError, OverflowError, RecursionError, TypeError, ValueError):
        return None


# ── Per-file AST walk ────────────────────────────────────────────────────────

def _decorator_short(node: ast.AST) -> Optional[str]:
    """Return the last dotted component of a decorator expression, if any."""
    target = node
    if isinstance(node, ast.Call):
        target = node.func
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Name):
        return target.id
    return None


def _is_entrypoint(node: ast.AST, short_name: str) -> bool:
    if short_name.startswith("test_"):
        return True
    if short_name in ("main", "__main__"):
        return True
    # Dunder methods (__init__, __call__, __enter__, …) are invoked implicitly.
    if short_name.startswith("__") and short_name.endswith("__"):
        return True
    for dec in getattr(node, "decorator_list", []):
        if _decorator_short(dec) in _ENTRYPOINT_DECORATORS:
            return True
    return False


class _FileVisitor(ast.NodeVisitor):
    """Single-pass walk collecting defs + call sites with enclosing scope."""

    def __init__(self, relpath: str, import_table: dict[str, str]):
        self.relpath = relpath
        self.import_table = import_table
        self.defs: list[SymbolDef] = []
        # Module-level assignments (constants/variables) — non-callable symbols
        # kept OUT of `defs` so they never enter call resolution (see VALUE_KINDS).
        self.variables: list[SymbolDef] = []
        self.calls: list[CallSite] = []
        self._scope: list[str] = []       # qualname components
        self._class_stack: list[str] = []  # class names (for method detection)
        self._func_depth = 0

    def _qual(self, name: str) -> str:
        return ".".join(self._scope + [name]) if self._scope else name

    def _sym(self, qualname: str) -> str:
        return f"{self.relpath}::{qualname}"

    def _current_caller(self) -> str:
        # Calls at module top-level are attributed to a synthetic "<module>" node.
        return self._sym(".".join(self._scope)) if self._scope else self._sym("<module>")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qual = self._qual(node.name)
        self.defs.append(SymbolDef(
            symbol=self._sym(qual), file=self.relpath, qualname=qual,
            short_name=node.name, lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            kind="class", is_entrypoint=True,  # classes are instantiated externally
        ))
        self._scope.append(node.name)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()
        self._scope.pop()

    def _visit_func(self, node: ast.AST) -> None:
        # A method is a function defined directly in a class body (not nested in
        # another function). @staticmethod takes NO implicit receiver, so its
        # first parameter must NOT be dropped — else its signature is corrupted.
        decs = {_decorator_short(d) for d in getattr(node, "decorator_list", [])}
        is_static = "staticmethod" in decs
        is_method = (
            bool(self._class_stack) and self._func_depth == 0 and not is_static
        )
        qual = self._qual(node.name)
        self.defs.append(SymbolDef(
            symbol=self._sym(qual), file=self.relpath, qualname=qual,
            short_name=node.name, lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            kind="method" if is_method else "function",
            is_entrypoint=_is_entrypoint(node, node.name),
            signature=_signature_from_node(node, is_method=is_method),
            properties=_complexity_properties(node, node.name),
        ))
        self._scope.append(node.name)
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1
        self._scope.pop()

    def _is_nested_in_func(self) -> bool:
        # True if the nearest enclosing scope component is a function, not the class.
        return self._func_depth > 0

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Assign(self, node: ast.Assign) -> None:
        # `name = lambda ...: ...` — promote to a proper def + scoped walk, the
        # Python-side twin of ts_extract's `_maybe_callable_const` (`const f =
        # () => ...`). Without this a lambda export is never emitted as a def
        # at all: no callers/callees, and its own calls get mis-attributed to
        # the enclosing scope's synthetic caller instead of the lambda's own
        # node. Only a SINGLE simple-Name target is promoted — a tuple/starred/
        # attribute target has no one unambiguous declared name to key on.
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Lambda)):
            name = node.targets[0].id
            is_method = bool(self._class_stack) and self._func_depth == 0
            qual = self._qual(name)
            lam = node.value
            self.defs.append(SymbolDef(
                symbol=self._sym(qual), file=self.relpath, qualname=qual,
                short_name=name, lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                kind="method" if is_method else "function",
                is_entrypoint=_is_entrypoint(lam, name),
                signature=_signature_from_node(lam, is_method=is_method),
                properties=_complexity_properties(lam, name),
            ))
            self._scope.append(name)
            self._func_depth += 1
            self.generic_visit(lam)
            self._func_depth -= 1
            self._scope.pop()
            return
        # Module-level constant/variable: `NAME = <anything>`. Only at true module
        # scope (not inside a class or function), and only for single simple Name
        # targets — a tuple/attribute/subscript target has no one declared name.
        if (not self._scope and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            self._record_value(node.targets[0].id, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """`NAME: T = value` / `NAME: T` at module scope — same symbol, typed."""
        if not self._scope and isinstance(node.target, ast.Name):
            self._record_value(node.target.id, node)
        self.generic_visit(node)

    def _record_value(self, name: str, node: ast.AST) -> None:
        """Index one module-level assignment as a non-callable SymbolDef."""
        self.variables.append(SymbolDef(
            symbol=self._sym(name), file=self.relpath, qualname=name,
            short_name=name, lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None) or node.lineno,
            kind=KIND_CONSTANT if name.isupper() else KIND_VARIABLE,
            # Never an entrypoint and never carries call-complexity properties —
            # both fields only mean something for a callable.
            is_entrypoint=False,
        ))

    def visit_Call(self, node: ast.Call) -> None:
        short, hint_self, hint_module = self._resolve_call_target(node.func)
        if short is not None:
            n_pos = sum(1 for a in node.args if not isinstance(a, ast.Starred))
            has_star = any(isinstance(a, ast.Starred) for a in node.args) or any(
                k.arg is None for k in node.keywords  # **kwargs spread
            )
            kws = tuple(k.arg for k in node.keywords if k.arg is not None)
            # End position of `node.func` is the end of the callee NAME token
            # for both `name(...)` (ast.Name) and `recv.name(...)` (ast.Attribute),
            # which is exactly where a goto-definition must point. Guarded:
            # end_lineno/end_col_offset are Optional in the ast typeshed and a
            # synthesized node (see the lambda promotion above) may lack them.
            name_line = getattr(node.func, "end_lineno", None) or 0
            name_col = getattr(node.func, "end_col_offset", None)
            name_col = name_col if isinstance(name_col, int) else 0
            self.calls.append(CallSite(
                caller=self._current_caller(),
                callee_short=short,
                n_positional=n_pos,
                has_star=has_star,
                keywords=kws,
                hint_self=hint_self,
                hint_module=hint_module,
                lineno=node.lineno,
                name_line=name_line,
                name_col=name_col,
            ))
        self.generic_visit(node)

    def _resolve_call_target(
        self, func: ast.AST
    ) -> tuple[Optional[str], bool, Optional[str]]:
        """Return (callee_short_name, is_self_call, module_hint)."""
        if isinstance(func, ast.Name):
            return func.id, False, None
        if isinstance(func, ast.Attribute):
            base = func.value
            if isinstance(base, ast.Name):
                if base.id in ("self", "cls"):
                    return func.attr, True, None
                return func.attr, False, base.id  # possible module.func
            return func.attr, False, None  # chained/att expr — short-name match only
        return None, False, None


def _build_import_table(tree: ast.AST, repo_root: Path, file_dir: Path) -> dict[str, str]:
    """Map local module aliases to the relpath of the imported source file (if local)."""
    table: dict[str, str] = {}

    def _resolve(parts: list[str]) -> Optional[str]:
        if not parts:
            return None
        candidate = repo_root / Path(*parts)
        for c in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if c.exists():
                try:
                    return str(c.relative_to(repo_root))
                except ValueError:
                    return None
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names or []:
                rel = _resolve(alias.name.split("."))
                if rel:
                    table[alias.asname or alias.name.split(".")[0]] = rel
        elif isinstance(node, ast.ImportFrom):
            # `from pkg.mod import thing` — alias `thing` may itself be a submodule file.
            base_parts: Optional[list[str]] = None
            if node.level == 0 and node.module:
                base_parts = node.module.split(".")
            elif node.level and node.level > 0:
                parent = file_dir
                for _ in range(node.level - 1):
                    parent = parent.parent
                try:
                    rel_parent = parent.relative_to(repo_root)
                    base_parts = list(rel_parent.parts) + (node.module.split(".") if node.module else [])
                except ValueError:
                    base_parts = None
            if base_parts is not None:
                base_file = _resolve(base_parts)
                for alias in node.names or []:
                    local = alias.asname or alias.name
                    # Case A: the imported name is itself a submodule/file
                    # (`from pkg import submod as m` then `m.func()`).
                    rel = _resolve(base_parts + [alias.name])
                    if rel:
                        table[local] = rel
                    elif base_file:
                        # Case B: a plain symbol imported from a module file —
                        # encode the target file + the ORIGINAL declared name.
                        # The def in that file is indexed under its own name
                        # (`alias.name`), not the local binding, so a bare
                        # call using a RENAMED import (`import X as y; y()`)
                        # needs the original name to find it
                        # (see _resolve_callee step 2b / _decode_import_entry).
                        table[local] = (
                            base_file if local == alias.name
                            else f"{base_file}::{alias.name}"
                        )
    return table


def _extract_source(source: str, relpath: str) -> list[SymbolDef]:
    """Extract just the symbol DEFS (with signatures) from raw source text.

    Used for cheap baseline snapshots (e.g. `git show base:file`) where we only
    need each symbol's signature/existence, not its call edges or import table.
    Never raises → returns [] on a parse error.
    """
    try:
        tree = ast.parse(source, filename=relpath)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    visitor = _FileVisitor(relpath, {})
    try:
        visitor.visit(tree)
    except RecursionError:
        return []
    return visitor.defs


def _extract_file(file_path: Path, repo_root: Path) -> _FileExtract:
    """Parse one Python file into defs + calls + import table. Never raises.

    A failure is recorded as a coverage STATUS on the returned extract rather
    than swallowed into an indistinguishable empty result — "no symbols here"
    and "this file never parsed" are different answers (project rule #158).
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        relpath = str(file_path.relative_to(repo_root))
    except (OSError, ValueError) as exc:
        return _FileExtract(defs=[], calls=[], import_table={},
                            status=COVERAGE_SKIPPED,
                            status_reason=f"unreadable: {type(exc).__name__}")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        return _FileExtract(defs=[], calls=[], import_table={},
                            status=COVERAGE_SKIPPED,
                            status_reason=f"syntax error at line {exc.lineno or '?'}")
    except (ValueError, MemoryError, RecursionError) as exc:
        return _FileExtract(defs=[], calls=[], import_table={},
                            status=COVERAGE_SKIPPED,
                            status_reason=f"parse failed: {type(exc).__name__}")

    import_table = _build_import_table(tree, repo_root, file_path.parent)
    visitor = _FileVisitor(relpath, import_table)
    try:
        visitor.visit(tree)
    except RecursionError:
        # The tree parsed but the walk bailed — a genuinely PARTIAL extract.
        return _FileExtract(defs=visitor.defs, calls=visitor.calls,
                            variables=visitor.variables,
                            import_table=import_table,
                            status=COVERAGE_PARSE_PARTIAL,
                            status_reason="walk hit the recursion limit")
    return _FileExtract(defs=visitor.defs, calls=visitor.calls,
                        variables=visitor.variables, import_table=import_table)


# ── Content-hash cache ───────────────────────────────────────────────────────

def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_cache(cache_path: Path) -> dict:
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, RecursionError, UnicodeError, ValueError):
        return {}
    if (not isinstance(raw, dict) or raw.get("version") != _CACHE_VERSION
            or raw.get("extractor_version") != _EXTRACTOR_VERSION):
        return {}
    return raw.get("files", {}) if isinstance(raw.get("files"), dict) else {}


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace a cache file atomically; cache writes never expose partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(content)
            handle.flush()
        os.replace(temp_path, path)
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _save_cache(cache_path: Path, files: dict) -> None:
    try:
        payload = json.dumps({
            "version": _CACHE_VERSION,
            "extractor_version": _EXTRACTOR_VERSION,
            "files": files,
        })
        _atomic_write_text(cache_path, payload)
    except (OSError, RecursionError, TypeError, ValueError) as exc:
        # EFS hiccup / read-only / serialization fault: cache is an optimization, never fatal.
        logger.debug("call_graph: could not persist cache to %s: %s", cache_path, exc)


# ── The graph ────────────────────────────────────────────────────────────────

@dataclass
class SymbolRef:
    """Lightweight caller/callee reference returned to consumers."""
    symbol: str
    file: str
    qualname: str
    lineno: int
    kind: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class FileCoverage:
    """One file's place in the coverage taxonomy (see COVERAGE_*)."""
    file: str
    status: str
    reason: str = ""
    symbols: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class CodeGraph:
    """Immutable-ish view over a repo's symbols + call edges.

    Build via build_graph(). Edges point caller -> callee. Unresolved calls land
    on synthetic "unresolved:<short>" nodes so their callers stay queryable.
    """

    def __init__(self) -> None:
        self.defs: dict[str, SymbolDef] = {}          # symbol id -> def
        self._by_short: dict[str, list[str]] = {}     # short name -> [symbol ids]
        # Module-level constants/variables (VALUE_KINDS). A SEPARATE namespace
        # from `defs`/`_by_short` on purpose: these are findable by name but are
        # never call-resolution candidates, never rank in pagerank/god_functions,
        # and never count as dead code (a constant has no callers by definition).
        self.variables: dict[str, SymbolDef] = {}
        self._vars_by_short: dict[str, list[str]] = {}
        self._edges: dict[str, set[str]] = {}         # caller -> {callee symbol ids}
        self._rev: dict[str, set[str]] = {}           # callee -> {caller symbol ids}
        self._edge_confidence: dict[tuple[str, str], float] = {}  # (caller, callee) -> confidence
        # (caller, callee) -> STRATEGY_* that produced the WINNING confidence.
        # Tracked alongside confidence (not derived from it) so a consumer can
        # say "type-resolved" vs "name-matched" without reverse-engineering a
        # float, and so the LSP layer's contribution is measurable in prod.
        self._edge_strategy: dict[tuple[str, str], str] = {}
        # Diagnostics for the type-resolution pass — how many low-confidence
        # Python call sites were asked, and how many got a definitive answer.
        # A layer that silently resolves nothing is the failure mode this makes
        # visible (rule #158).
        self.lsp_stats: dict[str, object] = {
            "enabled": False, "backend": "none", "consulted": 0,
            "resolved": 0, "overridden": 0, "files": 0,
        }
        self.calls: list[CallSite] = []               # every recorded call site
        self._nx: Optional["nx.DiGraph"] = None
        # relpath -> content sha the graph was built from (for the explore
        # verbatim contract's staleness check — see code_graph_tools.explore).
        self.file_hashes: dict[str, str] = {}
        # relpath set of files whose on-disk content DIFFERED from the persisted
        # index cache at build time (i.e. the saved index had lagged an edit
        # before this fresh build refreshed it). Powers explore's staleness
        # banner: a target in this set means "the file changed since the last
        # index save" — an honest signal that a prior consumer's edges lagged.
        self.rebuilt_files: set[str] = set()
        # relpath -> FileCoverage for EVERY candidate source file, indexed or not.
        # The honest denominator behind every "no results" answer this graph gives.
        self.coverage: dict[str, FileCoverage] = {}

    # ---- coverage ----

    def coverage_summary(self) -> dict[str, int]:
        """Counts per coverage status (every status key always present)."""
        counts = {s: 0 for s in COVERAGE_STATUSES}
        for c in self.coverage.values():
            counts[c.status] = counts.get(c.status, 0) + 1
        return counts

    def incomplete_files(self) -> list[FileCoverage]:
        """Files the graph could not fully read — the reason a search may under-report."""
        return sorted(
            (c for c in self.coverage.values() if c.status in COVERAGE_INCOMPLETE),
            key=lambda c: (c.status, c.file),
        )

    def coverage_status(self, relpath: str) -> str:
        """Return explicit uncertainty for paths outside the indexed universe."""
        c = self.coverage.get(relpath)
        return c.status if c is not None else COVERAGE_UNKNOWN

    def index_token(self) -> str:
        """A short, stable token identifying this index build (hash-of-hashes).

        Serves as the "verbatim source as of <token>" anchor in explore output —
        two builds of the same on-disk content produce the same token.
        """
        coverage = "".join(
            f"{path}:{item.status}:{item.reason}:{item.symbols};"
            for path, item in sorted(self.coverage.items())
        )
        joined = (
            "coverage-policy:1;"
            + "".join(f"{k}:{v};" for k, v in sorted(self.file_hashes.items()))
            + coverage
        )
        return _hash_bytes(joined.encode())[:12] if joined else "empty"

    # ---- construction helpers ----

    def _index_def(self, d: SymbolDef) -> None:
        self.defs[d.symbol] = d
        self._by_short.setdefault(d.short_name, []).append(d.symbol)

    def _index_variable(self, d: SymbolDef) -> None:
        """Index a module-level constant/variable in the non-callable namespace.

        A re-assignment of the same module-level name (`X = 1` then `X = 2`)
        collapses to ONE symbol id, so the first (declaring) site wins and the
        short-name list never grows duplicates.
        """
        if d.symbol in self.variables:
            return
        self.variables[d.symbol] = d
        self._vars_by_short.setdefault(d.short_name, []).append(d.symbol)

    def _add_edge(
        self, caller: str, callee: str, confidence: float = 0.0,
        strategy: str = STRATEGY_NAME_MATCH,
    ) -> None:
        self._edges.setdefault(caller, set()).add(callee)
        self._rev.setdefault(callee, set()).add(caller)
        # Multiple call sites can collapse to the same deduplicated edge (e.g. two
        # call sites in the same function calling the same target). MAX-aggregate:
        # if any one call site resolves confidently, the edge is trusted — this
        # mirrors the module's existing over-report bias (rule #158).
        key = (caller, callee)
        if confidence > self._edge_confidence.get(key, 0.0) or key not in self._edge_confidence:
            self._edge_confidence[key] = confidence
            self._edge_strategy[key] = strategy

    def edge_confidence(self, caller: str, callee: str) -> float:
        """Confidence [0.0-1.0] that `caller` really calls `callee`. 0.0 if no such edge."""
        return self._edge_confidence.get((caller, callee), 0.0)

    def edge_strategy(self, caller: str, callee: str) -> str:
        """Which resolver produced this edge's confidence (STRATEGY_*).

        Defaults to STRATEGY_NAME_MATCH for an unknown edge — the honest answer
        for anything this graph did not type-resolve.
        """
        return self._edge_strategy.get((caller, callee), STRATEGY_NAME_MATCH)

    def _resolve_callee(
        self, call: CallSite, file_import_tables: dict[str, dict[str, str]]
    ) -> list[tuple[str, float]]:
        """Resolve a call site to concrete (callee symbol id, confidence) pairs (over-reporting)."""
        caller_file = call.caller.split("::", 1)[0]

        # 1. self.method — precise: match the method on the caller's own class(es).
        if call.hint_self:
            caller_qual = call.caller.split("::", 1)[1] if "::" in call.caller else ""
            # enclosing class = qual minus the trailing method component
            targets = [
                sym for sym in self._by_short.get(call.callee_short, [])
                if self.defs[sym].file == caller_file
                and self.defs[sym].kind == "method"
                and self.defs[sym].qualname.rsplit(".", 1)[0]
                == caller_qual.rsplit(".", 1)[0]
            ]
            if targets:
                return [(t, _CONF_SELF_METHOD) for t in targets]
            # fall through to short-name over-report if the method lives on a base class

        # 2. module.func — resolve via the caller file's import table.
        if call.hint_module:
            entry = file_import_tables.get(caller_file, {}).get(call.hint_module)
            if entry:
                target_file, _ = _decode_import_entry(entry)
                targets = [
                    sym for sym in self._by_short.get(call.callee_short, [])
                    if self.defs[sym].file == target_file
                ]
                if targets:
                    return [(t, _CONF_MODULE_FUNC) for t in targets]

        # 2b. bare identifier whose name matches a known import alias in this
        # file's import table — resolves via the imported symbol's SOURCE FILE
        # (and, when the local alias renamed it, the ORIGINAL declared name)
        # instead of falling straight to an unscoped short-name over-report.
        # This closes two gaps at once: an aliased named import
        # (`import { x as y } from './m'; y()`) that previously resolved to
        # NOTHING (no def anywhere is named `y`), and the same-short-name-
        # different-module ambiguity for a call whose import is unambiguous
        # even though the short name isn't repo-wide.
        if not call.hint_module and not call.hint_self:
            entry = file_import_tables.get(caller_file, {}).get(call.callee_short)
            if entry:
                target_file, orig_name = _decode_import_entry(entry)
                lookup_name = orig_name or call.callee_short
                targets = [
                    sym for sym in self._by_short.get(lookup_name, [])
                    if self.defs[sym].file == target_file
                ]
                if targets:
                    return [(t, _CONF_MODULE_FUNC) for t in targets]

        # 3. bare name or unresolved — link to every def sharing the short name.
        targets = list(self._by_short.get(call.callee_short, []))
        if targets:
            conf = _CONF_BARE_SINGLE if len(targets) == 1 else _CONF_BARE_MULTI
            return [(t, conf) for t in targets]

        # 4. genuinely unknown — synthetic node keeps the caller reachable.
        return [(f"unresolved:{call.callee_short}", _CONF_UNRESOLVED)]

    # ---- type-resolution (LSP) layer ----

    def _defs_at(self, file: str, lineno: int, name: str) -> list[str]:
        """Symbol ids for a definition site the type engine pointed at.

        Matches on (file, lineno) first — jedi reports the `def`/`class` line,
        which is exactly `SymbolDef.lineno`. Falls back to (file, short name) so
        a one-line disagreement (a decorator convention, a future extractor
        tweak) degrades to a still-correct file-scoped match instead of dropping
        the answer on the floor.
        """
        exact = [
            sym for sym in self._by_short.get(name, [])
            if self.defs[sym].file == file and self.defs[sym].lineno == lineno
        ]
        if exact:
            return exact
        return [
            sym for sym in self._by_short.get(name, [])
            if self.defs[sym].file == file
        ]

    def _worth_asking(
        self, call: "CallSite", file_import_tables: dict[str, dict[str, str]]
    ) -> bool:
        """Can a type answer for this call site possibly land on a REPO symbol?

        The type-resolution pass is budgeted, so what it is NOT asked matters as
        much as what it is. Measured on a 203-file repo: 10,427 of 23,625
        low-confidence Python sites were `unresolved` — `logger.debug(...)`,
        `path.exists()`, `d.get(...)` — calls into the stdlib or a third-party
        package. The graph only indexes repo files, so an answer for those is
        DISCARDED by construction (`_defs_at` finds nothing); asking burns ~44%
        of the budget to learn nothing.

        So an `unresolved` site is asked only when there is IMPORT EVIDENCE that
        the name refers to something local: the file imports that exact name.
        That is precisely the alias / re-export case the layer exists to fix
        (`from pkg.facade import compute_total as ct` — `ct` matches no def
        anywhere, so name matching cannot scope it, but the import table proves
        it is ours). Sites that already have repo candidates (bare-single /
        bare-multi) are always worth asking — that ambiguity IS the target.
        """
        if call.confidence > _CONF_UNRESOLVED:
            return True
        caller_file = call.caller.split("::", 1)[0]
        table = file_import_tables.get(caller_file) or {}
        return call.callee_short in table or (
            call.hint_module is not None and call.hint_module in table
        )

    def _apply_lsp_resolution(
        self,
        pending: list[tuple["CallSite", float]],
        repo_root: Optional[Path],
        file_hashes: Optional[dict[str, str]],
        lsp_cache: Optional[dict],
    ) -> dict:
        """Second resolution pass: re-ask a real type engine about LOW-confidence
        Python call sites and override the winners. NEVER raises.

        Returns the new LSP cache dict (empty when nothing was resolved).
        """
        if repo_root is None or not pending:
            return {}
        try:
            from . import lsp_resolve  # local import: optional dep, cycle-free
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("call_graph: lsp_resolve unavailable: %s", exc)
            return {}

        self.lsp_stats["enabled"] = lsp_resolve.is_enabled()
        self.lsp_stats["backend"] = lsp_resolve.backend_name()
        if not lsp_resolve.is_enabled():
            return {}

        # Group by file so each file pays exactly one type-engine setup, and
        # order MOST-AMBIGUOUS FIRST within each file. The pass is budgeted, so
        # ordering decides what a truncated build actually buys: a 0.4
        # bare-multi site (the graph is guessing between several same-named
        # defs) is worth far more than a 0.7 bare-single one (probably already
        # right). Stable within a tier via the call's own position.
        pending = sorted(pending, key=lambda pc: (pc[1], pc[0].name_line, pc[0].name_col))
        sites_by_file: dict[str, list] = {}
        by_key: dict[tuple[str, int, int], list["CallSite"]] = {}
        for call, _conf in pending:
            relpath = call.caller.split("::", 1)[0]
            ref = lsp_resolve.CallSiteRef(
                line=call.name_line, col=call.name_col, name=call.callee_short)
            key = (relpath, ref.line, ref.col)
            if key not in by_key:
                sites_by_file.setdefault(relpath, []).append(ref)
            # Two identical call sites cannot exist, but a lambda promoted to a
            # def can duplicate a position — keep every CallSite for the key.
            by_key.setdefault(key, []).append(call)

        self.lsp_stats["consulted"] = sum(len(v) for v in sites_by_file.values())
        self.lsp_stats["files"] = len(sites_by_file)

        try:
            hits, new_cache = lsp_resolve.resolve_repo(
                repo_root, sites_by_file, file_hashes=file_hashes, cache=lsp_cache)
        except Exception as exc:  # the layer promises not to raise; log unexpected errors
            logger.debug("call_graph: lsp resolution pass failed: %s", exc)
            return {}

        resolved = overridden = 0
        for relpath, positions in hits.items():
            for (line, col), target in positions.items():
                resolved += 1
                targets = self._defs_at(target.file, target.lineno, target.name)
                if not targets:
                    # The type engine pointed somewhere the graph does not index
                    # (an excluded/unparsed file). Fail open — never invent a node.
                    continue
                for call in by_key.get((relpath, line, col), ()):
                    call.confidence = _CONF_LSP
                    call.strategy = STRATEGY_LSP
                    for sym in targets:
                        self._add_edge(call.caller, sym, _CONF_LSP, STRATEGY_LSP)
                    overridden += 1
        self.lsp_stats["resolved"] = resolved
        self.lsp_stats["overridden"] = overridden
        if overridden:
            logger.debug(
                "call_graph: lsp pass — %d/%d low-confidence site(s) type-resolved "
                "across %d file(s) via %s",
                overridden, self.lsp_stats["consulted"], self.lsp_stats["files"],
                self.lsp_stats["backend"],
            )
        return new_cache

    def _finalize(
        self,
        file_import_tables: dict[str, dict[str, str]],
        *,
        repo_root: Optional[Path] = None,
        file_hashes: Optional[dict[str, str]] = None,
        lsp_cache: Optional[dict] = None,
    ) -> dict:
        """Resolve every call site into edges. Returns the new LSP cache dict.

        `repo_root` is what enables the optional type-resolution pass; omitting
        it (the pre-LSP call shape, still used by tests and any in-memory
        caller) yields byte-identical name-matching behaviour.
        """
        # PASS 1 — name matching. Unchanged, and always the baseline: even when
        # the type engine later overrides an edge, the name-matched candidates
        # STAY on the graph. Over-reporting is the module's deliberate bias
        # (rule #158) and a type engine that is wrong about a dynamic
        # receiver must not be able to delete a real edge.
        pending: list[tuple[CallSite, float]] = []
        for call in self.calls:
            resolved = self._resolve_callee(call, file_import_tables)
            call.confidence = max((c for _, c in resolved), default=0.0)
            call.strategy = STRATEGY_NAME_MATCH
            for callee, conf in resolved:
                self._add_edge(call.caller, callee, conf, STRATEGY_NAME_MATCH)
            # Only LOW-confidence PYTHON sites with a captured name position are
            # worth a type query. `self.method()` at 1.0 and import-table hits at
            # 0.9 are already right — re-asking spends budget to confirm.
            if (
                call.confidence < _LSP_CONSULT_BELOW
                and call.name_line > 0
                and call.caller.split("::", 1)[0].endswith((".py", ".pyi"))
                and self._worth_asking(call, file_import_tables)
            ):
                pending.append((call, call.confidence))

        # PASS 2 — real type resolution for what name matching could not pin down.
        new_lsp_cache = self._apply_lsp_resolution(
            pending, repo_root, file_hashes, lsp_cache)

        if _NX_AVAILABLE:
            g = nx.DiGraph()
            g.add_nodes_from(self.defs.keys())
            for caller, callees in self._edges.items():
                for callee in callees:
                    g.add_edge(caller, callee)
            self._nx = g
        return new_lsp_cache

    # ---- public query API ----

    def find_definition(self, symbol: str) -> Optional[SymbolRef]:
        """Resolve a symbol id OR a bare/short name to its definition."""
        d = self.defs.get(symbol)
        if d is None:
            # allow short-name / qualname lookups for ergonomic tool use
            candidates = self._by_short.get(symbol.split("::")[-1].split(".")[-1], [])
            if len(candidates) == 1:
                d = self.defs[candidates[0]]
            elif candidates:
                # ambiguous — prefer exact qualname match if the caller gave one
                for c in candidates:
                    if self.defs[c].qualname == symbol or self.defs[c].symbol == symbol:
                        d = self.defs[c]
                        break
                if d is None:
                    d = self.defs[candidates[0]]
        if d is None:
            # Not a callable — try the module-level constant/variable namespace.
            # Checked AFTER callables so a name that is both never stops
            # resolving to the function it used to resolve to.
            d = self._find_variable(symbol)
        if d is None:
            return None
        return SymbolRef(d.symbol, d.file, d.qualname, d.lineno, d.kind)

    def _find_variable(self, symbol: str) -> Optional[SymbolDef]:
        """Resolve a symbol id / bare name to a module-level constant or variable."""
        d = self.variables.get(symbol)
        if d is not None:
            return d
        candidates = self._vars_by_short.get(
            symbol.split("::")[-1].split(".")[-1], [])
        if not candidates:
            return None
        for c in candidates:  # prefer an exact id/qualname match when ambiguous
            if self.variables[c].symbol == symbol or self.variables[c].qualname == symbol:
                return self.variables[c]
        return self.variables[candidates[0]]

    def symbol_def(self, symbol: str) -> Optional[SymbolDef]:
        """The SymbolDef behind a resolved symbol id — callable OR value kind.

        Consumers that used `graph.defs.get(id)` directly silently returned None
        for a constant once constants became findable; this is the accessor that
        covers both namespaces.
        """
        return self.defs.get(symbol) or self.variables.get(symbol)

    def search_names(self, query: str, *, limit: int = 8) -> list[SymbolRef]:
        """Structural NAME matching for a query no exact lookup resolved.

        Deliberately cheap and dependency-free: the index is already in memory,
        so this is a single pass over the short-name keys with an exact /
        prefix / substring / difflib-similarity ladder. It exists so `explore`
        can never dead-end — a near-miss ("classify_failures", "PLAN_LOCK")
        comes back as the EXACT names that do exist, which the agent can then
        feed to find_definition/find_callers verbatim.

        Ranked best-first. Callables outrank value kinds at equal score, because
        the overwhelmingly common intent behind a code question is a callable.
        """
        q = (query or "").strip()
        if not q:
            return []
        # Tokens the caller might have meant: identifier-ish runs in the query,
        # longest first (a natural-language question contributes several).
        tokens = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", q) if len(t) >= 3]
        tokens.sort(key=len, reverse=True)
        tokens = tokens[:6] or [q]
        lowered = [t.lower() for t in tokens]

        try:
            from difflib import SequenceMatcher  # stdlib, no dep
        except ImportError:  # pragma: no cover - stdlib is always present
            SequenceMatcher = None

        def _score(name: str) -> float:
            nl = name.lower()
            best = 0.0
            for i, t in enumerate(lowered):
                # Earlier (longer) tokens are the likelier subject of the query.
                weight = 1.0 - (0.05 * i)
                if nl == t:
                    s = 1.0
                elif nl.startswith(t) or t.startswith(nl):
                    s = 0.85
                elif t in nl or nl in t:
                    s = 0.7
                elif SequenceMatcher is not None:
                    s = SequenceMatcher(None, nl, t).ratio()
                    if s < 0.72:      # below this it is noise, not a near-miss
                        s = 0.0
                    else:
                        s *= 0.9      # a fuzzy hit never outranks a real substring
                else:                 # pragma: no cover
                    s = 0.0
                best = max(best, s * weight)
            return best

        scored: list[tuple[float, int, SymbolRef]] = []
        for short, ids in self._by_short.items():
            s = _score(short)
            if s <= 0:
                continue
            for sym in ids:
                d = self.defs[sym]
                scored.append((s, 0, SymbolRef(d.symbol, d.file, d.qualname, d.lineno, d.kind)))
        for short, ids in self._vars_by_short.items():
            s = _score(short)
            if s <= 0:
                continue
            for sym in ids:
                d = self.variables[sym]
                scored.append((s, 1, SymbolRef(d.symbol, d.file, d.qualname, d.lineno, d.kind)))
        scored.sort(key=lambda t: (-t[0], t[1], t[2].file, t[2].lineno))
        return [r for _s, _k, r in scored[: max(1, int(limit))]]

    def _ids_for(self, symbol: str) -> list[str]:
        """Expand a symbol id / short name to concrete node ids.

        Handles three cases: an exact def id, a synthetic edge node that is not a
        def (e.g. "unresolved:<short>" — needed so diff() can trace callers of a
        removed symbol), and a bare/qualified short name (over-reporting).
        """
        if symbol in self.defs:
            return [symbol]
        if symbol in self._rev or symbol in self._edges:
            return [symbol]
        return list(self._by_short.get(symbol.split("::")[-1].split(".")[-1], []))

    def definition_candidates(self, symbol: str) -> list[SymbolRef]:
        """Return all exact candidates without silently choosing a short-name match."""
        if symbol in self.defs:
            ids = [symbol]
        else:
            short = symbol.split("::")[-1].split(".")[-1]
            ids = list(self._by_short.get(short, ()))
            exact = [candidate for candidate in ids
                     if self.defs[candidate].qualname == symbol
                     or self.defs[candidate].symbol == symbol]
            if exact:
                ids = exact
            if not ids:
                variable_ids = list(self._vars_by_short.get(short, ()))
                exact_variables = [candidate for candidate in variable_ids
                                   if self.variables[candidate].qualname == symbol
                                   or self.variables[candidate].symbol == symbol]
                ids = exact_variables or variable_ids
                return self._as_refs(ids)
        return self._as_refs(ids)

    def _bounded_reach(
            self,
            seeds: Iterable[str],
            adj: dict[str, set[str]],
            *,
            max_depth: int = DEFAULT_TRAVERSAL_MAX_DEPTH,
            max_nodes: int = DEFAULT_TRAVERSAL_MAX_NODES,
            max_edges: int = DEFAULT_TRAVERSAL_MAX_EDGES,
    ) -> tuple[set[str], dict[str, object]]:
        """Traverse a graph with work limits and an honest continuation summary."""
        depth_limit = max(0, int(max_depth))
        node_limit = max(1, int(max_nodes))
        edge_limit = max(1, int(max_edges))
        queue = deque((seed, 0) for seed in sorted(set(seeds)))
        seen = {seed for seed, _depth in queue}
        frontier: set[str] = set()
        examined_edges = 0
        truncated = len(seen) > node_limit
        while queue and not truncated:
            current, depth = queue.popleft()
            for nxt in sorted(adj.get(current, ())):
                examined_edges += 1
                if examined_edges > edge_limit:
                    frontier.add(nxt)
                    truncated = True
                    break
                if nxt in seen:
                    continue
                if depth >= depth_limit or len(seen) >= node_limit:
                    frontier.add(nxt)
                    truncated = True
                    break
                seen.add(nxt)
                queue.append((nxt, depth + 1))
            if truncated:
                break
        return seen, {
            "visited": len(seen),
            "examined_edges": examined_edges,
            "max_depth": depth_limit,
            "truncated": truncated,
            "frontier": sorted(frontier)[:24],
        }

    def find_callers_bounded(
            self,
            symbol: str,
            *,
            max_depth: int = DEFAULT_TRAVERSAL_MAX_DEPTH,
            max_nodes: int = DEFAULT_TRAVERSAL_MAX_NODES,
            max_edges: int = DEFAULT_TRAVERSAL_MAX_EDGES,
    ) -> tuple[list[SymbolRef], dict[str, object]]:
        """Find transitive callers while bounding traversal before row slicing."""
        targets = self._ids_for(symbol)
        if not targets:
            return [], {"visited": 0, "examined_edges": 0, "max_depth": max_depth,
                        "truncated": False, "frontier": []}
        found, work = self._bounded_reach(
            targets, self._rev, max_depth=max_depth, max_nodes=max_nodes, max_edges=max_edges,
        )
        return self._as_refs(found - set(targets)), work

    def find_callers(self, symbol: str, *, transitive: bool = False) -> list[SymbolRef]:
        targets = self._ids_for(symbol)
        if not targets:
            return []
        found: set[str] = set()
        if transitive:
            found = self._reach(targets, self._rev)
        else:
            for t in targets:
                found |= self._rev.get(t, set())
        return self._as_refs(found - set(targets))

    def find_callees(self, symbol: str, *, transitive: bool = False) -> list[SymbolRef]:
        targets = self._ids_for(symbol)
        if not targets:
            return []
        found: set[str] = set()
        if transitive:
            found = self._reach(targets, self._edges)
        else:
            for t in targets:
                found |= self._edges.get(t, set())
        return self._as_refs(found - set(targets))

    def _reach(self, seeds: Iterable[str], adj: dict[str, set[str]]) -> set[str]:
        """BFS transitive closure (works with or without networkx)."""
        seen: set[str] = set()
        stack = list(seeds)
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):  # noqa: E741
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def _as_refs(self, ids: Iterable[str]) -> list[SymbolRef]:
        refs = []
        for sym in ids:
            d = self.symbol_def(sym)
            if d is not None:
                refs.append(SymbolRef(d.symbol, d.file, d.qualname, d.lineno, d.kind))
        return sorted(refs, key=lambda r: (r.file, r.lineno))

    def signature_of(self, symbol: str) -> Optional[Signature]:
        for sym in self._ids_for(symbol):
            return self.defs[sym].signature
        return None

    def pagerank(self, top_n: Optional[int] = 25) -> list[SymbolRef]:
        """Rank symbols by call-graph centrality (repo-map ordering).

        ``top_n=None`` returns the FULL ranking so a caller can report an exact
        pre-truncation total alongside the rows it chooses to show.
        """
        if not _NX_AVAILABLE or self._nx is None or self._nx.number_of_nodes() == 0:
            # degree fallback: rank by number of callers
            ranked = sorted(
                self.defs.values(),
                key=lambda d: len(self._rev.get(d.symbol, ())),
                reverse=True,
            )
            return [SymbolRef(d.symbol, d.file, d.qualname, d.lineno, d.kind)
                    for d in ranked[:top_n]]
        try:
            scores = nx.pagerank(self._nx)
        except Exception as exc:  # pragma: no cover
            logger.debug("call_graph: pagerank failed: %s", exc)
            return []
        ranked = sorted(
            (s for s in scores if s in self.defs),
            key=lambda s: scores[s],
            reverse=True,
        )
        return [self._as_refs([s])[0] for s in ranked[:top_n] if self._as_refs([s])]

    def tests_reaching(self, symbol: str) -> list[SymbolRef]:
        """Transitive callers of `symbol` that are test functions (test_* names)."""
        return [
            r for r in self.find_callers(symbol, transitive=True)
            if r.symbol in self.defs and self.defs[r.symbol].short_name.startswith("test_")
        ]

    def symbols_in_files(self, files: set[str]) -> list[str]:
        return [s for s, d in self.defs.items() if d.file in files]

    def properties_of(self, symbol: str) -> dict:
        """Static complexity properties for a symbol id / short name ({} if none)."""
        for sym in self._ids_for(symbol):
            d = self.defs.get(sym)
            if d is not None:
                return dict(d.properties or {})
        return {}

    def by_property(
        self, key: str, *, minimum=None, is_true: bool = False,
        top_n: Optional[int] = None,
    ) -> list[tuple[SymbolRef, dict]]:
        """Symbols ranked by a complexity property. Empty when nothing qualifies.

        ``minimum`` filters + ranks a NUMERIC property descending; ``is_true``
        filters a BOOLEAN property (ranked by length, longest first — the most
        expensive one to read is the one worth looking at). ``top_n=None``
        returns everything so the caller can report an exact total.
        """
        rows: list[tuple[SymbolRef, dict, float]] = []
        for d in self.defs.values():
            props = d.properties or {}
            if not props:
                continue
            val = props.get(key)
            if is_true:
                if not val:
                    continue
                rank = float(props.get("length", 0) or 0)
            else:
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    continue
                if minimum is not None and val < minimum:
                    continue
                rank = float(val)
            rows.append(
                (SymbolRef(d.symbol, d.file, d.qualname, d.lineno, d.kind), props, rank)
            )
        rows.sort(key=lambda t: (-t[2], t[0].file, t[0].lineno))
        return [(r, p) for r, p, _ in rows[:top_n]]

    def multi_source_impact(
        self, seeds: Iterable[str], *, depth: int = 2,
    ) -> list[tuple[SymbolRef, int]]:
        """ONE multi-source inbound BFS from `seeds` → [(ref, hop), …], hop-ordered.

        This is `impact_radius` generalised to a whole seed SET (a diff), run as
        a single traversal rather than N independent ones — so a symbol reachable
        from several seeds is visited once and carries its SHORTEST hop distance.
        Seeds are excluded from their own impact (a changed function is not
        "impacted by" itself); classes still expand to their methods so a class
        edit reaches every caller of every method (the impact_radius rule (b)).
        """
        seed_ids: set[str] = set()
        for s in seeds:
            seed_ids.update(self._ids_for(s))
        for s in list(seed_ids):
            seed_ids.update(self._child_methods(s))
        if not seed_ids:
            return []

        hops: dict[str, int] = {}
        frontier = set(seed_ids)
        for hop in range(1, max(0, int(depth)) + 1):
            nxt: set[str] = set()
            for node in frontier:
                for caller in self._rev.get(node, ()):
                    if caller in seed_ids or caller in hops:
                        continue
                    hops[caller] = hop
                    nxt.add(caller)
            if not nxt:
                break
            frontier = nxt

        out: list[tuple[SymbolRef, int]] = []
        for sym, hop in hops.items():
            d = self.defs.get(sym)
            if d is None:
                continue  # synthetic unresolved node — not a reviewable location
            out.append((SymbolRef(d.symbol, d.file, d.qualname, d.lineno, d.kind), hop))
        out.sort(key=lambda t: (t[1], t[0].file, t[0].lineno))
        return out

    def god_functions(self, min_callers: int = 8, top_n: Optional[int] = 20) -> list[SymbolRef]:
        """Highest fan-in symbols (many callers) — architectural hotspots to change carefully."""
        scored = [
            (d, len(self._rev.get(d.symbol, ())))
            for d in self.defs.values() if d.kind != "class"
        ]
        scored = [(d, n) for d, n in scored if n >= min_callers]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [SymbolRef(d.symbol, d.file, d.qualname, d.lineno, d.kind)
                for d, _ in scored[:top_n]]

    def orphans(self) -> list[SymbolRef]:
        """Defs with zero callers that are not entrypoints (dead-code candidates)."""
        out = []
        for d in self.defs.values():
            if d.kind == "class" or d.is_entrypoint:
                continue
            if not self._rev.get(d.symbol):
                out.append(SymbolRef(d.symbol, d.file, d.qualname, d.lineno, d.kind))
        return sorted(out, key=lambda r: (r.file, r.lineno))

    # ---- container relationships (methods of a class) ----
    #
    # Our graph records only CALL edges — there is no explicit `contains` edge
    # kind. The containment relation (class -> its methods) is recoverable from
    # the qualname prefix: "app/x.py::Foo.bar" is contained by "app/x.py::Foo".
    # This is enough to port CodeGraph's two impact refinements (their #536) —
    # (a) NOT climbing a containment edge upward, and (b) expanding a class
    # downward to its methods — WITHOUT adding a whole edge kind.

    def _child_methods(self, class_symbol: str) -> list[str]:
        """Symbol ids of methods/nested defs contained by a class def id.

        Membership is by-file + qualname-prefix ("Foo." for a class "Foo"), so a
        same-named class in another file never captures foreign methods.
        """
        d = self.defs.get(class_symbol)
        if d is None or d.kind != "class":
            return []
        prefix = d.qualname + "."
        return [
            sym for sym, sd in self.defs.items()
            if sd.file == d.file and sd.qualname.startswith(prefix)
        ]

    def impact_radius(self, symbol: str, *, max_depth: int = 6) -> list[SymbolRef]:
        """Blast radius of changing `symbol` — transitive INCOMING callers, with
        CodeGraph's two containment refinements (their #536).

        Differs from ``find_callers(transitive=True)`` in exactly two ways:
          (a) EXCLUDE-CONTAINS-WHEN-CLIMBING — a leaf method's impact never
              balloons to its whole class. We already store no upward containment
              edge, so climbing is pure call edges; this is preserved explicitly
              by seeding ONLY the resolved target ids (never their container).
          (b) CONTAINER EXPANSION — when the target is a CLASS, changing it can
              break any caller of ANY of its methods, so we seed the class AND
              all its child methods, then climb callers from all of them.

        Depth-bounded BFS over ``_rev`` (call edges only). Over-reports by design
        (project rule #158): a too-wide radius is safe to review; a missed caller is
        the silent break we exist to catch.
        """
        seeds = set(self._ids_for(symbol))
        if not seeds:
            return []
        # (b) container expansion — add each class target's methods as seeds.
        for s in list(seeds):
            seeds.update(self._child_methods(s))
        found: set[str] = set()
        frontier = set(seeds)
        depth = 0
        while frontier and depth < max_depth:
            nxt: set[str] = set()
            for node in frontier:
                for caller in self._rev.get(node, ()):  # incoming call edge only
                    if caller not in found and caller not in seeds:
                        found.add(caller)
                        nxt.add(caller)
            frontier = nxt
            depth += 1
        return self._as_refs(found - seeds)


# ── Dynamic-dispatch boundary detection ──────────────────────────────────────
#
# Ported (idea + patterns) from CodeGraph's src/mcp/dynamic-boundaries.ts
# (colbymchenry/codegraph, MIT). When a static call graph can't connect a flow
# because the next hop is chosen at RUNTIME (a computed member call, getattr
# dispatch, or a string-keyed bus emit), a graph that stays silent — or worse,
# guesses an edge — misleads the agent. CodeGraph's insight: HONESTLY ANNOUNCE
# the boundary instead. We port only the Python-relevant forms.

# Attribute-method names that route by a runtime STRING KEY (event/command bus,
# Neuron emits, SQS publish → a consumer resolved elsewhere). A call to one of
# these is a flow that "continues dynamically" at the matching handler/consumer.
_STRING_KEYED_DISPATCH = frozenset({
    "emit", "dispatch", "send", "publish", "trigger", "fire",
    "handle", "route", "call_tool", "invoke",
})


@dataclass
class DispatchBoundary:
    """A runtime-resolved call site the static graph cannot follow through."""
    file: str
    lineno: int
    kind: str          # "getattr" | "computed_subscript" | "string_keyed_bus"
    key_expr: str      # the token/expression that selects the target at runtime
    note: str          # human/agent-readable "flow continues dynamically" note


def _expr_src(node: ast.AST) -> str:
    """Best-effort source text for a small expression node (key display only)."""
    try:
        return ast.unparse(node)  # py3.9+
    except Exception:  # pragma: no cover - unparse is available on our runtimes
        return getattr(node, "id", None) or getattr(node, "attr", None) or "?"


class _DispatchVisitor(ast.NodeVisitor):
    """Detect the Python dynamic-dispatch forms CodeGraph announces."""

    def __init__(self, relpath: str):
        self.relpath = relpath
        self.boundaries: list[DispatchBoundary] = []

    def _add(self, node: ast.AST, kind: str, key_expr: str, note: str) -> None:
        self.boundaries.append(DispatchBoundary(
            file=self.relpath, lineno=getattr(node, "lineno", 0),
            kind=kind, key_expr=key_expr, note=note,
        ))

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # 1. computed-subscript call — handlers[x](...) / registry[key](...)
        if isinstance(func, ast.Subscript):
            base = _expr_src(func.value)
            key = _expr_src(func.slice)
            self._add(
                node, "computed_subscript", key,
                f"flow continues dynamically here (key={key}); the callee is looked up "
                f"in `{base}` at runtime — grep for where `{base}` is populated to find targets",
            )
        # 2. getattr(obj, name)(...) — reflection dispatch. The OUTER call's func
        #    is the `getattr(...)` Call node itself, not an Attribute.
        elif (isinstance(func, ast.Call) and isinstance(func.func, ast.Name)
                and func.func.id == "getattr" and len(func.args) >= 2):
            key = _expr_src(func.args[1])
            self._add(
                node, "getattr", key,
                f"flow continues dynamically here (key={key}); the method is resolved by "
                f"getattr at runtime — grep for the string values `{key}` can take",
            )
        elif isinstance(func, ast.Attribute):
            # 3. string-keyed bus emit — bus.emit('x', ...), neuron.publish('x')
            if func.attr in _STRING_KEYED_DISPATCH and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    key = repr(first.value)
                    recv = _expr_src(func.value)
                    self._add(
                        node, "string_keyed_bus", key,
                        f"flow continues dynamically here (key={key}); `{recv}.{func.attr}` "
                        f"routes to a handler/consumer registered under {key} elsewhere — "
                        f"grep for {key} to find the receiving end",
                    )
        self.generic_visit(node)


def dispatch_boundaries_in_file(source: str, relpath: str) -> list[DispatchBoundary]:
    """Scan one file's source for runtime dispatch sites. Never raises → [].

    Python uses the stdlib-ast visitor; TS/JS/Vue use a tree-sitter scan for the
    JS-flavoured dynamic forms (computed-member call, dynamic import(), string-
    keyed bus emit). Other tree-sitter languages have no boundary scan yet → [].
    """
    ext = "." + relpath.rsplit(".", 1)[-1].lower() if "." in relpath else ""
    if ext != ".py":
        try:
            from . import ts_dispatch  # noqa: PLC0415
            return ts_dispatch.dispatch_boundaries_in_file(source, relpath)
        except Exception:  # pragma: no cover - degrade if the ts scan/module is unavailable
            return []
    try:
        tree = ast.parse(source, filename=relpath)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    v = _DispatchVisitor(relpath)
    try:
        v.visit(tree)
    except RecursionError:
        return []
    return v.boundaries


def dispatch_boundaries_near(
    repo_root: Path, relpath: str, lineno: int, end_lineno: int
) -> list[DispatchBoundary]:
    """Dispatch boundaries whose call site falls inside [lineno, end_lineno] of a file.

    Used by the `explore` action to annotate a target's own body: when the flow
    OUT of the shown symbol goes through a runtime-resolved edge, say so instead
    of leaving the agent to assume the graph's callee set is complete.
    """
    try:
        src = (repo_root / relpath).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lo, hi = min(lineno, end_lineno), max(lineno, end_lineno)
    return [b for b in dispatch_boundaries_in_file(src, relpath) if lo <= b.lineno <= hi]


# ── Skeletonization (adaptive-budget helper) ─────────────────────────────────

def signature_skeleton(source_lines: list[str], d: "SymbolDef") -> str:
    """A signature-only skeleton of a def: its `def ...:`/`class ...:` header
    lines (through the colon) plus a `    ...  # body elided (N lines)` marker.

    Ported idea from CodeGraph's adaptive budget: OFF-SPINE symbols are shown as
    skeletons, not full bodies, so a large `explore` answer stays within budget
    while still telling the agent the symbol's shape and where it lives.
    Multi-line signatures (wrapped params) are preserved through the first line
    that ends the header (the one containing the `:` that opens the block).
    """
    start = max(1, d.lineno)
    end = min(len(source_lines), max(start, d.end_lineno))
    header: list[str] = []
    # Capture the def/class header through the line whose bracket-depth returns to
    # 0 AND ends the signature with a colon (handles wrapped multi-line params).
    depth = 0
    for i in range(start, end + 1):
        line = source_lines[i - 1]
        header.append(line)
        depth += line.count("(") + line.count("[") + line.count("{")
        depth -= line.count(")") + line.count("]") + line.count("}")
        if depth <= 0 and line.rstrip().endswith(":"):
            break
    body_lines = end - (start + len(header) - 1)
    indent = " " * (len(header[-1]) - len(header[-1].lstrip()) + 4) if header else "    "
    skel = "\n".join(header)
    if body_lines > 0:
        skel += f"\n{indent}...  # body elided ({body_lines} line{'s' if body_lines != 1 else ''})"
    return skel


# ── Build ────────────────────────────────────────────────────────────────────

def _iter_py_files(root: Path) -> Iterable[Path]:
    for py_file in sorted(root.rglob("*.py")):
        try:
            parts = py_file.relative_to(root).parts
        except ValueError:
            continue
        if any(p in _SKIP_DIRS or p.endswith(".egg-info") for p in parts):
            continue
        yield py_file


# Extensions the tree-sitter multi-language extractor (ts_extract.py) claims.
# .py always stays on the proven stdlib-ast path in THIS module — never migrated.
try:
    from .ts_extract import SUPPORTED_EXTENSIONS as _TS_EXTS
except Exception:  # pragma: no cover - degrade to Python-only if the module/deps are absent
    _TS_EXTS = frozenset()

_GRAPH_EXTENSIONS = frozenset({".py"}) | _TS_EXTS


def _iter_source_files(root: Path) -> Iterable[Path]:
    """Every graph-relevant source file (Python + any tree-sitter language).

    Thin view over `_scan_source_files` — ONE walk implementation, so the
    indexed set can never drift from the set `coverage` reports on.
    """
    return _scan_source_files(root)[0]


def _excluded_reason(parts: tuple[str, ...]) -> Optional[str]:
    """Why a graph-extension file under `parts` is deliberately not indexed."""
    for p in parts:
        if p in _SKIP_DIRS:
            return f"in excluded directory '{p}'"
        if p.endswith(".egg-info"):
            return "in packaging metadata directory"
    return None


def _scan_source_files(root: Path) -> tuple[list[Path], list[tuple[str, str]]]:
    """(indexable source files, [(relpath, exclusion reason), …]).

    One walk produces BOTH halves — the excluded half exists so `coverage` can
    say "vendored/generated code was deliberately not indexed" instead of
    leaving a hole the agent has to guess at. Only files with a graph-relevant
    EXTENSION are recorded as excluded; the millions of non-code files under
    node_modules are simply not candidates and would drown the report.
    """
    files: list[Path] = []
    excluded: list[tuple[str, str]] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in _GRAPH_EXTENSIONS:
            continue
        try:
            rel = f.relative_to(root)
        except ValueError:
            continue
        reason = _excluded_reason(rel.parts)
        if reason is not None:
            excluded.append((str(rel), reason))
            continue
        files.append(f)
    return files, excluded


def _extract_any_file(file_path: Path, repo_root: Path,
                      repo_files: Optional[set[str]] = None) -> "_FileExtract":
    """Dispatch a file to its language extractor by extension. Never raises.

    Python → the stdlib-ast path in this module (unchanged, default for .py).
    Everything else → the tree-sitter path in ts_extract.py (TS/JS/Vue/Go/Rust/
    Java). An unknown/failed extension degrades to an empty extract.
    """
    ext = file_path.suffix.lower()
    try:
        if file_path.stat().st_size > _MAX_PARSE_BYTES:
            return _FileExtract(defs=[], calls=[], import_table={},
                                status=COVERAGE_SKIPPED,
                                status_reason=f"oversized (> {_MAX_PARSE_BYTES} bytes)")
    except OSError:
        pass  # the extractor below will record the real read failure
    if ext == ".py":
        return _extract_file(file_path, repo_root)
    try:
        from . import ts_extract  # noqa: PLC0415
        relpath = str(file_path.relative_to(repo_root))
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError) as exc:
        return _FileExtract(defs=[], calls=[], import_table={},
                            status=COVERAGE_SKIPPED,
                            status_reason=f"unreadable: {type(exc).__name__}")
    if ext == ".vue":
        return ts_extract.extract_vue(source, relpath, repo_files)
    lang = ts_extract._LANG_FOR_EXT.get(ext)
    if lang is None:
        return _FileExtract(defs=[], calls=[], import_table={},
                            status=COVERAGE_SKIPPED,
                            status_reason=f"no extractor for '{ext}'")
    return ts_extract.extract_ts_like(source, relpath, lang, repo_files)


# ── Cross-session cache warming ──────────────────────────────────────────────
#
# Every Code Studio session is a FRESH CLONE, so its `.mc_codegraph_cache.json`
# starts empty and the first `build_graph` pays a full cold parse of the repo.
# Measured 2026-08-02: only 30 of 67 live session workspaces had a warm cache, and
# `code_graph` averaged 6.4s per call across a run.
#
# Sharing the cache between sessions is SAFE, and specifically because of how this
# cache is keyed: entries are looked up by repo-relative path and accepted only
# when the stored sha256 of the file's bytes matches the file on disk. A clone with
# different mtimes, a different absolute path, or genuinely different content
# simply MISSES and is re-extracted. So a warmed cache can make a build faster; it
# cannot make it wrong.
#
# Fail-soft everywhere: a missing, unreadable or corrupt shared cache just means a
# cold build, exactly as today.

_SHARED_CACHE_ROOT = Path(os.getenv("MC_CODEGRAPH_CACHE_DIR", "/workspace/.codegraph-cache"))


def repo_cache_key(repo_path: str) -> str | None:
    """A key that identifies the REPO, stable across sessions. None if unknowable.

    The obvious choice — the workspace path — is WRONG: every Code Studio session
    is a fresh directory, so keying on it gives every session its own "shared"
    cache and the warming is silently inert. (I wrote that version first.)

    The git remote is stable, but the URL embeds a per-session credential
    (`http://git-service:<token>@host/path`), so the userinfo MUST be stripped or
    the key changes every session AND a token gets written into a filename.
    """
    try:
        from kernel_git_view import legacy_git_query
        url = (legacy_git_query(Path(repo_path), ["config", "--get", "remote.origin.url"]) or "").strip()
        if not url:
            return None
        parsed = urlsplit(url)
        if parsed.scheme:
            # Drop userinfo; keep host + path.
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            return f"{parsed.scheme}://{host}{port}{parsed.path}"
        # scp-style (git@host:path) — strip anything before '@'.
        return url.split("@", 1)[-1]
    except Exception as exc:  # noqa: BLE001
        logger.debug("code_graph: repo key unavailable (%s)", exc)
        return None


def _shared_cache_path(repo_key: str) -> Path:
    safe = hashlib.sha256(repo_key.encode("utf-8")).hexdigest()[:32]
    return _SHARED_CACHE_ROOT / f"{safe}.json"


def _shared_lsp_cache_path(repo_key: str) -> Path:
    """Sidecar path for the type-resolution cache (same key, own file).

    Kept SEPARATE from the extraction cache rather than merged into it: the two
    have independent version numbers and independent failure modes, and a fresh
    clone that inherits one but not the other must still be correct (both are
    content-hash keyed, so a miss is just a recompute).
    """
    safe = hashlib.sha256(repo_key.encode("utf-8")).hexdigest()[:32]
    return _SHARED_CACHE_ROOT / f"{safe}.lsp.json"


def _copy_shared_sidecar(src: Path, dest: Path) -> None:
    """Best-effort JSON file copy for the LSP sidecar. Never raises, never clobbers."""
    try:
        if dest.exists() or not src.is_file():
            return
        data = src.read_text(encoding="utf-8")
        json.loads(data)  # refuse to seed a corrupt file
        _atomic_write_text(dest, data)
    except Exception as exc:  # noqa: BLE001 — an optimisation, never required
        logger.debug("code_graph: lsp cache sidecar copy skipped (%s)", exc)


def warm_cache_from_shared(repo_path: str, repo_key: str) -> bool:
    """Seed a fresh clone's cache from the shared per-repo copy. True if seeded."""
    try:
        dest = repo_cache_file(repo_path)
        if dest.exists():
            return False  # already warm — never clobber a live cache
        src = _shared_cache_path(repo_key)
        if not src.is_file():
            return False
        data = src.read_text(encoding="utf-8")
        json.loads(data)  # refuse to seed a corrupt file
        _atomic_write_text(dest, data)
        # Carry the type-resolution cache along on the same trip — a fresh clone
        # otherwise re-pays the whole LSP pass, which is the expensive half.
        from . import lsp_resolve
        _copy_shared_sidecar(
            _shared_lsp_cache_path(repo_key), repo_cache_file(repo_path, lsp_resolve.CACHE_FILENAME))
        logger.info("code_graph: warmed cache for %s from shared copy", repo_key)
        return True
    except Exception as exc:  # noqa: BLE001 — warming is an optimisation, never required
        logger.debug("code_graph: cache warm skipped (%s)", exc)
        return False


def publish_cache_to_shared(repo_path: str, repo_key: str) -> bool:
    """Publish this session's cache so the NEXT session starts warm."""
    try:
        src = repo_cache_file(repo_path)
        if not src.is_file():
            return False
        _SHARED_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        dest = _shared_cache_path(repo_key)
        _atomic_write_text(dest, src.read_text(encoding="utf-8"))
        from . import lsp_resolve
        lsp_src = repo_cache_file(repo_path, lsp_resolve.CACHE_FILENAME)
        lsp_dest = _shared_lsp_cache_path(repo_key)
        try:
            if lsp_src.is_file():
                _atomic_write_text(lsp_dest, lsp_src.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("code_graph: lsp cache publish skipped (%s)", exc)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("code_graph: cache publish skipped (%s)", exc)
        return False


# ── In-process build memo ────────────────────────────────────────────────────
# The disk cache makes a COLD build cheap; it does nothing about the fact that
# every tool call re-reads, re-hashes and re-finalizes the whole tree. In the
# 2026-08 A/B eval that per-call rebuild was the single biggest contributor to
# code_graph being 8x slower than grep across a 12-call budget.
#
# So: keep the last few built graphs in-process, keyed by repo root, and reuse
# one only when a CHEAP fingerprint of the tree is unchanged. The fingerprint
# stats (never reads) every candidate source file — size + mtime_ns + the file
# set — so an edit, an add and a delete all invalidate, and no stale graph can
# outlive the content it describes.
_MEMO_MAX = 4
_memo: "OrderedDict[str, tuple[str, CodeGraph]]" = OrderedDict()
_memo_lock = threading.Lock()
# One lock per repo so two concurrent callers never build the same tree twice,
# while different repos still build in parallel (a build can take seconds and
# must not be serialized behind an unrelated one).
_build_locks: dict[str, threading.Lock] = {}


def _tree_fingerprint(root: Path) -> str:
    """Cheap stat-only signature of a repo's source tree. '' when unknowable."""
    try:
        source_files, _excluded = _scan_source_files(root)
    except Exception as exc:  # never let fingerprinting break a build
        logger.debug("call_graph: fingerprint scan failed for %s: %s", root, exc)
        return ""
    parts: list[str] = []
    for f in sorted(source_files):
        try:
            st = f.stat()
        except OSError:
            # A file that vanished mid-scan still perturbs the fingerprint.
            parts.append(f"{f}:gone")
            continue
        parts.append(f"{f}:{st.st_size}:{st.st_mtime_ns}")
    if not parts:
        return ""
    return _hash_bytes("\n".join(parts).encode())


def _memo_hit(graph: CodeGraph) -> CodeGraph:
    """Hand back a memoized graph, retiring its one-shot staleness signal.

    `rebuilt_files` means "the PERSISTED index lagged these files when this
    graph was built" — that build refreshed and re-saved the index, so it is
    true exactly once. Carrying it across memo hits would make explore's
    staleness banner fire forever for a file nothing has touched since, which is
    a permanently-on warning, i.e. no warning at all (project rule #189).
    """
    graph.rebuilt_files = set()
    return graph


def _memo_lock_for(key: str) -> threading.Lock:
    with _memo_lock:
        lock = _build_locks.get(key)
        if lock is None:
            lock = _build_locks[key] = threading.Lock()
        return lock


def clear_build_memo() -> None:
    """Drop every memoized graph (tests, and any caller that must force a rebuild)."""
    with _memo_lock:
        _memo.clear()


def build_graph(
    repo_path: str, *, use_cache: bool = True, use_memo: bool = False
) -> CodeGraph:
    """Build the code graph for a repo. Cache-aware; never raises.

    On a warm build only files whose content hash changed are re-parsed — the
    chiasmus 60× trick. The cache lives at <repo>/.mc_codegraph_cache.json.

    ``use_memo=True`` additionally reuses the in-process graph object when the
    tree's stat fingerprint is unchanged, so repeated tool calls inside ONE task
    do not re-index at all. Off by default: a caller that mutates files and
    rebuilds in the same process (tests, the impact/diff paths) must always see
    a genuine rebuild.
    """
    if not use_memo:
        return _build_graph_uncached(repo_path, use_cache=use_cache)

    root = Path(repo_path)
    key = str(root.resolve()) if root.exists() else str(root)
    fingerprint = _tree_fingerprint(root)
    if not fingerprint:  # unknowable tree — never memoize a guess
        return _build_graph_uncached(repo_path, use_cache=use_cache)

    with _memo_lock:
        hit = _memo.get(key)
        if hit is not None and hit[0] == fingerprint:
            _memo.move_to_end(key)
            return _memo_hit(hit[1])

    with _memo_lock_for(key):
        # Re-check under the per-repo lock: a concurrent caller may have built
        # the very graph we were about to duplicate.
        with _memo_lock:
            hit = _memo.get(key)
            if hit is not None and hit[0] == fingerprint:
                _memo.move_to_end(key)
                return _memo_hit(hit[1])
        graph = _build_graph_uncached(repo_path, use_cache=use_cache)
        with _memo_lock:
            _memo[key] = (fingerprint, graph)
            _memo.move_to_end(key)
            while len(_memo) > _MEMO_MAX:
                _memo.popitem(last=False)
        return graph


def _build_graph_uncached(repo_path: str, *, use_cache: bool = True) -> CodeGraph:
    """The real build (no in-process memo). See build_graph."""
    graph = CodeGraph()
    root = Path(repo_path)
    if not root.exists():
        logger.warning("call_graph: repo path does not exist: %s", repo_path)
        return graph

    cache_path = repo_cache_file(root)
    cache = _load_cache(cache_path) if use_cache else {}
    new_cache: dict = {}
    file_import_tables: dict[str, dict[str, str]] = {}

    # Precompute the repo-wide source relpath set so non-Python extractors can
    # resolve local relative imports (./x → the concrete file). Python's import
    # resolution is filesystem-based and does not need this.
    source_files, excluded = _scan_source_files(root)
    repo_files: set[str] = {str(f.relative_to(root)) for f in source_files}
    for rel, reason in excluded:
        graph.coverage[rel] = FileCoverage(
            file=rel, status=COVERAGE_EXCLUDED, reason=reason)

    for src_file in source_files:
        relpath = str(src_file.relative_to(root))
        if relpath == _CACHE_FILENAME:
            continue
        try:
            raw = src_file.read_bytes()
        except OSError as exc:
            graph.coverage[relpath] = FileCoverage(
                file=relpath, status=COVERAGE_SKIPPED,
                reason=f"unreadable: {type(exc).__name__}")
            continue
        digest = _hash_bytes(raw)

        entry = cache.get(relpath)
        extract = _cached_extract(entry, digest)
        if extract is None:
            extract = _extract_any_file(src_file, root, repo_files)
            # Any invalid persisted row is a miss; rebuilding it restores the
            # complete, current shape without making cache damage fatal.
            if entry is not None:
                graph.rebuilt_files.add(relpath)
        new_cache[relpath] = {
            "hash": digest,
            "extractor_version": _EXTRACTOR_VERSION,
            "extract": extract.to_dict(),
        }
        graph.file_hashes[relpath] = digest

        graph.coverage[relpath] = FileCoverage(
            file=relpath, status=extract.status, reason=extract.status_reason,
            symbols=len(extract.defs),
        )

        file_import_tables[relpath] = extract.import_table
        for d in extract.defs:
            graph._index_def(d)
        for v in extract.variables:
            graph._index_variable(v)
        graph.calls.extend(extract.calls)

    # The type-resolution layer keeps its OWN content-hash cache (a separate
    # file, its own version) so a warm build re-asks nothing — and so a change
    # to either cache's shape can never invalidate the other's.
    from . import lsp_resolve  # local import: optional dep, keeps import cost off cold paths
    lsp_cache = lsp_resolve.load_cache(root) if use_cache else {}
    new_lsp_cache = graph._finalize(
        file_import_tables,
        repo_root=root,
        file_hashes=graph.file_hashes,
        lsp_cache=lsp_cache,
    )

    if use_cache:
        _save_cache(cache_path, new_cache)
        if new_lsp_cache:
            lsp_resolve.save_cache(root, new_lsp_cache)
    logger.debug(
        "call_graph: built — %d symbols, %d call sites", len(graph.defs), len(graph.calls)
    )
    return graph


# ── Snapshot + impact diff ───────────────────────────────────────────────────

@dataclass
class GraphSnapshot:
    """Serializable baseline: symbol id -> signature dict + short name."""
    signatures: dict[str, dict]
    short_names: dict[str, str]

    def to_dict(self) -> dict:
        return {"signatures": self.signatures, "short_names": self.short_names}

    @classmethod
    def from_dict(cls, d: dict) -> "GraphSnapshot":
        return cls(
            signatures=dict(d.get("signatures", {})),
            short_names=dict(d.get("short_names", {})),
        )


def snapshot(graph: CodeGraph) -> GraphSnapshot:
    return GraphSnapshot(
        signatures={s: d.signature.to_dict() for s, d in graph.defs.items()},
        short_names={s: d.short_name for s, d in graph.defs.items()},
    )


def snapshot_from_sources(sources: dict[str, str]) -> GraphSnapshot:
    """Build a partial baseline snapshot from {relpath: source_text}.

    Only the given files' symbols are captured — intended for a PR baseline where
    `sources` is the pre-change content of the CHANGED files (via `git show
    base:file`). Pair with diff(..., restrict_files=set(sources)) so the impact
    report reasons only about changed files.
    """
    signatures: dict[str, dict] = {}
    short_names: dict[str, str] = {}
    for relpath, src in sources.items():
        for d in _extract_source(src, relpath):
            signatures[d.symbol] = d.signature.to_dict()
            short_names[d.symbol] = d.short_name
    return GraphSnapshot(signatures=signatures, short_names=short_names)


@dataclass
class ImpactFinding:
    symbol: str
    change: str            # "removed" | "signature_changed" | "added"
    severity: str          # "blocker" | "warning" | "info"
    transitive_callers: list[str]
    detail: str


@dataclass
class ImpactReport:
    findings: list[ImpactFinding]

    @property
    def blockers(self) -> list[ImpactFinding]:
        return [f for f in self.findings if f.severity == "blocker"]

    def has_blockers(self) -> bool:
        return any(f.severity == "blocker" for f in self.findings)

    def to_dict(self) -> dict:
        return {"findings": [f.__dict__ for f in self.findings]}


def _arg_mismatch(old: Signature, new: Signature) -> Optional[str]:
    """Detect a change that could break existing positional callers. None = compatible."""
    # More required positional args now → old call sites under-supply.
    if new.min_positional > old.min_positional:
        return (f"now requires {new.min_positional} positional args "
                f"(was {old.min_positional})")
    # Positional capacity shrank (removed a param, no *args) → old sites over-supply.
    old_max = old.max_positional
    new_max = new.max_positional
    if old_max is not None and new_max is not None and new_max < old_max:
        return f"accepts at most {new_max} positional args (was {old_max})"
    # A previously-optional kwonly became required.
    new_required_kw = set(new.kwonly) - set(new.kwonly_defaults)
    old_required_kw = set(old.kwonly) - set(old.kwonly_defaults)
    added_required_kw = new_required_kw - old_required_kw
    if added_required_kw:
        return f"new required keyword-only args: {sorted(added_required_kw)}"
    return None


def _symbol_file(sym: str) -> str:
    return sym.split("::", 1)[0]


def _looks_like_test_file(relpath: str) -> bool:
    norm = relpath.replace("\\", "/")
    parts = norm.split("/")
    base = parts[-1]
    return (
        base.startswith("test_") or base.endswith("_test.py")
        or "tests" in parts or "test" in parts
        or base == "conftest.py"
    )


@dataclass
class TestImpact:
    """Which tests to run for a changed set, and which changes lack any test."""
    selected_tests: list[str]       # test symbol ids that transitively reach a change
    coverage_gaps: list[str]        # changed source symbols no test reaches

    @property
    def selected_test_files(self) -> list[str]:
        return sorted({s.split("::", 1)[0] for s in self.selected_tests})


def test_impact(graph: CodeGraph, changed_files: Iterable[str]) -> TestImpact:
    """Map a changed file set to the tests that exercise it + the coverage gaps.

    selected_tests  — test_* functions whose transitive callees include a symbol
                      defined in the changed files (run only these for a fast loop).
    coverage_gaps   — non-trivial source symbols in the changed files that NO test
                      transitively reaches (the "backend isn't done without tests"
                      signal). Entrypoints (routes/main/dunders) and test/conftest
                      files are excluded to avoid false gaps.
    """
    changed = set(changed_files)
    selected: set[str] = set()
    gaps: list[str] = []
    for sym, d in graph.defs.items():
        if d.file not in changed or d.kind == "class":
            continue
        reaching = graph.tests_reaching(sym)
        selected.update(r.symbol for r in reaching)
        if (not reaching and not d.is_entrypoint
                and d.kind in ("function", "method")
                and not _looks_like_test_file(d.file)):
            gaps.append(sym)
    return TestImpact(selected_tests=sorted(selected), coverage_gaps=sorted(gaps))


def diff(
    baseline: GraphSnapshot,
    current: CodeGraph,
    *,
    restrict_files: Optional[set[str]] = None,
) -> ImpactReport:
    """Compare a baseline snapshot to the current graph → impact findings.

    When restrict_files is given, only symbols DEFINED in those files are
    considered (a PR/edit scope) — this keeps the 'added' set from ballooning
    to every unrelated symbol in the repo when the baseline is partial.

    Blockers (deterministic, not LLM-judged):
      * a symbol removed while a DANGLING call to its name survives (truly broken)
      * a signature change that existing positional/keyword callers can't satisfy
    A removed symbol whose callers now rebind to a same-named def elsewhere is a
    WARNING (possible silent behaviour change), not a hard blocker.
    """
    findings: list[ImpactFinding] = []

    def _in_scope(sym: str) -> bool:
        return restrict_files is None or _symbol_file(sym) in restrict_files

    cur_syms = {s for s in current.defs if _in_scope(s)}
    base_syms = {s for s in baseline.signatures if _in_scope(s)}

    # Removed symbols
    for sym in sorted(base_syms - cur_syms):
        short = baseline.short_names.get(sym, sym.split("::")[-1].split(".")[-1])
        # DANGLING callers — a call to `short` that now resolves to nothing (broken).
        dangling = sorted({c.symbol for c in
                           current.find_callers(f"unresolved:{short}", transitive=True)})
        # REBOUND callers — a call to `short` now bound to a same-named def elsewhere.
        rebound = sorted({c.symbol for c in
                          current.find_callers(short, transitive=True)} - set(dangling))
        if dangling:
            sev = "blocker"
            detail = (f"symbol removed; {len(dangling)} caller(s) now reference an "
                      f"undefined '{short}' — restore it, update the callers, or confirm intentional")
            callers = dangling
        elif rebound:
            sev = "warning"
            detail = (f"symbol removed; {len(rebound)} caller(s) of '{short}' now bind to a "
                      f"same-named definition elsewhere — verify the behaviour is intended")
            callers = rebound
        else:
            sev = "info"
            detail = "symbol removed; no surviving callers found"
            callers = []
        findings.append(ImpactFinding(
            symbol=sym, change="removed", severity=sev,
            transitive_callers=callers, detail=detail,
        ))

    # Signature changes
    for sym in sorted(base_syms & cur_syms):
        old = Signature.from_dict(baseline.signatures[sym])
        new = current.defs[sym].signature
        if old == new:
            continue
        callers = current.find_callers(sym, transitive=True)
        caller_ids = sorted({c.symbol for c in callers})
        mismatch = _arg_mismatch(old, new)
        if mismatch and caller_ids:
            sev, detail = "blocker", f"signature change {mismatch}; review call sites"
        elif mismatch:
            sev, detail = "warning", f"signature change {mismatch}; no callers detected"
        else:
            sev, detail = "info", "signature changed (backward-compatible)"
        findings.append(ImpactFinding(
            symbol=sym, change="signature_changed", severity=sev,
            transitive_callers=caller_ids, detail=detail,
        ))

    # Added symbols (info only)
    for sym in sorted(cur_syms - base_syms):
        findings.append(ImpactFinding(
            symbol=sym, change="added", severity="info",
            transitive_callers=[], detail="new symbol",
        ))

    return ImpactReport(findings=findings)


# ── Diff-seeded blast radius ─────────────────────────────────────────────────
#
# `impact_radius` answers "who breaks if I change THIS symbol". A working agent
# has already changed a SET of things, and asking symbol-by-symbol both costs N
# traversals and loses the shape of the answer (which module absorbed the change,
# what is one hop away vs three). This section closes that gap: git diff →
# changed line ranges → the SymbolDefs those lines fall inside (the seed set) →
# ONE multi-source inbound BFS → hop-ordered impact + tests + a module rollup.

# Candidate default branches, in the order a real repo is most likely to use.
_DEFAULT_BRANCH_CANDIDATES = (
    "origin/HEAD", "origin/main", "origin/master", "main", "master",
)
_GIT_TIMEOUT_SECONDS = 20
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _git(repo_root: Path, *args: str) -> Optional[str]:
    """Run a read-only git command in `repo_root`. None on ANY failure.

    Never raises and never mutates: every call site here is `diff`/`rev-parse`/
    `merge-base`. A repo that is not a git checkout simply yields None and the
    blast-radius action degrades to an explained empty result.
    """
    from kernel_git_view import legacy_git_query
    return legacy_git_query(repo_root, list(args), timeout=_GIT_TIMEOUT_SECONDS)


def resolve_base_ref(repo_root: Path, base_ref: Optional[str] = None) -> tuple[Optional[str], str]:
    """(resolved base ref, how it was chosen). (None, reason) when git is unusable.

    Explicit `base_ref` wins (verified to exist). Otherwise: the merge-base with
    the first default branch that resolves, then HEAD~1, then the empty tree (so
    a single-commit repo still diffs against something real).
    """
    if _git(repo_root, "rev-parse", "--git-dir") is None:
        return None, "not a git repository"

    if base_ref:
        out = _git(repo_root, "rev-parse", "--verify", f"{base_ref}^{{commit}}")
        if out is None:
            return None, f"base_ref {base_ref!r} does not resolve to a commit"
        return out.strip(), f"explicit base_ref {base_ref!r}"

    for cand in _DEFAULT_BRANCH_CANDIDATES:
        if _git(repo_root, "rev-parse", "--verify", f"{cand}^{{commit}}") is None:
            continue
        mb = _git(repo_root, "merge-base", "HEAD", cand)
        if mb and mb.strip():
            return mb.strip(), f"merge-base with {cand}"

    out = _git(repo_root, "rev-parse", "--verify", "HEAD~1^{commit}")
    if out and out.strip():
        return out.strip(), "HEAD~1 (no default branch found)"

    # A repo with a single commit has no HEAD~1 — diff against the empty tree so
    # the very first change set is still analysable instead of silently empty.
    empty = _git(repo_root, "hash-object", "-t", "tree", "/dev/null")
    if empty and empty.strip():
        return empty.strip(), "empty tree (repository has a single commit)"
    return None, "could not resolve any base commit"


def parse_diff_ranges(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse a unified diff into {relpath: [(start_line, end_line), …]} (NEW side).

    Only the post-image ranges matter: they are the lines that now exist and
    therefore the lines that map onto current SymbolDefs. A pure deletion hunk
    (`+N,0`) is recorded as a zero-width marker at N so the surrounding symbol
    is still seeded — deleting code inside a function changes that function.
    """
    ranges: dict[str, list[tuple[int, int]]] = {}
    current: Optional[str] = None
    for line in (diff_text or "").splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                current = None
            else:
                if path.startswith("b/"):
                    path = path[2:]
                current = path
                ranges.setdefault(current, [])
            continue
        if current is None or not line.startswith("@@"):
            continue
        m = _HUNK_RE.match(line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        end = start + max(0, count - 1)
        ranges[current].append((max(1, start), max(1, end)))
    return {k: v for k, v in ranges.items() if v}


def changed_ranges(
    repo_root: Path, base: str,
) -> dict[str, list[tuple[int, int]]]:
    """Changed NEW-side line ranges between `base` and the WORKING TREE.

    Working tree — not HEAD — deliberately: an agent asking for its own blast
    radius has usually not committed yet, and an answer that ignores the edit in
    front of it is worse than useless. `-U0` keeps hunks tight so the seed set is
    the changed symbols, not their neighbours.
    """
    out = _git(repo_root, "diff", "-U0", "--no-color", base)
    if out is None:
        return {}
    ranges = parse_diff_ranges(out)
    # Untracked-but-added files never appear in `git diff`; include them whole so
    # a brand-new module's symbols are seeds too.
    untracked = _git(repo_root, "ls-files", "--others", "--exclude-standard")
    for rel in (untracked or "").splitlines():
        rel = rel.strip()
        if not rel or rel in ranges:
            continue
        if Path(rel).suffix.lower() not in _GRAPH_EXTENSIONS:
            continue
        try:
            n = len((repo_root / rel).read_text(
                encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
        if n:
            ranges[rel] = [(1, n)]
    return ranges


def seeds_for_ranges(
    graph: CodeGraph, ranges: dict[str, list[tuple[int, int]]],
) -> list[str]:
    """SymbolDef ids whose [lineno, end_lineno] span overlaps a changed range.

    A changed line inside a method seeds the METHOD (the tightest enclosing
    def), never its whole class — the class would drag in every sibling method's
    callers, which is exactly the over-broad answer impact_radius rule (a) exists
    to prevent. Classes are seeded only when the change lands on the class's own
    lines outside any method (e.g. a class-level attribute).
    """
    seeds: set[str] = set()
    by_file: dict[str, list[SymbolDef]] = {}
    for d in graph.defs.values():
        if d.file in ranges:
            by_file.setdefault(d.file, []).append(d)

    for relpath, spans in ranges.items():
        defs = by_file.get(relpath, [])
        for start, end in spans:
            overlapping = [
                d for d in defs
                if d.lineno <= end and max(d.lineno, d.end_lineno) >= start
            ]
            if not overlapping:
                continue
            # Tightest enclosing def wins; ties keep every equally-tight match.
            narrowest = min(
                max(0, max(d.lineno, d.end_lineno) - d.lineno) for d in overlapping
            )
            for d in overlapping:
                if max(0, max(d.lineno, d.end_lineno) - d.lineno) == narrowest:
                    seeds.add(d.symbol)
    return sorted(seeds)


@dataclass
class BlastRadius:
    """Diff-seeded impact answer. Every total here is EXACT, pre-truncation."""
    base_ref: Optional[str]
    base_source: str                       # how base_ref was chosen (or why not)
    changed_files: list[str]
    seeds: list[SymbolRef]
    impacted: list[tuple[SymbolRef, int]]  # (symbol, hop) — hop-ordered
    impacted_tests: list[str]              # test symbol ids that reach the change
    coverage_gaps: list[str]               # changed symbols no test reaches
    module_rollup: dict[str, int]          # module path -> exact impacted count
    depth: int
    note: str = ""

    @property
    def impacted_total(self) -> int:
        return len(self.impacted)


def _module_of(relpath: str) -> str:
    """The rollup bucket for a file — its directory, or '<root>' at the top level."""
    norm = relpath.replace("\\", "/")
    return norm.rsplit("/", 1)[0] if "/" in norm else "<root>"


def blast_radius(
    graph: CodeGraph,
    repo_root: Path,
    *,
    base_ref: Optional[str] = None,
    diff_text: Optional[str] = None,
    depth: int = 2,
) -> BlastRadius:
    """Diff → seeds → ONE inbound BFS → hop-ordered impact + tests + rollup.

    `diff_text` short-circuits git entirely (an explicit diff the caller already
    has). Otherwise the base ref is resolved and the working tree diffed against
    it. Never raises: a non-git directory yields an empty, EXPLAINED result.
    """
    depth = max(1, min(int(depth or 2), 10))

    if diff_text is not None:
        ranges = parse_diff_ranges(diff_text)
        base, source = None, "explicit diff supplied by caller"
        note = "" if ranges else "the supplied diff contained no parsable hunks"
    else:
        base, source = resolve_base_ref(repo_root, base_ref)
        if base is None:
            return BlastRadius(
                base_ref=None, base_source=source, changed_files=[], seeds=[],
                impacted=[], impacted_tests=[], coverage_gaps=[], module_rollup={},
                depth=depth, note=f"no diff could be computed: {source}",
            )
        ranges = changed_ranges(repo_root, base)
        note = "" if ranges else "no changes between the base ref and the working tree"

    seed_ids = seeds_for_ranges(graph, ranges)
    changed_files = sorted(ranges)
    if ranges and not seed_ids:
        note = note or (
            "changed lines fall outside any indexed symbol (imports, constants, "
            "config, or a file the graph does not extract) — see the coverage action"
        )

    impacted = graph.multi_source_impact(seed_ids, depth=depth)
    ti = test_impact(graph, set(changed_files))

    rollup: dict[str, int] = {}
    for ref, _hop in impacted:
        key = _module_of(ref.file)
        rollup[key] = rollup.get(key, 0) + 1

    return BlastRadius(
        base_ref=base, base_source=source, changed_files=changed_files,
        seeds=graph._as_refs(seed_ids), impacted=impacted,
        impacted_tests=ti.selected_tests, coverage_gaps=ti.coverage_gaps,
        module_rollup=dict(sorted(rollup.items(), key=lambda kv: (-kv[1], kv[0]))),
        depth=depth, note=note,
    )
