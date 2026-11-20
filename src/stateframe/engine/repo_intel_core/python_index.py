"""Content-addressed Python facts for the graph, symbol registry and change ledger."""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

try:
    import sqlite3
except ImportError:  # Python can be built without the optional SQLite extension.
    sqlite3 = None

from . import call_graph as graph

if __package__ and __package__.endswith("engine.repo_intel_core"):
    from .. import kernel_parse_cache as cache
else:
    import kernel_parse_cache as cache

parse = cache.parse
logger = logging.getLogger(__name__)
# Bump when the visitor, graph output or row semantics change. Python ASTs and
# identifier rules can differ between interpreter versions.
_VERSION = f"python-facts-3-{sys.implementation.cache_tag}"
_MAX_ENTRY = 8 << 20
IDENTIFIER_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
IDENTIFIER = re.compile(rf"\b({IDENTIFIER_PATTERN})\b")


class TraversalDepthError(RecursionError):
    """Let each consumer apply its own historical handling for very deep trees."""


def import_record(node):
    level = node.level if isinstance(node, ast.ImportFrom) else None
    return getattr(node, "module", None), level, [(a.name, a.asname) for a in node.names]


def registry_signature(fn) -> str:
    # KEEP THE / AND * MARKERS. the audit showed `def f(a, /, *, key=1)` rendering as
    # `def f(a, key)` — which tells the agent it may pass `key` positionally when Python
    # will reject that. A signature that is subtly wrong is worse than none, because the
    # agent trusts it and writes a call that cannot work.
    a = fn.args
    parts = [x.arg for x in a.posonlyargs]
    if a.posonlyargs:
        parts.append("/")
    parts += [x.arg for x in a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    parts += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {fn.name}({', '.join(parts)})"

class _Visitor(graph._FileVisitor):
    def __init__(self, relative):
        super().__init__(relative)
        self.depth = 0
        self.registry = []
        # The legacy import and ledger scans are breadth-first. A depth bucket
        # preserves their stable ordering while the scoped graph visits depth-first.
        self.imports = {}
        self.ledger = {}
        self.metrics = None
        self.pending_metrics = None
        self.loop = self.guard = 0

    def _properties(self, node, name):
        self.pending_metrics = graph._FunctionMetrics(node, name)
        return self.pending_metrics.properties

    def generic_visit(self, node):
        previous = self.metrics, self.loop, self.guard
        scope = isinstance(node, graph._PY_NESTED_SCOPES)
        if scope:
            self.metrics, self.pending_metrics = self.pending_metrics, None
            self.loop = self.guard = 0
        elif self.metrics is not None:
            self.loop, self.guard = self.metrics.observe(node, self.loop, self.guard)
        self.depth += 1
        super().generic_visit(node)
        self.depth -= 1
        if scope and self.metrics is not None:
            self.metrics.finish()
        self.metrics, self.loop, self.guard = previous

    def _definition(self, node, kind, signature):
        cls = self._class_stack[-1] if self._class_stack and kind != "class" else None
        self.registry.append((node.name, node.lineno, cls, kind, signature))
        self.ledger.setdefault(self.depth, []).append(
            (node.name, node.lineno, type(node).__name__.removesuffix("Def").lower()))

    def visit_ClassDef(self, node):
        self._definition(node, "class", f"class {node.name}")
        super().visit_ClassDef(node)

    def visit_FunctionDef(self, node):
        self._definition(node, "method" if self._class_stack else "function",
                         registry_signature(node))
        super()._visit_func(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        kind = "alias" if isinstance(node.value, (ast.Name, ast.Attribute)) else "assignment"
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.ledger.setdefault(self.depth, []).append((target.id, node.lineno, kind))
                if target.id.isupper():
                    cls = self._class_stack[-1] if self._class_stack else None
                    self.registry.append((target.id, node.lineno, cls, "constant", target.id))
        super().visit_Assign(node)

    def visit_Import(self, node):
        self.imports.setdefault(self.depth, []).append(import_record(node))
        for alias in node.names:
            if isinstance(node, ast.ImportFrom):
                if alias.name == "*":
                    continue
                symbol = alias.asname or alias.name.rsplit(".", 1)[-1]
            else:
                symbol = alias.asname or alias.name.split(".", 1)[0]
            self.ledger.setdefault(self.depth, []).append((symbol, node.lineno, "import"))

    visit_ImportFrom = visit_Import


def _extract(source: str, relative: str) -> dict:
    tree = parse(source, filename=relative)
    visitor = _Visitor(relative)
    try:
        visitor.visit(tree)
    except RecursionError as exc:
        raise TraversalDepthError from exc
    # Store only line numbers: the ledger obtains display text from current
    # source. Each (identifier, line) is unique, already in ascending order.
    text_sites = {}
    for number, line in enumerate(source.splitlines(), 1):
        for symbol in dict.fromkeys(IDENTIFIER.findall(line)):
            text_sites.setdefault(symbol, []).append(number)
    extract = graph._FileExtract(visitor.defs, visitor.calls, {}, visitor.variables)
    return {
        "graph": (extract.to_dict(), [
            row for depth in sorted(visitor.imports) for row in visitor.imports[depth]]),
        "registry": visitor.registry,
        "ledger": ([row for depth in sorted(visitor.ledger) for row in visitor.ledger[depth]],
                   text_sites),
    }


# Read only the requested projection. The graph often outweighs the registry
# and ledger combined. Bind each checksum to both its file identity and column.
_SELECT = {part: f"SELECT {part} FROM files WHERE key=? AND length({part})<=?"
           for part in ("registry", "ledger", "graph")}


def _encode(key, part, facts):
    payload = json.dumps(facts, separators=(",", ":")).encode()
    return hashlib.sha256(key + part.encode() + payload).digest() + payload


def _valid_projection(part: str, rows: object, source: str) -> bool:
    if not isinstance(rows, list):
        return False
    if part == "graph":
        return (len(rows) == 2 and graph._valid_cached_extract(rows[0])
                and graph._valid_import_records(rows[1]))
    if part == "registry":
        return all(isinstance(row, list) and len(row) == 5
                   and isinstance(row[0], str) and type(row[1]) is int
                   and (row[2] is None or isinstance(row[2], str))
                   and isinstance(row[3], str) and isinstance(row[4], str)
                   for row in rows)
    line_count = len(source.splitlines())
    return (len(rows) == 2 and isinstance(rows[0], list) and isinstance(rows[1], dict)
            and all(isinstance(row, list) and len(row) == 3 and isinstance(row[0], str)
                    and type(row[1]) is int and isinstance(row[2], str) for row in rows[0])
            and all(isinstance(lines, list) and all(type(n) is int and 0 < n <= line_count for n in lines)
                    for lines in rows[1].values()))


class _Index:
    def __init__(self, connection: sqlite3.Connection | None):
        self.connection = connection
        self.pending = []
        self.pending_bytes = 0

    def optional_projection(self, source: str, relative: str, part: str):
        # The registry and ledger have wider admission rules than the graph.
        # Keep their legacy read-only analysis for excluded or oversized inputs.
        if (graph._excluded_reason(tuple(relative.split("/"))) is not None
                or len(source) > graph._MAX_PARSE_BYTES
                or (len(source) * 4 > graph._MAX_PARSE_BYTES and
                    len(source.encode("utf-8", "surrogatepass")) > graph._MAX_PARSE_BYTES)):
            return None
        try:
            return self.get(source, relative, part)
        except TraversalDepthError:
            return None

    def get(self, source: str, relative: str, part: str):
        if self.connection is None:
            facts = cache.derived_source(_VERSION, source, relative,
                                         lambda text: _extract(text, relative))
            return facts[part]
        identity = relative.encode("utf-8", "surrogatepass") + b"\0" + cache.source_digest(source).encode()
        key = hashlib.sha256(identity).digest()
        try:
            row = self.connection.execute(_SELECT[part], (key, _MAX_ENTRY)).fetchone()
            if row and isinstance(row[0], bytes):
                checksum, payload = row[0][:32], row[0][32:]
                if hashlib.sha256(key + part.encode() + payload).digest() == checksum:
                    projection = json.loads(payload)
                    if _valid_projection(part, projection, source):
                        return projection
        except (sqlite3.Error, ValueError, RecursionError) as exc:
            logger.debug("Python fact cache read failed: %s", type(exc).__name__)

        facts = _extract(source, relative)
        columns = tuple(_encode(key, name, facts[name]) for name in _SELECT)
        if max(map(len, columns)) <= _MAX_ENTRY:
            self.pending.append((key, *columns))
            self.pending_bytes += sum(map(len, columns))
            if len(self.pending) >= 64 or self.pending_bytes >= 1 << 20:
                self.flush()
        return facts[part]

    def flush(self) -> None:
        if not self.pending:
            return
        try:
            with self.connection:
                self.connection.executemany(
                    "INSERT OR REPLACE INTO files(key,registry,ledger,graph) VALUES(?,?,?,?)",
                    self.pending)
        except sqlite3.Error as exc:
            # A busy or read-only cache must never block an entire scan per file.
            logger.debug("Python fact cache write failed: %s", type(exc).__name__)
            if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL:
                try:
                    with self.connection:
                        self.connection.execute("DELETE FROM files WHERE rowid IN "
                                                "(SELECT rowid FROM files ORDER BY rowid LIMIT 1024)")
                except sqlite3.Error as eviction_error:
                    logger.debug("Python fact cache eviction failed: %s", type(eviction_error).__name__)
        finally:
            self.pending.clear()
            self.pending_bytes = 0


@contextmanager
def open_index(root: Path, *, persistent: bool = True):
    """One short-lived connection per scan. Nothing persists inside the task tree."""
    if sqlite3 is None or not persistent:
        yield _Index(None)
        return
    home = os.environ.get("STATEFRAME_HOME")
    db = None
    try:
        if home:
            directory = Path(home).resolve() / "python-index"
            path = directory / f"{_VERSION}.sqlite"
            if (not path.resolve().is_relative_to(root.resolve())
                    and not directory.is_symlink() and not path.is_symlink()):
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                # Only consume a private cache. Refuse redirected SQLite sidecars.
                if (directory.stat().st_mode & 0o077 or
                        any(item.is_symlink() or (item.exists() and item.stat().st_nlink != 1)
                            for item in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"),
                                         Path(str(path) + "-journal")))):
                    raise OSError("unsafe Python index cache path")
                db = sqlite3.connect(path, timeout=0.02)
                db.execute("PRAGMA trusted_schema=OFF")
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
                db.execute("PRAGMA max_page_count=65536")
                db.execute("CREATE TABLE IF NOT EXISTS files "
                           "(key BLOB PRIMARY KEY, registry BLOB, ledger BLOB, graph BLOB)")
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        logger.debug("Python fact cache unavailable: %s", type(exc).__name__)
        if db is not None:
            db.close()
        db = None
    index = _Index(db)
    try:
        yield index
    finally:
        index.flush()
        if db is not None:
            db.close()
