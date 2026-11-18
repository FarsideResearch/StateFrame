"""THE SYMBOL REGISTRY — ground truth about what actually exists, so a name cannot be invented.

Observed live, 2026-08-30 (probes/react_live.log), a ReAct agent repairing
`create_and_dispatch_agent_task`:

    s07 grep blocking_run_status             -> found an IMPORT of it
    s08 grep "def blocking_run_status"       -> (no matches)
    s09 grep "def blocking_run_status"       -> (no matches)     <- exact repeat
    s11 grep "def task_blocking_run_status"  -> (no matches)     <- invented a second name
    s12 grep -n "task_blocking_run_status"   -> (no matches)

Six of thirty-six steps — 17% of the budget — spent hunting a symbol that does not exist, then
hunting a name it made up. No amount of reasoning fixes that, because the model has no way to
know. The repository does know, and the answer is deterministic.

This is the "compile reasoning into the kernel" move applied to hallucination: the model never
has to REMEMBER what exists, because the kernel can always TELL it, exactly and instantly.

WHAT IT ANSWERS, all deterministically and offline:
    exists(name)            is this a real symbol in this repo?
    where(name)             every definition site: file, line, kind
    signature(name)         the exact parameter list — no invented kwargs
    near(name)              the closest real names, when a symbol was invented
    callers(name)           who calls it (for blast radius)

USED BY THE LOOP so a dead end costs ONE step instead of six: a search for a non-existent
symbol returns "no such symbol in this repository" plus the nearest real candidates, rather
than an empty result the model reads as "look harder".

Deliberately AST-derived rather than grep-derived: a grep for `def foo` misses decorators,
async defs, methods, and assignments, which is precisely how the agent above got a
false negative on a symbol whose import it had already seen.

    python3 symbol_registry.py build <repo_root>     # write the index
    python3 symbol_registry.py ask <repo_root> <name>
"""
from __future__ import annotations

import ast
import difflib
import json
import logging
import pathlib
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, islice

import kernel_parse_cache as _parse_cache
import kernel_workspace_truth as _workspace_truth
from mc_repo_intel_vendor import python_index

_READ_WINDOW = 64
_READ_CHARS = 64 * 1024


def _read_source(path: pathlib.Path, limit: int = -1) -> str | None:
    try:
        with path.open(errors="strict") as stream:
            return stream.read(limit)
    except (UnicodeDecodeError, OSError):
        return None


def _finish_read(path, future):
    if future is None:
        return path, _read_source(path)
    prefix = future.result()
    # Bound queued text to 64 Ki characters per path. Long inputs are read in
    # full, one at a time, on the consuming thread.
    if prefix is not None and len(prefix) > _READ_CHARS:
        return path, _read_source(path)
    return path, prefix


def _sources(root, skip):
    paths = (p for p in root.rglob("*.py") if not any(part in skip for part in p.parts))
    first = tuple(islice(paths, _READ_WINDOW))
    if len(first) < _READ_WINDOW:
        for path in first:
            yield path, _read_source(path)
        return
    # Cold workcopies can spend most of boot waiting on sequential opens.
    # Overlap reads while preserving rglob order and single-threaded analysis.
    pending = deque()
    parallel = True
    with ThreadPoolExecutor(max_workers=16) as pool:
        for path in chain(first, paths):
            future = None
            if parallel:
                try:
                    future = pool.submit(_read_source, path, _READ_CHARS + 1)
                except (RuntimeError, OSError) as exc:
                    logging.getLogger(__name__).debug("Registry prefetch unavailable: %s", type(exc).__name__)
                    parallel = False
            pending.append((path, future))
            if len(pending) == _READ_WINDOW:
                yield _finish_read(*pending.popleft())
        while pending:
            yield _finish_read(*pending.popleft())


class SymbolRegistry:
    """An index of every def/class/assignment a repository actually contains."""

    def __init__(self) -> None:
        # name -> [{file, line, kind, signature, cls}]
        self.defs: dict[str, list[dict]] = {}
        self.files_scanned = 0
        self.files_failed = 0
        self.root: pathlib.Path | None = None

    # ------------------------------------------------------------------ build
    def build(self, root: pathlib.Path, skip=("__pycache__", ".git", "node_modules",
                                               ".venv", "site-packages")) -> "SymbolRegistry":
        self.root = pathlib.Path(root).resolve()
        with python_index.open_index(self.root) as index:
            for p, src in _sources(root, skip):
                if src is None:
                    self.files_failed += 1
                    continue
                rel = p.relative_to(root).as_posix()
                try:
                    rows = index.optional_projection(src, rel, "registry")
                    if rows is None:
                        self._walk(_parse_cache.parse(src, filename=rel), rel, None)
                    else:
                        for name, line, cls, kind, signature in rows:
                            self.defs.setdefault(name, []).append({
                                "file": rel, "line": line, "cls": cls, "kind": kind,
                                "signature": signature,
                            })
                except (SyntaxError, UnicodeDecodeError, OSError):
                    # A file we cannot parse is an UNKNOWN, never a "does not exist". Counted so
                    # the registry can report its own blind spots rather than answering
                    # confidently about a file it never read.
                    self.files_failed += 1
                    continue
                self.files_scanned += 1
        return self

    def _walk(self, node, rel: str, cls: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs.setdefault(child.name, []).append({
                    "file": rel, "line": child.lineno, "cls": cls,
                    "kind": "method" if cls else "function",
                    "signature": self._sig(child),
                })
                self._walk(child, rel, cls)
            elif isinstance(child, ast.ClassDef):
                self.defs.setdefault(child.name, []).append({
                    "file": rel, "line": child.lineno, "cls": None,
                    "kind": "class", "signature": f"class {child.name}",
                })
                self._walk(child, rel, child.name)
            elif isinstance(child, ast.Assign):
                for t in child.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        self.defs.setdefault(t.id, []).append({
                            "file": rel, "line": child.lineno, "cls": cls,
                            "kind": "constant", "signature": t.id,
                        })
            else:
                self._walk(child, rel, cls)

    _sig = staticmethod(python_index.registry_signature)

    # ------------------------------------------------------------------ query
    def exists(self, name: str) -> bool:
        return name in self.defs

    def where(self, name: str) -> list[dict]:
        return self.defs.get(name, [])

    def near(self, name: str, n: int = 5) -> list[str]:
        """Closest REAL names — the answer to an invented symbol."""
        return difflib.get_close_matches(name, self.defs.keys(), n=n, cutoff=0.6)

    def answer(
            self, name: str, *, workspace_root: pathlib.Path | str | None = None,
            revision: str | None = None, journal: object = None) -> str:
        """Render a fresh symbol fact when the live-workspace oracle is enabled."""
        if _workspace_truth.enabled():
            fact = _workspace_truth.symbol_defined(
                workspace_root, name, revision=revision, journal=journal,
            )
            return _workspace_truth.render_fact(fact)
        hits = self.where(name)
        return _workspace_truth.render_legacy_symbol_answer(
            name, hits, self.near(name), len(self.defs),
            self.files_scanned, self.files_failed,
        )

    # ------------------------------------------------------------------ io
    def save(self, path: pathlib.Path) -> None:
        path.write_text(json.dumps({"defs": self.defs, "scanned": self.files_scanned,
                                    "failed": self.files_failed}))

    @classmethod
    def load(cls, path: pathlib.Path) -> "SymbolRegistry":
        d = json.loads(path.read_text())
        r = cls()
        r.defs = d["defs"]
        r.files_scanned = d.get("scanned", 0)
        r.files_failed = d.get("failed", 0)
        return r


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-3])
        return 2
    cmd, root = sys.argv[1], pathlib.Path(sys.argv[2])
    if cmd == "build":
        r = SymbolRegistry().build(root)
        out = root / ".symbol_registry.json"
        r.save(out)
        print(f"indexed {len(r.defs)} symbols from {r.files_scanned} files "
              f"({r.files_failed} unparseable) -> {out}")
        return 0
    if cmd == "ask":
        r = SymbolRegistry().build(root)
        print(r.answer(sys.argv[3], workspace_root=root))
        return 0
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
