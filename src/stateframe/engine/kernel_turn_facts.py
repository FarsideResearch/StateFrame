"""Pinned point and symbol queries for the turn facts port."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from hashlib import sha256
from heapq import merge
from itertools import islice
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from threading import Lock

from kernel_turn_contracts import (
    Artifact, Basis, Context, Corrupt, Dep, Facts, Key, Page, Publication, Ref, Snapshot, Stale,
    Unavailable, Workspace,
)
from kernel_turn_fact_store import FactStore, combine, encode, row_digest
from symbol_registry import SymbolRegistry


_SWITCH = "STATEFRAME_TURN_INDEXED_READS"
_PAGE_CODEC = "turn/fact-page/1"
_BLOB_CODEC = "turn/fact-blob/1"
_MAX_PAGE_BYTES = 256 * 1024
_EMPTY_PAGE_BYTES = len(b"[]")
_MAX_POINT_BYTES = 2_000_000
_MAX_CHANGED_PATHS = 128
_SKIPPED = frozenset({
    "__pycache__", ".git", "node_modules", ".venv", "site-packages",
    "backups", "evaluator", "grader", "graders", "gold", "hidden", "hidden_tests", "private",
})
_FIRST = (b"", -1)


def enabled() -> bool:
    return os.environ.get(_SWITCH) == "on"


def _path(raw: bytes) -> bytes:
    parts = raw.split(b"/")
    if (b"\0" in raw or b"\\" in raw or
            any(part in {b"", b".", b".."} or os.fsdecode(part).casefold() in _SKIPPED
                for part in parts)):
        raise Unavailable("turn fact path is outside scope")
    return raw


def _symbols(path: bytes, data: bytes) -> dict[str, list[dict]]:
    try:
        return SymbolRegistry.definitions_in_source(os.fsdecode(path), data)
    except (SyntaxError, UnicodeError, ValueError, RecursionError) as error:
        raise Unavailable("turn symbol projection is incomplete") from error


def _root_key(snapshot: Snapshot) -> str:
    return sha256(encode([snapshot.root, snapshot.admission])).hexdigest()


def _ref_row(ref: Ref, **fields: object) -> dict[str, object]:
    return {**fields, "blob": {"codec": ref.codec, "sha256": ref.sha256, "size": ref.size}}


class _Facts:
    def __init__(self, ctx: Context, workspace: Workspace, publication: Publication):
        namespace = sha256(encode([
            ctx.principal, ctx.run, ctx.profile_digest, sys.version, "symbols/1",
        ])).hexdigest()
        self._store = FactStore(ctx.private_dir / ("turn-facts-" + namespace))
        self._workspace = workspace
        self._publication = publication
        self._enabled = enabled()
        self._overlay_lock = Lock()
        self._overlay_key: tuple[str, str] | None = None
        self._overlay: dict[bytes, dict[str, list[dict]]] = {}

    def close(self) -> None:
        self._store.close()

    def build(self, parent: Snapshot, stop: Callable[[], bool]) -> None:
        if not self._enabled or stop():
            return
        with self._workspace.retain(parent.root) as held:
            if _root_key(held) != _root_key(parent):
                raise Stale("admission changed before turn fact build")
            root = _root_key(held)
            with self._store.open() as db:
                if db is not None and FactStore.complete(db, root):
                    return
            with self._store.open(create=True) as db:
                if db is None:
                    raise Unavailable("turn fact database was not created")
                if FactStore.complete(db, root):
                    return
            with TemporaryDirectory(prefix="build-", dir=self._store.directory) as temp:
                tree = Path(temp) / "workspace"
                self._workspace.materialize(held, tree)
                if stop():
                    return
                self._build_index(self._store, root, held, tree, stop)

    def _build_index(self, store: FactStore, root: str, held: Snapshot, tree: Path,
                     stop: Callable[[], bool]) -> None:
        with store.open(create=True) as db:
            if db is None:
                raise Unavailable("turn fact database was not created")
            with db:
                db.execute("BEGIN IMMEDIATE")
                if not FactStore.complete(db, root):
                    self._ingest(db, root, held, tree, stop)

    def _ingest(self, db: sqlite3.Connection, root: str, held: Snapshot,
                tree: Path, stop: Callable[[], bool]) -> None:
        def on_error(error: OSError) -> None:
            raise Unavailable("admitted tree cannot be enumerated") from error

        for directory, subdirs, names in os.walk(tree, onerror=on_error):
            if stop():
                db.rollback()
                return
            subdirs[:] = sorted(name for name in subdirs if name.casefold() not in _SKIPPED)
            for name in sorted(names):
                if stop():
                    db.rollback()
                    return
                if not name.endswith(".py"):
                    continue
                file = Path(directory) / name
                if not file.is_file() or file.is_symlink():
                    continue
                path = os.fsencode(file.relative_to(tree).as_posix())
                if b"\\" in path:
                    continue
                found = _symbols(path, self._workspace.read(held, path))
                for symbol, rows in found.items():
                    for position, row in enumerate(rows):
                        FactStore.put(db, root, symbol, path, position, row)
        if stop():
            db.rollback()
            return
        FactStore.finish(db, root)

    def _check_snapshot(self, snapshot: Snapshot, root: str) -> None:
        if snapshot.root != root:
            raise Stale("workspace retention returned a different root")

    def _changes(self, parent: Snapshot, live: Snapshot) -> dict[bytes, dict[str, list[dict]]]:
        key = _root_key(parent), _root_key(live)
        with self._overlay_lock:
            if self._overlay_key == key:
                return self._overlay
        changed: dict[bytes, dict[str, list[dict]]] = {}
        for number, path in enumerate(self._workspace.changes(parent, live), 1):
            if number > _MAX_CHANGED_PATHS:
                raise Unavailable("turn fact overlay exceeds its foreground limit")
            if not path.endswith(b".py"):
                continue
            try:
                _path(path)
            except Unavailable:
                continue
            try:
                source = self._workspace.read(live, path)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                changed[path] = {}
            else:
                changed[path] = _symbols(path, source)
        with self._overlay_lock:
            self._overlay_key, self._overlay = key, changed
        return changed

    @contextmanager
    def _symbol_query(self, at: Basis) -> Iterator[
            tuple[sqlite3.Connection, str, dict[bytes, dict[str, list[dict]]]]]:
        with self._workspace.retain(at.live) as live:
            self._check_snapshot(live, at.live)
            live_root = _root_key(live)
            with self._store.open() as db:
                if db is not None and FactStore.complete(db, live_root):
                    yield db, live_root, {}
                    return
                if db is not None and at.parent != at.live:
                    with self._workspace.retain(at.parent) as parent:
                        self._check_snapshot(parent, at.parent)
                        parent_root = _root_key(parent)
                        if FactStore.complete(db, parent_root):
                            yield db, parent_root, self._changes(parent, live)
                            return
            if self._enabled:
                raise Unavailable("turn symbol index is incomplete for the retained root")

            # Disabled consumers use the exact same projector over their retained root.
            with TemporaryDirectory(prefix="turn-facts-", dir=self._store.directory.parent) as temp:
                directory = Path(temp)
                tree = directory / "workspace"
                self._workspace.materialize(live, tree)
                with closing(FactStore(directory / "index")) as store:
                    self._build_index(store, live_root, live, tree, lambda: False)
                    with store.open() as db:
                        if db is None or not FactStore.complete(db, live_root):
                            raise Unavailable("turn fact index is incomplete")
                        yield db, live_root, {}

    @staticmethod
    def _membership(db: sqlite3.Connection, root: str, name: str,
                    overlay: dict[bytes, dict[str, list[dict]]],
                    ) -> tuple[str, int, int, list[tuple[bytes, int, bytes]]]:
        count, total = FactStore.membership(db, root, name)
        added = []
        for path, found in overlay.items():
            for digest in FactStore.file_digests(db, root, name, path):
                count -= 1
                total = combine(total, digest, subtract=True)
            for position, row in enumerate(found.get(name, ())):
                payload = encode(row)
                count += 1
                total = combine(total, row_digest(path, position, payload))
                added.append((path, position, payload))
        if count < 0:
            raise Corrupt("turn symbol membership count is invalid")
        return sha256(encode([name, count, total])).hexdigest(), count, total, sorted(added)

    def _stage_page(self, key: Key, revision: str,
                    rows: list[object], next_cursor: bytes | None = None) -> Page:
        ref = self._publication.stage(_PAGE_CODEC, encode(rows))
        return Page(Artifact(ref, (Dep(key, revision),)), next_cursor)

    def _path_query(self, at: Basis, key: Key) -> tuple[str, bytes | None]:
        path = _path(key.selector)
        with self._workspace.retain(at.live) as live:
            self._check_snapshot(live, at.live)
            try:
                data = self._workspace.read(live, path)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return sha256(b"absent").hexdigest(), None
        if len(data) > _MAX_POINT_BYTES:
            raise Unavailable("turn point lookup exceeds the product read limit")
        return sha256(b"present\0" + data).hexdigest(), data

    @staticmethod
    def _symbol_name(key: Key) -> str:
        try:
            name = key.selector.decode("utf-8")
        except UnicodeError as error:
            raise Unavailable("turn symbol selector is invalid") from error
        if not name.isidentifier():
            raise Unavailable("turn symbol selector is invalid")
        return name

    @staticmethod
    def _cursor(at: Basis, key: Key, cursor: bytes | None) -> tuple[str, tuple[bytes, int]]:
        identity = sha256(encode([at.parent, at.live, at.epoch, at.authority,
                                  key.family, key.selector.hex()])).hexdigest()
        if cursor is None:
            return identity, _FIRST
        try:
            binding, path_hex, position = json.loads(cursor)
            path = bytes.fromhex(path_hex)
            if (binding != identity or not path or type(position) is not int
                    or not 0 <= position < (1 << 63)):
                raise ValueError("different query")
        except (TypeError, ValueError, UnicodeError) as error:
            raise Stale("turn fact cursor is stale or malformed") from error
        return identity, (path, position)

    def _lookup_symbols(self, at: Basis, key: Key, cursor: bytes | None, limit: int) -> Page:
        identity, after = self._cursor(at, key, cursor)
        name = self._symbol_name(key)
        with self._symbol_query(at) as (db, root, overlay):
            revision, expected_count, expected_total, added = self._membership(
                db, root, name, overlay)
            stored = FactStore.rows(db, root, name, after, tuple(overlay), limit + 1)
            rows = merge(stored, (row for row in added if row[:2] > after))
            payload: list[object] = []
            size = _EMPTY_PAGE_BYTES
            position = None
            next_cursor = None
            total = 0
            for path, index, value in islice(rows, limit + 1):
                if len(payload) == limit:
                    next_cursor = encode([identity, position[0].hex(), position[1]])
                    break
                digest = row_digest(path, index, value)
                if len(value) + _EMPTY_PAGE_BYTES > _MAX_PAGE_BYTES:
                    row = _ref_row(self._publication.stage(_BLOB_CODEC, value))
                    value = encode(row)
                else:
                    row = json.loads(value)
                if size + len(value) + bool(payload) > _MAX_PAGE_BYTES:
                    next_cursor = encode([identity, position[0].hex(), position[1]])
                    break
                payload.append(row)
                size += len(value) + bool(position)
                position = path, index
                total = combine(total, digest)
            if cursor is None and (
                    len(payload) > expected_count or
                    (next_cursor is not None and len(payload) == expected_count) or
                    (next_cursor is None and
                     (len(payload), total) != (expected_count, expected_total))):
                raise Corrupt("turn symbol membership does not match the indexed rows")
            return self._stage_page(key, revision, payload, next_cursor)

    def lookup(self, at: Basis, key: Key, cursor: bytes | None, limit: int) -> Page:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("turn fact page limit must be 1 to 256 rows")
        if key.family == "workspace-path":
            if cursor is not None:
                raise Stale("turn point lookup has no cursor")
            revision, data = self._path_query(at, key)
            rows = [] if data is None else [
                _ref_row(self._publication.stage(_BLOB_CODEC, data), path=key.selector.hex())
            ]
            return self._stage_page(key, revision, rows)
        if key.family == "symbol":
            return self._lookup_symbols(at, key, cursor, limit)
        raise Unavailable("unknown turn fact family")

    def current(self, at: Basis, deps: tuple[Dep, ...]) -> bool:
        try:
            for dep in deps:
                if dep.key.family == "workspace-path":
                    revision, _ = self._path_query(at, dep.key)
                elif dep.key.family == "symbol":
                    name = self._symbol_name(dep.key)
                    with self._symbol_query(at) as (db, root, overlay):
                        revision, _, _, _ = self._membership(db, root, name, overlay)
                else:
                    return False
                if revision != dep.revision:
                    return False
        except Unavailable:
            return False
        return True


def open_port(ctx: Context, *, workspace: Workspace, publication: Publication) -> Facts:
    """Read the supplied namespace without a workspace scan or index hydration."""
    return _Facts(ctx, workspace, publication)
