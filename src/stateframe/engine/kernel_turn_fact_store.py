"""Private, keyed storage for immutable turn symbol projections."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import stat
from threading import local

from kernel_turn_contracts import Corrupt, Unavailable


def encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _require_private(info: os.stat_result, kind: Callable[[int], bool]) -> None:
    if not kind(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise Unavailable("turn fact storage is not private")


class FactStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.path = directory / "symbols.sqlite"
        self._uri = self.path.absolute().as_uri()
        self._readers = local()

    def close(self) -> None:
        reader = getattr(self._readers, "reader", None)
        if reader is not None:
            reader.close()
            del self._readers.reader
            del self._readers.identity

    @contextmanager
    def _read(self, identity: tuple[int, int]) -> Iterator[sqlite3.Connection]:
        if getattr(self._readers, "identity", None) != identity:
            self.close()
            self._readers.reader = sqlite3.connect(self._uri + "?mode=ro", uri=True, timeout=0.05)
            self._readers.identity = identity
        db = self._readers.reader
        db.execute("BEGIN")
        try:
            yield db
        finally:
            db.rollback()

    @contextmanager
    def open(self, *, create: bool = False) -> Iterator[sqlite3.Connection | None]:
        try:
            if create:
                self.directory.mkdir(mode=0o700, exist_ok=True)
            try:
                _require_private(self.directory.lstat(), stat.S_ISDIR)
                if create:
                    try:
                        self.path.touch(mode=0o600, exist_ok=False)
                    except FileExistsError:
                        pass
                info = self.path.lstat()
                _require_private(info, stat.S_ISREG)
            except FileNotFoundError:
                self.close()
                yield None
                return
            if not create:
                with self._read((info.st_dev, info.st_ino)) as db:
                    yield db
                return
            with closing(sqlite3.connect(self._uri + "?mode=rw", uri=True, timeout=0.05)) as db:
                db.execute("PRAGMA journal_mode = WAL")
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS complete (
                        root TEXT PRIMARY KEY
                    ) WITHOUT ROWID;
                    CREATE TABLE IF NOT EXISTS symbols (
                        root TEXT, name TEXT, path BLOB, pos INTEGER,
                        payload BLOB, digest BLOB,
                        PRIMARY KEY (root, name, path, pos)
                    ) WITHOUT ROWID;
                    CREATE TABLE IF NOT EXISTS membership (
                        root TEXT, name TEXT, count INTEGER, digest BLOB,
                        PRIMARY KEY (root, name)
                    ) WITHOUT ROWID;
                """)
                yield db
        except sqlite3.DatabaseError as error:
            if getattr(error, "sqlite_errorcode", 0) & 0xFF in {
                sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_FULL,
                sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_IOERR, sqlite3.SQLITE_READONLY,
            }:
                raise Unavailable("turn fact storage is unavailable") from error
            raise Corrupt("turn fact storage cannot be read") from error
        except OSError as error:
            raise Unavailable("turn fact storage is unavailable") from error

    @staticmethod
    def complete(db: sqlite3.Connection, root: str) -> bool:
        return db.execute("SELECT 1 FROM complete WHERE root = ?", (root,)).fetchone() is not None

    @staticmethod
    def put(db: sqlite3.Connection, root: str, name: str, path: bytes,
            pos: int, row: dict) -> None:
        payload = encode(row)
        digest = row_digest(path, pos, payload)
        db.execute("INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?)",
                   (root, name, path, pos, payload, digest))

    @staticmethod
    def finish(db: sqlite3.Connection, root: str) -> None:
        from itertools import groupby

        rows = db.execute("SELECT name, digest FROM symbols WHERE root = ? ORDER BY name", (root,))
        for name, group in groupby(rows, key=lambda row: row[0]):
            count = total = 0
            for _, digest in group:
                count += 1
                total = combine(total, digest)
            db.execute("INSERT INTO membership VALUES (?, ?, ?, ?)",
                       (root, name, count, total.to_bytes(32, "big")))
        db.execute("INSERT INTO complete VALUES (?)", (root,))

    @staticmethod
    def membership(db: sqlite3.Connection, root: str, name: str) -> tuple[int, int]:
        row = db.execute(
            "SELECT count, digest FROM membership WHERE root = ? AND name = ?", (root, name),
        ).fetchone()
        return (row[0], int.from_bytes(row[1], "big")) if row is not None else (0, 0)

    @staticmethod
    def file_digests(db: sqlite3.Connection, root: str, name: str,
                     path: bytes) -> Iterator[bytes]:
        rows = db.execute(
            "SELECT digest FROM symbols WHERE root = ? AND name = ? AND path = ?",
            (root, name, path),
        )
        return (row[0] for row in rows)

    @staticmethod
    def rows(db: sqlite3.Connection, root: str, name: str, after: tuple[bytes, int],
             excluded: tuple[bytes, ...], limit: int) -> Iterator[tuple[bytes, int, bytes]]:
        sql = ("SELECT path, pos, payload, digest FROM symbols "
               "WHERE root = ? AND name = ? AND (path, pos) > (?, ?)")
        if excluded:
            sql += " AND path NOT IN (" + ", ".join("?" for _ in excluded) + ")"
        sql += " ORDER BY path, pos LIMIT ?"
        for path, pos, payload, digest in db.execute(sql, (root, name, *after, *excluded, limit)):
            if row_digest(path, pos, payload) != digest:
                raise Corrupt("turn symbol row digest mismatch")
            yield path, pos, payload


def row_digest(path: bytes, pos: int, payload: bytes) -> bytes:
    return sha256(path + b"\0" + pos.to_bytes(8, "big") + payload).digest()


def combine(total: int, digest: bytes, *, subtract: bool = False) -> int:
    contribution = int.from_bytes(digest, "big")
    return (total - contribution if subtract else total + contribution) % (1 << 256)
