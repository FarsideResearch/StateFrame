"""Run-private durable manifests, immutable payloads and ordered journal events."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

if __package__:
    from .kernel_checkpoint import _check_registered_secrets
    from .kernel_turn_contracts import (
        Commit, Context, Corrupt, Event, Head, Key, Publication, Ref, Stale, Unavailable,
    )
else:
    from kernel_checkpoint import _check_registered_secrets
    from kernel_turn_contracts import (
        Commit, Context, Corrupt, Event, Head, Key, Publication, Ref, Stale, Unavailable,
    )

SWITCH = "STATEFRAME_TURN_ATOMIC_PUBLICATION"
GENESIS = Head(0, "0" * 64)
_COMMIT_CODEC = "turn/commit/1"
_HASH = re.compile(r"[0-9a-f]{64}")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS identity (slot INTEGER PRIMARY KEY CHECK (slot = 1), value BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS objects (
    codec TEXT NOT NULL, digest TEXT NOT NULL, data BLOB NOT NULL, PRIMARY KEY (codec, digest)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS boundaries (
    id TEXT PRIMARY KEY, input BLOB NOT NULL, manifest BLOB NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS head (slot INTEGER PRIMARY KEY CHECK (slot = 1), manifest BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY, checksum TEXT NOT NULL, value BLOB NOT NULL
);
"""


def enabled() -> bool:
    return os.environ.get(SWITCH, "off").strip().lower() == "on"


def _encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _decode(value: bytes) -> object:
    try:
        decoded = json.loads(value)
        if _encode(decoded) != value:
            raise ValueError("noncanonical data")
        return decoded
    except (TypeError, ValueError, RecursionError) as error:
        raise Corrupt("invalid publication record") from error


def _ref(codec: str, data: bytes) -> Ref:
    if not isinstance(codec, str) or not codec or len(codec) > 128:
        raise Corrupt("invalid publication codec")
    return Ref(codec, hashlib.sha256(data).hexdigest(), len(data))


def _checked_ref(value: Ref) -> Ref:
    if (not isinstance(value, Ref) or not isinstance(value.codec, str) or not value.codec
            or len(value.codec) > 128 or not isinstance(value.sha256, str)
            or not _HASH.fullmatch(value.sha256) or type(value.size) is not int or value.size < 0):
        raise Corrupt("invalid publication reference")
    return value


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_file(path: Path, data: bytes) -> None:
    """Replace a small metadata file and acknowledge after directory durability."""
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _private(path: Path, *, directory: bool) -> None:
    info = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise Unavailable("publication storage must be private to the host user")


class _Publication:
    def __init__(self, ctx: Context):
        self._arena = None
        private_root = ctx.private_dir.absolute()
        if not stat.S_ISDIR(private_root.lstat().st_mode):
            raise Unavailable("publication parent must be a real directory")
        parent = private_root / ".turn-publication"
        parent.mkdir(mode=0o700, exist_ok=True)
        _private(parent, directory=True)
        sync_directory(private_root)
        if not enabled():
            self._arena = TemporaryDirectory(dir=parent)
            parent = Path(self._arena.name)
        namespace = _encode([ctx.principal, ctx.run, ctx.profile_digest])
        directory = parent / (".turn-" + hashlib.sha256(namespace).hexdigest())
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            creating = False
        else:
            creating = True
        _private(directory, directory=True)
        self._path = directory / "publication.sqlite3"
        if creating:
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            sync_directory(directory)
        _private(self._path, directory=False)
        sync_directory(parent)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._event_tip: Head | None = None
        self._advanced = False
        with self._database() as db:
            db.executescript("BEGIN IMMEDIATE;\n" + _SCHEMA)
            db.execute("INSERT OR IGNORE INTO identity VALUES (1, ?)", (namespace,))
            if db.execute("SELECT value FROM identity WHERE slot = 1").fetchone() != (namespace,):
                raise Corrupt("publication namespace mismatch")
            db.commit()

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        try:
            with closing(sqlite3.connect(self._path.as_uri() + "?mode=rw", uri=True,
                                         timeout=5, isolation_level=None)) as db:
                # EXTRA also syncs the directory after deleting the rollback journal.
                db.execute("PRAGMA synchronous=EXTRA")
                yield db
        except sqlite3.DatabaseError as error:
            if getattr(error, "sqlite_errorcode", None) in (
                sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB,
            ):
                raise Corrupt("publication database failed integrity checks") from error
            raise Unavailable("publication database operation failed") from error

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            if self._connection is not None:
                raise Unavailable("nested publication transactions are not supported")
            with self._database() as db:
                db.execute("BEGIN IMMEDIATE")
                self._connection = db
                try:
                    yield
                    if self._event_tip is not None:
                        raise Corrupt("journal records were staged without a manifest")
                    if db.in_transaction:
                        db.commit()
                except BaseException:
                    db.rollback()
                    raise
                finally:
                    self._event_tip = None
                    self._connection = None
                    self._advanced = False

    @contextmanager
    def _using(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._connection is not None:
                yield self._connection
            elif write:
                with self.transaction():
                    assert self._connection is not None
                    yield self._connection
            else:
                with self._database() as db:
                    db.execute("BEGIN")
                    yield db

    def stage(self, codec: str, data: bytes) -> Ref:
        if not isinstance(data, bytes):
            raise Corrupt("publication payload must be bytes")
        _check_registered_secrets(data, failure_type=Unavailable)
        ref = _ref(codec, data)
        with self._using(write=True) as db:
            if self._advanced:
                raise Unavailable("publication transaction already committed")
            changed = db.execute("INSERT OR IGNORE INTO objects VALUES (?, ?, ?)",
                                 (ref.codec, ref.sha256, data)).rowcount
            if not changed and self._load(db, ref) != data:
                raise Corrupt("publication payload collision")
        return ref

    @staticmethod
    def _load(db: sqlite3.Connection, ref: Ref) -> bytes:
        _checked_ref(ref)
        row = db.execute("SELECT data FROM objects WHERE codec = ? AND digest = ?",
                         (ref.codec, ref.sha256)).fetchone()
        if row is None or not isinstance(row[0], bytes) or _ref(ref.codec, row[0]) != ref:
            raise Corrupt("publication payload missing or damaged")
        return row[0]

    def load(self, ref: Ref) -> bytes:
        with self._using() as db:
            return self._load(db, ref)

    def _manifest(self, db: sqlite3.Connection, encoded: bytes) -> tuple[Ref, dict]:
        try:
            ref = Ref(*_decode(encoded))
            raw = self._load(db, ref)
            if ref.codec != _COMMIT_CODEC:
                raise Corrupt("invalid publication manifest codec")
            result = _decode(raw)
            if not isinstance(result, dict):
                raise Corrupt("invalid publication manifest")
            return ref, result
        except (ValueError, TypeError) as error:
            raise Corrupt("invalid publication manifest reference") from error

    def _current(self, db: sqlite3.Connection) -> tuple[Ref | None, dict]:
        row = db.execute("SELECT manifest FROM head WHERE slot = 1").fetchone()
        if row is not None:
            return self._manifest(db, row[0])
        if db.execute("SELECT 1 FROM boundaries LIMIT 1").fetchone() is not None:
            raise Corrupt("publication history has no verified head")
        return None, {}

    def head(self) -> Ref | None:
        with self._using() as db:
            return self._current(db)[0]

    def commit(self, proposed: Commit) -> Ref:
        roots = dict(proposed.roots)
        if (not isinstance(proposed.boundary_id, str) or not proposed.boundary_id
                or len(roots) != len(proposed.roots)
                or any(not isinstance(name, str) or not name for name in roots)):
            raise Corrupt("invalid publication boundary or root names")
        desired = _encode([proposed.basis, proposed.previous, sorted(roots.items())])
        with self._using(write=True) as db:
            duplicate = db.execute("SELECT input, manifest FROM boundaries WHERE id = ?",
                                   (proposed.boundary_id,)).fetchone()
            if duplicate is not None:
                if duplicate[0] != desired:
                    raise Corrupt("publication boundary reused with conflicting content")
                return self._manifest(db, duplicate[1])[0]
            if self._advanced:
                raise Corrupt("publication transaction already advanced its head")
            current, previous = self._current(db)
            if current != proposed.previous:
                raise Stale("publication head changed")
            tip = self._event_tip if self._event_tip is not None else self._tip(previous)
            if proposed.basis.journal != tip:
                raise Corrupt("publication journal head lacks a contiguous event chain")
            for ref in roots.values():
                self._load(db, ref)
            inherited = dict(previous.get("roots", ()))
            inherited.update(roots)
            manifest = self.stage(_COMMIT_CODEC, _encode({
                "basis": proposed.basis, "previous": current, "boundary_id": proposed.boundary_id,
                "roots": sorted(inherited.items()),
            }))
            encoded = _encode(manifest)
            db.execute("INSERT INTO boundaries VALUES (?, ?, ?)",
                       (proposed.boundary_id, desired, encoded))
            db.execute("INSERT OR REPLACE INTO head VALUES (1, ?)", (encoded,))
            db.commit()
            self._event_tip = None
            self._advanced = True
            return manifest

    @staticmethod
    def _tip(manifest: dict) -> Head:
        try:
            return Head(*manifest["basis"][3]) if manifest else GENESIS
        except (KeyError, IndexError, TypeError) as error:
            raise Corrupt("invalid publication manifest basis") from error

    def record(self, event: Event) -> None:
        """Stage one event inside the same transaction as its eventual manifest."""
        with self._lock:
            db = self._connection
            if db is None or self._advanced:
                raise Unavailable("events require an open publication transaction")
            if (not isinstance(event.kind, str) or not event.kind
                    or event.after.seq != event.before.seq + 1 or not event.after.digest
                    or event.after.digest == event.before.digest):
                raise Corrupt("invalid publication event")
            value = _encode([event.before, event.after, event.kind,
                             [(key.family, key.selector.hex()) for key in event.changed], event.payload])
            fingerprint = hashlib.sha256(value).hexdigest()
            duplicate = db.execute("SELECT checksum, value FROM events WHERE seq = ?",
                                   (event.after.seq,)).fetchone()
            if duplicate is not None:
                if duplicate != (fingerprint, value):
                    raise Corrupt("publication event conflicts with its recorded sequence")
                return
            tip = self._event_tip
            if tip is None:
                tip = self._tip(self._current(db)[1])
            if event.before != tip:
                raise Corrupt("publication event does not extend the verified head")
            self._load(db, event.payload)
            db.execute("INSERT INTO events VALUES (?, ?, ?)", (event.after.seq, fingerprint, value))
            self._event_tip = event.after

    @staticmethod
    def _event(checksum: str, value: bytes) -> Event:
        try:
            if hashlib.sha256(value).hexdigest() != checksum:
                raise Corrupt("invalid publication event checksum")
            before, after, kind, changes, payload = _decode(value)
            return Event(Head(*before), Head(*after), kind,
                         tuple(Key(family, bytes.fromhex(key)) for family, key in changes),
                         Ref(*payload))
        except (TypeError, ValueError) as error:
            raise Corrupt("invalid publication event") from error

    def suffix(self, after: Head, through: Head) -> Iterator[Event]:
        if after.seq < 0 or through.seq < after.seq:
            raise Corrupt("invalid publication event range")
        with self._using() as db:
            tip = self._tip(self._current(db)[1])
            if through.seq > tip.seq:
                raise Stale("requested publication event is not committed")
            if after.seq == 0:
                start = GENESIS
            else:
                row = db.execute("SELECT checksum, value FROM events WHERE seq = ?", (after.seq,)).fetchone()
                if row is None:
                    raise Corrupt("publication event cursor is missing")
                start = self._event(*row).after
            if after != start:
                raise Corrupt("publication event cursor diverged")
            current = after
            rows = db.execute("SELECT checksum, value FROM events WHERE seq > ? AND seq <= ? ORDER BY seq",
                              (after.seq, through.seq))
            for row in rows:
                event = self._event(*row)
                if event.before != current or event.after.seq != current.seq + 1:
                    raise Corrupt("publication event chain has a gap or fork")
                self._load(db, event.payload)
                current = event.after
                yield event
            if current != through:
                raise Corrupt("publication event range does not match its verified end")


def open_port(ctx: Context) -> _Publication:
    """Construct the run's publication port, or a disposable arena when switched off."""
    try:
        return _Publication(ctx)
    except OSError as error:
        raise Unavailable("publication storage unavailable") from error



def record_event(port: Publication, event: Event) -> None:
    """Enroll a host-typed event while staging a manifest on this adapter."""
    if not isinstance(port, _Publication):
        raise Unavailable("publication adapter cannot record events")
    port.record(event)
