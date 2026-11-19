"""Single-writer SQLite catalog primitives for the data workspace.

Importing this module is side-effect free. Catalog writers hold a persistent sidecar
``flock`` from before SQLite opens until after SQLite closes; readers use a read-only
URI and never take ownership of the WAL or schema.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import threading
from typing import TypeAlias

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on platforms without flock
    fcntl = None  # type: ignore[assignment]


SQLITE_VERSION: tuple[int, int, int] = sqlite3.sqlite_version_info
SCHEMA_VERSION = 1
APPLICATION_ID = 1397113923  # 0x53464443 (SFDC)
_FORMAT_NAME = "stateframe-data-catalog"
_MINIMUM_STRICT_VERSION = (3, 37, 0)
_PRIVATE_BITS = 0o077
_REQUIRED_OPEN_FLAGS = ("O_CLOEXEC", "O_NOFOLLOW")
_LOCAL_FILESYSTEMS = frozenset({"ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "tmpfs", "overlay", "zfs"})

Parameters: TypeAlias = tuple[object, ...]


class CatalogRefused(RuntimeError):
    """A catalog operation was refused for a stable, caller-visible reason."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


_TABLE_DDL: dict[str, str] = {
    "catalog_meta": """CREATE TABLE catalog_meta (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
        format_name TEXT NOT NULL CHECK(format_name='stateframe-data-catalog'),
        schema_version INTEGER NOT NULL CHECK(schema_version=1)
    ) STRICT""",
    "objects": """CREATE TABLE objects (
        object_id TEXT PRIMARY KEY CHECK(length(object_id)=64 AND object_id NOT GLOB '*[^0-9a-f]*'),
        kind TEXT NOT NULL CHECK(kind IN (
            'source','snapshot','manifest','part','schema','query','result','packet','request','recipe','event'
        )),
        byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
        storage_state TEXT NOT NULL CHECK(storage_state IN ('present','missing','gc_pending')),
        created_us INTEGER NOT NULL CHECK(created_us >= 0)
    ) STRICT""",
    "resources": """CREATE TABLE resources (
        resource_id TEXT PRIMARY KEY CHECK(length(resource_id)>0),
        label TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        locator_redacted TEXT,
        state TEXT NOT NULL CHECK(state IN ('active','dropped')),
        created_us INTEGER NOT NULL CHECK(created_us >= 0)
    ) STRICT""",
    "resource_objects": """CREATE TABLE resource_objects (
        resource_id TEXT NOT NULL REFERENCES resources(resource_id),
        object_id TEXT NOT NULL REFERENCES objects(object_id),
        role TEXT NOT NULL CHECK(role IN ('source','metadata')),
        ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
        PRIMARY KEY(resource_id, role, ordinal)
    ) STRICT""",
    "snapshots": """CREATE TABLE snapshots (
        snapshot_id TEXT PRIMARY KEY CHECK(length(snapshot_id)>0),
        resource_id TEXT NOT NULL REFERENCES resources(resource_id),
        manifest_object_id TEXT NOT NULL REFERENCES objects(object_id),
        captured_us INTEGER NOT NULL CHECK(captured_us >= 0),
        UNIQUE(resource_id, manifest_object_id)
    ) STRICT""",
    "datasets": """CREATE TABLE datasets (
        dataset_id TEXT PRIMARY KEY CHECK(length(dataset_id)>0),
        label TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('active','dropped')),
        created_us INTEGER NOT NULL CHECK(created_us >= 0)
    ) STRICT""",
    "revisions": """CREATE TABLE revisions (
        revision_id TEXT PRIMARY KEY CHECK(length(revision_id)>0),
        dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
        revision_number INTEGER NOT NULL CHECK(revision_number >= 1),
        parent_revision_id TEXT,
        manifest_object_id TEXT NOT NULL REFERENCES objects(object_id),
        schema_object_id TEXT NOT NULL REFERENCES objects(object_id),
        coverage TEXT NOT NULL CHECK(coverage IN ('complete','partial','unavailable')),
        created_us INTEGER NOT NULL CHECK(created_us >= 0),
        UNIQUE(dataset_id, revision_number),
        UNIQUE(dataset_id, revision_id),
        FOREIGN KEY(dataset_id, parent_revision_id) REFERENCES revisions(dataset_id, revision_id)
    ) STRICT""",
    "revision_sources": """CREATE TABLE revision_sources (
        revision_id TEXT NOT NULL REFERENCES revisions(revision_id),
        source_ordinal INTEGER NOT NULL CHECK(source_ordinal >= 0),
        snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
        PRIMARY KEY(revision_id, source_ordinal)
    ) STRICT""",
    "revision_parts": """CREATE TABLE revision_parts (
        revision_id TEXT NOT NULL REFERENCES revisions(revision_id),
        part_ordinal INTEGER NOT NULL CHECK(part_ordinal >= 0),
        part_object_id TEXT NOT NULL REFERENCES objects(object_id),
        row_count INTEGER NOT NULL CHECK(row_count >= 0),
        PRIMARY KEY(revision_id, part_ordinal),
        UNIQUE(revision_id, part_object_id)
    ) STRICT""",
    "jobs": """CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY CHECK(length(job_id)>0),
        request_id TEXT NOT NULL UNIQUE CHECK(length(request_id)>0),
        operation TEXT NOT NULL CHECK(operation IN ('register','query','derive','put','export','drop','validate')),
        request_object_id TEXT NOT NULL REFERENCES objects(object_id),
        revision_id TEXT REFERENCES revisions(revision_id),
        state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed','cancelled')),
        phase TEXT NOT NULL,
        completed_units INTEGER NOT NULL CHECK(completed_units >= 0),
        error_code TEXT,
        created_us INTEGER NOT NULL CHECK(created_us >= 0),
        updated_us INTEGER NOT NULL CHECK(updated_us >= created_us),
        CHECK((state='failed' AND error_code IS NOT NULL AND length(error_code)>0)
           OR (state<>'failed' AND error_code IS NULL))
    ) STRICT""",
    "results": """CREATE TABLE results (
        result_id TEXT PRIMARY KEY CHECK(length(result_id)>0),
        job_id TEXT NOT NULL REFERENCES jobs(job_id),
        revision_id TEXT NOT NULL REFERENCES revisions(revision_id),
        query_object_id TEXT NOT NULL REFERENCES objects(object_id),
        result_object_id TEXT NOT NULL REFERENCES objects(object_id),
        status TEXT NOT NULL CHECK(status IN ('ok','empty')),
        coverage TEXT NOT NULL CHECK(coverage IN ('complete','partial','unavailable')),
        created_us INTEGER NOT NULL CHECK(created_us >= 0),
        UNIQUE(result_id, revision_id, status)
    ) STRICT""",
    "packets": """CREATE TABLE packets (
        packet_id TEXT PRIMARY KEY CHECK(length(packet_id)>0),
        revision_id TEXT NOT NULL REFERENCES revisions(revision_id),
        result_id TEXT,
        status TEXT NOT NULL CHECK(status IN ('ok','empty','failed')),
        packet_object_id TEXT NOT NULL REFERENCES objects(object_id),
        error_code TEXT,
        created_us INTEGER NOT NULL CHECK(created_us >= 0),
        UNIQUE(packet_id, revision_id),
        FOREIGN KEY(result_id, revision_id, status) REFERENCES results(result_id, revision_id, status),
        CHECK((status='failed' AND result_id IS NULL AND error_code IS NOT NULL AND length(error_code)>0)
           OR (status IN ('ok','empty') AND result_id IS NOT NULL AND error_code IS NULL))
    ) STRICT""",
    "roots": """CREATE TABLE roots (
        root_id TEXT PRIMARY KEY CHECK(length(root_id)>0),
        owner_kind TEXT NOT NULL CHECK(owner_kind IN ('run','binding','publication','user','head')),
        owner_id TEXT NOT NULL CHECK(length(owner_id)>0),
        object_id TEXT NOT NULL REFERENCES objects(object_id),
        retention_level TEXT NOT NULL CHECK(retention_level IN ('transcript_replay','evidence_reopen','recompute')),
        created_us INTEGER NOT NULL CHECK(created_us >= 0),
        UNIQUE(owner_kind, owner_id, object_id, retention_level)
    ) STRICT""",
    "provenance_edges": """CREATE TABLE provenance_edges (
        from_object_id TEXT NOT NULL REFERENCES objects(object_id),
        to_object_id TEXT NOT NULL REFERENCES objects(object_id),
        relation TEXT NOT NULL CHECK(length(relation)>0),
        PRIMARY KEY(from_object_id, to_object_id, relation),
        CHECK(from_object_id <> to_object_id)
    ) STRICT""",
    "publications": """CREATE TABLE publications (
        publication_id TEXT PRIMARY KEY CHECK(length(publication_id)>0),
        job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
        fence INTEGER NOT NULL CHECK(fence >= 1),
        owner_token TEXT NOT NULL CHECK(length(owner_token)>0),
        phase TEXT NOT NULL CHECK(phase IN (
            'intent','staging','validated','installed','committed','delivered','failed'
        )),
        candidate_object_id TEXT REFERENCES objects(object_id),
        committed_object_id TEXT REFERENCES objects(object_id),
        error_code TEXT,
        updated_us INTEGER NOT NULL CHECK(updated_us >= 0),
        CHECK(phase NOT IN ('installed','committed','delivered') OR candidate_object_id IS NOT NULL),
        CHECK((phase IN ('committed','delivered') AND committed_object_id IS NOT NULL)
           OR (phase NOT IN ('committed','delivered') AND committed_object_id IS NULL)),
        CHECK(committed_object_id IS NULL OR committed_object_id=candidate_object_id),
        CHECK((phase='failed' AND error_code IS NOT NULL AND length(error_code)>0)
           OR (phase<>'failed' AND error_code IS NULL))
    ) STRICT""",
    "journal_outbox": """CREATE TABLE journal_outbox (
        event_id TEXT PRIMARY KEY CHECK(length(event_id)>0),
        publication_id TEXT NOT NULL UNIQUE REFERENCES publications(publication_id),
        payload_object_id TEXT NOT NULL REFERENCES objects(object_id),
        state TEXT NOT NULL CHECK(state IN ('pending','delivered')),
        delivery_attempts INTEGER NOT NULL CHECK(delivery_attempts >= 0),
        journal_receipt TEXT,
        created_us INTEGER NOT NULL CHECK(created_us >= 0),
        CHECK((state='delivered' AND journal_receipt IS NOT NULL)
           OR (state='pending' AND journal_receipt IS NULL))
    ) STRICT""",
    "analysis_heads": """CREATE TABLE analysis_heads (
        analysis_id TEXT PRIMARY KEY CHECK(length(analysis_id)>0),
        dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
        revision_id TEXT NOT NULL,
        packet_id TEXT,
        generation INTEGER NOT NULL CHECK(generation >= 1),
        FOREIGN KEY(dataset_id, revision_id) REFERENCES revisions(dataset_id, revision_id),
        FOREIGN KEY(packet_id, revision_id) REFERENCES packets(packet_id, revision_id)
    ) STRICT""",
    "bindings": """CREATE TABLE bindings (
        binding_id TEXT PRIMARY KEY CHECK(length(binding_id)>0),
        consumer_id TEXT NOT NULL CHECK(length(consumer_id)>0),
        binding_name TEXT NOT NULL CHECK(length(binding_name)>0),
        dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
        revision_id TEXT NOT NULL,
        packet_id TEXT,
        state TEXT NOT NULL CHECK(state IN ('active','dropped')),
        generation INTEGER NOT NULL CHECK(generation >= 1),
        UNIQUE(consumer_id, binding_name),
        FOREIGN KEY(dataset_id, revision_id) REFERENCES revisions(dataset_id, revision_id),
        FOREIGN KEY(packet_id, revision_id) REFERENCES packets(packet_id, revision_id)
    ) STRICT""",
}

_INDEX_DDL: dict[str, str] = {
    "resource_objects_object": "CREATE INDEX resource_objects_object ON resource_objects(object_id)",
    "snapshots_manifest": "CREATE INDEX snapshots_manifest ON snapshots(manifest_object_id)",
    "revision_sources_snapshot": "CREATE INDEX revision_sources_snapshot ON revision_sources(snapshot_id)",
    "revision_parts_object": "CREATE INDEX revision_parts_object ON revision_parts(part_object_id)",
    "jobs_dispatch": "CREATE INDEX jobs_dispatch ON jobs(state, created_us, job_id)",
    "results_revision": "CREATE INDEX results_revision ON results(revision_id, result_id)",
    "results_job": "CREATE INDEX results_job ON results(job_id)",
    "packets_result": "CREATE INDEX packets_result ON packets(result_id, revision_id, status)",
    "packets_object": "CREATE INDEX packets_object ON packets(packet_object_id)",
    "roots_object": "CREATE INDEX roots_object ON roots(object_id, retention_level)",
    "provenance_reverse": "CREATE INDEX provenance_reverse ON provenance_edges(to_object_id)",
    "publications_resume": "CREATE INDEX publications_resume ON publications(phase, updated_us)",
    "outbox_pending": "CREATE INDEX outbox_pending ON journal_outbox(state, created_us, event_id)",
    "heads_revision": "CREATE INDEX heads_revision ON analysis_heads(dataset_id, revision_id)",
    "heads_packet": "CREATE INDEX heads_packet ON analysis_heads(packet_id, revision_id)",
    "bindings_revision": "CREATE INDEX bindings_revision ON bindings(dataset_id, revision_id)",
    "bindings_packet": "CREATE INDEX bindings_packet ON bindings(packet_id, revision_id)",
}


def _validate_sqlite_version(version: object) -> tuple[int, int, int]:
    if type(version) is not tuple or len(version) != 3:
        raise TypeError("SQLite version must be a three-element tuple")
    if any(type(part) is not int for part in version):
        raise TypeError("SQLite version components must be plain integers")
    if any(part < 0 for part in version):
        raise ValueError("SQLite version components must be non-negative")
    return version


def sqlite_wal_safe(version: tuple[int, int, int]) -> bool:
    """Return whether SQLite includes the WAL-reset fix for its release line."""
    checked = _validate_sqlite_version(version)
    return (
        checked >= (3, 51, 3)
        or (checked[:2] == (3, 50) and checked[2] >= 7)
        or (checked[:2] == (3, 44) and checked[2] >= 6)
    )


def _runtime_version() -> tuple[int, int, int]:
    version = _validate_sqlite_version(SQLITE_VERSION)
    if version < _MINIMUM_STRICT_VERSION:
        raise CatalogRefused("sqlite_strict_required", "SQLite 3.37.0 or newer is required")
    return version


def _unescape_mount_field(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _require_local_filesystem(directory: Path) -> None:
    """Fail closed when the catalog directory's mount cannot be verified as local."""
    if not sys.platform.startswith("linux"):
        raise CatalogRefused("writer_lock_unsupported", "Cannot verify local filesystem locking here")
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CatalogRefused("writer_lock_unsupported", "Cannot verify local filesystem locking") from exc

    target = directory.as_posix()
    mounts: list[tuple[int, str, str]] = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            mountpoint = _unescape_mount_field(fields[4])
            filesystem = fields[separator + 1]
        except (IndexError, ValueError):
            continue
        if target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/"):
            mounts.append((len(mountpoint), mountpoint, filesystem))
    if not mounts:
        raise CatalogRefused("writer_lock_unsupported", "Catalog mount could not be identified")
    filesystem = max(mounts)[2]
    if filesystem not in _LOCAL_FILESYSTEMS:
        raise CatalogRefused("writer_lock_unsupported", "Catalog must use a verified local filesystem")


def _validate_private_directory(directory: Path) -> None:
    try:
        info = directory.stat()
    except OSError as exc:
        raise CatalogRefused("catalog_directory_invalid", "Catalog directory is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise CatalogRefused("catalog_directory_invalid", "Catalog directory must be owned by this user")
    mode = stat.S_IMODE(info.st_mode)
    if mode & _PRIVATE_BITS or mode & 0o700 != 0o700:
        raise CatalogRefused("catalog_directory_invalid", "Catalog directory must be private")


def _resolve_paths(path: str | os.PathLike[str], *, reader: bool) -> tuple[Path, Path]:
    try:
        supplied = Path(path)
        if not supplied.name or supplied.name in {".", ".."}:
            raise ValueError("catalog file name is empty")
        parent = supplied.parent.resolve(strict=True)
    except (TypeError, ValueError, OSError) as exc:
        raise CatalogRefused("catalog_path_invalid", "Catalog path is invalid") from exc
    _validate_private_directory(parent)
    _require_local_filesystem(parent)
    database = parent / supplied.name
    lock = parent / f"{supplied.name}.writer.lock"
    _validate_regular_file(database, required=reader, label="catalog")
    _validate_regular_file(lock, required=reader, label="writer lock")
    return database, lock


def _validate_regular_file(path: Path, *, required: bool, label: str) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if required:
            code = "catalog_missing" if label == "catalog" else "writer_lock_missing"
            raise CatalogRefused(code, f"The managed {label} is missing")
        return None
    except OSError as exc:
        raise CatalogRefused("catalog_path_invalid", f"The managed {label} cannot be inspected") from exc
    if not stat.S_ISREG(info.st_mode):
        raise CatalogRefused("catalog_path_invalid", f"The managed {label} must be a regular file")
    if info.st_uid != os.geteuid() or info.st_nlink != 1:
        raise CatalogRefused("catalog_path_invalid", f"The managed {label} has an invalid owner or link count")
    mode = stat.S_IMODE(info.st_mode)
    if mode & _PRIVATE_BITS or not mode & 0o400:
        raise CatalogRefused("catalog_path_invalid", f"The managed {label} must be private")
    return info


def _file_identity_matches(path: Path, fd: int) -> bool:
    try:
        path_info = path.lstat()
        fd_info = os.fstat(fd)
    except OSError:
        return False
    return (
        stat.S_ISREG(path_info.st_mode)
        and path_info.st_dev == fd_info.st_dev
        and path_info.st_ino == fd_info.st_ino
        and path_info.st_nlink == 1
    )


def _open_writer_lock(path: Path) -> int:
    if fcntl is None or any(not hasattr(os, flag) for flag in _REQUIRED_OPEN_FLAGS):
        raise CatalogRefused("writer_lock_unsupported", "Required cross-process lock support is unavailable")
    _validate_regular_file(path, required=False, label="writer lock")
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CatalogRefused("writer_lock_failed", "Could not open the catalog writer lock") from exc
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & _PRIVATE_BITS or not _file_identity_matches(path, fd)):
            raise CatalogRefused("writer_lock_failed", "Catalog writer lock identity is unsafe")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                raise CatalogRefused("writer_busy", "Another catalog writer owns this workspace") from exc
            unsupported = {errno.ENOSYS, errno.ENOLCK}
            if hasattr(errno, "EOPNOTSUPP"):
                unsupported.add(errno.EOPNOTSUPP)
            if hasattr(errno, "ENOTSUP"):
                unsupported.add(errno.ENOTSUP)
            if exc.errno in unsupported:
                raise CatalogRefused("writer_lock_unsupported", "Cross-process file locking is unavailable") from exc
            raise CatalogRefused("writer_lock_failed", "Could not acquire the catalog writer lock") from exc
        if not _file_identity_matches(path, fd):
            raise CatalogRefused("writer_lock_failed", "Catalog writer lock identity changed")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _ensure_database_file(path: Path) -> None:
    existing = _validate_regular_file(path, required=False, label="catalog")
    if existing is None:
        if any(not hasattr(os, flag) for flag in _REQUIRED_OPEN_FLAGS):
            raise CatalogRefused("writer_lock_unsupported", "Required safe file-open support is unavailable")
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            pass
        except OSError as exc:
            raise CatalogRefused("catalog_open_failed", "Could not create the catalog file") from exc
        else:
            try:
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1
                        or stat.S_IMODE(info.st_mode) & _PRIVATE_BITS):
                    raise CatalogRefused("catalog_path_invalid", "New catalog file has unsafe ownership or mode")
            finally:
                os.close(fd)
    elif not stat.S_IMODE(existing.st_mode) & 0o200:
        raise CatalogRefused("catalog_path_invalid", "Catalog file is not writable by its owner")
    _validate_regular_file(path, required=True, label="catalog")


def _configure_writer(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=0")
    connection.execute("PRAGMA locking_mode=NORMAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("PRAGMA synchronous=FULL")
    checks = {
        "foreign_keys": 1,
        "busy_timeout": 0,
        "wal_autocheckpoint": 0,
        "synchronous": 2,
    }
    for name, expected in checks.items():
        row = connection.execute(f"PRAGMA {name}").fetchone()
        if row is None or row[0] != expected:
            raise CatalogRefused("catalog_configuration_failed", "SQLite writer settings could not be enforced")
    locking = connection.execute("PRAGMA locking_mode").fetchone()
    if locking is None or str(locking[0]).lower() != "normal":
        raise CatalogRefused("catalog_configuration_failed", "SQLite writer locking mode is unavailable")


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";")


def _schema_objects(connection: sqlite3.Connection, object_type: str) -> dict[str, str]:
    rows = connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE type=? AND name NOT LIKE 'sqlite_%'",
        (object_type,),
    ).fetchall()
    return {str(name): str(sql) for name, sql in rows if sql is not None}


def _verify_schema(connection: sqlite3.Connection) -> None:
    expected_tables = set(_TABLE_DDL)
    actual_tables = _schema_objects(connection, "table")
    if set(actual_tables) != expected_tables:
        raise CatalogRefused("catalog_schema_unsupported", "Catalog table set does not match schema version")
    for name, expected_sql in _TABLE_DDL.items():
        if _normalize_sql(actual_tables[name]) != _normalize_sql(expected_sql):
            raise CatalogRefused(
                "catalog_schema_unsupported", "Catalog table definition does not match schema version"
            )

    actual_indexes = _schema_objects(connection, "index")
    if set(actual_indexes) != set(_INDEX_DDL):
        raise CatalogRefused("catalog_schema_unsupported", "Catalog index set does not match schema version")
    for name, expected_sql in _INDEX_DDL.items():
        if _normalize_sql(actual_indexes[name]) != _normalize_sql(expected_sql):
            raise CatalogRefused(
                "catalog_schema_unsupported", "Catalog index definition does not match schema version"
            )

    extra_objects = connection.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('view','trigger') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    if extra_objects:
        raise CatalogRefused("catalog_schema_unsupported", "Catalog contains unsupported schema objects")

    table_list = connection.execute("PRAGMA table_list").fetchall()
    strict_tables = {
        str(row[1]): int(row[5])
        for row in table_list
        if len(row) >= 6 and row[0] == "main" and row[2] == "table" and not str(row[1]).startswith("sqlite_")
    }
    if set(strict_tables) != expected_tables or any(value != 1 for value in strict_tables.values()):
        raise CatalogRefused("catalog_schema_unsupported", "Every catalog table must be STRICT")

    metadata = connection.execute(
        "SELECT singleton, format_name, schema_version FROM catalog_meta"
    ).fetchall()
    if [tuple(row) for row in metadata] != [(1, _FORMAT_NAME, SCHEMA_VERSION)]:
        raise CatalogRefused("catalog_schema_unsupported", "Catalog format marker is missing or inconsistent")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise CatalogRefused("catalog_schema_unsupported", "Catalog contains a foreign-key violation")


def _classify_database(connection: sqlite3.Connection) -> bool:
    """Verify identity and existing schema; return True only for a genuinely empty database."""
    application_row = connection.execute("PRAGMA application_id").fetchone()
    version_row = connection.execute("PRAGMA user_version").fetchone()
    if application_row is None or version_row is None:
        raise CatalogRefused("catalog_identity_mismatch", "Catalog identity could not be read")
    application_id, user_version = int(application_row[0]), int(version_row[0])
    user_objects = connection.execute(
        "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    if application_id == 0 and user_version == 0:
        if user_objects:
            raise CatalogRefused("catalog_schema_unsupported", "Unidentified database has a partial catalog schema")
        return True
    if application_id != APPLICATION_ID:
        raise CatalogRefused("catalog_identity_mismatch", "Database application identity does not match")
    if user_version != SCHEMA_VERSION:
        raise CatalogRefused("catalog_schema_unsupported", "Catalog schema version is not supported")
    _verify_schema(connection)
    return False


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in _TABLE_DDL.values():
            connection.execute(statement)
        connection.execute(
            "INSERT INTO catalog_meta(singleton, format_name, schema_version) VALUES(1, ?, ?)",
            (_FORMAT_NAME, SCHEMA_VERSION),
        )
        for statement in _INDEX_DDL.values():
            connection.execute(statement)
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        connection.commit()
    except BaseException:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        raise
    _verify_schema(connection)


def _enable_wal(connection: sqlite3.Connection) -> None:
    try:
        row = connection.execute("PRAGMA journal_mode=WAL").fetchone()
    except sqlite3.Error as exc:
        raise CatalogRefused("wal_unavailable", "SQLite could not enable WAL mode") from exc
    if row is None or str(row[0]).lower() != "wal":
        raise CatalogRefused("wal_unavailable", "SQLite did not enter WAL mode")


def _close_after_failed_open(connection: sqlite3.Connection | None, lock_fd: int) -> None:
    if connection is not None:
        try:
            connection.close()
        except BaseException as exc:
            # Retain the flock if SQLite may still own write handles.
            raise CatalogRefused(
                "catalog_close_failed", "Could not safely close the failed catalog connection"
            ) from exc
    try:
        os.close(lock_fd)
    except OSError as exc:
        raise CatalogRefused("writer_lock_close_failed", "Could not close the catalog writer lock") from exc


class CatalogWriter:
    """Thread- and process-affine owner for the catalog's sole SQLite write connection."""

    def __init__(self, connection: sqlite3.Connection, lock_fd: int, lock_path: Path) -> None:
        self._connection = connection
        self._lock_fd = lock_fd
        self._lock_path = lock_path
        self._pid = os.getpid()
        self._thread_id = threading.get_ident()
        self._closed = False

    def _check_owner(self) -> None:
        if self._closed:
            raise CatalogRefused("writer_closed", "Catalog writer is closed")
        if os.getpid() != self._pid or threading.get_ident() != self._thread_id:
            raise CatalogRefused("writer_owner_mismatch", "Catalog writer cannot cross process or thread boundaries")
        if not _file_identity_matches(self._lock_path, self._lock_fd):
            raise CatalogRefused("writer_lock_lost", "Catalog writer lock identity changed")

    def execute(self, sql: str, parameters: Parameters = ()) -> sqlite3.Cursor:
        """Execute trusted host SQL; agent requests must never pass SQL through this API."""
        self._check_owner()
        return self._connection.execute(sql, parameters)

    def commit(self) -> None:
        self._check_owner()
        self._connection.commit()

    def rollback(self) -> None:
        self._check_owner()
        self._connection.rollback()

    def close(self) -> None:
        if self._closed:
            return
        if os.getpid() != self._pid or threading.get_ident() != self._thread_id:
            raise CatalogRefused("writer_owner_mismatch", "Catalog writer must close on its owning process and thread")
        rollback_error: BaseException | None = None
        try:
            if self._connection.in_transaction:
                self._connection.rollback()
        except BaseException as exc:
            rollback_error = exc
        try:
            self._connection.close()
        except BaseException as exc:
            # Keep the lock descriptor open unless SQLite close is known to have completed.
            raise CatalogRefused("catalog_close_failed", "Could not safely close the catalog connection") from exc
        self._closed = True
        try:
            os.close(self._lock_fd)
        except OSError as exc:
            # A leaked descriptor is safer than releasing ownership around an open database handle.
            raise CatalogRefused("writer_lock_close_failed", "Could not close the catalog writer lock") from exc
        if rollback_error is not None:
            raise CatalogRefused(
                "catalog_rollback_failed", "Could not roll back pending catalog work"
            ) from rollback_error

    def __enter__(self) -> CatalogWriter:
        self._check_owner()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        try:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        finally:
            self.close()
        return False


def open_catalog(path: str | os.PathLike[str], *, single_writer: bool = True) -> CatalogWriter:
    """Open or initialize a WAL catalog while retaining its cross-process writer lock."""
    if type(single_writer) is not bool:
        raise TypeError("single_writer must be bool")
    version = _runtime_version()  # Read SQLITE_VERSION at call time for runtime policy overrides.
    if not sqlite_wal_safe(version) and not single_writer:
        raise CatalogRefused("wal_unpatched_sqlite", "Unpatched SQLite requires explicit single-writer isolation")
    database, lock_path = _resolve_paths(path, reader=False)
    lock_fd = _open_writer_lock(lock_path)
    connection: sqlite3.Connection | None = None
    try:
        _ensure_database_file(database)
        connection = sqlite3.connect(
            str(database), timeout=0, isolation_level=None, check_same_thread=True,
        )
        _configure_writer(connection)
        fresh = _classify_database(connection)
        _enable_wal(connection)
        if fresh:
            _initialize_schema(connection)
        else:
            _verify_schema(connection)
        writer = CatalogWriter(connection, lock_fd, lock_path)
        connection = None
        lock_fd = -1
        return writer
    except BaseException as exc:
        if lock_fd >= 0:
            _close_after_failed_open(connection, lock_fd)
        if isinstance(exc, CatalogRefused):
            raise
        if isinstance(exc, sqlite3.Error):
            raise CatalogRefused("catalog_unavailable", "Catalog could not be opened safely") from exc
        raise


def open_catalog_reader(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open an initialized catalog in read-only, query-only mode without repairing it."""
    _runtime_version()
    database, _lock_path = _resolve_paths(path, reader=True)
    uri = f"{database.as_uri()}?mode=ro"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=0, isolation_level=None, check_same_thread=True)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=0")
        query_only = connection.execute("PRAGMA query_only").fetchone()
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()
        if (query_only is None or query_only[0] != 1 or foreign_keys is None or foreign_keys[0] != 1
                or busy_timeout is None or busy_timeout[0] != 0):
            raise CatalogRefused("catalog_configuration_failed", "SQLite reader settings could not be enforced")
        application_row = connection.execute("PRAGMA application_id").fetchone()
        version_row = connection.execute("PRAGMA user_version").fetchone()
        if application_row is None or int(application_row[0]) != APPLICATION_ID:
            raise CatalogRefused("catalog_identity_mismatch", "Database application identity does not match")
        if version_row is None or int(version_row[0]) != SCHEMA_VERSION:
            raise CatalogRefused("catalog_schema_unsupported", "Catalog schema version is not supported")
        journal = connection.execute("PRAGMA journal_mode").fetchone()
        if journal is None or str(journal[0]).lower() != "wal":
            raise CatalogRefused("wal_unavailable", "Catalog is not in WAL mode")
        _verify_schema(connection)
        return connection
    except BaseException as exc:
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error as close_error:
                raise CatalogRefused(
                    "catalog_close_failed", "Could not safely close the catalog reader"
                ) from close_error
        if isinstance(exc, CatalogRefused):
            raise
        if isinstance(exc, sqlite3.Error):
            raise CatalogRefused("catalog_unavailable", "Catalog could not be opened read-only") from exc
        raise


__all__ = [
    "APPLICATION_ID", "CatalogRefused", "CatalogWriter", "SCHEMA_VERSION", "SQLITE_VERSION",
    "open_catalog", "open_catalog_reader", "sqlite_wal_safe",
]
