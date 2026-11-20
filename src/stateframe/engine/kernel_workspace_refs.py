"""Authenticated workspace root registry and host-only run bindings."""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from kernel_request_auth_identity import _installation_home, _installation_secret
from kernel_workspace_objects import (
    MAX_OBJECT_BYTES,
    MAX_STREAM_BYTES,
    Digest,
    ObjectIntegrityError,
    ObjectLease,
    StoredRef,
    WorkspaceObjects,
    _child_directory,
    _directory,
    _lock,
    _open_namespace,
    _private_directory,
)

_COMPONENT = re.compile(r"[A-Za-z0-9_-]{1,128}")
_OID = re.compile(r"[0-9a-f]{64}")
_KINDS = frozenset(("tree", "node", "blob", "state", "meta", "record"))
_TEMP = re.compile(r"\.root-[0-9a-f]{32}")
_MAX_REF_BYTES = 4096
RUN_BINDING = ".workspace-objects.json"
Kind = Literal["tree", "node", "blob", "state", "meta", "record"]
Rel = tuple[bytes, ...]


class RootConflictError(ObjectIntegrityError):
    """The workspace root register changed since the caller observed it."""


@dataclass(frozen=True, slots=True)
class RootKey:
    run_id: str
    kind: str
    key: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and _COMPONENT.fullmatch(value)
                   for value in (self.run_id, self.kind, self.key)):
            raise ObjectIntegrityError("invalid workspace root key")


@dataclass(frozen=True, slots=True)
class Ref:
    kind: Kind
    oid: Digest

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _KINDS:
            raise ObjectIntegrityError("invalid workspace reference kind")
        if not isinstance(self.oid, str) or not _OID.fullmatch(self.oid):
            raise ObjectIntegrityError("invalid workspace reference digest")


def workspace_objects_enabled(env: Mapping[str, str]) -> bool:
    value = env.get("STATEFRAME_WORKSPACE_OBJECTS", "off").strip().lower()
    if value not in ("off", "on"):
        raise ValueError("STATEFRAME_WORKSPACE_OBJECTS must be off or on")
    return value == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _read_private(directory: int, name: str, limit: int, *, replaceable: bool = False) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(fd)
        links = (0, 1) if replaceable else (1,)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink not in links
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > limit):
            raise ObjectIntegrityError("invalid private workspace record")
        data = source.read(limit + 1)
        after = os.fstat(fd)
        if (len(data) != info.st_size or after.st_mtime_ns != info.st_mtime_ns
                or after.st_mode != info.st_mode or after.st_nlink not in links):
            raise ObjectIntegrityError("private workspace record changed while being read")
        return data


def _decode(data: bytes) -> object:
    try:
        value = json.loads(data)
        if _canonical(value) == data:
            return value
    except (ValueError, RecursionError):
        pass
    raise ObjectIntegrityError("invalid workspace record encoding")


def _key(home: Path) -> bytes:
    with _directory(home) as directory:
        _private_directory(directory)
        key = _read_private(directory, "identity.key", 32)
    if len(key) != 32:
        raise ObjectIntegrityError("invalid workspace signing key")
    return key


def _home(store: WorkspaceObjects) -> Path:
    parent = store._path.parent
    if parent.name == "1" and parent.parent.name == "workspace-state":
        return parent.parent.parent
    return _installation_home()


def _run_identity(directory: int) -> tuple[int, int]:
    info = os.fstat(directory)
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise ObjectIntegrityError("workspace run directory must be host-controlled")
    return info.st_dev, info.st_ino


def _signature(secret: bytes, domain: bytes, body: object) -> str:
    return hmac.digest(secret, domain + b"\0" + _canonical(body), hashlib.sha256).hex()


def _root_body(lease: ObjectLease, key: RootKey, value: Ref) -> list[str]:
    return [lease._store._path.name, key.run_id, key.kind, key.key, value.kind, value.oid]


@contextmanager
def _registry(lease: ObjectLease) -> Iterator[int]:
    lease._check()
    with _child_directory(lease._namespace, "refs") as directory:
        if _private_directory(directory) != lease._store._refs_identity:
            raise ObjectIntegrityError("workspace root registry was replaced")
        yield directory


@contextmanager
def _root_directory(lease: ObjectLease, key: RootKey, *, create: bool) -> Iterator[tuple[int, ...]]:
    with ExitStack() as stack:
        registry = stack.enter_context(_registry(lease))
        try:
            run = stack.enter_context(_child_directory(registry, key.run_id, create=create))
            _private_directory(run)
            kind = stack.enter_context(_child_directory(run, key.kind, create=create))
            _private_directory(kind)
        except FileNotFoundError:
            if create:
                raise
            yield ()
        else:
            yield kind, run, registry


@contextmanager
def _root_lock(directory: int, key: RootKey) -> Iterator[None]:
    name = ".lock-" + key.key
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(name, flags, 0o600, dir_fd=directory)
    except FileExistsError:
        fd = os.open(name, flags & ~(os.O_CREAT | os.O_EXCL), dir_fd=directory)
        created = False
    else:
        created = True
    try:
        if created:
            os.fchmod(fd, 0o600)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise ObjectIntegrityError("invalid workspace root lock")
        with _lock(fd, fcntl.LOCK_EX):
            yield
    finally:
        os.close(fd)


def _commit_bytes(directory: int, name: str, data: bytes) -> None:
    temporary = ".root-" + secrets.token_hex(16)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(fd)
        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=directory)


def _read_root(lease: ObjectLease, directory: int, key: RootKey, secret: bytes) -> Ref | None:
    try:
        data = _read_private(directory, key.key, _MAX_REF_BYTES, replaceable=True)
    except FileNotFoundError:
        return None
    document = _decode(data)
    if (not isinstance(document, list) or len(document) != 3
            or not all(isinstance(item, str) for item in document)):
        raise ObjectIntegrityError("invalid workspace root record")
    kind, oid, signature = document
    value = Ref(cast(Kind, kind), oid)
    expected = _signature(secret, b"workspace-root/1", _root_body(lease, key, value))
    if not signature.isascii() or not hmac.compare_digest(signature, expected):
        raise ObjectIntegrityError("workspace root authentication failed")
    return value


def read_root(lease: ObjectLease, key: RootKey) -> Ref | None:
    with _root_directory(lease, key, create=False) as directories:
        if not directories:
            return None
        return _read_root(lease, directories[0], key, _key(_home(lease._store)))


def commit_root(lease: ObjectLease, key: RootKey, value: Ref, *, expected: Ref | None) -> None:
    lease._check(write=True)
    secret = _key(_home(lease._store))
    # Validate the immediate object and recover any unfinished object-directory
    # sync before allowing its name to become retention authority.
    stored = lease.resolve(value.oid, max_bytes=MAX_STREAM_BYTES if value.kind == "blob"
                           else MAX_OBJECT_BYTES)
    directory = lease._check()
    with _lock(directory, fcntl.LOCK_EX):
        try:
            os.stat(".stage-" + stored.digest, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            from kernel_workspace_objects import _read_checked

            _read_checked(directory, stored)
            os.fsync(directory)
            os.unlink(".stage-" + stored.digest, dir_fd=directory)
    if value.kind == "tree":
        _validate_tree_root(lease, stored)
    with _root_directory(lease, key, create=True) as directories, _root_lock(directories[0], key):
        current = _read_root(lease, directories[0], key, secret)
        if current != value:
            if current != expected:
                raise RootConflictError("workspace root has a different value")
            signature = _signature(secret, b"workspace-root/1", _root_body(lease, key, value))
            _commit_bytes(directories[0], key.key, _canonical([value.kind, value.oid, signature]))
        for fd in directories:
            os.fsync(fd)


def _validate_tree_root(lease: ObjectLease, stored: StoredRef) -> None:
    from kernel_workspace_objects import ObjectRef
    from kernel_workspace_tree import _DOMAIN, _field

    if not isinstance(stored, ObjectRef) or stored.size > 128:
        raise ObjectIntegrityError("invalid workspace tree root size")
    prefix = _DOMAIN + _field(b"dir")
    data = lease.read(stored)
    rest = data[len(prefix):]
    if (not data.startswith(prefix) or len(rest) not in (16, 48)
            or rest[:4] != b"\x00\x00\x00\x08"
            or int.from_bytes(rest[4:12], "big") > 0o7777
            or int.from_bytes(rest[12:16], "big") != len(rest) - 16):
        raise ObjectIntegrityError("invalid workspace tree root preimage")


def retire_root(lease: ObjectLease, key: RootKey, *, expected: Ref) -> None:
    lease._check(write=True)
    with _root_directory(lease, key, create=False) as directories:
        if not directories:
            raise RootConflictError("workspace root is missing")
        with _root_lock(directories[0], key):
            current = _read_root(lease, directories[0], key, _key(_home(lease._store)))
            if current != expected:
                raise RootConflictError("workspace root has a different value")
            os.unlink(key.key, dir_fd=directories[0])
            os.fsync(directories[0])


def _component(name: str) -> str:
    if not _COMPONENT.fullmatch(name):
        raise ObjectIntegrityError("invalid name in workspace root registry")
    return name


def iter_roots(lease: ObjectLease) -> Iterator[tuple[RootKey, Ref]]:
    """Enumerate an authenticated registry while the namespace is exclusively held."""
    if lease._writable:
        raise RuntimeError("workspace root enumeration requires an exclusive namespace lease")
    with _registry(lease) as registry:
        # A retirement interrupted after unlink but before its fsync must become
        # durable before collection can remove the objects that it used to pin.
        os.fsync(registry)
        secret = None
        for run_id in os.listdir(registry):
            with _child_directory(registry, _component(run_id)) as run:
                _private_directory(run)
                os.fsync(run)
                for kind in os.listdir(run):
                    with _child_directory(run, _component(kind)) as directory:
                        _private_directory(directory)
                        os.fsync(directory)
                        for name in os.listdir(directory):
                            if _TEMP.fullmatch(name):
                                continue
                            if name.startswith(".lock-") and _COMPONENT.fullmatch(name[6:]):
                                continue
                            root = RootKey(run_id, kind, name)
                            if secret is None:
                                secret = _key(_home(lease._store))
                            ref = _read_root(lease, directory, root, secret)
                            if ref is None:
                                raise ObjectIntegrityError("workspace root disappeared during collection")
                            yield root, ref


def _edges(lease: ObjectLease, ref: Ref, stored: StoredRef) -> Iterator[Ref]:
    if ref.kind == "blob":
        return
    if ref.kind in ("tree", "node"):
        from kernel_workspace_tree_io import child_refs

        yield from child_refs(lease, ref)
        return
    if ref.kind == "meta":
        from kernel_workspace_objects import ObjectRef

        if isinstance(stored, ObjectRef):
            document = _decode(lease.read(stored))
            schema = document.get("schema") if isinstance(document, dict) else None
            if isinstance(schema, str) and schema.startswith("stateframe-restore-"):
                from kernel_workspace_tree_io import metadata_edges

                yield from metadata_edges(lease, ref)
                return
    from kernel_workspace_checkpoint import record_edges

    yield from record_edges(lease, ref)


def prune_root_stages(lease: ObjectLease) -> None:
    """Remove abandoned writes and retired locks after a validated registry sweep."""
    if lease._writable:
        raise RuntimeError("workspace root cleanup requires an exclusive namespace lease")
    with _registry(lease) as registry:
        for run_id in os.listdir(registry):
            with _child_directory(registry, _component(run_id)) as run:
                _private_directory(run)
                for kind in os.listdir(run):
                    with _child_directory(run, _component(kind)) as directory:
                        _private_directory(directory)
                        names = set(os.listdir(directory))
                        abandoned = [name for name in names if _TEMP.fullmatch(name)
                                     or (name.startswith(".lock-") and name[6:] not in names)]
                        for name in abandoned:
                            os.unlink(name, dir_fd=directory)
                        if abandoned:
                            os.fsync(directory)


def registered_objects(lease: ObjectLease) -> Iterator[StoredRef]:
    """Expand typed roots. A missing or malformed edge prevents any sweep."""
    seen: set[tuple[Kind, Digest]] = set()
    active: set[tuple[Kind, Digest]] = set()
    for _, root in iter_roots(lease):
        stack = [iter((root,))]
        ancestors: list[tuple[Kind, Digest]] = []
        while stack:
            ref = next(stack[-1], None)
            if ref is None:
                stack.pop()
                if ancestors:
                    identity = ancestors.pop()
                    active.remove(identity)
                    seen.add(identity)
                continue
            identity = ref.kind, ref.oid
            if identity in active:
                raise ObjectIntegrityError("cycle in workspace root graph")
            if identity in seen:
                continue
            stored = lease.resolve(ref.oid, max_bytes=MAX_STREAM_BYTES if ref.kind == "blob"
                                   else MAX_OBJECT_BYTES)
            yield stored
            active.add(identity)
            ancestors.append(identity)
            stack.append(iter(_edges(lease, ref, stored)))


def _create_store(home: Path, namespace: str) -> WorkspaceObjects:
    with _directory(home) as home_fd:
        _private_directory(home_fd)
        with _child_directory(home_fd, "workspace-state", create=True) as state:
            _private_directory(state)
            with _child_directory(state, "1", create=True) as version:
                _private_directory(version)
                store = _open_namespace(home / "workspace-state" / "1", namespace, create=True)
            os.fsync(state)
        os.fsync(home_fd)
    return store


def bind_run_store(run: Path, namespace: str, *, run_id: str | None = None) -> WorkspaceObjects:
    """Bind an owner-private run directory to a host-selected workspace namespace."""
    if not isinstance(namespace, str) or not _OID.fullmatch(namespace):
        raise ObjectIntegrityError("invalid run workspace namespace")
    run = Path(run).absolute()
    run_id = _component(run.name if run_id is None else run_id)
    home = _installation_home()
    # Use the established key initialization, then require its durable file.
    _installation_secret(home)
    secret = _key(home)
    store = _create_store(home, namespace)
    with _directory(run) as directory:
        identity = _run_identity(directory)
        body = [namespace, run_id]
        data = _canonical([*body, _signature(secret, b"workspace-run/1", [*identity, *body])])
        with _lock(directory, fcntl.LOCK_EX):
            try:
                existing = _read_private(directory, RUN_BINDING, _MAX_REF_BYTES)
            except FileNotFoundError:
                _commit_bytes(directory, RUN_BINDING, data)
            else:
                if existing != data:
                    raise RootConflictError("run has a different workspace binding")
            os.fsync(directory)
    return store


def _run_binding(run: Path) -> tuple[Path, str, str]:
    run = Path(run).absolute()
    with _directory(run) as directory:
        identity = _run_identity(directory)
        record = _decode(_read_private(directory, RUN_BINDING, _MAX_REF_BYTES))
    if (not isinstance(record, list) or len(record) != 3
            or not all(isinstance(value, str) for value in record)):
        raise ObjectIntegrityError("invalid run workspace binding")
    namespace, run_id, signature = record
    if not _OID.fullmatch(namespace) or not _COMPONENT.fullmatch(run_id):
        raise ObjectIntegrityError("invalid run workspace binding")
    home = _installation_home()
    expected = _signature(_key(home), b"workspace-run/1", [*identity, namespace, run_id])
    if not signature.isascii() or not hmac.compare_digest(signature, expected):
        raise ObjectIntegrityError("run workspace binding authentication failed")
    return home, namespace, run_id


def open_run_store(run: Path) -> WorkspaceObjects:
    home, namespace, _ = _run_binding(run)
    return _open_namespace(home / "workspace-state" / "1", namespace, create=False)


def workspace_run_id(run: Path) -> str:
    return _run_binding(run)[2]
