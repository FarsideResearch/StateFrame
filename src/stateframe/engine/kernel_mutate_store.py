"""Locked, content-addressed retained mutation bundles and references."""
from __future__ import annotations

import fcntl
import base64
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .kernel_mutate_wire import strict_loads
    from .kernel_mutate_sweep import _remove_tree
else:
    from kernel_mutate_wire import strict_loads
    from kernel_mutate_sweep import _remove_tree


class StoreError(ValueError):
    pass


def project_id(repo) -> str:
    path = Path(repo).resolve(strict=True)
    result = subprocess.run(["git", "-C", str(path), "rev-list", "--max-parents=0", "HEAD"],
                            capture_output=True, text=True, timeout=5, check=False)
    first = result.stdout.splitlines()[0] if result.returncode == 0 and result.stdout.strip() else ""
    return hashlib.sha256((str(path) + "\n" + first).encode()).hexdigest()


class MutationStore:
    def __init__(self, home):
        self.root = Path(home) / "mutations"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._directory(self.root)
        self.objects = self.root / "objects"
        self.objects.mkdir(exist_ok=True, mode=0o700)
        self._directory(self.objects)
        self.index = self.root / "index"
        self.index.mkdir(exist_ok=True, mode=0o700)
        self._directory(self.index)
        (self.index / "projects").mkdir(exist_ok=True, mode=0o700)
        self._directory(self.index / "projects")
        self.leases = self.root / "leases"
        self.leases.mkdir(exist_ok=True, mode=0o700)
        self._directory(self.leases)
        self.lock_path = self.root / "store.lock"

    @staticmethod
    def _directory(path):
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise StoreError("invalid store directory")

    @contextmanager
    def _locked(self):
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _digest(digest):
        if not isinstance(digest, str) or re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", digest) is None:
            raise StoreError("invalid source digest")
        return digest.removeprefix("sha256:")

    @staticmethod
    def _bundle(bundle):
        if not isinstance(bundle, dict) or not 2 <= len(bundle) <= 9 or "unit.json" not in bundle or any(
                not isinstance(name, str) or re.fullmatch(r"(?:unit\.json|[a-z0-9_]{1,40}\.py)", name) is None
                or not isinstance(data, bytes) for name, data in bundle.items()):
            raise StoreError("invalid bundle")
        if sum(len(data) for data in bundle.values()) > 64 << 10:
            raise StoreError("bundle too large")

    @staticmethod
    def _fsync_dir(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def publish(self, digest, bundle, *, scope=None):
        digest = self._digest(digest)
        self._bundle(bundle)
        index_path = self._index_path(scope) if scope is not None else None
        try:
            if __package__:
                from .kernel_mutate_build import source_digest
            else:
                from kernel_mutate_build import source_digest
            manifest = strict_loads(bundle["unit.json"])
            files = {name: data for name, data in bundle.items() if name != "unit.json"}
            if source_digest(files, manifest) != "sha256:" + digest:
                raise StoreError("source digest mismatch")
        except (KeyError, ValueError, TypeError) as error:
            raise StoreError("source digest mismatch") from error
        target = self.objects / digest
        with self._locked():
            if target.exists():
                if self._get_unlocked(digest) != bundle:
                    raise StoreError("conflicting source digest")
                if index_path is not None:
                    self._atomic_index(index_path, [*self._read_index(index_path), digest])
                return target
            temp = Path(tempfile.mkdtemp(prefix=".publish-", dir=self.objects))
            try:
                for name, data in bundle.items():
                    fd = os.open(temp / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                    try:
                        os.write(fd, data)
                        os.fsync(fd)
                        os.fchmod(fd, 0o400)
                    finally:
                        os.close(fd)
                self._fsync_dir(temp)
                os.chmod(temp, 0o500)
                os.rename(temp, target)
                self._fsync_dir(self.objects)
                if index_path is not None:
                    self._atomic_index(index_path, [*self._read_index(index_path), digest])
            finally:
                if temp.exists():
                    os.chmod(temp, 0o700)
                    shutil.rmtree(temp)
        return target

    def _get_unlocked(self, digest):
        target = self.objects / digest
        if not target.is_dir() or target.is_symlink():
            raise StoreError("missing or invalid object")
        result = {}
        for path in target.iterdir():
            if path.is_symlink() or not path.is_file():
                raise StoreError("invalid object member")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 64 << 10:
                    raise StoreError("invalid object member")
                data = bytearray()
                while len(data) < info.st_size:
                    block = os.read(fd, info.st_size - len(data))
                    if not block:
                        raise StoreError("object changed during read")
                    data.extend(block)
                result[path.name] = bytes(data)
            finally:
                os.close(fd)
        self._bundle(result)
        return result

    def get(self, digest, verify=True):
        digest = self._digest(digest)
        with self._locked():
            bundle = self._get_unlocked(digest)
        if verify and "unit.json" in bundle:
            try:
                if __package__:
                    from .kernel_mutate_build import source_digest
                else:
                    from kernel_mutate_build import source_digest
                manifest = strict_loads(bundle["unit.json"])
                files = {name: data for name, data in bundle.items() if name != "unit.json"}
                if source_digest(files, manifest) != "sha256:" + digest:
                    raise StoreError("source digest mismatch")
            except (ValueError, TypeError) as error:
                raise StoreError("source digest mismatch") from error
        return bundle

    def _index_path(self, scope):
        if scope == "user":
            return self.index / "user.json"
        if isinstance(scope, str) and re.fullmatch(r"[0-9a-f]{64}", scope):
            return self.index / "projects" / (scope + ".json")
        raise StoreError("invalid index scope")

    def _read_index(self, path):
        if not path.exists() and not path.is_symlink():
            return []
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1 << 20:
                    raise StoreError("invalid index file")
                data = os.read(fd, (1 << 20) + 1)
            finally:
                os.close(fd)
            value = json.loads(data)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise StoreError("invalid index file") from error
        if not isinstance(value, list) or any(not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in value):
            raise StoreError("invalid index")
        return value

    def _atomic_index(self, path, rows):
        if path.exists() or path.is_symlink():
            if not stat.S_ISREG(path.lstat().st_mode):
                raise StoreError("invalid index target")
        fd, name = tempfile.mkstemp(prefix=".index-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(json.dumps(sorted(set(rows)), separators=(",", ":")).encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
            self._fsync_dir(path.parent)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def index_add(self, scope, digest):
        digest = self._digest(digest)
        path = self._index_path(scope)
        with self._locked():
            self._get_unlocked(digest)
            self._atomic_index(path, [*self._read_index(path), digest])

    def index_remove(self, scope, digest):
        digest = self._digest(digest)
        path = self._index_path(scope)
        with self._locked():
            self._atomic_index(path, [row for row in self._read_index(path) if row != digest])

    @contextmanager
    def lease(self, run_uuid, digests):
        if not isinstance(run_uuid, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_uuid) is None:
            raise StoreError("invalid run id")
        rows = [self._digest(item) for item in digests]
        path = self.leases / run_uuid
        with self._locked():
            if path.exists():
                raise StoreError("lease already exists")
            self._atomic_index(path, rows)
            lease_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield path
        finally:
            try:
                with self._locked():
                    path.unlink(missing_ok=True)
                    self._fsync_dir(self.leases)
            finally:
                fcntl.flock(lease_fd, fcntl.LOCK_UN)
                os.close(lease_fd)

    def gc(self, *, powerup_refs=(), dry_run=False):
        with self._locked():
            roots = {self._digest(item) for item in powerup_refs}
            for path in [self.index / "user.json", *self.index.joinpath("projects").glob("*.json")]:
                roots.update(self._read_index(path))
            for path in self.leases.iterdir():
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        roots.update(self._read_index(path))
                    else:
                        if not dry_run:
                            path.unlink()
                            self._fsync_dir(self.leases)
                finally:
                    os.close(fd)
            removed = []
            objects_fd = os.open(self.objects, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                for name in os.listdir(objects_fd):
                    if name.startswith(".publish-"):
                        continue
                    digest = self._digest(name)
                    info = os.stat(name, dir_fd=objects_fd, follow_symlinks=False)
                    if not stat.S_ISDIR(info.st_mode):
                        raise StoreError("retained object is not a directory")
                    self._get_unlocked(digest)
                    if digest in roots:
                        continue
                    if not dry_run:
                        object_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                            dir_fd=objects_fd)
                        try:
                            current = os.fstat(object_fd)
                            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                                raise StoreError("retained object identity changed before GC")
                            _remove_tree(object_fd, parent_fd=objects_fd, name=name,
                                         inode=(info.st_dev, info.st_ino))
                        finally:
                            os.close(object_fd)
                    removed.append(digest)
            finally:
                os.close(objects_fd)
            if not dry_run:
                self._fsync_dir(self.objects)
            return removed


def _store(store_root):
    root = Path(store_root)
    if root.name != "mutations":
        raise StoreError("invalid mutation store root")
    return MutationStore(root.parent)


def _unit_row(digest, bundle, origin):
    manifest = strict_loads(bundle["unit.json"])
    if not isinstance(manifest, dict):
        raise StoreError("invalid retained unit manifest")
    return {"name": manifest["name"], "revision": manifest["revision"],
            "digest": "sha256:" + digest.removeprefix("sha256:"),
            "summary": manifest["summary"], "kind": manifest["kind"],
            "origin": origin, "input_schema": manifest["input_schema"],
            "output_schema": manifest["output_schema"], "requires": manifest["requires"]}


def _run_units(run):
    if __package__:
        from .kernel_mutate_records import Journal
    else:
        from kernel_mutate_records import Journal
    journal = Journal(Path(run))
    units = {}
    for row in journal.events():
        if row.get("kind") == "ADMISSION" and isinstance(row.get("blob"), str):
            value = strict_loads(journal.get_blob(row["blob"]))
            files = {name: base64.b64decode(source, validate=True)
                     for name, source in value["files"].items()}
            bundle = {"unit.json": json.dumps(value["manifest"], sort_keys=True,
                                             separators=(",", ":")).encode(), **files}
            units[value["manifest"]["name"]] = (row["source_digest"], bundle)
    return units


def list_units(store_root, *, scope=None, run=None):
    if run is not None:
        return [_unit_row(digest, bundle, "run") for digest, bundle in _run_units(run).values()]
    store = _store(store_root)
    index_scope = project_id(Path.cwd()) if scope == "project" else scope
    if index_scope is None:
        raise StoreError("scope or run is required")
    with store._locked():
        digests = store._read_index(store._index_path(index_scope))
    return [_unit_row(digest, store.get(digest), scope) for digest in digests]


def show_unit(store_root, name):
    store = _store(store_root)
    for scope in ("user", project_id(Path.cwd())):
        with store._locked():
            digests = store._read_index(store._index_path(scope))
        for digest in digests:
            row = _unit_row(digest, store.get(digest), scope)
            if row["name"] == name:
                return row
    return None


def retain_from_run(store_root, run, name, scope):
    units = _run_units(run)
    if name not in units:
        raise StoreError("unit is not present in run admission evidence")
    digest, bundle = units[name]
    store = _store(store_root)
    index_scope = project_id(Path.cwd()) if scope == "project" else scope
    return store.publish(digest, bundle, scope=index_scope)


def remove_unit(store_root, name, scope):
    store = _store(store_root)
    index_scope = project_id(Path.cwd()) if scope == "project" else scope
    with store._locked():
        digests = store._read_index(store._index_path(index_scope))
    for digest in digests:
        if _unit_row(digest, store.get(digest), scope)["name"] == name:
            store.index_remove(index_scope, digest)
            return "sha256:" + digest
    raise StoreError("unit is not retained in this scope")


def gc(store_root, *, dry_run=False):
    if dry_run:
        root = Path(store_root)
        if not root.exists():
            return []
        required = (root, root / "objects", root / "index", root / "index" / "projects",
                    root / "leases")
        if (not all(path.is_dir() and not path.is_symlink() for path in required)
                or not (root / "store.lock").is_file()
                or (root / "store.lock").is_symlink()):
            raise StoreError("dry-run requires an intact existing store; no files were changed")
    return _store(store_root).gc(dry_run=dry_run)
