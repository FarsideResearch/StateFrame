"""Content-addressed, no-follow workspace effect accounting and restoration."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Literal, Mapping, Protocol


class EffectError(RuntimeError):
    """The workspace cannot be inspected or its pre-call state cannot be trusted."""


class EffectRestoreError(EffectError):
    def __init__(self, paths: list[str], details: list[str] | None = None):
        self.paths = tuple(sorted(set(paths)))
        message = "workspace restoration failed for: " + ", ".join(self.paths)
        if details:
            message += " (" + "; ".join(details) + ")"
        super().__init__(message)


class ContentStore:
    """Immutable SHA-256 objects in ``run_dir/cas/aa/rest-of-digest``."""

    def __init__(self, run_dir: Path):
        self.root = Path(run_dir) / "cas"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise EffectError(f"invalid CAS digest: {digest!r}")
        return self.root / digest[:2] / digest[2:]

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        target = self._path(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if self.get(digest) != data:
                raise EffectError(f"CAS object does not match digest {digest}")
            return digest
        fd, temporary = tempfile.mkstemp(prefix=".cas-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o444)
            # The store is private to this run; replacing an identical concurrent object is safe.
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return digest

    def get(self, digest: str) -> bytes:
        path = self._path(digest)
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as source:
                data = source.read()
        except OSError as error:
            raise EffectError(f"CAS object {digest} unavailable: {error}") from error
        if hashlib.sha256(data).hexdigest() != digest:
            raise EffectError(f"CAS object {digest} is corrupt")
        return data

    def capture(self, workspace: Path, manifest: Manifest) -> None:
        """Seed pre-call bytes; reject a workspace changed since its manifest was scanned."""
        for relative, entry in manifest.items():
            if entry.kind != "file":
                continue
            data = _read_file(Path(workspace) / relative)
            if hashlib.sha256(data).hexdigest() != entry.digest or len(data) != entry.size:
                raise EffectError(f"workspace changed while capturing {relative}")
            self.put(data)


@dataclass(frozen=True)
class Entry:
    kind: Literal["file", "symlink", "dir", "other"]
    mode: int
    digest: str | None
    size: int


Manifest = Mapping[str, Entry]


def _read_file(path: Path) -> bytes:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise EffectError(f"not a regular unlinked file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise EffectError(f"file changed while opening: {path}")
        with os.fdopen(fd, "rb", closefd=False) as source:
            return source.read()
    finally:
        os.close(fd)


def scan(workspace: Path, *, exclude: Callable[[str], bool]) -> Manifest:
    """Hash every regular file; classify hardlinks as unsupported special entries."""
    root = Path(workspace)
    if not stat.S_ISDIR(os.lstat(root).st_mode):
        raise EffectError(f"workspace is not a directory: {root}")
    result: dict[str, Entry] = {}

    def visit(directory: Path) -> None:
        with os.scandir(directory) as children:
            for child in children:
                relative = (directory / child.name).relative_to(root).as_posix()
                if exclude(relative):
                    continue
                info = child.stat(follow_symlinks=False)
                mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISDIR(info.st_mode):
                    result[relative] = Entry("dir", mode, None, 0)
                    visit(Path(child.path))
                elif stat.S_ISLNK(info.st_mode):
                    link = os.readlink(child.path).encode("utf-8", errors="surrogateescape")
                    result[relative] = Entry("symlink", mode, hashlib.sha256(link).hexdigest(), len(link))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    digest = hashlib.sha256()
                    fd = os.open(child.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    try:
                        opened = os.fstat(fd)
                        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                            raise EffectError(f"workspace changed while scanning {relative}")
                        with os.fdopen(fd, "rb", closefd=False) as source:
                            while block := source.read(1024 * 1024):
                                digest.update(block)
                    finally:
                        os.close(fd)
                    result[relative] = Entry("file", mode, digest.hexdigest(), opened.st_size)
                else:
                    result[relative] = Entry("other", mode, None, info.st_size)

    visit(root)
    return result


@dataclass(frozen=True)
class Effect:
    path: str
    change: Literal["created", "modified", "deleted", "mode", "type"]
    before: Entry | None
    after: Entry | None


def diff(before: Manifest, after: Manifest) -> list[Effect]:
    effects = []
    for path in sorted(before.keys() | after.keys()):
        old, new = before.get(path), after.get(path)
        if old == new:
            continue
        if old is None:
            change = "created"
        elif new is None:
            change = "deleted"
        elif old.kind != new.kind:
            change = "type"
        elif old.digest != new.digest or old.size != new.size:
            change = "modified"
        else:
            change = "mode"
        effects.append(Effect(path, change, old, new))
    return effects


class EffectPolicy(Protocol):
    def authorize(self, effect: Effect) -> tuple[bool, str]: ...


@dataclass(frozen=True)
class WorkspaceEffectPolicy:
    protected: Callable[[str], bool]
    edit_scope: Callable[[str], bool] | None
    new_file_allowed: Callable[[str], bool]
    baseline_manifest: Manifest | None = None

    def authorize(self, effect: Effect) -> tuple[bool, str]:
        if self.protected(effect.path):
            return False, "protected test or inode alias"
        if effect.after and effect.after.kind in {"symlink", "other"}:
            return False, "symlinks, hardlinks, and special files are not allowed"
        if self.edit_scope is not None and not self.edit_scope(effect.path):
            return False, "outside edit scope"
        if effect.before is None and not self.new_file_allowed(effect.path):
            return False, "new path is not allowed"
        return True, "authorized workspace effect"


def effect_policy_for_run(workspace: Path, row: Mapping[str, object],
                          baseline_tests: set[str], *,
                          baseline_manifest: Manifest | None = None) -> WorkspaceEffectPolicy:
    """Adapt the engine's own path decisions; keep their policy in one place."""
    from probes.react_repair import _is_product_path, protected_test_path

    root = Path(workspace)
    files = row.get("files")
    if not isinstance(files, Mapping):
        raise EffectError("run row has no product files mapping")
    editable = frozenset(str(path) for path in files)
    original_paths = frozenset((baseline_manifest or scan(root, exclude=lambda _: False)).keys())
    protected_inodes = set()
    for name in baseline_tests:
        try:
            info = os.lstat(name)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            protected_inodes.add((info.st_dev, info.st_ino))

    def is_protected(path: str) -> bool:
        candidate = root / path
        if protected_test_path(candidate, baseline_tests):
            return True
        try:
            info = os.lstat(candidate)
        except FileNotFoundError:
            return False
        return (info.st_dev, info.st_ino) in protected_inodes

    return WorkspaceEffectPolicy(
        protected=is_protected,
        edit_scope=lambda path: path in editable or (path not in original_paths and
                                                     _is_product_path(path, row)),
        new_file_allowed=lambda path: _is_product_path(path, row),
        baseline_manifest=baseline_manifest,
    )


def _safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {".", ".."} for part in pure.parts):
        raise EffectError(f"invalid workspace path: {relative!r}")
    current = root
    for part in pure.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise EffectError(f"symlink ancestor while restoring {relative}")
    return root.joinpath(*pure.parts)


def _remove(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode):
        with os.scandir(path) as children:
            for child in children:
                _remove(Path(child.path))
        os.rmdir(path)
    else:
        os.unlink(path)


def _restore(root: Path, relative: str, before: Manifest, store: ContentStore,
             baseline_store: ContentStore | None, policy: EffectPolicy) -> None:
    target = _safe_path(root, relative)
    old = before.get(relative)
    if old is None:
        _remove(target)
        return
    if old.kind not in {"file", "dir"}:
        raise EffectError(f"cannot restore pre-call {old.kind} at {relative}")
    if old.kind == "dir":
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            _remove(target)
        target.mkdir(parents=True, exist_ok=True)
        os.chmod(target, old.mode | stat.S_IWUSR | stat.S_IXUSR)
        with os.scandir(target) as children:
            for child in children:
                child_relative = f"{relative}/{child.name}"
                if child_relative not in before:
                    _remove(Path(child.path))
        descendants = sorted((p for p in before if p.startswith(relative + "/")),
                             key=lambda p: (p.count("/"), p))
        for child in descendants:
            _restore(root, child, before, store, baseline_store, policy)
        os.chmod(target, old.mode)
        return
    if old.digest is None:
        raise EffectError(f"pre-call file has no digest: {relative}")
    use_baseline = (isinstance(policy, WorkspaceEffectPolicy) and
                    policy.baseline_manifest is not None and policy.protected(relative))
    if use_baseline and baseline_store is None:
        raise EffectError(f"protected baseline store unavailable for {relative}")
    source = baseline_store if use_baseline else store
    baseline = (policy.baseline_manifest.get(relative) if isinstance(policy, WorkspaceEffectPolicy)
                and policy.baseline_manifest else None)
    entry = baseline if use_baseline and baseline is not None else old
    if entry.digest is None:
        raise EffectError(f"baseline file has no digest: {relative}")
    data = source.get(entry.digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or (target.exists() and target.is_dir()):
        _remove(target)
    fd, temporary = tempfile.mkstemp(prefix=".effect-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, entry.mode)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class Reconciliation:
    applied: list[Effect]
    restored: list[tuple[Effect, str]]
    manifest: Manifest


def reconcile(workspace: Path, before: Manifest, after: Manifest, policy: EffectPolicy,
              store: ContentStore, baseline_store: ContentStore | None) -> Reconciliation:
    effects = diff(before, after)
    decisions = [(effect, *policy.authorize(effect)) for effect in effects]
    denied = [(effect, reason) for effect, allowed, reason in decisions if not allowed]
    changed_paths = {effect.path for effect in effects}
    roots = []
    for effect, _ in sorted(denied, key=lambda item: (item[0].path.count("/"), item[0].path)):
        candidate = effect.path
        for parent in PurePosixPath(effect.path).parents:
            if str(parent) in changed_paths:
                candidate = str(parent)
        if candidate not in roots and not any(candidate.startswith(parent + "/") for parent in roots):
            roots.append(candidate)
    failures = []
    failure_details = []
    for path in roots:
        try:
            _restore(Path(workspace), path, before, store, baseline_store, policy)
        except (OSError, EffectError) as error:
            failures.append(path)
            failure_details.append(f"{path}: {error}")
    if failures:
        raise EffectRestoreError(failures, failure_details)
    included = before.keys() | after.keys()
    prefixes = {str(parent) for path in included for parent in PurePosixPath(path).parents
                if str(parent) != "."}
    final = scan(Path(workspace), exclude=lambda path: path not in included and path not in prefixes)
    for effect, _ in denied:
        expected = before.get(effect.path)
        if isinstance(policy, WorkspaceEffectPolicy) and policy.baseline_manifest and policy.protected(effect.path):
            expected = policy.baseline_manifest.get(effect.path, expected)
        if final.get(effect.path) != expected:
            failures.append(effect.path)
    if failures:
        raise EffectRestoreError(failures)
    store.capture(Path(workspace), final)
    applied = []
    for effect, allowed, _ in decisions:
        if allowed:
            if final.get(effect.path) == effect.after:
                applied.append(effect)
            else:
                denied.append((effect, "parent restoration also removed this effect"))
    return Reconciliation(
        applied=applied,
        restored=denied,
        manifest=final,
    )


def manifest_digest(manifest: Manifest) -> str:
    payload = [(path, asdict(entry)) for path, entry in sorted(manifest.items())]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
