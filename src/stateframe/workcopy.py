"""Safe working copy of the user's repository.

Every run works on a private copy under the run directory. The original files are only
written when the user passes --in-place, and then only files the run changed, and only if
the original still has the bytes it had when the run started (no clobbering of edits the
user made meanwhile).
"""
from __future__ import annotations

import difflib
import fnmatch
import hashlib
import os
import secrets
import shutil
import stat
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

# Never copied: VCS metadata, environments, caches, build output.
IGNORE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
               ".tox", ".nox", ".venv", "venv", "env", ".env", "node_modules", "dist", "build",
               ".idea", ".vscode", ".eggs", "site-packages", ".stateframe"}
IGNORE_GLOBS = ("*.pyc", "*.pyo", "*.so", "*.dylib", "*.egg-info", ".DS_Store")
# Secrets never enter the copy, so they can never reach a request.
SECRET_GLOBS = (".env", ".env.*", "*.pem", "*.key", "id_rsa*", "*.p12", "*.pfx", ".netrc",
                ".npmrc", ".pypirc", "credentials*", "*secret*")


class WorkCopyError(RuntimeError):
    pass


@dataclass
class Limits:
    max_files: int = 5000
    max_file_bytes: int = 1024 * 1024
    max_total_bytes: int = 64 * 1024 * 1024


@dataclass
class WorkCopy:
    original: Path
    copy: Path
    baseline: dict[str, str] = field(default_factory=dict)  # relative path -> sha256 at start
    skipped: list[str] = field(default_factory=list)
    snapshot: Path | None = None  # immutable copy of the start state (diff baseline), if the run made one

    def text_files(self) -> dict[str, str]:
        """Every UTF-8 text file in the copy, keyed by POSIX relative path."""
        out: dict[str, str] = {}
        for rel in sorted(self.baseline):
            try:
                out[rel] = (self.copy / rel).read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
        return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ignored(name: str, is_dir: bool) -> bool:
    if is_dir:
        return name in IGNORE_DIRS or name.endswith(".egg-info")
    return any(fnmatch.fnmatch(name, pattern) for pattern in IGNORE_GLOBS + SECRET_GLOBS)


def frame_source_path_allowed(path: str) -> bool:
    """Use the copy's exclusion rules for source added after the copy was made."""
    parts = path.split("/") if isinstance(path, str) else []
    return bool(parts and all(part not in {"", ".", ".."} for part in parts)
                and not any(_ignored(part, True) for part in parts[:-1])
                and not _ignored(parts[-1], False))


def create(original: Path, destination: Path, limits: Limits | None = None) -> WorkCopy:
    """Copy ``original`` into ``destination`` (which must not exist), recording hashes."""
    limits = limits or Limits()
    original = original.resolve()
    if not original.is_dir():
        raise WorkCopyError(f"--repo is not a directory: {original}")
    destination = destination.resolve()
    if destination.exists():
        raise WorkCopyError(f"working copy destination already exists: {destination}")
    if destination.is_relative_to(original):
        raise WorkCopyError("run directory must be outside the repository")
    work = WorkCopy(original=original, copy=destination)
    total = 0
    destination.mkdir(parents=True)
    for directory, dirnames, filenames in os.walk(original, followlinks=False):
        here = Path(directory)
        dirnames[:] = sorted(d for d in dirnames
                             if not _ignored(d, True) and not (here / d).is_symlink())
        for name in sorted(filenames):
            source = here / name
            rel = source.relative_to(original).as_posix()
            if _ignored(name, False) or source.is_symlink():
                work.skipped.append(rel)
                continue
            size = source.stat().st_size
            if size > limits.max_file_bytes:
                work.skipped.append(rel)
                continue
            total += size
            if len(work.baseline) >= limits.max_files or total > limits.max_total_bytes:
                raise WorkCopyError(
                    f"repository is larger than the run limits ({limits.max_files} files, "
                    f"{limits.max_total_bytes // (1024 * 1024)} MB); point --repo at a smaller tree")
            data = source.read_bytes()
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            shutil.copymode(source, target)
            work.baseline[rel] = _sha(data)
    if not work.baseline:
        raise WorkCopyError("repository has no files to work on")
    return work


def changed_files(work: WorkCopy) -> dict[str, str]:
    """relative path -> 'modified' | 'added' | 'deleted' in the copy versus the baseline."""
    changes: dict[str, str] = {}
    current: dict[str, str] = {}
    for path in work.copy.rglob("*"):
        if path.is_file() and not path.is_symlink():
            rel = path.relative_to(work.copy).as_posix()
            if any(part in IGNORE_DIRS for part in Path(rel).parts[:-1]) or _ignored(path.name, False):
                continue
            current[rel] = _sha(path.read_bytes())
    for rel, digest in current.items():
        if rel not in work.baseline:
            changes[rel] = "added"
        elif work.baseline[rel] != digest:
            changes[rel] = "modified"
    for rel in work.baseline:
        if rel not in current:
            changes[rel] = "deleted"
    return dict(sorted(changes.items()))


def unified_diff(work: WorkCopy) -> str:
    """Diff of the copy against the original repository's bytes at run start."""
    chunks = []
    for rel, kind in changed_files(work).items():
        before = _read_original(work, rel) if kind != "added" else ""
        after = (work.copy / rel).read_text(encoding="utf-8", errors="replace") if kind != "deleted" else ""
        chunks.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{rel}" if kind != "added" else "/dev/null",
            tofile=f"b/{rel}" if kind != "deleted" else "/dev/null"))
        if chunks and not chunks[-1].endswith("\n"):
            chunks[-1] += "\n\\ No newline at end of file\n"
    return "".join(chunks)


def _read_original(work: WorkCopy, rel: str) -> str:
    """Baseline text, from the immutable start snapshot when there is one (else the original,
    which must still hash to its baseline)."""
    path = (work.snapshot if work.snapshot is not None else work.original) / rel
    data = path.read_bytes() if path.is_file() else b""
    if _sha(data) != work.baseline.get(rel):
        raise WorkCopyError(f"original {rel} changed during the run; diff would be wrong")
    return data.decode("utf-8", errors="replace")


def apply_files(work: WorkCopy, files: dict[str, bytes | None], *, declared: set[str]) -> list[str]:
    """Apply the sealed candidate's writes and deletions to the copy.

    Only paths the run declared to the engine (or new files under the copy) are written,
    each must stay inside the copy, and symlinks are refused.
    """
    changes: dict[str, bytes | None] = {}
    baseline: dict[str, str] = {}
    for rel, data in sorted(files.items()):
        _relative_parts(rel)
        if rel not in declared and rel in work.baseline:
            continue
        # Test/tool litter (caches, bytecode, junk) is never product output, even if the engine captured it.
        if any(_ignored(part, True) for part in Path(rel).parts[:-1]) or _ignored(Path(rel).name, False):
            continue
        try:
            with _parent_fd(work.copy, rel) as parent:
                current, _ = _read_at(parent, Path(rel).name)
        except FileNotFoundError:
            current = None
        except OSError as error:
            raise WorkCopyError(f"cannot read candidate destination {rel}: {error}") from error
        if current == data:
            continue
        if current is not None:
            baseline[rel] = _sha(current)
        changes[rel] = data
    return guarded_apply(work.copy, changes, baseline, what="candidate apply")


def _relative_parts(rel: str) -> tuple[str, ...]:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise WorkCopyError(f"path escapes the repository: {rel}")
    return path.parts


@contextmanager
def _parent_fd(root: Path, rel: str, *, create: bool = False):
    """Walk every component without resolving links; anchor writes to an open directory."""
    parts = _relative_parts(rel)
    root = root.absolute()  # absolute is lexical; resolve would hide symlinks
    if ".." in root.parts:
        raise WorkCopyError(f"parent traversal in repository root: {root}")
    descriptor = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in (*root.parts[1:], *parts[:-1]):
            try:
                info = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass  # the no-follow open below validates a concurrent creation
            else:
                if stat.S_ISLNK(info.st_mode):
                    raise WorkCopyError(f"a parent directory is a symlink: {rel}")
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _read_at(parent: int, name: str) -> tuple[bytes | None, int | None]:
    """Read only regular files, refusing dangling links and races at the open itself."""
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None, None
    if stat.S_ISLNK(info.st_mode):
        raise WorkCopyError(f"destination is a symlink: {name}")
    if not stat.S_ISREG(info.st_mode):
        raise WorkCopyError(f"destination is not a regular file: {name}")
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise WorkCopyError(f"destination is not a regular file: {name}")
        return source.read(), stat.S_IMODE(opened.st_mode)


def _check_bytes(current: bytes | None, baseline_sha: str | None) -> str | None:
    if baseline_sha is None:
        return "was created in the original meanwhile" if current is not None else None
    if current is None or _sha(current) != baseline_sha:
        return "changed in the original since the start"
    return None


def _conflict(original: Path, rel: str, baseline_sha: str | None) -> str | None:
    """Why writing ``rel`` into ``original`` would clobber something, or None if it is safe."""
    try:
        with _parent_fd(original, rel) as parent:
            current, _ = _read_at(parent, Path(rel).name)
    except FileNotFoundError:
        current = None
    except (OSError, WorkCopyError) as error:
        return str(error)
    return _check_bytes(current, baseline_sha)


def guarded_apply(original: Path, changes: dict[str, bytes | None], baseline: dict[str, str],
                  *, what: str) -> list[str]:
    """Write ``changes`` (rel -> new bytes, or None = delete) into ``original``.

    Every path is validated against ``baseline`` (sha at start; absent = new file) BEFORE anything is
    written, and re-checked right before its own write. Writes are atomic per file (temp + rename).
    The whole set is not atomic: on an I/O error the files already written stay written and the error
    names them; nothing is rolled back destructively.
    """
    conflicts = [f"{rel} ({why})" for rel in sorted(changes)
                 if (why := _conflict(original, rel, baseline.get(rel)))]
    if conflicts:
        raise WorkCopyError(f"refusing {what}: " + "; ".join(conflicts[:10]))
    applied: list[str] = []
    # For a rename, install the new path before deleting the old path. A later
    # I/O failure can then leave a duplicate, but cannot lose the only copy.
    for rel in sorted(changes, key=lambda path: (changes[path] is None, path)):
        why = _conflict(original, rel, baseline.get(rel))
        if why:
            raise WorkCopyError(f"refusing {what} at {rel} ({why}); already applied: {applied or 'nothing'}")
        data = changes[rel]
        name = Path(rel).name
        try:
            with _parent_fd(original, rel, create=data is not None) as parent:
                current, mode = _read_at(parent, name)
                why = _check_bytes(current, baseline.get(rel))
                if why:
                    raise WorkCopyError(why)
                if data is None:
                    if current is not None:
                        os.unlink(name, dir_fd=parent)
                else:
                    # O_EXCL prevents following or overwriting any pre-existing temp file.
                    temp = f".stateframe-{secrets.token_hex(16)}.tmp"
                    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                         0o666, dir_fd=parent)
                    try:
                        with os.fdopen(descriptor, "wb") as target:
                            if mode is not None:
                                os.fchmod(target.fileno(), mode)
                            target.write(data)
                        # Recheck ancestors and the leaf after staging, immediately before replace.
                        why = _conflict(original, rel, baseline.get(rel))
                        if why:
                            raise WorkCopyError(why)
                        current, _ = _read_at(parent, name)
                        why = _check_bytes(current, baseline.get(rel))
                        if why:
                            raise WorkCopyError(why)
                        os.replace(temp, name, src_dir_fd=parent, dst_dir_fd=parent)
                    finally:
                        try:
                            os.unlink(temp, dir_fd=parent)
                        except FileNotFoundError:
                            pass
        except (OSError, WorkCopyError) as error:
            raise WorkCopyError(f"{what} stopped at {rel}: {error}; already applied: {applied or 'nothing'}") \
                from error
        applied.append(rel)
    return applied


def apply_in_place(work: WorkCopy) -> list[str]:
    """Copy changed files back to the original repository, guarded against concurrent edits."""
    changes = {rel: (None if kind == "deleted" else (work.copy / rel).read_bytes())
               for rel, kind in changed_files(work).items()}
    return guarded_apply(work.original, changes, work.baseline, what="--in-place")
