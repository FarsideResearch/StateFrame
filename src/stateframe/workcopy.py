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
import sys
import tokenize
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .engine.kernel_edit_merge import ConflictHunk, MergeApplied, MergeConflict, merge_texts
from .engine.kernel_secret_paths import SECRET_DIRS, is_secret_file

# Bounds for the conflict report shown to the user (COLLAB-DRIFT piece F).
MAX_HUNKS_PER_FILE = 5
MAX_HUNK_LINES = 40
MAX_CONFLICT_FILES_SHOWN = 10

# Never copied: VCS metadata, environments, caches, build output.
IGNORE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
               ".tox", ".nox", ".venv", "venv", "env", "node_modules", "dist", "build",
               ".idea", ".vscode", ".eggs", "site-packages", ".stateframe"} | SECRET_DIRS
IGNORE_GLOBS = ("*.pyc", "*.pyo", "*.so", "*.dylib", "*.egg-info", ".DS_Store")


class WorkCopyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConflictReport:
    """One file whose run edits and outside edits changed the same lines. Line numbers are 1-based, inclusive."""

    path: str
    hunk_count: int
    hunks: tuple[dict[str, object], ...]  # bounded: first MAX_HUNKS_PER_FILE hunks, each side cut to MAX_HUNK_LINES
    spans: str  # one line for the refusal message, e.g. "run lines 2-2 vs original lines 2-2"


class MergeConflictError(WorkCopyError):
    """Outside and run edits overlap; nothing was written. ``conflicts`` carries the typed report."""

    def __init__(self, message: str, conflicts: tuple[ConflictReport, ...]) -> None:
        super().__init__(message)
        self.conflicts = conflicts


class WorkCopyScanError(WorkCopyError):
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
    snapshot_reader: Callable[[str], bytes] | None = field(default=None, repr=False)

    def text_files(self) -> dict[str, str]:
        """Text files in the copy, keyed by POSIX relative path."""
        out: dict[str, str] = {}
        for rel in sorted(self.baseline):
            path = self.copy / rel
            try:
                if path.suffix.lower() in {".py", ".pyi"}:
                    with tokenize.open(path) as source:
                        out[rel] = source.read()
                else:
                    out[rel] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError, SyntaxError) as error:
                if path.suffix.lower() in {".py", ".pyi"}:
                    raise WorkCopyError(
                        f"cannot decode or read Python source file {rel} in the working copy"
                    ) from error
                continue
        return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ignored(name: str, is_dir: bool) -> bool:
    if is_dir:
        return name in IGNORE_DIRS or name.casefold() in SECRET_DIRS or name.endswith(".egg-info")
    return is_secret_file(name) or any(fnmatch.fnmatch(name, pattern) for pattern in IGNORE_GLOBS)


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

    def scan_error(error: OSError) -> None:
        path = error.filename or original
        detail = error.strerror or str(error)
        raise WorkCopyScanError(f"cannot scan repository directory {path}: {detail}") from error

    for directory, dirnames, filenames in os.walk(original, followlinks=False, onerror=scan_error):
        here = Path(directory)
        dirnames[:] = sorted(d for d in dirnames
                             if not _ignored(d, True) and not (here / d).is_symlink())
        for name in sorted(filenames):
            source = here / name
            rel = source.relative_to(original).as_posix()
            if _ignored(name, False) or source.is_symlink():
                work.skipped.append(rel)
                continue
            try:
                size = source.stat().st_size
            except OSError as error:
                raise WorkCopyScanError(f"cannot inspect repository file {source}: {error}") from error
            if size > limits.max_file_bytes:
                work.skipped.append(rel)
                continue
            total += size
            if len(work.baseline) >= limits.max_files or total > limits.max_total_bytes:
                raise WorkCopyError(
                    f"repository is larger than the run limits ({limits.max_files} files, "
                    f"{limits.max_total_bytes // (1024 * 1024)} MB); point --repo at a smaller tree")
            try:
                data = source.read_bytes()
            except OSError as error:
                raise WorkCopyScanError(f"cannot read repository file {source}: {error}") from error
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

    def scan_error(error: OSError) -> None:
        path = error.filename or work.copy
        detail = error.strerror or str(error)
        raise WorkCopyScanError(f"cannot scan working copy directory {path}: {detail}") from error

    for directory, dirnames, filenames in os.walk(work.copy, followlinks=False, onerror=scan_error):
        here = Path(directory)
        dirnames[:] = sorted(name for name in dirnames
                             if not _ignored(name, True) and not (here / name).is_symlink())
        for name in sorted(filenames):
            path = here / name
            if path.is_symlink() or _ignored(name, False):
                continue
            if not path.is_file():
                continue
            rel = path.relative_to(work.copy).as_posix()
            try:
                current[rel] = _sha(path.read_bytes())
            except OSError as error:
                raise WorkCopyScanError(f"cannot read working copy file {path}: {error}") from error
    for rel, digest in current.items():
        if rel not in work.baseline:
            changes[rel] = "added"
        elif work.baseline[rel] != digest:
            changes[rel] = "modified"
    for rel in work.baseline:
        if rel not in current:
            changes[rel] = "deleted"
    return dict(sorted(changes.items()))


def unified_diff(work: WorkCopy, *, scan_bytes: Callable[[bytes], object] | None = None) -> str:
    """Diff exact changed bytes, scanning paths and both inputs before formatting when requested."""
    chunks = []
    for rel, kind in changed_files(work).items():
        if scan_bytes is not None:
            scan_bytes(os.fsencode(rel))
        before_bytes = _read_original_bytes(work, rel) if kind != "added" else b""
        after_bytes = (work.copy / rel).read_bytes() if kind != "deleted" else b""
        if scan_bytes is not None:
            scan_bytes(before_bytes)
            scan_bytes(after_bytes)
        before = _diff_text(before_bytes)
        after = _diff_text(after_bytes)
        if before is None or after is None:
            chunks.append(
                f"binary/line-ending change: {rel} "
                "(text diff omitted; byte-level change recorded).\n"
            )
            continue
        file_diff = list(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{rel}" if kind != "added" else "/dev/null",
            tofile=f"b/{rel}" if kind != "deleted" else "/dev/null"))
        for index, line in enumerate(file_diff):
            if line.startswith((" ", "+", "-")) and not line.endswith(("\n", "\r")):
                file_diff[index] = line + "\n\\ No newline at end of file\n"
        if not file_diff:
            chunks.append(
                f"binary/line-ending change: {rel} "
                "(text diff omitted; byte-level change recorded).\n"
            )
            continue
        chunks.extend(file_diff)
    return "".join(chunks)


def _diff_text(data: bytes) -> str | None:
    """Return displayable UTF-8 text, preserving physical line endings."""
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_original_bytes(work: WorkCopy, rel: str) -> bytes:
    """Baseline bytes from the immutable start snapshot, or the unchanged original."""
    if work.snapshot_reader is not None:
        try:
            data = work.snapshot_reader(rel)
        except (OSError, ValueError) as error:
            raise WorkCopyError(f"original {rel} snapshot is unavailable") from error
    else:
        path = (work.snapshot if work.snapshot is not None else work.original) / rel
        data = path.read_bytes() if path.is_file() else b""
    if _sha(data) != work.baseline.get(rel):
        raise WorkCopyError(f"original {rel} changed during the run; diff would be wrong")
    return data


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


@contextmanager
def _engine_importable():
    """Make the engine's flat kernel_* modules importable for the switch helpers (same pattern as outer_checks)."""
    engine = str(Path(__file__).resolve().parent / "engine")
    added = engine not in sys.path
    if added:
        sys.path.insert(0, engine)
    try:
        yield
    finally:
        if added:
            sys.path.remove(engine)


def _drift_merge_enabled() -> bool:
    """COLLAB on and STRICT off, read by kernel_edit_drift. Any value it rejects keeps today's refusal."""
    with _engine_importable():
        from kernel_edit_drift import drift_merge_enabled
        try:
            return drift_merge_enabled()
        except ValueError:
            return False


def _start_bytes(work: WorkCopy, rel: str) -> bytes | None:
    """Start-of-run bytes of ``rel`` from the immutable snapshot, only if they match the baseline hash."""
    expected = work.baseline.get(rel)
    if work.snapshot is None or expected is None:
        return None
    try:
        with _parent_fd(work.snapshot, rel) as parent:
            data, _ = _read_at(parent, Path(rel).name)
    except (OSError, WorkCopyError):
        return None
    if data is None or _sha(data) != expected:
        return None
    return data


def _line_span(lo: int, hi: int) -> list[int]:
    return [lo + 1, hi] if hi > lo else [lo + 1, lo]


def _line_text(span: tuple[int, int]) -> str:
    first, last = _line_span(*span)
    return f"{first}-{last}"


def _cut_lines(text: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    if len(lines) <= MAX_HUNK_LINES:
        return text, False
    return "".join(lines[:MAX_HUNK_LINES]), True


def _hunk_payload(hunk: ConflictHunk) -> dict[str, object]:
    base, base_cut = _cut_lines(hunk.base_text)
    run, run_cut = _cut_lines(hunk.ours_text)
    original, original_cut = _cut_lines(hunk.theirs_text)
    return {
        "base_lines": _line_span(*hunk.base_lines),
        "run_lines": _line_span(*hunk.ours_lines),
        "original_lines": _line_span(*hunk.theirs_lines),
        "base_text": base,
        "run_text": run,
        "original_text": original,
        "truncated": base_cut or run_cut or original_cut,
    }


def _merge_outside_edit(base: bytes, run: bytes, original: bytes, rel: str) -> bytes | ConflictReport | None:
    """Merged bytes, a conflict report, or None when the bytes are not mergeable text (today's refusal stands)."""
    if any(b"\x00" in data for data in (base, run, original)):
        return None
    try:
        base_text, run_text, original_text = (data.decode("utf-8") for data in (base, run, original))
    except UnicodeDecodeError:
        return None
    outcome = merge_texts(base_text, run_text, original_text)
    if isinstance(outcome, MergeApplied):
        return outcome.text.encode("utf-8")
    if isinstance(outcome, MergeConflict):
        shown = outcome.hunks[:MAX_HUNKS_PER_FILE]
        spans = ", ".join(f"run lines {_line_text(hunk.ours_lines)} vs original lines {_line_text(hunk.theirs_lines)}"
                          for hunk in shown)
        more = f", +{len(outcome.hunks) - len(shown)} more" if len(outcome.hunks) > len(shown) else ""
        return ConflictReport(path=rel, hunk_count=len(outcome.hunks),
                              hunks=tuple(_hunk_payload(hunk) for hunk in shown), spans=spans + more)
    return None


def _merge_outside_edits(work: WorkCopy, changes: dict[str, bytes | None],
                         baseline: dict[str, str]) -> tuple[dict[str, bytes | None], dict[str, str], list[str]]:
    """Three-way merge each run write whose original file moved since the start (COLLAB-DRIFT on only).

    A merged file's new bytes replace the run's write, and its baseline becomes the original bytes it was merged
    onto, so guarded_apply still rechecks that file right before its own write. Anything not mergeable (deleted or
    created by the user, non-text, no start bytes, ambiguous) is left alone and refused by guarded_apply as before.
    Any real overlap aborts the whole apply with a MergeConflictError, before anything is written.
    """
    merged_changes = dict(changes)
    merged_baseline = dict(baseline)
    merged_paths: list[str] = []
    reports: list[ConflictReport] = []
    for rel, data in list(changes.items()):
        if data is None or baseline.get(rel) is None:
            continue
        try:
            with _parent_fd(work.original, rel) as parent:
                current, _ = _read_at(parent, Path(rel).name)
        except (OSError, WorkCopyError):
            continue
        if current is None or _sha(current) == baseline[rel]:
            continue
        base = _start_bytes(work, rel)
        if base is None:
            continue
        outcome = _merge_outside_edit(base, data, current, rel)
        if outcome is None:
            continue
        if isinstance(outcome, ConflictReport):
            reports.append(outcome)
            continue
        merged_changes[rel] = outcome
        merged_baseline[rel] = _sha(current)
        merged_paths.append(rel)
    if reports:
        parts = [f"{report.path} (merge conflict: {report.spans})" for report in reports[:MAX_CONFLICT_FILES_SHOWN]]
        raise MergeConflictError(
            "refusing --in-place: " + "; ".join(parts) + "; nothing was written", tuple(reports))
    return merged_changes, merged_baseline, merged_paths


def apply_in_place(work: WorkCopy, *, merged: list[str] | None = None) -> list[str]:
    """Copy changed files back to the original repository, guarded against concurrent edits.

    With COLLAB-DRIFT on, a file the user edited meanwhile is three-way merged (start snapshot as base) instead of
    refused. Merged paths are appended to ``merged`` only after the whole apply succeeds.
    """
    changes = {rel: (None if kind == "deleted" else (work.copy / rel).read_bytes())
               for rel, kind in changed_files(work).items()}
    baseline: dict[str, str] = work.baseline
    merged_paths: list[str] = []
    if _drift_merge_enabled():
        changes, baseline, merged_paths = _merge_outside_edits(work, changes, baseline)
    applied = guarded_apply(work.original, changes, baseline, what="--in-place")
    if merged is not None:
        merged.extend(sorted(merged_paths))
    return applied
