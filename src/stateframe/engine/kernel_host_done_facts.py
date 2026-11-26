"""Host-owned completion facts.

The completion claim is model-facing input, not completion evidence.  This
module keeps the two records separate: ``claims`` are retained for diagnosis,
while :class:`DoneFacts` is reconstructed from the workspace and typed runner
receipts.  The acceptance predicate never reads a claim.

The module is deliberately usable without a ``StateFrameRuntime`` instance so
the release boundary and small offline probes can exercise the authority in
isolation.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
import stat
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from weakref import WeakKeyDictionary

if __package__:
    from .kernel_file_digest_memo import RACY_NS
else:
    from kernel_file_digest_memo import RACY_NS

if TYPE_CHECKING:
    from kernel_history_spool import HistorySpool
    from kernel_verdict_identity import RunIdentity, RunIdentityProvider
    from kernel_workspace_consumers import WorkspaceConsumers
    from kernel_workspace_tree import File, Symlink


_T = TypeVar("_T")
_SNAPSHOT_UNAVAILABLE = object()


MODE_ENV = "STATEFRAME_HOST_DONE_FACTS"
SCHEMA = "stateframe-host-done-facts/1"
DONE_FACTS_SCHEMA = SCHEMA

_PASS = frozenset({"PASS", "PASSED", "GREEN", "SUCCESS", "SATISFIED", "OK"})
_FAIL = frozenset({"FAIL", "FAILED", "RED", "ERROR", "REFUSED", "BLOCKED"})
_IGNORED_PARTS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".test-tmp", ".tox",
})


class SnapshotChanged(OSError):
    """A captured file can no longer be proven to match its recorded digest."""

    def __init__(self, path: str, reason: str = "snapshot copy changed"):
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


def _open_regular_nofollow(root: Path, relative: str) -> int:
    """Open a regular file below root without following any path component."""
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise OSError("invalid snapshot path")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    flags = os.O_RDONLY | nofollow | directory
    current = os.open(root, flags)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0),
                          dir_fd=current)
    finally:
        os.close(current)
    if not stat.S_ISREG(os.fstat(file_fd).st_mode):
        os.close(file_fd)
        raise OSError("snapshot entry is not a regular file")
    return file_fd


def _read_regular(root: Path, relative: str) -> tuple[bytes, str, int]:
    digest, size, body, _ = _capture_worker_file(
        root, relative, keep_body=True, small_body=False)
    assert body is not None
    return body, digest, size


def _hash_regular(root: Path, relative: str) -> tuple[str, int]:
    digest, size, _, _ = _capture_worker_file(
        root, relative, keep_body=False, small_body=False)
    return digest, size


_FileSignature = tuple[int, int, int, int, int, int]
_RootIdentity = tuple[Path, int, int]


def _file_signature(info: os.stat_result) -> _FileSignature:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode)


class SnapshotFiles(Mapping[str, bytes]):
    """Digest-backed file mapping with bounded small-file caching."""

    SMALL_FILE_LIMIT = 64 * 1024
    BODY_CACHE_LIMIT = 16 * 1024 * 1024

    def __init__(self, rows: Mapping[str, tuple[str, int]] | None = None, *,
                 bodies: Mapping[str, bytes] | None = None,
                 own: Sequence[str] | set[str] = (),
                 start_root: str | os.PathLike[str] | None = None,
                 live_root: str | os.PathLike[str] | None = None,
                 reused_baseline: WorkspaceBaseline | None = None,
                 reused_paths: set[str] | None = None):
        self._rows = dict(rows or {})
        self._bodies = dict(bodies or {})
        self._body_cache_size = sum(
            len(body) for body in self._bodies.values()
            if len(body) <= self.SMALL_FILE_LIMIT)
        self._start_root = Path(start_root) if start_root is not None else None
        self._live_root = Path(live_root) if live_root is not None else None
        self._live_races: set[str] = set()
        self._own = frozenset(own)  # no proven copy elsewhere: these bodies stay here
        self._unproven: set[str] = set()
        self._reused_baseline = reused_baseline
        self._reused_paths = reused_paths if reused_paths is not None else set()

    def __getitem__(self, path: str) -> bytes:
        if path not in self._rows:
            raise KeyError(path)
        body = self._bodies.get(path)
        if body is not None:
            if path in self._unproven:
                if hashlib.sha256(body).hexdigest() != self._rows[path][0]:
                    raise SnapshotChanged(path, "restored body differs from its digest")
                self._unproven.discard(path)
            return body
        if self._reused_baseline is not None and path in self._reused_paths:
            try:
                return self._reused_baseline[path]
            except SnapshotChanged:
                # A cold start copy can disappear. The live file still needs
                # its own verified read if the caller requests current bytes.
                self._reused_paths.discard(path)
        digest, size = self._rows[path]
        if self._live_root is not None:
            try:
                body, actual_digest, actual_size = _read_regular(self._live_root, path)
            except OSError as error:
                raise SnapshotChanged(path, "live file is unavailable") from error
            if actual_digest != digest or actual_size != size:
                self._rows[path] = (actual_digest, actual_size)
                self._live_races.add(path)
            self._cache_small(path, body)
            return body
        root = self._start_root
        if root is None:
            raise SnapshotChanged(path, "no start snapshot is available")
        try:
            body, actual_digest, actual_size = _read_regular(root, path)
        except OSError as error:
            raise SnapshotChanged(path, "start snapshot file is unavailable") from error
        if actual_digest != digest or actual_size != size:
            raise SnapshotChanged(path, "start snapshot digest differs")
        self._cache_small(path, body)
        return body

    def __iter__(self) -> Iterator[str]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, path: object) -> bool:
        return path in self._rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        if len(self) != len(other) or any(path not in other for path in self):
            return False
        try:
            return all(self[path] == other[path] for path in self)
        except (SnapshotChanged, OSError, KeyError):
            return False

    def digest_for_path(self, path: str) -> str | None:
        row = self._rows.get(path)
        return row[0] if row is not None else None

    def size_for_path(self, path: str) -> int | None:
        row = self._rows.get(path)
        return row[1] if row is not None else None

    def read_for_diff(self, path: str) -> bytes:
        """Load one changed body; live rows are rehashed from the bytes read."""
        return self[path]

    def changed_during_read(self, path: str) -> bool:
        return path in self._live_races

    def __deepcopy__(self, _memo: object) -> SnapshotFiles:
        return self  # the rows do not change, and a copy would lose the start root

    def rebind_start_snapshot(self, root: str | os.PathLike[str] | None) -> None:
        self._start_root = Path(root) if root is not None else None

    def rebind_live_root(self, root: str | os.PathLike[str] | None) -> None:
        self._live_root = Path(root) if root is not None else None

    def _cache_small(self, path: str, body: bytes) -> None:
        if (len(body) <= self.SMALL_FILE_LIMIT and path not in self._bodies
                and self._body_cache_size + len(body) <= self.BODY_CACHE_LIMIT):
            self._bodies[path] = body
            self._body_cache_size += len(body)

    def __getstate__(self) -> dict[str, object]:
        small_bodies: dict[str, bytes] = {}
        kept = 0
        for path in self._rows:
            body = self._bodies.get(path)
            if body is not None and path in self._own:
                small_bodies[path] = body
            elif (body is not None and len(body) <= self.SMALL_FILE_LIMIT
                    and kept + len(body) <= self.BODY_CACHE_LIMIT):
                small_bodies[path] = body
                kept += len(body)
        return {
            "_rows": self._rows,
            "_bodies": small_bodies,
            "_own": sorted(self._own),
            "_start_root": None,
            "_live_root": self._live_root,
            **({"_reused_baseline": self._reused_baseline,
                "_reused_paths": self._reused_paths} if self._reused_paths else {}),
        }

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self._rows = dict(state.get("_rows", {}))
        raw_bodies = state.get("_bodies", {})
        self._bodies = dict(raw_bodies) if isinstance(raw_bodies, Mapping) else {}
        self._body_cache_size = sum(len(body) for body in self._bodies.values())
        self._start_root = None
        raw_live_root = state.get("_live_root")
        self._live_root = Path(raw_live_root) if raw_live_root is not None else None
        self._live_races = set()
        self._own = frozenset(state.get("_own", ()))
        self._unproven = set(self._bodies)
        self._reused_baseline = state.get("_reused_baseline")
        self._reused_paths = set(state.get("_reused_paths", ()))


class WorkspaceBaseline(SnapshotFiles):
    """Starting bytes plus process-local proofs of unchanged worker files.

    Matching parent copies back the cold bodies. Owned bodies share the
    small-file cache budget and spill to checkpointable disk storage beyond it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._spool: HistorySpool | None = None
        self._spooled_paths: dict[str, int] = {}
        self._worker: _RootIdentity | None = None
        self._signatures: dict[str, _FileSignature] = {}
        self._observed_ns = 0

    @classmethod
    def capture(cls, root: str | os.PathLike[str], *,
                start_snapshot: str | os.PathLike[str] | None = None) -> WorkspaceBaseline:
        snapshot = cls()
        _scan_workspace(root, start_snapshot=start_snapshot, starting=snapshot)
        return snapshot

    def _reuse_for(self, worker: _RootIdentity, now_ns: int) -> bool:
        if self._worker is None:
            # Restore discards all old signatures. The first live scan can
            # build new proofs, and only from freshly verified bytes.
            self._worker = worker
        if self._worker != worker:
            return False
        if now_ns < self._observed_ns:
            self._signatures.clear()
        self._observed_ns = now_ns
        return True

    def _spill(self, path: str, body: bytes) -> None:
        if self._spool is None:
            from kernel_memfix_history import MemfixSpool

            self._spool = MemfixSpool()
        position = len(self._spool)
        self._spool.append(body)
        self._spooled_paths[path] = position

    def __getitem__(self, path: str) -> bytes:
        position = self._spooled_paths.get(path)
        if position is None or self._spool is None:
            return super().__getitem__(path)
        try:
            return self._spool[position]
        except OSError as error:
            raise SnapshotChanged(path, "spooled start file is unavailable") from error

    def __getstate__(self) -> dict[str, object]:
        if self._spool is not None:
            import kernel_checkpoint as checkpoint

            # Spool blocks can be persisted outside the guarded state stream.
            # Scan the complete image before serializing any of those blocks.
            if checkpoint._SECRETS:
                tail = None
                block_size = 1 << 20
                for start in range(0, self._spool.spool_bytes, block_size):
                    block = self._spool._read_exact(
                        start, min(block_size, self._spool.spool_bytes - start))
                    tail = checkpoint._check_registered_secrets(block, tail, final=False)
                checkpoint._check_registered_secrets(b"", tail)
        return {**super().__getstate__(), "_spool": self._spool,
                "_spooled_paths": self._spooled_paths}

    def __setstate__(self, state: Mapping[str, object]) -> None:
        super().__setstate__(state)
        self._spool = state["_spool"]
        self._spooled_paths = state["_spooled_paths"]
        # Worker stat proofs belong to the attempt that made them.
        self._worker = None
        self._signatures = {}
        self._observed_ns = 0


def host_done_facts_mode() -> str:
    """Return the strict opt-in value for this lane's authority."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def mode() -> str:
    return host_done_facts_mode()


def enabled() -> bool:
    return host_done_facts_mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _copy(value: object) -> object:
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return str(value)


def _json_safe(value: object) -> object:
    """Keep diagnostic claims serializable without granting them authority."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (bytes, bytearray)):
        return (bytes(value).decode("utf-8", "replace"),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _unique_text(values: Sequence[object] | None) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (_as_text(values),)
    found: list[str] = []
    for value in values or ():
        text = _as_text(value)
        if text and text not in found:
            found.append(text)
    return tuple(found)


def _content_bytes(value: object) -> bytes:
    """Accept both our byte snapshots and small test-friendly snapshot maps."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8", "surrogateescape")
    if isinstance(value, Mapping):
        for key in ("content", "bytes", "data", "text"):
            if key in value:
                return _content_bytes(value[key])
    return repr(value).encode("utf-8", "backslashreplace")


def _path_is_ignored(relative: Path) -> bool:
    return any(part in _IGNORED_PARTS for part in relative.parts)


def _capture_worker_file(
    root: Path, relative: str, *, keep_body: bool, small_body: bool,
) -> tuple[str, int, bytes | None, _FileSignature]:
    digest = hashlib.sha256()
    size = 0
    fd = _open_regular_nofollow(root, relative)
    with os.fdopen(fd, "rb") as stream:
        signature = _file_signature(os.fstat(fd))
        chunks: list[bytes] | None = (
            [] if keep_body or (small_body and signature[2] <= SnapshotFiles.SMALL_FILE_LIMIT)
            else None
        )
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        if (size != signature[2]
                or _file_signature(os.fstat(fd)) != signature
                or _file_signature(os.lstat(root / relative)) != signature):
            raise SnapshotChanged(relative, "file identity changed during read")
    body = b"".join(chunks) if chunks is not None else None
    return digest.hexdigest(), size, body, signature


def _capture_scan_file(
    root: Path, relative: str, *, keep_body: bool, small_body: bool,
) -> tuple[str, int, bytes | None, _FileSignature]:
    try:
        return _capture_worker_file(root, relative, keep_body=keep_body, small_body=small_body)
    except SnapshotChanged:
        # An atomic replacement can leave a readable file at the same path.
        return _capture_worker_file(root, relative, keep_body=keep_body, small_body=small_body)


def workspace_snapshot(root: str | os.PathLike[str], *,
                       start_snapshot: str | os.PathLike[str] | None = None,
                       live: bool = False) -> SnapshotFiles:
    """Capture regular workspace paths as digests with lazily loaded bodies.

    The snapshot deliberately excludes VCS and test-cache directories.  The
    runtime writes its measurement output outside the task workspace; this
    small exclusion set also prevents runner cache churn from becoming a
    solver edit.
    """
    return _scan_workspace(root, start_snapshot=start_snapshot, live=live)


def current_workspace_snapshot(root: str | os.PathLike[str],
                               baseline: Mapping[str, object]) -> SnapshotFiles:
    if isinstance(baseline, WorkspaceBaseline):
        return _scan_workspace(root, live=True, baseline=baseline)
    return workspace_snapshot(root, live=True)


def _scan_workspace(
    root: str | os.PathLike[str], *,
    start_snapshot: str | os.PathLike[str] | None = None, live: bool = False,
    starting: WorkspaceBaseline | None = None, baseline: WorkspaceBaseline | None = None,
) -> SnapshotFiles:
    now_ns = time.time_ns()
    cutoff_ns = now_ns - RACY_NS
    base = Path(os.path.realpath(root))
    try:
        root_stat = base.lstat()
    except OSError:
        return SnapshotFiles(live_root=base if live else None)
    if not stat.S_ISDIR(root_stat.st_mode):
        return SnapshotFiles(live_root=base if live else None)
    worker = (base, root_stat.st_dev, root_stat.st_ino)
    if starting is not None:
        starting._worker = worker
        starting._observed_ns = now_ns
    if baseline is not None and not baseline._reuse_for(worker, now_ns):
        baseline = None

    source_root = (Path(start_snapshot)
                   if start_snapshot is not None and not live else None)
    rows: dict[str, tuple[str, int]] = {}
    bodies: dict[str, bytes] = {}
    own: set[str] = set()
    reused: set[str] = set()
    small_body_bytes = 0
    for current, dirs, files, directory_fd in os.fwalk(base, follow_symlinks=False):
        current_path = Path(current)
        if current_path == base:
            walked = os.fstat(directory_fd)
            if (walked.st_dev, walked.st_ino) != worker[1:]:
                break
        relative_dir = current_path.relative_to(base)
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED_PARTS)
        for name in sorted(files):
            if name in _IGNORED_PARTS:
                continue
            relative_text = (relative_dir / name).as_posix()
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    continue
                if (baseline is not None and max(info.st_mtime_ns, info.st_ctime_ns) < cutoff_ns
                        and baseline._signatures.get(relative_text) == _file_signature(info)):
                    rows[relative_text] = baseline._rows[relative_text]
                    reused.add(relative_text)
                    continue

                source_row = None
                if source_root is not None:
                    try:
                        source_row = _hash_regular(source_root, relative_text)
                    except OSError:
                        pass
                # Keep the observed start body when no matching parent copy backs it.
                keep_body = not live and source_row is None
                digest, size, body, signature = _capture_scan_file(
                    base, relative_text, keep_body=keep_body, small_body=not keep_body,
                )
                if source_row is not None and source_row != (digest, size):
                    if body is None:
                        digest, size, body, signature = _capture_scan_file(
                            base, relative_text, keep_body=True, small_body=False)
                    keep_body = True
            except OSError:
                # Preserve the snapshot's handling of unreadable entries.
                continue
            row = (digest, size)
            rows[relative_text] = row
            stable = max(signature[3], signature[4]) < cutoff_ns
            if starting is not None and stable:
                starting._signatures[relative_text] = signature
            elif baseline is not None and stable and baseline._rows.get(relative_text) == row:
                # A worker created just before startup may be racy on its first
                # capture. A fresh stable read can prove the same start bytes.
                baseline._signatures[relative_text] = signature
            if body is not None:
                fits = (size <= SnapshotFiles.SMALL_FILE_LIMIT
                        and small_body_bytes + size <= SnapshotFiles.BODY_CACHE_LIMIT)
                if keep_body and starting is None:
                    bodies[relative_text] = body
                    own.add(relative_text)
                elif keep_body and not fits:
                    starting._spill(relative_text, body)
                elif fits:
                    bodies[relative_text] = body
                    small_body_bytes += size
                    if keep_body:
                        own.add(relative_text)
    ordered_rows = {relative: rows[relative] for relative in sorted(rows, key=Path)}
    ordered_bodies = {relative: bodies[relative] for relative in ordered_rows
                      if relative in bodies}
    if starting is not None:
        starting._rows = ordered_rows
        starting._bodies = ordered_bodies
        starting._own = frozenset(own)
        starting._body_cache_size = small_body_bytes
        starting.rebind_start_snapshot(source_root)
        return starting
    return SnapshotFiles(
        ordered_rows, bodies=ordered_bodies, own=own,
        start_root=source_root if not live else None,
        live_root=base if live else None,
        reused_baseline=baseline if reused else None, reused_paths=reused,
    )


snapshot_workspace = workspace_snapshot


def rebind_workspace_snapshots(runtime: object, start_snapshot: object) -> None:
    """Rebind restored lazy snapshots to this attempt's spec-owned parent tree."""
    if start_snapshot is None:
        return
    seen: set[int] = set()

    def rebind(value: object, *, live_root: object = None) -> None:
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        method = getattr(value, "rebind_start_snapshot", None)
        if callable(method):
            method(start_snapshot)
        live_method = getattr(value, "rebind_live_root", None)
        if live_root is not None and callable(live_method):
            live_method(live_root)

    owner = getattr(runtime, "owner", None)
    root = getattr(owner, "root", None)
    rebind(getattr(runtime, "_host_done_facts_baseline", None))
    rebind(getattr(runtime, "_delivery_contract_baseline", None))
    delivery = getattr(runtime, "delivery_contract", None)
    rebind(getattr(delivery, "baseline", None))
    pending = getattr(delivery, "_pending_workspace_snapshot", None)
    if isinstance(pending, tuple) and len(pending) == 3:
        rebind(pending[2], live_root=root)
    lock = getattr(runtime, "green_lock", None)
    sources = getattr(lock, "_starting_product_sources", None)
    rebind(getattr(sources, "_snapshot", None))


def _lines(content: bytes) -> list[str]:
    return content.decode("utf-8", "surrogateescape").splitlines(keepends=True)


def _line_change(before: bytes, after: bytes) -> tuple[int, int]:
    old = _lines(before)
    new = _lines(after)
    added = deleted = 0
    for tag, old_start, old_end, new_start, new_end in difflib.SequenceMatcher(
            a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        deleted += old_end - old_start
        added += new_end - new_start
    # Binary files do not have useful text opcodes.  Represent a changed binary
    # as one changed line so it still appears in the typed workspace diff.
    if before and after and not old and not new:
        return 1, 1
    return added, deleted


def _snapshot_digest(snapshot: Mapping[str, object], path: str) -> str:
    digest_for_path = getattr(snapshot, "digest_for_path", None)
    if callable(digest_for_path):
        digest = digest_for_path(path)
        if digest is not None:
            return str(digest)
    return hashlib.sha256(_content_bytes(snapshot[path])).hexdigest()


def _snapshot_revision(snapshot: Mapping[str, object]) -> str:
    rows = [(str(path), _snapshot_digest(snapshot, str(path)))
            for path in sorted(snapshot, key=str)]
    return "workspace:" + _digest(rows)


@dataclass
class _PinnedCompletion:
    tree_digest: bytes
    files: tuple[tuple[str, File | Symlink], ...]
    identities: dict[tuple[Path, ...], str] = field(default_factory=dict)
    baseline: Mapping[str, object] | None = None
    facts: tuple[tuple[FileDelta, ...], str | None] | None = None


_PROJECTIONS: WeakKeyDictionary[WorkspaceConsumers, _PinnedCompletion] = WeakKeyDictionary()


def _pin_completion(
        root: Path, boundary: str, project: Callable[[_PinnedCompletion], _T],
) -> _T | None:
    """Project the full registered workspace under its live reader fence."""
    from kernel_workspace_consumers import consumers_for
    from kernel_workspace_state import WorkspaceUnavailable
    from kernel_workspace_tree import Directory, File, Symlink, children

    consumers = consumers_for(root)
    if consumers is None:
        return None
    state = consumers.state
    with state.pin(boundary) as observation:
        # The fallback uses the original filesystem scope if the pinned tree
        # excludes private storage or cannot attest to externally linked files.
        if ((state._home is not None and state._home.is_relative_to(state.fence.root))
                or state._blocked or state._external_links):
            return None
        projection = _PROJECTIONS.get(consumers)
        if projection is None or projection.tree_digest != observation.tree.digest:
            files = []
            pending = [("", observation.tree)]
            while pending:
                prefix, directory = pending.pop()
                for name, entry in children(directory):
                    path = prefix + os.fsdecode(name)
                    if isinstance(entry, Directory):
                        pending.append((path + "/", entry))
                    elif isinstance(entry, (File, Symlink)):
                        files.append((path, entry))
            projection = _PinnedCompletion(observation.tree.digest, tuple(sorted(files, key=lambda row: row[0])))
            _PROJECTIONS[consumers] = projection
        result = project(projection)
        if not observation.validate_for_use(boundary):
            raise WorkspaceUnavailable("completion workspace observation changed")
        return result


def completion_workspace_generation(root: Path) -> bytes | None:
    return _pin_completion(root, "completion-verdict", lambda projection: projection.tree_digest)


def pinned_run_identity(provider: RunIdentityProvider) -> tuple[bytes, RunIdentity] | None:
    """Reproduce the legacy run identity from matching pinned file metadata."""
    from kernel_verdict_identity import RunIdentity, _canonical as identity_bytes, _ignored, _sha256
    from kernel_workspace_tree import File

    def project(projection):
        excluded = provider.excluded_paths
        digest = projection.identities.get(excluded)
        if digest is None:
            rows = []
            for path, entry in projection.files:
                if _ignored(Path(path)):
                    continue
                if any((provider.root / path).is_relative_to(item) for item in excluded):
                    continue
                if isinstance(entry, File):
                    rows.append({"path": path, "kind": "file", "mode": entry.mode,
                                 "size": entry.size, "sha256": "sha256:" + entry.sha256.hex()})
                else:
                    rows.append({"path": path, "kind": "symlink",
                                 "target": os.fsdecode(entry.target)})
            digest = _sha256(identity_bytes({"schema": "stateframe-workspace-tree/1", "files": rows}))
            if len(projection.identities) >= 4:
                projection.identities.clear()
            projection.identities[excluded] = digest
        return projection.tree_digest, RunIdentity(
            provider.run_id, digest, provider.release_id, provider.harness_version)

    return _pin_completion(provider.root, "completion-identity", project)


def _pinned_done_workspace(
        root: Path, baseline: Mapping[str, object], *, needs_revision: bool,
        journal: object | None = None,
) -> tuple[tuple[FileDelta, ...], str | None] | None:
    from kernel_workspace_state import WorkspaceUnavailable
    from kernel_workspace_tree import File

    def project(projection):
        if (projection.baseline is baseline and projection.facts is not None
                and (not needs_revision or projection.facts[1] is not None)):
            return projection.facts
        after = {path: entry.sha256.hex() for path, entry in projection.files
                 if isinstance(entry, File) and not _path_is_ignored(Path(path))}
        before_paths = {str(path) for path in baseline}
        before = {path: _snapshot_hash(baseline, path) for path in before_paths}
        changed = {path for path in before_paths
                   if before[path] != after.get(path)}
        changed.update(after.keys() - before_paths)
        current_changes = {}
        for path in sorted(changed & after.keys()):
            content = (root / path).read_bytes()
            if hashlib.sha256(content).hexdigest() != after[path]:
                raise WorkspaceUnavailable("completion file changed during pinned read")
            current_changes[path] = content
        before_changes: dict[str, object] = {}
        for path in changed & before_paths:
            try:
                before_changes[path] = _snapshot_body(baseline, path)
            except SnapshotChanged:
                before_changes[path] = _SNAPSHOT_UNAVAILABLE
        deltas = []
        for path in sorted(changed):
            if before_changes.get(path) is _SNAPSHOT_UNAVAILABLE:
                _log_snapshot_changed(journal, "host_done_facts", path)
                deltas.append(FileDelta(
                    path=path, status="unavailable", before_sha256=before.get(path),
                    after_sha256=after.get(path)))
            else:
                old = ({path: before_changes[path]} if path in before_changes else {})
                new = ({path: current_changes[path]} if path in current_changes else {})
                deltas.extend(diff_snapshots(old, new))
        revision = "workspace:" + _digest(sorted(after.items())) if needs_revision else None
        projection.baseline = baseline
        projection.facts = (tuple(deltas), revision)
        return projection.facts

    return _pin_completion(root, "completion-diff", project)


def _revision(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("product_revision", "current_revision", "workspace_revision",
                    "revision", "artifact"):
            if value.get(key) is not None:
                candidate = _revision(value[key])
                if candidate:
                    return candidate
        return None
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
        return text if text else None
    return _digest(value)


@dataclass(frozen=True)
class FileDelta:
    path: str
    status: str
    added_lines: int
    deleted_lines: int
    before_lines: int
    after_lines: int
    before_sha256: str | None = None
    after_sha256: str | None = None

    @property
    def net_lines(self) -> int:
        return self.added_lines - self.deleted_lines

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "status": self.status,
            "added_lines": self.added_lines,
            "deleted_lines": self.deleted_lines,
            "net_lines": self.net_lines,
            "before_lines": self.before_lines,
            "after_lines": self.after_lines,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


def _snapshot_body(snapshot: Mapping[str, object], path: str) -> bytes:
    reader = getattr(snapshot, "read_for_diff", None)
    value = reader(path) if callable(reader) else snapshot[path]
    return _content_bytes(value)


def _snapshot_hash(snapshot: Mapping[str, object], path: str) -> str | None:
    return _snapshot_digest(snapshot, path) if path in snapshot else None


def _log_snapshot_changed(journal: object | None, consumer: str, path: str) -> None:
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("workspace_snapshot_changed", producer="kernel_host_done_facts",
             dispatch=False, consumer=consumer, path=path,
             reason_code="SNAPSHOT_CHANGED")


def diff_snapshots(before: Mapping[str, object], after: Mapping[str, object], *,
                   journal: object | None = None,
                   consumer: str = "workspace_diff") -> tuple[FileDelta, ...]:
    """Return deterministic deltas, loading bodies only after digest differences."""
    deltas: list[FileDelta] = []
    old_paths = {str(path) for path in before}
    new_paths = {str(path) for path in after}
    for path in sorted(old_paths | new_paths):
        before_hash = _snapshot_hash(before, path)
        after_hash = _snapshot_hash(after, path)
        if before_hash is not None and after_hash is not None and before_hash == after_hash:
            continue
        try:
            before_bytes = _snapshot_body(before, path) if before_hash is not None else b""
            after_bytes = _snapshot_body(after, path) if after_hash is not None else b""
        except SnapshotChanged:
            _log_snapshot_changed(journal, consumer, path)
            deltas.append(FileDelta(
                path=path, status="unavailable", added_lines=0, deleted_lines=0,
                before_lines=0, after_lines=0,
                before_sha256=before_hash, after_sha256=after_hash,
            ))
            continue
        # A file that changed back while it was read has no delta: the bytes read decide.
        if before_hash is not None and after_hash is not None and before_bytes == after_bytes:
            continue
        added, deleted = _line_change(before_bytes, after_bytes)
        deltas.append(FileDelta(
            path=path,
            status="added" if before_hash is None else "deleted" if after_hash is None else "modified",
            added_lines=added,
            deleted_lines=deleted,
            before_lines=len(_lines(before_bytes)) if before_hash is not None else 0,
            after_lines=len(_lines(after_bytes)) if after_hash is not None else 0,
            before_sha256=before_hash,
            after_sha256=_snapshot_hash(after, path) if after_hash is not None else None,
        ))
    return tuple(deltas)


diff_workspace = diff_snapshots


@dataclass(frozen=True)
class CheckResult:
    status: str
    selectors: tuple[str, ...] = ()
    revision: str | None = None
    current: bool = False
    exit_code: int | None = None
    receipt_id: str | None = None
    source: str = "runner"
    reason_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        result = {
            "status": self.status,
            "result": self.status,
            "selectors": list(self.selectors),
            "revision": self.revision,
            "current": self.current,
            "exit_code": self.exit_code,
            "receipt_id": self.receipt_id,
            "source": self.source,
        }
        if self.reason_code:
            result["reason_code"] = self.reason_code
        return result


def _selectors(row: Mapping[str, object]) -> tuple[str, ...]:
    values: list[object] = []
    for key in ("selectors", "public_selectors", "tests"):
        values.extend(_sequence(row.get(key)))
    for key in ("selector", "test_selector", "test", "name"):
        if row.get(key) is not None:
            values.append(row[key])
    return _unique_text(values)


def _status(row: Mapping[str, object]) -> str:
    from kernel_verdict import execution_stop_cause
    if execution_stop_cause(row) is not None:
        return "UNRESOLVED"
    exit_code = row.get("exit_code", row.get("returncode"))
    if isinstance(exit_code, bool):
        exit_code = int(exit_code)
    if isinstance(exit_code, int):
        return "PASS" if exit_code == 0 else "FAIL"
    for key in ("status", "result_class", "outcome", "result", "verdict"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip().upper()
        if text in _PASS or text.endswith("_PASS") or text.endswith("_GREEN"):
            return "PASS"
        if text in _FAIL or text.endswith("_FAIL") or text.endswith("_RED"):
            return "FAIL"
    return "UNKNOWN"


def _row_revision(row: Mapping[str, object]) -> str | None:
    for key in ("after_binding", "binding", "current", "revision_info"):
        value = row.get(key)
        candidate = _revision(value)
        if candidate:
            return candidate
    for key in ("current_revision", "workspace_revision", "product_revision", "revision",
                "artifact"):
        candidate = _revision(row.get(key))
        if candidate:
            return candidate
    return None


def _receipt_id(row: Mapping[str, object]) -> str | None:
    for key in ("receipt_id", "event_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def _is_current(row: Mapping[str, object], current_revision: str | None) -> bool:
    if row.get("binding_stable") is False or row.get("stale") is True:
        return False
    candidate = _row_revision(row)
    if candidate is not None and current_revision is not None:
        return candidate == current_revision
    # A typed receipt without a revision is still runner-owned evidence for
    # the revision at which the runtime emitted it.  Explicit revision data
    # above takes precedence and catches stale checks.
    return True


def _normalise_receipts(receipts: object) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for item in _sequence(receipts):
        if not isinstance(item, Mapping):
            continue
        nested = item.get("receipt")
        if isinstance(nested, Mapping) and not any(
                key in item for key in ("exit_code", "selectors", "result_class", "status")):
            rows.append(nested)
        else:
            rows.append(item)
    return tuple(rows)


def _selector_covers(required: str, observed: str) -> bool:
    if required == observed:
        return True
    required_file = required.split("::", 1)[0]
    observed_file = observed.split("::", 1)[0]
    return required_file == observed or required == observed_file or (
        observed.startswith(required + "::") or required.startswith(observed + "::"))


def _claims(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return _json_safe(dict(value))  # type: ignore[return-value]
    if value is None:
        return {}
    return {"summary": _json_safe(value)}


@dataclass(frozen=True)
class DoneFacts:
    """A typed, runner-derived completion record.

    ``claims`` is intentionally present for audit/debugging, but is never read
    by :meth:`_rejection_reasons` or any acceptance helper.
    """

    current_revision: str
    changed_files: tuple[str, ...] = ()
    file_deltas: tuple[FileDelta, ...] = ()
    per_file_line_deltas: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    last_check_result: CheckResult | None = None
    public_tests_required: tuple[str, ...] = ()
    public_tests_ran: tuple[str, ...] = ()
    public_tests_passed: tuple[str, ...] = ()
    unresolved_gating_items: tuple[str, ...] = ()
    empty_diff: bool = True
    evidence_digest: str = ""
    claims: Mapping[str, object] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()
    accepted: bool = field(init=False)

    def __post_init__(self) -> None:
        deltas = tuple(self.file_deltas)
        changed = tuple(self.changed_files) or tuple(delta.path for delta in deltas)
        line_deltas = {
            str(path): {
                "added_lines": int(values.get("added_lines", values.get("added", 0))),
                "deleted_lines": int(values.get("deleted_lines", values.get("deleted", 0))),
                "net_lines": int(values.get(
                    "net_lines", values.get("net", values.get("added_lines", 0) -
                    values.get("deleted_lines", 0)))),
            }
            for path, values in self.per_file_line_deltas.items()
            if isinstance(values, Mapping)
        }
        if not line_deltas:
            line_deltas = {
                delta.path: {"added_lines": delta.added_lines,
                             "deleted_lines": delta.deleted_lines,
                             "net_lines": delta.net_lines}
                for delta in deltas
            }
        unresolved = _unique_text(self.unresolved_gating_items)
        passed = _unique_text(self.public_tests_passed)
        if not passed and self.last_check_result is not None:
            if self.last_check_result.current and self.last_check_result.status == "PASS":
                passed = self.last_check_result.selectors
        reasons = self._rejection_reasons(
            empty_diff=bool(self.empty_diff),
            last_check=self.last_check_result,
            required=self.public_tests_required,
            ran=self.public_tests_ran,
            passed=passed,
            unresolved=unresolved,
        )
        if not self.evidence_digest:
            evidence = {
                "current_revision": self.current_revision,
                "changed_files": list(changed),
                "per_file_line_deltas": line_deltas,
                "last_check_result": (self.last_check_result.to_dict()
                                       if self.last_check_result else None),
                "public_tests_required": list(self.public_tests_required),
                "public_tests_ran": list(self.public_tests_ran),
                "public_tests_passed": list(passed),
                "unresolved_gating_items": list(unresolved),
                "empty_diff": bool(self.empty_diff),
            }
            evidence_digest = _digest(evidence)
        else:
            evidence_digest = str(self.evidence_digest)
        object.__setattr__(self, "changed_files", changed)
        object.__setattr__(self, "file_deltas", deltas)
        object.__setattr__(self, "per_file_line_deltas", line_deltas)
        object.__setattr__(self, "unresolved_gating_items", unresolved)
        object.__setattr__(self, "public_tests_passed", passed)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "evidence_digest", evidence_digest)
        object.__setattr__(self, "claims", _claims(self.claims))
        object.__setattr__(self, "accepted", not reasons)

    @staticmethod
    def _rejection_reasons(*, empty_diff: bool, last_check: CheckResult | None,
                           required: Sequence[str], ran: Sequence[str],
                           passed: Sequence[str],
                           unresolved: Sequence[str]) -> tuple[str, ...]:
        reasons: list[str] = []
        if empty_diff:
            reasons.append("EMPTY_DIFF")
        if required:
            if last_check is None:
                reasons.append("CHECK_MISSING")
            elif not last_check.current:
                reasons.append("CHECK_NOT_CURRENT")
            elif last_check.status != "PASS":
                reasons.append(last_check.reason_code or "CHECK_FAILED")
        pass_selectors = tuple(passed)
        missing = [selector for selector in required
                   if not any(_selector_covers(selector, observed) for observed in pass_selectors)]
        if missing:
            reasons.append("PUBLIC_TESTS_MISSING")
        if unresolved:
            reasons.append("UNRESOLVED_GATING_ITEMS")
        return tuple(reasons)

    @property
    def is_acceptable(self) -> bool:
        return self.accepted

    @property
    def host_check_state(self) -> str:
        return "REQUIRED_PUBLIC_CHECKS" if self.public_tests_required else "NO_HOST_CHECKS"

    @property
    def changed_file_paths(self) -> tuple[str, ...]:
        return self.changed_files

    @property
    def line_deltas(self) -> Mapping[str, Mapping[str, int]]:
        return self.per_file_line_deltas

    @property
    def last_check(self) -> CheckResult | None:
        return self.last_check_result

    @property
    def unresolved_gates(self) -> tuple[str, ...]:
        return self.unresolved_gating_items

    def to_dict(self) -> dict[str, object]:
        line_deltas = {path: dict(values)
                       for path, values in self.per_file_line_deltas.items()}
        return {
            "schema": SCHEMA,
            "current_revision": self.current_revision,
            "changed_files": list(self.changed_files),
            "changed_file_paths": list(self.changed_files),
            "file_deltas": [delta.to_dict() for delta in self.file_deltas],
            "per_file_line_deltas": line_deltas,
            "line_deltas": copy.deepcopy(line_deltas),
            "last_check_result": (self.last_check_result.to_dict()
                                   if self.last_check_result is not None else None),
            "public_tests_required": list(self.public_tests_required),
            "public_tests_ran": list(self.public_tests_ran),
            "public_tests_passed": list(self.public_tests_passed),
            "host_check_state": self.host_check_state,
            "no_host_checks": not bool(self.public_tests_required),
            "unresolved_gating_items": list(self.unresolved_gating_items),
            "empty_diff": self.empty_diff,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "evidence_digest": self.evidence_digest,
            # Explicitly labelled non-authoritative at the serialization edge.
            "claims": _copy(self.claims),
        }

    as_dict = to_dict

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DoneFacts":
        raw_deltas = value.get("file_deltas")
        deltas: list[FileDelta] = []
        for item in _sequence(raw_deltas):
            if not isinstance(item, Mapping):
                continue
            deltas.append(FileDelta(
                path=_as_text(item.get("path")),
                status=_as_text(item.get("status")) or "modified",
                added_lines=int(item.get("added_lines", item.get("added", 0)) or 0),
                deleted_lines=int(item.get("deleted_lines", item.get("deleted", 0)) or 0),
                before_lines=int(item.get("before_lines", 0) or 0),
                after_lines=int(item.get("after_lines", 0) or 0),
                before_sha256=(str(item["before_sha256"])
                               if item.get("before_sha256") is not None else None),
                after_sha256=(str(item["after_sha256"])
                              if item.get("after_sha256") is not None else None),
            ))
        raw_check = value.get("last_check_result")
        check = None
        if isinstance(raw_check, Mapping):
            exit_code = raw_check.get("exit_code")
            check = CheckResult(
                status=_status(raw_check),
                selectors=_unique_text(_sequence(raw_check.get("selectors"))),
                revision=_revision(raw_check.get("revision")),
                current=bool(raw_check.get("current")),
                exit_code=(int(exit_code) if isinstance(exit_code, int) else None),
                receipt_id=(str(raw_check["receipt_id"])
                            if raw_check.get("receipt_id") is not None else None),
                source=_as_text(raw_check.get("source")) or "runner",
                reason_code=(str(raw_check["reason_code"])
                             if raw_check.get("reason_code") is not None else None),
            )
        raw_lines = value.get("per_file_line_deltas", value.get("line_deltas", {}))
        line_deltas = raw_lines if isinstance(raw_lines, Mapping) else {}
        raw_changed = value.get("changed_files", value.get("changed_file_paths", ()))
        changed_values = [item.get("path") if isinstance(item, Mapping) else item
                          for item in _sequence(raw_changed)]
        changed = _unique_text(changed_values)
        if not changed and deltas:
            changed = tuple(delta.path for delta in deltas)
        required = _unique_text(_sequence(value.get("public_tests_required")))
        ran = _unique_text(_sequence(value.get("public_tests_ran")))
        passed = _unique_text(_sequence(value.get("public_tests_passed")))
        return cls(
            current_revision=_as_text(value.get("current_revision")) or "unknown",
            changed_files=changed,
            file_deltas=tuple(deltas),
            per_file_line_deltas=line_deltas,
            last_check_result=check,
            public_tests_required=required,
            public_tests_ran=ran,
            public_tests_passed=passed,
            unresolved_gating_items=_unique_text(
                _sequence(value.get("unresolved_gating_items"))),
            empty_diff=bool(value.get("empty_diff", not changed)),
            evidence_digest=_as_text(value.get("evidence_digest")),
            claims=_claims(value.get("claims")),
        )

    from_dict = from_mapping


def compute_done_facts(
    root: str | os.PathLike[str],
    baseline: Mapping[str, object] | str | os.PathLike[str] | None = None,
    *,
    current_revision: object = None,
    edit_receipts: object = (),
    test_receipts: object = (),
    required_public_tests: Sequence[object] = (),
    public_tests: Sequence[object] = (),
    unresolved_gating_items: Sequence[object] = (),
    claims: object = None,
    runner_evidence: Mapping[str, object] | None = None,
    pinned_baseline: bool = False,
    journal: object | None = None,
) -> DoneFacts:
    """Build facts solely from a workspace snapshot and typed host receipts.

    ``edit_receipts`` is accepted as an audit input for API symmetry, but the
    authoritative changed-file calculation is always the workspace diff.  A
    caller may pass a mapping of runner evidence to make release-boundary use
    explicit; neither that mapping nor ``claims`` is consulted for acceptance.
    """
    del edit_receipts  # The filesystem is the source of truth for changed files.
    evidence = runner_evidence if isinstance(runner_evidence, Mapping) else {}
    if not test_receipts:
        test_receipts = evidence.get("test_receipts", evidence.get("checks", ()))
    if not unresolved_gating_items:
        unresolved_gating_items = _sequence(evidence.get("unresolved_gating_items"))
    if claims is None and "claims" in evidence:
        claims = evidence.get("claims")
    if baseline is None:
        before: Mapping[str, object] = {}
    elif isinstance(baseline, (str, os.PathLike)):
        before = workspace_snapshot(baseline)
    else:
        before = baseline
    revision = _revision(current_revision)
    pinned = None
    if pinned_baseline:
        try:
            pinned = _pinned_done_workspace(Path(root), before,
                                            needs_revision=revision is None,
                                            journal=journal)
        except (OSError, ValueError):
            # Preserve legacy snapshot behavior for roots or paths the fence cannot admit.
            pass
    if pinned is None:
        after = current_workspace_snapshot(root, before)
        deltas = diff_snapshots(before, after, journal=journal,
                                consumer="host_done_facts")
        revision = revision or _snapshot_revision(after)
    else:
        deltas, snapshot_revision = pinned
        revision = revision or snapshot_revision
    required = _unique_text(required_public_tests)
    if not required:
        required = _unique_text(public_tests)
    rows = _normalise_receipts(test_receipts)
    checks: list[CheckResult] = []
    current_rows: list[CheckResult] = []
    for row in rows:
        from kernel_verdict import execution_stop_cause
        stop_cause = execution_stop_cause(row)
        check = CheckResult(
            status="UNRESOLVED" if stop_cause is not None else _status(row),
            selectors=_selectors(row),
            revision=_row_revision(row) or revision,
            current=_is_current(row, revision),
            exit_code=(int(row["exit_code"]) if isinstance(row.get("exit_code"), int)
                       else None),
            receipt_id=_receipt_id(row),
            source=_as_text(row.get("source")) or "runner",
            reason_code=("CHECK_STOPPED:" + stop_cause if stop_cause is not None else None),
        )
        checks.append(check)
        if check.current:
            current_rows.append(check)
    last_check = current_rows[-1] if current_rows else (checks[-1] if checks else None)
    public_ran: list[str] = []
    public_passed: list[str] = []
    for check in current_rows:
        for selector in check.selectors:
            if selector not in public_ran:
                public_ran.append(selector)
            if check.status == "PASS" and selector not in public_passed:
                public_passed.append(selector)
    line_deltas = {
        delta.path: {"added_lines": delta.added_lines,
                     "deleted_lines": delta.deleted_lines,
                     "net_lines": delta.net_lines}
        for delta in deltas
    }
    unavailable = ["snapshot_unavailable:" + delta.path for delta in deltas
                   if delta.status == "unavailable"]
    return DoneFacts(
        current_revision=revision,
        changed_files=tuple(delta.path for delta in deltas),
        file_deltas=deltas,
        per_file_line_deltas=line_deltas,
        last_check_result=last_check,
        public_tests_required=required,
        public_tests_ran=tuple(public_ran),
        public_tests_passed=tuple(public_passed),
        unresolved_gating_items=_unique_text(_sequence(unresolved_gating_items)
                                             + tuple(unavailable)),
        empty_diff=not bool(deltas),
        claims=_claims(claims),
    )


build_done_facts = compute_done_facts
derive_done_facts = compute_done_facts
from_runner_evidence = compute_done_facts


def claims_record(*, summary: object = None, claim: object = None,
                  evidence_refs: object = ()) -> dict[str, object]:
    """Return the diagnostic claim envelope without making it authoritative."""
    return {
        "summary": _json_safe(summary),
        "claim": _json_safe(claim),
        "evidence_refs": _json_safe(list(_sequence(evidence_refs))),
    }


def decorate_outcome(result: Mapping[str, object]) -> dict[str, object]:
    """Attach a host-authoritative completion view to a release outcome."""
    decorated = dict(result)
    done_seal = result.get("done_seal")
    if (isinstance(done_seal, Mapping)
            and done_seal.get("schema") == "stateframe-done-seal/1"
            and done_seal.get("stored") is True):
        decorated["completion"] = {
            "accepted": True,
            "authority": "done_seal",
            "schema": SCHEMA,
            "rejection_reasons": [],
            "evidence_digest": done_seal.get("artifact_digest"),
        }
        decorated["completion_accepted"] = True
        decorated["completion_authority"] = "done_seal"
        return decorated
    if "claims" not in decorated:
        claim_values = {key: result[key] for key in ("summary", "claim", "evidence_refs")
                        if key in result}
        if claim_values:
            decorated["claims"] = _claims(claim_values)
    raw = result.get("done_facts")
    if isinstance(raw, Mapping):
        facts = DoneFacts.from_mapping(raw)
        decorated["done_facts"] = facts.to_dict()
        submission_blocked = (
            result.get("accepted") is False
            or str(result.get("result", result.get("outcome", ""))).upper()
            in {"INCOMPLETE", "RUN_INCOMPLETE", "AUTHORITY_FAILURE"}
        )
        completion_accepted = bool(facts.accepted and not submission_blocked)
        completion = {
            "accepted": completion_accepted,
            "done_facts_accepted": facts.accepted,
            "authority": "host_done_facts",
            "schema": SCHEMA,
            "rejection_reasons": list(facts.rejection_reasons),
            "evidence_digest": facts.evidence_digest,
        }
        if submission_blocked:
            completion["rejection_reasons"].append("SUBMISSION_NOT_ACCEPTED")
    else:
        completion = {
            "accepted": False,
            "authority": "host_done_facts",
            "schema": SCHEMA,
            "rejection_reasons": ["DONE_FACTS_MISSING"],
        }
    decorated["completion"] = completion
    decorated["completion_accepted"] = bool(completion["accepted"])
    decorated["completion_authority"] = "host_done_facts"
    return decorated


def outcome_uses_done_facts(result: Mapping[str, object]) -> bool:
    return (isinstance(result.get("completion"), Mapping)
            and result["completion"].get("authority") == "host_done_facts")
