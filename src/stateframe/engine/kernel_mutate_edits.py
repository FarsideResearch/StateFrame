"""Crash-recoverable, directory-fd-bound mutation edit proposals."""
from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
import platform
import re
import stat
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from kernel_mutate_inputs import scope_path_matches

MAX_EDIT_BYTES = 256 << 10
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class StagedEdit:
    call: str
    root_identity: tuple[int, int]
    files: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class EditResult:
    committed: bool
    paths: tuple[str, ...]


class EditConflict(ValueError):
    pass


def file_digest(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_EDIT_BYTES:
            raise EditConflict("edit target is not a bounded, singly linked regular file")
        data = os.read(fd, MAX_EDIT_BYTES + 1)
        if os.fstat(fd).st_size != len(data):
            raise EditConflict("edit target changed during read")
        return hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)


def _parts(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, str) or not raw or raw.startswith("/") or len(raw) > 4096:
        raise EditConflict("invalid edit path")
    parts = tuple(raw.split("/"))
    if any(part in ("", ".", "..") or "\\" in part or "\x00" in part for part in parts):
        raise EditConflict("invalid edit path")
    return parts


def _parent(root_fd: int, parts: tuple[str, ...]) -> int:
    # A single constrained lookup closes the component-walk race, not the later
    # lookup-to-rename race. No Linux rename syscall accepts RESOLVE_BENEATH.
    if sys.platform == "linux":
        if platform.machine() not in {"x86_64", "aarch64"}:
            raise EditConflict("constrained edit lookup is unavailable on this architecture")
        class OpenHow(ctypes.Structure):
            _fields_ = [("flags", ctypes.c_uint64), ("mode", ctypes.c_uint64),
                        ("resolve", ctypes.c_uint64)]
        how = OpenHow(os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC, 0, 0x08 | 0x04 | 0x02)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        fd = libc.syscall(ctypes.c_long(437), ctypes.c_int(root_fd),
                          ctypes.c_char_p(os.fsencode("/".join(parts[:-1]) or ".")),
                          ctypes.byref(how), ctypes.c_size_t(ctypes.sizeof(how)))
        if fd < 0:
            number = ctypes.get_errno()
            raise EditConflict("edit parent constrained lookup failed") from OSError(
                number, os.strerror(number))
        return fd
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except OSError as error:
        os.close(fd)
        raise EditConflict("edit parent is missing, replaced or linked") from error


def _read_at(parent_fd: int, name: str) -> tuple[bytes | None, int | None]:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    except FileNotFoundError:
        return None, None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_EDIT_BYTES:
            raise EditConflict("edit target is not a bounded, singly linked regular file")
        data = os.read(fd, MAX_EDIT_BYTES + 1)
        if os.fstat(fd).st_size != len(data):
            raise EditConflict("edit target changed during read")
        return data, info.st_mode & 0o777
    finally:
        os.close(fd)


def _identity(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


# Files a test runner or interpreter executes or reads on its own: a unit writing one could make an
# unshipped file produce a shipped side effect (or run code at collection), so no grant covers them.
_COLLECTION_FILES = {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml", "noxfile.py",
                     "sitecustomize.py", "usercustomize.py"}


def _collection_or_startup(parts) -> bool:
    name = parts[-1]
    return (name in _COLLECTION_FILES or name.endswith(".pth")
            or any(part in {"test", "tests"} for part in parts[:-1]))


class _WriteTimeConflict(OSError):
    """An edit path changed during commit; handled by rollback and surfaced as EditConflict."""


_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2


def _rename_at2(src_fd: int, src: str, dst_fd: int, dst: str, flags: int) -> None:
    """Use atomic no-clobber or exchange; fail closed where the kernel lacks it."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise _WriteTimeConflict("atomic edit rename is unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(src_fd, os.fsencode(src), dst_fd, os.fsencode(dst), flags) != 0:
        number = ctypes.get_errno()
        if number in (errno.ENOSYS, errno.EOPNOTSUPP):
            raise _WriteTimeConflict("atomic edit rename is unavailable")
        raise OSError(number, os.strerror(number))


def _before_edit_operation() -> None:
    """Test seam immediately before an edit changes its target entry."""


@contextmanager
def _current_parent(root_fd: int, parts: tuple[str, ...], identity: tuple[int, int]):
    """Walk from the workspace again immediately before using a parent for an edit."""
    try:
        fd = _parent(root_fd, parts)
    except EditConflict as error:
        raise _WriteTimeConflict("edit parent moved before replacement") from error
    try:
        if _identity(fd) != identity:
            raise _WriteTimeConflict("edit parent changed before replacement")
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _writer_lock(journal):
    path = journal.root / "edit.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _entry_state(parent_fd: int, name: str) -> tuple[object, ...] | None:
    """Identify an entry and its bytes; equal bytes in another inode are not ours."""
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    try:
        data, mode = _read_at(parent_fd, name)
    except EditConflict as error:
        raise _CompensationConflict("edit entry is no longer a safe regular file") from error
    after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
        raise _WriteTimeConflict("edit entry changed during validation")
    return after.st_dev, after.st_ino, data, mode


class _CompensationConflict(_WriteTimeConflict):
    """Keep undo evidence when concurrent interference makes compensation unsafe."""


@dataclass
class _BoundEdit:
    parent_fd: int
    identity: tuple[int, int]
    name: str
    temporary: str
    before: tuple[object, ...] | None
    replacement: tuple[object, ...] | None
    changed: bool = False

    def validate(self, *, changed: bool) -> None:
        # Compensation deliberately checks object authority, NOT current location:
        # refusing all access to a moved directory would prevent undoing our write.
        # This is a narrow inverse capability, never permission to write new bytes.
        target = self.replacement if changed else self.before
        temporary = self.before if changed else self.replacement
        if (_identity(self.parent_fd) != self.identity
                or _entry_state(self.parent_fd, self.name) != target
                or _entry_state(self.parent_fd, self.temporary) != temporary):
            raise _CompensationConflict("edit entries changed; compensation is indeterminate")

    def undo(self) -> None:
        if not self.changed:
            return
        self.validate(changed=True)
        if self.before is None:
            _rename_at2(self.parent_fd, self.name, self.parent_fd, self.temporary,
                        _RENAME_NOREPLACE)
        elif self.replacement is None:
            _rename_at2(self.parent_fd, self.temporary, self.parent_fd, self.name,
                        _RENAME_NOREPLACE)
        else:
            _rename_at2(self.parent_fd, self.temporary, self.parent_fd, self.name,
                        _RENAME_EXCHANGE)
        self.changed = False
        self.validate(changed=False)
        os.fsync(self.parent_fd)

    def cleanup(self) -> None:
        state = self.before if self.changed else self.replacement
        if (_identity(self.parent_fd) != self.identity
                or _entry_state(self.parent_fd, self.temporary) != state):
            raise _CompensationConflict("edit temporary changed; cleanup is indeterminate")
        if state is not None:
            os.unlink(self.temporary, dir_fd=self.parent_fd)
            os.fsync(self.parent_fd)


@dataclass(frozen=True)
class _WriteContext:
    root_fd: int
    parts: tuple[str, ...]
    identity: tuple[int, int]
    expected: bytes | None
    pending: list[_BoundEdit] | None


_WRITE_CONTEXT: ContextVar[_WriteContext | None] = ContextVar("mutation_edit_write", default=None)


@contextmanager
def _bound_write_context(root_fd: int, parts: tuple[str, ...], identity: tuple[int, int],
                         expected: bytes | None, pending: list[_BoundEdit] | None = None):
    token = _WRITE_CONTEXT.set(_WriteContext(root_fd, parts, identity, expected, pending))
    try:
        yield
    finally:
        _WRITE_CONTEXT.reset(token)


def _undo_edits(receipts: list[_BoundEdit]) -> None:
    failure = None
    for receipt in reversed(receipts):
        try:
            receipt.undo()
            receipt.cleanup()
        except (EditConflict, OSError) as error:
            # Preserve ambiguous evidence, but still undo other independent edits.
            if failure is None:
                failure = error
    if failure is not None:
        raise _CompensationConflict("mutation edit rollback is indeterminate") from failure


def _write_bound(root_fd: int, parts: tuple[str, ...], identity: tuple[int, int],
                 parent_fd: int, data: bytes | None, mode: int, expected: bytes | None,
                 pending: list[_BoundEdit]) -> None:
    def verify_parent() -> None:
        with _current_parent(root_fd, parts, identity):
            if _identity(parent_fd) != identity:
                raise _WriteTimeConflict("retained edit parent changed")

    verify_parent()
    name = parts[-1]
    before = _entry_state(parent_fd, name)
    if (before[2] if before is not None else None) != expected:
        raise _WriteTimeConflict("edit target bytes changed")
    temporary = ".sf-mutate-" + uuid.uuid4().hex
    receipt = _BoundEdit(os.dup(parent_fd), identity, name, temporary, before, None)
    pending.append(receipt)
    if data is not None:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     mode & ~0o111, dir_fd=parent_fd)
        view = memoryview(data)
        try:
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("short mutation edit write")
                view = view[count:]
            os.fsync(fd)
            os.fchmod(fd, mode & ~0o111)
        finally:
            try:
                info = os.fstat(fd)
                receipt.replacement = (info.st_dev, info.st_ino,
                                       data[:len(data) - len(view)], info.st_mode & 0o777)
            finally:
                os.close(fd)
    _before_edit_operation()
    verify_parent()
    receipt.validate(changed=False)
    if data is None:
        _rename_at2(parent_fd, name, parent_fd, temporary, _RENAME_NOREPLACE)
    else:
        _rename_at2(parent_fd, temporary, parent_fd, name,
                    _RENAME_EXCHANGE if before is not None else _RENAME_NOREPLACE)
    receipt.changed = True
    os.fsync(parent_fd)
    verify_parent()
    receipt.validate(changed=True)


def _write_at(parent_fd: int, name: str, data: bytes | None, mode: int) -> None:
    # Keep the original four-argument edit seam; its context binds the parent
    # to a fresh root-relative lookup before the fd can authorize a write.
    context = _WRITE_CONTEXT.get()
    if context is None or name != context.parts[-1] or _identity(parent_fd) != context.identity:
        raise _WriteTimeConflict("edit parent changed before replacement")
    own_pending = context.pending is None
    receipts: list[_BoundEdit] = [] if own_pending else context.pending
    try:
        _write_bound(context.root_fd, context.parts, context.identity, parent_fd,
                     data, mode, context.expected, receipts)
        if own_pending:
            for receipt in receipts:
                receipt.cleanup()
    except OSError:
        if own_pending:
            _undo_edits(receipts)
        raise
    finally:
        if own_pending:
            for receipt in receipts:
                os.close(receipt.parent_fd)


class MutationEdits:
    def __init__(self, runtime):
        self.runtime = runtime
        self.failure_point: str | None = None

    def stage(self, call: str, proposals: list[dict]) -> StagedEdit:
        runtime = self.runtime
        if runtime.repo is None or not getattr(runtime.policy, "write_globs", ()):
            raise EditConflict("mutation writes are not granted")
        if not isinstance(proposals, list) or not 1 <= len(proposals) <= 16:
            raise EditConflict("edit set must contain one to sixteen files")
        root_fd = os.open(runtime.repo, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            files = []
            seen = set()
            total = 0
            effect_policy = None
            if runtime.effect_row is not None:
                from kernel_effects import effect_policy_for_run

                baseline_tests = {str(path if Path(path).is_absolute() else runtime.repo / path)
                                  for path in runtime.protected_paths}
                baseline_tests.update(str(path) for pattern in ("test_*.py", "conftest.py")
                                      for path in runtime.repo.rglob(pattern))
                effect_policy = effect_policy_for_run(runtime.repo, runtime.effect_row, baseline_tests)
            for proposal in proposals:
                if not isinstance(proposal, dict) or set(proposal) - {"op", "path", "base_sha256", "text"}:
                    raise EditConflict("invalid edit proposal")
                path = proposal.get("path")
                parts = _parts(path)
                if path in seen:
                    raise EditConflict("duplicate edit path")
                seen.add(path)
                if not any(scope_path_matches(path, glob) for glob in runtime.policy.write_globs):
                    raise EditConflict("edit path exceeds mutation write grant")
                if runtime.edit_scope is not None and path not in runtime.edit_scope:
                    raise EditConflict("edit path exceeds --edit scope")
                if (parts[-1].startswith("test_") or parts[-1] == "conftest.py"
                        or "tests" in parts or path in runtime.protected_paths):
                    raise EditConflict("protected test cannot be edited")
                if _collection_or_startup(parts):
                    raise EditConflict("edit path is a collection or startup file")
                op = proposal.get("op")
                if op not in ("replace", "create", "delete"):
                    raise EditConflict("invalid edit operation")
                base = proposal.get("base_sha256")
                if base is not None and (not isinstance(base, str) or _DIGEST.fullmatch(base) is None):
                    raise EditConflict("invalid base sha256")
                text = proposal.get("text")
                if op == "delete":
                    if text not in (None, ""):
                        raise EditConflict("delete cannot include text")
                    post = None
                elif not isinstance(text, str):
                    raise EditConflict("edit text must be UTF-8")
                else:
                    post = text.encode("utf-8", "strict")
                    total += len(post)
                if total > MAX_EDIT_BYTES:
                    raise EditConflict("edit set exceeds byte limit")
                parent_fd = _parent(root_fd, parts)
                try:
                    parent_identity = _identity(parent_fd)
                    pre, mode = _read_at(parent_fd, parts[-1])
                finally:
                    os.close(parent_fd)
                digest = hashlib.sha256(pre).hexdigest() if pre is not None else None
                if digest != base or (op == "create") != (pre is None):
                    raise EditConflict("edit base sha256 conflict")
                if mode is not None and mode & 0o111:
                    raise EditConflict("executable edit target is not allowed")
                if effect_policy is not None:
                    from kernel_effects import Effect, Entry

                    before = Entry("file", mode, digest, len(pre)) if pre is not None else None
                    after = Entry("file", mode or 0o644,
                                  hashlib.sha256(post).hexdigest(), len(post)) if post is not None else None
                    allowed, reason = effect_policy.authorize(Effect(
                        path, {"replace": "modified", "create": "created", "delete": "deleted"}[op],
                        before, after))
                    if not allowed:
                        raise EditConflict(reason)
                files.append({"path": path, "parts": parts, "parent_identity": parent_identity,
                              "pre": pre, "post": post, "mode": mode or 0o644})
            return StagedEdit(call, _identity(root_fd), tuple(files))
        finally:
            os.close(root_fd)

    def commit(self, staged: StagedEdit, *, cleanup_ok: bool) -> EditResult:
        runtime = self.runtime
        if not cleanup_ok:
            raise EditConflict("payload cleanup is unverified")
        with _writer_lock(runtime.journal):
            root_fd = os.open(runtime.repo, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            parents = []
            pending: list[_BoundEdit] = []
            committed = False
            try:
                if _identity(root_fd) != staged.root_identity:
                    raise EditConflict("workspace identity changed")
                # The complete set is validated before the first write.
                for file in staged.files:
                    parent_fd = _parent(root_fd, file["parts"])
                    parents.append(parent_fd)
                    if _identity(parent_fd) != file["parent_identity"]:
                        raise EditConflict("edit parent identity changed")
                    current, _mode = _read_at(parent_fd, file["parts"][-1])
                    if current != file["pre"]:
                        raise EditConflict("edit base sha256 conflict")
                evidence = []
                for file in staged.files:
                    pre, post = file["pre"], file["post"]
                    pre_blob = runtime.journal.put_blob(pre) if pre is not None else None
                    post_blob = runtime.journal.put_blob(post) if post is not None else None
                    evidence.append({"path": file["path"], "pre_sha256": pre_blob,
                                     "post_sha256": post_blob, "pre_blob": pre_blob,
                                     "post_blob": post_blob})
                runtime.journal.append("EFFECT_PREPARED", call=staged.call, files=evidence,
                                       root_identity=list(staged.root_identity),
                                       parent_identities=[list(file["parent_identity"]) for file in staged.files])
                # Tool dispatch is sequential and jobs see a read-only working copy, so no
                # engine writer runs concurrently; still compare each target just before
                # replacing it and roll the whole set back if anything else changed it.
                for index, file in enumerate(staged.files):
                    with _current_parent(root_fd, file["parts"], file["parent_identity"]) as parent_fd:
                        if _read_at(parent_fd, file["parts"][-1])[0] != file["pre"]:
                            raise _WriteTimeConflict("edit target changed before replacement")
                        with _bound_write_context(root_fd, file["parts"], file["parent_identity"],
                                                  file["pre"], pending):
                            _write_at(parent_fd, file["parts"][-1], file["post"], file["mode"])
                    if index == 0 and self.failure_point == "after_first_rename":
                        raise RuntimeError("injected crash after first rename")
                for file in staged.files:
                    with _current_parent(root_fd, file["parts"], file["parent_identity"]) as parent_fd:
                        if _read_at(parent_fd, file["parts"][-1])[0] != file["post"]:
                            raise _WriteTimeConflict("edit target changed after replacement")
                runtime.journal.append("EFFECT_COMMITTED", call=staged.call,
                                       files=evidence, root_identity=list(staged.root_identity))
                committed = True
                for receipt in pending:
                    receipt.cleanup()
                return EditResult(True, tuple(file["path"] for file in staged.files))
            except (EditConflict, OSError) as error:
                if committed:
                    runtime.journal.append("EFFECT_INDETERMINATE", call=staged.call)
                    raise EditConflict("mutation edit cleanup is indeterminate") from error
                try:
                    _undo_edits(pending)
                except (EditConflict, OSError) as rollback_error:
                    runtime.journal.append("EFFECT_INDETERMINATE", call=staged.call)
                    raise EditConflict("mutation edit rollback is indeterminate") from rollback_error
                runtime.journal.append("EFFECT_REFUSED", call=staged.call)
                if isinstance(error, (EditConflict, _WriteTimeConflict)):
                    raise EditConflict(str(error)) from error
                raise
            finally:
                for receipt in pending:
                    os.close(receipt.parent_fd)
                for fd in parents:
                    os.close(fd)
                os.close(root_fd)

    def recover(self, call: str, *, allow_rollforward: bool = True) -> EditResult:
        runtime = self.runtime
        rows = [row for row in runtime.journal.events() if row.get("call") == call]
        prepared = [row for row in rows if row["kind"] == "EFFECT_PREPARED"]
        if len(prepared) != 1:
            raise EditConflict("effect preparation evidence is missing")
        if any(row["kind"] in {"EFFECT_REFUSED", "EFFECT_INDETERMINATE"} for row in rows):
            raise EditConflict("refused or indeterminate mutation edit cannot roll forward")
        event = prepared[0]
        files = event["files"]
        with _writer_lock(runtime.journal):
            root_fd = os.open(runtime.repo, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                if list(_identity(root_fd)) != event["root_identity"]:
                    raise EditConflict("workspace identity changed during recovery")
                parents = []
                try:
                    for index, file in enumerate(files):
                        parent_fd = _parent(root_fd, _parts(file["path"]))
                        parents.append(parent_fd)
                        if list(_identity(parent_fd)) != event["parent_identities"][index]:
                            raise EditConflict("edit parent identity changed during recovery")
                        current, _mode = _read_at(parent_fd, _parts(file["path"])[-1])
                        digest = hashlib.sha256(current).hexdigest() if current is not None else None
                        allowed = ((file["pre_sha256"], file["post_sha256"])
                                   if allow_rollforward else (file["post_sha256"],))
                        if digest not in allowed:
                            raise EditConflict("indeterminate edit bytes during recovery")
                    for index, file in enumerate(files):
                        parts = _parts(file["path"])
                        identity = tuple(event["parent_identities"][index])
                        with _current_parent(root_fd, parts, identity) as parent_fd:
                            current, mode = _read_at(parent_fd, parts[-1])
                            digest = hashlib.sha256(current).hexdigest() if current is not None else None
                            allowed = ((file["pre_sha256"], file["post_sha256"])
                                       if allow_rollforward else (file["post_sha256"],))
                            if digest not in allowed:
                                raise EditConflict("indeterminate edit bytes during recovery")
                            if digest != file["post_sha256"]:
                                if not allow_rollforward:
                                    raise EditConflict("committed effect bytes changed")
                                post = (runtime.journal.get_blob(file["post_blob"])
                                        if file["post_blob"] is not None else None)
                                with _bound_write_context(root_fd, parts, identity, current):
                                    _write_at(parent_fd, parts[-1], post, mode or 0o644)
                    if not any(row["kind"] == "EFFECT_COMMITTED" for row in rows):
                        runtime.journal.append("EFFECT_COMMITTED", call=call, files=files,
                                               root_identity=event["root_identity"])
                    return EditResult(True, tuple(file["path"] for file in files))
                except _WriteTimeConflict as error:
                    kind = ("EFFECT_INDETERMINATE" if isinstance(error, _CompensationConflict)
                            else "EFFECT_REFUSED")
                    runtime.journal.append(kind, call=call)
                    raise EditConflict(str(error)) from error
                finally:
                    for fd in parents:
                        os.close(fd)
            finally:
                os.close(root_fd)
