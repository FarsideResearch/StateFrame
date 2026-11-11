"""Crash-recoverable, directory-fd-bound mutation edit proposals."""
from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import os
import re
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


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
    """A target changed during commit; handled by the rollback path, surfaced as EditConflict."""


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


def _write_at(parent_fd: int, name: str, data: bytes | None, mode: int) -> None:
    if data is None:
        os.unlink(name, dir_fd=parent_fd)
    else:
        temporary = ".sf-mutate-" + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     mode & ~0o111, dir_fd=parent_fd)
        try:
            view = memoryview(data)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("short mutation edit write")
                view = view[count:]
            os.fsync(fd)
            os.fchmod(fd, mode & ~0o111)
        finally:
            os.close(fd)
        try:
            os.rename(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
    os.fsync(parent_fd)


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
                if not any(fnmatch.fnmatchcase(path, glob) for glob in runtime.policy.write_globs):
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
            written = 0
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
                for index, (file, parent_fd) in enumerate(zip(staged.files, parents)):
                    if _read_at(parent_fd, file["parts"][-1])[0] != file["pre"]:
                        raise _WriteTimeConflict("edit target changed before replacement")
                    _write_at(parent_fd, file["parts"][-1], file["post"], file["mode"])
                    written = index + 1
                    if index == 0 and self.failure_point == "after_first_rename":
                        raise RuntimeError("injected crash after first rename")
                for file, parent_fd in zip(staged.files, parents):
                    if _read_at(parent_fd, file["parts"][-1])[0] != file["post"]:
                        raise _WriteTimeConflict("edit target changed after replacement")
                runtime.journal.append("EFFECT_COMMITTED", call=staged.call,
                                       files=evidence, root_identity=list(staged.root_identity))
                return EditResult(True, tuple(file["path"] for file in staged.files))
            except EditConflict:
                runtime.journal.append("EFFECT_REFUSED", call=staged.call)
                raise
            except OSError as error:
                try:
                    # Only targets this commit replaced are rolled back; an untouched target
                    # that something else changed is left as it is.
                    for file, parent_fd in list(zip(staged.files, parents))[:written]:
                        current, _mode = _read_at(parent_fd, file["parts"][-1])
                        if current == file["post"] and current != file["pre"]:
                            _write_at(parent_fd, file["parts"][-1], file["pre"], file["mode"])
                        elif current != file["pre"]:
                            raise EditConflict("indeterminate edit bytes during rollback")
                except (EditConflict, OSError) as rollback_error:
                    runtime.journal.append("EFFECT_INDETERMINATE", call=staged.call)
                    raise EditConflict("mutation edit rollback is indeterminate") from rollback_error
                runtime.journal.append("EFFECT_REFUSED", call=staged.call)
                if isinstance(error, _WriteTimeConflict):
                    raise EditConflict(str(error)) from error
                raise
            finally:
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
                    for file, parent_fd in zip(files, parents):
                        current, mode = _read_at(parent_fd, _parts(file["path"])[-1])
                        digest = hashlib.sha256(current).hexdigest() if current is not None else None
                        if digest != file["post_sha256"]:
                            if not allow_rollforward:
                                raise EditConflict("committed effect bytes changed")
                            post = (runtime.journal.get_blob(file["post_blob"])
                                    if file["post_blob"] is not None else None)
                            _write_at(parent_fd, _parts(file["path"])[-1], post, mode or 0o644)
                    if not any(row["kind"] == "EFFECT_COMMITTED" for row in rows):
                        runtime.journal.append("EFFECT_COMMITTED", call=call, files=files,
                                               root_identity=event["root_identity"])
                    return EditResult(True, tuple(file["path"] for file in files))
                finally:
                    for fd in parents:
                        os.close(fd)
            finally:
                os.close(root_fd)
