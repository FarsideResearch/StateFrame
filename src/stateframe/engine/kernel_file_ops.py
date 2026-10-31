"""Guarded product-file removal and relocation in the private workspace."""
from __future__ import annotations

import os
import hashlib
import json
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable


class FileOperationRefused(ValueError):
    """A requested file effect violates the workspace or task boundary."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")

    def reply(self) -> str:
        return f"REFUSED {self}"


@dataclass(frozen=True)
class FileOperationEffect:
    changed_paths: tuple[str, ...]
    source: str
    destination: str | None = None


class FileOperationReplay:
    """Bind native mutations to durable intent and verified local replay."""

    def __init__(self, root: Path, name: str, arguments: dict, call_id: str, *,
                 authority: dict, environment_identity: str):
        from kernel_effects import scan
        from kernel_shell_tool import workspace_identity
        from kernel_tool_tape import (replay_mode_active, tape_for_run, replay_decision,
                                      verify_intent, ToolReplayRefused)

        self.root, self.call_id = root, call_id
        self.identity = environment_identity
        self.tape = tape_for_run()
        self.record = self.tape.lookup(call_id) if replay_mode_active() else None
        intent = dict(tool=name, args=arguments, replay_class="local",
                      capability_digest=hashlib.sha256(json.dumps(
                          authority, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                      definition_digest=None, env_identity=self.identity,
                      pre_manifest_digest=workspace_identity(scan(root, exclude=lambda _: False)))
        if replay_mode_active():
            reason = replay_decision(self.record, replay_external_allowed=False).refuse_reason
            if not reason:
                reason = verify_intent(self.record, **intent)
            if reason:
                raise ToolReplayRefused(reason)
        else:
            self.tape.intent(call_id=call_id, **intent)

    def complete(self, result: str, observation: str) -> str:
        from kernel_effects import scan
        from kernel_shell_tool import workspace_identity
        from kernel_tool_tape import verify_reexecution, ToolReplayRefused

        completion = dict(exit_code={"APPLIED": 0, "REFUSED": 1, "FAILED": 2}[result],
                          post_manifest_digest=workspace_identity(scan(self.root, exclude=lambda _: False)),
                          env_identity=self.identity)
        if self.record is not None:
            reason = verify_reexecution(self.record, **completion)
            if reason:
                raise ToolReplayRefused(reason)
            return self.record.observation
        self.tape.complete(call_id=self.call_id, observation=observation, **completion)
        return observation


def _path(root: Path, raw: str) -> tuple[Path, str]:
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise FileOperationRefused("OUTSIDE_WORKSPACE", "use a nonempty repository-relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {".", "..", ""} for part in raw.split("/")):
        raise FileOperationRefused("OUTSIDE_WORKSPACE", f"path {raw!r} is not repository-relative")
    current = root
    for part in pure.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        except NotADirectoryError as error:
            raise FileOperationRefused(
                "PATH_NOT_FOUND", f"path {raw!r} traverses a non-directory") from error
        if stat.S_ISLNK(mode):
            raise FileOperationRefused("SYMLINK", f"path {raw!r} contains a symlink")
    if not current.resolve().is_relative_to(root.resolve()):
        raise FileOperationRefused("OUTSIDE_WORKSPACE", f"path {raw!r} escapes the workspace")
    return current, pure.as_posix()


def _source(path: Path, relative: str, *, protected: Callable[[Path], bool],
            editable: Callable[[str], bool]) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise FileOperationRefused("PATH_NOT_FOUND", f"{relative!r} does not exist") from error
    if stat.S_ISDIR(info.st_mode):
        raise FileOperationRefused("IS_DIRECTORY", f"{relative!r} is a directory")
    if not stat.S_ISREG(info.st_mode):
        raise FileOperationRefused("NOT_EDITABLE", f"{relative!r} is not a regular file")
    if protected(path):
        raise FileOperationRefused("PROTECTED_TEST", f"{relative!r} is a protected test or inode alias")
    if not editable(relative):
        raise FileOperationRefused("NOT_EDITABLE", f"{relative!r} is outside --edit scope")


def file_delete(root: Path, raw_path: str, *, protected: Callable[[Path], bool],
                editable: Callable[[str], bool]) -> FileOperationEffect:
    path, relative = _path(root, raw_path)
    _source(path, relative, protected=protected, editable=editable)
    path.unlink()
    return FileOperationEffect((relative,), relative)


def file_move(root: Path, raw_source: str, raw_destination: str, *,
              protected: Callable[[Path], bool], editable: Callable[[str], bool],
              new_file_allowed: Callable[[str], bool]) -> FileOperationEffect:
    source, source_rel = _path(root, raw_source)
    destination, destination_rel = _path(root, raw_destination)
    _source(source, source_rel, protected=protected, editable=editable)
    if source.stat().st_mode & 0o111:
        raise FileOperationRefused("UNSUPPORTED_MODE", "moving executable files is not supported by the sealed artifact")
    if source_rel == destination_rel:
        raise FileOperationRefused("DESTINATION_EXISTS", "source and destination are identical")
    if protected(destination):
        raise FileOperationRefused("PROTECTED_TEST", f"{destination_rel!r} is a protected test path")
    if not new_file_allowed(destination_rel):
        raise FileOperationRefused("NEW_FILE_FORBIDDEN", f"{destination_rel!r} is outside new-file policy")
    # A case-insensitive volume resolves a case-only destination to the source inode.
    case_only = (source.parent == destination.parent
                 and source.name.casefold() == destination.name.casefold())
    if destination.exists() and not (case_only and source.samefile(destination)):
        raise FileOperationRefused("DESTINATION_EXISTS", f"{destination_rel!r} already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if case_only and destination.exists():
        temporary = source.with_name(f".stateframe-move-{secrets.token_hex(16)}")
        source.rename(temporary)
        try:
            temporary.rename(destination)
        except OSError:
            temporary.rename(source)
            raise
    else:
        # Refuse a late collision. os.link + unlink keeps destination creation exclusive.
        try:
            os.link(source, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise FileOperationRefused("DESTINATION_EXISTS", f"{destination_rel!r} already exists") from error
        try:
            source.unlink()
        except OSError:
            destination.unlink()
            raise
    return FileOperationEffect((source_rel, destination_rel), source_rel, destination_rel)
