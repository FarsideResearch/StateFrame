"""Sandboxed hook execution and durable observation replay."""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
from typing import Callable, Mapping, Sequence

if __package__:
    from .kernel_effects import (
        ContentStore, Effect, EffectError, EffectPolicy, Entry, Reconciliation,
        WorkspaceEffectPolicy, diff, manifest_digest, reconcile, scan,
    )
    from .kernel_exec_capability import (
        _macos_developer_roots, ExecCapability, ExecLimits,
        executable_read_roots, symlink_hops,
    )
    from .kernel_exec_session import ExecSession, ExecSessionError
    from .kernel_hooks import (
        GATING_EVENTS, HookArgvError, HookConfigError, HookDefinition,
        HookOutcome, HookSet, PanelEntry, build_payload, check_operands,
        event_id, matches, parse_outcome, resolve_argv, sanitize,
        _RUN_PATH, _pairs, _constant, _file_sha, _regular_no_symlink,
    )
else:
    from kernel_effects import (
        ContentStore, Effect, EffectError, EffectPolicy, Entry, Reconciliation,
        WorkspaceEffectPolicy, diff, manifest_digest, reconcile, scan,
    )
    from kernel_exec_capability import (
        _macos_developer_roots, ExecCapability, ExecLimits,
        executable_read_roots, symlink_hops,
    )
    from kernel_exec_session import ExecSession, ExecSessionError
    from kernel_hooks import (
        GATING_EVENTS, HookArgvError, HookConfigError, HookDefinition,
        HookOutcome, HookSet, PanelEntry, build_payload, check_operands,
        event_id, matches, parse_outcome, resolve_argv, sanitize,
        _RUN_PATH, _pairs, _constant, _file_sha, _regular_no_symlink,
    )


class HookContainmentError(RuntimeError):
    """A hook may have escaped its execution boundary."""


class HookReplayRefused(BaseException):
    """A recorded hook cannot be replayed faithfully."""


class _HookSessionSetupError(RuntimeError):
    """A sandbox session could not be prepared before hook launch."""


class HookTape:
    """Logical-run tape; each append is durable before the next transition."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "hook-tape.jsonl"
        if self.path.is_symlink():
            raise ValueError("hook-tape.jsonl must not be a symlink")
        self._records: dict[str, dict] = {}
        if self.path.exists():
            for number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), 1
            ):
                try:
                    value = json.loads(
                        line, object_pairs_hook=_pairs, parse_constant=_constant
                    )
                    if type(value) is not dict or value.get("kind") not in (
                        "intent",
                        "complete",
                    ):
                        raise ValueError("invalid record kind")
                    record_id = value.get("record_id")
                    if type(record_id) is not str or not record_id:
                        raise ValueError("missing record id")
                    if value["kind"] == "intent":
                        if (
                            set(value) != {"kind", "record_id", "identity"}
                            or type(value["identity"]) is not dict
                            or record_id in self._records
                        ):
                            raise ValueError("duplicate or invalid intent")
                        self._records[record_id] = {
                            "identity": value["identity"],
                            "result": None,
                            "completed": False,
                        }
                    else:
                        if (
                            set(value) != {"kind", "record_id", "result"}
                            or type(value["result"]) is not dict
                            or record_id not in self._records
                            or self._records[record_id]["completed"]
                        ):
                            raise ValueError("completion without unique intent")
                        self._records[record_id].update(
                            result=value["result"], completed=True
                        )
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"hook-tape.jsonl line {number}: {error}"
                    ) from error

    def _append(self, value: dict) -> None:
        encoded = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        created = not self.path.exists()
        fd = os.open(
            self.path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("hook-tape.jsonl is not a regular private file")
            with os.fdopen(fd, "ab", closefd=False) as stream:
                stream.write(encoded)
                stream.flush()
            os.fsync(fd)
        finally:
            os.close(fd)
        if created:
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def intent(self, record_id: str, identity: dict) -> None:
        if (
            type(record_id) is not str
            or not record_id
            or type(identity) is not dict
            or record_id in self._records
        ):
            raise ValueError("duplicate or invalid hook intent")
        self._append({"kind": "intent", "record_id": record_id, "identity": identity})
        self._records[record_id] = {
            "identity": identity,
            "result": None,
            "completed": False,
        }

    def complete(self, record_id: str, result: dict) -> None:
        if (
            type(result) is not dict
            or record_id not in self._records
            or self._records[record_id]["completed"]
        ):
            raise ValueError("missing or already completed hook intent")
        self._append({"kind": "complete", "record_id": record_id, "result": result})
        self._records[record_id].update(result=result, completed=True)

    def lookup(self, record_id: str) -> dict | None:
        return self._records.get(record_id)

    def pending(self) -> tuple[str, ...]:
        return tuple(
            key for key, value in self._records.items() if not value["completed"]
        )


@dataclass(frozen=True)
class HookRunResult:
    hook: str
    event: str
    event_id: str
    outcome: HookOutcome
    replayed: bool
    skipped: str | None


@dataclass(frozen=True)
class WriterEffect:
    path: str
    pre_sha256: str
    post_sha256: str


@dataclass(frozen=True)
class WriterResult:
    hook: str
    authorized: tuple[WriterEffect, ...]
    restored: tuple[str, ...]
    reconciliation: Reconciliation


@dataclass(frozen=True)
class EventResult:
    decision: str
    deny_hook: str | None
    deny_code: str | None
    reason: str
    entries: tuple[PanelEntry, ...]
    runs: tuple[HookRunResult, ...]
    writes: tuple[WriterResult, ...] = ()


class _ScopedWriterPolicy:
    def __init__(
        self,
        run_policy: EffectPolicy,
        allowed_paths: frozenset[str],
        protected_paths: frozenset[str],
    ):
        self.run_policy = run_policy
        self.allowed_paths = allowed_paths
        self.protected_paths = protected_paths

    def authorize(self, effect: Effect) -> tuple[bool, str]:
        if effect.path in self.protected_paths:
            return False, "protected hook input"
        if effect.path not in self.allowed_paths:
            return False, "outside triggering action paths"
        if (
            effect.change != "modified"
            or effect.before is None
            or effect.after is None
            or effect.before.kind != "file"
            or effect.after.kind != "file"
        ):
            return False, "writer may only modify existing regular files"
        if effect.before.mode != effect.after.mode:
            return False, "writer may not change file mode"
        return self.run_policy.authorize(effect)


def _entry_data(entry: Entry | None) -> dict | None:
    return asdict(entry) if entry is not None else None


def _effect_data(effect: Effect) -> dict:
    return {
        "path": effect.path,
        "change": effect.change,
        "before": _entry_data(effect.before),
        "after": _entry_data(effect.after),
    }


def _entry_from_data(value: object) -> Entry | None:
    if value is None:
        return None
    if (
        type(value) is not dict
        or set(value) != {"kind", "mode", "digest", "size"}
        or value["kind"] not in {"file", "symlink", "dir", "other"}
        or type(value["mode"]) is not int
        or (value["digest"] is not None and type(value["digest"]) is not str)
        or type(value["size"]) is not int
    ):
        raise ValueError("invalid writer entry")
    return Entry(**value)


def _effect_from_data(value: object) -> Effect:
    if (
        type(value) is not dict
        or set(value) != {"path", "change", "before", "after"}
        or type(value["path"]) is not str
        or value["change"] not in {"created", "modified", "deleted", "mode", "type"}
    ):
        raise ValueError("invalid writer effect")
    return Effect(
        value["path"],
        value["change"],
        _entry_from_data(value["before"]),
        _entry_from_data(value["after"]),
    )


def _reconciliation_data(value: Reconciliation) -> dict:
    return {
        "applied": [_effect_data(effect) for effect in value.applied],
        "restored": [
            {"effect": _effect_data(effect), "reason": reason}
            for effect, reason in value.restored
        ],
        "manifest": {
            path: asdict(entry) for path, entry in sorted(value.manifest.items())
        },
    }


def _reconciliation_from_data(value: object) -> Reconciliation:
    if (
        type(value) is not dict
        or set(value) != {"applied", "restored", "manifest"}
        or type(value["applied"]) is not list
        or type(value["restored"]) is not list
        or type(value["manifest"]) is not dict
    ):
        raise ValueError("invalid writer reconciliation")
    applied = [_effect_from_data(effect) for effect in value["applied"]]
    restored = []
    for item in value["restored"]:
        if (
            type(item) is not dict
            or set(item) != {"effect", "reason"}
            or type(item["reason"]) is not str
        ):
            raise ValueError("invalid restored writer effect")
        restored.append((_effect_from_data(item["effect"]), item["reason"]))
    manifest = {}
    for path, entry in value["manifest"].items():
        if type(path) is not str:
            raise ValueError("invalid writer manifest path")
        parsed = _entry_from_data(entry)
        if parsed is None:
            raise ValueError("missing writer manifest entry")
        manifest[path] = parsed
    return Reconciliation(applied, restored, manifest)


def _replay_target(workspace: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or "\0" in relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {".", ".."} for part in pure.parts)
    ):
        raise HookReplayRefused("writer replay contains an invalid workspace path")
    target = Path(workspace)
    for part in pure.parts[:-1]:
        target /= part
        info = os.lstat(target)
        if not stat.S_ISDIR(info.st_mode):
            raise HookReplayRefused(
                f"writer replay path has a non-directory ancestor: {relative}"
            )
    return target / pure.parts[-1]


def _write_replay_file(
    workspace: Path, effect: WriterEffect, data: bytes, mode: int
) -> None:
    relative = effect.path
    try:
        target = _replay_target(workspace, relative)
        original = os.lstat(target)
        if (
            not stat.S_ISREG(original.st_mode)
            or original.st_nlink != 1
            or stat.S_IMODE(original.st_mode) != mode
            or _file_sha(target) != effect.pre_sha256
        ):
            raise HookReplayRefused(
                f"writer replay pre-state changed: {relative}"
            )
        fd, temporary = tempfile.mkstemp(prefix=".hook-replay-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, mode)
            current = os.lstat(target)
            if (
                (current.st_dev, current.st_ino)
                != (original.st_dev, original.st_ino)
                or not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or stat.S_IMODE(current.st_mode) != mode
                or _file_sha(target) != effect.pre_sha256
            ):
                raise HookReplayRefused(
                    f"writer replay pre-state changed while applying: {relative}"
                )
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    except HookReplayRefused:
        raise
    except (OSError, HookConfigError) as error:
        raise HookReplayRefused(
            f"writer replay could not apply recorded bytes to {relative}: {error}"
        ) from error


def _read_replay_pre_bytes(
    workspace: Path, effect: WriterEffect, mode: int, size: int
) -> bytes:
    target = _replay_target(workspace, effect.path)
    fd = os.open(
        target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != mode
        ):
            raise HookReplayRefused(f"writer replay pre-state changed: {effect.path}")
        data = source.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != effect.pre_sha256:
        raise HookReplayRefused(f"writer replay pre-state changed: {effect.path}")
    return data


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _labeled(value: object, roots: Mapping[str, str]) -> object:
    if isinstance(value, str):
        for path, label in sorted(roots.items(), key=lambda item: -len(item[0])):
            value = value.replace(path, label)
        return value
    if isinstance(value, dict):
        return {key: _labeled(item, roots) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_labeled(item, roots) for item in value]
    return value


class HookRunner:
    def __init__(
        self,
        run_dir: Path,
        hooks: HookSet,
        *,
        bundle_dir: Path,
        workspace: Path,
        shell_network: str,
        run_seed: str,
        run_id: str,
        version: str,
        replay: Callable[[], bool] = lambda: False,
        platform_backend: str | None = None,
        effect_policy: EffectPolicy | None = None,
        protected_paths: frozenset[str] = frozenset(),
    ):
        network_order = {"none": 0, "loopback": 1, "full": 2}
        if shell_network not in network_order or any(
            hook.network not in network_order
            or network_order[hook.network] > network_order[shell_network]
            or hook.writes == "changed"
            and (hook.event != "post_tool" or hook.network == "full" or not hook.changed)
            for hook in hooks.definitions
        ):
            raise HookConfigError("hook network exceeds --shell-network")
        if any(type(path) is not str or not path for path in protected_paths):
            raise ValueError("protected hook paths must be nonempty strings")
        self.run_dir = Path(run_dir).resolve(strict=True)
        self.hooks = hooks
        self.bundle_dir = Path(bundle_dir).resolve(strict=True)
        self.workspace = Path(workspace).resolve(strict=True)
        self.shell_network = shell_network
        self.run_seed = run_seed
        self.run_id = run_id
        self.version = version
        self.replay = replay
        self.platform_backend = platform_backend
        self.effect_policy = effect_policy
        self.protected_paths = frozenset(protected_paths)
        self.tape = HookTape(self.run_dir)
        self.sessions: dict[tuple[str, bool], ExecSession] = {}
        self.scratch_root = self.run_dir / "hook-scratch"
        self.scratch_root.mkdir(mode=0o700, exist_ok=True)
        self.records_dir = self.run_dir / "hooks"
        (self.records_dir / "out").mkdir(mode=0o700, parents=True, exist_ok=True)
        if any(
            path.is_symlink()
            for path in (self.scratch_root, self.records_dir, self.records_dir / "out")
        ):
            raise HookContainmentError("hook output directory must not be a symlink")
        self.driver = Path(__file__).with_name("kernel_hook_driver.py").resolve()

    def _capability(self, hook: HookDefinition, scratch: Path) -> ExecCapability:
        roots = [
            Path(p)
            for p in (
                "/usr",
                "/bin",
                "/sbin",
                "/lib",
                "/lib64",
                "/System",
                "/Library/Apple",
                "/private/var/select",
            )
            if Path(p).exists()
        ]
        roots.extend(
            (
                Path(sys.base_prefix),
                Path(sys.prefix),
                Path(sys.executable),
                self.driver,
                self.bundle_dir / str(hook.source_index),
            )
        )
        roots.extend(_macos_developer_roots())
        system_gitconfig = Path("/etc/gitconfig")
        if sys.platform == "darwin" and system_gitconfig.is_file():
            roots.extend(symlink_hops(system_gitconfig))
        if hook.executable:
            roots.extend(executable_read_roots(Path(hook.executable)))
        for item in hook.files:
            if item.executable:
                path = self.bundle_dir / str(hook.source_index) / item.relative
                with path.open("rb") as source:
                    first = source.readline(4096)
                if first.startswith(b"#!"):
                    words = first[2:].split()
                    if words:
                        roots.extend(executable_read_roots(Path(os.fsdecode(words[0]))))
        for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
            if (prefix / "Cellar").is_dir():
                roots.extend(
                    p
                    for name in ("Cellar", "opt", "lib")
                    if (p := prefix / name).is_dir()
                )
        if hook.network == "full":
            for name in (
                "/etc/hosts",
                "/etc/resolv.conf",
                "/etc/nsswitch.conf",
                "/etc/ssl",
                "/etc/pki",
            ):
                path = Path(name)
                if path.exists():
                    roots.append(path)
                    # ExecCapability requires symlinked roots' resolved targets
                    # to have explicit read authority as well.
                    roots.append(path.resolve(strict=True))
        executable_dir = (
            Path(hook.executable).parent
            if hook.executable
            else self.bundle_dir / str(hook.source_index)
        )
        env = {
            "PATH": os.pathsep.join(
                dict.fromkeys(
                    (str(executable_dir), "/usr/bin", "/bin", "/usr/sbin", "/sbin")
                )
            ),
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
            "LC_ALL": "C.UTF-8",
            "STATEFRAME_HOOK_EVENT": hook.event,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if sys.platform == "darwin" and not Path("/etc/gitconfig").is_file():
            # Same rule as the shell capability: an absent file outside the read roots yields EPERM, not ENOENT.
            env["GIT_CONFIG_NOSYSTEM"] = "1"
        return ExecCapability(
            "hook",
            self.workspace,
            hook.writes == "changed",
            tuple(roots),
            (scratch,),
            hook.network,
            env,
            ExecLimits(wall_seconds=hook.timeout + 10, output_bytes=256 << 10),
        )

    def _bundle_unchanged(self, hook: HookDefinition) -> bool:
        for item in hook.files:
            try:
                path = _regular_no_symlink(
                    self.bundle_dir, f"{hook.source_index}/{item.relative}"
                )
                mode = 0o500 if item.executable else 0o400
                if (
                    _file_sha(path, 1 << 20) != item.sha256
                    or path.stat().st_mode & 0o777 != mode
                ):
                    return False
            except (OSError, HookConfigError):
                return False
        return True

    def _session(
        self, network: str, writable: bool, cap: ExecCapability
    ) -> ExecSession:
        key = (network, writable)
        session = self.sessions.get(key)
        if session is None:
            session = None
            try:
                session = ExecSession(
                    self.run_dir, platform_backend=self.platform_backend
                )
                # The supervisor's Linux mount envelope is pinned once. Each target
                # still receives only its own fresh scratch directory as write authority.
                session.configure(replace(cap, write_roots=(self.scratch_root,)))
            except (ExecSessionError, OSError, ValueError) as error:
                if session is not None:
                    try:
                        session.close()
                    except (ExecSessionError, OSError) as cleanup_error:
                        raise HookContainmentError(
                            f"hook session setup cleanup failed: {cleanup_error}"
                        ) from cleanup_error
                raise _HookSessionSetupError(
                    "hook sandbox session could not be configured"
                ) from error
            self.sessions[key] = session
        return session

    def _record(self, value: dict) -> None:
        path = self.records_dir / "records.jsonl"
        fd = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("hook records path is not a regular file")
            with os.fdopen(fd, "ab", closefd=False) as stream:
                stream.write(
                    (
                        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
                stream.flush()
            os.fsync(fd)
        finally:
            os.close(fd)

    def record_delivery(
        self,
        event_ids: Sequence[str],
        *,
        request_id: str,
        admitted_bytes: int,
        omitted_bytes: int,
        dropped: bool,
    ) -> None:
        """Durably join hook invocations to the request that carried their text."""
        if not isinstance(event_ids, Sequence) or isinstance(event_ids, (str, bytes)):
            raise ValueError("invalid hook delivery event ids")
        ids = tuple(event_ids)
        if (
            type(request_id) is not str
            or not request_id
            or any(type(item) is not str or not item for item in ids)
            or type(admitted_bytes) is not int
            or admitted_bytes < 0
            or type(omitted_bytes) is not int
            or omitted_bytes < 0
            or type(dropped) is not bool
        ):
            raise ValueError("invalid hook delivery record")
        value = {
            "event_ids": list(ids),
            "request_id": request_id,
            "admitted_bytes": admitted_bytes,
            "omitted_bytes": omitted_bytes,
            "dropped": dropped,
        }
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        path = self.records_dir / "delivery.jsonl"
        created = not path.exists()
        try:
            fd = os.open(
                path,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    raise HookContainmentError("hook delivery path is not a private regular file")
                with os.fdopen(fd, "ab", closefd=False) as stream:
                    stream.write(encoded)
                    stream.flush()
                os.fsync(fd)
            finally:
                os.close(fd)
            if created:
                directory_fd = os.open(self.records_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except OSError as error:
            raise HookContainmentError(
                f"hook delivery record could not be written: {error}"
            ) from error

    def _store_streams(self, record_id: str, envelope: Mapping[str, object]) -> dict:
        digests = {}
        for field, suffix in (("stdout_b64", "out"), ("stderr_b64", "err")):
            raw = base64.b64decode(str(envelope[field]), validate=True)
            path = self.records_dir / "out" / f"{record_id}.{suffix}"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
            digests[suffix] = hashlib.sha256(raw).hexdigest()
        return digests

    def _writer_policy(self, paths: tuple[str, ...]) -> _ScopedWriterPolicy:
        if self.effect_policy is None:
            raise ValueError("writer effect policy is unavailable")
        return _ScopedWriterPolicy(
            self.effect_policy, frozenset(paths), self.protected_paths
        )

    def _restore_writer_workspace(
        self, hook: HookDefinition, before: Mapping[str, Entry], store: ContentStore
    ) -> None:
        try:
            after = scan(self.workspace, exclude=lambda _: False)
            if diff(before, after):
                restore_all = WorkspaceEffectPolicy(
                    lambda _: True, None, lambda _: False
                )
                reconcile(
                    self.workspace, before, after, restore_all, store, None
                )
        except (OSError, EffectError) as error:
            raise HookContainmentError(
                f"{hook.name}: writer workspace restoration failed: {error}"
            ) from error

    def _finish_writer(
        self,
        hook: HookDefinition,
        paths: tuple[str, ...],
        before: Mapping[str, Entry],
        store: ContentStore,
    ) -> tuple[WriterResult, dict]:
        if self.effect_policy is None:
            raise HookContainmentError(
                f"{hook.name}: writer executed without an effect policy"
            )
        try:
            after = scan(self.workspace, exclude=lambda _: False)
            reconciliation = reconcile(
                self.workspace,
                before,
                after,
                self._writer_policy(paths),
                store,
                None,
            )
            full_manifest = scan(self.workspace, exclude=lambda _: False)
            writer_store = ContentStore(self.run_dir / "hook-effects")
            authorized = []
            for effect in reconciliation.applied:
                if (
                    effect.before is None
                    or effect.after is None
                    or effect.before.kind != "file"
                    or effect.after.kind != "file"
                    or effect.before.digest is None
                    or effect.after.digest is None
                    or effect.path not in paths
                    or effect.before.mode != effect.after.mode
                    or effect.change != "modified"
                ):
                    raise EffectError(
                        f"scoped writer policy admitted an invalid effect: {effect.path}"
                    )
                post_bytes = store.get(effect.after.digest)
                if hashlib.sha256(post_bytes).hexdigest() != effect.after.digest:
                    raise EffectError(
                        f"post-writer content does not match its digest: {effect.path}"
                    )
                writer_store.put(post_bytes)
                authorized.append(
                    WriterEffect(
                        effect.path, effect.before.digest, effect.after.digest
                    )
                )
        except (OSError, EffectError, ValueError) as error:
            raise HookContainmentError(
                f"{hook.name}: writer effect reconciliation failed: {error}"
            ) from error

        restored = tuple(effect.path for effect, _ in reconciliation.restored)
        result = WriterResult(
            hook.name, tuple(authorized), restored, reconciliation
        )
        record = {
            "authorized": [asdict(effect) for effect in authorized],
            "restored": list(restored),
            "pre_digests": {
                path: before[path].digest for path in paths
            },
            "pre_modes": {path: before[path].mode for path in paths},
            "pre_workspace_digest": manifest_digest(before),
            "post_workspace_digest": manifest_digest(full_manifest),
            "reconciliation": _reconciliation_data(reconciliation),
        }
        return result, record

    def _replay_writer(
        self,
        hook: HookDefinition,
        record: object,
        paths: tuple[str, ...],
        before: Mapping[str, Entry],
    ) -> WriterResult:
        try:
            if self.effect_policy is None:
                raise HookReplayRefused(
                    f"hook {hook.name} writer has no effect policy on replay"
                )
            if (
                type(record) is not dict
                or set(record)
                != {
                    "authorized",
                    "restored",
                    "pre_digests",
                    "pre_modes",
                    "pre_workspace_digest",
                    "post_workspace_digest",
                    "reconciliation",
                }
                or type(record["authorized"]) is not list
                or type(record["restored"]) is not list
                or type(record["pre_digests"]) is not dict
                or type(record["pre_modes"]) is not dict
                or type(record["pre_workspace_digest"]) is not str
                or type(record["post_workspace_digest"]) is not str
            ):
                raise ValueError("invalid writer completion")
            if manifest_digest(before) != record["pre_workspace_digest"]:
                raise HookReplayRefused(
                    f"hook {hook.name} workspace pre-state changed at replay"
                )
            if set(record["pre_digests"]) != set(paths) or set(
                record["pre_modes"]
            ) != set(paths):
                raise ValueError("writer path set changed")
            for path in paths:
                entry = before.get(path)
                if (
                    entry is None
                    or entry.kind != "file"
                    or entry.digest != record["pre_digests"][path]
                    or type(record["pre_modes"][path]) is not int
                    or entry.mode != record["pre_modes"][path]
                ):
                    raise HookReplayRefused(
                        f"hook {hook.name} pre-digest changed for {path}"
                    )

            reconciliation = _reconciliation_from_data(record["reconciliation"])
            authorized = []
            seen = set()
            writes = []
            writer_store = ContentStore(self.run_dir / "hook-effects")
            policy = self._writer_policy(paths)
            applied_by_path = {effect.path: effect for effect in reconciliation.applied}
            if len(applied_by_path) != len(reconciliation.applied):
                raise ValueError("duplicate applied writer path")
            for item in record["authorized"]:
                if (
                    type(item) is not dict
                    or set(item) != {"path", "pre_sha256", "post_sha256"}
                    or any(type(item[key]) is not str for key in item)
                    or item["path"] in seen
                    or item["path"] not in paths
                ):
                    raise ValueError("invalid authorized writer effect")
                seen.add(item["path"])
                effect = applied_by_path.get(item["path"])
                if (
                    effect is None
                    or effect.change != "modified"
                    or effect.before is None
                    or effect.after is None
                    or effect.before.kind != "file"
                    or effect.after.kind != "file"
                    or effect.before.mode != effect.after.mode
                    or effect.before.digest != item["pre_sha256"]
                    or effect.after.digest != item["post_sha256"]
                ):
                    raise ValueError("writer receipt does not match reconciliation")
                allowed, _ = policy.authorize(effect)
                if not allowed:
                    raise HookReplayRefused(
                        f"hook {hook.name} writer effect is no longer authorized: {item['path']}"
                    )
                if type(item["pre_sha256"]) is not str or type(
                    item["post_sha256"]
                ) is not str:
                    raise ValueError("invalid writer digest")
                data = writer_store.get(item["post_sha256"])
                if hashlib.sha256(data).hexdigest() != item["post_sha256"]:
                    raise ValueError("recorded writer bytes do not match digest")
                authorized.append(WriterEffect(**item))
                writes.append((authorized[-1], data, effect.before.mode))
            if seen != set(applied_by_path):
                raise ValueError("writer receipts do not cover applied effects")
            restored = tuple(effect.path for effect, _ in reconciliation.restored)
            if record["restored"] != list(restored):
                raise ValueError("writer restored paths do not match reconciliation")
            pre_bytes = {
                effect.path: _read_replay_pre_bytes(
                    self.workspace, effect, mode, before[effect.path].size
                )
                for effect, _, mode in writes
            }
            try:
                for effect, data, mode in writes:
                    _write_replay_file(self.workspace, effect, data, mode)
                post_manifest = scan(self.workspace, exclude=lambda _: False)
                if manifest_digest(post_manifest) != record["post_workspace_digest"]:
                    raise HookReplayRefused(
                        f"hook {hook.name} post-state changed during replay"
                    )
            except BaseException as error:
                try:
                    for effect, _, mode in reversed(writes):
                        target = _replay_target(self.workspace, effect.path)
                        current = os.lstat(target)
                        if (
                            not stat.S_ISREG(current.st_mode)
                            or current.st_nlink != 1
                            or stat.S_IMODE(current.st_mode) != mode
                        ):
                            raise HookReplayRefused(
                                f"writer replay restoration found changed file type or mode: {effect.path}"
                            )
                        current_sha = _file_sha(target)
                        if current_sha == effect.pre_sha256:
                            continue
                        if current_sha != effect.post_sha256:
                            raise HookReplayRefused(
                                f"writer replay restoration found unknown bytes: {effect.path}"
                            )
                        _write_replay_file(
                            self.workspace,
                            WriterEffect(effect.path, effect.post_sha256, effect.pre_sha256),
                            pre_bytes[effect.path],
                            mode,
                        )
                    restored_manifest = scan(self.workspace, exclude=lambda _: False)
                    if manifest_digest(restored_manifest) != record["pre_workspace_digest"]:
                        raise HookReplayRefused(
                            f"hook {hook.name} workspace pre-state could not be restored"
                        )
                except BaseException as restore_error:
                    raise HookContainmentError(
                        f"{hook.name}: writer replay restoration failed: {restore_error}"
                    ) from restore_error
                raise HookReplayRefused(
                    f"hook {hook.name} writer replay could not apply recorded bytes: {error}"
                ) from error
            return WriterResult(hook.name, tuple(authorized), restored, reconciliation)
        except HookReplayRefused:
            raise
        except (KeyError, TypeError, ValueError, OSError, EffectError) as error:
            raise HookReplayRefused(
                f"hook {hook.name} writer record is invalid: {error}"
            ) from error

    def _execute(
        self, hook: HookDefinition, argv: list[str], payload: bytes, cap: ExecCapability
    ) -> tuple[HookOutcome, dict | None, float]:
        request = {
            "argv": argv,
            "payload_b64": base64.b64encode(payload).decode("ascii"),
            "timeout": hook.timeout,
            "stream_cap": 32768,
        }
        rpc_size = (
            len(
                json.dumps(
                    {
                        "op": "start",
                        "argv": [sys.executable, "-I", str(self.driver)],
                        "cap": cap.as_dict(),
                        "cwd": str(self.workspace),
                        "stdin": base64.b64encode(
                            json.dumps(request, separators=(",", ":")).encode()
                        ).decode(),
                    },
                    separators=(",", ":"),
                ).encode()
            )
            + 1
        )
        if rpc_size > 1 << 20:
            return (
                _error_outcome(
                    "input_size", "hook invocation exceeds execution request bound"
                ),
                None,
                0.0,
            )
        start = time.monotonic()
        try:
            session = self._session(
                hook.network, hook.writes == "changed", cap
            )
        except _HookSessionSetupError:
            return (
                _error_outcome("session", "hook sandbox session unavailable"),
                None,
                time.monotonic() - start,
            )
        try:
            result = session.run(
                [sys.executable, "-I", str(self.driver)],
                cap,
                stdin=json.dumps(request, separators=(",", ":")).encode(),
                cwd=self.workspace,
            )
        except ExecSessionError as error:
            message = str(error).lower()
            if (
                "no unsandboxed fallback" in message
                or "sandbox unavailable" in message
                or "requires bwrap" in message
            ):
                return (
                    _error_outcome("sandbox_unavailable", "hook sandbox unavailable"),
                    None,
                    time.monotonic() - start,
                )
            raise HookContainmentError(
                f"{hook.name}: execution supervisor failure after launch: {error}"
            ) from error
        if (
            not result.cleanup.ok
            or result.output_truncated
            or result.timed_out
            or result.exit_code != 0
            or result.output_total_bytes != len(result.output)
        ):
            raise HookContainmentError(
                f"{hook.name}: driver transport or cleanup failed"
            )
        try:
            envelope = json.loads(result.output)
        except (ValueError, UnicodeError) as error:
            raise HookContainmentError(
                f"{hook.name}: malformed driver envelope"
            ) from error
        outcome = parse_outcome(hook.event, envelope)
        if outcome.decision == "containment":
            raise HookContainmentError(f"{hook.name}: {outcome.reason}")
        return outcome, envelope, result.wall_seconds

    def run_event(
        self,
        event: str,
        *,
        boundary: str,
        data: Mapping[str, object],
        match: Mapping[str, object] | None = None,
        label: str = "",
    ) -> EventResult:
        token = _RUN_PATH.set(str(self.run_dir))
        try:
            return self._run_event(
                event, boundary=boundary, data=data, match=match, label=label
            )
        finally:
            _RUN_PATH.reset(token)

    def _run_event(
        self,
        event: str,
        *,
        boundary: str,
        data: Mapping[str, object],
        match: Mapping[str, object] | None,
        label: str,
    ) -> EventResult:
        if event not in (
            "session_start",
            "prompt_submit",
            "pre_tool",
            "post_tool",
            "stop",
            "session_end",
        ):
            raise ValueError(f"invalid hook event: {event}")
        active = self.hooks.for_event(event)
        if not active:
            return EventResult("continue", None, None, "", (), ())
        match_values = dict(match or {})
        applied_paths_available = match_values.pop("applied_paths_available", True)
        if type(applied_paths_available) is not bool:
            raise ValueError("post_tool applied-path availability must be a boolean")
        selected: list[tuple[int, HookDefinition]] = []
        for index, hook in enumerate(self.hooks.definitions):
            if (
                event == "post_tool"
                and hook.event == event
                and hook.writes == "changed"
                and not applied_paths_available
                and isinstance(match_values.get("tool"), str)
                and any(
                    fnmatch.fnmatchcase(match_values["tool"], pattern)
                    for pattern in hook.tools
                )
            ):
                # Unknown effect provenance cannot narrow a writer's authority.
                # Record the writer as skipped without scanning the workspace.
                selected.append((index, hook))
            elif hook.event == event and matches(hook, **match_values):
                selected.append((index, hook))
        runs: list[HookRunResult] = []
        entries: list[PanelEntry] = []
        writes: list[WriterResult] = []
        for index, hook in selected:
            ident = event_id(self.run_seed, boundary, index)
            hook_data = dict(data)
            changed_paths = tuple(
                match_values.get("changed_paths", data.get("changed_paths", ()))
            )
            deleted_paths = set(
                match_values.get("deleted_paths", data.get("deleted_paths", ()))
            )
            matching_paths = tuple(
                path
                for path in changed_paths
                if path not in deleted_paths
                and (
                    not hook.paths
                    or type(path) is not str
                    or any(fnmatch.fnmatchcase(path, pattern) for pattern in hook.paths)
                )
            )
            if event == "post_tool" and hook.paths:
                hook_data["changed_paths"] = list(matching_paths)
            attempted_paths = matching_paths
            if event == "pre_tool" and hook.paths:
                hook_data["paths"] = [
                    path
                    for path in match_values.get("arg_paths", ())
                    if type(path) is str
                    and any(
                        fnmatch.fnmatchcase(path, pattern) for pattern in hook.paths
                    )
                ]
            path_error = None
            is_replay = self.replay()
            before = None
            store = None
            writer_paths: tuple[str, ...] = ()
            writer_skip = None
            if hook.writes == "changed":
                if not applied_paths_available:
                    writer_skip = "applied_effects_unavailable"
                else:
                    try:
                        before = scan(self.workspace, exclude=lambda _: False)
                        writer_paths = tuple(
                            sorted(
                                {
                                    path
                                    for path in matching_paths
                                    if type(path) is str
                                    and before.get(path) is not None
                                    and before[path].kind == "file"
                                }
                            )
                        )
                    except (OSError, EffectError) as error:
                        raise HookContainmentError(
                            f"{hook.name}: writer workspace preflight failed: {error}"
                        ) from error
                matching_paths = writer_paths
                hook_data["changed_paths"] = list(writer_paths)
                if writer_skip is not None:
                    pass
                elif not writer_paths:
                    writer_skip = "empty_write_set"
                elif self.effect_policy is None:
                    writer_skip = "writer_policy"
                elif not is_replay:
                    try:
                        store = ContentStore(self.run_dir)
                        store.capture(self.workspace, before)
                    except (OSError, EffectError) as error:
                        raise HookContainmentError(
                            f"{hook.name}: writer pre-state capture failed: {error}"
                        ) from error

            if "{paths}" in hook.argv:
                # Validate the exact list the hook receives, including its limits.
                try:
                    resolve_argv(hook, self.bundle_dir, paths=matching_paths)
                except HookArgvError as error:
                    if hook.writes == "changed" and writer_skip is None and not writer_paths:
                        writer_skip = "empty_write_set"
                    else:
                        path_error = error
                if hook.writes != "changed":
                    matching_paths = tuple(
                        path
                        for path in matching_paths
                        if (self.workspace / path).exists()
                    )
                    if not matching_paths and path_error is None:
                        continue
                    hook_data["changed_paths"] = list(
                        attempted_paths
                        if path_error
                        else matching_paths
                    )

            payload = build_payload(
                event,
                hook,
                event_id=ident,
                run_id=self.run_id,
                cwd=str(self.workspace),
                data=hook_data,
                version=self.version,
            )
            scratch = Path(tempfile.mkdtemp(prefix="hook-", dir=self.scratch_root))
            try:
                if writer_skip is not None:
                    cap = None
                    cap_error = None
                else:
                    try:
                        cap = self._capability(hook, scratch)
                    except (OSError, ValueError) as error:
                        cap = None
                        cap_error = error
                    else:
                        cap_error = None
                labels = {
                    str(self.workspace): "<workspace>",
                    str(self.run_dir): "<run>",
                    str(self.bundle_dir): "<bundle>",
                    str(scratch): "<scratch>",
                }
                identity = {
                    "event_id": ident,
                    "hook": hook.name,
                    "event": event,
                    "definition": hook.definition_digest(),
                    "input": payload.identity_digest,
                    "capability": (
                        _digest(_labeled(cap.as_dict(), labels))
                        if cap is not None
                        else "not_required"
                        if writer_skip is not None
                        else "unavailable"
                    ),
                }
                if hook.writes == "changed":
                    identity.update(
                        writer_set=list(writer_paths),
                        writer_pre_digests={
                            path: before[path].digest for path in writer_paths
                        },
                        writer_pre_modes={
                            path: before[path].mode for path in writer_paths
                        },
                        writer_pre_workspace_digest=(
                            manifest_digest(before) if before is not None else None
                        ),
                        writer_policy=self.effect_policy is not None,
                        writer_protected_paths=sorted(self.protected_paths),
                    )
                # A replay never starts a sandbox or rechecks a mutable executable.
                writer_result = None
                writer_record = None
                if is_replay:
                    record = self.tape.lookup(ident)
                    if record is None or not record["completed"]:
                        raise HookReplayRefused(
                            f"hook {hook.name} has no completed record at {boundary}"
                        )
                    if record["identity"] != identity:
                        raise HookReplayRefused(
                            f"hook {hook.name} input or definition changed at {boundary}"
                        )
                    try:
                        outcome = HookOutcome(**record["result"]["outcome"])
                        skipped = record["result"].get("skipped")
                        duration = record["result"].get("duration", 0.0)
                        executed = record["result"].get("executed", False)
                        stream_digests = record["result"].get("stream_digests")
                        cleanup_ok = record["result"].get("cleanup_ok")
                        writer_record = record["result"].get("writer")
                    except (KeyError, TypeError, ValueError) as error:
                        raise HookReplayRefused(
                            f"hook {hook.name} has a malformed completion"
                        ) from error
                    if hook.writes == "changed":
                        if writer_record is not None:
                            if before is None:
                                raise HookReplayRefused(
                                    f"hook {hook.name} writer has no replay pre-state"
                                )
                            writer_result = self._replay_writer(
                                hook, writer_record, writer_paths, before
                            )
                        elif executed:
                            raise HookReplayRefused(
                                f"hook {hook.name} writer completion has no effects record"
                            )
                    replayed = True
                else:
                    self.tape.intent(ident, identity)
                    skipped = None
                    replayed = False
                    if before is None and writer_skip != "applied_effects_unavailable":
                        try:
                            before = scan(self.workspace, exclude=lambda _: False)
                            store = ContentStore(self.run_dir)
                            store.capture(self.workspace, before)
                        except (OSError, EffectError) as error:
                            raise HookContainmentError(
                                f"{hook.name}: workspace preflight failed: {error}"
                            ) from error
                    envelope = None
                    duration = 0.0
                    executed = False
                    stream_digests = None
                    cleanup_ok = None
                    try:
                        if writer_skip in {
                            "empty_write_set", "applied_effects_unavailable",
                        }:
                            outcome = HookOutcome(
                                "continue", "", "", None, None, None, False, 0, 0, 0, 0
                            )
                            skipped = writer_skip
                        elif writer_skip == "writer_policy":
                            outcome = _error_outcome(
                                "writer_policy",
                                "writer hooks require a workspace effect policy",
                            )
                            skipped = writer_skip
                        elif payload.error:
                            outcome = _error_outcome(
                                payload.error,
                                "hook input exceeds 256 KiB"
                                if payload.error == "input_size"
                                else "hook input is not canonical JSON",
                            )
                            skipped = payload.error
                        elif cap_error is not None:
                            outcome = _error_outcome(
                                "capability",
                                "hook execution capability unavailable",
                            )
                            skipped = "capability"
                        else:
                            if path_error is not None:
                                raise path_error
                            check_operands(hook, self.workspace)
                            argv = resolve_argv(
                                hook, self.bundle_dir, paths=matching_paths
                            )
                            if __package__:
                                from .kernel_hooks import _execution_pin
                            else:
                                from kernel_hooks import _execution_pin

                            try:
                                pinned = _execution_pin(
                                    hook.executable,
                                    hook.files,
                                    bundle_dir=self.bundle_dir,
                                    source_index=hook.source_index,
                                )
                            except (OSError, HookConfigError):
                                pinned = None
                            if (
                                pinned != hook.executable_digest
                                or not self._bundle_unchanged(hook)
                            ):
                                outcome = _error_outcome(
                                    "changed",
                                    "hook executable, interpreter or bundle changed since load",
                                )
                                skipped = "changed"
                            else:
                                outcome, envelope, duration = self._execute(
                                    hook, argv, payload.payload, cap
                                )
                                executed = (
                                    envelope is not None
                                    and envelope.get("launch_error") is None
                                )
                                if outcome.error_code == "input_size":
                                    skipped = "input_size"
                                elif outcome.error_code == "session":
                                    skipped = "session"
                                elif outcome.error_code == "sandbox_unavailable":
                                    skipped = "sandbox_unavailable"
                    except HookArgvError as error:
                        outcome = _error_outcome("argv", str(error))
                        skipped = "argv"
                    finally:
                        if (before is not None
                                and writer_skip != "applied_effects_unavailable"):
                            try:
                                after = scan(self.workspace, exclude=lambda _: False)
                                if hook.writes == "changed":
                                    if executed:
                                        if store is None:
                                            raise HookContainmentError(
                                                f"{hook.name}: writer pre-state is unavailable"
                                            )
                                        try:
                                            writer_result, writer_record = (
                                                self._finish_writer(
                                                    hook, writer_paths, before, store
                                                )
                                            )
                                        except HookContainmentError:
                                            if store is not None:
                                                self._restore_writer_workspace(
                                                    hook, before, store
                                                )
                                            raise
                                    elif diff(before, after):
                                        if store is None:
                                            raise HookContainmentError(
                                                f"{hook.name}: unexpected workspace change without a restore store"
                                            )
                                        self._restore_writer_workspace(
                                            hook, before, store
                                        )
                                        raise HookContainmentError(
                                            f"{hook.name}: writer changed workspace before launch"
                                        )
                                elif diff(before, after):
                                    policy = WorkspaceEffectPolicy(
                                        lambda _: True, None, lambda _: False
                                    )
                                    reconcile(
                                        self.workspace, before, after, policy, store, None
                                    )
                                    raise HookContainmentError(
                                        f"{hook.name}: hook changed workspace"
                                    )
                            except HookContainmentError:
                                raise
                            except (OSError, EffectError) as error:
                                raise HookContainmentError(
                                    f"{hook.name}: workspace inspection or restoration failed: {error}"
                                ) from error
                    if envelope is not None:
                        try:
                            stream_digests = self._store_streams(ident, envelope)
                            cleanup_ok = envelope["cleanup_ok"]
                        except (OSError, ValueError) as error:
                            raise HookContainmentError(
                                f"{hook.name}: raw stream recording failed: {error}"
                            ) from error
                    try:
                        self.tape.complete(
                            ident,
                            {
                                "outcome": asdict(outcome),
                                "skipped": skipped,
                                "duration": duration,
                                "reduced": payload.reduced,
                                "executed": executed,
                                "stream_digests": stream_digests,
                                "cleanup_ok": cleanup_ok,
                                "writer": writer_record,
                            },
                        )
                    except (OSError, ValueError) as error:
                        raise HookContainmentError(
                            f"{hook.name}: hook completion recording failed: {error}"
                        ) from error
                try:
                    self._record(
                        {
                            "event_id": ident,
                            "hook": hook.name,
                            "event": event,
                            "matched": True,
                            "executed": executed and not replayed,
                            "replayed": replayed,
                            "skipped": skipped,
                            "decision": outcome.decision,
                            "error_code": outcome.error_code,
                            "exit_code": outcome.exit_code,
                            "signal": outcome.signal,
                            "timed_out": outcome.timed_out,
                            "duration": duration,
                            "input_digest": payload.identity_digest,
                            "definition_digest": hook.definition_digest(),
                            "capability_digest": identity["capability"],
                            "input_bytes": len(payload.payload or b""),
                            "stdout_total": outcome.stdout_total,
                            "stderr_total": outcome.stderr_total,
                            "stream_digests": stream_digests,
                            "cleanup_ok": cleanup_ok,
                            "capability": {
                                "network": hook.network,
                                "workspace_writable": hook.writes == "changed",
                                "write_roots": 1,
                            },
                            "request_id": None,
                            "admitted_text_bytes": len(
                                (outcome.reason + outcome.context).encode()
                            ),
                            "omitted_text_bytes": outcome.omitted_bytes,
                            "replaced_literals": outcome.replaced_literals,
                            "candidate_digest": data.get("candidate_digest")
                            if event == "stop"
                            else None,
                        }
                    )
                except (OSError, ValueError) as error:
                    raise HookContainmentError(
                        f"{hook.name}: hook record write failed: {error}"
                    ) from error
                runs.append(
                    HookRunResult(hook.name, event, ident, outcome, replayed, skipped)
                )
                if writer_result is not None:
                    writes.append(writer_result)
                if outcome.context and event in (
                    "session_start",
                    "prompt_submit",
                    "post_tool",
                    "session_end",
                ):
                    entries.append(
                        PanelEntry(
                            "note" if event == "post_tool" else "start",
                            hook.name,
                            event,
                            label,
                            outcome.context,
                        )
                    )
                if event in GATING_EVENTS and (
                    outcome.decision == "deny"
                    or outcome.decision == "error"
                    and hook.on_error == "block"
                ):
                    return EventResult(
                        "deny",
                        hook.name,
                        "HOOK_DENIED" if outcome.decision == "deny" else "HOOK_FAILED",
                        outcome.reason,
                        tuple(entries),
                        tuple(runs),
                        tuple(writes),
                    )
            finally:
                if scratch.exists():
                    try:
                        shutil.rmtree(scratch)
                    except OSError as error:
                        raise HookContainmentError(
                            f"{hook.name}: scratch cleanup failed: {error}"
                        ) from error
        return EventResult(
            "continue", None, None, "", tuple(entries), tuple(runs), tuple(writes)
        )

    def close(self) -> None:
        failures = []
        for session in self.sessions.values():
            try:
                session.close()
            except ExecSessionError as error:
                failures.append(str(error))
        if failures:
            raise HookContainmentError(
                "hook session cleanup failed: " + "; ".join(failures)
            )


def _error_outcome(code: str, message: str) -> HookOutcome:
    cleaned = sanitize(message, limit=1024)
    return HookOutcome(
        "error",
        cleaned.text,
        "",
        code,
        None,
        None,
        False,
        0,
        0,
        cleaned.replaced_literals,
        cleaned.omitted_bytes,
    )
