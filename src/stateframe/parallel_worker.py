"""Prepare and launch ordinary engine children for compiled parallel subframes."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Mapping

from .endpoint import CHAT_SUFFIX, EndpointError, normalize_base_url
from .engine.kernel_parallel_admission import DirectorySnapshot
from .engine.kernel_custom_tools import load_tool_files
from .engine.kernel_parallel_records import UnitSpec
from .engine.kernel_tool_settings import ToolSettings, ToolSettingsError
from .engine.kernel_exec_tracker import (ContainmentUnverified, Descendants, LaunchIdentity, ProcessIdentity,
                                          ProcessTable, capture_launch_identity,
                                          child_exited_unreaped, recorded_launch_identity,
                                          signal_launch_group, kill_launch_unit, live_group_rows,
                                          load_tolerant_seconds)


# Parent process values admitted at the subframe spawn boundary. Run-specific values and the
# provider credential names are copied separately below from the recorded worker spec.
CHILD_ENV_ALLOWLIST = frozenset({
    "PATH",  # Resolve only the child tools selected by the run.
    "HOME",  # Give user-scoped tools a stable home directory.
    "LANG",  # Preserve the run's locale and default text encoding.
    "LC_ALL",  # Preserve an explicit locale override.
    "LC_CTYPE",  # Preserve locale-aware text decoding.
    "TMPDIR",  # Keep temporary files in the configured workspace temp area.
    "USER",  # Preserve the operating-system user identity for child tools.
    "LOGNAME",  # Keep login identity consistent with USER.
    "SHELL",  # Honor the configured shell for allowed shell tools.
    "SYSTEMROOT",  # Required by subprocess startup on Windows.
    "TERM",  # Preserve terminal capability selection for child tools.
    "LLAMA_CPP_TOKENIZE_URL",  # Keep the run's explicitly selected local tokenizer endpoint.
    "STATEFRAME_TEST_PYTHON",  # Use the run's recorded interpreter for project checks.
    "PYTHONHASHSEED",  # Set hash randomization before the engine interpreter starts.
    "PYTHONDONTWRITEBYTECODE",  # Prevent the child from writing bytecode into its workspace.
    "PYTHONUNBUFFERED",  # Flush child progress promptly to the worker log.
    "SF_SUBFRAME_INCARNATION",  # Worker-set identity marker for this launch attempt.
})


def kernel_start_identity(pid: int, *, row: ProcessIdentity | None = None) -> str | None:
    """Return a kernel birth token, independent of wall time and PID reuse."""
    row = ProcessTable().identity(pid) if row is None else row
    if row is None or row.start is None:
        return None
    if sys.platform == "linux":
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", boot_id):
            raise WorkerLaunchError("Linux boot identity is invalid")
        return f"linux:{boot_id}:{row.start[0]}"
    if sys.platform == "darwin":
        return f"darwin:{row.start[0]}:{row.start[1]}"
    raise WorkerLaunchError(f"subframe process identity is unsupported on {sys.platform}")


def mac_process_argv(pid: int) -> list[str] | None:
    """Read KERN_PROCARGS2, retaining the argument boundaries lost by ps."""
    mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2, pid
    libc = ctypes.CDLL(None, use_errno=True)
    libc.sysctl.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint,
                            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                            ctypes.c_void_p, ctypes.c_size_t]
    size = ctypes.c_size_t()
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            return None
        raise OSError(error, "cannot size subframe process arguments", pid)
    if size.value < 8 or size.value > 1 << 20:
        raise WorkerLaunchError("subframe process argument size is invalid")
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            return None
        raise OSError(error, "cannot read subframe process arguments", pid)
    data = buf.raw[:size.value]
    count = int.from_bytes(data[:4], sys.byteorder, signed=True)
    if count < 1 or count > 4096:
        raise WorkerLaunchError("subframe process argument count is invalid")
    cursor = data.find(b"\0", 4)
    if cursor < 0:
        raise WorkerLaunchError("subframe process argument header is invalid")
    cursor += 1
    while cursor < len(data) and data[cursor] == 0:
        cursor += 1
    argv = []
    for _ in range(count):
        end = data.find(b"\0", cursor)
        if end < 0:
            raise WorkerLaunchError("subframe process arguments are truncated")
        argv.append(os.fsdecode(data[cursor:end]))
        cursor = end + 1
    return argv


def run_wall_seconds(spec: Mapping[str, object]) -> float | None:
    """The run wall named by a worker spec: None when unset, an error when not numeric."""
    raw = (spec.get("environment") or {}).get("STATEFRAME_RUN_WALL_SECONDS")
    if not isinstance(raw, str):
        return None
    try:
        return float(raw)
    except ValueError as error:
        raise WorkerLaunchError("subframe run-wall setting is not numeric") from error


def cleanup_seconds() -> float:
    """Allow scheduler pressure to delay exit without declaring containment lost."""
    return load_tolerant_seconds("STATEFRAME_PARALLEL_CLEANUP_SECONDS")


class WorkerPreparationError(RuntimeError):
    """A subframe spec or snapshot cannot safely produce an ordinary subframe run."""


class WorkerLaunchError(RuntimeError):
    """A prepared subframe could not be started or supervised."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise WorkerPreparationError(f"subframe spec is not JSON serializable: {error}") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise WorkerPreparationError(f"refusing to overwrite existing subframe artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_canonical_json(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise WorkerPreparationError(f"subframe artifact appeared during write: {path}") from error
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path, label: str) -> dict:
    try:
        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise WorkerPreparationError(f"{label} has duplicate JSON key {key!r}")
                result[key] = item
            return result

        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkerPreparationError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise WorkerPreparationError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class SnapshotManifest:
    digest: str
    source_digest: str
    files: dict[str, int]
    file_digests: dict[str, str]
    directories: dict[str, int]


def read_snapshot_manifest(root: Path, expected_digest: str, *,
                           verify_contents: bool = True,
                           verify_inventory: bool = True) -> SnapshotManifest:
    """Authenticate frozen snapshot bytes and recover the source's original modes."""
    root = Path(root)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise WorkerPreparationError("snapshot manifest digest must be a lowercase sha256")
    if root.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise WorkerPreparationError("subframe start snapshot root is not read-only")
    manifest_path = root / "snapshot-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise WorkerPreparationError("subframe snapshot manifest is missing or unsafe")
    try:
        descriptor = os.open(manifest_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > (16 << 20):
                raise WorkerPreparationError("subframe snapshot manifest is not a bounded regular file")
            chunks = []
            remaining = (16 << 20) + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1 << 20))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        if len(raw) > 16 << 20:
            raise WorkerPreparationError("subframe snapshot manifest exceeds 16 MiB")
        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise WorkerPreparationError(f"subframe snapshot manifest has duplicate JSON key {key!r}")
                result[key] = item
            return result

        value = json.loads(raw, object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkerPreparationError(f"subframe snapshot manifest is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise WorkerPreparationError("subframe snapshot manifest must be a JSON object")
    if (set(value) != {"schema", "files", "directories", "source_manifest_digest"}
            or value["schema"] != "stateframe-snapshot-manifest/1"
            or not isinstance(value["files"], list) or not isinstance(value["directories"], list)
            or not isinstance(value["source_manifest_digest"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["source_manifest_digest"])):
        raise WorkerPreparationError("subframe snapshot manifest has invalid fields")
    digest = hashlib.sha256(_canonical_json(value)).hexdigest()
    if digest != expected_digest:
        raise WorkerPreparationError("subframe snapshot manifest digest mismatch")

    files: dict[str, int] = {}
    file_digests: dict[str, str] = {}
    for row in value["files"]:
        if (not isinstance(row, dict) or set(row) != {"path", "kind", "mode", "sha256"}
                or row["kind"] != "file" or not isinstance(row["path"], str)
                or type(row["mode"]) is not int or not 0 <= row["mode"] <= 0o7777
                or not isinstance(row["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise WorkerPreparationError("subframe snapshot manifest has an invalid file entry")
        name = row["path"]
        if (not name or name.startswith("/") or "\\" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or name == "snapshot-manifest.json" or name in files):
            raise WorkerPreparationError("subframe snapshot manifest has an unsafe file path")
        if verify_inventory:
            path = root / name
            if (path.is_symlink() or not path.is_file()
                    or stat.S_IMODE(path.stat().st_mode) != 0o444
                    or (verify_contents
                        and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"])):
                raise WorkerPreparationError(f"subframe snapshot file differs: {name}")
        files[name] = row["mode"]
        file_digests[name] = row["sha256"]
    if list(files) != sorted(files):
        raise WorkerPreparationError("subframe snapshot file inventory is not sorted")
    source_digest = hashlib.sha256(b"".join(_canonical_json(row) + b"\n" for row in value["files"])).hexdigest()
    if source_digest != value["source_manifest_digest"]:
        raise WorkerPreparationError("subframe snapshot source digest mismatch")

    directories: dict[str, int] = {}
    for row in value["directories"]:
        if (not isinstance(row, dict) or set(row) != {"path", "mode"}
                or not isinstance(row["path"], str) or type(row["mode"]) is not int
                or not 0 <= row["mode"] <= 0o7777):
            raise WorkerPreparationError("subframe snapshot manifest has an invalid directory entry")
        name = row["path"]
        if (not name or name.startswith("/") or "\\" in name
                or any(part in {"", ".", ".."} for part in name.split("/")) or name in directories):
            raise WorkerPreparationError("subframe snapshot manifest has an unsafe directory path")
        if verify_inventory:
            path = root / name
            if path.is_symlink() or not path.is_dir() or stat.S_IMODE(path.stat().st_mode) != 0o555:
                raise WorkerPreparationError(f"subframe snapshot directory differs: {name}")
        directories[name] = row["mode"]
    if list(directories) != sorted(directories):
        raise WorkerPreparationError("subframe snapshot directory inventory is not sorted")
    if verify_inventory:
        actual = list(root.rglob("*"))
        if (any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in actual)
                or set(files) != {path.relative_to(root).as_posix() for path in actual
                                  if path.is_file() and path != manifest_path}
                or set(directories) != {path.relative_to(root).as_posix() for path in actual if path.is_dir()}):
            raise WorkerPreparationError("subframe snapshot inventory differs from its manifest")
    return SnapshotManifest(digest, source_digest, files, file_digests, directories)


def _restore_snapshot_modes(root: Path, manifest: SnapshotManifest) -> None:
    try:
        if root.is_symlink() or not root.is_dir():
            raise WorkerPreparationError("cloned subframe snapshot root is unsafe")
        root.chmod(0o755)
        manifest_path = root / "snapshot-manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise WorkerPreparationError("cloned subframe snapshot manifest is unsafe")
        manifest_path.unlink()
        for name, mode in manifest.files.items():
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise WorkerPreparationError(f"cloned subframe file is unsafe: {name}")
            path.chmod(mode)
        for name, mode in sorted(manifest.directories.items(), key=lambda item: (-item[0].count("/"), item[0])):
            path = root / name
            if path.is_symlink() or not path.is_dir():
                raise WorkerPreparationError(f"cloned subframe directory is unsafe: {name}")
            path.chmod(mode)
    except OSError as error:
        raise WorkerPreparationError(f"cannot restore original subframe snapshot modes: {error}") from error


def _unit_task_builder():
    try:
        from .engine.kernel_unit_task import build_public_row
    except ImportError as error:
        raise WorkerPreparationError("subframe task builder is unavailable; install the subframe engine lane") from error
    return build_public_row


def _clone_snapshot(source: Path, destination: Path) -> str:
    """Try the native copy-on-write clone, then reflink, then a metadata-preserving copy."""
    methods = ([(["cp", "-cR", str(source), str(destination)], "apfs-clone")]
               if sys.platform == "darwin" else
               [(["cp", "--reflink=auto", "-R", str(source), str(destination)], "reflink-auto")]
               if sys.platform.startswith("linux") else [])
    for command, name in methods:
        try:
            result = subprocess.run(command, check=False, capture_output=True,
                                    timeout=load_tolerant_seconds("STATEFRAME_PARALLEL_PREPARE_SECONDS"))
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0 and destination.is_dir():
            return name
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
    try:
        shutil.copytree(source, destination, symlinks=True, copy_function=shutil.copy2)
    except OSError as error:
        raise WorkerPreparationError(f"cannot clone start snapshot: {error}") from error
    return "copytree"


def _start_files(spec: UnitSpec, snapshot: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    required = tuple(sorted(set(spec.contract.owns) | set(spec.contract.check.source_paths)))
    for relative in required:
        target = snapshot / relative
        try:
            mode = target.lstat().st_mode
        except OSError as error:
            raise WorkerPreparationError(f"owned start file is missing: {relative}") from error
        if not stat.S_ISREG(mode):
            raise WorkerPreparationError(f"owned start path is not a regular file: {relative}")
        try:
            files[relative] = target.read_bytes()
        except OSError as error:
            raise WorkerPreparationError(f"cannot read owned start file {relative}: {error}") from error
    for relative in spec.contract.creates:
        target = snapshot / relative
        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise WorkerPreparationError(f"cannot inspect declared create path {relative}: {error}") from error
        if not stat.S_ISREG(mode):
            raise WorkerPreparationError(f"declared create path exists and is not a regular file: {relative}")
        try:
            files[relative] = target.read_bytes()
        except OSError as error:
            raise WorkerPreparationError(f"cannot read declared create path {relative}: {error}") from error
    return files


def _validate_existing_workspace(workspace: Path, snapshot: Path, spec: UnitSpec) -> None:
    """Permit owned edits from an earlier incarnation while fencing all other changes."""
    after = DirectorySnapshot(workspace, spec.start_generation)
    before_paths = set(path for path in DirectorySnapshot(snapshot, spec.start_generation).list_files()
                       if path != "snapshot-manifest.json")
    after_paths = set(after.list_files())
    missing = before_paths - after_paths
    extra = after_paths - before_paths - set(spec.contract.creates)
    if missing or extra:
        raise WorkerPreparationError(f"existing subframe workspace has unexpected paths: "
                                     f"missing={sorted(missing)}, extra={sorted(extra)}")
    for relative in sorted(before_paths - set(spec.contract.owns)):
        source, target = snapshot / relative, workspace / relative
        if (stat.S_IFMT(source.lstat().st_mode) != stat.S_IFMT(target.lstat().st_mode)
                or (os.readlink(source) if source.is_symlink() else source.read_bytes()) !=
                   (os.readlink(target) if target.is_symlink() else target.read_bytes())):
            raise WorkerPreparationError(f"existing subframe changed an unowned path: {relative}")


def _worker_settings(parent_spec: Mapping[str, object]) -> ToolSettings:
    value = parent_spec.get("tool_settings")
    try:
        settings = ToolSettings.from_dict(value)
    except (ToolSettingsError, TypeError) as error:
        raise WorkerPreparationError(f"parent tool settings are invalid: {error}") from error
    return replace(settings, shell="off", shell_ask=False, shell_network="none", shell_allow=(),
                   shell_deny=(), shell_read=(), host_authorization=None, run_env=False,
                   git_tool=False, web=False, web_allow=(), tools_files=(), replay_external=False)


def _lease_endpoint(parent_spec: Mapping[str, object], unit_class: str) -> str:
    """Resolve the one server used by every request role in this subframe."""
    chat_url = parent_spec.get("chat_url")
    if not isinstance(chat_url, str) or not chat_url:
        raise WorkerPreparationError("parent spec lacks a model endpoint")
    recorded = parent_spec.get("routes")
    if recorded is None:
        return chat_url
    if not isinstance(recorded, dict) or recorded.get("schema") != "stateframe-routes/1":
        raise WorkerPreparationError("parent spec has an invalid recorded route table")
    roles, overrides = recorded.get("roles"), recorded.get("unit_overrides", {})
    if not isinstance(roles, dict) or not isinstance(overrides, dict):
        raise WorkerPreparationError("parent spec has an invalid recorded route table")
    digest_input = {"roles": roles, "unit_overrides": overrides} if overrides else roles
    digest = hashlib.sha256(json.dumps(digest_input, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode("utf-8")).hexdigest()
    if recorded.get("digest") != digest:
        raise WorkerPreparationError("parent spec recorded route digest differs from its settings")
    endpoints: set[str] = set()
    for role in ("read", "plan", "edit", "finish", "decide2"):
        settings = roles.get(role)
        if not isinstance(settings, dict):
            raise WorkerPreparationError(f"parent spec lacks the {role} route for this subframe")
        endpoint = settings.get("endpoint")
        for name in ("unit", f"unit.{unit_class}"):
            fields = overrides.get(name, [])
            if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
                raise WorkerPreparationError("parent spec has an invalid subframe route override")
            if "endpoint" in fields:
                overlay = roles.get(name)
                if not isinstance(overlay, dict):
                    raise WorkerPreparationError(f"parent spec lacks the {name} route override")
                endpoint = overlay.get("endpoint")
        if not isinstance(endpoint, str):
            raise WorkerPreparationError(f"parent spec has an invalid {role} route endpoint")
        try:
            endpoints.add(normalize_base_url(endpoint) + CHAT_SUFFIX)
        except EndpointError as error:
            raise WorkerPreparationError(f"parent spec has an invalid {role} route endpoint: {error}") from error
    if len(endpoints) != 1:
        raise WorkerPreparationError("routed subframe uses multiple model endpoints; configure one endpoint "
                                     "for every subframe request role")
    return endpoints.pop()


def _prepare_hook_bundle(parent_run: Path, run_dir: Path, hooks: object, *, reuse: bool) -> None:
    from .engine.kernel_hooks import HookConfigError, verify_bundle

    if type(hooks) is not dict or type(hooks.get("bundle_manifest")) is not dict:
        raise WorkerPreparationError("subframe hook bundle manifest is invalid")
    manifest = hooks["bundle_manifest"]
    try:
        if not reuse:
            # Inherit the pinned run artifact, never mutable hook source files.
            verify_bundle(manifest, parent_run / "hook-bundle")
            shutil.copytree(parent_run / "hook-bundle", run_dir / "hook-bundle", symlinks=True)
        verify_bundle(manifest, run_dir / "hook-bundle")
    except (HookConfigError, OSError) as error:
        raise WorkerPreparationError(f"subframe hook bundle cannot be verified: {error}") from error


def _bind_workspace_store(run_dir: Path, spec: UnitSpec, parent_spec: Mapping[str, object]) -> None:
    environment = parent_spec.get("environment")
    if not isinstance(environment, Mapping) or environment.get("STATEFRAME_WORKSPACE_OBJECTS") != "on":
        return
    engine_dir = parent_spec.get("engine_dir")
    if not isinstance(engine_dir, str) or not Path(engine_dir).is_dir():
        raise WorkerPreparationError("workspace object storage lacks its engine directory")

    # Stable identities make retries idempotent while keeping each worker in its own store.
    namespace = hashlib.sha256(
        f"stateframe-subframe-workspace/1\0{spec.run_seed}\0{spec.unit_id}\0{spec.execution_id}"
        .encode("utf-8")).hexdigest()
    engine_path = str(Path(engine_dir).resolve())
    added = engine_path not in sys.path
    if added:
        sys.path.insert(0, engine_path)
    try:
        from kernel_workspace_refs import bind_run_store

        bind_run_store(run_dir, namespace, run_id=namespace[:32])
    except (ImportError, OSError, ValueError) as error:
        raise WorkerPreparationError("subframe workspace object store could not be bound") from error
    finally:
        if added:
            sys.path.remove(engine_path)


def prepare_worker(spec: UnitSpec, spec_path: Path, run_dir: Path, start_snapshot: Path,
                   *, parent_spec: dict) -> Path:
    """Persist an ordinary subframe run spec and exactly one workspace clone of its start snapshot."""
    if not isinstance(spec, UnitSpec):
        raise TypeError("spec must be a UnitSpec")
    spec_path, run_dir, start_snapshot = Path(spec_path), Path(run_dir), Path(start_snapshot)
    if not isinstance(parent_spec, dict):
        raise TypeError("parent_spec must be a dict")
    from .parallel_resources import SlotPoolError, parse_slot_pool
    try:
        parallel_slots = parse_slot_pool(parent_spec.get("parallel_slots"),
                                         source="--parallel-slots in parent run spec")
    except SlotPoolError as error:
        raise WorkerPreparationError(f"subframe slot pool is invalid: {error}") from error
    if spec_path.is_symlink() or start_snapshot.is_symlink() or run_dir.is_symlink():
        raise WorkerPreparationError("subframe preparation paths must not be symlinks")
    if len(run_dir.parents) < 4:
        raise WorkerPreparationError("subframe run directory is outside the parallel subframe layout")
    parent_run = run_dir.parents[3]
    recorded_parent = parent_spec.get("parent_snapshot")
    if not isinstance(recorded_parent, str) or not recorded_parent:
        raise WorkerPreparationError("parent spec lacks its run-scoped parent snapshot")
    parent_snapshot = Path(recorded_parent)
    if (not parent_snapshot.is_absolute() or parent_snapshot.is_symlink()
            or parent_snapshot.resolve() != (parent_run / "parent").resolve()):
        raise WorkerPreparationError("parent spec snapshot does not match the subframe's parent run")
    if not isinstance(spec.run_seed, str) or re.fullmatch(r"[0-9a-f]{32}", spec.run_seed) is None:
        raise WorkerPreparationError("subframe run identity seed must be 32 lowercase hex characters")
    capabilities = parent_spec.get("capabilities")
    python = capabilities.get("python") if isinstance(capabilities, dict) else None
    if (not isinstance(python, str) or not Path(python).is_absolute()
            or not Path(python).is_file() or not os.access(python, os.X_OK)):
        raise WorkerPreparationError("parent spec lacks its recorded engine interpreter")
    try:
        on_disk = UnitSpec.from_json(_read_json(spec_path, "subframe spec"))
    except ValueError as error:
        raise WorkerPreparationError(f"subframe spec is invalid: {error}") from error
    if on_disk.digest() != spec.digest():
        raise WorkerPreparationError("subframe spec changed before subframe preparation")
    lease_endpoint = _lease_endpoint(parent_spec, spec.contract.model_class)
    expected_snapshot_digest = parent_spec.get("snapshot_manifest_digest")
    if not isinstance(expected_snapshot_digest, str):
        raise WorkerPreparationError("parent launch lacks the authenticated snapshot manifest digest")
    snapshot = read_snapshot_manifest(start_snapshot, expected_snapshot_digest)
    if snapshot.source_digest != spec.start_manifest_digest:
        raise WorkerPreparationError("start snapshot manifest digest does not match the subframe spec")

    run_dir.mkdir(parents=True, exist_ok=True)
    _bind_workspace_store(run_dir, spec, parent_spec)
    spec_file = run_dir / "spec.json"
    workspace = run_dir / "workspace"
    if spec_file.is_file():
        existing = _read_json(spec_file, "prepared subframe spec")
        parent = run_dir / "parent"
        settings_path = run_dir / "tool-settings.json"
        if (not workspace.is_dir() or workspace.is_symlink()
                or not parent.is_dir() or parent.is_symlink()
                or settings_path.is_symlink() or not settings_path.is_file()):
            raise WorkerPreparationError("existing subframe preparation does not match the frozen subframe spec")
        try:
            _validate_existing_workspace(workspace, start_snapshot, spec)
            if DirectorySnapshot(parent, spec.start_generation).manifest_digest != spec.start_manifest_digest:
                raise WorkerPreparationError("existing subframe parent snapshot differs from its start snapshot")
        except (OSError, ValueError, WorkerPreparationError) as error:
            raise WorkerPreparationError(f"existing subframe workspace cannot be verified: {error}") from error
        if (existing.get("unit_spec") != spec.to_json()
                or existing.get("hooks") != parent_spec.get("hooks")
                or existing.get("parallel_slots") != (list(parallel_slots) if parallel_slots is not None else None)
                or existing.get("parallel_parent_endpoint") != lease_endpoint
                or existing.get("parallel_workspace_clone") not in {"apfs-clone", "reflink-auto", "copytree"}
                or existing.get("parent_snapshot") != str(run_dir / "parent")
                or _read_json(settings_path, "subframe tool settings") != existing.get("tool_settings")):
            raise WorkerPreparationError("existing subframe preparation does not match the frozen subframe spec")
        if "hooks" in parent_spec:
            _prepare_hook_bundle(parent_run, run_dir, parent_spec["hooks"], reuse=True)
        return run_dir
    if (spec_file.exists() or spec_file.is_symlink() or workspace.exists() or workspace.is_symlink()
            or (run_dir / "parent").exists() or (run_dir / "parent").is_symlink()
            or (run_dir / "tool-settings.json").exists() or (run_dir / "tool-settings.json").is_symlink()):
        raise WorkerPreparationError("subframe run directory contains a partial or unsafe preparation")

    start_files = _start_files(spec, start_snapshot)
    try:
        public_row = _unit_task_builder()(spec, start_files)
    except (TypeError, ValueError) as error:
        raise WorkerPreparationError(f"cannot build the public subframe task: {error}") from error
    if not isinstance(public_row, dict):
        raise WorkerPreparationError("subframe task builder did not return a public row")
    parent_row = parent_spec.get("public_row")
    if isinstance(parent_row, dict) and "task_file_scope" in parent_row:
        task_scope = parent_row["task_file_scope"]
        if not isinstance(task_scope, list) or any(not isinstance(path, str) or not path for path in task_scope):
            raise WorkerPreparationError("parent task file scope is invalid")
        if task_scope:
            public_row["task_file_scope"] = list(task_scope)
    settings = _worker_settings(parent_spec)
    settings_path = run_dir / "tool-settings.json"
    # The baseline scanner requires a real directory, and resume rebuilds workspace/ from parent/.
    # Keep this immutable start copy separate from the writable candidate and the shared snapshot.
    temp_parent = run_dir / f".parent-{os.getpid()}-{time.monotonic_ns()}"
    _clone_snapshot(start_snapshot, temp_parent)
    _restore_snapshot_modes(temp_parent, snapshot)
    if DirectorySnapshot(temp_parent, spec.start_generation).manifest_digest != spec.start_manifest_digest:
        raise WorkerPreparationError("subframe parent copy differs from the start snapshot")
    os.replace(temp_parent, run_dir / "parent")
    _fsync_directory(run_dir)
    temp_workspace = run_dir / f".workspace-{os.getpid()}-{time.monotonic_ns()}"
    clone_method = _clone_snapshot(start_snapshot, temp_workspace)
    _restore_snapshot_modes(temp_workspace, snapshot)
    if DirectorySnapshot(temp_workspace, spec.start_generation).manifest_digest != spec.start_manifest_digest:
        raise WorkerPreparationError("subframe workspace clone differs from the start snapshot")
    os.replace(temp_workspace, workspace)
    _fsync_directory(run_dir)
    _write_json(settings_path, settings.as_dict())

    parent_environment = parent_spec.get("environment")
    if not isinstance(parent_environment, dict):
        raise WorkerPreparationError("parent profile environment must be an object")
    environment = dict(parent_environment)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
        raise WorkerPreparationError("parent profile environment must contain string keys and values")
    environment["STATEFRAME_TOOL_SETTINGS_FILE"] = str(settings_path.resolve())
    environment["STATEFRAME_REPO_CACHE_DIR"] = str((run_dir / "worker" / "repo-intel-cache").resolve())
    # Route scope is only meaningful (and only accepted by the switch preflight) when the run uses model routes.
    routed = parent_spec.get("routes") is not None
    if routed:
        environment["STATEFRAME_ROUTE_SCOPE"] = "unit"
        environment["STATEFRAME_ROUTE_UNIT_CLASS"] = spec.contract.model_class
    parent_sources = parent_spec.get("environment_sources", {})
    if not isinstance(parent_sources, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in parent_sources.items()):
        raise WorkerPreparationError("parent environment sources must contain string keys and values")
    environment_sources = dict(parent_sources)
    environment_sources.update({
        "STATEFRAME_TOOL_SETTINGS_FILE": "runner:unit-tool-settings",
        "STATEFRAME_REPO_CACHE_DIR": "runner:unit-repo-cache",
    })
    if routed:
        environment_sources.update({
            "STATEFRAME_ROUTE_SCOPE": "runner:unit-route",
            "STATEFRAME_ROUTE_UNIT_CLASS": "runner:unit-route",
        })

    permit_root = (parent_run / "parallel" / "check-permits").resolve()
    prepared = dict(parent_spec)
    if lease_endpoint != parent_spec["chat_url"]:
        # A parent slot can only be lent to a subframe using that same server.
        prepared["prompt_cache_slot"] = {"slot": None, "selection": "not_selected",
                                         "reason": "subframe uses another model endpoint"}
    if parallel_slots is None:
        prepared.pop("parallel_slots", None)
    else:
        prepared["parallel_slots"] = list(parallel_slots)
    prepared.pop("resume", None)
    prepared.pop("parallel_recovery", None)
    prepared.pop("run_env", None)
    prepared.pop("run_env_source_python", None)
    prepared.update({
        "schema": "stateframe-run/1",
        "parent_snapshot": str(run_dir / "parent"),
        "public_row": public_row,
        "tool_settings": settings.as_dict(),
        "custom_tools_digest": load_tool_files((), builtin_names=frozenset(),
                                               shell_network="none").digest,
        "custom_executables": {},
        "environment": environment,
        "environment_sources": environment_sources,
        "run_identity_seed": spec.run_seed,
        "parallel_mode": "off",
        "unit_spec": spec.to_json(),
        "parallel_permit_root": str(permit_root),
        "parallel_parent_endpoint": lease_endpoint,
        "parallel_send_ledger": str((run_dir / "sends-ledger.jsonl").resolve()),
        "parallel_workspace_clone": clone_method,
        "parallel_snapshot_manifest_digest": snapshot.digest,
        "test_cmd": None,
        "in_place": False,
    })
    prepared["parallel_engine_python"] = python
    if "hooks" in prepared:
        _prepare_hook_bundle(parent_run, run_dir, prepared["hooks"], reuse=False)
    _write_json(spec_file, prepared)
    return run_dir


class TickProcessTable(ProcessTable):
    """Process-table scans shared across workers and threads.

    Per-worker supervision used to enumerate the whole process table for each worker
    (several times per tick, plus once per status request), so a tick cost workers x host
    process count. Two sharing rules keep the meaning of each observation intact:

    * Observation (`tick_snapshot`): inside an open `tick()` the first call scans and the
      rest reuse that view. Observers still re-read each recorded member by PID with its
      birth stamp (`Descendants.update`), PID-namespace checks stay live, and a process
      born after the scan is found by the next tick.
    * Signalling and sweeps (`snapshot`): always a scan that began after the caller asked.
      Concurrent callers wait for and share one such scan instead of each running their
      own; a scan already in flight when a caller arrives is never reused. `invalidate()`
      drops the tick view after a signal so later observers rescan.

    The tick view belongs to the thread that opened the tick: ticks nest per thread, so a
    tick held open on another thread (a prepare or channel thread polling a worker) never
    keeps the supervisor's view alive across its own ticks. Every signal bumps a generation
    stamp; a view scanned before the bump, or a scan that overlapped one, is never reused.
    A scan that fails in any way publishes that failure to every caller sharing it: a
    failed scan is never an empty table, so a unit cannot be sealed on "no live group".
    """

    def __init__(self) -> None:
        super().__init__()
        self.require_complete = True
        self._lock = threading.Lock()  # guards the generation stamp
        self._flight = threading.Condition(threading.Lock())
        self._generation = 0  # bumped by invalidate(); a view stamped earlier is stale
        self._serial = 0  # numbers published views; see view_token()
        self._tick_state = threading.local()  # per-thread depth and tick view
        self._started = 0  # scans begun
        self._running = False
        self._result_id = 0
        self._result: dict[int, ProcessIdentity] | None = None
        self._result_error: Exception | None = None
        self._result_started_ns = 0
        self.scans = 0  # full-table enumerations performed; read by tests

    def __getstate__(self) -> dict:
        # Per-thread tick views belong to the thread that opened them; a thread-local cannot be
        # pickled, so it is dropped and rebuilt empty on the other side.
        state = dict(self.__dict__)
        state.pop("_tick_state", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._tick_state = threading.local()
        # A scan in flight in the saving process has no thread here: never wait on it, and never
        # reuse the result it left, which describes that process's table.
        self._running = False
        self._result = None
        self._result_error = None

    def _scan(self) -> dict[int, ProcessIdentity]:
        """One full enumeration, shared by every caller that arrived before it began."""
        with self._flight:
            target = self._started + 1
            while True:
                if self._result_id >= target:
                    if self._result_error is not None:
                        raise self._result_error
                    self._tick_state.fresh_started_ns = self._result_started_ns
                    return dict(self._result or {})
                if not self._running:
                    break
                self._flight.wait()
            self._started += 1
            scan_id = self._started
            self._running = True
        scan_started_ns = time.monotonic_ns()
        rows: dict[int, ProcessIdentity] | None = None
        error: Exception | None = None
        try:
            self.scans += 1
            rows = super().snapshot()
        except (OSError, RuntimeError, ValueError) as caught:
            error = caught
        except BaseException as caught:
            # Any other failure (an IndexError on a garbled stat read, an interrupt) must
            # reach every caller sharing this scan as a failure, never as an empty table.
            error = RuntimeError(f"process table scan failed: {caught!r}")
            error.__cause__ = caught
            if not isinstance(caught, Exception):
                raise  # an interrupt keeps its type here; waiters get the failure below
        finally:
            with self._flight:
                self._running = False
                self._result_id, self._result, self._result_error = scan_id, rows, error
                self._result_started_ns = scan_started_ns
                self._flight.notify_all()
        if error is not None:
            raise error
        self._tick_state.fresh_started_ns = scan_started_ns
        return dict(rows or {})

    def snapshot(self) -> dict[int, ProcessIdentity]:
        # An explicit fresh scan also supersedes this thread's older observation.
        # Later passive probes can reuse it; another snapshot still forces a scan.
        state = self._state()
        state.cached, state.cached_error = None, None
        return self.tick_snapshot()

    def _state(self) -> threading.local:
        state = self._tick_state
        if not hasattr(state, "depth"):
            state.depth = 0
            state.cached = None
            state.cached_error = None
            state.cached_serial = None
            state.stamp = -1
        return state

    @contextmanager
    def tick(self):
        """Open (or join) this thread's supervisor tick whose observations share one scan."""
        state = self._state()
        if state.depth == 0:
            state.cached, state.cached_error = None, None
        state.depth += 1
        try:
            yield self
        finally:
            state.depth -= 1
            if state.depth == 0:
                state.cached, state.cached_error = None, None

    def invalidate(self) -> None:
        """Drop every tick view; the next observation on any thread scans again."""
        with self._lock:
            self._generation += 1

    def tick_snapshot(self) -> dict[int, ProcessIdentity]:
        """Return a private copy of this tick's table view (a fresh scan outside a tick)."""
        state = self._state()
        if state.depth == 0:
            return self._scan()
        with self._lock:
            generation = self._generation
        if state.stamp == generation:
            if state.cached_error is not None:
                raise state.cached_error  # a failing scan is not retried once per worker
            if state.cached is not None:
                return dict(state.cached)
        # Scan outside the lock: invalidate() and other threads' ticks never wait behind it.
        state.cached, state.cached_error, state.stamp = None, None, generation
        try:
            rows = self._scan()
        except (OSError, RuntimeError, ValueError) as error:
            state.cached_error = error
            raise
        with self._lock:
            current = self._generation
            if current == generation:
                self._serial += 1
                state.cached_serial = self._serial
        if current == generation:
            state.cached = rows  # a signal during the scan leaves the view unpublished
        return dict(rows)

    def view_token(self) -> int | None:
        """Name the view this thread's open tick holds, or None when there is none to reuse.

        Two observations carrying the same token saw the same table, so the second adds nothing
        and may reuse the first's result. A new tick, a signal, or a failed or unpublished scan
        never shares a token with an earlier view."""
        state = self._state()
        if state.depth == 0 or state.cached is None:
            return None
        with self._lock:
            return state.cached_serial if state.stamp == self._generation else None


@dataclass
class WorkerHandle:
    """A child process and its session/process-group identity."""

    process: subprocess.Popen
    pid: int
    pgid: int
    incarnation: int
    run_dir: Path
    log_path: Path
    started_at: float
    started_monotonic: float
    reader: threading.Thread
    timed_out: bool = False
    cancel_reason: str | None = None
    cancel_started: float | None = None
    kill_sent: bool = False
    exit_code: int | None = None
    _log_error: str | None = None
    launch_identity: LaunchIdentity | None = None
    _cleanup_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _family_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _family: Descendants | None = field(default=None, init=False, repr=False)
    _shared_table: TickProcessTable | None = field(default=None, init=False, repr=False)
    # Run wall from the worker spec, read on the first deadline check (None = the spec sets none).
    wall_seconds: float | None = field(default=None, init=False)
    _wall_loaded: bool = field(default=False, init=False, repr=False)
    deadline_origin: float = field(default=0.0, init=False)
    _running_noted: bool = field(default=False, init=False, repr=False)
    # The last shared view this handle observed and what it found there (see _live_family).
    _observed_token: int | None = field(default=None, init=False, repr=False)
    _observed_rows: list[ProcessIdentity] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.deadline_origin = self.started_monotonic
        if self.launch_identity is None:
            self.launch_identity = (recorded_launch_identity(self.process)
                                    or capture_launch_identity(self.process, session_leader=True))
        if self.launch_identity is not None and self.launch_identity.start is not None:
            self._family = Descendants(self.pid, ProcessTable(), self.launch_identity)
            self._family.table.require_complete = True

    def attach_tick_table(self, table: TickProcessTable) -> None:
        """Observe through the supervisor's shared per-tick table instead of a private one."""
        with self._family_lock:
            self._shared_table = table
            if self._family is not None:
                self._family.table = table

    def _live_family(self, *, shared: bool = False) -> list[ProcessIdentity]:
        """Update and return the live family. `shared` accepts the tick's scan (observation
        only); signalling and sweeps always take a fresh scan."""
        with self._family_lock:
            if self._family is None:
                return []
            table = self._family.table
            try:
                observe = getattr(table, "tick_snapshot", None) if shared else None
                token_of = getattr(table, "view_token", None) if observe is not None else None
                token = token_of() if token_of is not None else None
                if token is not None and token == self._observed_token:
                    # This handle already folded this exact view in (enforce_deadline and the
                    # host's own poll both observe each tick): folding it again finds nothing new.
                    return list(self._observed_rows)
                live = self._family.update(observe() if observe is not None else table.snapshot())
                if token_of is not None:
                    self._observed_token, self._observed_rows = token_of(), list(live)
                return live
            except (OSError, RuntimeError, ValueError) as error:
                raise ContainmentUnverified("subframe descendant inspection is unavailable") from error

    def _live_unit(self) -> bool:
        """Family or group liveness. When both are read through one table (the supervisor's
        shared table) they come from ONE fresh scan; they used to cost two serial scans."""
        try:
            with self._family_lock:
                family = self._family
                group_table = self._shared_table if self._shared_table is not None else ProcessTable()
                if self._shared_table is None:
                    group_table.require_complete = True
                if family is not None:
                    rows = family.table.snapshot()
                    if family.update(rows):
                        return True
                    if group_table is family.table:
                        return self._group_alive(rows)
                return self._group_alive(group_table.snapshot())
        except (OSError, RuntimeError, ValueError) as error:
            raise ContainmentUnverified("subframe descendant inspection is unavailable") from error

    def _live_group(self) -> bool:
        try:
            table = self._shared_table if self._shared_table is not None else ProcessTable()
            if self._shared_table is None:
                table.require_complete = True
            return self._group_alive(table.snapshot())
        except (OSError, RuntimeError, ValueError) as error:
            raise ContainmentUnverified("subframe group inspection is unavailable") from error

    def _group_alive(self, rows: dict[int, ProcessIdentity]) -> bool:
        """Is any live process in this worker's process group, judged by birth and not by pgid alone?

        Once the leader is reaped its PID can be recycled, and a recycled PID that becomes a
        group leader is a stranger that shares the pgid number. Once the reap is reported, only
        processes born no later than it can belong to this group; later ones are never ours.
        Before the reap is reported every member of the group counts (the safe direction)."""
        family = self._family
        bound = getattr(family, "reaped_bound", None)
        if sys.platform == "darwin":
            return bool(live_group_rows(rows, self.pgid, bound))
        return any(row.pgid == self.pgid and not row.zombie
                   and (bound is None or row.start <= bound)
                   for row in rows.values())

    def group_live(self) -> bool:
        """Group liveness from a scan that began after this call (coalesced with other callers).

        Host-side recheck used before numbering an exit or sealing: it never runs a private
        full scan and never mistakes a recycled group number for this worker's group."""
        return self._live_unit()

    def _signal_group(self, sig: signal.Signals) -> None:
        with self._family_lock:
            # Snapshot before signalling the leader: detached checks can reparent as it dies.
            live = self._live_family()
            try:
                if (not signal_launch_group(self.process, self.launch_identity, sig)
                        and self._live_group()):
                    raise ContainmentUnverified(
                        f"subframe containment unverified: launch identity for group {self.pgid} is unavailable")
                if self._family is not None:
                    self._family.signal([row for row in live if row.pgid != self.pgid], sig)
            finally:
                invalidate = getattr(getattr(self._family, "table", None), "invalidate", None)
                if invalidate is not None:
                    invalidate()  # the table changed under us, even on a partial signal: rescan

    def _sweep_group(self) -> None:
        if not self._live_unit():
            return
        self._signal_group(signal.SIGTERM)
        until = time.monotonic() + cleanup_seconds()
        while self._live_unit() and time.monotonic() < until:
            time.sleep(0.05)
        if self._live_unit():
            self._signal_group(signal.SIGKILL)
            until = time.monotonic() + cleanup_seconds()
            while self._live_unit() and time.monotonic() < until:
                time.sleep(0.05)
        if self._live_unit():
            raise ContainmentUnverified(f"subframe containment unverified: worker family {self.pgid} remains live")

    def poll(self) -> int | None:
        # Popen.poll() reaps the leader and releases the PID before group cleanup.
        # Retain birth-verified descendants while their parent links are still observable.
        self._live_family(shared=True)
        if self.process.returncode is not None:
            self.exit_code = self.process.returncode
            return self.exit_code
        return 0 if child_exited_unreaped(self.process) else None

    def cancel(self, reason: str) -> None:
        with self._family_lock:
            if self.cancel_started is not None:
                return
            self.cancel_reason = str(reason)[:300]
            try:
                self._signal_group(signal.SIGTERM)
                # Process inspection can itself be slow. Give the worker its full
                # grace after delivery, rather than spending it preparing the signal.
                self.cancel_started = time.monotonic()
            except PermissionError as error:
                raise WorkerLaunchError(f"cannot terminate subframe group {self.pgid}: {error}") from error

    def enforce_deadline(self) -> None:
        if self.cancel_reason is not None and self.cancel_started is None:
            # A previous signal could not establish containment. Retry the
            # explicit cancellation once inspection becomes available again.
            self.cancel(self.cancel_reason)
            return
        if self.poll() is not None or self.cancel_started is not None:
            return
        if not self._wall_loaded:
            # Read once per handle, not once per worker per tick; a corrupt spec still raises.
            self.wall_seconds = run_wall_seconds(_read_json(self.run_dir / "spec.json", "subframe spec"))
            self._wall_loaded = True
        seconds = self.wall_seconds
        if seconds is None or not math.isfinite(seconds) or seconds <= 0:
            return  # an explicit non-positive run wall means "no deadline"
        if time.monotonic() - self.deadline_origin < seconds:
            return
        grace = load_tolerant_seconds("STATEFRAME_PARALLEL_WATCHDOG_GRACE_SECONDS")
        if time.monotonic() - self.deadline_origin >= seconds + grace:
            self.timed_out = True
            self.cancel("subframe watchdog deadline expired")

    def note_running(self, acquired_epoch: float) -> None:
        """Start the run-wall clock at the slot lease: waiting for a model slot is not run time."""
        if self._running_noted:
            return
        self._running_noted = True
        self.deadline_origin = self.started_monotonic + max(0.0, acquired_epoch - self.started_at)

    def escalate(self, *, grace: float | None = None) -> None:
        with self._family_lock:
            if self.cancel_started is None or self.kill_sent:
                return
            if grace is None:
                grace = cleanup_seconds()
            if time.monotonic() - self.cancel_started < grace:
                return
            try:
                self._signal_group(signal.SIGKILL)
                self.kill_sent = True
            except PermissionError as error:
                raise WorkerLaunchError(f"cannot kill subframe group {self.pgid}: {error}") from error

    def reap(self, timeout: float = 0.0) -> int | None:
        with self._cleanup_lock:
            until = time.monotonic() + max(0.0, timeout)
            while not child_exited_unreaped(self.process) and self.process.returncode is None:
                self._live_family()
                remaining = until - time.monotonic()
                if remaining <= 0:
                    return None
                time.sleep(min(0.02, remaining))
            self._sweep_group()
            try:
                code = self.process.wait(timeout=max(0.0, until - time.monotonic()))
            except subprocess.TimeoutExpired:
                return None
            if self._family is not None:
                with self._family_lock:
                    self._family.leader_reaped()
                    # This immutable proof survives a crash between containment and the
                    # host's EXITED commit. Member lists alone are never a complete family.
                    leader = self._family.members.get(self.pid)
                    if (leader is None or self.launch_identity is None
                            or leader.start != self.launch_identity.start):
                        raise ContainmentUnverified("subframe family exit lacks its launch identity")
                    proof = {"pid": self.pid, "pgid": self.pgid,
                             "kernel_identity": kernel_start_identity(self.pid, row=leader)}
                # Release the family lock before these durability barriers.
                proof_file = self.run_dir / "proc" / str(self.incarnation) / "family-exited.json"
                try:
                    if proof_file.is_symlink():
                        raise WorkerPreparationError("subframe family exit proof is unsafe")
                    if proof_file.exists():
                        if _read_json(proof_file, "subframe family exit proof") != proof:
                            raise WorkerPreparationError("subframe family exit proof differs")
                        _fsync_directory(proof_file.parent)  # retry a prior lost directory ACK
                    else:
                        _write_json(proof_file, proof)
                except (OSError, WorkerPreparationError) as error:
                    raise ContainmentUnverified("subframe family exit could not be persisted") from error
            self.exit_code = code
            self.reader.join(timeout=cleanup_seconds())
            return code


def _pump_output(stream, path: Path, handle_ref: dict[str, WorkerHandle | None]) -> None:
    handle = handle_ref.get("handle")
    try:
        with path.open("a", encoding="utf-8", errors="replace") as log:
            for line in stream:
                log.write(line)
                log.flush()
    except OSError as error:
        if handle is not None:
            handle._log_error = f"{type(error).__name__}: {error}"
    finally:
        stream.close()


def _child_environment(spec: Mapping[str, object]) -> dict[str, str]:
    environment = spec.get("environment")
    if not isinstance(environment, dict):
        raise WorkerLaunchError("subframe spec lacks its profile environment")

    # Start only from named host basics; a parent STATEFRAME_* setting is not ambient child state.
    env = {key: value for key, value in os.environ.items()
           if key in CHILD_ENV_ALLOWLIST and key != "SF_SUBFRAME_INCARNATION"}
    # These run-local settings are recorded in the spec and override ambient copies when present.
    # The installation-identity pin is the runner's startup authority for the run: every
    # engine child, subframes included, verifies the installation home against it.
    for name in ("LLAMA_CPP_TOKENIZE_URL", "STATEFRAME_TEST_PYTHON", "PYTHONHASHSEED",
                 "STATEFRAME_RUNNER_HOME_IDENTITY"):
        value = environment.get(name)
        if isinstance(value, str):
            if value.strip() or name == "PYTHONHASHSEED":
                env[name] = value
            else:
                env.pop(name, None)
    if environment.get("STATEFRAME_WORKSPACE_OBJECTS") == "on":
        stateframe_home = environment.get("STATEFRAME_HOME")
        if isinstance(stateframe_home, str) and stateframe_home:
            env["STATEFRAME_HOME"] = stateframe_home

    # Credentials are named in the spec; values stay in the launch environment only.
    key_names = [spec.get("api_key_env")]
    recorded = spec.get("routes")
    if recorded is not None:
        if not isinstance(recorded, dict):
            raise WorkerLaunchError("subframe route table is invalid")
        roles, overrides = recorded.get("roles"), recorded.get("unit_overrides", {})
        if not isinstance(roles, dict) or not isinstance(overrides, dict):
            raise WorkerLaunchError("subframe route table is invalid")
        unit_class = environment.get("STATEFRAME_ROUTE_UNIT_CLASS", "default")
        for role in ("read", "plan", "edit", "finish", "decide2"):
            settings = roles.get(role)
            if not isinstance(settings, dict):
                raise WorkerLaunchError("subframe route settings are invalid")
            key_name = settings.get("api_key_env")
            for name in ("unit", f"unit.{unit_class}"):
                fields = overrides.get(name, [])
                if not isinstance(fields, list):
                    raise WorkerLaunchError("subframe route override is invalid")
                if "api_key_env" in fields:
                    overlay = roles.get(name)
                    if not isinstance(overlay, dict) or "api_key_env" not in overlay:
                        raise WorkerLaunchError("subframe route credential override is invalid")
                    key_name = overlay["api_key_env"]
            key_names.append(key_name)
    for key_name in key_names:
        if key_name is None:
            continue
        if not isinstance(key_name, str) or not key_name:
            raise WorkerLaunchError("subframe API-key environment name is invalid")
        if key_name not in os.environ:
            raise WorkerLaunchError(f"subframe API-key environment {key_name} is not set")
        env[key_name] = os.environ[key_name]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _abort_failed_launch(process: subprocess.Popen, identity: LaunchIdentity | None,
                         reason: str) -> None:
    """Keep the launch PID reserved until its group has been swept."""
    if identity is None:
        try:
            # This Popen is still unreaped and no other thread has its handle.
            # Popen.kill() calls poll(), which could release the group PID first.
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            raise WorkerLaunchError(
                f"{reason}; subframe containment unverified: cannot kill unidentified leader: {error}") from error
        raise WorkerLaunchError(
            f"{reason}; subframe containment unverified: launch identity unavailable; "
            "leader left unreaped for recovery")
    try:
        survivors = kill_launch_unit(process, launch_identity=identity)
    except (ContainmentUnverified, OSError, RuntimeError) as error:
        raise WorkerLaunchError(f"{reason}; subframe containment unverified: {error}") from error
    if survivors:
        raise WorkerLaunchError(
            f"{reason}; subframe containment unverified: surviving PIDs {survivors}")
    raise WorkerLaunchError(reason)


def _launch_spec(run_dir: Path) -> dict:
    """Read and validate the prepared worker spec: its engine child script and interpreter."""
    spec = _read_json(run_dir / "spec.json", "subframe spec")
    if spec.get("schema") != "stateframe-run/1" or not isinstance(spec.get("public_row"), dict):
        raise WorkerLaunchError("subframe spec is not an ordinary stateframe-run/1 spec")
    engine_dir = Path(str(spec.get("engine_dir", "")))
    child = engine_dir.parent / "_child.py"
    python = spec.get("parallel_engine_python")
    if (not isinstance(python, str) or not Path(python).is_absolute()
            or not Path(python).is_file() or not os.access(python, os.X_OK)):
        raise WorkerLaunchError("subframe spec lacks its recorded parent engine interpreter")
    if not child.is_file():
        raise WorkerLaunchError(f"engine child script is missing: {child}")
    return spec


def prepare_relaunch(run_dir: Path, incarnation: int) -> None:
    """Archive the prior attempt and rebuild the workspace for a relaunch (incarnation > 1).

    This copies the whole repository, so a supervisor runs it before taking its host lock;
    `launch_worker(..., relaunch_prepared=True)` then only starts the process."""
    if type(incarnation) is not int or incarnation < 1:
        raise ValueError("incarnation must be a positive integer")
    if incarnation == 1:
        return
    run_dir = Path(run_dir).resolve()
    spec = _launch_spec(run_dir)
    engine_dir = Path(str(spec.get("engine_dir", "")))
    prior_path = run_dir / "proc" / str(incarnation - 1) / "process.json"
    if prior_path.is_symlink() or not prior_path.is_file():
        raise WorkerLaunchError("prior incarnation process record is missing or unsafe")
    prior = _read_json(prior_path, "prior incarnation process record")
    if (prior.get("incarnation") != incarnation - 1
            or type(prior.get("pid")) is not int or prior["pid"] < 1
            or prior.get("pgid") != prior["pid"]):
        raise WorkerLaunchError("prior incarnation process record has invalid identity")
    from . import resume as resume_module
    try:
        state = resume_module.assess(run_dir)
        resume_module.archive_attempt(state, engine_dir, use_checkpoint=False)
        resume_module.fresh_workcopy(run_dir, Path(str(spec["repo"])), preserve_symlinks=True)
        unit_spec = UnitSpec.from_json(spec["unit_spec"])
        snapshot_root = run_dir.parents[1] / "snapshots" / f"g{unit_spec.start_generation}"
        manifest = read_snapshot_manifest(snapshot_root, str(spec["parallel_snapshot_manifest_digest"]))
        if manifest.source_digest != unit_spec.start_manifest_digest:
            raise WorkerLaunchError("resumed subframe snapshot source digest differs")
        if (DirectorySnapshot(run_dir / "workspace", unit_spec.start_generation).manifest_digest !=
                unit_spec.start_manifest_digest):
            raise WorkerLaunchError("resumed subframe workspace differs from its immutable start snapshot")
    except (OSError, ValueError, WorkerPreparationError, resume_module.ResumeRefused) as error:
        raise WorkerLaunchError(f"cannot resume subframe execution: {error}") from error


def launch_worker(
    run_dir: Path, incarnation: int, relaunch_prepared: bool = False, *,
    admission: Callable[[], AbstractContextManager[Callable[[WorkerHandle], None]]] | None = None,
) -> WorkerHandle:
    """Start the prepared engine child in a new process group and drain its log.

    A relaunch (incarnation > 1) first rebuilds the workspace via `prepare_relaunch`, unless
    the caller already did so outside its own locks (`relaunch_prepared=True`). An optional
    admission fence covers spawning and handle installation, never durable filesystem I/O."""
    if type(incarnation) is not int or incarnation < 1:
        raise ValueError("incarnation must be a positive integer")
    run_dir = Path(run_dir).resolve()
    if incarnation > 1 and not relaunch_prepared:
        prepare_relaunch(run_dir, incarnation)
    spec = _launch_spec(run_dir)
    engine_dir = Path(str(spec.get("engine_dir", "")))
    child = engine_dir.parent / "_child.py"
    python = spec["parallel_engine_python"]

    environment = _child_environment(spec)
    if "SF_SUBFRAME_INCARNATION" not in CHILD_ENV_ALLOWLIST:
        raise WorkerLaunchError("subframe incarnation marker is missing from the child environment allowlist")
    environment["SF_SUBFRAME_INCARNATION"] = str(incarnation)
    if "unit_spec" in spec:
        # The child must durably publish its independent slot claim before lease_slot
        # returns the slot to engine code that can start owned descendants.
        environment["SF_SUBFRAME_SLOT_CLAIM"] = "durable-v1"
    process_dir = run_dir / "proc" / str(incarnation)
    try:
        process_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise WorkerLaunchError(f"subframe incarnation already exists: {incarnation}") from error
    except OSError as error:
        raise WorkerLaunchError(f"cannot create subframe process directory: {error}") from error
    log_path = process_dir / "engine.log"
    command = [python, "-B", str(child), str(run_dir)]
    try:
        _fsync_directory(run_dir)
        _fsync_directory(process_dir.parent)
        log_path.touch(mode=0o600, exist_ok=False)
        _write_json(process_dir / "launch-intent.json", {
            "schema": "stateframe-subframe-launch-intent/1",
            "incarnation": incarnation,
            "run_dir": str(run_dir),
            "command": command,
        })
    except (OSError, WorkerPreparationError) as error:
        raise WorkerLaunchError(f"cannot persist subframe launch intent at {process_dir}: {error}") from error
    process: subprocess.Popen | None = None
    launch_identity = None
    try:
        with (admission() if admission is not None else nullcontext()) as install:
            try:
                process = subprocess.Popen(command, cwd=run_dir, env=environment,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           encoding="utf-8", errors="replace",
                                           bufsize=1, start_new_session=True, close_fds=True)
            except (OSError, ValueError) as error:
                raise WorkerLaunchError(f"cannot start subframe engine child: {error}") from error
            try:
                launch_identity = capture_launch_identity(process, session_leader=True)
                if launch_identity is None:
                    raise WorkerLaunchError("subframe launch identity could not be captured")
                birth_identity = kernel_start_identity(process.pid)
                if birth_identity is None:
                    raise WorkerLaunchError("subframe process exited before its kernel identity was recorded")
            except (OSError, RuntimeError, ValueError) as error:
                raise WorkerLaunchError(f"cannot record subframe kernel identity: {error}") from error
            if process.stdout is None:
                raise WorkerLaunchError("subframe child did not provide its output pipe")
            handle_ref: dict[str, WorkerHandle | None] = {"handle": None}
            reader = threading.Thread(target=_pump_output, args=(process.stdout, log_path, handle_ref), daemon=True,
                                      name=f"stateframe-worker-log-{process.pid}")
            started_at = time.time()
            handle = WorkerHandle(process, process.pid, process.pid, incarnation, run_dir, log_path, started_at,
                                  time.monotonic(), reader, launch_identity=launch_identity)
            handle_ref["handle"] = handle
            reader.start()
            if install is not None:
                install(handle)
    except (OSError, RuntimeError, ValueError) as error:
        if process is not None:
            # Release admission before a failed spawn's containment sweep can wait.
            _abort_failed_launch(process, launch_identity, str(error))
        raise
    # Preserve members already retained by the handle, including any that have left
    # the launch PGID. This is signal authority, not proof of a complete family.
    launch_record = {
        "incarnation": incarnation,
        "pid": process.pid,
        "pgid": process.pid,
        "started_at": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "started_epoch": started_at,
        "kernel_identity": birth_identity,
        "verified_members": {},
        "fenced": False,
    }
    try:
        family = getattr(handle, "_family", None)
        if family is not None:
            verified_members: dict[str, str] = {}
            with handle._family_lock:
                for member, row in family.members.items():
                    if member != process.pid:
                        member_birth = kernel_start_identity(member, row=row)
                        if member_birth is None:
                            raise WorkerLaunchError("subframe member birth identity is unavailable")
                        verified_members[str(member)] = member_birth
            launch_record["verified_members"] = verified_members
        _write_json(process_dir / "process.json", launch_record)
    except (OSError, WorkerPreparationError, WorkerLaunchError) as error:
        try:
            handle.cancel("subframe process record could not be written")
            handle.escalate()
            if handle.reap(timeout=cleanup_seconds()) is None:
                handle.escalate(grace=0.0)
                handle.reap(timeout=cleanup_seconds())
        except (OSError, WorkerLaunchError):
            pass
        raise WorkerLaunchError(f"cannot record subframe process identity: {error}") from error
    return handle
