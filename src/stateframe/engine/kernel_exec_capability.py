"""Immutable host-issued execution authority, independent of test policy."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG
import sys
from threading import Lock
from types import MappingProxyType
from typing import Literal

NetworkMode = Literal["none", "loopback", "full"]
PathFingerprint = tuple[int, int, int, int, int]
PathFingerprints = dict[str, PathFingerprint]


def _file_fingerprint(status: os.stat_result) -> PathFingerprint:
    return status.st_dev, status.st_ino, status.st_mtime_ns, status.st_ctime_ns, status.st_mode


def _remember_path(inputs: PathFingerprints, path: Path, status: os.stat_result) -> None:
    fingerprint = _file_fingerprint(status)
    if inputs.setdefault(str(path), fingerprint) != fingerprint:
        raise OSError(f"shell PATH authority changed during scanning: {path}")


_PATH_ACCESS = {"read": os.R_OK, "execute": os.X_OK, "list": os.R_OK | os.X_OK}
_PATH_CHECKS = ("entry", "parent", "target", *_PATH_ACCESS)
_MISSING_TARGET = {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}
_LOCAL_PATH_FILESYSTEMS = {b"ext2", b"ext3", b"ext4", b"xfs", b"btrfs", b"tmpfs"}
_MAX_PATH_CHECKS = 32_768


def _inventory_stamp(kind: str, status: os.stat_result) -> tuple[int, ...]:
    identity = (status.st_dev, status.st_ino, status.st_mode, status.st_uid, status.st_gid)
    if kind == "parent":
        # Sibling churn in traversal parents does not change resolution.
        return identity
    return (*identity, status.st_mtime_ns, status.st_ctime_ns, status.st_size)


class PathInventoryStamps:
    """Record the metadata and access decisions used by an executable scan."""

    def __init__(self) -> None:
        self.values: dict[str, dict[str, tuple[int, ...] | bool | None]] = {
            kind: {} for kind in _PATH_CHECKS
        }
        self.complete = True
        self._count = 0

    def remember(self, kind: str, path: Path, value: tuple[int, ...] | bool | None) -> None:
        if not self.complete:
            return
        values, name = self.values[kind], str(path)
        if name not in values:
            self._count += 1
            if self._count > _MAX_PATH_CHECKS:
                self.complete = False
                return
        if values.setdefault(name, value) != value:
            self.complete = False

    def status(self, kind: str, path: Path, status: os.stat_result) -> None:
        self.remember(kind, path, _inventory_stamp(kind, status))

    def missing_target(self, path: Path, error: OSError) -> None:
        if error.errno in _MISSING_TARGET:
            self.remember("target", path, None)
        else:
            self.complete = False

    def skipped_target(self, path: Path, *, directory: bool = False) -> None:
        try:
            status = os.stat(path)
        except OSError as error:
            self.missing_target(path, error)
        else:
            if directory and not S_ISDIR(status.st_mode):
                self.status("target", path, status)
            else:
                self.complete = False


def _walk_symlinks(
    path: Path, inputs: PathFingerprints | None = None, *,
    stamps: PathInventoryStamps | None = None,
) -> tuple[tuple[Path, ...], dict[Path, str], Path, os.stat_result]:
    """Return full-path hops and links, without admitting a link's whole directory.

    Resolving only the endpoint loses the paths the kernel checks during exec.
    Keep the links themselves, never broaden authority to their parent directories.
    """
    absolute = Path(path).absolute()
    pending = list(absolute.parts[1:])
    current = Path("/")
    hops: set[Path] = set()
    links: dict[Path, str] = {}
    traversals = 0
    while pending:
        component = pending.pop(0)
        if component == "..":
            current = current.parent
            continue
        current = current / component
        status = current.lstat() if stamps is not None else None
        is_link = current.is_symlink() if status is None else S_ISLNK(status.st_mode)
        if is_link:
            traversals += 1
            if traversals > 40:
                raise ValueError(f"executable symlink chain exceeds 40 hops: {path}")
            hops.add(current.joinpath(*pending))
            if inputs is not None:
                _remember_path(inputs, current, status if status is not None else current.lstat())
            if stamps is not None:
                stamps.status("entry", current, status)
            links[current] = os.readlink(current)
            target = Path(links[current])
            current = Path("/") if target.is_absolute() else current.parent
            pending = list(target.parts[1:] if target.is_absolute() else target.parts) + pending
        elif stamps is not None and pending and S_ISDIR(status.st_mode):
            stamps.status("parent", current, status)
    status = (current if inputs is None and stamps is None else absolute).stat()
    if inputs is not None:
        _remember_path(inputs, current, status)
    return tuple(sorted(hops | {current})), links, current, status


def symlink_hops(path: Path) -> tuple[Path, ...]:
    """Admit each full executable path along a bounded symlink traversal."""
    return _walk_symlinks(path)[0]


_PUBLIC_CA_PATHS = ("/etc/ssl/certs", "/etc/ssl/cert.pem", "/etc/pki/tls/certs",
                     "/etc/pki/ca-trust/extracted", "/etc/ca-certificates")


def public_ca_read_roots() -> tuple[Path, ...]:
    """Linux public CA trust store paths (with their symlink chains); macOS admits /etc/ssl elsewhere."""
    if sys.platform == "darwin":
        return ()
    roots: list[Path] = []
    for raw in _PUBLIC_CA_PATHS:
        path = Path(raw)
        if path.exists():
            roots.extend(symlink_hops(path))
    return tuple(dict.fromkeys(roots))


def executable_symlinks(paths: tuple[Path, ...]) -> dict[Path, str]:
    """Links to recreate in a private mount tree; targets are admitted separately."""
    return {link: target for path in paths for link, target in _walk_symlinks(path)[1].items()}


def executable_read_roots(path: Path) -> tuple[Path, ...]:
    """Include a script's absolute shebang interpreter and its complete link chain."""
    return _executable_read_roots(path)


def _executable_read_roots(
    path: Path, inputs: PathFingerprints | None = None, expected: PathFingerprint | None = None,
    *, stamps: PathInventoryStamps | None = None,
) -> tuple[Path, ...]:
    roots: set[Path] = set()
    seen: set[Path] = set()
    pending = [Path(path)]
    while pending:
        candidate = pending.pop()
        if inputs is None and stamps is None:
            hops = symlink_hops(candidate)
            resolved = candidate.resolve(strict=True)
            regular = resolved.is_file()
        else:
            hops, _, resolved, status = _walk_symlinks(candidate, inputs, stamps=stamps)
            if expected is not None and candidate == path and _file_fingerprint(status) != expected:
                raise OSError(f"shell PATH entry changed during scanning: {path}")
            if stamps is not None:
                stamps.status("entry", resolved, status)
            regular = S_ISREG(status.st_mode)
        roots.update(hops)
        if resolved in seen or not regular:
            continue
        if len(seen) >= 40:
            raise ValueError(f"executable interpreter chain exceeds 40 files: {path}")
        seen.add(resolved)
        # Execute-only native binaries (for example macOS sudo) have no
        # readable script header. Their link authority still remains explicit.
        readable = os.access(resolved, os.R_OK)
        if stamps is not None:
            stamps.remember("read", resolved, readable)
        if not readable:
            continue
        with resolved.open("rb") as stream:
            line = stream.readline(4096)
        if line.startswith(b"#!"):
            words = line[2:].split()
            if words:
                interpreter = Path(os.fsdecode(words[0]))
                if interpreter.is_absolute():
                    if interpreter.exists():
                        pending.append(interpreter)
                    elif stamps is not None:
                        stamps.skipped_target(interpreter)
    return tuple(sorted(roots))


def python_runtime_read_roots(path: Path) -> tuple[Path, ...]:
    """Include Python installation roots reached by a pinned interpreter."""
    executable = Path(path)
    roots = set(executable_read_roots(executable))
    resolved = executable.resolve(strict=True)
    for candidate in (executable.absolute(), resolved):
        is_python = candidate.name.casefold().startswith("python")
        if is_python and candidate.parent.name in {"bin", "Scripts"}:
            prefix = candidate.parent.parent
            is_virtualenv = (prefix / "pyvenv.cfg").is_file()
            if prefix != Path("/") and (candidate == resolved or is_virtualenv):
                roots.add(prefix)
        for parent in candidate.parents:
            if parent.name.casefold().startswith("python") and parent.name.endswith(
                ".framework"
            ):
                roots.add(parent)
                break
    return tuple(sorted(roots))


@dataclass(frozen=True)
class ExecLimits:
    wall_seconds: float = 120.0
    cpu_seconds: float = 300.0
    memory_bytes: int = 2 << 30
    file_bytes: int = 1 << 30
    nofile: int = 1024
    nproc: int = 256
    output_bytes: int = 256 << 10
    kill_on_output: bool = False

    def __post_init__(self) -> None:
        for name in ("wall_seconds", "cpu_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if self.wall_seconds > 600:
            raise ValueError("wall_seconds exceeds shell maximum of 600")
        for name in ("memory_bytes", "file_bytes", "nofile", "nproc", "output_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.file_bytes > 1 << 30:
            raise ValueError("file_bytes exceeds shell maximum of 1 GiB")
        if self.nofile < 16:
            raise ValueError("nofile must be at least 16 for the launcher")
        if self.output_bytes > 4 << 20:
            raise ValueError("output_bytes exceeds maximum of 4 MiB")
        if type(self.kill_on_output) is not bool:
            raise TypeError("kill_on_output must be a boolean")


@dataclass(frozen=True)
class ExecCapability:
    purpose: Literal["shell", "job", "git", "custom_tool", "probe", "hook", "mutation", "mutation_broker"]
    workspace: Path
    workspace_writable: bool
    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]
    network: NetworkMode
    env: Mapping[str, str]
    limits: ExecLimits
    sandboxed: bool = True

    def __post_init__(self) -> None:
        if self.purpose not in {"shell", "job", "git", "custom_tool", "probe", "hook", "mutation", "mutation_broker"}:
            raise ValueError("invalid execution purpose")
        if self.network not in {"none", "loopback", "full"}:
            raise ValueError("network must be none, loopback or full")
        if type(self.sandboxed) is not bool or type(self.workspace_writable) is not bool:
            raise TypeError("sandboxed and workspace_writable must be booleans")
        if not isinstance(self.limits, ExecLimits):
            raise TypeError("limits must be ExecLimits")
        object.__setattr__(self, "workspace", Path(self.workspace).resolve(strict=True))
        for name in ("read_roots", "write_roots"):
            # Interpreter discovery belongs to trusted provisioning, not to
            # replace()/deserialization after a tool may have changed a script.
            roots = tuple(sorted({Path(p).absolute() if name == "read_roots"
                                  else Path(p).resolve(strict=True)
                                  for p in getattr(self, name)}))
            if name == "read_roots":
                resolved = {p: p.resolve(strict=True) for p in roots}
                canonical = {p for p in roots if p == resolved[p]}
                directories = {p for p in canonical if p.is_dir()}
                for path, target in resolved.items():
                    if target not in canonical and not any(parent in directories for parent in target.parents):
                        raise ValueError(f"read-root symlink target lacks explicit authority: {path}")
                roots = tuple(p for p in roots if p.is_symlink()
                              or not any(parent in directories for parent in p.parents))
            object.__setattr__(self, name, roots)
        environment = dict(self.env)
        for key, value in environment.items():
            if (not isinstance(key, str) or not key or "=" in key or "\0" in key
                    or not isinstance(value, str) or "\0" in value):
                raise ValueError("environment must contain valid string keys and values")
        object.__setattr__(self, "env", MappingProxyType(environment))
        if not self.workspace_writable:
            for root in self.write_roots:
                if root == self.workspace or root in self.workspace.parents or self.workspace in root.parents:
                    raise ValueError("write root overlaps read-only workspace")

    def as_dict(self) -> dict:
        limits = asdict(self.limits)
        if not self.limits.kill_on_output:
            del limits["kill_on_output"]
        return {"purpose": self.purpose, "workspace": str(self.workspace),
                "workspace_writable": self.workspace_writable,
                "read_roots": [str(p) for p in self.read_roots],
                "write_roots": [str(p) for p in self.write_roots],
                "network": self.network, "env": dict(self.env),
                "limits": limits, "sandboxed": self.sandboxed}

    def digest(self) -> str:
        data = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode()
        return hashlib.sha256(data).hexdigest()


def capability_from_dict(value: dict) -> ExecCapability:
    values = dict(value)
    values["limits"] = ExecLimits(**values["limits"])
    return ExecCapability(**values)


def _macos_developer_roots() -> list[Path]:
    """Read-only authority for the active macOS developer directory.

    /usr/bin/make, git, cc, clang and friends are shims that resolve the developer tools through
    /var/db/xcode_select_link. If the sandbox hides it, the shim concludes no tools are installed
    and asks launchd to open the "Install Command Line Developer Tools" dialog on the user's desktop.
    """
    if sys.platform != "darwin":
        return []
    roots = []
    for value in ("/private/var/db/xcode_select_link", "/var/select/developer_dir",
                  "/private/var/select/developer_dir"):
        link = Path(value)
        if link.is_symlink() and link.resolve().is_dir():
            roots.extend(symlink_hops(link))
    fallback = Path("/Library/Developer/CommandLineTools")
    if fallback.is_dir():
        roots.append(fallback)
    return roots


def system_runtime_read_roots() -> tuple[Path, ...]:
    """Read-only system files needed by shell commands and generic checks."""
    # macOS resolves /bin/sh through /private/var/select/sh; hiding that link
    # makes shell-script checks fail before they can inspect the repository.
    roots = [Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64",
                               "/System", "/Library/Apple", "/private/var/select")
             if Path(p).exists()]
    roots += _macos_developer_roots()
    if sys.platform == "darwin":
        gitconfig = Path("/etc/gitconfig")
        if gitconfig.is_file():
            roots.extend(symlink_hops(gitconfig))
        # Node and OpenSSL load configuration outside their installed-code
        # roots. Admit only SSL configuration, never all of /etc or Homebrew etc.
        ssl = Path("/etc/ssl")
        if ssl.exists():
            roots.extend(symlink_hops(ssl))
        for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
            etc = prefix / "etc"
            if etc.is_dir():
                for path in etc.glob("openssl*"):
                    if path.is_dir() or path.is_file():
                        roots.extend(symlink_hops(path))
    # Installed libraries may be in sibling formulae, not the executable's
    # directory. Homebrew var and the user's home are never admitted.
    for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
        if (prefix / "Cellar").is_dir():
            roots.extend(path for name in ("Cellar", "opt", "lib")
                         if (path := prefix / name).is_dir())
    return tuple(sorted(set(roots)))


SYSTEM_SHELL_PATHS = tuple(map(Path, ("/usr/bin", "/bin", "/usr/sbin", "/sbin")))


_SHELL_DIRECTORY_CACHE: OrderedDict[Path, tuple[tuple, frozenset[Path]]] = OrderedDict()
_SHELL_DIRECTORY_CACHE_LOCK = Lock()


def clear_shell_capability_cache() -> None:
    """Discard process-local executable authority snapshots."""
    with _SHELL_DIRECTORY_CACHE_LOCK:
        _SHELL_DIRECTORY_CACHE.clear()


def _directory_executables(
    directory: Path, *, stamps: PathInventoryStamps | None = None,
) -> tuple[tuple, tuple[Path, ...]]:
    stat = directory.stat()
    # File metadata detects edits in place, permission changes and shebang edits.
    # Link metadata also detects replacement even when its target stays the same.
    # One lstat per entry, plus a stat only for links: the admission test (a regular file
    # after following links, executable by us) reads the same snapshot the fingerprint records.
    executables, files = [], []
    if stamps is not None:
        stamps.status("entry", directory, stat)
        stamps.remember("list", directory, os.access(directory, _PATH_ACCESS["list"]))
    for path in sorted(directory.iterdir()):
        try:
            link = path.lstat()
        except OSError:
            if stamps is not None:
                stamps.complete = False
            continue
        if stamps is not None:
            stamps.status("parent" if S_ISDIR(link.st_mode) else "entry", path, link)
        try:
            target = path.stat() if S_ISLNK(link.st_mode) else link
        except OSError as error:
            # Vanished, dangling, looping or unreadable metadata (macOS protects some /usr/sbin
            # entries with EACCES): not admitted, as Path.is_file reports. Skipping only narrows
            # authority; the entry is never a read root.
            if stamps is not None:
                stamps.missing_target(path, error)
            continue
        if stamps is not None and S_ISLNK(link.st_mode):
            stamps.status("target", path, target)
        if not S_ISREG(target.st_mode):
            continue
        executable = os.access(path, os.X_OK)
        if stamps is not None:
            stamps.remember("execute", path, executable)
        if not executable:
            continue
        executables.append(path)
        files.append((path.name, _file_fingerprint(target), _file_fingerprint(link)))
    return (directory, _file_fingerprint(stat), tuple(files)), tuple(executables)


def path_executable_roots(
    directories: tuple[Path, ...] | list[Path], *, stamps: PathInventoryStamps | None = None,
) -> tuple[
    dict[Path, set[Path]], PathFingerprints,
]:
    """Scan each physical directory once, retaining authority for each PATH spelling."""
    scans = {}
    by_directory = {}
    inputs: PathFingerprints = {}
    for directory in dict.fromkeys(directories):
        if not directory.is_dir():
            if stamps is not None:
                stamps.skipped_target(directory, directory=True)
            continue
        _, _, resolved, status = _walk_symlinks(directory, inputs, stamps=stamps)
        if resolved not in scans:
            if stamps is None:
                key, _ = _directory_executables(resolved)
            else:
                key, _ = _directory_executables(resolved, stamps=stamps)
            if key[1] != _file_fingerprint(status):
                raise OSError(f"shell PATH directory changed during scanning: {directory}")
            scans[resolved] = key[2]
        by_directory[directory] = {
            root for name, target, _ in scans[resolved]
            for root in _executable_read_roots(directory / name, inputs, target, stamps=stamps)
        }
    return by_directory, inputs


def path_authority_fingerprint(inputs: Mapping) -> str:
    encoded = json.dumps(inputs, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _local_path_devices() -> set[bytes]:
    # Kernel-managed local filesystems supply the ctime semantics the proof uses.
    # Virtual and remote filesystems can change bytes without a comparable stamp.
    if sys.platform != "linux":
        return set()
    with open("/proc/self/mountinfo", "rb") as stream:
        data = stream.read((1 << 20) + 1)
    if len(data) > 1 << 20:
        return set()
    local, other = set(), set()
    for line in data.splitlines():
        left, _, right = line.partition(b" - ")
        fields, filesystem = left.split(), right.split()
        if len(fields) < 3 or not filesystem:
            return set()
        (local if filesystem[0] in _LOCAL_PATH_FILESYSTEMS else other).add(fields[2])
    return local - other


def current_path_authority_fingerprint(paths: list[str] | dict[str, list[str]]) -> str:
    """Check executable identities, script metadata and link inputs without rereading headers."""
    if isinstance(paths, list):
        return path_authority_fingerprint({path: _file_fingerprint(os.lstat(path)) for path in paths})
    if not isinstance(paths, dict) or paths.keys() != set(_PATH_CHECKS):
        raise OSError("invalid shell PATH inventory")
    if any(not isinstance(names, list) for names in paths.values()):
        raise OSError("invalid shell PATH dependencies")
    if sum(map(len, paths.values())) > _MAX_PATH_CHECKS:
        raise OSError("shell PATH inventory exceeds the dependency limit")
    observed: dict[str, dict] = {}
    local_devices = _local_path_devices()
    devices = set()
    for kind, names in paths.items():
        checks = observed[kind] = {}
        for name in names:
            if not isinstance(name, str) or not name.startswith("/") or "\0" in name or name in checks:
                raise OSError("invalid shell PATH dependency")
            if kind in _PATH_ACCESS:
                checks[name] = os.access(name, _PATH_ACCESS[kind])
                continue
            try:
                status = os.stat(name) if kind == "target" else os.lstat(name)
            except OSError as error:
                if kind != "target" or error.errno not in _MISSING_TARGET:
                    raise
                checks[name] = None
            else:
                checks[name] = _inventory_stamp(kind, status)
                devices.add(status.st_dev)
    if any(f"{os.major(device)}:{os.minor(device)}".encode() not in local_devices
           for device in devices):
        raise OSError("shell PATH inventory filesystem stamps are unavailable")
    return path_authority_fingerprint(observed)


def path_directory_stamp(path: Path, resolved: Path, stamps: PathInventoryStamps) -> None:
    """Bind a raw PATH spelling to the directory actually scanned."""
    _, _, observed, status = _walk_symlinks(path, stamps=stamps)
    stamps.status("parent", observed, status)
    if observed != resolved:
        stamps.complete = False


def shell_capability(*, workspace, run_env_dir, scratch_home, network, writable,
                     extra_read_roots=(), path_dirs=(),
                     include_path_executable_roots=True, path_snapshot=None) -> ExecCapability:
    if type(include_path_executable_roots) is not bool:
        raise TypeError("include_path_executable_roots must be a boolean")
    home = Path(scratch_home).resolve(strict=True)
    run_env = Path(run_env_dir).resolve(strict=True)
    paths = [run_env / "bin", *(Path(p).resolve(strict=True) for p in path_dirs),
             *SYSTEM_SHELL_PATHS]
    covered = set()
    admitted = set()
    if path_snapshot is not None and include_path_executable_roots:
        covered = set(map(Path, path_snapshot["directories"]))
        admitted = set(map(Path, path_snapshot["roots"]))
    # Runtime libraries are explicit authority in the resulting capability/hash.
    roots = [*system_runtime_read_roots(), Path(sys.base_prefix), Path(sys.prefix),
             Path(sys.executable), *extra_read_roots]
    system_gitconfig = Path("/etc/gitconfig")
    # Freeze executable and shebang authority for shell, jobs and derived custom
    # capabilities. Runtime libraries retain the existing resolved-root policy.
    # Offline commands still load TLS trust at import time (Debian's bundled pip reads its CA bundle
    # when imported, so ensurepip fails without it). Admit the public CA store only, never /etc/ssl/private.
    roots += public_ca_read_roots()
    if network == "full":
        roots += [Path(p).resolve() for p in ("/etc/hosts", "/etc/resolv.conf",
                  "/etc/nsswitch.conf", "/etc/ssl", "/etc/pki") if Path(p).exists()]
    env = {"PATH": os.pathsep.join(dict.fromkeys(map(str, paths))),
           "HOME": str(home), "TMPDIR": str(home), "LANG": "C", "LC_ALL": "C",
           "CI": "1", "NO_COLOR": "1", "GIT_PAGER": "cat",
           "PYTHONDONTWRITEBYTECODE": "1", "PIP_CACHE_DIR": str(run_env / "pip-cache"),
           "npm_config_cache": str(run_env / "npm-cache"),
           "CARGO_HOME": str(run_env / "cargo"), "GOPATH": str(run_env / "go"),
           "GOMODCACHE": str(run_env / "go-mod")}
    if sys.platform == "darwin" and not system_gitconfig.is_file():
        # An absent file outside the read roots yields EPERM rather than ENOENT.
        env["GIT_CONFIG_NOSYSTEM"] = "1"
    expanded: dict[Path, tuple[Path, ...]] = {}

    def read_roots(path: Path) -> tuple[Path, ...]:
        if path not in expanded:
            expanded[path] = executable_read_roots(path)
        return expanded[path]

    admitted.update(root for path in roots if Path(path) not in admitted
                    for root in read_roots(Path(path)))
    run_roots = (run_env, home, Path(workspace).resolve(strict=True))
    for directory in dict.fromkeys(paths):
        if (not include_path_executable_roots
                or (directory in covered and directory != run_env / "bin")
                or not directory.is_dir()):
            continue
        cacheable = not any(directory == root or root in directory.parents for root in run_roots)
        key, executables = _directory_executables(directory)
        with _SHELL_DIRECTORY_CACHE_LOCK:
            cached = _SHELL_DIRECTORY_CACHE.get(directory) if cacheable else None
            if cached is not None and cached[0] == key:
                _SHELL_DIRECTORY_CACHE.move_to_end(directory)
            else:
                cached = None
        if cached is None:
            first = {root for executable in executables for root in read_roots(executable)}
            directory_roots = frozenset(root for path in first for root in read_roots(path))
            current = directory.stat()
            if cacheable and _file_fingerprint(current) == key[1]:
                with _SHELL_DIRECTORY_CACHE_LOCK:
                    _SHELL_DIRECTORY_CACHE[directory] = (key, directory_roots)
                    _SHELL_DIRECTORY_CACHE.move_to_end(directory)
                    if len(_SHELL_DIRECTORY_CACHE) > 64:
                        _SHELL_DIRECTORY_CACHE.popitem(last=False)
        else:
            directory_roots = cached[1]
        admitted.update(directory_roots)
    return ExecCapability("shell", Path(workspace), writable, tuple(admitted),
                          (run_env, home), network, env, ExecLimits())
