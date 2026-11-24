"""Installation-private receipts for successful startup probes."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import secrets
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time

from .send1_wave4 import SPEC_KEY, SWITCH, child_selection, enabled, launch_selection
from .startup_cache_writes import active_writer, defer

_DIRECTORY = "startup-probes-v1"
_MAX_ENTRY = 256 << 10
_SLOTS = 1024
_MAX_STAMPS = 65_536
_DEFAULT_AGE = 86_400
ENGINE_VERSION = "engine-python-version"
_SCOPE = ContextVar("startup_probe_scope", default=None)
_CHILD_STORE_MISSING = False
STORE_MISSING_ENV = "STATEFRAME_STARTUP_PROBE_STORE_MISSING"
_PACKAGE = Path(__file__).resolve().parent


@contextmanager
def child_scope(*, store_missing=False):
    """Expose the recorded selection to all workers in the dedicated run child."""
    global _CHILD_STORE_MISSING
    previous = _CHILD_STORE_MISSING
    _CHILD_STORE_MISSING = store_missing
    try:
        with child_selection():
            yield
    finally:
        _CHILD_STORE_MISSING = previous


@contextmanager
def launch_scope(selected: bool, identity: dict | None):
    """Give a host call the current launch's selection and verified home."""
    token = _SCOPE.set(identity)
    try:
        with launch_selection(selected):
            yield
    finally:
        _SCOPE.reset(token)


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _open_dir(path, *, parent=None):
    return os.open(path, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                   dir_fd=parent)


def _private(info, mode: int) -> bool:
    return info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == mode


def _read_file(directory: int, name: str, limit: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    with os.fdopen(os.open(name, flags, dir_fd=directory), "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or not _private(before, 0o600)
                or before.st_nlink != 1 or not 0 < before.st_size <= limit):
            raise ValueError("untrusted probe file")
        data = stream.read(limit + 1)
        if len(data) != before.st_size or _stamp(before) != _stamp(os.fstat(stream.fileno())):
            raise ValueError("probe file changed while reading")
        return data, before


@dataclass(frozen=True)
class _ProbeStore:
    path: Path
    home: int
    directory: int
    secret: bytes
    key_stamp: tuple[int, ...]

    def current(self) -> bool:
        home = os.fstat(self.home)
        directory = os.fstat(self.directory)
        if not _private(home, 0o700) or not _private(directory, 0o700):
            return False
        named_home = self.path.stat(follow_symlinks=False)
        named_directory = os.stat(_DIRECTORY, dir_fd=self.home, follow_symlinks=False)
        key = os.stat("identity.key", dir_fd=self.home, follow_symlinks=False)
        return (os.path.samestat(home, named_home) and os.path.samestat(directory, named_directory)
                and key.st_nlink == 1 and _stamp(key) == self.key_stamp)

    def read(self, slot: str) -> bytes:
        if not self.current():
            raise ValueError("probe store authority changed")
        data, _ = _read_file(self.directory, slot, _MAX_ENTRY)
        return data


@contextmanager
def _store(*, create: bool):
    from .engine.kernel_request_auth_identity import _installation_home

    expected = _SCOPE.get()
    home = Path(expected["realpath"]) if expected else _installation_home()
    if not home.is_absolute() or ".." in home.parts:
        raise ValueError("invalid probe home")
    # Walk the path by descriptor so neither the home nor a parent component can
    # redirect the operation through a symlink, including during publication.
    directory = _open_dir(home.anchor)
    try:
        for part in home.parts[1:]:
            child = _open_dir(part, parent=directory)
            os.close(directory)
            directory = child
        info = os.fstat(directory)
        if not _private(info, 0o700):
            raise ValueError("untrusted probe home")
        key, key_info = _read_file(directory, "identity.key", 32)
        identity = {
            "realpath": os.fspath(home),
            "home_st_dev": info.st_dev,
            "home_st_ino": info.st_ino,
            "key_st_dev": key_info.st_dev,
            "key_st_ino": key_info.st_ino,
            "key_st_ctime_ns": key_info.st_ctime_ns,
        }
        if len(key) != 32 or (expected is not None and expected != identity):
            raise ValueError("probe home identity changed")
        if create:
            try:
                os.mkdir(_DIRECTORY, 0o700, dir_fd=directory)
            except FileExistsError:
                pass
        cache = _open_dir(_DIRECTORY, parent=directory)
        try:
            if not _private(os.fstat(cache), 0o700):
                raise ValueError("untrusted probe directory")
            binding = _json([_DIRECTORY, str(home), info.st_dev, info.st_ino, _stamp(key_info)])
            yield _ProbeStore(home, directory, cache, hmac.digest(key, binding, "sha256"),
                              _stamp(key_info))
        finally:
            os.close(cache)
    finally:
        os.close(directory)


def engine_version_facts(facts: object) -> bool:
    return (type(facts) is dict and set(facts) == {"version"}
            and type(facts["version"]) is list and len(facts["version"]) == 3
            and all(type(part) is int and 0 <= part <= 2**31 - 1 for part in facts["version"]))


def _valid(kind, facts, validate):
    if type(facts) is not dict or not facts:
        return False
    if kind == ENGINE_VERSION and not engine_version_facts(facts):
        return False
    return validate is None or validate(facts) is True


def _entry_key(kind, fingerprint):
    if (not isinstance(kind, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", kind)
            or not fingerprint):
        raise ValueError("invalid probe key")
    key = _digest([_DIRECTORY, kind, fingerprint])
    return key, f"{int(key[:8], 16) % _SLOTS:04x}.json"


def _launch_reader():
    writer = active_writer()
    identity = _SCOPE.get()
    home = identity["realpath"] if identity else os.environ.get("STATEFRAME_HOME")
    pin = tuple(sorted(identity.items())) if identity else None
    return writer.reader((_DIRECTORY, home, pin), lambda: _store(create=False))


def reads_available() -> bool:
    return not _CHILD_STORE_MISSING and (active_writer() is None or _launch_reader() is not None)


def lookup(kind, fingerprint, max_age=_DEFAULT_AGE, *, validate=None) -> dict | None:
    """Read authenticated facts. Consumers can supply their typed schema predicate."""
    if not enabled() or _CHILD_STORE_MISSING:
        return None
    try:
        if not math.isfinite(max_age) or max_age <= 0:
            return None
        key, slot = _entry_key(kind, fingerprint)
        if active_writer() is not None:
            reader = _launch_reader()
            if reader is None:
                return None
            reader_scope = nullcontext(reader)
        else:
            reader_scope = _store(create=False)
        with reader_scope as reader:
            mac, payload = reader.read(slot).split(b"\n", 1)
            if not hmac.compare_digest(mac, hmac.digest(reader.secret, payload, "sha256").hex().encode()):
                return None
            record = json.loads(payload)
            if (type(record) is not dict or set(record) != {"key", "created", "clock", "boot", "facts"}
                    or record["key"] != key or record["boot"] != _boot_id()):
                return None
            age = time.time() - record["created"]
            elapsed = time.monotonic() - record["clock"]
            facts = record["facts"]
            if (0 <= age <= max_age and 0 <= elapsed <= max_age
                    and _valid(kind, facts, validate) and reader.current()):
                return facts
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError,
            subprocess.SubprocessError):
        pass
    return None


def publish(kind, fingerprint, facts, *, validate=None) -> bool:
    """Publish complete positive facts. A slot collision is a miss for its former key."""
    if not enabled():
        return False
    return defer(kind, lambda: _publish(kind, fingerprint, facts, validate))


def _publish(kind, fingerprint, facts, validate):
    try:
        key, slot = _entry_key(kind, fingerprint)
        payload = _json({"key": key, "created": time.time(), "clock": time.monotonic(),
                         "boot": _boot_id(), "facts": facts})
        # Validate the detached JSON representation that a future reader will see.
        if len(payload) + 65 > _MAX_ENTRY or not _valid(kind, json.loads(payload)["facts"], validate):
            return False
        with _store(create=True) as store:
            directory = store.directory
            data = hmac.digest(store.secret, payload, "sha256").hex().encode() + b"\n" + payload
            temporary = f".{slot}.{secrets.token_hex(12)}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
            descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, slot, src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
            finally:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError,
            subprocess.SubprocessError):
        if active_writer() is not None:
            raise
        return False
    return True


def _bounded_read(path: Path, limit: int = 1 << 20) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("unproved probe fingerprint input")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("probe fingerprint input exceeds its bound")
    return data


def _boot_id() -> str:
    if sys.platform == "linux":
        raw = _bounded_read(Path("/proc/sys/kernel/random/boot_id"), 64)
    elif sys.platform == "darwin":
        raw = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"], timeout=1, stderr=subprocess.DEVNULL,
        )
    else:
        raise ValueError("probe boot identity is unavailable")
    value = raw.decode("ascii").strip().lower()
    if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is None:
        raise ValueError("invalid probe boot identity")
    return value


def host_fingerprint() -> str | None:
    """Bind facts to a Linux boot, kernel, namespaces and process security context."""
    if sys.platform != "linux":
        return None
    try:
        proc = Path("/proc/self")
        fields = (b"Uid:", b"Gid:", b"Groups:", b"Cap", b"NoNewPrivs:", b"Seccomp", b"Umask:")
        context = [line.decode("ascii") for line in _bounded_read(proc / "status").splitlines()
                   if line.startswith(fields)]
        if len(context) < 10:
            return None
        namespaces = [os.readlink(proc / "ns" / name) for name in
                      ("user", "mnt", "pid", "net", "ipc", "uts", "cgroup")]
        proc_inputs = (Path("/proc/sys/kernel/random/boot_id"), proc / "cgroup",
                       proc / "uid_map", proc / "gid_map", proc / "mountinfo",
                       proc / "attr" / "current")
        return _digest([_DIRECTORY, os.uname(), sys.implementation.cache_tag,
                        context, namespaces,
                        [hashlib.sha256(_bounded_read(path)).hexdigest() for path in proc_inputs]])
    except (OSError, ValueError, UnicodeError):
        return None


class _Dependencies:
    """Bounded ctime/inode stamps for local, kernel-owned filesystems."""

    def __init__(self):
        self.inputs = {}
        self.mounts = {}
        self.tree_entries = 0
        for line in _bounded_read(Path("/proc/self/mountinfo")).splitlines():
            left, right = line.split(b" - ", 1)
            fields = left.split()
            mount = re.sub(rb"\\([0-7]{3})", lambda m: bytes([int(m[1], 8)]), fields[4])
            self.mounts[os.fsdecode(mount)] = right.split()[0]

    def _local(self, path: Path) -> None:
        for candidate in (path, *path.parents):
            kind = self.mounts.get(str(candidate))
            if kind is not None:
                if kind not in {b"ext4", b"xfs", b"btrfs", b"tmpfs"}:
                    raise ValueError("unverified filesystem stamps")
                return
        raise ValueError("no filesystem identity")

    def add(self, path, info, link=None):
        if len(self.inputs) >= _MAX_STAMPS:
            raise ValueError("too many probe dependencies")
        value = _stamp(info), link
        if self.inputs.setdefault(os.fspath(path), value) != value:
            raise ValueError("probe dependency changed while enumerating")

    def path(self, path: Path, *, optional=False) -> Path | None:
        # Include every symlink and intermediate directory's access identity. A
        # sibling creation under an ancestor does not change traversal authority.
        from .engine.kernel_exec_capability import _walk_symlinks

        try:
            _, links, resolved, info = _walk_symlinks(path)
        except FileNotFoundError:
            if not optional:
                raise
            # A broken symlink has unproved provenance, independent of its target.
            if path.is_symlink():
                raise
            self.inputs[str(path)] = None
            return None
        self._local(resolved)
        for link, target in links.items():
            self._local(link)
            self.add(link, link.lstat(), target)
        self.add(resolved, info)
        ancestors = {parent for item in (*links, resolved) for parent in item.parents}
        for parent in ancestors:
            self.inputs[str(parent) + "/"] = _stamp(parent.stat())[:5]
        return resolved

    def tree(self, root: Path) -> None:
        pending = [str(root)]
        tree = hashlib.sha256()
        stamp = struct.Struct("!QQqI")
        # The checked mount supplies the device and stamp semantics for the
        # whole tree. A nested mount or symlink needs a separate proof.
        mount_points = {path for path in self.mounts if path.startswith(str(root) + "/")}
        while pending:
            path = pending.pop()
            before = os.stat(path, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError("probe import directory changed")
            tree.update(os.fsencode(path) + b"\0")
            tree.update(stamp.pack(before.st_dev, before.st_ino, before.st_ctime_ns, before.st_mode))
            with os.scandir(path) as entries:
                # Bound the directory before allocating the sorted projection.
                names = []
                for entry in entries:
                    self.tree_entries += 1
                    if self.tree_entries > _MAX_STAMPS:
                        raise ValueError("too many probe import dependencies")
                    names.append(entry)
                for entry in sorted(names, key=lambda item: item.name):
                    info = entry.stat(follow_symlinks=False)
                    if entry.path in mount_points or (
                            not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode)):
                        raise ValueError("unproved import dependency")
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(entry.path)
                    tree.update(os.fsencode(entry.name) + b"\0")
                    tree.update(stamp.pack(info.st_dev, info.st_ino, info.st_ctime_ns, info.st_mode))
            if _stamp(before) != _stamp(os.stat(path, follow_symlinks=False)):
                raise ValueError("probe import membership changed")
        self.inputs[str(root) + "/tree"] = tree.hexdigest()


def _interpreter_dependencies(python: str) -> _Dependencies:
    import site

    dependencies = _Dependencies()
    binary = Path(python)
    target = dependencies.path(binary)
    base = Path(sys.base_prefix)
    host_binary = Path(sys.executable)
    host_target = dependencies.path(host_binary)
    if (not binary.is_absolute() or target != host_target
            or target.parent != base.resolve(strict=True) / "bin"
            or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", binary.name)
            or not os.access(binary, os.X_OK)
            or _bounded_header(target) != b"\x7fELF"):
        raise ValueError("unproved interpreter layout")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib = dependencies.path(base / sys.platlibdir / version)
    dependencies.tree(stdlib)
    _no_sitecustomize(stdlib)
    dynload = dependencies.path(stdlib / "lib-dynload", optional=True)
    if dynload is not None:
        _no_sitecustomize(dynload)
    # Standard CPython checks the zip path even when it was absent at build time.
    zip_name = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    if dependencies.path(base / sys.platlibdir / zip_name, optional=True) is not None:
        raise ValueError("unproved Python zip layout")

    prefixes = set()
    for executable in (binary, host_binary):
        prefix = executable.parent.parent
        if dependencies.path(executable.parent / "pyvenv.cfg", optional=True) is not None:
            raise ValueError("unproved interpreter configuration location")
        _reject_layout_files(dependencies, executable)
        cfg = dependencies.path(prefix / "pyvenv.cfg", optional=True)
        if cfg is None:
            if prefix.resolve() != base.resolve() or base in {Path("/usr"), Path("/usr/local")}:
                raise ValueError("unproved system interpreter layout")
        else:
            rows = [tuple(map(str.strip, line.split("=", 1)))
                    for line in _bounded_read(cfg, 16 << 10).decode().splitlines() if line.strip()]
            values = {name.lower(): value for name, value in rows}
            runtime_home = Path(values.get("home", ""))
            if (len(values) != len(rows) or not runtime_home.is_absolute()
                    or dependencies.path(runtime_home) != target.parent
                    or values.get("include-system-site-packages", "").lower() != "false"):
                raise ValueError("unproved interpreter configuration")
        prefixes.add(prefix)
    # The resolved launcher also starts with site enabled inside the sandbox.
    _reject_layout_files(dependencies, target)
    if dependencies.path(base / "pyvenv.cfg", optional=True) is not None:
        raise ValueError("unproved base interpreter configuration")
    prefixes.add(base)
    for raw in site.getsitepackages(sorted(map(str, prefixes))):
        directory = dependencies.path(Path(raw), optional=True)
        if directory is None:
            continue
        dependencies.tree(directory)
        _site_dependencies(directory, dependencies)

    # Loader config and the native runtime used by the current CPython. The
    # sandbox witness independently exercises live loader/filter initialization.
    for path in ("/etc/ld.so.cache", "/etc/ld.so.preload"):
        dependencies.path(Path(path), optional=True)
    maps = _bounded_read(Path("/proc/self/maps"))
    loaders = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith(b"/"):
            path = Path(os.fsdecode(fields[5].removesuffix(b" (deleted)")))
            if re.match(r"(?:ld-linux|lib(?:python|c[.-]|m[.-]|dl[.-]|pthread[.-]|util[.-]|rt[.-]))",
                        path.name):
                if fields[5].endswith(b" (deleted)"):
                    raise ValueError("runtime was replaced while mapped")
                loaders.add(path)
    if not any(path.name.startswith("ld-linux") for path in loaders):
        raise ValueError("unproved dynamic loader")
    for path in sorted(loaders):
        dependencies.path(path)
        if path.name.startswith("libpython"):
            _reject_layout_files(dependencies, path)
    return dependencies


def _reject_layout_files(dependencies: _Dependencies, executable: Path) -> None:
    for filename in (executable.with_suffix("._pth").name, executable.name + "._pth",
                     "pybuilddir.txt"):
        if dependencies.path(executable.with_name(filename), optional=True) is not None:
            raise ValueError("unproved alternate Python import layout")


def _bounded_header(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    with os.fdopen(os.open(path, flags), "rb") as stream:
        return stream.read(4)


def _site_dependencies(directory: Path, dependencies: _Dependencies) -> None:
    # A listing error invalidates the interpreter fingerprint. Path.glob can
    # suppress that error and make import hooks look absent.
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith(".") or not entry.name.endswith(".pth"):
            continue
        for line in _bounded_read(entry, 64 << 10).decode().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith(("import ", "import\t")):
                raise ValueError("executable Python site hook")
            candidate = Path(line.rstrip())
            if not candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("unproved Python site path")
            resolved = dependencies.path(candidate, optional=True)
            if resolved is not None:
                # CPython adds this path to sys.path but does not process .pth
                # files within it. Its importable files still need their stamps.
                dependencies.tree(resolved)
                _no_sitecustomize(resolved)
    _no_sitecustomize(directory)


def _no_sitecustomize(directory: Path) -> None:
    if any(entry.name.startswith("sitecustomize") for entry in directory.iterdir()):
        raise ValueError("executable Python site customization")


def interpreter_fingerprint(python: str | Path, *, require_pytest: bool = False) -> str | None:
    """Return an exact dependency stamp for eligible isolated CPython startup.

    Pytest imports and executable site hooks need the original live probe.
    """
    if (sys.platform != "linux" or sys.implementation.name != "cpython" or require_pytest
            or any(name.startswith("LD_") for name in os.environ)):
        return None
    try:
        host = host_fingerprint()
        if host is not None:
            return _digest([host, os.fspath(python), sys.version_info[:3], "-I", "-B",
                            _interpreter_dependencies(os.fspath(python)).inputs])
    except (OSError, ValueError, RuntimeError, TypeError, OverflowError, struct.error):
        pass
    return None


def _sandbox_backend() -> str | None:
    from .engine.kernel_test_isolation import sandbox_executable
    from .engine.kernel_safe_paths import container_active
    from .engine.sf_launch import _current_cgroup

    if container_active():
        return None
    if (platform.system() != "Linux" or os.getuid() == 0
            or threading.current_thread() is not threading.main_thread()
            or os.getuid() != os.geteuid() or os.getgid() != os.getegid()):
        return None
    binary = sandbox_executable("bwrap")
    system = "/usr/bin/bwrap"
    if binary is None or not os.path.samefile(binary, system):
        return None
    info = os.stat(system)
    # Only run the independent witness with the system installation, and retain
    # the supervised path where the command can use delegated cgroup controllers.
    overflow_uid = int(_bounded_read(Path("/proc/sys/kernel/overflowuid"), 32))
    if (info.st_uid not in {0, overflow_uid} or info.st_mode & (stat.S_ISUID | stat.S_ISGID | 0o022)
            or any(os.access(path, os.W_OK) for path in (system, "/usr/bin", "/usr"))):
        return None
    group = _current_cgroup()
    if group is None or os.access(group, os.W_OK):
        return None
    try:
        os.getxattr(system, "security.capability")
    except OSError as error:
        if error.errno != errno.ENODATA:
            return None
    else:
        return None
    return binary


_PROBE_SOURCES = (
    "kernel_safe_probe.py", "kernel_exec_session.py", "kernel_exec_supervisor.py",
    "kernel_exec_capability.py", "kernel_exec_tracker.py", "kernel_execution_policy.py",
    "kernel_test_isolation.py", "kernel_safe_paths.py", "kernel_job_output.py",
    "kernel_reason_log.py", "sf_launch.py",
)


def engine_fingerprint(python: str, code: str, wall: float) -> str | None:
    """Key a version result to the engine check and its executable sandbox."""
    if wall < 60 or "STATEFRAME_EXEC_RPC_SECONDS" in os.environ:
        return None
    try:
        backend = _sandbox_backend()
        if backend is None:
            return None
        interpreter = interpreter_fingerprint(python)
        if interpreter is None:
            return None
        dependencies = _Dependencies()
        for path in (Path(backend), Path("/usr/bin/bwrap"), Path(tempfile.gettempdir())):
            dependencies.path(path.absolute())
        scratch = Path(tempfile.gettempdir()).resolve()
        # Scratch membership is per launch. Its traversal and mount identity
        # matter, including a changed mode, inode, owner, or mount options.
        dependencies.inputs[str(scratch)] = _stamp(scratch.stat())[:5]
        implementation = []
        for source in (_PACKAGE / Path(__file__).name, _PACKAGE / "runner.py",
                       *(_PACKAGE / "engine" / name for name in _PROBE_SOURCES)):
            resolved = dependencies.path(source)
            implementation.append(hashlib.sha256(_bounded_read(resolved)).hexdigest())
        limits = [resource.getrlimit(getattr(resource, name)) for name in
                  ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_CORE", "RLIMIT_DATA",
                   "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_NPROC", "RLIMIT_STACK")]
        return _digest([ENGINE_VERSION, interpreter, "bwrap", backend, code, wall,
                        limits, implementation, dependencies.inputs])
    except (OSError, ValueError, RuntimeError, TypeError, OverflowError, struct.error):
        return None


def check_sandbox(python: str, wall: float) -> bool:
    """Exercise live nested mounts, namespaces, filters and exec on an eligible hit.

    The command is a system no-op. A refusal, timeout or unproved policy goes
    to the supervised interpreter probe.
    """
    from .engine.kernel_exec_capability import executable_read_roots, shell_capability
    from .engine.kernel_test_isolation import (
        _bwrap_userns_guard_supported, _interpreter_roots, capability_command,
    )

    try:
        bwrap = _sandbox_backend()
        if bwrap is None or not _bwrap_userns_guard_supported("/usr/bin/bwrap"):
            return False
        with tempfile.TemporaryDirectory(prefix="stateframe-startup-") as workspace, \
                tempfile.TemporaryDirectory(prefix="stateframe-probe-") as scratch:
            home, environment, session = (Path(scratch) / name for name in ("home", "env", "session"))
            for directory in (home, environment, session):
                directory.mkdir()
            executable = Path(python)
            cap = shell_capability(
                workspace=Path(workspace), run_env_dir=environment, scratch_home=home,
                network="none", writable=False, include_path_executable_roots=False,
                extra_read_roots=(*executable_read_roots(executable),
                                  *map(Path, _interpreter_roots(executable))),
            )
            command, _ = capability_command(
                ["/usr/bin/true"], cap, platform_backend="bwrap", shared_namespace=True,
            )
            # The supervisor resolves the inner sandbox with this fixed PATH.
            # Require the further-userns guard in the nested boundary too.
            if "--assert-userns-disabled" not in command:
                return False
            command[0] = "/usr/bin/bwrap"
            outer = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                     "--as-pid-1", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev"]
            for directory in (Path(workspace), home, session):
                outer.extend(("--bind", str(directory), str(directory)))
            result = subprocess.run(
                [*outer, "--", *command], env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, timeout=min(wall, 5), check=False,
            )
            return result.returncode == 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
        return False
