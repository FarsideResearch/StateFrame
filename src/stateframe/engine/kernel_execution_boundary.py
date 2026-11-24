"""Opt-in trusted-repository boundary; detection is a mistake guard, not security proof.
A root-owned regular /.dockerenv is runtime evidence with a non-initial PID namespace;
nested bwrap in that container remains recognized. Host mounts, network policy, and receipt integrity are unverified.
"""
from dataclasses import dataclass
import os, posixpath, re, stat
from typing import Sequence

BACKEND_PLATFORM = "platform"
BACKEND_CONTAINER = "container"
DETECTOR_VERSION = 1
BOUNDARY_STATEMENT = ("Execution boundary: current container only, for trusted repositories. "
    "The engine created no private filesystem, network or PID namespace. "
    "Code run here may read the provider key, reach the network and alter run records. "
    "Host mounts, network policy and receipt integrity were not verified.")
_INITIAL_PID_NAMESPACE = 4026531836
_CLASSIFICATIONS = frozenset(("container", "not_container", "unknown"))
_NAMESPACE_NAMES = ("pid", "mnt", "net", "user", "uts", "ipc", "cgroup", "time")
_RUNTIME_MOUNT_ROOTS = (("/var/lib/docker", "mount:docker"), ("/var/lib/kubelet", "mount:kubelet"),
    ("/var/lib/containerd", "mount:containerd"), ("/var/lib/containers/storage/overlay", "mount:containers-overlay"),
    ("/var/lib/crio", "mount:crio"), ("/run/containerd", "mount:containerd"), ("/run/containers/storage/overlay", "mount:containers-overlay"))
_INJECTED_MOUNTS = frozenset(("/etc/hosts", "/etc/hostname", "/etc/resolv.conf", "/dev/termination-log"))
_CGROUP_RUNTIME_FORMS = (
    (r"docker(?:-[0-9a-f]{64}\.scope)?", "cgroup:docker"),
    (r"kubepods(?:-(?:besteffort|burstable))?(?:-pod[0-9a-f_]{36})?(?:\.slice)?", "cgroup:kubepods"),
    (r"(?:containerd|cri-containerd)(?:-[0-9a-f]{64}\.scope)?", "cgroup:containerd"),
    (r"(?:cri-o|crio)(?:-[0-9a-f]{64}\.scope)?", "cgroup:crio"),
    (r"(?:podman|libpod)(?:-[0-9a-f]{64}\.scope)?", "cgroup:podman"),
)
_RECORD_KEYS = frozenset(("backend", "policy", "classification", "signals", "detector_version", "namespaces", "statement"))
_VALID_SIGNALS = frozenset((
    "marker:/.dockerenv", "marker:/run/.containerenv",
    *(signal for _, signal in _RUNTIME_MOUNT_ROOTS),
    "cgroup:docker", "cgroup:kubepods", "cgroup:podman", "cgroup:containerd", "cgroup:crio",
))

@dataclass(frozen=True)
class Detection:
    classification: str
    signals: tuple[str, ...]
    namespaces: dict[str, str]
    detector_version: int

class BoundaryRefused(Exception):
    def __init__(self, code: str):
        self.code = code; super().__init__(code)

@dataclass(frozen=True)
class Boundary:
    backend: str
    detection: Detection
    def record(self) -> dict:
        is_container = self.backend == BACKEND_CONTAINER
        return {
            "backend": self.backend,
            "policy": "allow-container" if is_container else "platform",
            "classification": self.detection.classification,
            "signals": list(self.detection.signals),
            "detector_version": self.detection.detector_version,
            "namespaces": dict(self.detection.namespaces),
            "statement": BOUNDARY_STATEMENT if is_container else "",
        }
    @classmethod
    def from_record(cls, record: dict) -> "Boundary":
        if type(record) is not dict or set(record) != _RECORD_KEYS:
            raise ValueError("invalid execution boundary record")
        backend, policy, classification, statement = (record[key] for key in ("backend", "policy", "classification", "statement"))
        container = backend == BACKEND_CONTAINER
        if (type(backend) is not str or backend not in (BACKEND_PLATFORM, BACKEND_CONTAINER) or
                any(type(value) is not str for value in (policy, classification, statement)) or
                policy != ("allow-container" if container else "platform") or classification not in _CLASSIFICATIONS or
                statement != (BOUNDARY_STATEMENT if container else "") or (container and classification != "container")):
            raise ValueError("invalid execution boundary record")
        version, signals, namespaces = (record[key] for key in ("detector_version", "signals", "namespaces"))
        if (type(version) is not int or version != DETECTOR_VERSION or type(signals) is not list or
                any(type(item) is not str or item not in _VALID_SIGNALS for item in signals) or
                len(set(signals)) != len(signals) or type(namespaces) is not dict or
                any(type(key) is not str or key not in _NAMESPACE_NAMES or type(value) is not str or not value.isascii() or not value.isdigit()
                    for key, value in namespaces.items())):
            raise ValueError("invalid execution boundary record")
        if classification == "container" and ("pid" not in namespaces or namespaces["pid"] == str(_INITIAL_PID_NAMESPACE) or not signals):
            raise ValueError("invalid execution boundary evidence")
        return cls(backend, Detection(classification, tuple(signals), dict(namespaces), version))

def _marker_signals(fs_root: str) -> tuple[list[str], bool]:
    signals, unreadable = [], False
    for relative, signal in ((".dockerenv", "marker:/.dockerenv"), ("run/.containerenv", "marker:/run/.containerenv")):
        try:
            info = os.stat(os.path.join(fs_root, relative), follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            unreadable = True; continue
        if info.st_uid == 0 and stat.S_ISREG(info.st_mode):
            signals.append(signal)
    return signals, unreadable

def _unescape_mountinfo(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)

def _runtime_mount_signal(path: str) -> str | None:
    path = posixpath.normpath(path)
    return next((signal for root, signal in _RUNTIME_MOUNT_ROOTS if path == root or path.startswith(root + "/")), None)

def _mount_signals(contents: str) -> list[str]:
    signals = []
    for line in contents.splitlines():
        halves = line.split(" - ", 1)
        if len(halves) != 2:
            continue
        before, after = halves[0].split(), halves[1].split()
        if (len(before) < 6 or len(after) < 3 or not before[0].isdigit() or not before[1].isdigit()
                or re.fullmatch(r"\d+:\d+", before[2]) is None or not before[3].startswith("/") or not before[4].startswith("/")):
            continue
        root, point = (_unescape_mountinfo(field) for field in before[3:5])
        if point == "/":
            if signal := _runtime_mount_signal(root):
                signals.append(signal)
            if after[0] == "overlay":
                options = (option.partition("=") for field in after[2:] for option in _unescape_mountinfo(field).split(","))
                signals.extend(signal for key, sep, value in options if sep and key in ("lowerdir", "upperdir")
                               for layer in value.split(":") if (signal := _runtime_mount_signal(layer)))
        elif point in _INJECTED_MOUNTS and (signal := _runtime_mount_signal(root)):
            signals.append(signal)
    return signals

def _cgroup_signals(contents: str) -> list[str]:
    signals = []
    for line in contents.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[2].startswith("/"):
            continue
        path = fields[2]
        for part in path.split("/"):
            signals.extend(signal for pattern, signal in _CGROUP_RUNTIME_FORMS if re.fullmatch(pattern, part))
    return signals

def detect(proc_root: str = "/proc", fs_root: str = "/") -> Detection:
    """Read kernel/runtime evidence without consulting environment or cwd."""
    try:
        ns_root = os.path.join(proc_root, "self", "ns")
        pid_inode = str(os.stat(os.path.join(ns_root, "pid")).st_ino)
        namespaces, signals = {"pid": pid_inode}, []
        for name in _NAMESPACE_NAMES[1:]:
            try:
                namespaces[name] = str(os.stat(os.path.join(ns_root, name)).st_ino)
            except OSError:
                pass
        markers, fs_unreadable = _marker_signals(fs_root)
        signals.extend(markers)
        unreadable = fs_unreadable
        for name, parser in (("mountinfo", _mount_signals), ("cgroup", _cgroup_signals)):
            try:
                with open(os.path.join(proc_root, "self", name), encoding="utf-8", errors="replace") as stream: signals.extend(parser(stream.read()))
            except OSError:
                unreadable = True
        signals = list(dict.fromkeys(signals))
        noninitial = int(pid_inode) != _INITIAL_PID_NAMESPACE
        classification = ("unknown" if unreadable else "container" if signals else "not_container") if noninitial else "not_container"
        return Detection(classification, tuple(signals), namespaces, DETECTOR_VERSION)
    except Exception:
        return Detection("unknown", (), {}, DETECTOR_VERSION)

def authorize(selected: bool, *, conflicts: Sequence[str], detection: Detection, uid: int) -> Boundary:
    if not selected:
        return Boundary(BACKEND_PLATFORM, detection)
    if "mutations" in conflicts:
        raise BoundaryRefused("CONTAINER_MUTATIONS")
    if conflicts:
        raise BoundaryRefused("CONTAINER_CONFLICT")
    if uid == 0:
        raise BoundaryRefused("CONTAINER_ROOT_UID")
    if detection.classification != "container":
        raise BoundaryRefused("CONTAINER_UNKNOWN" if detection.classification == "unknown" else "CONTAINER_NOT_DETECTED")
    return Boundary(BACKEND_CONTAINER, detection)

def check_resume(recorded: dict | None, current: Boundary) -> None:
    if recorded is None:
        if current.backend == BACKEND_CONTAINER:
            raise BoundaryRefused("CONTAINER_RESUME_MISMATCH")
        return
    try:
        previous = Boundary.from_record(recorded)
    except (TypeError, ValueError, KeyError):
        raise BoundaryRefused("CONTAINER_RESUME_MISMATCH") from None
    if any(previous.record()[key] != current.record()[key] for key in ("backend", "policy", "classification")):
        raise BoundaryRefused("CONTAINER_RESUME_MISMATCH")

_active_boundary: Boundary | None = None
_DEFAULT_BOUNDARY = Boundary(BACKEND_PLATFORM, Detection("not_container", (), {}, DETECTOR_VERSION))

def activate(boundary: Boundary) -> None:
    global _active_boundary
    if not isinstance(boundary, Boundary): raise TypeError("boundary must be a Boundary")
    if _active_boundary is not None: raise RuntimeError("execution boundary already activated in this process")
    _active_boundary = boundary

def active() -> Boundary:
    return _active_boundary if _active_boundary is not None else _DEFAULT_BOUNDARY
