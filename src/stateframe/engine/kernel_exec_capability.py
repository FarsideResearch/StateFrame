"""Immutable host-issued execution authority, independent of test policy."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Literal

NetworkMode = Literal["none", "loopback", "full"]


def _walk_symlinks(path: Path) -> tuple[tuple[Path, ...], dict[Path, str]]:
    """Return full-path hops and links, without admitting a link's whole directory.

    Resolving only the endpoint loses the paths the kernel checks during exec.
    Keep the links themselves, never broaden authority to their parent directories.
    """
    pending = list(Path(path).absolute().parts[1:])
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
        if current.is_symlink():
            traversals += 1
            if traversals > 40:
                raise ValueError(f"executable symlink chain exceeds 40 hops: {path}")
            hops.add(current.joinpath(*pending))
            links[current] = os.readlink(current)
            target = Path(links[current])
            current = Path("/") if target.is_absolute() else current.parent
            pending = list(target.parts[1:] if target.is_absolute() else target.parts) + pending
    current.stat()  # Missing targets are not executable authority.
    return tuple(sorted(hops | {current})), links


def symlink_hops(path: Path) -> tuple[Path, ...]:
    """Admit each full executable path along a bounded symlink traversal."""
    return _walk_symlinks(path)[0]


def executable_symlinks(paths: tuple[Path, ...]) -> dict[Path, str]:
    """Links to recreate in a private mount tree; targets are admitted separately."""
    return {link: target for path in paths for link, target in _walk_symlinks(path)[1].items()}


def executable_read_roots(path: Path) -> tuple[Path, ...]:
    """Include a script's absolute shebang interpreter and its complete link chain."""
    roots: set[Path] = set()
    seen: set[Path] = set()
    pending = [Path(path)]
    while pending:
        candidate = pending.pop()
        hops = symlink_hops(candidate)
        roots.update(hops)
        resolved = candidate.resolve(strict=True)
        if resolved in seen or not resolved.is_file():
            continue
        if len(seen) >= 40:
            raise ValueError(f"executable interpreter chain exceeds 40 files: {path}")
        seen.add(resolved)
        # Execute-only native binaries (for example macOS sudo) have no
        # readable script header. Their link authority still remains explicit.
        if not os.access(resolved, os.R_OK):
            continue
        with resolved.open("rb") as stream:
            line = stream.readline(4096)
        if line.startswith(b"#!"):
            words = line[2:].split()
            if words:
                interpreter = Path(os.fsdecode(words[0]))
                if interpreter.is_absolute() and interpreter.exists():
                    pending.append(interpreter)
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


@dataclass(frozen=True)
class ExecCapability:
    purpose: Literal["shell", "job", "git", "custom_tool", "probe"]
    workspace: Path
    workspace_writable: bool
    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]
    network: NetworkMode
    env: Mapping[str, str]
    limits: ExecLimits
    sandboxed: bool = True

    def __post_init__(self) -> None:
        if self.purpose not in {"shell", "job", "git", "custom_tool", "probe"}:
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
        return {"purpose": self.purpose, "workspace": str(self.workspace),
                "workspace_writable": self.workspace_writable,
                "read_roots": [str(p) for p in self.read_roots],
                "write_roots": [str(p) for p in self.write_roots],
                "network": self.network, "env": dict(self.env),
                "limits": asdict(self.limits), "sandboxed": self.sandboxed}

    def digest(self) -> str:
        data = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode()
        return hashlib.sha256(data).hexdigest()


def capability_from_dict(value: dict) -> ExecCapability:
    values = dict(value)
    values["limits"] = ExecLimits(**values["limits"])
    return ExecCapability(**values)


def shell_capability(*, workspace, run_env_dir, scratch_home, network, writable,
                     extra_read_roots=(), path_dirs=()) -> ExecCapability:
    home = Path(scratch_home).resolve(strict=True)
    run_env = Path(run_env_dir).resolve(strict=True)
    paths = [run_env / "bin", *(Path(p).resolve(strict=True) for p in path_dirs),
             Path("/usr/bin"), Path("/bin"), Path("/usr/sbin"), Path("/sbin")]
    # Runtime libraries are explicit authority in the resulting capability/hash.
    # /private/var/select holds macOS's /bin/sh selection link (sh -> bash|dash); without it
    # every /bin/sh start prints "Error opening /private/var/select/sh".
    roots = [Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64",
                              "/System", "/Library/Apple", "/private/var/select") if Path(p).exists()]
    roots += [Path(sys.base_prefix), Path(sys.prefix),
              Path(sys.executable), *extra_read_roots, *path_dirs]
    # Freeze executable and shebang authority for shell, jobs and derived custom
    # capabilities. Runtime libraries retain the existing resolved-root policy.
    for directory in dict.fromkeys(paths):
        if directory.is_dir():
            for executable in sorted(directory.iterdir()):
                if executable.is_file() and os.access(executable, os.X_OK):
                    roots.extend(executable_read_roots(executable))
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
    admitted = {root for path in roots for root in executable_read_roots(Path(path))}
    return ExecCapability("shell", Path(workspace), writable, tuple(admitted),
                          (run_env, home), network, env, ExecLimits())
