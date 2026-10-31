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
            roots = tuple(sorted({Path(p).resolve(strict=True) for p in getattr(self, name)}))
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
    roots = [Path(p) for p in ("/usr", "/bin", "/sbin", "/lib", "/lib64",
                              "/System", "/Library/Apple") if Path(p).exists()]
    roots += [Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(),
              Path(sys.executable).resolve(), *extra_read_roots, *path_dirs]
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
    return ExecCapability("shell", Path(workspace), writable, tuple(roots),
                          (run_env, home), network, env, ExecLimits())
