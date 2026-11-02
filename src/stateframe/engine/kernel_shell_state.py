"""Persistent shell state (STATEFRAME_SHELL_STATE): cwd and exported variables survive between shell calls.

Off by default. When on, each foreground `shell` call runs through a fixed /bin/sh wrapper that starts in
the previous call's directory with the previous call's exported variables, runs the command with `eval`, and
on exit writes the final directory and environment to the scratch home. The engine keeps only the cwd
relative to the workspace and the variables that differ from the sandbox's base environment.

The state is part of each call's tape intent, and the state a call produced is stored in the run folder, so
a replayed call restores exactly the state its live run left without executing again.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

MODE_ENV = "STATEFRAME_SHELL_STATE"
_MODES = frozenset({"off", "on"})
MAX_VARIABLES = 256
MAX_ENV_BYTES = 64 << 10
STATE_DIR = "shell-state"

# $1 start dir, $2 output prefix, $3 command. The trap records state however the command exits.
WRAPPER = (
    'cd -- "$1" || exit 125\n'
    '__sf_out=$2; __sf_cmd=$3; shift 3\n'
    "trap '__sf_rc=$?; pwd -P > \"$__sf_out.cwd\" 2>/dev/null; env -0 > \"$__sf_out.env\" 2>/dev/null; "
    "exit $__sf_rc' EXIT\n"
    'eval "$__sf_cmd"\n'
)


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in _MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


@dataclass(frozen=True)
class ShellState:
    cwd: str = "."
    set_vars: dict[str, str] = field(default_factory=dict)
    unset_vars: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {"cwd": self.cwd, "set": dict(sorted(self.set_vars.items())),
                "unset": sorted(self.unset_vars)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ShellState":
        cwd, set_vars, unset_vars = value.get("cwd"), value.get("set"), value.get("unset")
        if (not isinstance(cwd, str) or not isinstance(set_vars, dict) or not isinstance(unset_vars, list)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in set_vars.items())
                or not all(isinstance(name, str) for name in unset_vars)):
            raise ValueError("invalid shell state record")
        return cls(cwd, dict(set_vars), tuple(unset_vars))

    def environment(self, base: Mapping[str, str]) -> dict[str, str]:
        result = {key: value for key, value in base.items() if key not in self.unset_vars}
        result.update(self.set_vars)
        return result

    def directory(self, root: Path) -> Path:
        return root if self.cwd == "." else root / self.cwd


def _parse_env(raw: bytes) -> dict[str, str] | None:
    result: dict[str, str] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        name, separator, value = entry.partition(b"=")
        if not separator:
            return None
        try:
            result[name.decode()] = value.decode()
        except UnicodeDecodeError:
            return None
    return result


# Variables the shell itself maintains: the wrapper sets the directory, so these are never carried.
_SHELL_MANAGED = frozenset({"PWD", "OLDPWD", "SHLVL", "_"})


def _admissible(name: str) -> bool:
    return (bool(name) and name not in _SHELL_MANAGED and name.replace("_", "a").isalnum()
            and not name[0].isdigit() and not name.startswith("STATEFRAME_")
            and not name.startswith("__sf_"))


def capture(prefix: Path, root: Path, base: Mapping[str, str], previous: ShellState) -> tuple[ShellState, str | None]:
    """Read the wrapper's output. Returns the next state and a note when something was not kept."""
    try:
        cwd_text = (prefix.parent / (prefix.name + ".cwd")).read_text(encoding="utf-8").strip()
        env_raw = (prefix.parent / (prefix.name + ".env")).read_bytes()
    except (OSError, UnicodeDecodeError):
        return previous, "shell state not recorded (command did not exit normally); kept the previous state"
    finally:
        for suffix in (".cwd", ".env"):
            try:
                (prefix.parent / (prefix.name + suffix)).unlink()
            except FileNotFoundError:
                pass
    notes = []
    cwd = previous.cwd
    try:
        relative = Path(cwd_text).resolve().relative_to(root)
        cwd = relative.as_posix() or "."
    except (OSError, ValueError):
        notes.append("directory outside the workspace was not kept")
    environment = _parse_env(env_raw)
    if environment is None:
        notes.append("environment could not be read; kept the previous variables")
        return ShellState(cwd, previous.set_vars, previous.unset_vars), "; ".join(notes)
    set_vars = {name: value for name, value in environment.items()
                if base.get(name) != value and _admissible(name)}
    unset_vars = tuple(sorted(name for name in base if name not in environment and _admissible(name)))
    if (len(set_vars) > MAX_VARIABLES
            or sum(len(k) + len(v) for k, v in set_vars.items()) > MAX_ENV_BYTES):
        notes.append(f"exported variables exceed {MAX_VARIABLES} names or {MAX_ENV_BYTES} bytes; kept the previous variables")
        set_vars, unset_vars = previous.set_vars, previous.unset_vars
    return ShellState(cwd, set_vars, unset_vars), ("; ".join(notes) or None)


def store(run_dir: Path, call_id: str, state: ShellState) -> None:
    directory = run_dir / STATE_DIR
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / f"{_safe(call_id)}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state.as_dict(), sort_keys=True))
    os.replace(temporary, path)


def load(run_dir: Path, call_id: str) -> ShellState | None:
    path = run_dir / STATE_DIR / f"{_safe(call_id)}.json"
    try:
        return ShellState.from_dict(json.loads(path.read_text()))
    except FileNotFoundError:
        return None


def _safe(call_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in call_id)[:200]
