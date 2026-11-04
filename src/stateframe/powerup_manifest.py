"""Strict, bounded powerup manifests and source tree fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .engine.kernel_custom_tools import CustomToolError, _pattern
from .engine.kernel_safe_paths import safe_expanduser
from .skills import MAX_DIRECTORY_BYTES, NAME


class PowerupError(ValueError):
    """A powerup definition, setting, or trusted source is invalid."""


MODES = frozenset({"off", "offer", "on", "auto"})
SETTING_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
SETTING_TYPES = frozenset({"string", "integer", "number", "boolean", "enum"})


@dataclass(frozen=True)
class Manifest:
    path: Path
    name: str
    version: str
    summary: str
    default: str
    offer: dict
    activate: dict
    settings: dict


def _keys(value: object, allowed: set[str], where: str) -> dict:
    if type(value) is not dict:
        raise PowerupError(f"{where}: expected table")
    extra = set(value) - allowed
    if extra:
        raise PowerupError(f"{where}: unknown key {sorted(extra)[0]}")
    return value


def _string_list(value: object, where: str, *, maximum: int = 16) -> list[str]:
    if (
        type(value) is not list
        or len(value) > maximum
        or any(
            type(item) is not str or not item or len(item) > 256 or "\0" in item
            for item in value
        )
    ):
        raise PowerupError(f"{where}: expected up to {maximum} bounded strings")
    return value


TREE_DIGEST_FORMAT = "stateframe-powerup-tree/2"
TREE_DIGEST_VERSION = 2


def tree_digest(path: Path) -> str:
    """Hash paths, modes, and bytes, while refusing links and oversized trees.

    Version 2 hashes one canonical JSON document: the format tag and a sorted list of
    [type, path, mode, sha256(content)] records. JSON quoting delimits every field, so
    no two trees share an encoding. Pins made with the old concatenated encoding
    carry no digest version and are refused until the user enables again.
    """
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise PowerupError(f"powerup directory missing or linked: {root}")
    entries: list[list] = []
    total = 0
    count = 0
    def unreadable(error: OSError) -> None:
        raise PowerupError(f"{root}: unreadable entry: {error}")

    for current, directories, files in os.walk(root, onerror=unreadable, followlinks=False):
        for name in sorted([*directories, *files]):
            item = Path(current) / name
            relative = item.relative_to(root).as_posix()
            info = item.lstat()
            count += 1
            if count > 2000:
                raise PowerupError(f"{root}: too many entries")
            if stat.S_ISDIR(info.st_mode):
                entries.append(["dir", relative, 0, ""])
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                remaining = MAX_DIRECTORY_BYTES - total
                if info.st_size > remaining:
                    raise PowerupError(f"{root}: directory exceeds 1 MiB")
                descriptor = os.open(item, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as source:
                    opened = os.fstat(source.fileno())
                    if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                            or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
                        raise PowerupError(f"{item}: changed during hashing")
                    content = source.read(remaining + 1)
                    finished = os.fstat(source.fileno())
                if (len(content) != info.st_size or finished.st_size != opened.st_size
                        or finished.st_mtime_ns != opened.st_mtime_ns):
                    raise PowerupError(f"{item}: changed during hashing")
                total += len(content)
                entries.append(["file", relative, stat.S_IMODE(info.st_mode),
                                hashlib.sha256(content).hexdigest()])
            else:
                raise PowerupError(
                    f"{item}: symlinks, hard links, and special files are not allowed"
                )
    document = {"format": TREE_DIGEST_FORMAT, "entries": sorted(entries)}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _setting(spec: object, where: str) -> dict:
    item = _keys(spec, {"type", "default", "min", "max", "max_length", "values"}, where)
    kind = item.get("type")
    if type(kind) is not str or kind not in SETTING_TYPES or "default" not in item:
        raise PowerupError(f"{where}: type and default are required")
    if kind in {"integer", "number"}:
        if set(item) - {"type", "default", "min", "max"}:
            raise PowerupError(f"{where}: invalid numeric constraint")
        if "min" not in item or "max" not in item:
            raise PowerupError(f"{where}: numeric bounds are required")
        if kind == "integer":

            def valid(v):
                return type(v) is int
        else:

            def valid(v):
                return type(v) in (int, float) and math.isfinite(v)

        if (
            not all(valid(item[key]) for key in ("default", "min", "max"))
            or not item["min"] <= item["default"] <= item["max"]
        ):
            raise PowerupError(f"{where}: invalid numeric bounds or default")
    elif kind == "string":
        if (
            set(item) - {"type", "default", "max_length"}
            or type(item.get("max_length")) is not int
            or not 1 <= item["max_length"] <= 1000
        ):
            raise PowerupError(f"{where}: max_length must be 1..1000")
        if (
            type(item["default"]) is not str
            or len(item["default"]) > item["max_length"]
            or any(c in item["default"] for c in "\0{}")
        ):
            raise PowerupError(f"{where}: invalid string default")
    elif kind == "boolean":
        if set(item) != {"type", "default"} or type(item["default"]) is not bool:
            raise PowerupError(f"{where}: invalid boolean default")
    else:
        values = item.get("values")
        invalid_values = (
            not isinstance(values, list)
            or not 1 <= len(values) <= 20
            or any(
                type(v) is not str
                or not v
                or len(v) > 100
                or any(c in v for c in "\0{}")
                for v in values
            )
        )
        if (
            set(item) != {"type", "default", "values"}
            or invalid_values
            or len(set(values)) != len(values)
            or item["default"] not in values
        ):
            raise PowerupError(f"{where}: invalid enum values or default")
    return item


def parse_manifest(path: Path) -> Manifest:
    root = safe_expanduser(Path(path))
    tree_digest(root)
    source = root / "powerup.toml"
    try:
        raw = source.read_bytes()
        if len(raw) > 64 * 1024:
            raise PowerupError("powerup.toml exceeds 64 KiB")
        value = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise PowerupError(f"{source}: invalid manifest: {error}") from error
    item = _keys(
        value,
        {
            "schema",
            "name",
            "version",
            "summary",
            "default",
            "offer",
            "activate",
            "settings",
        },
        str(source),
    )
    if item.get("schema") != "stateframe-powerup/1":
        raise PowerupError(f"{source}: expected stateframe-powerup/1")
    name = item.get("name")
    if type(name) is not str or not NAME.fullmatch(name) or name != root.name:
        raise PowerupError(f"{source}: name must match directory")
    if (
        type(item.get("version")) is not str
        or not 1 <= len(item["version"]) <= 64
        or any(c in item["version"] for c in "\0\r\n")
    ):
        raise PowerupError(f"{source}: version is required")
    if (
        type(item.get("summary")) is not str
        or not 1 <= len(item["summary"]) <= 120
        or any(c in item["summary"] for c in "\0\r\n")
    ):
        raise PowerupError(
            f"{source}: summary must be one line of at most 120 characters"
        )
    if type(item.get("default")) is not str or item["default"] not in MODES:
        raise PowerupError(f"{source}: invalid default mode")
    offer = _keys(
        item.get("offer", {}),
        {"any_files", "all_of", "task_matches", "loops"},
        f"{source}:offer",
    )
    for key in ("any_files", "all_of", "task_matches"):
        _string_list(offer.get(key, []), f"{source}:offer.{key}")
    loops = _string_list(
        offer.get("loops", ["composed"]), f"{source}:offer.loops", maximum=3
    )
    if not loops or set(loops) - {"composed", "plain", "lean"}:
        raise PowerupError(f"{source}: invalid offer loops")
    for pattern in offer.get("task_matches", []):
        expression = pattern.removeprefix("(?i)")
        if len(pattern) > 128:
            raise PowerupError(f"{source}: unsafe task_matches pattern")
        try:
            _pattern(expression, f"{source}:offer.task_matches")
            re.compile(pattern)
        except (re.error, CustomToolError) as error:
            raise PowerupError(f"{source}: invalid task_matches: {error}") from error
    activate = _keys(
        item.get("activate", {}),
        {"skill", "commands", "hooks", "mcp"},
        f"{source}:activate",
    )
    if any(type(v) is not bool for v in activate.values()):
        raise PowerupError(f"{source}: activate values must be booleans")
    settings = item.get("settings", {})
    if type(settings) is not dict:
        raise PowerupError(f"{source}:settings: expected table")
    if len(settings) > 32:
        raise PowerupError(f"{source}: too many settings")
    for key, spec in settings.items():
        if not SETTING_NAME.fullmatch(key):
            raise PowerupError(f"{source}: invalid setting name {key}")
        _setting(spec, f"{source}:settings.{key}")
    return Manifest(
        root.resolve(),
        name,
        item["version"],
        item["summary"],
        item["default"],
        offer,
        activate,
        settings,
    )


def setting_value(spec: dict, value: object, where: str) -> object:
    kind = spec["type"]
    if kind == "boolean":
        if type(value) is str:
            value = {"true": True, "false": False}.get(value.lower(), value)
        valid = type(value) is bool
    elif kind == "integer":
        if type(value) is str:
            try:
                value = int(value)
            except ValueError:
                pass
        valid = type(value) is int and spec["min"] <= value <= spec["max"]
    elif kind == "number":
        if type(value) is str:
            try:
                value = float(value)
            except ValueError:
                pass
        valid = (
            type(value) in (int, float)
            and math.isfinite(value)
            and spec["min"] <= value <= spec["max"]
        )
    elif kind == "string":
        valid = (
            type(value) is str
            and len(value) <= spec["max_length"]
            and not any(c in value for c in "\0{}")
        )
    else:
        valid = (
            type(value) is str
            and value in spec["values"]
            and not any(c in value for c in "\0{}")
        )
    if not valid:
        raise PowerupError(f"{where}: invalid {kind} value or out of bounds")
    return value
