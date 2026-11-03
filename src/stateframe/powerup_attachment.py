"""Validate the shared isolated-unit attachment contract without executing it."""

from __future__ import annotations

import json
import hashlib
import os
import re
import stat
from pathlib import Path

from . import hook_host
from .powerup_manifest import PowerupError

SCHEMA = "stateframe-isolated-unit-attachment/1"
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
BINDING_NAME = re.compile(r"[a-z][a-z0-9_]{1,40}\Z")


def read_attachments(directory: Path) -> tuple[dict, ...]:
    path = directory / "units.json"
    if not path.exists() and not path.is_symlink():
        return ()
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1 << 20:
            raise PowerupError("units.json must be a bounded regular file")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as source:
            opened = os.fstat(source.fileno())
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise PowerupError("units.json changed while opening")
            raw = source.read((1 << 20) + 1)
        if len(raw) > 1 << 20:
            raise PowerupError("units.json exceeds 1 MiB")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=hook_host._strict_pairs,
            parse_constant=hook_host._reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise PowerupError(f"invalid units.json: {error}") from error
    entries = (
        value["units"] if type(value) is dict and set(value) == {"units"} else value
    )
    if type(entries) is not list or len(entries) > 6:
        raise PowerupError("units.json requires a bounded units list")
    for item in entries:
        required = {
            "schema",
            "unit_digest",
            "executor",
            "capability_ceiling_digest",
            "capability_ceiling",
            "replay_policy",
            "binding",
        }
        if type(item) is not dict or set(item) != required or item["schema"] != SCHEMA:
            raise PowerupError("invalid isolated unit attachment schema")
        if any(
            type(item[key]) is not str or not SHA.fullmatch(item[key])
            for key in ("unit_digest", "capability_ceiling_digest")
        ):
            raise PowerupError("invalid isolated unit attachment digest")
        if type(item["executor"]) is not str or item["executor"] not in {
            "shell",
            "mutation",
        }:
            raise PowerupError("invalid isolated unit executor")
        ceiling = item["capability_ceiling"]
        if (
            type(ceiling) is not dict
            or set(ceiling) - {"network", "writes", "read_globs", "domains"}
            or type(ceiling.get("network")) is not str
            or ceiling.get("network") not in {"none", "public-web"}
            or any(
                type(ceiling.get(key, [])) is not list
                or len(ceiling.get(key, [])) > 32
                or any(
                    type(entry) is not str or len(entry) > 256
                    for entry in ceiling.get(key, [])
                )
                for key in ("writes", "read_globs", "domains")
            )
        ):
            raise PowerupError("invalid isolated unit capability ceiling")
        actual = hashlib.sha256(json.dumps(ceiling, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if item["capability_ceiling_digest"] != "sha256:" + actual:
            raise PowerupError("isolated unit capability ceiling digest mismatch")
        replay = item["replay_policy"]
        if type(replay) is not str or replay not in {
            "observation",
            "local",
            "external",
        }:
            raise PowerupError("invalid isolated unit replay policy")
        if ceiling["network"] != "none" and replay != "external":
            raise PowerupError("networked isolated units require external replay")
        if item["executor"] == "mutation" and replay == "local":
            raise PowerupError("mutation units cannot use local replay")
        binding = item["binding"]
        if (
            type(binding) is not dict
            or set(binding) - {"kind", "name", "event"}
            or type(binding.get("kind")) is not str
            or binding.get("kind") not in {"command", "hook"}
            or type(binding.get("name")) is not str
            or (item["executor"] == "mutation" and not BINDING_NAME.fullmatch(binding["name"]))
            or (item["executor"] == "shell" and not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", binding["name"]))
            or (
                binding["kind"] == "hook"
                and binding.get("event") not in (
                    {"post_tool", "session_end"} if item["executor"] == "mutation"
                    else {"session_start", "prompt_submit", "pre_tool", "post_tool", "stop", "session_end"}
                )
            )
            or (binding["kind"] == "command" and "event" in binding)
        ):
            raise PowerupError("invalid isolated unit binding")
    return tuple(entries)


def shell_attachments(directory: Path, parts: tuple[str, ...], unit_digest: str) -> tuple[dict, ...]:
    """Describe each admitted shell binding using the shared attachment schema."""
    from .powerup_materialize import _read_json

    records = []
    for part, filename, key, kind in (
        ("commands", "commands.json", "tools", "command"),
        ("hooks", "hooks.json", "hooks", "hook"),
        ("mcp", "mcp.json", "servers", "command"),
    ):
        if part not in parts:
            continue
        items = _read_json(directory / filename).get(key, [])
        if type(items) is not list or any(type(item) is not dict for item in items):
            raise PowerupError(f"{filename}: invalid {key} list")
        for item in items:
            network = item.get("network", "none")
            writes = item.get("writes", "none")
            ceiling = {"network": "none" if network == "none" else "public-web",
                       "writes": [] if writes == "none" else [writes], "domains": []}
            digest = hashlib.sha256(json.dumps(ceiling, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            binding = {"kind": kind, "name": item.get("name")}
            if kind == "hook":
                binding["event"] = item.get("event")
            records.append({"schema": SCHEMA, "unit_digest": "sha256:" + unit_digest,
                "executor": "shell", "capability_ceiling_digest": "sha256:" + digest,
                "capability_ceiling": ceiling,
                "replay_policy": ("external" if network != "none" else
                                  ("local" if writes != "none" else "observation")
                                  if kind == "hook" else item.get("replay", "external")),
                "binding": binding})
    return tuple(records)
