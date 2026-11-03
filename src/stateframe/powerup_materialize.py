"""Materialize bounded powerup settings and validate ordinary unit parts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Sequence

from . import hook_host
from .engine.kernel_custom_tools import CustomToolError, load_tool_files
from .engine.kernel_hooks import HookConfigError, load_hook_files
from .engine.kernel_mcp_tools import load_mcp_files
from .powerup_attachment import read_attachments, shell_attachments
from .powerup_manifest import Manifest, PowerupError, tree_digest
from .powerup_state import home
from .skills import SkillError, parse_skill
from .state_dirs import private_directory

BUILTINS = frozenset(
    (
        "grep",
        "list_files",
        "read",
        "edit",
        "str_replace",
        "insert_before",
        "authored_check",
        "file_create",
        "run_tests",
        "read_check_receipt",
        "recall",
        "done",
        "shell",
        "job_wait",
        "job_output",
        "job_stop",
        "file_delete",
        "file_move",
        "git",
        "web_fetch",
        "git_commit",
        "git_branch",
        "skill_load",
    )
)
PLACEHOLDER = re.compile(r"^\{setting:([a-z][a-z0-9_]{0,40})\}$")


def _read_json(path: Path) -> dict:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1 << 20:
            raise PowerupError(f"{path}: expected bounded regular JSON file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            opened = os.fstat(source.fileno())
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or
                    (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
                raise PowerupError(f"{path}: JSON file changed while opening")
            raw = source.read((1 << 20) + 1)
        if len(raw) > 1 << 20:
            raise PowerupError(f"{path}: exceeds 1 MiB")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=hook_host._strict_pairs,
            parse_constant=hook_host._reject_constant,
        )
    except PowerupError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise PowerupError(f"{path}: invalid JSON: {error}") from error
    if type(value) is not dict:
        raise PowerupError(f"{path}: expected JSON object")
    return value


def _render_argv(
    value: dict, settings: dict, root: Path, *, commands: bool, final_root: Path
) -> dict:
    key = "tools" if commands else "hooks"
    items = value.get(key, [])
    if type(items) is not list:
        return value  # The existing loader reports the schema error.
    for item in items:
        if type(item) is not dict or type(item.get("argv")) is not list:
            continue  # existing loader reports the exact schema error
        rendered = []
        for argument in item["argv"]:
            if type(argument) is not str:
                rendered.append(argument)
                continue
            if commands and ".." in Path(argument).parts:
                raise PowerupError("command argv may not traverse outside the unit")
            match = PLACEHOLDER.fullmatch(argument)
            if match:
                name = match.group(1)
                if name not in settings:
                    raise PowerupError(f"unknown argv setting {name}")
                current = settings[name]
                rendered.append(
                    str(current).lower() if type(current) is bool else str(current)
                )
            elif "{setting:" in argument:
                raise PowerupError(
                    "setting placeholder must occupy a whole argv argument"
                )
            elif (
                commands
                and argument not in {"{paths}"}
                and not Path(argument).is_absolute()
                and (root / argument).is_file()
            ):
                rendered.append(str(final_root / argument))
            else:
                rendered.append(argument)
        item["argv"] = rendered
        if not commands and item.get("event") in {"prompt_submit", "pre_tool", "stop"}:
            item.setdefault("on_error", "continue")
    return value


def _materialize(
    manifest: Manifest,
    settings: dict,
    *,
    shell_network: str,
    repo: Path,
    cache_root: Path | None = None,
    trust_gating: bool = False,
    filter_generated_hooks: bool = True,
) -> tuple[Path, str]:
    source_sha = tree_digest(manifest.path)
    key = hashlib.sha256(
        json.dumps([source_sha, settings, trust_gating, filter_generated_hooks], sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()
    cache = cache_root or home() / "powerup-cache"
    private_directory(cache)
    target = cache / key / manifest.name
    temporary = Path(tempfile.mkdtemp(prefix=".powerup-", dir=cache))
    try:
        copied = temporary / manifest.name
        shutil.copytree(manifest.path, copied, symlinks=True)
        if tree_digest(copied) != source_sha:
            # The copy, not the source, is what gets used: it must match the pin exactly.
            raise PowerupError(f"{manifest.name}: copied tree does not match its pin")
        if tree_digest(manifest.path) != source_sha:
            raise PowerupError(
                f"{manifest.name}: source changed during materialization"
            )
        for filename, commands in (
            ("commands.json", True),
            ("hooks.json", False),
            ("mcp.json", False),
        ):
            path = copied / filename
            if not path.is_file():
                continue
            value = _read_json(path)
            if filename == "hooks.json" and filter_generated_hooks:
                metadata = copied / "install.json"
                provenance = _read_json(metadata).get("provenance") if metadata.is_file() else None
                if type(provenance) is str and provenance.startswith("promoted from mutation "):
                    hooks = value.get("hooks")
                    if type(hooks) is list:
                        allowed = {"post_tool", "session_end"}
                        if trust_gating:
                            allowed.update({"prompt_submit", "pre_tool", "stop"})
                        value["hooks"] = [item for item in hooks if type(item) is not dict
                                          or item.get("event") in allowed]
            if filename == "mcp.json":
                servers = value.get("servers", [])
                if type(servers) is not list:
                    servers = []  # The MCP loader reports the schema error.
                for item in servers:
                    if type(item) is dict and type(item.get("argv")) is list:
                        if any(
                            type(arg) is str and ".." in Path(arg).parts
                            for arg in item["argv"]
                        ):
                            raise PowerupError(
                                "MCP argv may not traverse outside the unit"
                            )
                        item["argv"] = [
                            str(target / arg)
                            if type(arg) is str
                            and not Path(arg).is_absolute()
                            and (copied / arg).is_file()
                            else arg
                            for arg in item["argv"]
                        ]
            else:
                value = _render_argv(
                    value, settings, copied, commands=commands, final_root=target
                )
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        expected = tree_digest(copied)
        directory = cache / key
        if directory.exists() or directory.is_symlink():
            if (
                directory.is_symlink()
                or target.is_symlink()
                or tree_digest(target) != expected
            ):
                raise PowerupError(f"{target}: materialized cache changed")
        else:
            try:
                temporary.rename(directory)
            except FileExistsError:
                if directory.is_symlink() or tree_digest(target) != expected:
                    raise PowerupError(f"{target}: materialized cache changed")
        return target, expected
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _validate(
    manifest: Manifest,
    materialized: Path,
    parts: tuple[str, ...],
    *,
    repo: Path,
    shell_network: str,
    earlier_tools: Sequence[Path] = (),
    earlier_mcp: Sequence[Path] = (),
    earlier_hooks: Sequence[Path] = (),
) -> None:
    if (manifest.path / "tools.json").exists():
        raise PowerupError("tools.json is reserved for skills; use commands.json")
    if (materialized / "SKILL.md").is_file():
        try:
            skill = parse_skill(materialized)
        except SkillError as error:
            raise PowerupError(str(error)) from error
        if skill.description != manifest.summary:
            raise PowerupError("SKILL.md description must match manifest summary")
    attachments = read_attachments(materialized)
    shell_parts = tuple(part for part, filename in (
        ("commands", "commands.json"), ("hooks", "hooks.json"),
        ("mcp", "mcp.json")) if (materialized / filename).is_file())
    actual_shell = shell_attachments(materialized, shell_parts, tree_digest(materialized))
    for declared in attachments:
        if declared["executor"] != "shell":
            continue
        if not any(actual["binding"] == declared["binding"]
                   and actual["capability_ceiling"] == declared["capability_ceiling"]
                   and actual["replay_policy"] == declared["replay_policy"]
                   for actual in actual_shell):
            raise PowerupError("isolated shell attachment binding or capability mismatch")
    for filename, section in (("commands.json", "tools"), ("mcp.json", "servers")):
        path = materialized / filename
        if path.is_file():
            items = _read_json(path).get(section, [])
            for item in items if type(items) is list else ():
                if (
                    type(item) is dict
                    and item.get("network", "none") != "none"
                    and item.get("replay", "external") != "external"
                ):
                    raise PowerupError(
                        "networked powerup units require external replay"
                    )
    hooks_path = materialized / "hooks.json"
    if hooks_path.is_file():
        hooks = _read_json(hooks_path).get("hooks", [])
        for hook in hooks if type(hooks) is list else ():
            if type(hook) is not dict or hook.get("network", "none") == "none":
                continue
            if not any(
                item["binding"]["kind"] == "hook"
                and item["binding"]["name"] == hook.get("name")
                and item["binding"].get("event") == hook.get("event")
                and item["replay_policy"] == "external"
                and item["capability_ceiling"]["network"] == "public-web"
                for item in attachments
            ):
                raise PowerupError(
                    "networked powerup hook requires an external replay attachment"
                )
    try:
        if (materialized / "commands.json").is_file():
            load_tool_files(
                [
                    *(earlier_tools if "commands" in parts else ()),
                    materialized / "commands.json",
                ],
                builtin_names=BUILTINS,
                shell_network=shell_network,
            )
        if (materialized / "hooks.json").is_file():
            load_hook_files(
                (
                    *(earlier_hooks if "hooks" in parts else ()),
                    materialized / "hooks.json",
                ),
                repo=repo,
                shell_network=shell_network,
            )
        if (materialized / "mcp.json").is_file():
            load_mcp_files(
                [*(earlier_mcp if "mcp" in parts else ()), materialized / "mcp.json"],
                shell_network=shell_network,
            )
    except (CustomToolError, HookConfigError) as error:
        raise PowerupError(str(error)) from error
