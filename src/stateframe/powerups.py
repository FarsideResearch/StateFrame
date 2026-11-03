"""Resolve trusted powerups into existing skill, command, hook, and MCP inputs."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

from .powerup_attachment import read_attachments, shell_attachments
from .powerup_management import disable, enable, install, promote, remove, trust
from .powerup_manifest import (
    MODES,
    Manifest,
    PowerupError,
    parse_manifest,
    TREE_DIGEST_VERSION,
    tree_digest,
)
from .powerup_materialize import BUILTINS as BUILTINS
from .powerup_materialize import _materialize, _read_json, _validate
from .powerup_offer import _offer_matches
from .powerup_state import (
    effective_state,
    home,
    project_state,
    trusted_digest,
    user_state,
)
from .skills import NAME, SkillError

__all__ = [
    "PowerupError",
    "parse_manifest",
    "resolve_units",
    "attach_units",
    "register_executor",
    "verify_records",
    "install",
    "remove",
    "enable",
    "disable",
    "trust",
    "promote",
    "effective_state",
]


@dataclass(frozen=True)
class Unit:
    name: str
    source: Path
    source_sha256: str
    mode: str
    settings: dict
    materialized_dir: Path
    materialized_sha256: str
    parts: tuple[str, ...]
    attachments: tuple[dict, ...] = ()

    def record(self) -> dict:
        return {
            "name": self.name,
            "source": str(self.source),
            "tree_digest_version": TREE_DIGEST_VERSION,
            "tree_sha256": self.source_sha256,
            "mode": self.mode,
            "settings": self.settings,
            "materialized_dir": str(self.materialized_dir),
            "rendered_dir_sha256": self.materialized_sha256,
            "parts": list(self.parts),
            "attachments": list(self.attachments),
        }


@dataclass
class Selection:
    active: list[Unit] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def record(self) -> dict:
        return {
            "active": [unit.record() for unit in self.active],
            "skipped": self.skipped,
        }


def _parts(manifest: Manifest, mode: str) -> tuple[str, ...]:
    if mode == "offer":
        return ("skill",) if (manifest.path / "SKILL.md").is_file() else ()
    return tuple(
        key
        for key, filename in (
            ("skill", "SKILL.md"),
            ("commands", "commands.json"),
            ("hooks", "hooks.json"),
            ("mcp", "mcp.json"),
        )
        if manifest.activate.get(key, True) and (manifest.path / filename).is_file()
    )


def _requested(value: str) -> tuple[str, str | None]:
    name, separator, mode = value.rpartition("=")
    if separator and mode in MODES:
        return name, mode
    return value, None


def resolve_units(
    repo: Path,
    task: str,
    profile: str,
    requests: Sequence[str],
    sets: Sequence[str],
    no_powerups: bool,
    *,
    loop: str = "composed",
    shell_network: str = "loopback",
    user_tools: Sequence[Path] = (),
    user_mcp: Sequence[Path] = (),
    user_hooks: Sequence[Path] = (),
    user_skills: int = 0,
    printer=print,
) -> Selection:
    selected = Selection()
    if no_powerups:
        return selected
    root = Path(repo).resolve()
    cli_modes: dict[str, str] = {}
    explicit: dict[str, Path] = {}
    for request in requests:
        name, mode = _requested(request)
        path = Path(name).expanduser()
        if path.is_dir() or os.sep in name:
            explicit[path.name] = path.absolute()
            cli_modes[path.name] = mode or "on"
        else:
            cli_modes[name] = mode or "on"
    cli_settings: dict[str, dict[str, str]] = {}
    for value in sets:
        key, separator, setting = value.partition("=")
        name, dot, field = key.partition(".")
        if not separator or not dot or not name or not field:
            raise PowerupError(f"invalid --powerup-set: {value}")
        cli_settings.setdefault(name, {})[field] = setting
    empty = {"global": {}, "profiles": {}}
    try:
        user = user_state()
    except (PowerupError, OSError, ValueError) as error:
        user = empty
        selected.skipped.append({"name": "state", "code": "state_invalid", "reason": str(error)})
        printer(f"powerup state disabled: {error}")
    try:
        project = project_state(root)
    except (PowerupError, OSError, ValueError) as error:
        project = empty
        selected.skipped.append({"name": "project-state", "code": "state_invalid", "reason": str(error)})
        printer(f"powerup project state disabled: {error}")
    if any(item.get("code") == "state_invalid" for item in selected.skipped):
        # Fail closed: an unreadable state file may hold the user's "off", so attach nothing.
        printer("powerups disabled for this run: fix or remove the invalid state file")
        return selected
    project_root = root / ".stateframe" / "powerups"
    user_root = home() / "powerups"
    names = (
        set(cli_modes)
        | set(cli_settings)
        | set(user["global"])
        | set(user["profiles"].get(profile, {}))
        | set(project["global"])
    )
    for directory in (project_root, user_root):
        try:
            if directory.is_dir() and not directory.is_symlink():
                names.update(
                    path.name
                    for path in directory.iterdir()
                    if path.is_dir() and not path.is_symlink() and NAME.fullmatch(path.name)
                )
        except OSError as error:
            selected.skipped.append({"name": str(directory), "code": "discovery_invalid",
                                     "reason": str(error)})
            printer(f"powerup directory disabled: {error}")
    for name in sorted(names):
        source = explicit.get(name)
        source_kind = "explicit" if source else ""
        project_path = project_root / name
        project_rejected = False
        project_error = None
        if source is None and project_path.is_dir() and not project_path.is_symlink():
            try:
                pinned = trusted_digest(root, project_path)
                unchanged = pinned is not None and tree_digest(project_path) == pinned
                if pinned is not None and not unchanged:
                    project_error = (
                        "re-trust required: project powerup changed or pinned with an older "
                        f"digest format; run 'stateframe powerups trust --repo {root}' after review"
                    )
            except (PowerupError, OSError) as error:
                unchanged = False
                project_error = str(error)
            if unchanged:
                source, source_kind = project_path, "project"
            else:
                project_rejected = True
        user_path = user_root / name
        if source is None and user_path.is_dir() and not user_path.is_symlink():
            source, source_kind = user_path, "user"
        if source_kind == "explicit" and (project_path.is_dir() or user_path.is_dir()):
            printer(
                f"powerup {name}: explicit path takes precedence over discovered copies"
            )
        if source_kind == "project" and user_path.is_dir():
            printer(
                f"powerup {name}: trusted project copy takes precedence over installed copy"
            )
        if source_kind == "user" and project_rejected:
            printer(
                f"powerup {name}: project copy untrusted or changed; using installed copy"
            )
        if source is None:
            if name in cli_modes or name in cli_settings or project_path.is_dir():
                reason = (
                    project_error or "project powerup is untrusted or changed"
                    if project_path.is_dir()
                    else "not found"
                )
                selected.skipped.append({"name": name, "reason": reason})
                printer(f"powerup {name} disabled: {reason}")
            continue
        try:
            manifest = parse_manifest(source)
            state = effective_state(
                manifest,
                profile,
                project=project,
                cli=cli_modes,
                cli_settings=cli_settings.get(name),
                installed=source_kind == "user",
                project_source=source_kind == "project",
                user=user,
            )
            mode = state["mode"]
            if source_kind == "explicit":
                state["digest"] = None
            if mode == "off":
                continue
            source_sha = tree_digest(source)
            if source_kind == "user" and name not in cli_modes and not state["digest"]:
                raise PowerupError(
                    "not enabled; run 'stateframe powerups enable NAME' after review"
                )
            current_pins = state["digest_version"] == TREE_DIGEST_VERSION
            if state["digest"] and not current_pins:
                raise PowerupError(
                    "re-trust required: pinned with an older digest format; "
                    f"run 'stateframe powerups enable {name}' after review"
                )
            if state["digest"] and state["digest"] != source_sha:
                raise PowerupError(
                    "changed since enabled; run 'stateframe powerups enable NAME' after review"
                )
            if loop != "composed":
                raise PowerupError("requires the composed loop")
            if mode == "auto":
                if not _offer_matches(manifest, root, task, loop):
                    continue
                mode = "on"
            parts = _parts(manifest, mode)
            if (
                "skill" in parts
                and user_skills + sum("skill" in unit.parts for unit in selected.active)
                >= 16
            ):
                raise PowerupError("maximum 16 selected skills")
            gating_trusted = current_pins and state["trust_gating"] == source_sha
            materialized, materialized_sha = _materialize(
                manifest, state["settings"], shell_network=shell_network,
                repo=root, trust_gating=gating_trusted,
            )
            _validate(
                manifest,
                materialized,
                parts,
                repo=root,
                shell_network=shell_network,
                earlier_tools=user_tools,
                earlier_mcp=user_mcp,
                earlier_hooks=user_hooks,
            )
            if tree_digest(source) != source_sha or tree_digest(materialized) != materialized_sha:
                raise PowerupError("powerup changed between validation and attachment")
            if "commands" in parts:
                count = len(_read_json(materialized / "commands.json").get("tools", []))
                if count > 6:
                    raise PowerupError("maximum 6 powerup tools")
            if "hooks" in parts and source_kind == "user":
                source_hooks = _read_json(source / "hooks.json").get("hooks", [])
                rendered_hooks = _read_json(materialized / "hooks.json").get("hooks", [])
                if type(source_hooks) is list and type(rendered_hooks) is list:
                    attached = {(item.get("name"), item.get("event")) for item in rendered_hooks
                                if type(item) is dict}
                    for item in source_hooks:
                        if type(item) is dict and (item.get("name"), item.get("event")) not in attached:
                            gating = item.get("event") in {"prompt_submit", "pre_tool", "stop"}
                            selected.skipped.append({"name": name,
                                "code": "gating_untrusted" if gating else "event_unavailable",
                                "reason": (f"hook {item.get('name', '?')} requires --trust-gating"
                                           if gating else f"hook {item.get('name', '?')} event unavailable")})
            attachments = (() if mode == "offer" else (
                *(item for item in read_attachments(materialized)
                  if item["executor"] != "shell"),
                *shell_attachments(materialized, parts, materialized_sha)))
            selected.active.append(
                Unit(
                    name,
                    source.resolve(),
                    source_sha,
                    mode,
                    state["settings"],
                    materialized,
                    materialized_sha,
                    parts,
                    attachments,
                )
            )
        except (PowerupError, SkillError, OSError, ValueError, shutil.Error) as error:
            reason = str(error)
            selected.skipped.append({"name": name, "reason": reason})
            printer(f"powerup {name} disabled: {reason}")
    return selected


_EXECUTORS: dict[str, Callable[[object, Unit], None]] = {}


def register_executor(name: str, handler: Callable[[object, Unit], None]) -> None:
    """Register a host adapter for an isolated unit executor before a run starts."""
    if name != "mutation" or not callable(handler):
        raise PowerupError("unsupported isolated unit executor registration")
    _EXECUTORS[name] = handler


def filter_unavailable_executors(selection: Selection, *, printer=print) -> None:
    kept = []
    for unit in selection.active:
        missing = next((item["executor"] for item in unit.attachments
                        if item["executor"] not in {"shell", *_EXECUTORS}), None)
        if missing is None or unit.mode == "offer":
            kept.append(unit)
            continue
        selection.skipped.append({"name": unit.name, "code": "executor_unavailable",
                                  "reason": f"executor {missing} unavailable"})
        printer(f"powerup {unit.name} disabled: executor {missing} unavailable")
    selection.active = kept


def attach_units(options, units: Selection | Sequence[Unit], *, printer=print) -> None:
    """Append selected units after all explicitly supplied inputs in stable name order."""
    selection = units if isinstance(units, Selection) else None
    if selection is not None:
        filter_unavailable_executors(selection, printer=printer)
    candidates = units.active if selection is not None else units
    if not candidates:
        return
    skills = list(options.skills)
    hooks = list(options.hooks_files)
    settings = options.tool_settings
    tools = list(settings.tools_files)
    mcp = list(settings.mcp_files)
    kept = []
    mutation_adapters = []
    for unit in sorted(candidates, key=lambda item: item.name):
        if unit.mode != "offer" and any(
            item["executor"] == "mutation" for item in unit.attachments
        ):
            adapter = _EXECUTORS.get("mutation")
            if adapter is None:
                printer(f"powerup {unit.name} disabled: executor mutation unavailable")
                continue
            mutation_adapters.append((adapter, unit))
        if "skill" in unit.parts:
            skills.append(str(unit.materialized_dir))
        if "commands" in unit.parts:
            tools.append(
                {
                    "path": str(unit.materialized_dir / "commands.json"),
                    "digest": "pending",
                }
            )
        if "hooks" in unit.parts:
            hooks.append(unit.materialized_dir / "hooks.json")
        if "mcp" in unit.parts:
            mcp.append(
                {"path": str(unit.materialized_dir / "mcp.json"), "digest": "pending"}
            )
        kept.append(unit)
    if selection is not None:
        selection.active = kept
    if not kept:
        return
    options.skills = tuple(skills)
    options.hooks_files = tuple(hooks)
    options.tool_settings = replace(
        settings, tools_files=tuple(tools), mcp_files=tuple(mcp)
    )
    for adapter, unit in mutation_adapters:
        adapter(options, unit)


def verify_records(value: object) -> None:
    if (
        type(value) is not dict
        or set(value) != {"active", "skipped"}
        or type(value["active"]) is not list
    ):
        raise PowerupError("POWERUP_CHANGED: invalid powerup spec")
    for record in value["active"]:
        if type(record) is dict and record.get("tree_digest_version") != TREE_DIGEST_VERSION:
            raise PowerupError(
                f"POWERUP_CHANGED: {record.get('name', 'unit')}: pinned with an older "
                "digest format; re-trust required"
            )
        required = {
            "tree_digest_version",
            "name",
            "source",
            "tree_sha256",
            "mode",
            "settings",
            "materialized_dir",
            "rendered_dir_sha256",
            "parts",
        }
        if type(record) is not dict or not required.issubset(record) or set(record) - (required | {"attachments"}):
            raise PowerupError("POWERUP_CHANGED: invalid pinned unit")
        if (type(record["name"]) is not str or not NAME.fullmatch(record["name"])
                or type(record["materialized_dir"]) is not str
                or type(record["rendered_dir_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", record["rendered_dir_sha256"]) is None
                or type(record["mode"]) is not str or record["mode"] not in {"on", "offer"}
                or type(record["parts"]) is not list
                or any(type(part) is not str or part not in {"skill", "commands", "hooks", "mcp"}
                       for part in record["parts"])
                or type(record["settings"]) is not dict):
            raise PowerupError("POWERUP_CHANGED: invalid pinned unit fields")
        directory = Path(record["materialized_dir"])
        try:
            current = tree_digest(directory)
        except (PowerupError, OSError, ValueError) as error:
            raise PowerupError(f"POWERUP_CHANGED: {record['name']}: {error}") from error
        if current != record["rendered_dir_sha256"]:
            raise PowerupError(
                f"POWERUP_CHANGED: {record['name']}: materialized unit changed"
            )
        if "attachments" in record:
            try:
                expected = (() if record["mode"] == "offer" else (
                    *(item for item in read_attachments(directory)
                      if item["executor"] != "shell"),
                    *shell_attachments(directory, tuple(record["parts"]), current)))
            except (PowerupError, OSError, ValueError, TypeError) as error:
                raise PowerupError(f"POWERUP_CHANGED: {record['name']}: {error}") from error
            if type(record["attachments"]) is not list or record["attachments"] != list(expected):
                raise PowerupError(f"POWERUP_CHANGED: {record['name']}: attachment record changed")
