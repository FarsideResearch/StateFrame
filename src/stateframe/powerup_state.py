"""Powerup state and project trust using the hook trust store contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
from pathlib import Path

from . import hook_host
from .engine.kernel_safe_paths import safe_expanduser, safe_home
from .powerup_manifest import (
    MODES, TREE_DIGEST_VERSION, Manifest, PowerupError, setting_value, tree_digest,
)
from .skills import NAME
from .state_dirs import private_directory

SCHEMA = "stateframe-powerup-state/1"


def home() -> Path:
    configured = os.environ.get("STATEFRAME_HOME")
    if configured:
        return safe_expanduser(Path(configured))
    user_home = safe_home()
    if user_home is None:
        raise PowerupError("cannot find the StateFrame home: set HOME or STATEFRAME_HOME")
    return safe_expanduser(user_home / ".stateframe")


def _state_bytes(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1 << 20:
        raise PowerupError(f"{path}: expected bounded regular state file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or
                (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)):
            raise PowerupError(f"{path}: state file changed while opening")
        raw = source.read((1 << 20) + 1)
    if len(raw) > 1 << 20:
        raise PowerupError(f"{path}: state file exceeds 1 MiB")
    return raw


def _read(path: Path) -> dict:
    if not path.exists() and not path.is_symlink():
        return {"schema": SCHEMA, "global": {}, "profiles": {}}
    try:
        value = json.loads(
            _state_bytes(path).decode("utf-8"),
            object_pairs_hook=hook_host._strict_pairs,
            parse_constant=hook_host._reject_constant,
        )
    except PowerupError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise PowerupError(f"{path}: invalid JSON: {error}") from error
    if (
        type(value) is not dict
        or value.get("schema") != SCHEMA
        or set(value) - {"schema", "global", "profiles"}
    ):
        raise PowerupError(f"{path}: expected {SCHEMA}")
    global_state, profiles = value.get("global", {}), value.get("profiles", {})
    if (
        type(global_state) is not dict
        or type(profiles) is not dict
        or len(global_state) > 256
        or len(profiles) > 64
    ):
        raise PowerupError(f"{path}: invalid state sections")
    for section in [global_state, *profiles.values()]:
        if type(section) is not dict or len(section) > 256:
            raise PowerupError(f"{path}: invalid profile section")
        for name, row in section.items():
            if (
                type(name) is not str
                or not NAME.fullmatch(name)
                or type(row) is not dict
                or set(row) - {"mode", "settings", "digest", "trust_gating", "digest_version"}
            ):
                raise PowerupError(f"{path}: invalid state entry")
            if "mode" in row and (
                type(row["mode"]) is not str or row["mode"] not in MODES
            ):
                raise PowerupError(f"{path}: invalid mode for {name}")
            if "settings" in row and (
                type(row["settings"]) is not dict or len(row["settings"]) > 32
            ):
                raise PowerupError(f"{path}: invalid settings for {name}")
            if "digest" in row and (
                type(row["digest"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", row["digest"]) is None
            ):
                raise PowerupError(f"{path}: invalid digest for {name}")
            if "digest_version" in row and (
                type(row["digest_version"]) is not int or row["digest_version"] < 1
            ):
                raise PowerupError(f"{path}: invalid digest version for {name}")
            if "trust_gating" in row and (
                type(row["trust_gating"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", row["trust_gating"]) is None
            ):
                raise PowerupError(f"{path}: invalid gating trust for {name}")
    return {"schema": SCHEMA, "global": global_state, "profiles": profiles}


def _write(path: Path, value: dict) -> None:
    private_directory(path.parent)
    if path.is_symlink():
        raise PowerupError(f"{path}: linked state file")
    if path.exists():
        _read(path)
    temporary = path.parent / ("." + path.name + "-" + secrets.token_hex(12))
    try:
        with temporary.open("x", encoding="utf-8") as output:
            os.chmod(temporary, 0o600)
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def user_home_or_none() -> Path | None:
    """The StateFrame home for READING user-scope powerups on a run, or None when there is no home at all.

    No home (HOME unset, no passwd entry, no STATEFRAME_HOME) means there can be no user powerups or user state,
    so a run treats them as empty. Commands that WRITE (install, enable, trust, gc) keep using home() and its
    clear error."""
    try:
        return home()
    except PowerupError:
        return None


def user_state() -> dict:
    user_home = user_home_or_none()
    if user_home is None:
        return {"schema": SCHEMA, "global": {}, "profiles": {}}
    return _read(user_home / "powerups.json")


def update(
    name: str,
    *,
    mode: str | None = None,
    settings: dict | None = None,
    digest: str | None = None,
    profile: str | None = None,
    trust_gating: str | None = None,
) -> None:
    if not NAME.fullmatch(name):
        raise PowerupError(f"invalid powerup name: {name}")
    path = home() / "powerups.json"
    value = _read(path)
    section = value["profiles"].setdefault(profile, {}) if profile else value["global"]
    row = dict(section.get(name, {}))
    if mode is not None:
        if type(mode) is not str or mode not in MODES:
            raise PowerupError(f"invalid powerup mode: {mode}")
        row["mode"] = mode
    if settings is not None:
        row["settings"] = {**row.get("settings", {}), **settings}
    if digest is not None:
        row["digest"] = digest
    if trust_gating is not None:
        row["trust_gating"] = trust_gating
    if digest is not None or trust_gating is not None:
        row["digest_version"] = TREE_DIGEST_VERSION
    section[name] = row
    _write(path, value)


def mark_installed_off(name: str) -> None:
    """Clear stale profile enables when a new installation takes this name."""
    if not NAME.fullmatch(name):
        raise PowerupError(f"invalid powerup name: {name}")
    path = home() / "powerups.json"
    value = _read(path)
    value["global"][name] = {"mode": "off"}
    for section in value["profiles"].values():
        section.pop(name, None)
    _write(path, value)


def trusted_digest(repo: Path, path: Path) -> str | None:
    root = Path(repo).resolve()
    source = Path(path).resolve()
    try:
        entries = hook_host._read_trust_store()
    except hook_host.HookTrustError as error:
        raise PowerupError(str(error)) from error
    for row in entries:
        if row["repo"] == str(root) and row["file"] == str(source):
            return row["digest"]
    return None


def project_state(repo: Path) -> dict:
    path = Path(repo).resolve() / ".stateframe" / "powerups.json"
    if not path.exists():
        return {"schema": SCHEMA, "global": {}, "profiles": {}}
    digest = trusted_digest(repo, path)
    if digest is None:
        return {"schema": SCHEMA, "global": {}, "profiles": {}}
    if digest != tree_digest_file(path):
        # A trusted file that changed may hold the user's "off": fail closed, visibly.
        raise PowerupError(
            f"re-trust required: {path} changed since it was trusted; run "
            f"'stateframe powerups trust --repo {Path(repo).resolve()}' after review"
        )
    return _read(path)


def tree_digest_file(path: Path) -> str:
    return hashlib.sha256(_state_bytes(path)).hexdigest()


def trust(repo: Path, *, remove: bool = False) -> list[str]:
    root = Path(repo).resolve(strict=True)
    sources = [root / ".stateframe" / "powerups.json"]
    directory = root / ".stateframe" / "powerups"
    if directory.is_dir() and not directory.is_symlink():
        sources += sorted(
            path
            for path in directory.iterdir()
            if path.is_dir() and not path.is_symlink() and NAME.fullmatch(path.name)
        )
    entries = hook_host._read_trust_store()
    source_names = {str(path) for path in sources}
    project_prefix = str(directory) + os.sep
    kept = [
        row
        for row in entries
        if not (
            row["repo"] == str(root)
            and (row["file"] in source_names or row["file"].startswith(project_prefix))
        )
    ]
    if not remove:
        for source in sources:
            if source.is_file() or source.is_dir():
                if source.is_file():
                    _read(source)
                digest = (
                    tree_digest(source) if source.is_dir() else tree_digest_file(source)
                )
                kept.append({"repo": str(root), "file": str(source), "digest": digest})
    kept.sort(key=lambda row: (row["repo"], row["file"], row["digest"]))
    hook_host._write_trust_store(kept)
    return [
        row["file"]
        for row in kept
        if row["repo"] == str(root) and row["file"] in source_names
    ]


def effective_state(
    manifest: Manifest,
    profile: str,
    *,
    project: dict | None = None,
    cli: dict | None = None,
    cli_settings: dict | None = None,
    installed: bool = False,
    project_source: bool = False,
    user: dict | None = None,
) -> dict:
    state = user if user is not None else user_state()
    installed_mode = installed or (manifest.path / "install.json").is_file()
    mode = "off" if installed_mode else manifest.default
    mode_source = "installed" if installed_mode else "manifest"
    settings = {key: spec["default"] for key, spec in manifest.settings.items()}
    setting_sources = {key: "manifest" for key in settings}
    digest = None
    gating_digest = None
    digest_version = None
    for label, section in (
        ("user-global", state["global"]),
        (f"profile:{profile}", state["profiles"].get(profile, {})),
    ):
        row = section.get(manifest.name, {})
        if "mode" in row:
            mode, mode_source = row["mode"], label
        settings.update(row.get("settings", {}))
        setting_sources.update({key: label for key in row.get("settings", {})})
        if "digest" in row or "trust_gating" in row:
            digest_version = row.get("digest_version")
        digest = row.get("digest", digest)
        gating_digest = row.get("trust_gating", gating_digest)
    if project:
        row = project.get("global", {}).get(manifest.name, {})
        proposed = row.get("mode", mode)
        if "mode" in row and (proposed != "on" or project_source):
            mode, mode_source = proposed, "project"
        settings.update(row.get("settings", {}))
        setting_sources.update({key: "project" for key in row.get("settings", {})})
    if cli and manifest.name in cli:
        mode, mode_source = cli[manifest.name], "cli"
    if cli_settings:
        settings.update(cli_settings)
        setting_sources.update({key: "cli" for key in cli_settings})
    if set(settings) - set(manifest.settings):
        raise PowerupError(
            f"unknown setting {sorted(set(settings) - set(manifest.settings))[0]}"
        )
    settings = {
        key: setting_value(manifest.settings[key], value, f"{manifest.name}.{key}")
        for key, value in settings.items()
    }
    return {
        "mode": mode,
        "mode_source": mode_source,
        "settings": settings,
        "setting_sources": setting_sources,
        "digest": digest,
        "trust_gating": gating_digest,
        "digest_version": digest_version,
    }
