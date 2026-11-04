"""Install, enable, and promote reviewed powerup units."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from .engine.kernel_safe_paths import safe_expanduser
from .powerup_manifest import (
    Manifest,
    PowerupError,
    parse_manifest,
    setting_value,
    tree_digest,
)
from .powerup_state import home, mark_installed_off, update
from .skills import NAME
from .state_dirs import private_directory


def install(
    path: Path, *, force: bool = False, provenance: str | None = None
) -> Manifest:
    from .powerups import _materialize, _parts, _validate

    supplied = safe_expanduser(Path(path))
    if supplied.is_symlink():
        raise PowerupError(f"powerup directory is linked: {supplied}")
    source = supplied.resolve(strict=True)
    manifest = parse_manifest(source)
    validated_sha = tree_digest(source)
    if (source / "install.json").exists():
        raise PowerupError("install.json is reserved for installation provenance")
    with tempfile.TemporaryDirectory(prefix="stateframe-powerup-check-") as scratch:
        rendered, _ = _materialize(
            manifest,
            {key: spec["default"] for key, spec in manifest.settings.items()},
            shell_network="full",
            repo=source,
            cache_root=Path(scratch) / "cache",
            trust_gating=True,
            filter_generated_hooks=False,
        )
        _validate(
            manifest,
            rendered,
            _parts(manifest, "on"),
            repo=source,
            shell_network="full",
        )
    destination_root = home() / "powerups"
    private_directory(destination_root)
    destination = destination_root / manifest.name
    if (destination.exists() or destination.is_symlink()) and not force:
        raise PowerupError(f"powerup {manifest.name} is already installed")
    temporary = Path(tempfile.mkdtemp(prefix=".install-", dir=destination_root))
    try:
        copied = temporary / manifest.name
        shutil.copytree(source, copied, symlinks=True)
        if tree_digest(copied) != validated_sha or tree_digest(source) != validated_sha:
            raise PowerupError("source changed during installation")
        (copied / "install.json").write_text(
            json.dumps({"provenance": provenance or str(source)}) + "\n"
        )
        copied_manifest = parse_manifest(copied)
        with tempfile.TemporaryDirectory(prefix="stateframe-powerup-publish-check-") as scratch:
            rendered, _ = _materialize(
                copied_manifest,
                {key: spec["default"] for key, spec in manifest.settings.items()},
                shell_network="full", repo=copied, cache_root=Path(scratch) / "cache",
                trust_gating=True, filter_generated_hooks=False,
            )
            _validate(copied_manifest, rendered, _parts(copied_manifest, "on"), repo=copied,
                      shell_network="full")
        if destination.exists() or destination.is_symlink():
            try:
                previous_suffix = tree_digest(destination)[:12]
            except PowerupError:
                previous_suffix = secrets.token_hex(12)
            preserved = destination_root / (
                manifest.name + ".previous-" + previous_suffix
            )
            if preserved.exists() or preserved.is_symlink():
                raise PowerupError(f"cannot preserve prior installation: {preserved}")
            destination.rename(preserved)
        copied.rename(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    mark_installed_off(manifest.name)
    return parse_manifest(destination)


def remove(name: str) -> None:
    if not NAME.fullmatch(name):
        raise PowerupError(f"invalid powerup name: {name}")
    source = home() / "powerups" / name
    if source.is_symlink() or not source.is_dir():
        raise PowerupError(f"powerup {name} is not installed")
    shutil.rmtree(source)
    update(name, mode="off")


def enable(
    name: str,
    *,
    mode: str = "on",
    profile: str | None = None,
    settings: dict | None = None,
    trust_gating: bool = False,
) -> Manifest:
    from .powerups import _materialize, _parts, _read_json, _validate

    if type(mode) is not str or mode not in {"on", "offer", "auto"}:
        raise PowerupError(f"invalid enable mode: {mode}")
    manifest = parse_manifest(home() / "powerups" / name)
    provenance = _read_json(manifest.path / "install.json").get("provenance", "")
    if type(provenance) is not str:
        raise PowerupError("invalid installation provenance")
    source_digest = tree_digest(manifest.path)
    values = {
        key: setting_value(manifest.settings[key], value, f"{name}.{key}")
        for key, value in (settings or {}).items()
        if key in manifest.settings
    }
    if settings and set(settings) - set(manifest.settings):
        raise PowerupError(
            f"unknown setting {sorted(set(settings) - set(manifest.settings))[0]}"
        )
    rendered, _ = _materialize(
        manifest,
        {
            key: values.get(key, spec["default"])
            for key, spec in manifest.settings.items()
        },
        shell_network="full",
        repo=manifest.path,
        trust_gating=True,
        filter_generated_hooks=False,
    )
    _validate(
        manifest,
        rendered,
        _parts(manifest, "on"),
        repo=manifest.path,
        shell_network="full",
    )
    if tree_digest(manifest.path) != source_digest:
        raise PowerupError("powerup changed during enablement")
    update(
        name,
        mode=mode,
        settings=values,
        digest=source_digest,
        profile=profile,
        trust_gating=source_digest if trust_gating else None,
    )
    return manifest


def disable(name: str, *, profile: str | None = None) -> None:
    update(name, mode="off", profile=profile)


def trust(repo: Path, *, remove: bool = False) -> list[str]:
    from .powerup_state import trust as trust_project

    return trust_project(repo, remove=remove)


def promote(
    run_dir: Path, mutation_id: str, name: str, reader: Callable[[Path, str], Path]
) -> Manifest:
    if not NAME.fullmatch(name) or not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,80}", mutation_id
    ):
        raise PowerupError("invalid promotion name or mutation id")
    supplied = Path(reader(Path(run_dir), mutation_id))
    if supplied.is_symlink():
        raise PowerupError(f"recorded mutation unit is linked: {supplied}")
    recorded = supplied.resolve(strict=True)
    tree_digest(recorded)
    with tempfile.TemporaryDirectory(prefix="stateframe-promote-") as scratch:
        source = Path(scratch) / name
        shutil.copytree(recorded, source)
        skill = source / "SKILL.md"
        summary = "Promoted command from a recorded mutation."
        if skill.is_file():
            body = skill.read_text()
            match = re.search(r"^description:\s*(.+)$", body, re.MULTILINE)
            if match:
                summary = match.group(1)[:120]
            body = re.sub(
                r"^name:\s*[^\n]+$", f"name: {name}", body, count=1, flags=re.MULTILINE
            )
            skill.write_text(body)
        (source / "powerup.toml").write_text(
            f'schema = "stateframe-powerup/1"\nname = "{name}"\nversion = "1.0.0"\n'
            f'summary = {json.dumps(summary)}\ndefault = "off"\n'
        )
        return install(
            source,
            provenance=f"promoted from mutation {mutation_id} in {Path(run_dir).resolve()}",
        )
