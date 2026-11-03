"""Powerup management commands, separate from run/chat argument handling."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

from . import powerups, profiles
from .powerup_manifest import tree_digest
from .powerup_materialize import _read_json
from .powerup_state import home
from .skills import NAME


def register(commands: argparse._SubParsersAction) -> None:
    root = commands.add_parser("powerups", help="manage installed and trusted powerups")
    actions = root.add_subparsers(dest="powerups_action", required=True)
    listing = actions.add_parser("list", help="list installed powerups")
    listing.add_argument("--profile", default=profiles.DEFAULT)
    show = actions.add_parser(
        "show", help="show manifest, parts and effective settings"
    )
    show.add_argument("name")
    show.add_argument("--profile", default=profiles.DEFAULT)
    installed = actions.add_parser("install", help="validate and install a powerup")
    installed.add_argument("path", type=Path)
    installed.add_argument("--force", action="store_true")
    removed = actions.add_parser("remove", help="remove an installed powerup")
    removed.add_argument("name")
    enabled = actions.add_parser("enable", help="enable and pin an installed powerup")
    enabled.add_argument("name")
    enabled.add_argument("--mode", choices=("on", "offer", "auto"), default="on")
    enabled.add_argument("--profile")
    enabled.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    enabled.add_argument(
        "--trust-gating", action="store_true", help="allow promoted hooks to gate a run"
    )
    disabled = actions.add_parser("disable", help="disable an installed powerup")
    disabled.add_argument("name")
    disabled.add_argument("--profile")
    trusted = actions.add_parser(
        "trust", help="trust exact project powerup tree digests"
    )
    trusted.add_argument("--repo", type=Path, required=True)
    trusted.add_argument("--remove", action="store_true")
    promoted = actions.add_parser(
        "promote", help="install a recorded mutation as a disabled powerup"
    )
    promoted.add_argument("run_dir", type=Path)
    promoted.add_argument("mutation_id")
    promoted.add_argument("--name", required=True)


def _reader(run_dir: Path, mutation_id: str) -> Path:
    """Read a digest-pinned attach-unit index when no external reader is supplied."""
    index = run_dir / "mutations" / "index.json"
    value = _read_json(index)
    units = value.get("units")
    record = units.get(mutation_id) if type(units) is dict else None
    if (
        type(record) is not dict
        or set(record) != {"directory", "sha256"}
        or type(record["directory"]) is not str
        or type(record["sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
    ):
        raise powerups.PowerupError("mutation attach-unit index has no valid record")
    relative = Path(record["directory"])
    if relative.is_absolute() or ".." in relative.parts:
        raise powerups.PowerupError("mutation unit path escapes run records")
    source = run_dir / "mutations" / relative
    if not source.resolve().is_relative_to((run_dir / "mutations").resolve()):
        raise powerups.PowerupError("mutation unit path escapes run records")
    if tree_digest(source) != record["sha256"]:
        raise powerups.PowerupError("recorded mutation unit digest changed")
    return source


def handle(
    args: argparse.Namespace, *, reader: Callable[[Path, str], Path] | None = None
) -> int:
    action = args.powerups_action
    if action in {"list", "show"}:
        names = (
            [args.name]
            if action == "show"
            else sorted(
                path.name
                for path in (home() / "powerups").iterdir()
                if path.is_dir() and NAME.fullmatch(path.name)
            )
            if (home() / "powerups").is_dir()
            else []
        )
        for name in names:
            path = home() / "powerups" / name
            try:
                manifest = powerups.parse_manifest(path)
                state = powerups.effective_state(manifest, args.profile, installed=True)
            except (powerups.PowerupError, OSError) as error:
                if action == "show":
                    raise
                print(f"{name}\t{path}\tinvalid\t-\tinvalid\t-\t{str(error)[:200]}")
                continue
            parts = [
                part
                for part, file in (
                    ("skill", "SKILL.md"),
                    ("commands", "commands.json"),
                    ("hooks", "hooks.json"),
                    ("mcp", "mcp.json"),
                )
                if (manifest.path / file).is_file()
            ]
            digest = tree_digest(manifest.path)
            status = (
                "changed"
                if state["digest"] and state["digest"] != digest
                else "current"
            )
            if action == "list":
                print(
                    f"{name}\t{manifest.path}\t{state['mode']}\t{','.join(parts)}\t{status}\t{digest[:12]}"
                )
            else:
                print(
                    json.dumps(
                        {
                            "name": name,
                            "summary": manifest.summary,
                            "version": manifest.version,
                            "mode": state["mode"],
                            "mode_source": state["mode_source"],
                            "settings": state["settings"],
                            "setting_sources": state["setting_sources"],
                            "parts": parts,
                            "source": str(manifest.path),
                            "digest_status": status,
                            "tree_sha256": digest,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
    elif action == "install":
        manifest = powerups.install(args.path, force=args.force)
        print(
            f"installed {manifest.name}; mode off; sha256 {tree_digest(manifest.path)}"
        )
    elif action == "remove":
        powerups.remove(args.name)
        print(f"removed {args.name}")
    elif action == "enable":
        settings = {}
        for value in args.set:
            key, separator, setting = value.partition("=")
            if not separator or not key:
                raise powerups.PowerupError(f"invalid --set: {value}")
            settings[key] = setting
        manifest = powerups.enable(
            args.name,
            mode=args.mode,
            profile=args.profile,
            settings=settings,
            trust_gating=args.trust_gating,
        )
        for filename, section in (
            ("commands.json", "tools"),
            ("hooks.json", "hooks"),
            ("mcp.json", "servers"),
        ):
            path = manifest.path / filename
            if path.is_file():
                for item in json.loads(path.read_text())[section]:
                    print(
                        json.dumps(
                            {
                                key: item.get(key)
                                for key in (
                                    "name",
                                    "event",
                                    "argv",
                                    "network",
                                    "writes",
                                    "replay",
                                )
                                if key in item
                            },
                            sort_keys=True,
                        )
                    )
        print(
            f"enabled {manifest.name} as {args.mode}; sha256 {tree_digest(manifest.path)}"
        )
    elif action == "disable":
        powerups.disable(args.name, profile=args.profile)
        print(f"disabled {args.name}")
    elif action == "trust":
        paths = powerups.trust(args.repo, remove=args.remove)
        if args.remove:
            print(f"removed project powerup trust for {args.repo.resolve()}")
        for path in paths:
            print(f"trusted {path}")
    else:
        manifest = powerups.promote(
            args.run_dir, args.mutation_id, args.name, reader or _reader
        )
        print(f"installed {manifest.name} from mutation {args.mutation_id}; mode off")
    return 0
