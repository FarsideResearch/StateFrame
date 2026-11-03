"""Bounded one-time auto-offer rules."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from .powerup_manifest import Manifest, PowerupError


def _offer_matches(manifest: Manifest, repo: Path, task: str, loop: str) -> bool:
    offer = manifest.offer
    if loop not in offer.get("loops", ["composed"]):
        return False
    if not any(offer.get(key) for key in ("any_files", "all_of", "task_matches")):
        return False
    paths: list[str] = []
    if offer.get("any_files") or offer.get("all_of"):
        pending = [repo]
        visited = 0
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as iterator:
                entries = []
                for entry in iterator:
                    visited += 1
                    if visited > 20_000:
                        raise PowerupError("offer scan exceeds 20000 entries")
                    entries.append(entry)
            for entry in sorted(entries, key=lambda item: item.name, reverse=True):
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in {
                        ".git",
                        ".stateframe",
                        ".venv",
                        "node_modules",
                    }:
                        pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    paths.append(Path(entry.path).relative_to(repo).as_posix())
    any_files = offer.get("any_files", [])
    all_of = offer.get("all_of", [])
    patterns = offer.get("task_matches", [])
    matched_file = any(
        fnmatch.fnmatchcase(path, pat) for path in paths for pat in any_files
    )
    matched_all = all(
        any(fnmatch.fnmatchcase(path, pat) for path in paths) for pat in all_of
    )
    matched_task = any(re.search(pattern, task[:8192]) for pattern in patterns)
    return matched_all and (
        matched_file or matched_task or bool(all_of) and not any_files and not patterns
    )
