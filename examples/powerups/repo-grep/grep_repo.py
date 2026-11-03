#!/usr/bin/env python3
"""Search literal text with bounded traversal and output."""

import fnmatch
import os
import stat
import sys
from pathlib import Path

MAX_ENTRIES = 20_000


def _search_file(path: Path, relative: str, pattern: str, remaining: int) -> int:
    """Read one regular file without following a link after inspection."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, encoding="utf-8") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1 << 20:
                return 0
            shown = 0
            for number, line in enumerate(source, 1):
                if pattern in line:
                    print(f"{relative}:{number}: {line.strip()[:200]}")
                    shown += 1
                    if shown >= remaining:
                        return shown
            return shown
    except (OSError, UnicodeError):
        return 0


def main() -> int:
    if len(sys.argv) != 3 or not sys.argv[1] or not sys.argv[2]:
        return 2
    pattern, glob = sys.argv[1:]
    root = Path.cwd()
    shown = 0
    visited = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = []
                for entry in iterator:
                    visited += 1
                    if visited > MAX_ENTRIES:
                        print(f"search stopped at {MAX_ENTRIES} directory entries")
                        return 0
                    entries.append(entry)
        except OSError:
            continue
        children = []
        for entry in sorted(entries, key=lambda item: item.name):
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in {
                        ".git",
                        ".venv",
                        "node_modules",
                        "__pycache__",
                    }:
                        children.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    path = Path(entry.path)
                    relative = path.relative_to(root).as_posix()
                    if fnmatch.fnmatchcase(relative, glob):
                        shown += _search_file(path, relative, pattern, 50 - shown)
                        if shown >= 50:
                            print("search stopped at 50 matches")
                            return 0
            except OSError:
                continue
        pending.extend(reversed(children))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
