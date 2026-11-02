#!/usr/bin/python3
"""Trim trailing whitespace and ensure a final newline in changed text files."""

from pathlib import Path
import sys


workspace = Path.cwd().resolve()
for argument in sys.argv[1:]:
    relative = Path(argument)
    if relative.is_absolute() or ".." in relative.parts:
        print("formatter received an invalid workspace path", file=sys.stderr)
        raise SystemExit(1)
    target = workspace / relative
    if target.is_symlink() or not target.is_file():
        print("formatter received a missing or linked file", file=sys.stderr)
        raise SystemExit(1)
    resolved = target.resolve(strict=True)
    if resolved != workspace and workspace not in resolved.parents:
        print("formatter path escaped the workspace", file=sys.stderr)
        raise SystemExit(1)
    try:
        text = target.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print("formatter could not read a UTF-8 file: " + str(error), file=sys.stderr)
        raise SystemExit(1) from error
    lines = [line.rstrip() for line in text.splitlines()]
    formatted = "\n".join(lines)
    if formatted:
        formatted += "\n"
    if formatted != text:
        target.write_text(formatted, encoding="utf-8")
    print("checked formatting: " + relative.as_posix())
