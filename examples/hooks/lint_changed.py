#!/usr/bin/python3
"""Parse changed Python files and report syntax errors."""

import ast
import json
from pathlib import Path
import sys


payload = json.load(sys.stdin)
data = payload.get("data", {})
paths = data.get("changed_paths", []) if isinstance(data, dict) else []
workspace = Path.cwd().resolve()
failed = False
for name in paths:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        print("lint received an invalid workspace path", file=sys.stderr)
        failed = True
        continue
    target = workspace / relative
    if target.is_symlink() or not target.is_file():
        print("lint received a missing or linked file", file=sys.stderr)
        failed = True
        continue
    try:
        resolved = target.resolve(strict=True)
        if resolved != workspace and workspace not in resolved.parents:
            print("lint path escaped the workspace", file=sys.stderr)
            failed = True
            continue
        source = target.read_text(encoding="utf-8")
        ast.parse(source, filename=relative.as_posix())
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        print(str(error), file=sys.stderr)
        failed = True
    else:
        print("syntax ok: " + relative.as_posix())

if failed:
    raise SystemExit(1)
