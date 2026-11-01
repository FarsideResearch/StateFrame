#!/usr/bin/python3
"""Refuse simple shell command strings that contain selected risky commands."""

import json
import re
import sys


payload = json.load(sys.stdin)
data = payload.get("data", {})
arguments = data.get("args", {})
command = arguments.get("command", "") if isinstance(arguments, dict) else ""

blocked = (
    re.search(r"\bgit\s+push(?:\s|$)", command)
    or re.search(r"\brm\s+-[^\s]*r[^\s]*f[^\s]*(?:\s|$)", command)
    or re.search(r"\brm\s+-[^\s]*f[^\s]*r[^\s]*(?:\s|$)", command)
)
if blocked:
    print("This example guard refuses git push and rm -rf command text.", file=sys.stderr)
    raise SystemExit(2)

print("Shell command passed the example text check.")
