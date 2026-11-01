#!/usr/bin/python3
"""Require a completed passing check before allowing a submission."""

import json
import sys


payload = json.load(sys.stdin)
data = payload.get("data", {})
checks = data.get("checks", {}) if isinstance(data, dict) else {}
passed = (
    isinstance(checks, dict)
    and checks.get("status") == "PASS"
    and checks.get("exit_code") == 0
    and checks.get("execution_complete") is True
)
if not passed:
    print("Run a check that passes before submitting this result.", file=sys.stderr)
    raise SystemExit(2)

print("The latest check passed.")
