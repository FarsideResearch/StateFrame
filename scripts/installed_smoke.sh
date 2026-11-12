#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'FAIL: installed mutation smoke: %s\n' "$*" >&2
  exit 1
}

if [[ -z "${STATEFRAME_BIN:-}" ]]; then
  fail "STATEFRAME_BIN must be set"
fi
if [[ "$STATEFRAME_BIN" != /* ]]; then
  resolved_bin="$(command -v "$STATEFRAME_BIN" || true)"
  [[ -n "$resolved_bin" ]] || fail "STATEFRAME_BIN is not executable"
  STATEFRAME_BIN="$resolved_bin"
fi
[[ -x "$STATEFRAME_BIN" ]] || fail "STATEFRAME_BIN is not executable"
bin_dir="$(cd "$(dirname "$STATEFRAME_BIN")" && pwd -P)" || fail "cannot resolve STATEFRAME_BIN directory"
STATEFRAME_BIN="$bin_dir/$(basename "$STATEFRAME_BIN")"
STATEFRAME_PYTHON="$bin_dir/python"
[[ -x "$STATEFRAME_PYTHON" ]] || fail "venv Python is missing next to STATEFRAME_BIN"

temp_root="$(mktemp -d "${TMPDIR:-/tmp}/stateframe-installed-mutation.XXXXXX")" \
  || fail "cannot create temporary directory"
STATEFRAME_HOME="$temp_root/home"
mkdir -m 700 "$STATEFRAME_HOME" || fail "cannot create temporary STATEFRAME_HOME"
export STATEFRAME_HOME
trap 'rm -rf "$temp_root"' EXIT
unset PYTHONPATH
cd "$temp_root"

os_name="$(uname -s 2>/dev/null)" || fail "cannot identify operating system"
if [[ "$os_name" == "Linux" ]]; then
  report="$temp_root/selftest.json"
  errors="$temp_root/selftest.stderr"
  if "$STATEFRAME_BIN" mutations selftest --json >"$report" 2>"$errors"; then
    cli_status=0
  else
    cli_status=$?
  fi
  if ! "$STATEFRAME_PYTHON" - "$report" "$cli_status" <<'PY'
import json
from pathlib import Path
import sys

try:
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, UnicodeError, ValueError) as error:
    print(f"selftest output is not valid JSON: {type(error).__name__}: {error}", file=sys.stderr)
    raise SystemExit(1)

if not isinstance(report, dict):
    print("selftest report is not an object", file=sys.stderr)
    raise SystemExit(1)

checks = report.get("checks", report.get("units", []))
failed = False
if not isinstance(checks, list):
    print("selftest checks are not a list", file=sys.stderr)
    failed = True
else:
    for check in checks:
        if not isinstance(check, dict) or check.get("ok") is not True:
            name = check.get("name", "unknown") if isinstance(check, dict) else "unknown"
            error = (check.get("detail") or check.get("error") or check.get("reason")
                     or "check did not report ok=true") if isinstance(check, dict) else str(check)
            print(f"{name}: {error}", file=sys.stderr)
            failed = True

if report.get("ok") is not True:
    if not failed:
        print("selftest: report ok is not true", file=sys.stderr)
    failed = True
if int(sys.argv[2]) != 0:
    print(f"selftest command exited {sys.argv[2]}", file=sys.stderr)
    failed = True
raise SystemExit(1 if failed else 0)
PY
  then
    if [[ -s "$errors" ]]; then
      cat "$errors" >&2
    fi
    fail "Linux selftest did not report ok=true with exit 0"
  fi
else
  refusal_out="$temp_root/refusal.stdout"
  refusal_err="$temp_root/refusal.stderr"
  if "$STATEFRAME_BIN" mutations selftest --json >"$refusal_out" 2>"$refusal_err"; then
    cat "$refusal_out" "$refusal_err" >&2
    fail "mutation selftest unexpectedly succeeded on $os_name"
  fi
  if ! grep -Fq "MUTATION_PLATFORM_UNSUPPORTED" "$refusal_out" "$refusal_err"; then
    cat "$refusal_out" "$refusal_err" >&2
    fail "mutation selftest did not report MUTATION_PLATFORM_UNSUPPORTED"
  fi
  if grep -Fq "Traceback" "$refusal_out" "$refusal_err"; then
    cat "$refusal_out" "$refusal_err" >&2
    fail "mutation selftest printed a traceback"
  fi
  if ! "$STATEFRAME_BIN" mutations list >"$temp_root/list.stdout" 2>"$temp_root/list.stderr"; then
    cat "$temp_root/list.stdout" "$temp_root/list.stderr" >&2
    fail "read-only mutations list failed"
  fi
fi

printf 'PASS: installed mutation smoke (%s)\n' "$os_name"
