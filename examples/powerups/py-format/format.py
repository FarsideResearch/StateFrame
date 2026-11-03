"""Bounded ruff formatter hook; missing ruff is an optional capability."""

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    bundled = Path(__file__).with_name("ruff")
    executable = (
        str(bundled)
        if bundled.is_file() and bundled.stat().st_mode & 0o111
        else shutil.which("ruff")
    )
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        if executable is None:
            print(
                "py-format: ruff is unavailable; automatic formatting is inactive for this run."
            )
        return 0
    if executable is None:
        return 0
    if len(sys.argv) < 4 or sys.argv[1] != "--line-length":
        return 1
    try:
        length = int(sys.argv[2])
    except ValueError:
        return 1
    paths = [path for path in sys.argv[3:] if path.endswith(".py")]
    if not paths:
        return 0
    try:
        result = subprocess.run(
            [executable, "format", "--line-length", str(length), "--", *paths],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return 0
    except subprocess.TimeoutExpired:
        sys.stderr.write("ruff format timed out")
        return 1
    if result.returncode:
        sys.stderr.write(result.stderr[-2048:])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
