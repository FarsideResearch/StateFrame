"""Run one configured check without a shell, returning bounded veto output."""

import os
import selectors
import shlex
import shutil
import subprocess
import sys
import time

MAX_OUTPUT_BYTES = 1 << 20
TAIL_BYTES = 2048


def _run_check(argv: list[str]) -> int:
    try:
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except FileNotFoundError as error:
        sys.stderr.write(f"test-gate: configured check could not start: {error}")
        return 2
    except OSError as error:
        sys.stderr.write(f"configured check could not start: {error}")
        return 2
    assert process.stdout is not None
    deadline = time.monotonic() + 120
    tail = bytearray()
    total = 0
    failure = None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = "configured check timed out"
                    break
                if not selector.select(remaining):
                    continue
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                total += len(chunk)
                tail.extend(chunk)
                del tail[:-TAIL_BYTES]
                if total > MAX_OUTPUT_BYTES:
                    failure = "configured check output limit exceeded"
                    break
        if failure is None:
            try:
                code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                failure = "configured check timed out"
            else:
                if code:
                    sys.stderr.write(tail.decode("utf-8", errors="replace"))
                    return 2
                return 0
    finally:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        process.wait()
        process.stdout.close()
    sys.stderr.write(failure or "configured check failed")
    return 2


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"--check-program", "--run"}:
        return 1
    try:
        argv = shlex.split(sys.argv[2])
    except ValueError as error:
        sys.stderr.write(str(error))
        return 2 if sys.argv[1] == "--run" else 0
    program = shutil.which(argv[0]) if argv else None
    if sys.argv[1] == "--check-program":
        if program is None:
            print(
                "test-gate: check program is unavailable; submission will be refused until it is installed."
            )
        return 0
    if program is None:
        sys.stderr.write(f"test-gate: check program {argv[0] if argv else '(empty)'} is not installed")
        return 2
    return _run_check([program, *argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
