"""Compare raw no-server model request files against a 0.2.1 source tree.

Run with --baseline /path/to/97cd52d --candidate /path/to/this/tree.
Both runs use the same temporary repository and run-directory path in sequence.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile


def source_env(tree: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(tree / "src"),
        str(tree / "src/stateframe/engine")))
    return environment


def write_launcher(path: Path) -> None:
    path.write_text("import runpy, secrets\n"
                    "_sequence = 0\n"
                    "def fixed_hex(size=32):\n"
                    "    global _sequence\n"
                    "    _sequence += 1\n"
                    "    return format(_sequence, 'x').zfill(size * 2)[-size * 2:]\n"
                    "secrets.token_hex = fixed_hex\n"
                    "runpy.run_module('stateframe', run_name='__main__')\n")


def capture(tree: Path, python: str, launcher: Path, script: Path,
            repo: Path, run_dir: Path, *, legacy_mode: bool = False) -> bytes:
    environment = source_env(tree)
    endpoint = subprocess.Popen([python, "-m", "stateframe.scripted_endpoint",
        "--script", str(script), "--port", "0"], cwd=tree, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert endpoint.stdout is not None
        ready, _, _ = select.select([endpoint.stdout], [], [], 15)
        if not ready:
            raise RuntimeError("scripted endpoint did not start")
        url = endpoint.stdout.readline().strip()
        if not url.startswith("http://127.0.0.1:"):
            raise RuntimeError(f"unexpected scripted endpoint URL: {url!r}")
        command = [python, str(launcher), "run", "--repo", str(repo),
            "--task", "Fix the pricing code so the tests pass.", "--endpoint", url,
            "--model", "scripted", "--run-dir", str(run_dir), "--test-python", python,
            "--test-cmd", "python -m unittest discover -s tests -q", "--max-turns", "32",
            "--no-run-env", "--quiet",
            *(("--runner-kind", "pytest", "--protect-tests", "all") if legacy_mode else ())]
        result = subprocess.run(command, cwd=tree, env=environment, capture_output=True,
                                text=True, timeout=300, check=False)
        # The comparison is about request bytes, so a run whose checks fail still counts;
        # only a run that never wrote its summary or requests is a capture failure.
        files = sorted((run_dir / "requests").glob("*.json"))
        if not files or not (run_dir / "summary.json").is_file():
            raise RuntimeError(f"{tree}: run produced no requests or summary ({result.returncode}):\n"
                               f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        return b"\n".join(path.read_bytes() for path in files)
    finally:
        endpoint.terminate()
        try:
            endpoint.wait(timeout=5)
        except subprocess.TimeoutExpired:
            endpoint.kill()
            endpoint.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="source tree at origin/main 97cd52d")
    parser.add_argument("--candidate", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    baseline = args.baseline.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    script = candidate / "examples/discount_bug.script.json"
    with tempfile.TemporaryDirectory(prefix="sf-no-mcp-diff-") as temporary:
        root = Path(temporary)
        repo = root / "repo"
        shutil.copytree(candidate / "examples/discount_bug", repo)
        launcher = root / "fixed_seed_run.py"
        write_launcher(launcher)
        run_dir = root / "run"
        previous = capture(baseline, args.python, launcher, script, repo, run_dir)
        run_dir.rename(root / "baseline-run")
        current = capture(candidate, args.python, launcher, script, repo, run_dir)
        for label, body in (("0.2.1", previous), ("candidate", current)):
            print(f"{label}: {len(body)} bytes sha256={hashlib.sha256(body).hexdigest()}")
        if previous == current:
            print("Model request files are byte-identical.")
            return 0
        before = previous.decode(errors="replace").splitlines(keepends=True)
        after = current.decode(errors="replace").splitlines(keepends=True)
        sys.stdout.writelines(difflib.unified_diff(before, after, fromfile="0.2.1",
                                                   tofile="candidate", n=3))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
