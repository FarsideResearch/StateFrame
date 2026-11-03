"""Compare raw no-server request files against the 0.2.3 source tree that test_no_mcp_release_comparison pins.

Run with --baseline /path/to/33c6c51 (0.2.3) --candidate /path/to/this/tree.
Both runs use the same temporary repository and run-directory path in sequence.
The candidate pins the settings whose defaults changed after 0.2.3 (pytest runner, all tests
protected, stable tool head off), so the request files are expected to be byte-identical;
pass --current-defaults to see how today's defaults differ instead.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from stateframe.request_records import list_request_records, read_request_record


def source_env(tree: Path, stable_tool_head: str | None, *,
               source_on_pythonpath: bool = True) -> dict[str, str]:
    environment = dict(os.environ)
    if source_on_pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join((str(tree / "src"),
            str(tree / "src/stateframe/engine")))
    else:
        environment.pop("PYTHONPATH", None)
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if stable_tool_head is not None:
        environment["STATEFRAME_STABLE_TOOL_HEAD"] = stable_tool_head
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
            repo: Path, run_dir: Path, *, legacy_mode: bool = False,
            stable_tool_head: str | None,
            source_on_pythonpath: bool = True,
            test_python: str | None = None) -> bytes:
    # stable_tool_head has no default (see tools/changed_defaults.py): every caller states its arm.
    environment = source_env(tree, stable_tool_head=stable_tool_head,
                             source_on_pythonpath=source_on_pythonpath)
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
            "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", test_python or python,
            "--test-cmd", "python -m unittest discover -s tests -q", "--max-turns", "32",
            "--no-run-env", "--quiet",
            *(("--runner-kind", "pytest", "--protect-tests", "all") if legacy_mode else ())]
        result = subprocess.run(command, cwd=tree, env=environment, capture_output=True,
                                text=True, timeout=300, check=False)
        # The comparison is about request bytes, so a run whose checks fail still counts;
        # only a run that never wrote its summary or requests is a capture failure.
        names = list_request_records(run_dir)
        if not names or not (run_dir / "summary.json").is_file():
            raise RuntimeError(f"{tree}: run produced no requests or summary ({result.returncode}):\n"
                               f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        return b"\n".join(read_request_record(run_dir, name) for name in names)
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
                        help="untouched source tree of 0.2.3 (33c6c51)")
    parser.add_argument("--candidate", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter that runs the example's tests (each tree runs under its own)")
    parser.add_argument("--current-defaults", action="store_true",
                        help="run the candidate with today's defaults instead of the pinned 0.2.3 settings")
    args = parser.parse_args()
    baseline = args.baseline.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    script = candidate / "examples/discount_bug.script.json"
    # The engine child drops PYTHONPATH, so a shared interpreter would import whichever stateframe it has
    # installed. Give each tree its own interpreter and prove what it imports before comparing bytes.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from acceptance._support import assert_tree_imports, tree_python
    with tempfile.TemporaryDirectory(prefix="sf-no-mcp-diff-") as temporary:
        root = Path(temporary)
        repo = root / "repo"
        shutil.copytree(candidate / "examples/discount_bug", repo)
        launcher = root / "fixed_seed_run.py"
        write_launcher(launcher)
        run_dir = root / "run"
        pythons = {}
        for label, tree in (("baseline", baseline), ("candidate", candidate)):
            pythons[label] = tree_python(tree / "src", root / f"{label}-python")
            assert_tree_imports(pythons[label], tree / "src", root)
        previous = capture(baseline, pythons["baseline"], launcher, script, repo, run_dir,
                           stable_tool_head=None, source_on_pythonpath=False, test_python=args.python)
        run_dir.rename(root / "baseline-run")
        current = capture(candidate, pythons["candidate"], launcher, script, repo, run_dir,
                          legacy_mode=not args.current_defaults,
                          stable_tool_head=None if args.current_defaults else "off",
                          source_on_pythonpath=False, test_python=args.python)
        for label, body in (("baseline", previous), ("candidate", current)):
            print(f"{label}: {len(body)} bytes sha256={hashlib.sha256(body).hexdigest()}")
        if previous == current:
            print("Model request files are byte-identical.")
            return 0
        before = previous.decode(errors="replace").splitlines(keepends=True)
        after = current.decode(errors="replace").splitlines(keepends=True)
        sys.stdout.writelines(difflib.unified_diff(before, after, fromfile="baseline",
                                                   tofile="candidate", n=3))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
