"""Capture raw request files for the shipped examples with mutations off.

The same repository copy and run-directory path are used for the baseline and the candidate in
sequence, so request bytes can be compared exactly. Mirrors compare_no_mcp_requests.capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys

from stateframe.request_records import list_request_records, read_request_record

EXAMPLES = {
    "discount_bug": ("Fix the pricing code so the tests pass.",
                     "python -m pytest -q tests"),
    "tiny_site": ("Build a small static shop site in site/ and make tests/test_site.py pass.",
                  "python -m pytest -q tests/test_site.py"),
    "brief_to_report": ("Read input/brief.md and input/tickets.csv and write report/summary.json "
                        "and report/report.md so tests/test_report.py passes.",
                        "python -m pytest -q tests/test_report.py"),
}


def source_env(tree: Path, *, stable_tool_head: str) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_MUTATION")}
    environment["PYTHONPATH"] = os.pathsep.join((str(tree / "src"),
                                                 str(tree / "src/stateframe/engine")))
    environment["STATEFRAME_STABLE_TOOL_HEAD"] = stable_tool_head
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def write_launcher(path: Path) -> None:
    """Fixed token stream so both trees allocate identical run identifiers."""
    path.write_text("import runpy, secrets\n"
                    "_sequence = 0\n"
                    "def fixed_hex(size=32):\n"
                    "    global _sequence\n"
                    "    _sequence += 1\n"
                    "    return format(_sequence, 'x').zfill(size * 2)[-size * 2:]\n"
                    "secrets.token_hex = fixed_hex\n"
                    "runpy.run_module('stateframe', run_name='__main__')\n")


def capture(tree: Path, python: str, launcher: Path, example: str, repo: Path,
            run_dir: Path, *, stable_tool_head: str,
            resume: bool = False) -> tuple[bytes, list[list[str]], dict]:
    """Return (joined request bytes, tool-name order per request, spec.json)."""
    task, test_cmd = EXAMPLES[example]
    if repo.exists():
        shutil.rmtree(repo)
    shutil.copytree(tree / "examples" / example, repo)
    environment = source_env(tree, stable_tool_head=stable_tool_head)
    endpoint = subprocess.Popen([python, "-m", "stateframe.scripted_endpoint",
        "--script", str(tree / "examples" / f"{example}.script.json"), "--port", "0"],
        cwd=tree, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert endpoint.stdout is not None
        ready, _, _ = select.select([endpoint.stdout], [], [], 15)
        if not ready:
            raise RuntimeError("scripted endpoint did not start")
        url = endpoint.stdout.readline().strip()
        command = [python, str(launcher), "run", "--repo", str(repo), "--task", task,
                   "--endpoint", url, "--model", "scripted", "--run-dir", str(run_dir),
                   "--test-python", python, "--test-cmd", test_cmd,
                   "--max-turns", "1" if resume else "32",
                   "--no-run-env", "--quiet"]
        result = subprocess.run(command, cwd=tree, env=environment, capture_output=True,
                                text=True, timeout=600, check=False)
        initial_raw = []
        if resume:
            initial_raw = [read_request_record(run_dir, name)
                           for name in list_request_records(run_dir)]
            if not initial_raw or not (run_dir / "summary.json").is_file():
                raise RuntimeError(f"{tree}: no requests before resume:\n{result.stderr[-2000:]}")
            command = [python, str(launcher), "resume", str(run_dir), "--endpoint", url,
                       "--max-turns", "32", "--quiet"]
            result = subprocess.run(command, cwd=tree, env=environment, capture_output=True,
                                    text=True, timeout=600, check=False)
        # Finished runs may pack request records (requests.pack.xz); the reader handles both formats.
        records = list_request_records(run_dir) if run_dir.is_dir() else []
        if result.returncode != 0 or not records or not (run_dir / "summary.json").is_file():
            raise RuntimeError(f"{tree}: no requests or summary ({result.returncode}):\n"
                               f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        raw = initial_raw + [read_request_record(run_dir, name) for name in records]
        names = []
        for body in raw:
            tools = json.loads(body).get("tools") or []
            names.append([tool.get("function", {}).get("name", "") for tool in tools])
        spec = json.loads((run_dir / "spec.json").read_text()) if (run_dir / "spec.json").is_file() else {}
        return b"\n".join(raw), names, spec
    finally:
        endpoint.terminate()
        try:
            endpoint.wait(timeout=5)
        except subprocess.TimeoutExpired:
            endpoint.kill()
            endpoint.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="new directory for isolated captures and raw request artifacts")
    parser.add_argument("--examples", nargs="+", choices=tuple(EXAMPLES), default=list(EXAMPLES))
    parser.add_argument("--resume-example", choices=tuple(EXAMPLES), default="discount_bug",
                        help="also compare a one-turn run resumed with a larger turn budget")
    parser.add_argument("--stable-tool-head", choices=("on", "off"), default="on")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    launcher = work / "launcher.py"
    write_launcher(launcher)
    # Keep user-installed extensions out of both trees and all generated state in the capture.
    os.environ["STATEFRAME_HOME"] = str(work / "home")
    cases = [(example, False) for example in args.examples] + [(args.resume_example, True)]
    identical = True
    for example, resume in cases:
        case = example + ("-resume" if resume else "")
        case_dir = work / case
        case_dir.mkdir()
        captures = []
        for label, tree in (("baseline", args.baseline), ("candidate", args.candidate)):
            raw, names, _spec = capture(tree.resolve(), args.python, launcher, example,
                case_dir / "repo", case_dir / "run", stable_tool_head=args.stable_tool_head,
                resume=resume)
            (case_dir / f"{label}.requests.bin").write_bytes(raw)
            (case_dir / "run").rename(case_dir / f"{label}-run")
            captures.append(raw)
            print(f"{case} {label}: requests={len(names)} bytes={len(raw)} "
                  f"sha256={hashlib.sha256(raw).hexdigest()}", flush=True)
        match = captures[0] == captures[1]
        identical = identical and match
        print(f"{case}: byte-identical={match}", flush=True)
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
