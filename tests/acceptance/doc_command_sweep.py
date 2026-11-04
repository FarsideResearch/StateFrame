"""Clean-install sweep of runnable documentation commands and bundled examples."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from engine_run import run_monitored  # noqa: E402


@dataclass(frozen=True)
class ExampleCommand:
    label: str
    arguments: tuple[str, ...]
    example: str


def commands(source: Path) -> tuple[ExampleCommand, ...]:
    """Parse real run commands; reject every unmapped bundled example command."""
    found: list[ExampleCommand] = []
    for document in (source / "QUICKSTART.md", source / "README.md"):
        blocks = re.findall(r"(?ms)^```(?:bash|sh)\n(.*?)^```", document.read_text())
        for block in blocks:
            for line in block.replace("\\\n", " ").splitlines():
                line = line.strip()
                if not line.startswith("stateframe run "):
                    continue
                argv = shlex.split(line)
                if "--repo" not in argv:
                    raise AssertionError(f"{document.name}: run command has no --repo: {line}")
                repo = argv[argv.index("--repo") + 1]
                if not repo.startswith("examples/"):
                    if document.name == "README.md" and repo == "~/code/shop":
                        continue  # the documented sketch for a reader's own repository
                    raise AssertionError(f"{document.name}: unmapped run command: {line}")
                name = repo.removeprefix("examples/")
                if not (source / "examples" / name).is_dir() or not (
                    source / "examples" / f"{name}.script.json"
                ).is_file():
                    raise AssertionError(f"{document.name}: no bundled script for {line}")
                found.append(ExampleCommand(f"{document.name}: {line}", tuple(argv[1:]), name))
    scripted = {path.name.removesuffix(".script.json")
                for path in (source / "examples").glob("*.script.json")}
    for name in sorted(scripted - {item.example for item in found}):
        repo = source / "examples" / name
        readme = repo / "README.md"
        if not repo.is_dir() or not readme.is_file():
            raise AssertionError(f"example {name} has no runnable command or README")
        task = "Follow the example specification and make its checks pass:\n" + readme.read_text()
        found.append(ExampleCommand(
            f"{readme.relative_to(source)}: stateframe run --repo examples/{name} --task <README>",
            ("run", "--repo", f"examples/{name}", "--task", task), name,
        ))
    if not found:
        raise AssertionError("no runnable example commands were discovered")
    return tuple(found)


def outcome(requests: int, result: str) -> bool:
    return requests >= 1 and result == "SUBMITTED"


def sweep(source: Path, work: Path, *, cases: tuple[ExampleCommand, ...] | None = None
          ) -> list[tuple[str, int, str]]:
    cases = cases if cases is not None else commands(source)
    wheel_dir = work / "wheels"
    wheel_dir.mkdir(parents=True)
    build = run_monitored(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--wheel-dir", str(wheel_dir), str(source)],
        capture_output=True, text=True, expected_seconds=180,
        step="build StateFrame wheel", watch_paths=(wheel_dir,),
    )
    if build.returncode:
        raise AssertionError(build.stdout + build.stderr)
    wheels = list(wheel_dir.glob("stateframe-*.whl"))
    if len(wheels) != 1:
        raise AssertionError(f"expected one wheel, found {wheels}")
    environment_dir = work / "venv"
    virtualenv = run_monitored(
        [sys.executable, "-m", "venv", str(environment_dir)],
        capture_output=True, text=True, expected_seconds=180,
        step="create StateFrame example virtual environment",
        watch_paths=(environment_dir,),
    )
    if virtualenv.returncode:
        raise AssertionError(virtualenv.stdout + virtualenv.stderr)
    python = environment_dir / "bin/python"
    executable = environment_dir / "bin/stateframe"
    for arguments in (("--no-deps", str(wheels[0])), ("-r", str(source / "requirements.txt"))):
        install = run_monitored(
            [str(python), "-m", "pip", "install", *arguments],
            capture_output=True, text=True, expected_seconds=180,
            step="install StateFrame example environment", watch_paths=(executable,),
        )
        if install.returncode:
            raise AssertionError(install.stdout + install.stderr)
    home = work / "home"
    home.mkdir()
    environment = {**os.environ, "HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("PYTHONPATH", None)
    environment["PATH"] = os.environ.get("PATH", "")
    results = []
    for index, case in enumerate(cases):
        script = source / "examples" / f"{case.example}.script.json"
        server = subprocess.Popen(
            [str(python), "-m", "stateframe.scripted_endpoint", "--script", str(script),
             "--port", "0"], cwd=source, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        try:
            ready, _, _ = select.select([server.stdout], [], [], 20)
            if not ready:
                raise AssertionError(f"endpoint did not start for {case.label}")
            endpoint = server.stdout.readline().strip()
            if not endpoint.startswith("http://127.0.0.1:"):
                raise AssertionError(f"invalid endpoint for {case.label}: {endpoint}")
            run_dir = work / f"run-{index}"
            argv = list(case.arguments)
            argv[argv.index("--repo") + 1] = str(source / "examples" / case.example)
            process = run_monitored(
                [str(executable), *argv, "--endpoint", endpoint, "--model", "scripted",
                 "--run-dir", str(run_dir)], cwd=source, env=environment,
                capture_output=True, text=True, expected_seconds=300,
                run_dir=run_dir, step=case.label,
            )
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            requests = int(summary.get("provider_sends", 0))
            result = str(summary.get("engine_result", "MISSING"))
            results.append((case.label, requests, result))
            if process.returncode and outcome(requests, result):
                tail = (process.stderr or process.stdout or "").strip()[-600:]
                results[-1] = (case.label, requests, f"EXIT_{process.returncode}: {tail}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("work", type=Path)
    options = parser.parse_args()
    try:
        results = sweep(options.source, options.work)
    except Exception as error:
        print(f"FAIL: {error}", flush=True)
        return 1
    for label, requests, result in results:
        print(f"{label} | requests={requests} | result={result}", flush=True)
    passed = all(outcome(requests, result) for _, requests, result in results)
    print(f"{'PASS' if passed else 'FAIL'}: {len(results)} commands", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
