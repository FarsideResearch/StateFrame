"""Run the public CLI against a real scripted endpoint and inspect durable artifacts."""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PYTHON = sys.executable


@dataclass(frozen=True)
class Case:
    repo: Path
    run_dir: Path
    process: subprocess.CompletedProcess[str]
    summary: dict[str, object] | None
    requests: tuple[dict, ...]

    @property
    def transcript(self) -> str:
        return "\n".join(json.dumps(request, sort_keys=True) for request in self.requests)

    @property
    def diff(self) -> str:
        path = self.run_dir / "final.diff"
        return path.read_text() if path.exists() else ""

    def assert_reply(self, fragment: str) -> None:
        assert fragment in self.transcript, (
            f"missing model-visible {fragment!r}; stdout={self.process.stdout!r}; "
            f"stderr={self.process.stderr!r}; summary={self.summary!r}")

    def assert_offered(self, tool: str, *, request: int = 0) -> None:
        offered = {entry["function"]["name"] for entry in self.requests[request].get("tools", [])}
        assert tool in offered, (tool, offered)

    def assert_not_offered(self, tool: str) -> None:
        assert all(tool not in {entry["function"]["name"] for entry in request.get("tools", [])}
                   for request in self.requests)


def environment() -> dict[str, str]:
    # Acceptance runs must observe the packaged profile, not a caller's switch overrides.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("STATEFRAME_")}
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"),
                                        str(ROOT / "src" / "stateframe" / "engine")))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def copy_repo(tmp_path: Path, name: str = "simple_repo") -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURES / name, repo, symlinks=True)
    return repo


def script_path(tmp_path: Path, name: str, substitutions: dict[str, str] | None = None) -> Path:
    source = FIXTURES / "scripts" / name
    if not substitutions:
        return source
    document = json.loads(source.read_text())

    def replace(value: object) -> object:
        if isinstance(value, str):
            for key, replacement in substitutions.items():
                if value == "${" + key + "}" and key == "PORT":
                    return int(replacement)
                value = value.replace("${" + key + "}", replacement)
            return value
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    target = tmp_path / "case.script.json"
    target.write_text(json.dumps(replace(document)))
    return target


class Endpoint:
    def __init__(self, script: Path, *, force_missing_tool: bool = False):
        command = ([PYTHON, str(FIXTURES / "force_endpoint.py")]
                   if force_missing_tool else [PYTHON, "-m", "stateframe.scripted_endpoint"])
        self.process = subprocess.Popen(
            [*command, "--script", str(script), "--port", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment())
        assert self.process.stdout is not None
        ready, _, _ = select.select([self.process.stdout], [], [], 15)
        if not ready:
            self.close()
            raise AssertionError("scripted endpoint did not start within 15 seconds")
        self.url = self.process.stdout.readline().strip()
        assert self.url.startswith("http://127.0.0.1:") and self.url.endswith("/v1"), self.url

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def __enter__(self) -> "Endpoint":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_requests(run_dir: Path) -> tuple[dict, ...]:
    return tuple(json.loads(path.read_text())
                 for path in sorted((run_dir / "requests").glob("*.json")))


def run_case(tmp_path: Path, script: str, *, repo_name: str = "simple_repo",
             substitutions: dict[str, str] | None = None, flags: tuple[str, ...] = (),
             repo: Path | None = None, task: str = "Exercise the requested tool and verify the result.",
             timeout: int = 240, force_missing_tool: bool = False,
             default_test_cmd: bool = True,
             env_overrides: dict[str, str] | None = None) -> Case:
    repo = repo or copy_repo(tmp_path, repo_name)
    run_dir = tmp_path / "run"
    script = str(script_path(tmp_path, script, substitutions))
    with Endpoint(Path(script), force_missing_tool=force_missing_tool) as endpoint:
        process = subprocess.run(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", task,
             "--endpoint", endpoint.url, "--model", "scripted", "--run-dir", str(run_dir),
             "--test-python", PYTHON,
             *(["--test-cmd", "python -m unittest discover -s tests -q"] if default_test_cmd else []),
             "--max-turns", "32",
             "--quiet", *flags], capture_output=True, text=True,
            env={**environment(), **(env_overrides or {})},
            timeout=timeout, check=False)
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text()) if path.is_file() else None
    return Case(repo, run_dir, process, summary, read_requests(run_dir))
