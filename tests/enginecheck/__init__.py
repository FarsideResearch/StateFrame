"""Model-free engine runs: a declared scenario drives the scripted endpoint and returns the wire and journal.

A scenario is the only input: its files become a fresh git repo, its steps script the model side, and the
engine runs as a subprocess against the in-process scripted endpoint. Every run is checked against the
scenario's needs so a run that did not exercise what it claims fails loudly instead of passing vacuously.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from _journal import measurement_events
from _stateframe_home import isolated_stateframe_home
from engine_run import run_monitored
from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[2]
_GIT_IDENTITY = ("-c", "user.name=enginecheck", "-c", "user.email=enginecheck@localhost",
                 "-c", "commit.gpgsign=false")
_ENGINE_TIMEOUT_SECONDS = 600


class VacuousScenario(AssertionError):
    """A scenario ran but one of its declared needs did not hold."""


class HarnessError(RuntimeError):
    """The engine run could not be driven to a provider request."""


@dataclass(frozen=True)
class Need:
    description: str
    check: Callable[["Trace"], bool]


@dataclass(frozen=True)
class Scenario:
    files: dict[str, bytes]
    task: str
    steps: list[dict]
    switches: dict[str, str] = field(default_factory=dict)
    max_turns: int = 6
    needs: list[Need] = field(default_factory=list)


@dataclass(frozen=True)
class Trace:
    requests: list[bytes]
    request_json: list[dict]
    events: list[dict]
    exit_code: int
    switches: dict[str, str]
    run_dir: Path
    repo: Path


def _write_repo(repo: Path, files: dict[str, bytes]) -> None:
    """Materialize files byte-exact in a fresh git repo with one commit."""
    repo.mkdir(parents=True)
    for name, content in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, *_GIT_IDENTITY, "commit", "-q", "-m", "scenario fixture")


def _git(repo: Path, *args: str) -> None:
    completed = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise HarnessError(f"git {args[0]} failed in scenario repo: {completed.stderr.strip()}")


def _engine_environment(switches: dict[str, str]) -> dict[str, str]:
    """Strip ambient engine switches, pin hash seed and home, then apply this scenario's switches."""
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("STATEFRAME_", "REACT_"))}
    environment.update({
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATEFRAME_HOME": isolated_stateframe_home(),
        # Match the scripted-parity prior art: resolve tools from the explicit PATH only.
        "PATH": os.defpath,
    })
    environment.update(switches)
    return environment


def _applied_switches(environment: dict[str, str]) -> dict[str, str]:
    """The STATEFRAME_* switches the engine subprocess actually received (STATEFRAME_HOME is harness plumbing)."""
    return {name: value for name, value in environment.items()
            if name.startswith("STATEFRAME_") and name != "STATEFRAME_HOME"}


def _request_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "wire").glob("*.request.json"), key=lambda p: int(p.name.split(".")[0]))


def run(scenario: Scenario, tmp_path: Path) -> Trace:
    """Run one scenario model-free and return its wire requests and journal; raise if a need fails."""
    case = tmp_path / "case"
    repo = case / "repo"
    run_dir = case / "run"
    _write_repo(repo, scenario.files)

    script = Script({"model": "scripted", "steps": scenario.steps, "tokenize": {"mode": "ok"}})
    endpoint = serve(script)
    arguments = [
        sys.executable, "-m", "stateframe", "run",
        "--repo", str(repo), "--task", scenario.task,
        "--endpoint", f"http://127.0.0.1:{endpoint.server_port}/v1",
        "--model", "scripted", "--run-dir", str(run_dir), "--quiet", "--json",
        "--max-turns", str(scenario.max_turns), "--test-python", sys.executable,
    ]
    try:
        environment = _engine_environment(scenario.switches)
        try:
            completed = run_monitored(arguments, env=environment,
                                      expected_seconds=_ENGINE_TIMEOUT_SECONDS, run_dir=run_dir,
                                      step="enginecheck scenario", check=False)
        except OSError as exc:
            raise HarnessError(f"could not start the engine: {exc}") from exc
    finally:
        endpoint.shutdown()
        endpoint.server_close()

    paths = _request_paths(run_dir)
    if not paths:
        raise HarnessError(
            f"engine exited {completed.returncode} without sending a provider request: "
            f"{completed.stdout[-2000:]}{completed.stderr[-2000:]}")
    try:
        script.assert_consumed()
    except AssertionError as exc:
        raise HarnessError(f"scripted steps were not consumed: {exc}") from exc

    requests = [path.read_bytes() for path in paths]
    trace = Trace(
        requests=requests,
        request_json=[json.loads(raw) for raw in requests],
        events=measurement_events(run_dir),
        exit_code=completed.returncode,
        switches=_applied_switches(environment),
        run_dir=run_dir,
        repo=repo,
    )
    for need in scenario.needs:
        if not need.check(trace):
            raise VacuousScenario(need.description)
    return trace


def run_pair(scenario: Scenario, tmp_path: Path, switch_env: dict[str, str]) -> tuple[Trace, Trace]:
    """Run the scenario with switch_env absent (off) and applied on top of its switches (on), in separate dirs."""
    off = run(replace(scenario), tmp_path / "off")
    on = run(replace(scenario, switches={**scenario.switches, **switch_env}), tmp_path / "on")
    return off, on
