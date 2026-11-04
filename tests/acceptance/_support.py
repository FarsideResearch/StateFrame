"""Run the public CLI against a real scripted endpoint and inspect durable artifacts."""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import select
import shutil
import site
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

from _stateframe_home import isolated_stateframe_home
from stateframe.request_records import list_request_records, read_request_record
from engine_run import run_monitored

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PYTHON = sys.executable
_PASS_ENV = ("PATH", "HOME", "LANG", "TMPDIR")

# Slowest stateframe subprocess wall time per case, in seconds. The acceptance cases use
# 3 reps on the slowest gate host (2x Xeon E5-2680 v1 container, 1 worker; 850db00 at
# load5 61-70 and f8e8925 at load5 70-90, 2026-11-03). The recorded blocked case uses
# Mike's completed 122.9s host run on 2026-11-03; remeasure it after the host can run
# nested sandbox-exec. Gates rarely run solo, so the loaded maximum is the reference.
# Re-measure when a case changes; tests/acceptance/test_acceptance_budgets.py checks that
# every case is used and that measured callers carry no bare deadline literal.
MEASURED_SECONDS = {
    "e19_resume": 307,
    "no_skills_run": 193,
    "recorded_blocked_run": 123,
}
HEADROOM = 1.5


def budget(case: str) -> int:
    """Progress-idle expectation for a measured case: HEADROOM x its slowest wall time."""
    return math.ceil(HEADROOM * MEASURED_SECONDS[case])


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
    env["STATEFRAME_HOME"] = isolated_stateframe_home()  # never the real ~/.stateframe (env seeds pile up there)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"),
                                        str(ROOT / "src" / "stateframe" / "engine")))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def pass_environment() -> dict[str, str]:
    """Return only variables the engine child is allowed to inherit for import checks."""
    return {key: os.environ[key] for key in _PASS_ENV if key in os.environ}


def _copy_startup_pytest(site_packages: Path) -> None:
    """Keep pytest's startup imports inside the interpreter prefix admitted by probes."""
    for name in ("pytest", "pluggy", "iniconfig", "packaging", "pygments"):
        distribution = importlib.metadata.distribution(name)
        for relative in distribution.files or ():
            if relative.is_absolute() or ".." in relative.parts or relative.suffix in {".pyc", ".pth"}:
                continue
            source = Path(distribution.locate_file(relative))
            if source.is_file():
                target = site_packages / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def tree_python(tree_src: Path, work: Path) -> str:
    """Create a Python that imports one source tree and shares only installed packages."""
    tree_src = Path(tree_src).resolve()
    if not tree_src.is_dir():
        raise ValueError(f"tree source directory does not exist: {tree_src}")

    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    venv = work / "python-venv"
    if venv.exists():
        raise FileExistsError(f"tree Python environment already exists: {venv}")

    base_python = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
    subprocess.run([base_python, "-m", "venv", "--without-pip", str(venv)],
                   check=True, env=pass_environment())
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site_result = subprocess.run(
        [str(python), "-B", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        check=True, capture_output=True, text=True, env=pass_environment())
    venv_site = Path(site_result.stdout.strip()).resolve()
    if not venv_site.is_dir():
        raise RuntimeError(f"tree Python site-packages directory is missing: {venv_site}")
    _copy_startup_pytest(venv_site)

    current_paths = [*site.getsitepackages()]
    configured_paths = sysconfig.get_paths()
    current_paths.extend(configured_paths.get(name) for name in ("purelib", "platlib"))
    import_paths = [tree_src]
    seen = {tree_src}
    for candidate in current_paths:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if path.is_dir() and path not in seen:
            import_paths.append(path)
            seen.add(path)

    if any("\n" in str(path) or "\r" in str(path) for path in import_paths):
        raise ValueError("Python import paths cannot contain line breaks")
    # Plain path entries expose dependencies without processing their editable-install .pth files.
    with (venv_site / "stateframe-tree-paths.pth").open("x", encoding="utf-8") as stream:
        stream.writelines(f"{path}\n" for path in import_paths)
    # Keep the venv path: resolving its symlink can make Python ignore the venv prefix.
    return str(python)


def assert_tree_imports(python: str, tree_src: Path, cwd: Path) -> None:
    """Fail unless an isolated interpreter imports StateFrame from its source tree."""
    tree_src = Path(tree_src).resolve()
    cwd = Path(cwd).resolve()
    tree_root = tree_src.parent
    assert not cwd.is_relative_to(tree_root), (cwd, tree_root)
    preflight = subprocess.run(
        [python, "-B", "-c", "import stateframe.workcopy; print(stateframe.workcopy.__file__)"],
        capture_output=True, text=True, cwd=cwd, env=pass_environment(), check=False)
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    imported = Path(preflight.stdout.strip()).resolve()
    assert imported.is_relative_to(tree_src), (imported, tree_src)


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
    return tuple(json.loads(read_request_record(run_dir, name))
                 for name in list_request_records(run_dir))


def read_measurement_events(run_dir: Path) -> tuple[dict, ...]:
    import kernel_measurement as measurement

    rows = []
    for path in measurement.journal_paths(run_dir / "worker" / "measurement-runs"):
        with measurement.open_journal(path) as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return tuple(rows)


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
        process = run_monitored(
            [PYTHON, "-m", "stateframe", "run", "--repo", str(repo), "--task", task,
             "--endpoint", endpoint.url, "--model", "scripted", "--run-dir", str(run_dir),
             "--test-python", PYTHON,
             *(["--test-cmd", "python -m unittest discover -s tests -q"] if default_test_cmd else []),
             "--max-turns", "32",
             "--quiet", *flags], capture_output=True, text=True,
            env={**environment(), **(env_overrides or {})},
            expected_seconds=timeout, check=False)
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text()) if path.is_file() else None
    return Case(repo, run_dir, process, summary, read_requests(run_dir))
