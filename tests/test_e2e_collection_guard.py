"""Keep top-level e2e tests collectible while fixture repositories stay data."""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import pytest

from _timeouts import scaled
from _stateframe_home import isolated_stateframe_home
from fixtures.process_signals import monitored_group_exists, signal_run_owned


pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]
E2E_TESTS = Path("tests/e2e")
COLLECTION_QUIET_PERIOD = 60.0
COLLECTION_POLL_INTERVAL = 0.1


class CollectionWatchdogError(AssertionError):
    def __init__(self, process_group_id: int, detail: str):
        self.process_group_id = process_group_id
        super().__init__(f"pytest --collect-only child process group {process_group_id} {detail}")


def _linux_proc_available() -> bool:
    return sys.platform == "linux" and Path("/proc/self/stat").is_file()


def _linux_group_snapshot(process_group_id: int) -> tuple[int, bool] | None:
    """Return group CPU ticks and whether any member is running, or None if /proc is unavailable."""
    try:
        processes = os.scandir("/proc")
    except OSError:
        return None

    cpu_ticks = 0
    has_running_process = False
    try:
        with processes:
            for entry in processes:
                if not entry.name.isdecimal():
                    continue
                try:
                    stat = Path(entry.path, "stat").read_text(encoding="ascii")
                    fields = stat[stat.rfind(")") + 2 :].split()
                    if len(fields) < 13 or int(fields[2]) != process_group_id:
                        continue
                    cpu_ticks += int(fields[11]) + int(fields[12])
                    has_running_process |= fields[0] not in {"Z", "X"}
                except (OSError, ValueError, IndexError):
                    # Processes can exit while /proc is being scanned.
                    continue
    except OSError:
        return None
    return cpu_ticks, has_running_process


def _process_group_is_alive(process_group_id: int, process: subprocess.Popen | None = None) -> bool:
    if _linux_proc_available():
        snapshot = _linux_group_snapshot(process_group_id)
        if snapshot is not None:
            return snapshot[1]
    if os.name == "posix":
        return monitored_group_exists(process_group_id)
    return process is not None and process.poll() is None


def _signal_collection_group(process: subprocess.Popen, signum: int) -> None:
    try:
        signal_run_owned(
            Path.cwd(),
            process,
            signum,
            deadline=time.monotonic(),
            what="pytest collection watchdog cleanup",
            require_checkpoint_saved=False,
            group=os.name == "posix",
        )
    except ProcessLookupError:
        pass


def _stop_collection_process(process: subprocess.Popen) -> None:
    """Stop the isolated child group and reap the collection process."""
    _signal_collection_group(process, signal.SIGKILL)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _signal_collection_group(process, signal.SIGKILL)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"collection process {process.pid} did not exit after SIGKILL") from None

    if _linux_proc_available():
        cleanup_deadline = time.monotonic() + 2
        while _process_group_is_alive(process.pid, process) and time.monotonic() < cleanup_deadline:
            time.sleep(0.02)
        if _process_group_is_alive(process.pid, process):
            raise RuntimeError(f"collection process group {process.pid} remained after SIGKILL")


def _decode_output(output: bytearray) -> str:
    return bytes(output).decode(errors="replace")


def _run_linux_collection(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    quiet_period: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=0,
    )
    selector = selectors.DefaultSelector()
    output = {"stdout": bytearray(), "stderr": bytearray()}
    for name in output:
        stream = getattr(process, name)
        assert stream is not None
        selector.register(stream, selectors.EVENT_READ, data=name)

    started = time.monotonic()
    last_progress = started
    stopped = False
    try:
        snapshot = _linux_group_snapshot(process.pid)
        if snapshot is None:
            _stop_collection_process(process)
            stopped = True
            raise RuntimeError("cannot read process CPU usage from /proc for collection watchdog")
        last_cpu_ticks = snapshot[0]
        while True:
            if selector.get_map():
                events = selector.select(timeout=COLLECTION_POLL_INTERVAL)
                for key, _ in events:
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if chunk:
                        output[key.data].extend(chunk)
                        last_progress = time.monotonic()
                    else:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
            else:
                time.sleep(COLLECTION_POLL_INTERVAL)

            if process.poll() is not None and not selector.get_map():
                process.wait()
                if _process_group_is_alive(process.pid, process):
                    _stop_collection_process(process)
                return subprocess.CompletedProcess(
                    command,
                    process.returncode,
                    _decode_output(output["stdout"]),
                    _decode_output(output["stderr"]),
                )

            now = time.monotonic()
            snapshot = _linux_group_snapshot(process.pid)
            if snapshot is None:
                _stop_collection_process(process)
                stopped = True
                raise RuntimeError("cannot read process CPU usage from /proc for collection watchdog")
            if snapshot[0] > last_cpu_ticks:
                last_cpu_ticks = snapshot[0]
                last_progress = now
            if now - last_progress >= quiet_period:
                _stop_collection_process(process)
                stopped = True
                raise CollectionWatchdogError(
                    process.pid,
                    f"made no progress for {quiet_period:g} seconds; stopped its process group",
                )
    finally:
        selector.close()
        if not stopped and process.poll() is None:
            _stop_collection_process(process)
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _run_fallback_collection(
    command: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_collection_process(process)
        raise CollectionWatchdogError(
            process.pid,
            f"exceeded the scaled wall limit of {timeout:g} seconds; stopped its process group",
        ) from None
    if _process_group_is_alive(process.pid, process):
        _stop_collection_process(process)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run_collection(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    quiet_period: float,
    fallback_timeout: float,
) -> subprocess.CompletedProcess[str]:
    if quiet_period <= 0 or fallback_timeout <= 0:
        raise ValueError("collection watchdog limits must be greater than zero")
    if _linux_proc_available():
        return _run_linux_collection(
            command,
            cwd=cwd,
            env=env,
            quiet_period=quiet_period,
        )
    return _run_fallback_collection(command, cwd=cwd, env=env, timeout=fallback_timeout)


def _e2e_python_sources(root: Path) -> list[str]:
    e2e_root = root / E2E_TESTS
    return [path.relative_to(root).as_posix() for path in e2e_root.rglob("*.py")]


def _fixture_repo_conftests(root: Path, python_sources: list[str] | None = None) -> list[str]:
    if python_sources is None:
        python_sources = _e2e_python_sources(root)
    return sorted(
        path
        for path in python_sources
        if PurePosixPath(path).name == "conftest.py" and PurePosixPath(path).parent != E2E_TESTS
    )


def _assert_e2e_collection(
    root: Path, *, quiet_period: float = COLLECTION_QUIET_PERIOD
) -> tuple[str, ...]:
    """Collect the suite in a fresh process and enforce the e2e/fixture boundary."""
    python_sources = _e2e_python_sources(root)
    fixture_conftests = _fixture_repo_conftests(root, python_sources)
    if fixture_conftests:
        raise AssertionError(
            "fixture-repository conftest.py files found under tests/e2e: "
            + ", ".join(fixture_conftests)
        )

    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATEFRAME_HOME": isolated_stateframe_home(),
    }
    # Fix root-relative node IDs in synthetic projects while preserving full suite collection.
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(root),
        "tests",
    ]
    result = _run_collection(
        command,
        cwd=root,
        env=environment,
        quiet_period=quiet_period,
        fallback_timeout=scaled(120),
    )
    node_ids = tuple(
        line.strip()
        for line in result.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    )
    expected_files = sorted(
        path
        for path in python_sources
        if PurePosixPath(path).parent == E2E_TESTS and PurePosixPath(path).name.startswith("test_")
    )
    collected_files = {node_id.split("::", 1)[0] for node_id in node_ids}
    missing_files = [path for path in expected_files if path not in collected_files]
    nested_node_ids = [
        node_id
        for node_id in node_ids
        if node_id.split("::", 1)[0].startswith("tests/e2e/")
        and len(PurePosixPath(node_id.split("::", 1)[0]).parts) > 3
    ]
    output = result.stdout + result.stderr
    nested_sources = [path for path in python_sources if PurePosixPath(path).parent != E2E_TESTS]
    imported_nested_sources = [path for path in nested_sources if path in output]

    problems = []
    if missing_files:
        problems.append(f"top-level e2e test files without node ids: {missing_files}")
    if nested_node_ids:
        problems.append(f"fixture-repository node ids were collected: {nested_node_ids}")
    if imported_nested_sources:
        problems.append(f"fixture-repository Python sources appeared during collection: {imported_nested_sources}")
    if problems:
        summary = next(
            (line.strip() for line in reversed(output.splitlines()) if " collected" in line),
            "collection summary unavailable",
        )
        raise AssertionError(
            f"e2e collection contract failed (pytest exit {result.returncode}; {summary}): "
            + "; ".join(problems)
        )
    return node_ids


def _write_planted_tree(root: Path, ignore_glob: str) -> Path:
    tests = root / "tests"
    e2e = tests / "e2e"
    repo = e2e / "repo"
    repo.mkdir(parents=True)
    (tests / "conftest.py").write_text(
        f"collect_ignore_glob = [{ignore_glob!r}]\n",
        encoding="utf-8",
    )
    (e2e / "test_x.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    imported = repo / "imported.txt"
    (repo / "test_inner.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('imported', encoding='utf-8')\n"
        "def test_inner():\n"
        "    pass\n",
        encoding="utf-8",
    )
    return imported


def test_full_suite_collects_e2e_modules_and_ignores_fixture_repositories():
    _assert_e2e_collection(ROOT)


@pytest.mark.skipif(not _linux_proc_available(), reason="the progress watchdog uses Linux process accounting")
def test_collection_progress_can_outlast_the_scaled_wall_limit(pytester, monkeypatch):
    e2e = pytester.path / "tests/e2e"
    e2e.mkdir(parents=True)
    finished = pytester.path / "collection-finished"
    (e2e / "test_slow.py").write_text(
        "import time\n"
        "from pathlib import Path\n"
        "end = time.process_time() + 0.35\n"
        "while time.process_time() < end:\n"
        "    pass\n"
        f"Path({str(finished)!r}).write_text('done', encoding='utf-8')\n"
        "def test_slow():\n"
        "    pass\n",
        encoding="utf-8",
    )
    # Simulate the old scaled(120) deadline without waiting 120 seconds.
    monkeypatch.setattr(sys.modules[__name__], "scaled", lambda seconds: 0.1 if seconds == 120 else seconds)

    started = time.monotonic()
    node_ids = _assert_e2e_collection(pytester.path, quiet_period=5.0)

    assert time.monotonic() - started > 0.1
    assert finished.read_text(encoding="utf-8") == "done"
    assert any(node_id.startswith("tests/e2e/test_slow.py::") for node_id in node_ids)


@pytest.mark.skipif(not _linux_proc_available(), reason="the progress watchdog uses Linux process accounting")
def test_collection_watchdog_stops_sleeping_child_and_leaves_no_process(tmp_path):
    with pytest.raises(CollectionWatchdogError, match="made no progress.*stopped its process group") as error:
        _run_collection(
            [sys.executable, "-c", "import time; time.sleep(3600)"],
            cwd=tmp_path,
            env=dict(os.environ),
            quiet_period=0.2,
            fallback_timeout=2.0,
        )

    assert not _process_group_is_alive(error.value.process_group_id)


def test_collection_uses_scaled_wall_limit_when_proc_is_unavailable(tmp_path, monkeypatch):
    marker = tmp_path / "child-started"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import time; "
        f"Path({str(marker)!r}).write_text('started', encoding='utf-8'); time.sleep(3600)",
    ]
    monkeypatch.setattr(sys.modules[__name__], "_linux_proc_available", lambda: False)
    monkeypatch.setattr(sys.modules[__name__], "scaled", lambda seconds: 0.5)

    with pytest.raises(CollectionWatchdogError, match="exceeded the scaled wall limit") as error:
        _run_collection(
            command,
            cwd=tmp_path,
            env=dict(os.environ),
            quiet_period=60.0,
            fallback_timeout=scaled(120),
        )

    assert marker.read_text(encoding="utf-8") == "started"
    assert not _process_group_is_alive(error.value.process_group_id)


def test_collection_guard_detects_the_old_glob_and_accepts_the_fixed_glob(pytester):
    imported = _write_planted_tree(pytester.path, "e2e/*")

    with pytest.raises(AssertionError, match=r"tests/e2e/test_x\.py"):
        _assert_e2e_collection(pytester.path)
    assert not imported.exists()

    (pytester.path / "tests/conftest.py").write_text(
        'collect_ignore_glob = ["e2e/*/*"]\n',
        encoding="utf-8",
    )
    node_ids = _assert_e2e_collection(pytester.path)
    assert any(node_id.startswith("tests/e2e/test_x.py::") for node_id in node_ids)
    assert not any("tests/e2e/repo/" in node_id for node_id in node_ids)
    assert not imported.exists()

    conftest = pytester.path / "tests/e2e/repo/conftest.py"
    conftest.write_text(
        "from pathlib import Path\n"
        f"Path({str(imported)!r}).write_text('imported', encoding='utf-8')\n",
        encoding="utf-8",
    )
    assert _fixture_repo_conftests(pytester.path) == ["tests/e2e/repo/conftest.py"]
    with pytest.raises(AssertionError, match=r"tests/e2e/repo/conftest\.py"):
        _assert_e2e_collection(pytester.path)
    assert not imported.exists()

    conftest.unlink()
    assert _fixture_repo_conftests(pytester.path) == []
    _assert_e2e_collection(pytester.path)
    assert not imported.exists()
