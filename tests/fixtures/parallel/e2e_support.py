"""Helpers for real shipped-profile subframe CLI runs."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from _journal import measurement_events, measurement_journal_paths
from engine_run import EngineProcess, run_monitored, start_monitored
from fixtures.process_signals import signal_run_owned

from stateframe.engine.kernel_parallel_records import ParallelResult

ROOT = Path(__file__).resolve().parents[3]
REQUIREMENTS = ROOT / "requirements.txt"
ALLOWED_SUBFRAME_TOOLS = frozenset({
    "read", "grep", "str_replace", "insert_before", "file_create", "run_tests", "done", "subframe_blocked",
})


def _clean_environment(*, home: Path | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    for name in tuple(environment):
        if name.startswith("STATEFRAME_"):
            environment.pop(name, None)
    if home is not None:
        home.mkdir(parents=True, exist_ok=True)
        environment["HOME"] = str(home)
    return environment


def create_engine_venv(destination: Path) -> Path:
    if not REQUIREMENTS.is_file():
        raise AssertionError(f"locked E2E requirements are missing: {REQUIREMENTS}")
    created = subprocess.run([sys.executable, "-I", "-m", "venv", str(destination)],
                             capture_output=True, text=True, timeout=90, check=False)
    assert created.returncode == 0, created.stdout + created.stderr
    python = destination / "bin" / "python"
    environment = _clean_environment()
    for command in (
        [str(python), "-I", "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", "--no-input",
         "--no-build-isolation", "--no-deps", str(ROOT)],
        [str(python), "-I", "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", "--no-input",
         "-r", str(REQUIREMENTS)],
    ):
        completed = subprocess.run(command, env=environment, capture_output=True, text=True,
                                   timeout=300, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    return python


def cli_command(python: Path, endpoint: str, repo: Path, run_dir: Path, task: str, *,
                parallel: str = "2", arguments: Iterable[str] = (),
                max_turns: int = 0, profile: str = "r2hc3") -> list[str]:
    return [str(python), "-I", "-m", "stateframe", "run", "--repo", str(repo), "--task", task,
            "--endpoint", endpoint, "--model", "scripted", "--profile", profile,
            "--run-dir", str(run_dir), "--max-turns", str(max_turns), "--parallel", parallel,
            "--python", str(python), "--test-python", str(python), "--shell", "off", "--json", *arguments]


def run_cli(python: Path, endpoint: str, repo: Path, run_dir: Path, task: str, *,
            parallel: str = "2", arguments: Iterable[str] = (), max_turns: int = 0,
            profile: str = "r2hc3",
            extra_environment: dict[str, str] | None = None,
            timeout: int = 900) -> subprocess.CompletedProcess[str]:
    environment = _clean_environment(home=run_dir.parent / "home")
    environment.update(extra_environment or {})
    return run_monitored(cli_command(python, endpoint, repo, run_dir, task, parallel=parallel,
                                     arguments=arguments, max_turns=max_turns, profile=profile),
                         cwd=ROOT, env=environment, run_dir=run_dir,
                         expected_seconds=timeout, step="parallel CLI run", check=False)


def start_cli(python: Path, endpoint: str, repo: Path, run_dir: Path, task: str, *,
              parallel: str = "2", arguments: Iterable[str] = (), max_turns: int = 0,
              profile: str = "r2hc3", extra_environment: dict[str, str] | None = None
              ) -> EngineProcess:
    environment = _clean_environment(home=run_dir.parent / "home")
    environment.update(extra_environment or {})
    return start_monitored(
        cli_command(python, endpoint, repo, run_dir, task, parallel=parallel,
                    arguments=arguments, max_turns=max_turns, profile=profile),
        cwd=ROOT, env=environment, run_dir=run_dir,
        expected_seconds=900, step="parallel CLI run",
    )


def start_resume(python: Path, endpoint: str, run_dir: Path, *,
                 extra_environment: dict[str, str] | None = None,
                 capture_output: bool = False) -> EngineProcess:
    # Keep the existing call signature; the monitor always captures diagnostic output.
    environment = _clean_environment(home=run_dir.parent / "home")
    environment.update(extra_environment or {})
    return start_monitored(
        [str(python), "-I", "-m", "stateframe", "resume", str(run_dir),
         "--endpoint", endpoint, "--model", "scripted", "--keep-checkpoints", "--json"],
        cwd=ROOT, env=environment, run_dir=run_dir,
        expected_seconds=900, step="parallel CLI resume",
    )


def _engine_lock_holder(run_dir: Path) -> int | None:
    lock_path = run_dir / ".engine.lock"
    if not lock_path.is_file():
        return None
    with lock_path.open("r+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            value = handle.read().strip()
            if not value.isdecimal() or int(value) <= 0:
                raise AssertionError(f"engine lock has no valid process ID: {value!r}")
            return int(value)
        fcntl.flock(handle, fcntl.LOCK_UN)
    return None


def interrupt_run(process: EngineProcess, run_dir: Path, *, require_checkpoint_saved: bool = True) -> None:
    """Stop the owned CLI and wait for its watched engine child to release the lock."""
    engine_pid = _engine_lock_holder(run_dir)
    assert engine_pid is not None, "engine lock was not held before interruption"
    assert engine_pid != process.pid, "engine must have a separate process session"
    assert process.poll() is None, f"CLI exited before interruption: pid={process.pid}"
    try:
        signal_run_owned(run_dir, process, signal.SIGTERM,
                         deadline=time.monotonic() + 30,
                         what="owned CLI interruption",
                         require_checkpoint_saved=require_checkpoint_saved)
    except ProcessLookupError:
        raise AssertionError(f"CLI exited during interruption: pid={process.pid}") from None
    try:
        exit_code = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"owned CLI did not exit after SIGTERM: pid={process.pid}; "
            f"engine_lock_holder={_engine_lock_holder(run_dir)}") from None
    assert exit_code != 0
    deadline = time.monotonic() + 30
    while _engine_lock_holder(run_dir) is not None:
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"engine lock remained held after CLI exit: cli_pid={process.pid}; "
                f"cli_exit={exit_code}; initial_holder={engine_pid}; "
                f"current_holder={_engine_lock_holder(run_dir)}")
        time.sleep(0.05)


def copy_example(tmp_path: Path, name: str = "parallel_fixes") -> Path:
    source = ROOT / "examples" / name
    destination = tmp_path / "repo"
    shutil.copytree(source, destination)
    for path in destination.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        mode = path.stat().st_mode
        path.chmod(0o755 if mode & 0o111 else 0o644)
    return destination


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict), f"expected JSON object at {path}"
    return value


def summary_for(run_dir: Path) -> dict[str, object]:
    return read_json(run_dir / "summary.json")


def _parent_run_dirs(run_dir: Path) -> list[Path]:
    attempts = run_dir / "attempts"
    roots = (sorted(attempts.glob("[0-9]*/worker/measurement-runs"),
                    key=lambda path: int(path.parents[1].name)) if attempts.is_dir() else [])
    return [*(root.parents[1] for root in roots), run_dir]


def parent_event_journals(run_dir: Path) -> list[Path]:
    return sorted(journal for root in _parent_run_dirs(run_dir)
                  for journal in measurement_journal_paths(root))


def parent_measurement_events(run_dir: Path) -> list[dict[str, object]]:
    return [event for root in _parent_run_dirs(run_dir) for event in measurement_events(root)]


def parallel_call_roots(run_dir: Path) -> list[Path]:
    return sorted(path for path in (run_dir / "parallel").iterdir()
                  if path.is_dir() and (path / "intent.json").is_file())


def wire_document(path: Path) -> dict[str, object]:
    return read_json(path)


def wire_message_text(path: Path) -> str:
    request = wire_document(path)
    messages = request.get("messages")
    assert isinstance(messages, list), f"wire request has no messages: {path}"
    return "\n".join(message.get("content", "") for message in messages
                     if isinstance(message, dict) and isinstance(message.get("content"), str))


def scenario_script(parent_steps: list[dict[str, object]],
                    subframe_steps: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {"parent": parent_steps, "subframes": subframe_steps}


def tool_step(tool: str, arguments: dict[str, object] | None = None, *, required: bool = True) -> dict[str, object]:
    return {"tool": tool, "arguments": arguments or {}, "required": required}


def crash_hook_arguments(phase: str, marker: Path) -> tuple[str, ...]:
    """Deliver crash settings through the run spec, not the filtered CLI environment."""
    return ("--test-env", f"SF_PARALLEL_E2E_CRASH={phase}",
            "--test-env", f"SF_PARALLEL_E2E_CRASH_MARKER={marker}")


def install_startup_hook(python: Path, module: str, source: str) -> Path:
    """Load a test hook even when the distribution supplies sitecustomize first."""
    if not module.isidentifier() or not module.isascii():
        raise AssertionError(f"invalid startup hook module: {module!r}")
    site_roots = sorted((python.parent.parent / "lib").glob("python*/site-packages"))
    if len(site_roots) != 1:
        raise AssertionError(f"expected one isolated site-packages directory, found {site_roots}")
    hook = site_roots[0] / f"{module}.py"
    files = ((hook, source), (hook.with_suffix(".pth"), f"import {module}\n"))
    for path, content in files:
        if path.exists() and (not path.is_file() or path.read_text(encoding="utf-8") != content):
            raise AssertionError(f"refusing to replace existing startup hook: {path}")
    for path, content in files:
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    return hook


def install_crash_hook(python: Path) -> Path:
    """Install an opt-in process-crash hook at the transaction durability seams."""
    hook_source = (
        "from __future__ import annotations\n"
        "import importlib.abc\n"
        "import importlib.machinery\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "class _CrashLoader(importlib.abc.Loader):\n"
        "    def __init__(self, wrapped, module_kind):\n"
        "        self.wrapped, self.module_kind = wrapped, module_kind\n"
        "    def create_module(self, spec):\n"
        "        create = getattr(self.wrapped, 'create_module', None)\n"
        "        return create(spec) if create is not None else None\n"
        "    def exec_module(self, module):\n"
        "        self.wrapped.exec_module(module)\n"
        "        phase = os.environ.get('SF_PARALLEL_E2E_CRASH')\n"
        "        marker = os.environ.get('SF_PARALLEL_E2E_CRASH_MARKER')\n"
        "        def crash():\n"
        "            if marker and Path(marker).exists():\n"
        "                return\n"
        "            if marker:\n"
        "                Path(marker).write_text(phase or '')\n"
        "            os._exit(86)\n"
        "        if self.module_kind == 'transaction' and phase == 'after_decided':\n"
        "            original = module.TxnStore.decide\n"
        "            def decide(store, record, blobs):\n"
        "                original(store, record, blobs)\n"
        "                if record.txn_id == 1:\n"
        "                    crash()\n"
        "            module.TxnStore.decide = decide\n"
        "        elif self.module_kind == 'transaction' and phase == 'after_materialized':\n"
        "            original = module.TxnStore.materialize\n"
        "            def materialize(store, txn_id, workspace):\n"
        "                original(store, txn_id, workspace)\n"
        "                if txn_id == 1:\n"
        "                    crash()\n"
        "            module.TxnStore.materialize = materialize\n"
        "        elif self.module_kind == 'transaction' and phase == 'between_materialized_files':\n"
        "            replace = module.os.replace\n"
        "            count = 0\n"
        "            def replace_after_first_parallel_file(source, destination, *args, **kwargs):\n"
        "                nonlocal count\n"
        "                result = replace(source, destination, *args, **kwargs)\n"
        "                target = Path(destination)\n"
        "                continuations = Path(marker).parent / 'run' / 'worker' / 'continuation-workspaces' if marker else None\n"
        "                if (continuations is not None and Path(source).name.startswith('.parallel-') and\n"
        "                        target.parent.parent.parent.parent == continuations and\n"
        "                        target.parent.parent.name == 'repo' and target.parent.name == 'pkg' and\n"
        "                        target.name in {'a.py', 'b.py'}):\n"
        "                    count += 1\n"
        "                    if count == 1:\n"
        "                        crash()\n"
        "                return result\n"
        "            module.os.replace = replace_after_first_parallel_file\n"
        "        elif self.module_kind == 'parallel' and phase == 'after_published':\n"
        "            original = module.TxnStore.mark_published\n"
        "            def mark_published(store, txn_id, publication_id):\n"
        "                original(store, txn_id, publication_id)\n"
        "                if txn_id == 1:\n"
        "                    crash()\n"
        "            module.TxnStore.mark_published = mark_published\n"
        "        elif self.module_kind == 'parallel' and phase == 'after_ack':\n"
        "            original = module._record_outcome\n"
        "            def record_outcome(call_root, outcome, *args, **kwargs):\n"
        "                original(call_root, outcome, *args, **kwargs)\n"
        "                if outcome.txn_id == 1:\n"
        "                    crash()\n"
        "            module._record_outcome = record_outcome\n"
        "        elif self.module_kind == 'tape' and phase == 'after_input_refusal_intent':\n"
        "            original = module.ToolTape.intent\n"
        "            def intent(tape, **kwargs):\n"
        "                original(tape, **kwargs)\n"
        "                args = kwargs.get('args')\n"
        "                if (kwargs.get('tool') == 'parallel' and isinstance(args, dict) and\n"
        "                        args.get('schema') == 'stateframe-parallel-input-refusal/1'):\n"
        "                    crash()\n"
        "            module.ToolTape.intent = intent\n"
        "\n"
        "class _CrashFinder(importlib.abc.MetaPathFinder):\n"
        "    modules = {'kernel_parallel_txn': 'transaction',\n"
        "               'stateframe.engine.kernel_parallel_txn': 'transaction',\n"
        "               'kernel_parallel': 'parallel',\n"
        "               'stateframe.engine.kernel_parallel': 'parallel',\n"
        "               'kernel_tool_tape': 'tape',\n"
        "               'stateframe.engine.kernel_tool_tape': 'tape'}\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        module_kind = self.modules.get(fullname)\n"
        "        if module_kind is None:\n"
        "            return None\n"
        "        spec = importlib.machinery.PathFinder.find_spec(fullname, path)\n"
        "        if spec is not None and spec.loader is not None:\n"
        "            spec.loader = _CrashLoader(spec.loader, module_kind)\n"
        "        return spec\n"
        "\n"
        "sys.meta_path.insert(0, _CrashFinder())\n"
    )
    return install_startup_hook(python, "_stateframe_e2e_crash", hook_source)


def parent_wire_files(run_dir: Path) -> list[Path]:
    attempts = sorted((run_dir / "attempts").glob("[0-9]*/wire/*.request.json"),
                      key=lambda path: (int(path.parents[1].name), path.name))
    paths = [*attempts, *sorted((run_dir / "wire").glob("*.request.json"))]
    assert paths, f"parent run has no recorded wire requests: {run_dir}"
    return paths


def wait_for_parent_wire(run_dir: Path, number: int, process: EngineProcess,
                         *, timeout: float) -> Path:
    """Wait for a request rebuilt by resume, including one replayed from the tape."""
    request = run_dir / "wire" / f"{number:04d}.request.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if request.is_file():
            return request
        if process.poll() is not None:
            raise AssertionError(f"resume exited before rebuilding {request}")
        time.sleep(0.1)
    raise AssertionError(f"resume did not rebuild {request} within {timeout}s")


def subframe_directories(run_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for call_root in sorted((run_dir / "parallel").glob("*/")):
        units_root = call_root / "units"
        if not units_root.is_dir():
            continue
        for subframe in sorted(units_root.iterdir()):
            if subframe.is_dir() and (subframe / "wire").is_dir():
                found[subframe.name] = subframe
    return found


def subframe_wire_files(subframe_dir: Path) -> list[Path]:
    attempts = sorted((subframe_dir / "attempts").glob("[0-9]*/wire/*.request.json"),
                      key=lambda path: (int(path.parents[1].name), path.name))
    paths = [*attempts, *sorted((subframe_dir / "wire").glob("*.request.json"))]
    assert paths, f"subframe {subframe_dir.name} has no recorded wire requests"
    return paths


def _f4_payloads(request: dict[str, object], request_path: Path) -> list[dict[str, object]]:
    messages = request.get("messages")
    assert isinstance(messages, list), f"wire request has no messages: {request_path}"
    payloads: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith("[StateFrame]\n"):
            continue
        frame = content[len("[StateFrame]\n"):].split("\n[", 1)[0]
        try:
            payload = json.loads(frame)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == "stateframe-f4/1":
            payloads.append(payload)
    assert payloads, f"shipped F4 frame is absent from wire bytes: {request_path}"
    assert any(isinstance(payload.get("state"), dict)
               and isinstance(payload["state"].get("framefit5"), dict)
               and payload["state"]["framefit5"].get("mode") == "FRAME_ONLY"
               for payload in payloads), f"F4 frame-only state is absent from wire bytes: {request_path}"
    return payloads


def assert_shipped_wire(request_paths: Iterable[Path], *, task_marker: str | None = None) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for request_path in request_paths:
        request = read_json(request_path)
        current = _f4_payloads(request, request_path)
        payloads.extend(current)
        if task_marker is not None:
            messages = request.get("messages")
            assert isinstance(messages, list)
            contents = [message.get("content") for message in messages if isinstance(message, dict)]
            assert any(isinstance(content, str) and task_marker in content for content in contents), (
                f"wire request omitted task marker {task_marker!r}: {request_path}")
    return payloads


def tool_names(request_path: Path) -> set[str]:
    request = read_json(request_path)
    tools = request.get("tools")
    assert isinstance(tools, list), f"wire request has no tool schemas: {request_path}"
    names = {tool["function"]["name"] for tool in tools
             if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
             and isinstance(tool["function"].get("name"), str)}
    assert names, f"wire request has no callable tools: {request_path}"
    return names


def assert_subframe_surface(request_path: Path) -> dict[str, object]:
    names = tool_names(request_path)
    assert names <= ALLOWED_SUBFRAME_TOOLS, (request_path, sorted(names - ALLOWED_SUBFRAME_TOOLS))
    payloads = assert_shipped_wire([request_path])
    for payload in payloads:
        state = payload.get("state")
        frame = state.get("framefit5") if isinstance(state, dict) else None
        # Only re-prepared requests carry an owner menu; when the key is absent there is nothing to validate.
        if not isinstance(frame, dict) or "owner_protocol" not in frame:
            continue
        owner = frame["owner_protocol"]
        assert isinstance(owner, dict), f"F4 owner menu is invalid: {request_path}"
        assert owner.get("schema") == "stateframe-framefit5-owner/1", f"F4 owner menu is invalid: {request_path}"
        if "admitted_actions" not in owner:
            # attach_owner_protocol emits a schema-only block when the owner contract names no menu;
            # route/menu_source are only ever written together with admitted_actions.
            assert "route" not in owner and "menu_source" not in owner, f"F4 owner menu is incomplete: {request_path}"
            continue
        actions = owner.get("admitted_actions")
        if actions is None:
            # Scoped children can inherit a schema-only owner protocol. The
            # request's tool schemas remain the authoritative executable menu.
            assert "route" not in owner and "menu_source" not in owner, (
                f"F4 owner menu is incomplete: {request_path}")
        else:
            assert isinstance(actions, list) and all(isinstance(action, str) for action in actions), (
                f"F4 owner menu is invalid: {request_path}")
            assert set(actions) <= ALLOWED_SUBFRAME_TOOLS, (
                request_path, sorted(set(actions) - ALLOWED_SUBFRAME_TOOLS))
    return payloads[0]


def result_records(run_dir: Path) -> dict[str, ParallelResult]:
    results: dict[str, ParallelResult] = {}
    for path in sorted((run_dir / "parallel").glob("*/result.json")):
        raw = read_json(path)
        result = ParallelResult.from_json(raw)
        results[result.call_id] = result
    return results


def all_subframe_wire_bytes(run_dir: Path) -> list[bytes]:
    return [path.read_bytes() for subframe in subframe_directories(run_dir).values()
            for path in subframe_wire_files(subframe)]
