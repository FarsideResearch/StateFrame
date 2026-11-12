"""Linux mutation regressions reproduced through the CLI and scripted endpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest


_SOURCE = Path(__file__).resolve().parents[1] / "src"
_LINUX_REASON = "mutation launch and resume regressions require Linux bubblewrap"
_EXPECTED_REPORT = (
    b"# Daily Invoice Summary\n\n"
    b"Invoices processed: 3\n"
    b"Total: $50.00\n"
)
_REPORT_SOURCE = r'''import csv
import io

def handle(input, ctx):
    text = ctx.input_text(input["input_id"])
    rows = list(csv.DictReader(io.StringIO(text)))
    total = sum(int(row["amount_cents"]) for row in rows)
    cents = f"${total // 100}.{total % 100:02d}"
    report = ("# Daily Invoice Summary\n\n"
              f"Invoices processed: {len(rows)}\n"
              f"Total: {cents}\n")
    ctx.propose_edit("create", "reports/daily-summary.md", report)
    return {"invoice_count": len(rows), "total_cents": total}
'''
_OBSERVER_SOURCE = "def handle(input, ctx):\n    return {}\n"


def _script(steps):
    from stateframe.scripted_endpoint import Script, serve

    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": steps})
    script.request_peer = threading.local()
    server = serve(script)
    original_handler = server.RequestHandlerClass

    class PeerRecordingHandler(original_handler):
        def do_POST(self) -> None:  # noqa: N802
            script.request_peer.addresses = (self.client_address,
                                             self.connection.getsockname())
            try:
                super().do_POST()
            finally:
                script.request_peer.addresses = None

    server.RequestHandlerClass = PeerRecordingHandler
    return script, server


def _mutate(op, ident=None, args=None):
    arguments = {"op": op}
    if ident is not None:
        arguments["id"] = ident
    if args is not None:
        arguments["args"] = json.dumps(args, separators=(",", ":"))
    return {"tool": "mutate", "required": True, "arguments": arguments}


def _tool(name, arguments):
    return {"tool": name, "required": True, "arguments": arguments}


def _done(summary, *, repeat=False):
    step = _tool("done", {"summary": summary})
    if repeat:
        step["repeat"] = True
    return step


def _environment(home: Path, state_home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    prefixes = ("STATEFRAME_", "REACT_", "KERNEL_", "DS_", "PILOT_", "UNIT_PLAN_")
    for key in list(environment):
        if key.startswith(prefixes) or key in {"PYTHONHOME", "PYTHONPATH"}:
            environment.pop(key, None)
    environment.update({
        "HOME": str(home),
        "STATEFRAME_HOME": str(state_home),
        "PYTHONPATH": str(_SOURCE),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    home.mkdir(parents=True, exist_ok=True)
    return environment


def _profile(state_home: Path, destination: Path) -> Path:
    source = _SOURCE / "stateframe" / "profiles" / "r2hc3.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    document["switches"]["STATEFRAME_HOME"] = str(state_home)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return destination


def _run_command(repo: Path, run_dir: Path, profile: Path, endpoint: str, task: str,
                 *, mutations: str, extra_args=(), max_turns: int = 32) -> list[str]:
    return [
        sys.executable, "-m", "stateframe", "run",
        "--profile", str(profile),
        "--mutations", mutations,
        "--repo", str(repo),
        "--task", task,
        "--endpoint", endpoint,
        "--model", "scripted",
        "--run-dir", str(run_dir),
        "--quiet",
        "--max-turns", str(max_turns),
        "--test-python", sys.executable,
        "--allow-source-tree",
        *extra_args,
    ]


def _events(run_dir: Path) -> list[dict]:
    path = run_dir / "mutations" / "journal.jsonl"
    if not path.is_file():
        return []
    data = path.read_bytes()
    lines = data.splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not data.endswith(b"\n"):
                break
            raise
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _count(rows: list[dict], kind: str, *, name: str | None = None) -> int:
    return sum(row.get("kind") == kind and (name is None or row.get("name") == name)
               for row in rows)


class _ResponseGate:
    """Hold one scripted response after its journal predicate becomes durable."""

    def __init__(self, script, run_dir: Path, predicate, *, timeout: float = 420):
        self.script = script
        self.run_dir = run_dir
        self.predicate = predicate
        self.timeout = timeout
        self.arrived = threading.Event()
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.error: str | None = None
        self.socket_owner_pid: int | None = None
        self.request_index: int | None = None
        self.held_step: dict | None = None
        self._replay_request_index: int | None = None
        self._lock = threading.Lock()
        original_next_call = script.next_call

        def gated_next_call(offered):
            with self._lock:
                request_index = len(script.requests)
                if request_index == self._replay_request_index:
                    self._replay_request_index = None
                    if self.held_step is None:
                        raise AssertionError("held scripted response is missing")
                    return dict(self.held_step)

                step = original_next_call(offered)
                if self.arrived.is_set() or not self.predicate(_events(run_dir)):
                    return step

                self.request_index = request_index
                self.held_step = dict(step)
                peer = script.request_peer.addresses
                self.socket_owner_pid = _sender_pid_for_peer(*peer) if peer is not None else None
                self.waiting.set()
                self.arrived.set()
                if not self.release.wait(self.timeout):
                    self.error = f"scripted request {request_index} was not released"
                    self.waiting.clear()
                    raise TimeoutError(self.error)
                self._replay_request_index = request_index + 1
                self.waiting.clear()
                return step

        script.next_call = gated_next_call

    @property
    def in_flight(self) -> bool:
        return (self.request_index is not None and self.arrived.is_set()
                and self.waiting.is_set() and not self.release.is_set())


def _proc_socket_address(address: tuple) -> str:
    host, port = address[:2]
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    packed = socket.inet_pton(family, host)
    if sys.byteorder == "little":
        if family == socket.AF_INET:
            packed = packed[::-1]
        else:
            packed = b"".join(packed[index:index + 4][::-1]
                              for index in range(0, len(packed), 4))
    return f"{packed.hex().upper()}:{port:04X}"


def _sender_pid_for_peer(peer: tuple, local: tuple) -> int | None:
    peer_address = _proc_socket_address(peer)
    local_address = _proc_socket_address(local)
    table = "/proc/net/tcp6" if len(peer_address.split(":")[0]) == 32 else "/proc/net/tcp"
    try:
        rows = Path(table).read_text(encoding="ascii").splitlines()[1:]
    except OSError:
        return None
    inodes = {
        fields[9] for row in rows if len(fields := row.split()) > 9
        and fields[1] == peer_address and fields[2] == local_address
        and fields[3] == "01"
    }
    if len(inodes) != 1:
        return None
    target = f"socket:[{next(iter(inodes))}]"
    owners = set()
    for process in os.scandir("/proc"):
        if not process.name.isdigit():
            continue
        try:
            with os.scandir(f"/proc/{process.name}/fd") as descriptors:
                for descriptor in descriptors:
                    try:
                        if os.readlink(descriptor.path) == target:
                            owners.add(int(process.name))
                            break
                    except OSError:
                        continue
        except OSError:
            continue
    return next(iter(owners)) if len(owners) == 1 else None


def _engine_child_identity(pid: int) -> tuple[int, str] | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    if fields.get("State", "").strip().startswith(("Z", "X")) or not cmdline:
        return None
    ppid = int(fields["PPid"].strip())
    argv = cmdline.rstrip(b"\0").split(b"\0")
    if any(arg.endswith(b"/_child.py") or arg == b"stateframe._child"
           for arg in argv):
        return ppid, "engine"
    if (argv[-1] == b"--http-child"
            and any(arg.endswith(b"/kernel_bounded_http.py") for arg in argv)):
        return ppid, "http"
    return None


def _proc_engine_children(parent_pid: int) -> list[int]:
    children = []
    for process in os.scandir("/proc"):
        if process.name.isdigit():
            identity = _engine_child_identity(int(process.name))
            if identity == (parent_pid, "engine"):
                children.append(int(process.name))
    return sorted(children)


@pytest.mark.skipif(sys.platform != "linux", reason=_LINUX_REASON)
@pytest.mark.parametrize("family, host", [
    (socket.AF_INET, "127.0.0.1"),
    (socket.AF_INET6, "::1"),
])
def test_gated_request_peer_resolves_socket_owner_linux(family, host):
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        listener.listen()
        with socket.create_connection(listener.getsockname()[:2]) as client:
            accepted, _ = listener.accept()
            with accepted:
                assert _sender_pid_for_peer(accepted.getpeername(),
                                            accepted.getsockname()) == os.getpid()


@pytest.mark.skipif(sys.platform != "linux", reason=_LINUX_REASON)
def test_gated_request_peer_resolves_separate_client_pid_linux():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        client = subprocess.Popen(
            [sys.executable, "-c",
             "import socket, sys; "
             "s = socket.create_connection(('127.0.0.1', int(sys.argv[1]))); "
             "sys.stdin.buffer.read(1); s.close()",
             str(listener.getsockname()[1])],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            accepted, _ = listener.accept()
            with accepted:
                assert client.pid != os.getpid()
                assert _engine_child_identity(client.pid) is None
                assert _sender_pid_for_peer(accepted.getpeername(),
                                            accepted.getsockname()) == client.pid
        finally:
            client.communicate(input=b"x", timeout=5)


def _stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _interrupt_and_resume(command: list[str], environment: dict[str, str], run_dir: Path,
                          endpoint: str, gate: _ResponseGate, *, resume_args=(),
                          before_resume=None) -> list[dict]:
    process = subprocess.Popen(
        command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + gate.timeout
        while time.monotonic() < deadline:
            if gate.arrived.wait(0.1):
                break
            if process.poll() is not None:
                output = process.communicate(timeout=5)[0]
                raise AssertionError(
                    "CLI exited before the journal-gated request arrived:\n"
                    + output[-5000:]
                )
        else:
            raise AssertionError("timed out waiting for the journal-gated request")

        assert gate.in_flight, "endpoint did not hold its response"
        children = _proc_engine_children(process.pid)
        assert len(children) == 1, f"expected one engine child, found {children}"
        sender_pid = children[0]
        assert _engine_child_identity(sender_pid) == (process.pid, "engine"), (
            f"gated request sender {sender_pid} is not a live direct engine child"
        )
        assert gate.socket_owner_pid is not None, (
            "could not resolve the held request's TCP client socket owner"
        )
        owner_identity = _engine_child_identity(gate.socket_owner_pid)
        assert (gate.socket_owner_pid == sender_pid
                and owner_identity == (process.pid, "engine")) or (
                    owner_identity == (sender_pid, "http")), (
            f"socket owner {gate.socket_owner_pid} is neither the gated engine "
            f"{sender_pid} nor its bounded HTTP child (identity={owner_identity})"
        )
        os.kill(sender_pid, signal.SIGKILL)
        process.wait(timeout=90)
        initial_output = process.communicate(timeout=10)[0]
        assert process.returncode != 0, (
            "CLI returned success after the engine child was killed:\n" + initial_output[-5000:]
        )

        rows_before_resume = _events(run_dir)
        if before_resume is not None:
            before_resume(rows_before_resume)
    except BaseException:
        gate.release.set()
        _stop_process_group(process)
        raise
    finally:
        gate.release.set()
        _stop_process_group(process)

    resumed = subprocess.run(
        [sys.executable, "-m", "stateframe", "resume", str(run_dir),
         "--endpoint", endpoint, *resume_args],
        env=environment, capture_output=True, text=True, timeout=420, check=False,
    )
    assert resumed.returncode == 0, (
        "CLI resume failed:\n" + resumed.stdout[-5000:] + resumed.stderr[-5000:]
    )
    gate.script.assert_consumed()
    return _events(run_dir)


def _seed_qualification(state_home: Path, *, ok: bool) -> None:
    from kernel_mutate_qualify import _binding

    report = {
        "ok": ok,
        "platform": "linux",
        "runtime_excluded": [],
        "checks": [{
            "name": "regression_fixture",
            "ok": ok,
            "detail": "fixture qualification state",
        }],
    }
    path = state_home / "mutations" / "qualified.json"
    path.write_text(json.dumps({
        "schema": "stateframe-mutation-qualified/1",
        "binding": _binding(),
        "report": report,
    }, separators=(",", ":")), encoding="utf-8")


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason=_LINUX_REASON)
def test_m1_interrupt_resume_preserves_committed_report_effect_linux(tmp_path):
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    (repo / "reports" / ".gitkeep").touch()
    (repo / "tests").mkdir()
    (repo / "tests" / "test_report.py").write_text(
        "from pathlib import Path\n\n"
        "def test_daily_summary_bytes():\n"
        f"    assert Path('reports/daily-summary.md').read_bytes() == {_EXPECTED_REPORT!r}\n",
        encoding="utf-8",
    )
    (repo / "invoices.csv").write_text(
        "invoice_id,amount_cents\nINV-001,1250\nINV-002,2750\nINV-003,1000\n",
        encoding="utf-8",
    )
    state_home = tmp_path / "state-home"
    profile = _profile(state_home, tmp_path / "profile.json")
    environment = _environment(tmp_path / "home", state_home)
    run_dir = tmp_path / "run"

    script, server = _script([
        _mutate("enter", "daily_invoice_summary", {
            "goal": "Summarize invoices and propose the daily report file",
            "kind": "reader",
            "inputs": ["invoices.csv"],
            "needs": {"writes": True},
        }),
        _mutate("submit", args={
            "code": _REPORT_SOURCE,
            "summary": "Create the invoice summary",
            "output_schema": {
                "type": "object",
                "properties": {
                    "invoice_count": {"type": "integer"},
                    "total_cents": {"type": "integer"},
                },
                "required": ["invoice_count", "total_cents"],
                "additionalProperties": False,
            },
            "expect": {"invoice_count": 3, "total_cents": 5000},
        }),
        _mutate("use", "daily_invoice_summary", {
            "input": {"input_id": "in0"},
            "inputs": ["invoices.csv"],
        }),
        _done("The report contains three invoices totaling $50.00.", repeat=True),
    ])
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    gate = _ResponseGate(
        script, run_dir,
        lambda rows: (_count(rows, "EFFECT_COMMITTED") == 1
                      and _count(rows, "TERMINAL") == 1
                      and _count(rows, "CLEANUP_VERIFIED") == 1),
    )

    def assert_committed_before_resume(_rows):
        candidate = run_dir / "workspace" / "reports" / "daily-summary.md"
        assert candidate.read_bytes() == _EXPECTED_REPORT
        assert not (repo / "reports" / "daily-summary.md").exists()

    try:
        rows = _interrupt_and_resume(
            _run_command(
                repo, run_dir, profile, endpoint,
                "Write reports/daily-summary.md with the invoice totals from invoices.csv.",
                mutations="authored",
                extra_args=(
                    "--mutation-write", "reports/daily-summary.md",
                    "--check", "tests/test_report.py",
                    "--in-place",
                ),
            ),
            environment, run_dir, endpoint, gate, before_resume=assert_committed_before_resume,
        )
    finally:
        server.shutdown()
        server.server_close()

    candidate = run_dir / "workspace" / "reports" / "daily-summary.md"
    in_place = repo / "reports" / "daily-summary.md"
    assert candidate.read_bytes() == _EXPECTED_REPORT
    assert in_place.read_bytes() == _EXPECTED_REPORT
    assert _count(rows, "EFFECT_COMMITTED") == 1
    assert _count(rows, "EFFECT_REFUSED") == 0


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason=_LINUX_REASON)
def test_m2_interrupt_resume_keeps_disabled_hook_disabled_linux(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Demo note\n\nThe project opened in 2026.\n",
                                    encoding="utf-8")

    state_one = tmp_path / "state-one"
    profile_one = _profile(state_one, tmp_path / "profile-author.json")
    environment_one = _environment(tmp_path / "home-one", state_one)
    first_run = tmp_path / "run-retain"
    first_script, first_server = _script([
        _mutate("enter", "observer", {"goal": "Observe tool metadata", "kind": "hook"}),
        _mutate("submit", args={
            "code": _OBSERVER_SOURCE,
            "summary": "Observe tool metadata",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "expect": {},
        }),
        _done("The observer hook is ready."),
    ])
    first_endpoint = f"http://127.0.0.1:{first_server.server_port}/v1"
    try:
        first = subprocess.run(
            _run_command(repo, first_run, profile_one, first_endpoint,
                         "Create an offline observer for tool metadata.", mutations="authored",
                         max_turns=12),
            env=environment_one, capture_output=True, text=True, timeout=420, check=False,
        )
        assert first.returncode == 0, first.stdout[-5000:] + first.stderr[-5000:]
        first_script.assert_consumed()
    finally:
        first_server.shutdown()
        first_server.server_close()

    retained = subprocess.run(
        [sys.executable, "-m", "stateframe", "mutations", "retain",
         str(first_run), "observer", "--scope", "user"],
        env=environment_one, capture_output=True, text=True, timeout=120, check=False,
    )
    assert retained.returncode == 0, retained.stdout[-3000:] + retained.stderr[-3000:]

    state_two = tmp_path / "state-two"
    state_two.mkdir()
    shutil.copytree(
        state_one / "mutations", state_two / "mutations",
        ignore=shutil.ignore_patterns("qualified.json", ".qualified-*"),
    )
    user_index = state_two / "mutations" / "index" / "user.json"
    assert user_index.is_file() and json.loads(user_index.read_text(encoding="utf-8"))

    # C5 gets this refusal by masking ldd in a disposable container. A binding-valid
    # qualification record reproduces the same durable platform-unqualified state safely.
    _seed_qualification(state_two, ok=False)
    profile_two = _profile(state_two, tmp_path / "profile-resume.json")
    environment_two = _environment(tmp_path / "home-two", state_two)
    run_dir = tmp_path / "run-refusal"
    second_script, second_server = _script([
        _tool("read", {"path": "README.md"}),
        _tool("read", {"path": "README.md"}),
        _done("README.md says the project opened in 2026.", repeat=True),
    ])
    endpoint = f"http://127.0.0.1:{second_server.server_port}/v1"
    gate = _ResponseGate(
        second_script, run_dir,
        lambda rows: (_count(rows, "HOOK_REFUSED") == 1
                      and _count(rows, "HOOK_DISABLED") == 1),
    )

    def qualify_hook_for_reattach(rows_before_resume):
        refused = [row for row in rows_before_resume if row.get("kind") == "HOOK_REFUSED"]
        disabled = [row for row in rows_before_resume if row.get("kind") == "HOOK_DISABLED"]
        assert len(refused) == 1 and refused[0].get("code") == "MUTATION_PLATFORM_UNQUALIFIED"
        assert len(disabled) == 1 and disabled[0].get("reason") == "platform_unqualified"
        _seed_qualification(state_two, ok=True)

    try:
        rows = _interrupt_and_resume(
            _run_command(
                repo, run_dir, profile_two, endpoint,
                "Read README.md and summarize its note.", mutations="authored",
                extra_args=("--mutation-hook", "observer:post_tool"),
            ),
            environment_two, run_dir, endpoint, gate,
            before_resume=qualify_hook_for_reattach,
        )
    finally:
        second_server.shutdown()
        second_server.server_close()

    assert _count(rows, "HOOK_DISABLED") == 1
    assert _count(rows, "HOOK_TRIGGER_INTENT", name="observer") == 1
    assert not [row for row in rows
                if row.get("kind") == "HOOK_TRIGGER_COMPLETE" and row.get("name") == "observer"]


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason=_LINUX_REASON)
def test_m3_broker_fetch_returns_status_and_body_linux(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "page_source.py").write_text(
        'PAGE_URL = "https://example.com/"\n', encoding="utf-8")
    state_home = tmp_path / "state-home"
    profile = _profile(state_home, tmp_path / "profile.json")
    environment = _environment(tmp_path / "home", state_home)
    run_dir = tmp_path / "run"
    script, server = _script([
        _mutate("catalog"),
        _mutate("use", "html_extract", {
            "input": {"url": "https://example.com/"},
            "inputs": [],
        }),
        _done("The public page title and text were extracted.", repeat=True),
    ])
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    gate = _ResponseGate(
        script, run_dir,
        lambda rows: (_count(rows, "BROKER_COMPLETE") == 1
                      and _count(rows, "TERMINAL") == 1
                      and _count(rows, "CLEANUP_VERIFIED") == 1),
    )

    def assert_broker_body_before_resume(rows):
        from stateframe.engine.kernel_mutate_records import Journal

        completion = next(row for row in rows if row.get("kind") == "BROKER_COMPLETE")
        assert completion.get("status") == 200, completion
        assert completion.get("error") is None, completion
        digest = completion.get("body_digest")
        body = Journal(run_dir).get_blob(digest)
        assert body and completion.get("bytes") == len(body)

    try:
        rows = _interrupt_and_resume(
            _run_command(
                repo, run_dir, profile, endpoint,
                "Extract the public title and readable text from the URL in page_source.py.",
                mutations="prebuilt",
                extra_args=(
                    "--target", "page_source.py",
                    "--mutation-network", "public-web",
                    "--mutation-domain", "example.com",
                    "--shell-network", "full",
                ),
            ),
            environment, run_dir, endpoint, gate,
            resume_args=("--replay-external",),
            before_resume=assert_broker_body_before_resume,
        )
    finally:
        server.shutdown()
        server.server_close()

    from stateframe.engine.kernel_mutate_records import Journal

    completions = [row for row in rows if row.get("kind") == "BROKER_COMPLETE"]
    assert len(completions) == 1
    assert completions[0].get("status") == 200, completions[0]
    assert completions[0].get("error") is None, completions[0]
    body = Journal(run_dir).get_blob(completions[0]["body_digest"])
    assert body and completions[0].get("bytes") == len(body)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux mutation bubblewrap command construction")
def test_m4_mutation_sandbox_command_binds_python_and_loader(monkeypatch, tmp_path):
    import kernel_test_isolation as isolation
    from kernel_exec_capability import ExecCapability, ExecLimits

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mounts = {}
    for name in ("boot", "unit", "ctl", "in", "resp"):
        root = workspace / name
        root.mkdir()
        mounts[name] = str(root)

    prefix = Path(sys.prefix).resolve(strict=True)
    executable = Path(sys.executable).resolve(strict=True)
    capability = ExecCapability(
        purpose="mutation",
        workspace=workspace,
        workspace_writable=False,
        read_roots=(prefix, executable),
        write_roots=(),
        network="none",
        env={},
        limits=ExecLimits(),
    )

    monkeypatch.setattr(isolation, "sandbox_executable", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(isolation, "_bwrap_userns_guard_supported", lambda _name: False)
    real_isfile = isolation.os.path.isfile
    monkeypatch.setattr(
        isolation.os.path, "isfile",
        lambda path: True if path == "/etc/ld.so.cache" else real_isfile(path),
    )

    command, _environment = isolation.capability_command(
        ["/bin/true"], capability, platform_backend="bwrap", mutation_mounts=mounts,
    )
    boundary = command.index("--")
    bind_rows = [
        command[index:index + 3]
        for index, value in enumerate(command[:boundary - 2])
        if value in {"--ro-bind", "--bind"}
    ]
    assert ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"] in bind_rows
    readonly_destinations = {
        Path(row[2]) for row in bind_rows if row[0] == "--ro-bind"
    }
    assert any(executable.is_relative_to(destination) for destination in readonly_destinations), (
        f"bwrap does not expose Python executable {executable}: {bind_rows}"
    )
