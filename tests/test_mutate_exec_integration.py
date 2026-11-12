"""Host namespace and cleanup checks for the mutation executor."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import pytest
from fixtures.process_signals import signal_run_owned

from kernel_exec_tracker import ProcessTable
from kernel_mutate_exec import LaunchSpec, MutationExecutor, MutationLimits
from mutate_leftovers import stage_test_unit


pytestmark = pytest.mark.host_sandbox


def _fd_count():
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux /proc/self/fd verifies pidfd release",
)
def test_executor_create_run_close_releases_pidfds_five_times(tmp_path):
    baseline = {os.readlink(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")
                if (tmp := f"/proc/self/fd/{fd}") and os.path.exists(tmp)}
    unit = _unit(tmp_path, "def handle(input,ctx): return {'ok':True}\n")
    for index in range(5):
        root = tmp_path / f"runtime-{index}"
        root.mkdir()
        with MutationExecutor(root, network_envelope="none") as executor:
            assert executor.run(_spec(unit)).cleanup.ok
        current = {os.readlink(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")
                   if (tmp := f"/proc/self/fd/{fd}") and os.path.exists(tmp)}
        assert current == baseline


def test_two_real_broker_fetches_reuse_boot_and_leave_no_stage(tmp_path, monkeypatch):
    import kernel_mutate_exec as module

    root = tmp_path / "runtime"
    root.mkdir()
    engine = Path(module.__file__).parent
    copied = tmp_path / "engine-copy"
    copied.mkdir()
    for name in ("sf_mutate_boot.py", "sf_mutate_sdk.py", "sf_launch.py"):
        (copied / name).write_bytes((engine / name).read_bytes())
    (copied / "sf_mutate_fetch.py").write_text("import sys\nsys.stdin.buffer.read()\nsys.stdout.buffer.write(b'OK')\n")
    monkeypatch.setattr(module, "_ENGINE", copied)
    with MutationExecutor(root, network_envelope="full") as executor:
        for _ in range(2):
            launch = uuid.uuid4().hex
            doc = {"call": "c1", "launch": launch, "role": "broker", "request": {}}
            spec = LaunchSpec("broker", "c1", launch, None, {}, {}, json.dumps(doc).encode(), MutationLimits())
            assert executor.run(spec).stdout == b"OK"
            assert not list((root / "stage").iterdir())


def test_launch_error_never_completes_tape_before_stage_cleanup(tmp_path, monkeypatch):
    import kernel_mutate_exec as module
    root = tmp_path / "runtime"
    root.mkdir()
    unit = _unit(tmp_path, "def handle(input,ctx): return {}\n")
    with MutationExecutor(root, network_envelope="none") as executor:
        def fail_spawn(*_args, **_kwargs):
            raise ValueError("launch refused")
        monkeypatch.setattr(executor._session, "_spawn", fail_spawn)
        with pytest.raises(ValueError, match="launch refused"):
            executor.run(_spec(unit))
        assert not list((root / "stage").iterdir())
        with monkeypatch.context() as patch:
            original_remove = module._remove_tree
            def refuse_stage_cleanup(path):
                if path.parent == root / "stage":
                    raise OSError("stage still present")
                return original_remove(path)
            patch.setattr(module, "_remove_tree", refuse_stage_cleanup)
            import kernel_tool_tape
            from kernel_mutate_records import Journal
            from kernel_mutate_runtime import MutationRuntime
            class Tape:
                def __init__(self):
                    self.intents = []
                    self.completions = []
                def intent(self, **fields):
                    self.intents.append(fields)
                def complete(self, **fields):
                    self.completions.append(fields)
            tape = Tape()
            patch.setattr(kernel_tool_tape, "tape_for_run", lambda: tape)
            patch.setattr(kernel_tool_tape, "replay_mode_active", lambda: False)
            runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "journal"))
            with pytest.raises(module.MutationContainmentError, match="stage still present"):
                runtime._taped({"op": "use"}, "tape-case", lambda: executor.run(_spec(unit)))
            assert len(tape.intents) == 1 and tape.completions == []
        with monkeypatch.context() as patch:
            original_record = executor._record
            def fail_failure_record(spec, state, identity=None):
                if state == "CLEANUP_FAILED":
                    raise OSError("ledger unavailable")
                return original_record(spec, state, identity)
            patch.setattr(executor, "_record", fail_failure_record)
            patch.setattr(module, "_remove_tree", refuse_stage_cleanup)
            with pytest.raises(module.MutationContainmentError):
                executor.run(_spec(unit))
        assert list((root / "stage").iterdir())
        with monkeypatch.context() as patch:
            def fail_staging(spec):
                (root / "stage" / spec.launch).mkdir()
                raise ValueError("staging refused")
            patch.setattr(executor, "_stage_spec", fail_staging)
            patch.setattr(module, "_remove_tree", refuse_stage_cleanup)
            with pytest.raises(module.MutationContainmentError, match="stage still present"):
                executor.run(_spec(unit))


def test_constructor_failure_before_owner_is_recoverable(tmp_path, monkeypatch):
    import kernel_mutate_exec as module
    root = tmp_path / "runtime"
    root.mkdir()
    original = module._copy_regular
    def fail_first(source, target):
        if target.name == "sf_mutate_sdk.py":
            raise OSError("injected constructor failure")
        return original(source, target)
    monkeypatch.setattr(module, "_copy_regular", fail_first)
    with pytest.raises(OSError, match="injected constructor failure"):
        MutationExecutor(root, network_envelope="none")
    assert not root.exists()


def _unit(root: Path, source: str) -> Path:
    import kernel_mutate_build

    return stage_test_unit(kernel_mutate_build, root / "stage" / "runtime", source).dir


def _spec(unit: Path, *, wall=15.0, cpu=5.0, cap=4096, role="payload") -> LaunchSpec:
    launch = uuid.uuid4().hex
    doc = {"wire": 1, "call": "c1", "launch": launch, "phase": 0, "role": role,
           "input": {}, "inputs": [], "responses": [], "grants": {"network": False, "writes": False},
           "max_fetch": 4}
    return LaunchSpec(role, "c1", launch, unit, {}, {},
                      json.dumps(doc).encode(), MutationLimits(wall, cpu, 256, cap))


def _scan(root: Path, before: int):
    assert not (root / "stage").exists() or not list((root / "stage").iterdir())
    owner = json.loads((root / "owner.json").read_text())
    assert isinstance(owner.get("ns"), str) and owner["ns"].startswith("pid:[")
    assert len(owner.get("cmdline_sha256", "")) == 64
    table = ProcessTable()
    for line in (root / "launches.jsonl").read_text().splitlines():
        row = json.loads(line)
        identity = row.get("identity")
        if identity:
            assert isinstance(identity.get("ns"), str) and identity["ns"].startswith("pid:[")
            assert len(identity.get("cmdline_sha256", "")) == 64
            current = table.identity(identity["pid"])
            assert current is None or current.zombie or list(current.start) != identity["start"]
    assert _fd_count() <= before


@pytest.mark.parametrize("name,source", [
    ("stdin", "import os\ndef handle(input, ctx):\n try: os.write(0,b'x')\n except OSError: return {'ro':True}\n return {'ro':False}\n"),
    ("fork", "import os\ndef handle(input,ctx):\n try: os.fork()\n except OSError: return {'denied':True}\n return {'denied':False}\n"),
    ("fork_bomb", "import os\ndef handle(input,ctx):\n while True: os.fork()\n"),
    ("proc_write", "def handle(input,ctx):\n try: open('/proc/self/comm','w').write('changed')\n except OSError: return {'denied':True}\n return {'denied':False}\n"),
    ("stdout", "import os\ndef handle(input,ctx):\n for _ in range(16384): os.write(1,b'x'*65536)\n return {}\n"),
    ("loop", "def handle(input,ctx):\n while True: pass\n"),
    ("cpu_loop", "import os\ndef handle(input,ctx):\n os.write(1,b'BUSY')\n while True: pass\n"),
    ("sigkill", "import os,signal\ndef handle(input,ctx):\n os.kill(os.getpid(),signal.SIGKILL)\n"),
    ("ignore_term", "import signal,time\ndef handle(input,ctx):\n signal.signal(signal.SIGTERM,signal.SIG_IGN)\n while True: time.sleep(.1)\n"),
    ("syntax", "def handle(:\n pass\n"),
])
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux /proc/self/fd and procfs verify launch cleanup",
)
def test_adversarial_launch_cleanup(tmp_path, name, source):
    before = _fd_count()
    root = tmp_path / "runtime"
    root.mkdir()
    unit = _unit(tmp_path, source)
    spec = _spec(unit, role="validator" if name == "syntax" else "payload",
                 cpu=1.0 if name == "cpu_loop" else 5.0,
                 wall=30.0 if name == "cpu_loop" else 15.0,
                 cap=2048 if name == "stdout" else 4096)
    with MutationExecutor(root, network_envelope="none") as executor:
        result = executor.run(spec)
    _scan(root, before)
    assert result.cleanup.ok
    if name == "stdin":
        assert b'"ro":true' in result.stdout
    elif name == "fork":
        assert b'"denied":true' in result.stdout
    elif name == "proc_write":
        assert b'"denied":true' in result.stdout
    elif name == "stdout":
        assert result.stdout_overflow and result.exit_code != 0
        assert result.wall_s < spec.limits.wall_s
    elif name == "cpu_loop":
        assert b"BUSY" in result.stdout and result.exit_code != 0
        assert not result.timed_out
    elif name == "syntax":
        assert b'"ok":false' in result.stdout
    else:
        assert result.exit_code != 0


def test_registration_precedes_release(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    unit = _unit(tmp_path, "def handle(input,ctx): return {'ok':True}\n")
    with MutationExecutor(root, network_envelope="none") as executor:
        outcome = executor.run(_spec(unit))
    assert outcome.exit_code == 0, outcome.stdout
    rows = [json.loads(line) for line in (root / "launches.jsonl").read_text().splitlines()]
    states = [row["state"] for row in rows]
    assert states == ["PREPARED", "STARTED", "REGISTERED", "RELEASED", "EXITED", "REAPED"]
    assert rows[2]["identity"] == outcome.identity


def test_invalid_document_leaves_no_stage(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    unit = _unit(tmp_path, "def handle(input,ctx): return {}\n")
    spec = _spec(unit)
    invalid = LaunchSpec(spec.role, spec.call, spec.launch, spec.unit_dir, {}, {},
                         b'{"wire":1,"call":"wrong","launch":"wrong","role":"payload"}', spec.limits)
    with MutationExecutor(root, network_envelope="none") as executor:
        with pytest.raises(ValueError, match="identity"):
            executor.run(invalid)
    assert not (root / "stage").exists()


def test_runtime_metadata_has_private_permissions(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    with MutationExecutor(root, network_envelope="none"):
        assert root.stat().st_mode & 0o077 == 0
        assert (root / "launches.jsonl").stat().st_mode & 0o077 == 0


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux /proc/self/fd and PID identity tracking",
)
def test_broker_starts_only_after_registration(tmp_path, monkeypatch):
    import kernel_mutate_exec as module

    engine = Path(module.__file__).parent
    mocked = tmp_path / "engine-copy"
    mocked.mkdir()
    for name in ("sf_mutate_boot.py", "sf_mutate_sdk.py", "sf_launch.py"):
        (mocked / name).write_bytes((engine / name).read_bytes())
    (mocked / "sf_mutate_fetch.py").write_text(
        "import sys\n"
        "assert sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(b'BROKER')\n")
    monkeypatch.setattr(module, "_ENGINE", mocked)
    root = tmp_path / "runtime"
    root.mkdir()
    launch = uuid.uuid4().hex
    doc = {"call": "c1", "launch": launch, "role": "broker", "request": {}}
    spec = LaunchSpec("broker", "c1", launch, None, {}, {}, json.dumps(doc).encode(),
                      MutationLimits(wall_s=15))
    with MutationExecutor(root, network_envelope="full") as executor:
        result = executor.run(spec)
    assert result.exit_code == 0 and result.stdout == b"BROKER"
    states = [json.loads(line)["state"] for line in (root / "launches.jsonl").read_text().splitlines()]
    assert states.index("REGISTERED") < states.index("RELEASED")
    _scan(root, _fd_count())


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux /proc/self/fd and PID identity tracking",
)
def test_engine_sigkill_reaps_payload_and_stage(tmp_path):
    before = _fd_count()
    root = tmp_path / "runtime"
    root.mkdir()
    unit = _unit(tmp_path, "import time\ndef handle(input,ctx):\n while True: time.sleep(.1)\n")
    launch = uuid.uuid4().hex
    helper = tmp_path / "owner.py"
    helper.write_text(
        "import json,sys\n"
        f"sys.path.insert(0,{str(Path(__file__).resolve().parents[1] / 'src/stateframe/engine')!r})\n"
        "from pathlib import Path\n"
        "from kernel_mutate_exec import MutationExecutor,LaunchSpec,MutationLimits\n"
        f"root=Path({str(root)!r}); unit=Path({str(unit)!r}); launch={launch!r}\n"
        "doc={'wire':1,'call':'c1','launch':launch,'role':'payload','input':{},"
        "'inputs':[],'responses':[],'grants':{'network':False,'writes':False}}\n"
        "spec=LaunchSpec('payload','c1',launch,unit,{},{},json.dumps(doc).encode(),"
        "MutationLimits(wall_s=60,cpu_s=30))\n"
        "with MutationExecutor(root,network_envelope='none') as ex: ex.run(spec)\n")
    process = subprocess.Popen([sys.executable, "-I", str(helper)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ledger = root / "launches.jsonl"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if ledger.exists() and '"state":"RELEASED"' in ledger.read_text():
                break
            if process.poll() is not None:
                pytest.fail(f"executor owner exited before release: {process.returncode}")
            time.sleep(.05)
        else:
            pytest.fail("executor owner did not publish release")
        signal_run_owned(root, process, signal.SIGKILL, deadline=time.monotonic(),
                         what="executor owner", require_checkpoint_saved=False)
        process.wait(timeout=10)
        from kernel_mutate_exec import MutationContainmentError, recover_orphaned_stages
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                recover_orphaned_stages(root)
                break
            except MutationContainmentError:
                time.sleep(.05)
        _scan(root, before)
    finally:
        if process.poll() is None:
            signal_run_owned(root, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="executor owner cleanup", require_checkpoint_saved=False)
            process.wait(timeout=10)
