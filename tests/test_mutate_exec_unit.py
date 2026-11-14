"""Mutation execution contracts that do not need a host namespace."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from kernel_exec_capability import ExecCapability, ExecLimits, capability_from_dict
from kernel_exec_session import ExecSession, ExecSessionError


@pytest.mark.host_sandbox
def test_launch_stage_hook_observes_fsynced_ledger(tmp_path):
    from kernel_mutate_exec import MutationExecutor

    root = tmp_path / ("a" * 32)
    root.mkdir()
    seen = []

    def hook(stage):
        rows = (root / "launches.jsonl").read_text().splitlines()
        assert rows and json.loads(rows[-1])["state"] == stage
        seen.append(stage)

    executor = MutationExecutor(root, network_envelope="none", launch_stage_hook=hook)
    try:
        executor._record(SimpleNamespace(launch="l", call="c", role="payload"), "PREPARED")
        assert seen == ["PREPARED"]
    finally:
        executor.close()

def _cap(root: Path, **changes) -> ExecCapability:
    base = ExecCapability("mutation", root, False, (root,), (), "none", {}, ExecLimits())
    return replace(base, **changes)


def test_mutation_purposes_and_output_limit_serialization(tmp_path):
    cap = _cap(tmp_path)
    assert "kill_on_output" not in cap.as_dict()["limits"]
    assert capability_from_dict(cap.as_dict()).digest() == cap.digest()
    broker = replace(cap, purpose="mutation_broker", network="full",
                     limits=replace(cap.limits, kill_on_output=True))
    assert broker.as_dict()["limits"]["kill_on_output"] is True
    assert capability_from_dict(broker.as_dict()).digest() == broker.digest()
    with pytest.raises((TypeError, ValueError)):
        replace(cap.limits, kill_on_output=1)


def test_session_rejects_network_and_read_widening(tmp_path):
    extra = tmp_path.parent / (tmp_path.name + "-extra")
    extra.mkdir()
    with ExecSession(tmp_path) as session:
        session.configure(_cap(tmp_path))
        session._start(_cap(tmp_path))
        with pytest.raises(ExecSessionError, match="network"):
            session._start(_cap(tmp_path, network="full"))
        with pytest.raises(ExecSessionError, match="read"):
            session._start(_cap(tmp_path, read_roots=(tmp_path, extra)))


def test_unconfigured_session_also_pins_first_envelope(tmp_path):
    extra = tmp_path.parent / (tmp_path.name + "-extra")
    extra.mkdir()
    with ExecSession(tmp_path) as session:
        session._start(_cap(tmp_path))
        with pytest.raises(ExecSessionError, match="network"):
            session._start(_cap(tmp_path, network="full"))
        with pytest.raises(ExecSessionError, match="read"):
            session._start(_cap(tmp_path, read_roots=(tmp_path, extra)))


def test_sdk_need_more_is_not_an_exception():
    from sf_mutate_sdk import NeedMore, MutationContext
    assert issubclass(NeedMore, BaseException)
    assert not issubclass(NeedMore, Exception)
    ctx = MutationContext({"call": "c", "inputs": [], "responses": [],
                           "grants": {"network": True, "writes": False}})
    with pytest.raises(NeedMore):
        try:
            ctx.fetch("https://example.test/a", "text/html")
        except Exception:
            pytest.fail("unit caught NeedMore as Exception")
    assert ctx.pending_fetches == [{"method": "GET", "url": "https://example.test/a",
                                    "accept": "text/html"}]


def test_boot_release_gate_exits_before_unit_import(tmp_path, monkeypatch):
    from sf_mutate_boot import wait_for_release
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    assert wait_for_release(ctl, "launch", "nonce", timeout=0.01) is None
    (ctl / "release.json").write_text(json.dumps({"launch": "launch", "nonce": "wrong"}))
    assert wait_for_release(ctl, "launch", "nonce", timeout=0.01) is None


def test_boot_release_gate_allows_scheduled_host_delay(tmp_path, monkeypatch):
    import sf_mutate_boot as boot

    ctl = tmp_path / "ctl"
    ctl.mkdir()
    release_path = ctl / "release.json"
    release_path.write_text(json.dumps({"launch": "launch", "nonce": "nonce",
                                        "deadline": 100, "identity": {"start": "1"}}))
    now = [0.0]

    class DelayedFile:
        def __init__(self, path):
            self.path = Path(path)

        def read_bytes(self):
            if self.path == release_path and now[0] < 3:
                raise FileNotFoundError(self.path)
            return self.path.read_bytes()

    class DelayedPath:
        def __init__(self, path):
            self.path = Path(path)

        def __truediv__(self, name):
            return DelayedFile(self.path / name)

    monkeypatch.setattr(boot, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda _seconds: now.__setitem__(0, now[0] + 1)))
    monkeypatch.setattr(boot, "Path", DelayedPath)

    assert boot.wait_for_release(ctl, "launch", "nonce") == {
        "launch": "launch", "nonce": "nonce", "deadline": 100, "identity": {"start": "1"}}
    assert now[0] == 3


@pytest.mark.parametrize("raw", [b'{"wire":1,"wire":1}', b'{"n":1e999}',
                                  b'{"n":12345678901234567890123456789012345678901}'])
def test_boot_rejects_nonstrict_json(monkeypatch, raw):
    import sf_mutate_boot as boot
    monkeypatch.setattr(boot.os, "read", lambda _fd, _size: raw)
    with pytest.raises(ValueError):
        boot._read_doc()


def test_macos_profile_deny_default():
    from sf_mutate_boot import macos_profile
    profile = macos_profile(("/usr/lib", "/sf/boot", "/sf/unit", "/in", "/resp"))
    for rule in ("(deny default)", "(deny process-fork)", "(deny process-exec)",
                 "(deny mach-lookup)", '"/dev/null"'):
        assert rule in profile
    assert "(allow default)" not in profile


def test_macos_broker_profile_grants_network_without_file_writes():
    from sf_mutate_boot import macos_profile
    profile = macos_profile(("/usr/lib", "/sf/boot"), network_full=True)
    assert "(deny default)" in profile
    assert "(allow network*)" in profile
    assert "(deny process-fork)" in profile and "(deny process-exec)" in profile
    assert '(allow file-write* (literal "/dev/null"))' in profile


def test_macos_activation_fails_closed(monkeypatch):
    import sf_mutate_boot as boot

    class SandboxCall:
        def __call__(self, *_args):
            return -1

    class Library:
        sandbox_init_with_parameters = SandboxCall()

    monkeypatch.setattr(boot.ctypes, "CDLL", lambda *_args, **_kwargs: Library())
    with pytest.raises(RuntimeError, match="activation failed"):
        boot.activate_macos_sandbox(("/sf/boot", "/sf/unit"))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux seccomp syscall 435 and prctl are required")
def test_post_init_filter_deny_syscalls_in_child(tmp_path):
    import subprocess
    code = ("import ctypes, errno, os, sys; "
            "sys.path.insert(0, sys.argv[1]); from sf_launch import install_syscall_filter; "
            "install_syscall_filter(network_none=True, post_init=True); "
            "assert ctypes.CDLL(None, use_errno=True).syscall(435, 0, 0) == -1; "
            "assert ctypes.get_errno() == errno.EPERM; "
            "print('DENIED')")
    engine = str(Path(__file__).resolve().parents[1] / "src/stateframe/engine")
    result = subprocess.run([sys.executable, "-I", "-c", code, engine],
                            capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == b"DENIED"


def test_executor_cleanup_failure_is_containment_error(tmp_path, monkeypatch):
    from kernel_exec_session import CleanupReport
    from kernel_mutate_exec import MutationContainmentError, MutationExecutor
    root = tmp_path / "runtime"
    root.mkdir()
    executor = MutationExecutor(root, network_envelope="none")
    monkeypatch.setattr(executor._session, "stop_all",
                        lambda *_args: CleanupReport(False, 0, (123,), "survivor"))
    with pytest.raises(MutationContainmentError, match="survivor"):
        executor.close()


def test_runtime_closure_scan_is_cached(monkeypatch):
    import kernel_mutate_exec as module
    module._runtime_roots_cached.cache_clear()
    original = module.subprocess.run
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", observed)
    assert module._runtime_roots() == module._runtime_roots()
    assert len(calls) == (1 if sys.platform == "linux" else 0)


def test_host_identity_skips_foreign_same_tick_process(monkeypatch, tmp_path):
    """A process of another user with the same start tick must not break identity translation."""
    import sys
    from types import SimpleNamespace

    import kernel_exec_session

    if not sys.platform.startswith("linux"):
        pytest.skip("host PID translation is Linux-only")
    rows = {
        101: SimpleNamespace(pid=101, start=(7, 7), zombie=False, pgid=101, sid=101),
        202: SimpleNamespace(pid=202, start=(7, 7), zombie=False, pgid=202, sid=202),
    }

    class FakeTable:
        def snapshot(self):
            return rows

        @staticmethod
        def pid_namespace(pid):
            if pid == 101:
                raise PermissionError(13, "Permission denied", f"/proc/{pid}/ns/pid")
            return "pid:[1]"

    proc = tmp_path / "proc"
    (proc / "202").mkdir(parents=True)
    (proc / "202" / "status").write_text("Name:\tpython3\nNSpid:\t202\t5\n")
    real_path = kernel_exec_session.Path
    monkeypatch.setattr(kernel_exec_session, "ProcessTable", FakeTable)
    monkeypatch.setattr(kernel_exec_session, "Path",
                        lambda value, *rest: real_path(str(proc) if value == "/proc" else value, *rest))
    identity = {"pid": 5, "start": [7, 7], "ns": "pid:[1]", "cmdline_sha256": "c" * 64}
    monkeypatch.setattr(kernel_exec_session, "process_cmdline_digest",
                        lambda pid: identity["cmdline_sha256"] if pid == 202 else pytest.fail("unexpected PID"))
    resolved = kernel_exec_session._host_mutation_identity(identity)
    assert resolved["pid"] == 202 and resolved["ns_pid"] == 5


def test_sdk_input_resolves_id_or_unambiguous_file_name(tmp_path):
    """Resolve input paths as well as IDs, and report available inputs on failure."""
    from sf_mutate_sdk import MutationContext, MutationError

    (tmp_path / "in0").write_bytes(b"alpha")
    (tmp_path / "in1").write_bytes(b"beta")
    doc = {"call": "c", "inputs": [
        {"id": "in0", "path": str(tmp_path / "in0"), "bytes": 5, "name": "events.b64"},
        {"id": "in1", "path": str(tmp_path / "in1"), "bytes": 4, "name": "notes.txt"}]}
    ctx = MutationContext(doc)
    assert ctx.input_bytes("in0") == b"alpha"
    assert ctx.input_bytes("data/events.b64") == b"alpha"
    assert ctx.input_bytes("notes.txt") == b"beta"
    with pytest.raises(MutationError, match=r"inputs: in0 \(events.b64\), in1 \(notes.txt\)"):
        ctx.input_bytes("data/missing.bin")
    twin = MutationContext({"call": "c", "inputs": [
        {"id": "in0", "path": str(tmp_path / "in0"), "bytes": 5, "name": "a.txt"},
        {"id": "in1", "path": str(tmp_path / "in1"), "bytes": 4, "name": "a.txt"}]})
    with pytest.raises(MutationError, match="unknown input id"):
        twin.input_bytes("a.txt")
