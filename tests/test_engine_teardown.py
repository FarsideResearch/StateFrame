"""Real processes exercise the engine's completion, cancellation, and pipe ownership."""
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from textwrap import dedent

import pytest
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from _timeouts import scaled
from stateframe import runner
from stateframe.engine.kernel_exec_tracker import ProcessTable, capture_launch_identity

SRC = Path(__file__).resolve().parents[1] / "src"
RECEIPT = {"result": {"result": "SUBMITTED"}, "exit_status": 0}
END = '[sf] {"kind":"engine_end","result":"SUBMITTED"}'


def _drive(tmp_path, monkeypatch, source, printer=print, teardown_seconds=0.3):
    child = tmp_path / "engine.py"
    child.write_text(dedent(source), encoding="utf-8")
    monkeypatch.setattr(runner, "CHILD", child)
    monkeypatch.setattr(runner, "ENGINE_TEARDOWN_SECONDS", scaled(teardown_seconds))
    monkeypatch.setattr(runner, "OUTPUT_DRAIN_SECONDS", scaled(0.3))
    options = runner.RunOptions(repo=tmp_path, task="exercise teardown", endpoint=None)
    return runner.drive(tmp_path, options, printer=printer)


def _wait_for(predicate, message, timeout=60):
    """Wait for test-process progress; the long bound only catches a genuine hang."""
    deadline = time.monotonic() + scaled(timeout)
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail(message)
        time.sleep(0.05)


@pytest.mark.parametrize("status", [0, 1])
def test_published_result_survives_stuck_exit_and_inherited_output(tmp_path, monkeypatch, status):
    receipt = json.dumps({**RECEIPT, "exit_status": status})
    assert _drive(tmp_path, monkeypatch, f"""
        import subprocess, sys, time
        from pathlib import Path
        peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(12)"],
                                start_new_session=True)
        Path("peer").write_text(str(peer.pid))
        Path("engine-result.json").write_text({receipt!r})
        print({END!r}, flush=True)
        print("unterminated tail", end="", flush=True)
        time.sleep(12)
        """) == status
    assert (tmp_path / "engine-result.json").read_text() == receipt
    assert (tmp_path / "engine.log").read_text().endswith("unterminated tail")
    peer = ProcessTable().identity(int((tmp_path / "peer").read_text()))
    assert peer is None or peer.zombie


@pytest.mark.parametrize("stale_receipt", [False, True])
def test_unconfirmed_marker_and_eof_do_not_stop_live_work(tmp_path, monkeypatch, stale_receipt):
    if stale_receipt:
        (tmp_path / "engine-result.json").write_text(json.dumps(RECEIPT))
    assert _drive(tmp_path, monkeypatch, f"""
        import os, time
        from pathlib import Path
        print({END!r}, flush=True)
        os.close(1)
        os.close(2)
        time.sleep({scaled(0.8)})
        Path("work-finished").touch()
        """) == 0
    assert (tmp_path / "work-finished").exists()


def test_exit_with_open_inherited_pipe_drains_only_for_grace(tmp_path, monkeypatch):
    assert _drive(tmp_path, monkeypatch, """
        import subprocess, sys
        from pathlib import Path
        peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(12)"])
        Path("peer").write_text(str(peer.pid))
        print("last line", flush=True)
        """) == 0
    assert "last line\n" in (tmp_path / "engine.log").read_text()
    pid = int((tmp_path / "peer").read_text())
    # SIGKILL delivery can precede scheduling the orphan to exit.
    for _ in range(30):
        peer = ProcessTable().identity(pid)
        if peer is None or peer.zombie:
            break
        threading.Event().wait(scaled(0.02))
    else:
        pytest.fail("process holding the inherited pipe survived engine exit")


def test_child_watchdog_retains_result_if_terminal_line_is_garbled(tmp_path, monkeypatch):
    receipt = json.dumps(RECEIPT)
    assert _drive(tmp_path, monkeypatch, f"""
        import sys, threading
        from pathlib import Path
        sys.path.insert(0, {str(SRC)!r})
        from stateframe import _child
        _child.TEARDOWN_GRACE_SECONDS = {scaled(0.3)}
        _child._reap_workers = threading.Event().wait

        def main(argv, *, teardown):
            Path("engine-result.json").write_text({receipt!r})
            print("inherited output " + {END!r}, flush=True)
            teardown.begin()
            return 0

        _child.main = main
        raise SystemExit(_child._run_cli(sys.argv[:1]))
        """) == 0
    assert (tmp_path / "engine-result.json").read_text() == receipt
    assert "inherited output " + END in (tmp_path / "engine.log").read_text()


@pytest.mark.parametrize("responds", [False, True])
def test_interrupt_requests_stop_before_escalating(tmp_path, monkeypatch, responds):
    grace = scaled(1.0)
    descendant_source = f"""
        import atexit, os, signal, time
        from pathlib import Path
        root = Path.cwd()

        def stop(signum, _frame):
            (root / ("descendant-signal-" + str(signum))).touch()
            (root / "descendant-stop-requested").touch()
            if {responds!r}:
                raise SystemExit(0)

        signal.signal(signal.SIGTERM, stop)
        atexit.register(lambda: (root / "descendant-exited").touch())
        ready_tmp = root / "descendant-ready.tmp"
        ready_tmp.write_text(
            "{{}} {{}}".format(os.getpid(), os.getpgid(0)), encoding="ascii")
        ready_tmp.replace(root / "descendant-ready")
        while True:
            time.sleep(0.02)
    """
    source = f"""
        import os, signal, subprocess, sys, time
        from pathlib import Path
        root = Path.cwd()
        descendant_source = {descendant_source!r}
        descendant = subprocess.Popen(
            [sys.executable, "-c", descendant_source],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        (root / "descendant-pid").write_text(str(descendant.pid), encoding="ascii")
        ready_deadline = time.monotonic() + {scaled(60)!r}
        while not (root / "descendant-ready").exists():
            if time.monotonic() >= ready_deadline:
                raise SystemExit("descendant failed to become ready")
            time.sleep(0.02)

        def stop(*_):
            (root / "stop-requested").touch()
            if {responds!r}:
                while not (root / "descendant-exited").exists():
                    time.sleep(0.01)
                descendant.wait()
                print("graceful cleanup", flush=True)
                raise SystemExit(0)

        signal.signal(signal.SIGTERM, stop)
        (root / "leader-ready").write_text(
            "{{}} {{}}".format(os.getpid(), os.getpgid(0)), encoding="ascii")
        print('[sf] {{"kind":"engine_start","turns":1}}', flush=True)
        while True:
            time.sleep(0.02)
    """

    process_holder = []
    escalation_order = []
    stop_request_started = []
    original_popen = runner.subprocess.Popen

    def record_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        process_holder.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", record_process)
    original_request = runner._request_engine_stop

    def record_stop_request_time(*args, **kwargs):
        stop_request_started.append(time.monotonic())
        return original_request(*args, **kwargs)

    monkeypatch.setattr(runner, "_request_engine_stop", record_stop_request_time)

    original_kill_tree = runner._kill_engine_tree
    original_stop_group = runner._stop_group

    def record_kill_tree(process, reason):
        escalation_order.append(("engine tree", (tmp_path / "descendant-stop-requested").exists(),
                                 time.monotonic()))
        return original_kill_tree(process, reason)

    def record_stop_group(process):
        escalation_order.append(("process group", (tmp_path / "descendant-stop-requested").exists(),
                                 time.monotonic()))
        return original_stop_group(process)

    monkeypatch.setattr(runner, "_kill_engine_tree", record_kill_tree)
    monkeypatch.setattr(runner, "_stop_group", record_stop_group)

    descendant_identity = []
    launched_after = time.time()

    def interrupt(_):
        ready = tmp_path / "descendant-ready"
        descendant_identity.append(record_identity(
            int(ready.read_text().split()[0]), launched_after=launched_after,
            command_fragment="descendant-ready"))
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            _drive(tmp_path, monkeypatch, source, printer=interrupt, teardown_seconds=1.0)
        _wait_for(lambda: (tmp_path / "descendant-ready").exists(),
                  "descendant readiness handshake did not complete")
        leader_pid, leader_pgid = map(int, (tmp_path / "leader-ready").read_text().split())
        descendant_pid, descendant_pgid = map(int, (tmp_path / "descendant-ready").read_text().split())
        assert process_holder[0].pid == leader_pid
        assert leader_pid == leader_pgid
        assert descendant_pgid == leader_pgid, "descendant escaped the engine launch group"
        assert (tmp_path / "stop-requested").exists()
        assert (tmp_path / "descendant-stop-requested").exists(), (
            "descendant did not acknowledge SIGTERM from the launch group")
        assert (tmp_path / f"descendant-signal-{signal.SIGTERM}").exists()
        assert escalation_order, "runner performed no final process-group cleanup"
        assert all(received for _, received, _ in escalation_order), (
            f"escalation preceded descendant SIGTERM acknowledgement: {escalation_order!r}")
        assert ("engine tree" in [kind for kind, _, _ in escalation_order]) is (not responds)
        if not responds:
            assert stop_request_started, "runner never began the stop request"
            assert all(at - stop_request_started[0] >= grace for _, _, at in escalation_order), (
                f"runner sent SIGKILL before the configured grace elapsed: {escalation_order!r}")
        assert process_holder[0].returncode is not None, "driver left the child running"
        if responds:
            assert process_holder[0].returncode == 0
        engine_log = (tmp_path / "engine.log").read_text()
        assert ("graceful cleanup" in engine_log) is responds
        if not responds:
            assert engine_log.splitlines() == ['[sf] {"kind":"engine_start","turns":1}']
        _wait_for(
            lambda: (lambda identity: identity is None or identity.zombie)(
                ProcessTable().identity(descendant_pid)),
            "descendant survived process-group cleanup")
    finally:
        if process_holder:
            # Signal the group only while the leader is unreaped: a reaped leader's group id
            # may already belong to someone else.
            try:
                if process_holder[0].poll() is None:
                    signal_run_owned(tmp_path, process_holder[0], signal.SIGKILL, deadline=0,
                                     what="teardown cleanup", require_checkpoint_saved=False,
                                     group=True)
            except OSError:
                pass
            try:
                process_holder[0].wait(timeout=scaled(60))
            except subprocess.TimeoutExpired:
                pytest.fail("test engine leader did not exit after process-group cleanup")
        for expected in descendant_identity:
            try:
                signal_run_verified(tmp_path, expected.pid, expected, signal.SIGKILL, deadline=0,
                                    what="teardown cleanup", require_checkpoint_saved=False)
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass  # already gone, exiting, or no longer the recorded process


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux orphan adoption")
def test_child_reaps_orphans_and_allows_normal_python_finalizers(tmp_path):
    script = dedent("""
        import atexit, os, signal, sys, time
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from stateframe import _child
        root = Path(sys.argv[2])

        def main(argv, *, teardown):
            if (launcher := os.fork()) == 0:
                if os.fork() == 0:
                    os.setsid()
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    (root / "peer").write_text(str(os.getpid()))
                    time.sleep(12)
                os._exit(0)
            os.waitpid(launcher, 0)
            while not (root / "peer").exists():
                time.sleep(.01)
            if (zombie := os.fork()) == 0:
                os._exit(0)
            os.waitid(os.P_PID, zombie, os.WEXITED | os.WNOWAIT)
            (root / "zombie").write_text(str(zombie))
            atexit.register(lambda: (root / "finalized").touch())
            teardown.begin()
            return 0

        _child.main = main
        raise SystemExit(_child._run_cli(sys.argv[:1]))
        """)
    with subprocess.Popen([sys.executable, "-c", script, str(SRC), str(tmp_path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True) as process:
        capture_launch_identity(process, session_leader=True)
        try:
            _, stderr = process.communicate(timeout=scaled(5))
            assert process.returncode == 0, stderr.decode()
        finally:
            if process.returncode is None:
                runner._kill_engine_tree(process, "test child did not exit")
                runner._stop_group(process)
    assert (tmp_path / "finalized").exists()
    table = ProcessTable()
    for name in ("peer", "zombie"):
        assert table.identity(int((tmp_path / name).read_text())) is None


def test_lookup_timeout_keeps_completed_sites_and_misses(tmp_path, monkeypatch, caplog):
    from repo_intel_core import lsp_resolve as resolver

    release = threading.Event()
    sites = [resolver.CallSiteRef(n, 0, "target") for n in range(1, 4)]
    (tmp_path / "sample.py").touch()
    target = resolver.ResolvedTarget("sample.py", 3, "target", "function")

    def lookup(root, path, source, refs, deadline, results):
        results[refs[0].key] = target
        results[refs[1].key] = None
        release.wait(scaled(3))
        results[refs[2].key] = target

    monkeypatch.setattr(resolver, "is_enabled", lambda: True)
    monkeypatch.setattr(resolver, "_HARD_TIMEOUT_GRACE_SECONDS", scaled(0.1))
    monkeypatch.setattr(resolver, "_resolve_sites_blocking", lookup)
    try:
        result = resolver.resolve_file_full(tmp_path, "sample.py", sites, timeout=0)
    finally:
        release.set()
    assert result == {sites[0].key: target, sites[1].key: None}
    assert "salvaged 2/3" in caplog.text
