"""Real processes exercise the engine's completion, cancellation, and pipe ownership."""
import json
import subprocess
import sys
import threading
from pathlib import Path
from textwrap import dedent

import pytest
from _timeouts import scaled
from stateframe import runner
from stateframe.engine.kernel_exec_tracker import ProcessTable, capture_launch_identity

SRC = Path(__file__).resolve().parents[1] / "src"
RECEIPT = {"result": {"result": "SUBMITTED"}, "exit_status": 0}
END = '[sf] {"kind":"engine_end","result":"SUBMITTED"}'


def _drive(tmp_path, monkeypatch, source, printer=print):
    child = tmp_path / "engine.py"
    child.write_text(dedent(source), encoding="utf-8")
    monkeypatch.setattr(runner, "CHILD", child)
    monkeypatch.setattr(runner, "ENGINE_TEARDOWN_SECONDS", scaled(0.3))
    monkeypatch.setattr(runner, "OUTPUT_DRAIN_SECONDS", scaled(0.3))
    options = runner.RunOptions(repo=tmp_path, task="exercise teardown", endpoint=None)
    return runner.drive(tmp_path, options, printer=printer)


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
    def interrupt(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _drive(tmp_path, monkeypatch, f"""
            import signal, time
            from pathlib import Path
            def stop(*_):
                Path("stop-requested").touch()
                if {responds!r}:
                    print("graceful cleanup", flush=True)
                    raise SystemExit(0)
            signal.signal(signal.SIGTERM, stop)
            print('[sf] {{"kind":"engine_start","turns":1}}', flush=True)
            time.sleep(12)
            """, printer=interrupt)
    assert (tmp_path / "stop-requested").exists()
    assert ("graceful cleanup" in (tmp_path / "engine.log").read_text()) == responds


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
