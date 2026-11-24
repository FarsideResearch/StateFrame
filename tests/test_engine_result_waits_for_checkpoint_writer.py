"""Exercise real checkpoint children held across terminal result publication."""

from contextlib import ExitStack, contextmanager, suppress
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest

from fixtures.process_signals import (
    kernel_start_identity, record_identity, signal_run_owned, signal_run_verified,
)


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="uses fork, waitid, and procfs")

@pytest.fixture(autouse=True)
def default_sigint():
    # A background job may start pytest with SIGINT ignored, and children inherit that.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


ENGINE_CHILD = r"""
import io
import json
import os
from pathlib import Path
import signal
import sys
import time

import kernel_checkpoint as checkpoint
from probes import react_repair
from stateframe import _child

run = Path(sys.argv[1])
mode = sys.argv[2]
notice_fd, gate_fd = map(int, sys.argv[3:5])
root = run / "checkpoints"
root.mkdir()

if mode == "blocked-alarm":
    checkpoint.CHILD_TIMEOUT_S = 1
elif mode == "progress":
    checkpoint.CHILD_TIMEOUT_S = 2

def notice(kind, **fields):
    os.write(notice_fd, (json.dumps({"kind": kind, **fields}) + "\n").encode())

def fail_write():
    raise OSError(28, "synthetic disk failure")

def write_checkpoint():
    if mode in {"early-exit", "already-reaped"}:
        os._exit(91)
    notice("writer_ready", pid=os.getpid())
    os.read(gate_fd, 1)
    if mode in {"exit1", "broken-stderr"}:
        fail_write()
    if mode == "progress":
        started = time.monotonic()
        writer = checkpoint._StateWriter(io.BytesIO())
        for _ in range(9):
            time.sleep(0.3)
            writer.write(b"serialized state")
        writer.finish()
        notice("progress_complete", elapsed=time.monotonic() - started)
    return {"sends": 1, "state_bytes": 0, "chunks": ()}

def work(_row, **_kwargs):
    if mode == "broken-stderr":
        for step in (5, 6):
            checkpoint._fork_writer(run, step, root / f"saved-{step}", fail_write, root / f"tmp-{step}")
            checkpoint.finish_background_writer()
    tmp = root / "temporary"
    tmp.mkdir()
    (tmp / "partial").touch()
    checkpoint._fork_writer(run, 7, root / "saved", write_checkpoint, tmp)
    pid = checkpoint._STATE["child"]
    emit.writer_pid = pid
    if mode == "already-reaped":
        assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 91
    if mode == "broken-stderr":
        reader, writer = os.pipe()
        os.close(reader)
        os.dup2(writer, 2)
        os.close(writer)
    notice("task_returning", pid=pid)
    if mode == "engine-error":
        raise ValueError("synthetic task failure")
    return {"result": "SUBMITTED"}

def save_and_restore_next_checkpoint():
    checkpoint._is_state_module = lambda *_args: False
    checkpoint._context_items = lambda: []
    checkpoint.contextvars.copy_context = checkpoint.contextvars.Context
    checkpoint.install_continuous_clock = lambda _saved: None
    checkpoint._verify_stateframe_home_identity = lambda *_args: None
    checkpoint.register_signing_key(b"checkpoint-test-key-0123456789abcdef")

    def checkpoint_roundtrip():
        value = "saved"
        checkpoint._STATE["setup"] = {
            "code": sys._getframe().f_code, "line": 0, "funcs": frozenset(),
        }
        target = checkpoint.save(sys._getframe(), 8, run)
        value = "changed"
        checkpoint.restore(sys._getframe(), target, run)
        assert value == "saved"
        notice("next_checkpoint_restored", checkpoint=target.name, value=value)

    checkpoint_roundtrip()

waitpid = os.waitpid
wait_announced = False

def observe_waitpid(pid, options):
    global wait_announced
    if pid > 0 and pid == checkpoint._STATE.get("child") and checkpoint._STATE["child_step"] == 7:
        if not wait_announced:
            wait_announced = True
            assert not (run / "engine-result.json").exists(), "receipt published before writer wait"
            notice("waiting_for_writer", pid=pid)
        assert not options & os.WUNTRACED, "a job stop must not be taken for an end"
    return waitpid(pid, options)

original_emit = _child.emit

def emit(kind, **fields):
    if kind == "engine_end":
        pid = emit.writer_pid
        try:
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            reaped = True
        else:
            reaped = False
        notice(kind, reaped=reaped, repeated=checkpoint.finish_background_writer())
    original_emit(kind, **fields)

checkpoint.os.waitpid = observe_waitpid
react_repair.work = work
_child.emit = emit
checkpoint.verify_stateframe_home_identity = lambda *_args, **_kwargs: None
finish_checkpoint_writer = _child._finish_checkpoint_writer

def finish_and_restore_next_checkpoint(teardown):
    outcome = finish_checkpoint_writer(teardown)
    if mode == "job-control":
        save_and_restore_next_checkpoint()
    return outcome

_child._finish_checkpoint_writer = finish_and_restore_next_checkpoint
if mode == "before-main":
    def wait_before_main(_teardown):
        notice("before_main")
        os.read(gate_fd, 1)

    _child.watch_parent = wait_before_main
raise SystemExit(_child._run_cli(["engine", str(run)]))
"""


class EngineRun:
    def __init__(self, process, notices, gate, run, launched_at):
        self.process = process
        self.notices = notices
        self.gate = gate
        self.run = run
        self.launched_at = launched_at
        self.buffer = b""
        self.events = {}

    def observe(self, kind):
        until = time.monotonic() + 8
        while kind not in self.events:
            assert "engine_end" not in self.events, f"engine ended before {kind}; saw {self.events}"
            if b"\n" not in self.buffer:
                ready, _, _ = select.select([self.notices], [], [], max(0, until - time.monotonic()))
                assert ready, f"missing {kind}; saw {self.events}"
                new = os.read(self.notices, 4096)
                assert new, f"engine closed notice pipe before {kind}; saw {self.events}"
                self.buffer += new
                continue
            line, self.buffer = self.buffer.split(b"\n", 1)
            event = json.loads(line)
            self.events[event["kind"]] = event
        return self.events[kind]

    def assert_writer_is_held(self):
        pid = self.observe("task_returning")["pid"]
        assert self.observe("writer_ready")["pid"] == pid
        assert self.observe("waiting_for_writer")["pid"] == pid
        assert not (self.run / "engine-result.json").exists()

    def release(self):
        if self.gate is not None:
            os.close(self.gate)
            self.gate = None

    def signal_writer(self, sig):
        pid = self.events["task_returning"]["pid"]
        identity = record_identity(pid, launched_after=self.launched_at, command_fragment=str(self.run))
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        assert int(stat[1]) == self.process.pid
        signal_run_verified(self.run, pid, identity, sig, deadline=time.monotonic(),
                            what="held checkpoint writer", require_checkpoint_saved=False)

    def process_state(self, pid, identity):
        assert kernel_start_identity(pid) == identity.kernel_identity
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0]

    def wait_for_process_state(self, pid, identity, *, stopped):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            state = self.process_state(pid, identity)
            if (state in {"T", "t"}) is stopped:
                return state
            time.sleep(0.01)
        raise AssertionError(f"process {pid} did not become {'stopped' if stopped else 'running'}")

    def stop(self, *, group):
        self.signal_engine(signal.SIGTERM, group=group,
                           what="engine waiting for its checkpoint writer")

    def signal_engine(self, sig, *, group=False, what):
        signal_run_owned(self.run, self.process, sig, deadline=time.monotonic(),
                         what=what, require_checkpoint_saved=False, group=group)

    def assert_result(self, expected_writer, *, status=0, result="SUBMITTED"):
        output, _ = self.process.communicate(timeout=8)
        assert self.process.returncode == status, output
        terminal = json.loads((self.run / "engine-result.json").read_text())
        assert terminal["exit_status"] == status
        assert terminal["result"]["result"] == result
        assert set(terminal) == {"result", "seconds", "exit_status"}
        assert self.observe("engine_end") == {
            "kind": "engine_end", "reaped": True, "repeated": expected_writer,
        }
        events = [json.loads(line) for line in (self.run / "checkpoints/events.jsonl").read_text().splitlines()]
        finishes = [row for row in events if row["event"] == "checkpoint_writer_finished" and row["step"] == 7]
        assert len(finishes) == 1
        assert finishes[0]["exit_code"] == expected_writer["exit_code"]
        if expected_writer["outcome"] == "failed":
            assert not (self.run / "checkpoints/temporary").exists()


@contextmanager
def launch(tmp_path, mode, *, blocked=(), ignored=()):
    src = Path(__file__).resolve().parents[1] / "src"
    engine = src / "stateframe" / "engine"
    run = tmp_path / "run"
    repo = tmp_path / "repo"
    run.mkdir()
    repo.mkdir()
    (repo / "sample.py").write_text("def sample():\n    return 1\n")
    (run / "spec.json").write_text(json.dumps({
        "engine_dir": str(engine),
        "parent_snapshot": str(repo),
        "public_row": {"msg": "synthetic checkpoint teardown"},
        "model": "fixture",
        "chat_url": "http://127.0.0.1:1/v1/chat/completions",
        "max_turns": 1,
        "environment": {"STATEFRAME_HOME": str(tmp_path / "home"), "STATEFRAME_CHECKPOINT": "off"},
    }))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(src), str(engine))))
    for name in ("VIRTUAL_ENV", "CONDA_PREFIX", "CODE_GRAPH_LSP"):
        env.pop(name, None)

    with ExitStack() as fds:
        notices, notice_writer = os.pipe()
        fds.callback(os.close, notices)
        gate_reader, gate = os.pipe()
        launched_at = time.time()
        previous_handlers = {sig: signal.getsignal(sig) for sig in ignored}
        for sig in ignored:
            signal.signal(sig, signal.SIG_IGN)
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", ENGINE_CHILD, str(run), mode,
                 str(notice_writer), str(gate_reader)],
                cwd=run, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                pass_fds=(notice_writer, gate_reader), process_group=0,
            )
        except BaseException:
            os.close(gate)
            raise
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            os.close(notice_writer)
            os.close(gate_reader)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        fixture = EngineRun(process, notices, gate, run, launched_at)
        try:
            yield fixture
        finally:
            fixture.release()
            try:
                process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                if "task_returning" in fixture.events:
                    with suppress(ProcessLookupError):
                        fixture.signal_writer(signal.SIGKILL)
                with suppress(ProcessLookupError):
                    fixture.stop(group=True)
                process.communicate(timeout=8)


def writer_outcome(code, reason="unexpected writer error"):
    outcome = {"step": 7, "outcome": "saved" if code == 0 else "failed", "exit_code": code}
    if code != 0:
        outcome["reason"] = reason
    return outcome


@pytest.mark.parametrize("mode,code", [("saved", 0), ("killed", -signal.SIGKILL),
                                       ("exit1", 1),
                                       ("engine-error", 0)])
def test_engine_end_waits_for_last_checkpoint_writer(tmp_path, mode, code):
    with launch(tmp_path, mode) as fixture:
        fixture.assert_writer_is_held()
        if mode == "killed":
            fixture.signal_writer(signal.SIGKILL)
        else:
            fixture.release()
        if mode == "engine-error":
            fixture.assert_result(writer_outcome(code), status=1, result="ENGINE_ERROR")
        elif mode == "exit1":
            fixture.assert_result(writer_outcome(code, "checkpoint write failed: ENOSPC"))
        else:
            fixture.assert_result(writer_outcome(code))


def test_stopped_writer_is_waited_for_and_its_save_completes(tmp_path):
    with launch(tmp_path, "saved") as fixture:
        fixture.assert_writer_is_held()
        fixture.signal_writer(signal.SIGSTOP)
        time.sleep(0.3)
        assert fixture.process.poll() is None
        assert not (fixture.run / "engine-result.json").exists()
        fixture.signal_writer(signal.SIGCONT)
        fixture.release()
        fixture.assert_result(writer_outcome(0))


def test_operator_stop_ends_a_stopped_writer(tmp_path):
    with launch(tmp_path, "stop", blocked={signal.SIGTERM}) as fixture:
        fixture.assert_writer_is_held()
        fixture.signal_writer(signal.SIGSTOP)
        fixture.stop(group=False)
        output, _ = fixture.process.communicate(timeout=8)
        assert fixture.process.returncode == 0, output
        assert (fixture.run / "engine-result.json").exists()
        assert output.count("run was stopped while the last checkpoint was still being written") == 1, output


def test_writer_alarm_works_when_launcher_blocks_it(tmp_path):
    with launch(tmp_path, "blocked-alarm", blocked={signal.SIGALRM}) as fixture:
        fixture.assert_writer_is_held()
        fixture.assert_result(writer_outcome(-signal.SIGALRM))


def test_writer_can_work_for_longer_than_its_watchdog_interval(tmp_path):
    with launch(tmp_path, "progress") as fixture:
        fixture.assert_writer_is_held()
        fixture.release()
        fixture.assert_result(writer_outcome(0))
        assert fixture.observe("progress_complete")["elapsed"] > 2


@pytest.mark.parametrize("group", [True, False])
def test_sigterm_during_final_wait_keeps_completed_task_result(tmp_path, group):
    with launch(tmp_path, "stop", blocked={signal.SIGTERM}) as fixture:
        fixture.assert_writer_is_held()
        fixture.stop(group=group)
        output, _ = fixture.process.communicate(timeout=5)
        assert fixture.process.returncode == 0, output
        actual = [row for row in map(json.loads, (fixture.run / "checkpoints/events.jsonl").read_text().splitlines())
                  if row["event"] == "checkpoint_writer_finished" and row["step"] == 7][-1]
        assert actual["exit_code"] in {-signal.SIGTERM, -signal.SIGKILL}
        fixture.assert_result(writer_outcome(actual["exit_code"]))


def test_sigint_during_final_wait_keeps_completed_task_result(tmp_path):
    with launch(tmp_path, "stop", blocked={signal.SIGTERM}) as fixture:
        fixture.assert_writer_is_held()
        fixture.signal_engine(
            signal.SIGINT, what="engine waiting for checkpoint writer after SIGINT")
        fixture.process.wait(timeout=5)
        assert fixture.process.returncode == 0
        actual = [row for row in map(json.loads, (fixture.run / "checkpoints/events.jsonl").read_text().splitlines())
                  if row["event"] == "checkpoint_writer_finished" and row["step"] == 7][-1]
        assert actual["exit_code"] in {-signal.SIGTERM, -signal.SIGKILL}
        fixture.assert_result(writer_outcome(actual["exit_code"]))


def test_sigint_before_finalization_keeps_keyboard_interrupt_exit(tmp_path):
    with launch(tmp_path, "before-main") as fixture:
        fixture.observe("before_main")
        fixture.signal_engine(signal.SIGINT, what="engine before task finalization")
        output, _ = fixture.process.communicate(timeout=5)
        assert fixture.process.returncode == 130, output
        assert not (fixture.run / "engine-result.json").exists()


SIGINT_CHILD = r"""
import os, signal, sys, time
from stateframe import _child

seen = []
def interrupted():
    try:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.2)
    except KeyboardInterrupt:
        return True
    return False

def main(_argv, *, teardown):
    # Both interrupts unwind and neither starts the hard-stop clock.
    seen.extend([interrupted(), interrupted(), teardown.deadline is None])
    teardown.finalizing_checkpoint = True
    seen.extend([interrupted(), interrupted(), teardown.deadline is not None])
    return 0

_child.main = main
_child.watch_parent = lambda _teardown: None
_child._run_cli(["engine", "."])
print(seen)
"""


def test_sigint_starts_the_stop_clock_only_while_finalizing():
    src = Path(__file__).resolve().parents[1] / "src"
    done = subprocess.run(
        [sys.executable, "-B", "-c", SIGINT_CHILD], capture_output=True, text=True, timeout=30,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(src), str(src / "stateframe" / "engine")))))
    assert done.stdout.strip() == "[True, True, True, False, False, True]", done.stderr


EARLY_SIGTERM_CHILD = r"""
import os, signal, time
from stateframe import _child

seen = []
def main(_argv, *, teardown):
    time.sleep(0.2)
    seen.append("main")
    return 0

_child.main = main
_child.watch_parent = lambda _teardown: seen.append("watch_parent")
os.kill(os.getpid(), signal.SIGTERM)
assert signal.SIGTERM in signal.sigpending()
status = _child._run_cli(["engine", "."])
print(status, seen)
"""


def test_sigterm_pending_from_a_blocking_launcher_stops_before_the_parent_watch():
    src = Path(__file__).resolve().parents[1] / "src"
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        done = subprocess.run(
            [sys.executable, "-B", "-c", EARLY_SIGTERM_CHILD], capture_output=True, text=True, timeout=30,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(src), str(src / "stateframe" / "engine")))))
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    assert done.stdout.strip() == "130 []", done.stderr


def test_inherited_ignored_sigint_stays_ignored(tmp_path):
    with launch(tmp_path, "before-main", ignored={signal.SIGINT}) as fixture:
        fixture.observe("before_main")
        fixture.signal_engine(signal.SIGINT, what="engine with inherited ignored SIGINT")
        assert fixture.process.poll() is None
        fixture.release()
        fixture.assert_result(writer_outcome(0))


def test_job_control_group_stop_and_continue_preserves_writer_and_restores_next_checkpoint(tmp_path):
    with launch(tmp_path, "job-control") as fixture:
        fixture.assert_writer_is_held()
        writer_pid = fixture.events["task_returning"]["pid"]
        engine_identity = record_identity(
            fixture.process.pid, launched_after=fixture.launched_at,
            command_fragment=str(fixture.run))
        writer_identity = record_identity(
            writer_pid, launched_after=fixture.launched_at,
            command_fragment=str(fixture.run))
        assert os.getpgid(fixture.process.pid) == fixture.process.pid
        assert os.getpgid(writer_pid) == fixture.process.pid

        signal_run_verified(
            fixture.run, fixture.process.pid, engine_identity, signal.SIGTSTP,
            deadline=time.monotonic(), what="checkpoint job-control stop",
            require_checkpoint_saved=False)
        fixture.wait_for_process_state(fixture.process.pid, engine_identity, stopped=True)
        fixture.wait_for_process_state(writer_pid, writer_identity, stopped=True)

        # Resume the writer while the engine remains stopped so its state is settled
        # before the engine resumes and inspects waitpid's stopped-child notification.
        signal_run_verified(
            fixture.run, writer_pid, writer_identity, signal.SIGCONT,
            deadline=time.monotonic(), what="continued checkpoint writer",
            require_checkpoint_saved=False)
        fixture.wait_for_process_state(writer_pid, writer_identity, stopped=False)
        (fixture.run / "writer-already-continued").touch()
        fixture.wait_for_process_state(fixture.process.pid, engine_identity, stopped=True)

        # This leader-bound signal continues the whole private process group, as `fg` does.
        signal_run_verified(
            fixture.run, fixture.process.pid, engine_identity, signal.SIGCONT,
            deadline=time.monotonic(), what="checkpoint job-control continue",
            require_checkpoint_saved=False)
        fixture.wait_for_process_state(fixture.process.pid, engine_identity, stopped=False)
        fixture.release()
        fixture.assert_result(writer_outcome(0))

        restored = fixture.observe("next_checkpoint_restored")
        assert restored["value"] == "saved"
        saved = fixture.run / "checkpoints" / restored["checkpoint"]
        assert (saved / "checkpoint.json").is_file()
        assert (saved / "state.pkl").is_file()


def test_checkpoint_writer_warning_survives_broken_stderr(tmp_path):
    with launch(tmp_path, "broken-stderr") as fixture:
        fixture.assert_writer_is_held()
        fixture.release()
        fixture.assert_result(writer_outcome(1, "checkpoint write failed: ENOSPC"))


def test_checkpoint_writer_can_exit_before_announcing_readiness(tmp_path):
    with launch(tmp_path, "early-exit") as fixture:
        fixture.observe("task_returning")
        fixture.assert_result(writer_outcome(91))
        assert "writer_ready" not in fixture.events


def test_already_reaped_writer_is_reported_with_unknown_status(tmp_path):
    with launch(tmp_path, "already-reaped") as fixture:
        fixture.assert_result(writer_outcome(None))
