"""A real mutation run checkpoints with a live executor, is SIGKILLed, and resumes from its checkpoint.

The launch ledger is opened by path so the checkpoint pickler can carry the
MutationExecutor. This test drives a scripted run end to end: one mutation
completes (its launch rows are on disk and checkpoints keep saving), a second
slow mutation is in flight when the engine is SIGKILLed, and a fresh process
resumes the run. The resume must restore the engine checkpoint (not replay from
the start), keep the completed launch's durable evidence without launching it
again, and reconcile the in-flight launch as interrupted instead of re-running it.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _step(op: str, ident: str | None = None, args: dict | None = None) -> dict:
    fields: dict[str, object] = {"op": op}
    if ident is not None:
        fields["id"] = ident
    if args is not None:
        fields["args"] = json.dumps(args)
    return {"tool": "mutate", "required": True, "arguments": fields}


def _rows(path: Path) -> list[dict]:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            break
    return rows


def _use_calls(tape_rows: list[dict], name: str) -> list[str]:
    return [row["call_id"] for row in tape_rows if row.get("kind") == "intent"
            and row.get("tool") == "mutate" and row.get("args", {}).get("op") == "use"
            and row["args"].get("id") == name]


def _checkpoint_events(run: Path) -> list[dict]:
    return _rows(run / "checkpoints" / "events.jsonl")


def _engine_child(runner_pid: int) -> int:
    children = Path(f"/proc/{runner_pid}/task/{runner_pid}/children").read_text().split()
    matching = [int(pid) for pid in children
                if "_child.py" in Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()]
    assert len(matching) == 1, matching
    return matching[0]


SLOW_READER_MANIFEST = {
    "schema": "stateframe-mutation-unit/1", "name": "slow_reader",
    "revision": 1, "parent": None, "entry": "main.py", "summary": "Read odd rows",
    "kind": "reader", "input_schema": {"type": "object"},
    "output_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                      "required": ["value"], "additionalProperties": False},
    "requires": {"network": False, "writes": False},
    "smoke": [{"input": {"mode": "smoke"}, "expect": {"value": 7}}],
    "limits": {"wall_s": 25, "cpu_s": 5, "mem_mb": 256}}
SLOW_READER_SOURCE = ("import time\n"
                      "def handle(input, ctx):\n"
                      "    if input.get('mode') == 'smoke': return {'value': 7}\n"
                      "    time.sleep(20)\n"
                      "    return {'value': int(ctx.input_text(input['input_id']).split('|')[1])}\n")


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="mutation launches require the Linux sandbox")
def test_mutation_run_checkpoints_and_resumes_after_engine_sigkill(tmp_path):
    from stateframe.scripted_endpoint import Script, serve
    from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
    from mutate_leftovers import assert_no_leftovers, snapshot_leftovers

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_report.py").write_text("def test_report(): assert True\n")
    (repo / "reader.py").write_text("def read_report(): return None\n")
    (repo / "report.odd").write_text("name|41\n")
    (repo / "sample.txt").write_bytes(b"first\nsecond\n")

    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        _step("catalog"),
        _step("use", "file_facts", {"input": {"input_id": "in0"}, "inputs": ["sample.txt"]}),
        _step("enter", args={"goal": "Read the odd report"}),
        _step("submit", args={"manifest": SLOW_READER_MANIFEST, "files": {"main.py": SLOW_READER_SOURCE}}),
        _step("use", "slow_reader", {"input": {"mode": "real", "input_id": "in0"}, "inputs": ["report.odd"]}),
        {"tool": "done", "required": True, "arguments": {"summary": "Both readers ran."}},
    ]})
    server = serve(script)
    run = tmp_path / "run"
    home = tmp_path / "home"
    home.mkdir()
    state_home = home / ".stateframe"
    runtime_parent = state_home / "rt" / "mutations"
    before = snapshot_leftovers(runtime_parent)
    environment = {**os.environ, "PYTHONPATH": str(SRC), "STATEFRAME_MUTATIONS": "authored",
                   "HOME": str(home), "STATEFRAME_HOME": str(state_home)}
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    command = [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
               "--task", "Read sample.txt with file_facts, then report.odd with a slow authored reader.",
               "--target", "reader.py", "--endpoint", endpoint, "--model", "scripted",
               "--run-dir", str(run), "--quiet", "--max-turns", "20", "--test-python", sys.executable]
    launched_at = time.time()
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    ledger: Path | None = None
    pre_kill_rows: list[dict] = []
    pre_kill_events: list[dict] = []
    try:
        # Wait until the first mutation has launched (ledger rows exist), the slow one is running,
        # and a checkpoint has been saved after the executor came into existence.
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"run exited before the slow payload started: {process.stdout.read()}")
            spec_path = run / "spec.json"
            if ledger is None and spec_path.is_file():
                seed = json.loads(spec_path.read_text()).get("run_identity_seed")
                if seed:
                    ledger = runtime_parent / str(seed) / "launches.jsonl"
            if ledger is not None:
                rows = _rows(ledger)
                slow_calls = _use_calls(_rows(run / "tool-tape.jsonl"), "slow_reader")
                slow_started = any(row.get("call") in slow_calls and row.get("role") == "payload"
                                   and row.get("state") == "STARTED" for row in rows)
                failed = [event for event in _checkpoint_events(run) if event.get("event") == "checkpoint_failed"]
                assert not failed, failed
                saved = [event for event in _checkpoint_events(run) if event.get("event") == "checkpoint_saved"]
                if slow_started and saved:
                    pre_kill_rows = rows
                    pre_kill_events = saved
                    break
            time.sleep(0.1)
        assert pre_kill_rows, ("slow reader payload or checkpoint not seen",
                               [(event.get("event"), event.get("step"), event.get("reason"),
                                 event.get("failure_reason")) for event in _checkpoint_events(run)],
                               [(row.get("call"), row.get("role"), row.get("state"))
                                for row in (_rows(ledger) if ledger else [])][-12:],
                               [(row.get("kind"), row.get("args", {}).get("op"), row.get("args", {}).get("id"))
                                for row in _rows(run / "tool-tape.jsonl")][-12:])
        assert ledger is not None

        # Crash the verified engine process (it leads its own session, so the mutation supervisor
        # goes with it) after a checkpoint is on disk; the runner then exits on its own.
        engine_pid = _engine_child(process.pid)
        identity = record_identity(engine_pid, launched_after=launched_at, command_fragment="_child.py")
        signal_run_verified(run, engine_pid, identity, signal.SIGKILL, deadline=deadline,
                            what="mutation run")
        process.communicate(timeout=90)
        assert process.returncode != 0, process.returncode

        # --keep-checkpoints keeps the restored attempt's checkpoint events after it finishes.
        resumed = subprocess.run([sys.executable, "-m", "stateframe", "resume", str(run),
                                  "--endpoint", endpoint, "--keep-checkpoints"], env=environment,
                                 capture_output=True, text=True, timeout=420)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        script.assert_consumed()

        # Checkpoints kept saving while the executor was alive (asserted before the kill; the
        # resume archives the earlier attempt's checkpoint events), and the resumed attempt did not
        # fail to serialize either.
        assert any(event.get("event") == "checkpoint_saved" for event in pre_kill_events), pre_kill_events
        events = _checkpoint_events(run)
        assert not [event for event in events if event.get("event") == "checkpoint_failed"], events

        # The resume restores the newest checkpoint instead of replaying from request 1: the
        # spec records the restored checkpoint (send count and replayed tail), the engine logs
        # checkpoint_restored, and no restore was refused (a refusal falls back to full replay).
        spec = json.loads((run / "spec.json").read_text())
        checkpoint = spec.get("resume", {}).get("checkpoint") or {}
        assert checkpoint.get("status") != "skipped", (checkpoint, resumed.stdout)
        assert isinstance(checkpoint.get("sends"), int) and checkpoint.get("dir"), (checkpoint, resumed.stdout)
        assert "checkpoint: engine state restored as of send" in resumed.stdout, resumed.stdout
        restored = [event for event in events if event.get("event") == "checkpoint_restored"]
        refused = [event for event in events if event.get("event") == "checkpoint_restore_refused"]
        assert len(restored) == 1 and not refused, events
        assert restored[0].get("sends") == checkpoint["sends"], (restored, checkpoint)
        # Only the exchanges after the checkpoint were replayed, and the earlier attempt is the only
        # archived one (a refused restore would have archived another attempt and replayed it all).
        summary = json.loads((run / "summary.json").read_text())
        assert summary["resume"]["attempt"] == 2, summary["resume"]
        assert summary["resume"]["replayed"] == checkpoint["tail"] >= 1, (summary["resume"], checkpoint)
        assert sorted(path.name for path in (run / "attempts").iterdir()) == ["1"]

        # No mutation launched twice: one use intent per unit, one admission and one result per call.
        tape = _rows(run / "tool-tape.jsonl")
        first_call = _use_calls(tape, "file_facts")
        slow_call = _use_calls(tape, "slow_reader")
        assert len(first_call) == 1 and len(slow_call) == 1, (first_call, slow_call)
        assert {row.get("call") for row in pre_kill_rows} >= {first_call[0], slow_call[0]}, pre_kill_rows
        # Each pre-kill launch ran once: the restored engine never starts another payload for a call
        # the pre-kill ledger already holds (the ledger is ephemeral and swept, so the durable
        # journal is the post-resume witness: one LAUNCHED/STARTED row set per call at most).
        pre_kill_launches = {(row.get("call"), row.get("role")) for row in pre_kill_rows
                             if row.get("state") == "STARTED"}
        assert (first_call[0], "payload") in pre_kill_launches, pre_kill_rows
        assert (slow_call[0], "payload") in pre_kill_launches, pre_kill_rows
        journal = _rows(run / "mutations" / "journal.jsonl")
        for call, outcome in ((first_call[0], "TERMINAL"), (slow_call[0], "INTERRUPTED")):
            admissions = [row for row in journal if row.get("kind") == "ADMISSION" and row.get("call") == call]
            assert len(admissions) == 1, (call, admissions)
            results = [row for row in journal if row.get("call") == call
                       and row.get("kind") in {"INTERRUPTED", "TERMINAL"}]
            assert [row["kind"] for row in results] == [outcome], (call, results)
        interrupted = [row for row in journal if row.get("call") == slow_call[0] and row.get("kind") == "INTERRUPTED"]
        assert interrupted[0]["observation"].split()[:2] == ["REFUSED", "MUTATION_INTERRUPTED"], interrupted

        # The interrupted runtime root (and its ledger, ephemeral ownership evidence) is swept.
        assert not runtime_parent.exists() or not list(runtime_parent.iterdir())
        assert_no_leftovers(before, snapshot_leftovers(runtime_parent))
    finally:
        if process.poll() is None:
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="test cleanup", require_checkpoint_saved=False)
            process.wait(timeout=10)
        if process.stdout is not None:
            process.stdout.close()
        server.shutdown()
        server.server_close()


# --- Shared crash/resume driver for the edge tests below -----------------------------------------

NOTES_PRE = b"alpha\nbeta\n"
NOTES_POST = NOTES_PRE + b"gamma rolled forward\n"

WRITER_MANIFEST = {
    "schema": "stateframe-mutation-unit/1", "name": "notes_writer",
    "revision": 1, "parent": None, "entry": "main.py", "summary": "Append a line to notes.txt",
    "kind": "reader", "input_schema": {"type": "object"},
    "output_schema": {"type": "object", "properties": {"lines": {"type": "integer"}},
                      "required": ["lines"], "additionalProperties": False},
    "requires": {"network": False, "writes": True},
    "smoke": [{"input": {"mode": "smoke"}, "expect": {"lines": 0}}],
    "limits": {"wall_s": 25, "cpu_s": 5, "mem_mb": 256}}
WRITER_SOURCE = ("import hashlib\n"
                 "def handle(input, ctx):\n"
                 "    if input.get('mode') == 'smoke': return {'lines': 0}\n"
                 "    text = ctx.input_text(input['input_id'])\n"
                 "    ctx.propose_edit('replace', 'notes.txt', text + 'gamma rolled forward\\n',\n"
                 "                     base_sha256=hashlib.sha256(text.encode()).hexdigest())\n"
                 "    return {'lines': text.count('\\n') + 1}\n")


def _crash_after(tmp_path: Path, steps: list[dict], ready, *, extra_args: tuple[str, ...] = (),
                 extra_env: dict | None = None) -> tuple[Path, subprocess.CompletedProcess, dict]:
    """Run a scripted mutation run, SIGKILL its engine once ``ready(run, ledger_rows)`` returns a
    truthy snapshot, then resume it. Returns (run dir, resume result, pre-kill snapshot)."""
    from stateframe.scripted_endpoint import Script, serve
    from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_report.py").write_text("def test_report(): assert True\n")
    (repo / "reader.py").write_text("def read_report(): return None\n")
    (repo / "report.odd").write_text("name|41\n")
    (repo / "sample.txt").write_bytes(b"first\nsecond\n")
    (repo / "notes.txt").write_bytes(NOTES_PRE)
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": steps})
    server = serve(script)
    run = tmp_path / "run"
    home = tmp_path / "home"
    home.mkdir()
    state_home = home / ".stateframe"
    runtime_parent = state_home / "rt" / "mutations"
    environment = {**os.environ, "PYTHONPATH": str(SRC), "STATEFRAME_MUTATIONS": "authored",
                   "HOME": str(home), "STATEFRAME_HOME": str(state_home), **(extra_env or {})}
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    command = [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
               "--task", "Run the scripted mutation units.", "--target", "reader.py",
               "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run), "--quiet",
               "--max-turns", "20", "--test-python", sys.executable, *extra_args]
    launched_at = time.time()
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    snapshot: dict | None = None
    ledger: Path | None = None
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"run exited before the crash point: {process.stdout.read()}")
            spec_path = run / "spec.json"
            if ledger is None and spec_path.is_file():
                seed = json.loads(spec_path.read_text()).get("run_identity_seed")
                if seed:
                    ledger = runtime_parent / str(seed) / "launches.jsonl"
            if ledger is not None:
                failed = [e for e in _checkpoint_events(run) if e.get("event") == "checkpoint_failed"]
                assert not failed, failed
                snapshot = ready(run, _rows(ledger))
                if snapshot:
                    break
            time.sleep(0.1)
        assert snapshot, ("crash point not reached",
                          [(e.get("event"), e.get("step")) for e in _checkpoint_events(run)],
                          [(r.get("kind"), r.get("call")) for r in _rows(run / "mutations" / "journal.jsonl")])
        engine_pid = _engine_child(process.pid)
        identity = record_identity(engine_pid, launched_after=launched_at, command_fragment="_child.py")
        signal_run_verified(run, engine_pid, identity, signal.SIGKILL, deadline=deadline,
                            what="mutation run")
        process.communicate(timeout=90)
        assert process.returncode != 0, process.returncode
        resumed = subprocess.run([sys.executable, "-m", "stateframe", "resume", str(run),
                                  "--endpoint", endpoint, "--keep-checkpoints"], env=environment,
                                 capture_output=True, text=True, timeout=420)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        script.assert_consumed()
        events = _checkpoint_events(run)
        restored = [e for e in events if e.get("event") == "checkpoint_restored"]
        refused = [e for e in events if e.get("event") == "checkpoint_restore_refused"]
        assert len(restored) == 1 and not refused, (events, resumed.stdout)
        return run, resumed, snapshot
    finally:
        if process.poll() is None:
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="test cleanup", require_checkpoint_saved=False)
            process.wait(timeout=10)
        if process.stdout is not None:
            process.stdout.close()
        server.shutdown()
        server.server_close()


def _slow_payload_started(run: Path, ledger_rows: list[dict]) -> bool:
    slow_calls = _use_calls(_rows(run / "tool-tape.jsonl"), "slow_reader")
    return any(row.get("call") in slow_calls and row.get("role") == "payload"
               and row.get("state") == "STARTED" for row in ledger_rows)


def _newest_checkpoint_files(run: Path, name: str) -> list[bytes]:
    """Every run/workspace copy of ``name`` in the newest saved checkpoint (engine-internal
    continuation repos under worker/ carry their own copy and are not the run workspace)."""
    saved = [e for e in _checkpoint_events(run) if e.get("event") == "checkpoint_saved"]
    if not saved:
        return []
    directory = run / "checkpoints" / saved[-1]["dir"]
    return [path.read_bytes() for path in directory.rglob(name)
            if path.is_file() and path.parent.name == "workspace"]


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="mutation launches require the Linux sandbox")
def test_resume_rolls_forward_edit_committed_after_checkpoint_exactly_once(tmp_path):
    """A writer commits notes.txt after the newest checkpoint; the engine dies before the next one.
    The restored workspace holds the checkpoint's preimage, so the resume must roll the committed
    edit forward: notes.txt ends byte-exact at the postimage (not the preimage, not applied twice)
    and the writer never launches again."""
    steps = [
        _step("enter", args={"goal": "Read the odd report"}),
        _step("submit", args={"manifest": SLOW_READER_MANIFEST, "files": {"main.py": SLOW_READER_SOURCE}}),
        _step("enter", args={"goal": "Append a line to notes.txt", "needs": {"writes": True}}),
        _step("submit", args={"manifest": WRITER_MANIFEST, "files": {"main.py": WRITER_SOURCE}}),
        _step("use", "notes_writer", {"input": {"input_id": "in0"}, "inputs": ["notes.txt"]}),
        _step("use", "slow_reader", {"input": {"mode": "real", "input_id": "in0"}, "inputs": ["report.odd"]}),
        {"tool": "done", "required": True, "arguments": {"summary": "Notes written."}},
    ]

    def ready(run: Path, ledger_rows: list[dict]) -> dict | None:
        journal = _rows(run / "mutations" / "journal.jsonl")
        committed = [row for row in journal if row.get("kind") == "EFFECT_COMMITTED"]
        if not committed or not _slow_payload_started(run, ledger_rows):
            return None
        return {"committed": committed,
                "workspace": (run / "workspace" / "notes.txt").read_bytes(),
                "checkpoint_copies": _newest_checkpoint_files(run, "notes.txt")}

    # Save every other step (2, 4, 6) so the writer's step 5 commits with no checkpoint after it,
    # and the engine dies in step 6 while the slow reader runs, before that step's save.
    run, resumed, before = _crash_after(tmp_path, steps, ready, extra_args=("--mutation-write", "notes.txt", "--in-place"),
                                        extra_env={"STATEFRAME_CHECKPOINT_EVERY": "2"})

    # Preconditions: the commit landed live, and the checkpoint the resume restores predates it.
    assert len(before["committed"]) == 1, before["committed"]
    assert before["workspace"] == NOTES_POST, before["workspace"]
    assert before["checkpoint_copies"] and all(copy == NOTES_PRE for copy in before["checkpoint_copies"]), \
        before["checkpoint_copies"]

    # Right after the restore (before finish, which re-materializes committed effects from the
    # journal), the resumed attempt's first checkpoint saw the rolled-forward postimage, exactly once.
    after_restore = _newest_checkpoint_files(run, "notes.txt")
    assert after_restore and all(copy == NOTES_POST for copy in after_restore), after_restore
    saved = [e for e in _checkpoint_events(run) if e.get("event") == "checkpoint_saved"]
    restored = [e for e in _checkpoint_events(run) if e.get("event") == "checkpoint_restored"]
    assert saved and restored and saved[-1]["ts"] > restored[0]["ts"], (saved, restored)

    # The resumed workspace (and, with --in-place, the repo) holds the postimage exactly once.
    # Mutation effects land in the workspace directly; they are not part of the sealed candidate.
    assert (run / "workspace" / "notes.txt").read_bytes() == NOTES_POST
    assert (tmp_path / "repo" / "notes.txt").read_bytes() == NOTES_POST
    journal = _rows(run / "mutations" / "journal.jsonl")
    tape = _rows(run / "tool-tape.jsonl")
    writer_calls = _use_calls(tape, "notes_writer")
    assert len(writer_calls) == 1, writer_calls
    assert [row["kind"] for row in journal if row.get("call") == writer_calls[0]
            and row.get("kind") in {"EFFECT_COMMITTED", "EFFECT_REFUSED"}] == ["EFFECT_COMMITTED"], journal
    assert len([row for row in journal if row.get("kind") == "ADMISSION"
                and row.get("call") == writer_calls[0]]) == 1, journal


WATCH_STEPS_TAIL = [
    _step("enter", args={"goal": "Read the odd report"}),
    _step("submit", args={"manifest": SLOW_READER_MANIFEST, "files": {"main.py": SLOW_READER_SOURCE}}),
    _step("use", "slow_reader", {"input": {"mode": "real", "input_id": "in0"}, "inputs": ["report.odd"]}),
    {"tool": "done", "required": True, "arguments": {"summary": "Watched and read."}},
]


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="mutation launches require the Linux sandbox")
def test_resume_stops_restored_watch_with_reason_resume_and_never_fires_twice(tmp_path):
    """A watch is live when the engine dies. The resuming host journals watch_stop reason=resume
    for it exactly once, and no watch tick (intent or completion) is journaled twice."""
    # every_turns=3: the watch ticks once before the crash (each tick launches the unit, and the
    # run's payload budget must still admit the slow reader) and is still live when the engine dies.
    steps = [
        _step("watch", "file_facts", {"input": {"input_id": "in0"}, "inputs": ["sample.txt"],
                                      "every_turns": 3, "max_ticks": 30}),
        *WATCH_STEPS_TAIL,
    ]

    def ready(run: Path, ledger_rows: list[dict]) -> dict | None:
        journal = _rows(run / "mutations" / "journal.jsonl")
        ticks = [row for row in journal if row.get("kind") == "WATCH_TICK_COMPLETE"]
        saved = [e for e in _checkpoint_events(run) if e.get("event") == "checkpoint_saved"]
        if not ticks or not saved or not _slow_payload_started(run, ledger_rows):
            return None
        return {"journal": journal}

    run, _resumed, before = _crash_after(tmp_path, steps, ready)

    starts = [row for row in before["journal"] if row.get("kind") == "watch_start"]
    assert len(starts) == 1, before["journal"]
    watch_id = starts[0]["watch"]
    assert not [row for row in before["journal"] if row.get("kind") == "watch_stop"], before["journal"]

    journal = _rows(run / "mutations" / "journal.jsonl")
    stops = [row for row in journal if row.get("kind") == "watch_stop"]
    resume_stops = [row for row in stops if row.get("reason") == "resume"]
    assert [row.get("watch") for row in resume_stops] == [watch_id], stops
    assert len([row for row in journal if row.get("kind") == "watch_start"]) == 1, journal
    for kind in ("WATCH_TICK_INTENT", "WATCH_TICK_COMPLETE"):
        invocations = [row["invocation_id"] for row in journal if row.get("kind") == kind]
        assert invocations and len(invocations) == len(set(invocations)), (kind, invocations)
    parents = [(row["watch"], row["parent_turn_id"]) for row in journal if row.get("kind") == "WATCH_TICK_INTENT"]
    assert len(parents) == len(set(parents)), parents
    # The stopped watch never ticks again after the resume stopped it.
    stop_index = journal.index(resume_stops[0])
    late = [row for row in journal[stop_index:]
            if row.get("kind") == "WATCH_TICK_INTENT" and row.get("watch") == watch_id]
    assert not late, late
