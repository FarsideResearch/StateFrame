"""The disabled startup pilot keeps its modules out of both launch processes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import sys
from textwrap import dedent
import time
from threading import Barrier, Event

import pytest
from _timeouts import scaled
from engine_run import run_monitored, start_monitored
from _journal import measurement_events, measurement_journal_paths
from fixtures.process_signals import signal_run_owned, wait_for_checkpoint_saved
from kernel_cache_layout_view import FINAL_LAYOUT_NAME
from tools.pinned_tree import pinned_tree

from stateframe import profiles
from stateframe.scripted_endpoint import Script, serve


ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "b34f7b667159e6562a774579c65986617b084b41"
MODULE_CAPTURE = dedent("""\
    import sys

    def capture(frame, event, arg):
        if event != "call":
            return
        if frame.f_code.co_name == "main" and frame.f_code.co_filename.endswith("/_child.py"):
            sys.modules["threading"].setprofile(capture)
        elif (frame.f_code.co_name == "build_qualified_request"
                and frame.f_globals.get("__name__") == "kernel_final_wire_allocation"):
            sys.setprofile(None)
            sys.modules["threading"].setprofile(None)
            with open({child_path!r}, "x", encoding="utf-8") as output:
                output.write("\\n".join(sorted(sys.modules)))

    sys.setprofile(capture)
""")
CLI_CAPTURE = dedent("""\
    import subprocess
    import sys

    launch = subprocess.Popen

    def capture_child(command, *args, **kwargs):
        if isinstance(command, list) and command[-1] == "stateframe._child":
            command = [*command[:3], {child_code!r} + command[3], command[4]]
        elif isinstance(command, list) and len(command) == 4 and command[2].endswith("/_child.py"):
            bootstrap = (
                "import os, runpy\\n"
                "sys.argv.pop(0)\\n"
                "sys.path[0] = os.path.dirname(sys.argv[0])\\n"
                "runpy.run_path(sys.argv[0], run_name='__main__')"
            )
            command = [*command[:2], "-c", {child_code!r} + bootstrap, *command[2:]]
        else:
            return launch(command, *args, **kwargs)
        with open({parent_path!r}, "w", encoding="utf-8") as output:
            output.write("\\n".join(sorted(sys.modules)))
        return launch(command, *args, **kwargs)

    subprocess.Popen = capture_child
    from stateframe.cli import main
    raise SystemExit(main(sys.argv[1:]))
""")
QUALIFICATION_LAUNCH = dedent("""\
    import secrets
    import sys
    from types import SimpleNamespace

    from stateframe import runner
    from stateframe.cli import main

    runner.secrets = SimpleNamespace(
        token_hex=lambda size: "0123456789abcdef" * 2 if size == 16 else secrets.token_hex(size),
    )
    raise SystemExit(main(sys.argv[1:]))
""")
IMPORT_GUARD = """
import sys

class StartupModuleGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {
            "stateframe.send1_probe_cache",
            "stateframe.startup_cache_writes",
            "stateframe.workspace_start_snapshot",
            "kernel_agent_registration",
        }:
            raise AssertionError(f"disabled pilot imported {fullname}")

sys.meta_path.insert(0, StartupModuleGuard())
"""


@pytest.fixture(scope="session")
def release_tree(tmp_path_factory):
    return pinned_tree(BASE_REVISION, tmp_path_factory)


@pytest.fixture
def startup_script():
    return Script({"tokenize": {"mode": "ok"}, "steps": [
        {"tool": "read", "arguments": {"path": "sample.py"}, "repeat": True},
    ]})


@pytest.fixture
def scripted_startup(tmp_path, startup_script):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def sample():\n    return 42\n", encoding="utf-8")
    with serve(startup_script) as server:
        def start(source, name, mode, *, launcher=("-m", "stateframe"), turns=1, action="run",
                  home=None, extra_env=None, task="Inspect sample.py."):
            run = tmp_path / name
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("STATEFRAME_", "REACT_", "SF_LAB_"))}
            env.update(PYTHONPATH=str(source / "src"), PATH=os.defpath,
                       STATEFRAME_CHECKPOINT="off",
                       STATEFRAME_HOME=str(home or tmp_path / f"home-{name}"))
            env.update(extra_env or {})
            if mode is not None:
                env["STATEFRAME_SEND1_WAVE4"] = mode
            endpoint = f"http://127.0.0.1:{server.server_port}/v1"
            if action in {"run", "launch"}:
                command = [
                    sys.executable, "-B", *launcher, "run", "--repo", str(repo),
                    "--run-dir", str(run), "--task", task,
                    "--endpoint", endpoint, "--model", "scripted",
                    "--test-python", str(Path(sys.executable).resolve()),
                    "--check-cmd", "true", "--max-turns", str(turns),
                    "--keep-checkpoints", "--quiet",
                ]
            elif action == "resume":
                command = [
                    sys.executable, "-B", *launcher, "resume", str(run),
                    "--endpoint", endpoint, "--max-turns", str(turns),
                    "--keep-checkpoints", "--quiet",
                ]
            else:
                raise ValueError(f"unsupported scripted startup action: {action}")
            if action == "launch":
                return start_monitored(command, env=env, cwd=tmp_path, run_dir=run, step=name)
            process = run_monitored(command, env=env, cwd=tmp_path, run_dir=run, step=name)
            if action == "run":
                assert (run / "requests" / f"{turns:04d}.json").is_file(), process.stdout + process.stderr
            elif process.returncode != 0:
                raise AssertionError(process.stdout + process.stderr)
            return run

        try:
            yield start
        finally:
            server.shutdown()


@pytest.mark.host_sandbox
@pytest.mark.parametrize("mode", [None, "off"], ids=["unset", "off"])
def test_disabled_cli_startup_module_set_matches_release(release_tree, scripted_startup, tmp_path, mode):
    def modules(source, name):
        parent = tmp_path / f"{name}-parent.txt"
        child = tmp_path / f"{name}-child.txt"
        launcher = CLI_CAPTURE.format(parent_path=str(parent),
                                      child_code=MODULE_CAPTURE.format(child_path=str(child)))
        scripted_startup(source, name, mode, launcher=("-c", launcher))
        result = {"CLI": parent.read_text().splitlines(), "engine": child.read_text().splitlines()}
        parent.unlink()
        child.unlink()
        return result

    release = modules(release_tree, "release")
    current = modules(ROOT, "current")
    differences = {
        process: {"added": sorted(set(current[process]) - set(expected)),
                  "missing": sorted(set(expected) - set(current[process]))}
        for process, expected in release.items() if current[process] != expected
    }
    assert current == release, differences


def qualification_records(run):
    assert len(measurement_journal_paths(run)) == 1
    selected_kinds = {
        "qualified_request_sealed", "framefit5_provider_boundary_checked", "working_set_prepared",
        "request_source_final_wire", "read_reuse_visibility_checked", "wire_truth_checked",
        "action_receipt", "action_receipt_amended", "native_batch_call_receipt", "read_repeat_assessed",
    }
    # These fields describe journal writes, so their wall-clock differences
    # also propagate through the hash chain. The payload and record order stay intact.
    envelope = {"event_id", "previous_hash", "record_hash", "timestamp_ns"}
    records = []
    for row in measurement_events(run):
        if row["kind"] not in selected_kinds:
            continue
        record = {key: value for key, value in row.items() if key not in envelope}
        if "receipt" in record:
            record["receipt"].pop("ts", None)
        records.append(record)
    assert sum(row["kind"] == "qualified_request_sealed" for row in records) == 2
    assert any(row["kind"] == "action_receipt_amended" for row in records)
    return json.dumps(records, sort_keys=True, indent=1).encode()


@pytest.mark.host_sandbox
def test_disabled_qualification_and_receipts_match_release(release_tree, scripted_startup, tmp_path):
    launcher = ("-c", QUALIFICATION_LAUNCH)
    run = scripted_startup(release_tree, "qualification", "off", launcher=launcher, turns=2,
                           task="Inspect the sample.")
    release = qualification_records(run)
    release_run = tmp_path / "release"
    run.rename(release_run)
    current_run = scripted_startup(ROOT, "qualification", "off", launcher=launcher, turns=2,
                                   task="Inspect the sample.")
    assert qualification_records(current_run) == release
    release_wire = sorted((release_run / "wire").glob("*.request.json"))
    current_wire = sorted((current_run / "wire").glob("*.request.json"))
    assert [path.name for path in release_wire] == [path.name for path in current_wire]
    assert len(release_wire) == 2
    for expected, actual in zip(release_wire, current_wire, strict=True):
        assert actual.read_bytes() == expected.read_bytes(), actual.name


@pytest.mark.host_sandbox
@pytest.mark.parametrize("layout", ["off", "on"])
def test_shipped_layout_and_startup_switch_keep_host_marker_out_of_wire(
        scripted_startup, startup_script, tmp_path, layout):
    bodies = []
    for mode in ("off", "on"):
        marker = f"PRIVATE-HOST-STARTUP-{mode.upper()}-9DEC3A"
        home = tmp_path / marker
        run = scripted_startup(
            ROOT, "matrix", mode, launcher=("-c", QUALIFICATION_LAUNCH), turns=2, home=home,
            extra_env={"STATEFRAME_FINAL_CACHE_LAYOUT": layout, "STATEFRAME_PAGE_LOG": "off"},
        )
        spec = json.loads((run / "spec.json").read_bytes())
        assert spec["profile"] == profiles.DEFAULT
        assert spec["environment"]["STATEFRAME_HOME"] == str(home)
        assert spec["environment"]["STATEFRAME_FINAL_CACHE_LAYOUT"] == layout
        assert "STATEFRAME_SEND1_WAVE4" not in spec["environment"]
        assert (spec.get("send1_wave4") is True) == (mode == "on")
        assert (run / "parent-start.json").is_file() == (mode == "on")

        sends = [json.loads(row) for row in (run / "sends.jsonl").read_bytes().splitlines()]
        assert [row["send"] for row in sends] == [1, 2]
        wire = [(run / "wire" / f"{send:04d}.request.json").read_bytes() for send in (1, 2)]
        assert all(b"PRIVATE-HOST-STARTUP-" not in body for body in wire)
        requests = [json.loads(body) for body in wire]
        assert len(startup_script.requests) == 2 * (len(bodies) + 1)
        assert requests == startup_script.requests[-2:]
        assert [body["messages"][-1].get("name") for body in requests] == [
            FINAL_LAYOUT_NAME if layout == "on" else None,
        ] * 2
        bodies.append(wire)
        run.rename(tmp_path / mode)
    assert bodies[0] == bodies[1]


@pytest.mark.host_sandbox
def test_two_engines_can_reach_first_send_with_one_startup_store(
        scripted_startup, startup_script, tmp_path, monkeypatch):
    arrive = Barrier(2, timeout=scaled(45))
    next_call = startup_script.next_call

    def accept_together(offered):
        arrive.wait()
        return next_call(offered)

    monkeypatch.setattr(startup_script, "next_call", accept_together)
    home = tmp_path / "shared-home"
    tasks = ["Inspect sample.py for ALPHA.", "Inspect sample.py for BRAVO."]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(scripted_startup, ROOT, f"parallel-{index}", "on", home=home, task=task)
                   for index, task in enumerate(tasks)]
        runs = {future.result(): task for future, task in zip(futures, tasks, strict=True)}

    assert len(startup_script.requests) == 2
    engine_pids = {(run / ".engine.lock").read_text().splitlines()[0] for run in runs}
    assert len(engine_pids) == 2
    for run, task in runs.items():
        spec = json.loads((run / "spec.json").read_bytes())
        assert spec["environment"]["STATEFRAME_HOME"] == str(home)
        assert (run / "parent-start.json").is_file()
        body = (run / "wire/0001.request.json").read_bytes()
        assert task.encode() in body
        assert all(other.encode() not in body for other in tasks if other != task)
        sends = [json.loads(row) for row in (run / "sends.jsonl").read_bytes().splitlines()]
        assert [row["send"] for row in sends] == [1]


def test_disabled_tokenizer_skips_parallel_schema_work_and_unpinned_capability_lookup(
        tmp_path, monkeypatch):
    from kernel_agent_schemas import prepare_schema_measurement
    from kernel_request_diet import RequestTokenMeter

    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "off")
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path))

    def unexpected(*args, **kwargs):
        pytest.fail("disabled pilot ran a speculative tokenizer check")

    monkeypatch.setattr(RequestTokenMeter, "overlap_http_counts", unexpected)
    monkeypatch.setattr(RequestTokenMeter, "_tokenizer_capability_proven", unexpected)
    count = prepare_schema_measurement(tokenizer=lambda _text: 2)
    assert count()["shell"] == {"tokens": 2, "method": "tokenize"}
    meter = RequestTokenMeter(endpoint="http://127.0.0.1:1/v1")
    assert meter._persistent_count_slot("identity", "content") is None


def test_disabled_path_listing_does_not_build_an_inventory_key(tmp_path, monkeypatch):
    from stateframe import shell_host

    executable = tmp_path / "example-tool"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "off")
    monkeypatch.setenv("PATH", str(tmp_path))

    def unexpected(_inputs):
        pytest.fail("path listing constructed a fingerprint with the pilot disabled")

    monkeypatch.setattr(shell_host, "path_authority_fingerprint", unexpected)
    directories, roots = shell_host.path_authority()
    assert directories == [str(tmp_path)]
    assert str(executable) in roots


def test_disabled_bootstrap_uses_release_environment_installer(tmp_path, monkeypatch):
    from stateframe import _child

    class Installed(Exception):
        pass

    spec = {"environment": {}}
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")

    def install(recorded_spec):
        assert recorded_spec == spec
        raise Installed

    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "off")
    monkeypatch.setattr(_child, "watch_parent", lambda: None)
    monkeypatch.setattr(_child, "hold_engine_lock", lambda _run: None)
    monkeypatch.setattr(_child, "_runner_home_identity_pin", lambda: None)
    monkeypatch.setattr(_child, "install_environment", install)
    with pytest.raises(Installed):
        _child.main(["_child.py", str(tmp_path)])


@pytest.mark.host_sandbox
def test_disabled_launch_uses_no_pilot_modules_and_matches_enabled_wire(tmp_path):
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import os, subprocess, sys\n"
        f"guard = {IMPORT_GUARD!r}\n"
        "if os.environ['STATEFRAME_SEND1_WAVE4'] == 'off':\n"
        "    exec(guard)\n"
        "    launch = subprocess.Popen\n"
        "    def guarded_launch(command, *args, **kwargs):\n"
        "        if isinstance(command, list) and command[-1] == 'stateframe._child':\n"
        "            command = [*command[:3], guard + '\\n' + command[3], command[4]]\n"
        "        elif isinstance(command, list) and len(command) == 4 and command[2].endswith('/_child.py'):\n"
        "            bootstrap = guard + '\\nimport os, runpy\\n'\n"
        "            bootstrap += 'sys.argv.pop(0); sys.path[0] = os.path.dirname(sys.argv[0]); runpy.run_path(sys.argv[0], run_name=\"__main__\")'\n"
        "            command = [*command[:2], '-c', bootstrap, *command[2:]]\n"
        "        return launch(command, *args, **kwargs)\n"
        "    subprocess.Popen = guarded_launch\n"
        "from stateframe.cli import main\n"
        "raise SystemExit(main(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def sample():\n    return 42\n", encoding="utf-8")
    script = Script({"tokenize": {"mode": "ok"}, "steps": [
        {"tool": "read", "arguments": {"path": "sample.py"}, "repeat": True},
    ]})
    results = []
    python = str(Path(sys.executable).resolve())
    with serve(script) as server:
        try:
            for mode in ("off", "on"):
                run = tmp_path / mode
                home = tmp_path / f"home-{mode}"
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(("STATEFRAME_", "REACT_"))}
                env.update(PYTHONPATH=str(ROOT / "src"), PATH=os.defpath,
                           STATEFRAME_SEND1_WAVE4=mode, STATEFRAME_CHECKPOINT="off",
                           STATEFRAME_HOME=str(home))
                command = [sys.executable, str(launcher), "run", "--repo", str(repo),
                           "--run-dir", str(run), "--task", "Inspect sample.py.",
                           "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                           "--model", "scripted", "--test-python", python,
                           "--check-cmd", "true", "--max-turns", "1", "--quiet"]
                process = run_monitored(command, env=env, run_dir=run,
                                        step=f"scripted startup with pilot {mode}")
                marker = run / "requests" / "0001.json"
                assert marker.is_file(), process.stdout + process.stderr
                spec = json.loads((run / "spec.json").read_text())
                assert "STATEFRAME_SEND1_WAVE4" not in spec["environment"]
                if mode == "off":
                    assert "send1_wave4" not in spec
                    assert not (home / "startup-probes-v1").exists()
                    assert not (run / "parent-start.json").exists()
                else:
                    assert spec["send1_wave4"] is True
                results.append((marker.read_bytes(), (run / "wire/0001.request.json").read_bytes()))
        finally:
            server.shutdown()
    assert results[0] == results[1]


@pytest.mark.host_sandbox
def test_wave4_checkpoint_saves_and_resumes_from_serialized_state(
        scripted_startup, startup_script, monkeypatch, tmp_path):
    # Hold a live request after a checkpoint is published, then resume that same
    # interrupted attempt from the checkpoint.
    next_call = startup_script.next_call
    run = tmp_path / "wave4-checkpoint"
    checkpoint_events = run / "checkpoints" / "events.jsonl"
    in_flight = Event()
    release_response = Event()

    def hold_third_response(offered):
        result = next_call(offered)
        if len(startup_script.requests) == 3:
            in_flight.set()
            if not release_response.wait(120):
                raise TimeoutError("the test did not release the third scripted response")
        return result

    monkeypatch.setattr(startup_script, "next_call", hold_third_response)
    process = scripted_startup(
        ROOT, "wave4-checkpoint", "on", action="launch", turns=0,
        extra_env={"STATEFRAME_CHECKPOINT": "on"},
    )
    try:
        assert in_flight.wait(120), "the scripted third request did not become in flight"
        assert (run / "requests" / "0003.json").is_file()
        wait_for_checkpoint_saved(
            run, deadline=time.monotonic() + 120, what="wave-4 checkpoint regression",
            process=process,
        )
        signal_run_owned(
            run, process, signal.SIGKILL, deadline=time.monotonic(),
            what="wave-4 checkpoint regression", require_checkpoint_saved=False, group=True,
        )
    finally:
        if process.poll() is None:
            try:
                signal_run_owned(
                    run, process, signal.SIGKILL, deadline=time.monotonic(),
                    what="wave-4 checkpoint test cleanup", require_checkpoint_saved=False, group=True,
                )
            except ProcessLookupError:
                pass  # the runner exited between poll() and identity verification
        process.wait(timeout=30)
        release_response.set()
        process.close()

    rows = [json.loads(line) for line in checkpoint_events.read_text().splitlines()]
    assert any(row.get("event") == "checkpoint_saved" for row in rows), rows
    assert not any(row.get("event") == "checkpoint_disabled" for row in rows), rows

    resume_next_call = startup_script.next_call

    def finish_resumed_run(offered):
        if len(startup_script.requests) >= 4:
            return {"tool": "done", "arguments": {"summary": "Checkpoint resume verified."}}
        return resume_next_call(offered)

    monkeypatch.setattr(startup_script, "next_call", finish_resumed_run)
    scripted_startup(
        ROOT, "wave4-checkpoint", "on", action="resume", turns=0,
        extra_env={"STATEFRAME_CHECKPOINT": "on"},
    )
    rows = [json.loads(line) for line in checkpoint_events.read_text().splitlines()]
    assert any(row.get("event") == "checkpoint_restored" for row in rows), rows
    assert not any(row.get("event") == "checkpoint_restore_refused" for row in rows), rows
