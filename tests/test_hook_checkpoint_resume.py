"""Checkpoint resume keeps hook authority and request bytes across an interrupted run."""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tarfile
import types
import zlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from _timeouts import scaled

import kernel_checkpoint
from kernel_hook_wiring import HookWiring
from kernel_hooks import create_bundle, load_hook_files
from kernel_checkpoint import SKIP_RUN_ENTRIES, _restore_files
from stateframe import _child, resume
from stateframe.request_records import PACK_NAME as REQUEST_PACK_NAME
from stateframe.request_records import list_request_records, seal_request_records
from stateframe.engine import kernel_journal_store
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve
from fixtures.process_signals import (record_identity, signal_run_owned, signal_run_verified,
                                      wait_for_checkpoint_saved)
from engine_run import run_monitored, start_monitored


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _legacy_direct_engine_checkpoint(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE", "1")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_child_treats_skipped_checkpoint_as_full_replay() -> None:
    skipped = {"status": "skipped", "reason": "hooks_configured"}
    selected = {"sends": 1, "step": 2, "dir": "/checkpoint", "tail": 1}
    assert _child._selected_resume_checkpoint({"checkpoint": skipped}) is None
    assert _child._selected_resume_checkpoint({"checkpoint": selected}) is selected
    assert _child._selected_resume_checkpoint({}) is None


def test_checkpoint_file_restore_keeps_complete_hook_tape(tmp_path: Path) -> None:
    """The live logical tape may contain records newer than the saved checkpoint."""
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "saved"
    (checkpoint / "run").mkdir(parents=True)
    tape = run / "hook-tape.jsonl"
    tape.write_text('checkpoint intent\nrecorded tail\n')
    assert tape.name in SKIP_RUN_ENTRIES

    _restore_files(checkpoint, run, {"entries": []})

    assert tape.read_text() == 'checkpoint intent\nrecorded tail\n'


def test_checkpoint_excludes_pinned_and_live_hook_entries(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run"
    bundle = run / "hook-bundle" / "0"
    bundle.mkdir(parents=True)
    (bundle / "script.py").write_text("print('pinned')\n")
    (bundle / "script.py").chmod(0o400)
    bundle.chmod(0o500)
    bundle.parent.chmod(0o500)
    records = run / "hooks"
    records.mkdir()
    (records / "records.jsonl").write_text("old record\n")
    (records / "delivery.jsonl").write_text("old delivery\n")
    scratch = run / "hook-scratch"
    scratch.mkdir()
    (scratch / "live").write_text("scratch\n")
    (run / "hook-tape.jsonl").write_text("old tape\n")
    worker = run / "worker" / "continuation-workspaces" / ("a" * 32)
    worker.mkdir(parents=True)
    (worker / "owner.lock").write_text("lease\n")

    old_state = kernel_checkpoint._STATE.copy()
    old_key = kernel_checkpoint._SIGNING_KEY.copy()
    try:
        monkeypatch.setattr(kernel_checkpoint, "_is_state_module", lambda *_args: False)
        monkeypatch.setattr(kernel_checkpoint, "_context_items", lambda: [])
        frame = types.SimpleNamespace(f_code=test_checkpoint_excludes_pinned_and_live_hook_entries.__code__,
                                      f_locals={"step": 1})
        kernel_checkpoint._STATE["setup"] = {"code": frame.f_code, "line": 0, "funcs": frozenset()}
        kernel_checkpoint.register_signing_key(b"checkpoint-test-key-0123456789abcdef")
        checkpoint = kernel_checkpoint.save(frame, 1, run)
        entries = json.loads((checkpoint / "checkpoint.json").read_text())["entries"]
        assert "worker" in entries
        assert set(entries).isdisjoint({"hook-bundle", "hooks", "hook-scratch", "hook-tape.jsonl"})
        assert not any((checkpoint / "run" / name).exists() for name in
                       ("hook-bundle", "hooks", "hook-scratch", "hook-tape.jsonl"))

        (records / "records.jsonl").write_text("old record\nnew record\n")
        (records / "delivery.jsonl").write_text("old delivery\nnew delivery\n")
        (run / "hook-tape.jsonl").write_text("old tape\nnew tape\n")
        _restore_files(checkpoint, run, {"entries": entries})
        assert (bundle / "script.py").read_text() == "print('pinned')\n"
        assert (records / "records.jsonl").read_text() == "old record\nnew record\n"
        assert (records / "delivery.jsonl").read_text() == "old delivery\nnew delivery\n"
        assert (run / "hook-tape.jsonl").read_text() == "old tape\nnew tape\n"
        assert (scratch / "live").read_text() == "scratch\n"
        assert (worker / "owner.lock").read_text() == "lease\n"
        resume._compact_archived_checkpoints(run / "checkpoints")
        assert not (checkpoint / "run").exists()
    finally:
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE.update(old_state)
        kernel_checkpoint._SIGNING_KEY[:] = old_key
        bundle.chmod(0o700)
        bundle.parent.chmod(0o700)
        (bundle / "script.py").chmod(0o600)


def test_checkpoint_refuses_live_hook_entry_before_moving_files(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "saved"
    (checkpoint / "run" / "hooks").mkdir(parents=True)
    records = run / "hooks"
    records.mkdir()
    (records / "delivery.jsonl").write_text("live\n")

    with pytest.raises(kernel_checkpoint.CheckpointError, match="live run entry 'hooks'"):
        _restore_files(checkpoint, run, {"entries": ["hooks"]})
    assert (records / "delivery.jsonl").read_text() == "live\n"
    assert not any((run / "checkpoints").glob("setup-discard.*"))


def test_hook_runner_rebinds_restored_workspace_and_pending_action(tmp_path: Path) -> None:
    run = tmp_path / "run"
    old_root = tmp_path / "setup" / "repo"
    restored_root = tmp_path / "restored" / "repo"
    for root in (old_root, restored_root):
        root.mkdir(parents=True)
        (root / "app.py").write_text("VALUE = 1\n")
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [
        {"name": "guard", "event": "pre_tool", "tools": ["str_replace"],
         "argv": [sys.executable, "-c", "print('ok')"]},
    ]}))
    hooks = load_hook_files([config], shell_network="none", repo=old_root)
    create_bundle(hooks, run / "hook-bundle")
    spec = {"repo": str(old_root), "run_identity_seed": "seed",
            "tool_settings": {"shell_network": "none"}, "hooks": {"start_context": []}}
    wiring = HookWiring(run, hooks, spec, replay=lambda: False)
    row = {"files": {"app.py": "VALUE = 1\n"}, "scoped_edits": False}
    emitted = []

    def runtime():
        return SimpleNamespace(journal=SimpleNamespace(
            run_id="run-id", emit=lambda kind, **fields: emitted.append((kind, fields))))

    try:
        setup_runtime = runtime()
        wiring.bind(old_root, row, set(), None, setup_runtime)
        wiring._actions["pending"] = {"refusal": None, "name": "str_replace"}
        saved = dict(wiring.state)
        wiring.state.clear()
        wiring.state.update(saved)
        restored_runtime = runtime()
        wiring.rebind_after_checkpoint(restored_root, row, set(), restored_runtime)
        assert wiring.runner.workspace == restored_root
        assert wiring.runtime is restored_runtime
        assert restored_runtime._hook_wiring is wiring
        assert wiring._actions is wiring.state["actions"]
        assert wiring._actions["pending"]["name"] == "str_replace"
        assert [kind for kind, _ in emitted] == ["user_hook_config"]
    finally:
        wiring.close()
        bundle = run / "hook-bundle"
        for directory in (bundle / "0", bundle):
            directory.chmod(0o700)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/__init__.py").write_text("")
    (repo / "pkg/value.py").write_text("VALUE = 1\n")
    (repo / "tests/test_value.py").write_text(
        "from pkg.value import VALUE\ndef test_value():\n    assert VALUE in (1, 3)\n"
    )
    return repo


def _steps(variant: str) -> list[dict]:
    def step(tool: str, **arguments: object) -> dict:
        return {"tool": tool, "required": True, "arguments": arguments}

    if variant in {"no_hooks", "routes"}:
        return [
            step("read", path="pkg/value.py"),
            step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 3"),
            step("read", path="pkg/value.py"),
            step("run_tests", target="tests/test_value.py"),
            step("done", summary="Checks passed after the edit."),
        ]
    if variant == "hooks_after_first_edit":
        return [
            step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 4"),
            step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 2"),
            step("str_replace", path="pkg/value.py", old_str="VALUE = 3", new_str="VALUE = 4"),
            step("done", summary="Checks passed after formatting."),
            step("done", summary="Checks passed after formatting."),
        ]
    return [
        step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 2"),
        step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 4"),
        step("str_replace", path="pkg/value.py", old_str="VALUE = 3", new_str="VALUE = 4"),
        step("done", summary="Checks passed after formatting."),
        step("done", summary="Checks passed after formatting."),
    ]


def _hooks() -> list[dict]:
    return [
        {"name": "start", "event": "session_start", "argv": [
            sys.executable, "-c", "print('start marker')"]},
        {"name": "prompt", "event": "prompt_submit", "argv": [
            sys.executable, "-c", "print('prompt marker')"]},
        {"name": "guard", "event": "pre_tool", "tools": ["str_replace"],
         "argv": [sys.executable, "-c",
                  "import json,sys;d=json.load(sys.stdin)['data'];"
                  "deny=d['args'].get('new_str')=='VALUE = 2';"
                  "print(json.dumps({'decision':'deny' if deny else 'continue',"
                  "'reason':'blocked' if deny else ''}))"]},
        {"name": "audit", "event": "post_tool", "tools": ["str_replace"],
         "argv": [sys.executable, "-c", "print('post marker')"]},
        {"name": "fmt", "event": "post_tool", "tools": ["str_replace"],
         "changed": True, "writes": "changed", "argv": [
             sys.executable, "-c",
             "import pathlib,sys;p=pathlib.Path(sys.argv[1]);"
             "p.write_text(p.read_text().replace('VALUE = 4','VALUE = 3'))",
             "{paths}"]},
        {"name": "gate", "event": "stop", "max_blocks": 2,
         "argv": [sys.executable, "-c",
                  "import json,sys;d=json.load(sys.stdin)['data'];"
                  "deny=d['blocks_so_far']==0;"
                  "sys.stderr.write('run checks' if deny else '');"
                  "sys.exit(2 if deny else 0)"]},
        {"name": "end", "event": "session_end", "argv": [
            sys.executable, "-c", "print('end marker')"]},
    ]


class InterruptAfterActions(Script):
    def __init__(self, document: dict, count: int):
        super().__init__(document)
        self.count = count
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.interrupted = threading.Event()

    def next_call(self, offered):
        if self.position == self.count and not self.interrupted.is_set():
            self.waiting.set()
            if not self.release.wait(scaled(60)):
                raise TimeoutError("interrupted request was not released")
            self.interrupted.set()
            raise ScriptStepUnavailable("request interrupted before a response")
        return super().next_call(offered)


def _launcher(tmp_path: Path) -> Path:
    launcher = tmp_path / "fixed-seed.py"
    launcher.write_text(
        "import secrets, sys\n"
        "from stateframe.cli import main\n"
        "original = secrets.token_hex\n"
        "used = False\n"
        "def fixed(size=None):\n"
        "    global used\n"
        "    if size == 16 and not used:\n"
        "        used = True\n"
        "        return 'a' * 32\n"
        "    return original(size)\n"
        "secrets.token_hex = fixed\n"
        "raise SystemExit(main(sys.argv[1:]))\n"
    )
    return launcher


def _requests(run: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))}


def _receipts_by_order(requests: dict[str, bytes], first: str) -> dict[str, bytes]:
    order: dict[bytes, bytes] = {}

    def token(match: re.Match) -> bytes:
        value = match.group(2)
        order.setdefault(value, b"receipt-%d" % len(order))
        return match.group(1) + order[value]

    pattern = re.compile(rb'(edit_receipt\\?":\\?")([0-9a-f]{20})')
    normalized = {name: pattern.sub(token, data) for name, data in sorted(requests.items())}
    return {name: data for name, data in normalized.items() if name >= first}


@pytest.mark.host_sandbox
@pytest.mark.parametrize("variant,completed_actions,checkpoint_every", (
    ("hooks", 3, 3),
    ("hooks_after_first_edit", 1, 1),
    ("hooks_routes", 3, 3),
    ("no_hooks", 3, 3),
    ("routes", 3, 3),
))
def test_checkpoint_resume_matches_uninterrupted_control(
    tmp_path: Path, variant: str, completed_actions: int, checkpoint_every: int,
) -> None:
    repo = _repo(tmp_path)
    config = tmp_path / "hooks.json"
    has_hooks = variant.startswith("hooks")
    if has_hooks:
        config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": _hooks()}))
    launcher = _launcher(tmp_path)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((
        str(ROOT / "src"), str(ROOT / "src/stateframe/engine")))}
    document = {"model": "scripted", "steps": _steps(variant), "tokenize": {"mode": "ok"}}
    script = InterruptAfterActions(document, completed_actions)
    server = serve(script)
    run = tmp_path / "interrupted"
    process = None
    try:
        profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
        profile["switches"]["STATEFRAME_CHECKPOINT_EVERY"] = str(checkpoint_every)
        profile_path = tmp_path / "checkpoint-profile.json"
        profile_path.write_text(json.dumps(profile))
        flags = ["--hooks", str(config)] if has_hooks else []
        if variant in {"routes", "hooks_routes"}:
            flags.extend(["--route", "read.temperature=0.75"])
        command = [
            sys.executable, str(launcher), "run", "--repo", str(repo),
            "--task", "Inspect and update VALUE.", "--endpoint",
            f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            "--run-dir", str(run), "--test-python", sys.executable,
            "--max-turns", "8", "--shell", "off", "--keep-checkpoints",
            "--profile", str(profile_path),
            *flags, "--quiet",
        ]
        process = start_monitored(command, env=environment, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, run_dir=run,
                                  expected_seconds=scaled(600), step="checkpoint resume interrupted run",
                                  watch_paths=(run / "checkpoints",))
        while not script.waiting.wait(0.5):
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"first attempt stopped before the interrupted request: {stdout}{stderr}")
        completed = {
            name: data for name, data in _requests(run).items()
            if (run / "wire" / name.replace(".request.", ".response.")).exists()
        }
        assert len(completed) == completed_actions
        first_new_request = f"{completed_actions + 1:04d}.request.json"
        pending = (run / "wire" / first_new_request).read_bytes()
        prior_tape_bytes = (run / "hook-tape.jsonl").read_bytes() if has_hooks else b""
        prior_tape = _rows(run / "hook-tape.jsonl") if has_hooks else []
        if variant == "hooks_after_first_edit":
            old_records = _rows(run / "hooks/records.jsonl")
            assert {"pre_tool", "post_tool"} <= {record["event"] for record in old_records}
            assert any(record["event"] == "pre_tool" and record["executed"] for record in old_records)
            assert any(record["event"] == "post_tool" and record["executed"] for record in old_records)
        checkpoint_events_path = run / "checkpoints/events.jsonl"
        checkpoint_deadline = time.monotonic() + scaled(120)
        wait_for_checkpoint_saved(run, deadline=checkpoint_deadline,
                                  what="hook checkpoint resume")
        if variant == "hooks_after_first_edit":
            saved = [row for row in _rows(checkpoint_events_path)
                     if row["event"] == "checkpoint_saved"]
            assert any(json.loads((run / "checkpoints" / row["dir"] / "checkpoint.json").read_text())
                       ["sends"] == 1 for row in saved)
        child_pids = subprocess.check_output(["pgrep", "-P", str(process.pid)],
                                             text=True).split()
        assert len(child_pids) == 1, child_pids
        child = int(child_pids[0])
        identity = record_identity(child, launched_after=0, command_fragment="_child.py")
        signal_run_verified(run, child, identity, signal.SIGKILL, deadline=checkpoint_deadline,
                            what="hook checkpoint resume")
        script.release.set()
        assert script.interrupted.wait(scaled(10))
        process.communicate(timeout=scaled(90))

        control_script = Script(document)
        control_server = serve(control_script)
        control = tmp_path / "control"
        try:
            control_command = command.copy()
            control_command[control_command.index(str(run))] = str(control)
            control_command[control_command.index(
                f"http://127.0.0.1:{server.server_port}/v1"
            )] = f"http://127.0.0.1:{control_server.server_port}/v1"
            result = run_monitored(control_command, env=environment, run_dir=control,
                                   step="uninterrupted checkpoint control",
                                   capture_output=True, text=True, expected_seconds=scaled(240))
            assert result.returncode == 0, result.stdout + result.stderr
            control_script.assert_consumed()
        finally:
            control_server.shutdown()
            control_server.server_close()
        control_requests = _requests(control)
        assert len(control_requests) >= len(_steps(variant))

        resumed = run_monitored(
            [sys.executable, "-m", "stateframe", "resume", str(run),
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--keep-checkpoints"],
            env=environment, capture_output=True, text=True, expected_seconds=scaled(240),
        )
        if resumed.returncode != 0:
            expected = script.steps[script.position - 1] if script.position else None
            offered = ([tool["function"]["name"] for tool in
                        script.requests[-1].get("tools", [])] if script.requests else [])
            pytest.fail(resumed.stdout + resumed.stderr +
                        f"\nscript step: {script.position}; expected: {expected}; offered: {offered}")
        script.assert_consumed()
        assert set(_requests(run).items()) >= set(completed.items())
        assert _requests(run)[first_new_request] == pending
        spec = json.loads((run / "spec.json").read_text())
        assert not (run / "checkpoint-refused.json").exists()
        assert not (run / "attempts/2").exists()
        checkpoint_events = _rows(run / "checkpoints/events.jsonl")
        assert not any(row["event"] == "checkpoint_restore_refused" for row in checkpoint_events)
        engine_events = [json.loads(line.removeprefix("[sf] ")) for line in
                         (run / "engine.log").read_text().splitlines()
                         if line.startswith("[sf] ")]
        assert not any(row["kind"] == "checkpoint_unfaithful" for row in engine_events)
        if variant == "no_hooks":
            assert 0 <= spec["resume"]["checkpoint"]["sends"] <= len(completed)
            assert any(row["event"] == "checkpoint_restored" for row in checkpoint_events)
            assert any(row["kind"] == "checkpoint_verified" for row in engine_events)
            assert "checkpoint: engine state restored as of send" in resumed.stdout
        else:
            reason = "routes_configured" if variant == "routes" else "hooks_configured"
            assert spec["resume"]["checkpoint"] == {
                "status": "skipped", "reason": reason,
            }
            assert sum(row["event"] == "checkpoint_skipped" and
                       row.get("reason") == reason for row in checkpoint_events) == 1
            assert not any(row["event"] == "checkpoint_restored" for row in checkpoint_events)
            assert not any(row["kind"] == "checkpoint_verified" for row in engine_events)
            setting = "per-step models" if variant == "routes" else "hooks"
            skip_line = (f"checkpoint: not restored because {setting} are configured for this run; "
                         "replaying the recorded exchanges instead")
            output_lines = resumed.stdout.splitlines()
            assert output_lines.count(skip_line) == 1
            assert output_lines[output_lines.index(skip_line) + 1].startswith("replay: ")

        # The interrupted run's own pre-kill bytes are checked exactly above. Against the separate
        # control run, edit receipt IDs differ between independent runs (KNOWN_LIMITATIONS), so they
        # are mapped to their order of first appearance; every other byte must match.
        resumed_requests = _requests(run)
        assert _receipts_by_order(resumed_requests, first_new_request) == _receipts_by_order(
            control_requests, first_new_request)
        assert (run / "workspace/pkg/value.py").read_bytes() == (
            control / "workspace/pkg/value.py").read_bytes()

        if has_hooks:
            tape = _rows(run / "hook-tape.jsonl")
            assert (run / "hook-tape.jsonl").read_bytes().startswith(prior_tape_bytes)
            assert tape[:len(prior_tape)] == prior_tape
            intents = [row["record_id"] for row in tape if row["kind"] == "intent"]
            assert all(count == 1 for count in Counter(intents).values())
            assert all(sum(row["record_id"] == ident and row["kind"] == "complete"
                           for row in tape) == 1 for ident in intents)
            old_ids = set(row["record_id"] for row in prior_tape)
            records = _rows(run / "hooks/records.jsonl")
            assert (run / "attempts/1/checkpoints/events.jsonl").is_file()
            replayed_old = [record for record in records if record["event_id"] in old_ids]
            assert replayed_old
            assert all(record["replayed"] and not record["executed"] for record in replayed_old)
            assert {record["event"] for record in records +
                    _rows(run / "attempts/1/hooks/records.jsonl")} == {
                        "session_start", "prompt_submit", "pre_tool", "post_tool",
                        "stop", "session_end",
                    }
    finally:
        if process is not None and process.poll() is None:
            # Failure cleanup may precede any completed checkpoint.
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="hook resume cleanup", require_checkpoint_saved=False)
            process.communicate(timeout=scaled(10))
        server.shutdown()
        server.server_close()


@pytest.mark.host_sandbox
def test_keyed_run_saves_secret_free_checkpoints_and_resumes_from_latest(
        tmp_path: Path) -> None:
    key = "CKPTW_LOCAL_DUMMY_KEY_123456789"
    repo = _repo(tmp_path)
    launcher = _launcher(tmp_path)
    home = tmp_path / "isolated-home"
    state_home = home / ".stateframe"
    state_home.mkdir(parents=True)
    environment = {**os.environ,
                   "PYTHONPATH": os.pathsep.join((
                       str(ROOT / "src"), str(ROOT / "src/stateframe/engine"))),
                   "HOME": str(home),
                   "LAB_KEY": key}
    environment["STATEFRAME_HOME"] = str(state_home)
    seed = subprocess.run(
        [sys.executable, "-c", (
            "from stateframe.engine.kernel_request_auth_identity import _installation_secret; "
            "_installation_secret()"
        )],
        env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if seed.returncode:
        pytest.fail("could not initialize the isolated identity key")
    steps = _steps("no_hooks")
    steps.insert(-1, {"tool": "git_commit", "required": True,
                      "arguments": {"message": "Record the value correction"}})
    document = {
        "model": "scripted",
        "steps": steps,
        "tokenize": {"mode": "ok"},
        "chat": {"require_key": key},
    }
    script = InterruptAfterActions(document, 2)
    server = serve(script)
    run = tmp_path / "keyed-run"
    process = None

    def output_without_key(result) -> str:
        return (result.stdout + result.stderr).replace(key, "[redacted]")

    def secret_marker_rows() -> tuple[tuple[str, bytes], ...]:
        key_path = state_home / "identity.key"
        if not key_path.is_file():
            pytest.fail("the keyed run did not create its isolated identity.key")
        identity_secret = key_path.read_bytes()
        if len(identity_secret) != 32:
            pytest.fail(f"identity.key length was {len(identity_secret)}, expected 32")
        identity_hash = hashlib.sha256(identity_secret).hexdigest()
        return (
            ("raw identity bytes", identity_secret),
            ("identity hex", identity_secret.hex().encode("ascii")),
            ("identity sha256", identity_hash.encode("ascii")),
        )

    def artifact_payloads(path: Path) -> list[bytes]:
        raw = path.read_bytes()
        payloads = [raw]
        for decoder in (zlib.decompress, lzma.decompress):
            try:
                decoded = decoder(raw)
            except (OSError, EOFError, lzma.LZMAError, zlib.error):
                continue
            if decoded != raw:
                payloads.append(decoded)
        for candidate in tuple(payloads):
            try:
                with tarfile.open(fileobj=io.BytesIO(candidate), mode="r:*") as archive:
                    for member in archive:
                        if member.isfile():
                            source = archive.extractfile(member)
                            if source is not None:
                                with source:
                                    payloads.append(source.read())
            except (OSError, EOFError, tarfile.TarError):
                pass
        if path.name == "pack-0001.sfpk":
            records, _valid_end = kernel_journal_store._scan_blob_pack(path)
            for blob_hash in records:
                packed = kernel_journal_store._read_packed_blob(path.parent, blob_hash, records)
                payloads.append(packed)
                try:
                    payloads.append(zlib.decompress(packed))
                except zlib.error:
                    pass
        return payloads

    def assert_artifact_tree_has_no_identity_markers(
            roots: tuple[Path, ...], marker_rows: tuple[tuple[str, bytes], ...],
            identity_sha256_prefix: str) -> None:
        for root in roots:
            if root.is_file() or root.is_symlink():
                paths = [root]
            elif root.is_dir():
                paths = root.rglob("*")
            else:
                continue
            for path in paths:
                if not path.is_file():
                    continue
                for payload in artifact_payloads(path):
                    for marker_name, marker in marker_rows:
                        if marker in payload:
                            pytest.fail(
                                f"{marker_name} found in {path.name}; identity key length=32, "
                                f"sha256={identity_sha256_prefix}"
                            )

    def assert_checkpoint_bytes_are_secret_free() -> None:
        checkpoint_root = run / "checkpoints"
        pickle_files = [path for path in checkpoint_root.rglob("*")
                        if path.is_file() and (
                            path.name == kernel_checkpoint.STATE_FILE
                            or path.name.endswith(".pkl")
                            or path.name.endswith(kernel_checkpoint.CHUNK_SUFFIX))]
        assert pickle_files, "no pickled checkpoint files were found"
        request_key = key.encode()
        request_key_sha256 = hashlib.sha256(request_key).hexdigest()[:12]
        for path in pickle_files:
            data = path.read_bytes()
            if request_key in data:
                pytest.fail(
                    f"registered request key found in {path.relative_to(run)} "
                    f"(length={len(request_key)}, sha256={request_key_sha256})"
                )
            if path.name.endswith(kernel_checkpoint.CHUNK_SUFFIX):
                if request_key in zlib.decompress(data):
                    pytest.fail(
                        f"registered request key found in decoded {path.relative_to(run)} "
                        f"(length={len(request_key)}, sha256={request_key_sha256})"
                    )

    try:
        profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
        profile["switches"]["STATEFRAME_CHECKPOINT_EVERY"] = "1"
        profile["switches"]["STATEFRAME_GIT_COMMIT"] = "on"
        profile_path = tmp_path / "keyed-checkpoint-profile.json"
        profile_path.write_text(json.dumps(profile))
        command = [
            sys.executable, str(launcher), "run", "--repo", str(repo),
            "--task", "Inspect and update VALUE.", "--endpoint",
            f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            "--run-dir", str(run), "--test-python", sys.executable,
            "--api-key-env", "LAB_KEY", "--max-turns", "8", "--shell", "off",
            "--keep-checkpoints", "--profile", str(profile_path), "--quiet",
        ]
        process = start_monitored(
            command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, run_dir=run, expected_seconds=scaled(600),
            step="keyed checkpoint resume", watch_paths=(run / "checkpoints",),
        )
        while not script.waiting.wait(0.2):
            if process.poll() is not None:
                result = process.communicate()
                pytest.fail("keyed run stopped before the interruption point: "
                            + output_without_key(SimpleNamespace(stdout=result[0], stderr=result[1])))

        events_path = run / "checkpoints/events.jsonl"
        deadline = time.monotonic() + 90
        saved_steps = set()
        while time.monotonic() < deadline:
            events = _rows(events_path) if events_path.exists() else []
            saved_steps = {row.get("step") for row in events
                           if row.get("event") == "checkpoint_saved"}
            if {1, 2} <= saved_steps:
                break
            failures = [row for row in events if row.get("event") == "checkpoint_failed"]
            if failures:
                pytest.fail("keyed checkpoint writer failed: "
                            + str(failures[-1].get("error", "unknown failure")).replace(
                                key, "[redacted]"))
            if process.poll() is not None:
                result = process.communicate()
                pytest.fail("keyed run exited before saving checkpoints: "
                            + output_without_key(SimpleNamespace(stdout=result[0], stderr=result[1])))
            time.sleep(0.05)
        assert {1, 2} <= saved_steps, "checkpoints at steps 1 and 2 were not both saved"
        assert script.chat_authorization_present and all(script.chat_authorized)
        latest_name = (run / "checkpoints/LATEST").read_text().strip()
        latest_meta = json.loads(
            (run / "checkpoints" / latest_name / kernel_checkpoint.META_FILE).read_text())
        assert latest_meta["step"] == 2
        assert_checkpoint_bytes_are_secret_free()

        child_pids = subprocess.check_output(
            ["pgrep", "-P", str(process.pid)], text=True,
        ).split()
        assert len(child_pids) == 1, child_pids
        child = int(child_pids[0])
        identity = record_identity(child, launched_after=0, command_fragment="_child.py")
        signal_run_verified(
            run, child, identity, signal.SIGKILL,
            deadline=deadline, what="keyed checkpoint resume",
        )
        script.release.set()
        assert script.interrupted.wait(scaled(10))
        process.communicate(timeout=scaled(90))

        resumed = run_monitored(
            [sys.executable, "-m", "stateframe", "resume", str(run),
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--api-key-env", "LAB_KEY", "--keep-checkpoints"],
            env=environment, capture_output=True, text=True, expected_seconds=scaled(240),
        )
        if resumed.returncode != 0:
            pytest.fail(output_without_key(resumed))
        script.assert_consumed()

        spec = json.loads((run / "spec.json").read_text())
        restored = [row for row in _rows(events_path)
                    if row.get("event") == "checkpoint_restored"]
        assert restored and restored[-1].get("step") == latest_meta["step"]
        assert Path(spec["resume"]["checkpoint"]["dir"]).resolve() == (
            run / "checkpoints" / latest_name).resolve()
        assert not any(row.get("event") in {"checkpoint_failed", "checkpoint_disabled"}
                       for row in _rows(events_path))
        assert script.chat_authorized and all(script.chat_authorized)
        assert_checkpoint_bytes_are_secret_free()
        request_key = key.encode()
        request_key_sha256 = hashlib.sha256(request_key).hexdigest()[:12]
        if request_key in (run / "engine.log").read_bytes():
            pytest.fail(
                f"registered request key found in engine.log "
                f"(length={len(request_key)}, sha256={request_key_sha256})"
            )

        marker_rows = secret_marker_rows()
        identity_sha256_prefix = hashlib.sha256(marker_rows[0][1]).hexdigest()[:12]
        seal_request_records(run)
        if list_request_records(run) and not (run / REQUEST_PACK_NAME).is_file():
            pytest.fail("run request bundle was not published")
        exported_patches = sorted((run / "commits").rglob("*.patch"))
        if not exported_patches:
            pytest.fail("run did not export its git_commit patch series")
        if not (run / "final.diff").is_file():
            pytest.fail("run did not export final.diff")
        assert_artifact_tree_has_no_identity_markers(
            (run,), marker_rows, identity_sha256_prefix,
        )
    finally:
        if process is not None and process.poll() is None:
            signal_run_owned(
                run, process, signal.SIGKILL, deadline=time.monotonic(),
                what="keyed checkpoint test cleanup", require_checkpoint_saved=False,
            )
            process.communicate(timeout=scaled(10))
        script.release.set()
        server.shutdown()
        server.server_close()
