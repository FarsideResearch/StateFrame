"""Actual CLI/provider/sandbox coverage; run these outside nested OS sandboxes."""
import json
import os
import re
import signal
import subprocess
import sys
import time
import zlib
from pathlib import Path

import pytest
from _timeouts import scaled
from fixtures.process_signals import signal_run_owned

from stateframe.scripted_endpoint import Script, serve
from stateframe.request_records import list_request_records, read_request_record
from engine_run import run_monitored, start_monitored
from _journal import measurement_events

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox


def step(tool, **arguments):
    return {"tool": tool, "required": True, "arguments": arguments}


def fixture_repo(tmp_path, *, failing=False):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/__init__.py").write_text("")
    (repo / "pkg/value.py").write_text("VALUE = 1\n")
    (repo / "tests/test_value.py").write_text(
        f"from pkg.value import VALUE\ndef test_value():\n    assert VALUE == {2 if failing else 1}\n")
    return repo


def environment(*, identity_epoch=None):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "src/stateframe/engine")))}
    if identity_epoch is not None:
        env["STATEFRAME_SHELL_ENV_IDENTITY_EPOCH"] = identity_epoch
    return env


def command(repo, run, url, flags=()):
    return [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Inspect the requested shell results, fix VALUE if needed, and verify with run_tests.",
            "--endpoint", url, "--model", "scripted", "--run-dir", str(run),
            "--test-python", sys.executable, "--max-turns", "16", "--quiet", *flags]


def run_script(tmp_path, steps, *, failing=False, flags=(), prepare=None,
               repo=None, run_name="run", identity_epoch=None):
    repo = repo or fixture_repo(tmp_path, failing=failing)
    if prepare:
        prepare(repo)
    script = Script({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}})
    server = serve(script)
    run = tmp_path / run_name
    try:
        result = run_monitored(
            command(repo, run, f"http://127.0.0.1:{server.server_port}/v1", flags),
            env=environment(identity_epoch=identity_epoch), capture_output=True, text=True,
            expected_seconds=scaled(240),
            run_dir=run,
        )
        summary_path = run / "summary.json"
        assert summary_path.is_file(), ("run wrote no summary.json\nSTDOUT:\n" + result.stdout[-4000:]
                                        + "\nSTDERR:\n" + result.stderr[-4000:])
        summary = json.loads(summary_path.read_text())
        assert result.returncode == 0, result.stdout + result.stderr + str(summary)
        assert summary["succeeded"], summary
        script.assert_consumed()
        requests = "\n".join(read_request_record(run, name).decode()
                               for name in list_request_records(run))
        return run, requests
    finally:
        server.shutdown()
        server.server_close()


def finish():
    return [step("run_tests", target="tests/test_value.py"), step("done", summary="Verified the product.")]


def test_echo_roundtrip(tmp_path):
    _, requests = run_script(tmp_path, [step("shell", command="echo hi"), *finish()])
    assert "$ echo hi" in requests and "hi" in requests


def test_readonly_touch_refusal(tmp_path):
    run, requests = run_script(tmp_path, [step("shell", command="touch x"), *finish()],
                               flags=("--shell", "read-only"))
    assert "REFUSED READ_ONLY_POLICY:" in requests
    assert not (run / "workspace/x").exists()


def test_first_send_defers_shell_baseline_then_restores_protected_test(tmp_path, monkeypatch):
    from kernel_workspace_refs import RootKey, open_run_store, read_root, workspace_run_id

    run = tmp_path / "run"
    first_send_retained = []
    next_call = Script.next_call

    def baseline_retained():
        if (run / ".workspace-objects.json").is_file():
            key = RootKey(workspace_run_id(run), "shell-baseline", "parent")
            with open_run_store(run).lease() as lease:
                return read_root(lease, key) is not None
        return any((run / "shell-baseline").glob("*/objects.pack"))

    def inspect_first_send(script, offered):
        if script.position == 0:
            first_send_retained.append(baseline_retained())
        return next_call(script, offered)

    monkeypatch.setattr(Script, "next_call", inspect_first_send)
    _, requests = run_script(tmp_path, [step("shell", command="printf 'assert False\\n' > tests/test_value.py"),
                                      *finish()], flags=("--protect-tests", "all"))
    assert first_send_retained == [False]
    assert baseline_retained()
    assert "restored (not allowed): tests/test_value.py" in requests


def test_shell_product_edit_refreshes_read_and_final_diff(tmp_path):
    run, requests = run_script(tmp_path, [step("read", path="pkg/value.py"),
        step("shell", command="printf 'VALUE = 2\\n' > pkg/value.py"),
        step("read", path="pkg/value.py"), *finish()], failing=True)
    assert "+VALUE = 2" in (run / "final.diff").read_text()
    assert "files changed by this command: modified: pkg/value.py" in requests
    events = measurement_events(run)
    assert '"shell_effect"' in json.dumps(events)
    # The read after the shell edit must see a NEW revision with the new bytes, not the stale page.
    revisions = re.findall(r"\[read page\] path=pkg/value\.py revision=(sha256:[0-9a-f]+)", requests)
    assert len(set(revisions)) >= 2, revisions
    assert "VALUE = 2" in requests


def test_ask_declines_without_tty(tmp_path):
    _, requests = run_script(tmp_path, [step("shell", command="echo hi"), *finish()],
                               flags=("--shell-ask",))
    assert "REFUSED ASK_DECLINED:" in requests


def local_package(repo):
    package = repo / "localpkg"
    package.mkdir()
    (package / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n')
    (package / "backend.py").write_text('''import zipfile
from pathlib import Path

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = "shell_fixture-1.0-py3-none-any.whl"
    with zipfile.ZipFile(Path(wheel_directory) / name, "w") as wheel:
        wheel.writestr("shell_fixture.py", "VALUE = 42\\n")
        wheel.writestr("shell_fixture-1.0.dist-info/METADATA", "Metadata-Version: 2.1\\nName: shell-fixture\\nVersion: 1.0\\n")
        wheel.writestr("shell_fixture-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n")
        wheel.writestr("shell_fixture-1.0.dist-info/RECORD", "")
    return name
''')
    (repo / "tests/test_value.py").write_text(
        "def test_value():\n    import shell_fixture\n    assert shell_fixture.VALUE == 42\n")


def test_local_pip_install_is_available_to_engine_and_final_checks(tmp_path, clean_test_environment):
    # A clean interpreter: the harness venv has StateFrame installed editable from outside the
    # repository, which import authority correctly refuses to clone into a run environment.
    python = clean_test_environment(tmp_path / "test-env")
    run, requests = run_script(tmp_path, [
        step("shell", command="pip install --no-index --no-build-isolation ./localpkg"),
        # The engine verifies automatically after the environment change; once green the menu
        # (correctly) no longer offers run_tests, so that step is optional here.
        {**step("run_tests", target="tests/test_value.py"), "required": False},
        step("done", summary="Installed the dependency and verified.")],
        flags=("--shell-network", "full", "--test-python", str(python)), prepare=local_package)
    assert "environment changed:" in requests
    assert "Successfully installed shell-fixture-1.0" in requests
    checks = [json.loads(zlib.decompress(path.read_bytes()))
              for path in sorted((run / "checks").glob("*.json.z"))]
    assert checks and all(check["argv"][0] == str(run / "env/bin/python") for check in checks)
    assert checks[-1]["returncode"] == 0
    summary = json.loads((run / "summary.json").read_text())
    assert summary["tests_after"]["command"].startswith(str(run / "env/bin/python"))
    assert summary["tests_after"]["passed"]


def test_kill_resume_serves_byte_identical_next_request(tmp_path):
    """Pause the second provider response, after the shell completion is durable."""
    import threading

    class PausedScript(Script):
        def __init__(self):
            super().__init__({"model": "scripted", "steps": [step("shell", command="echo hi"), *finish()],
                              "tokenize": {"mode": "ok"}})
            self.arrived = threading.Event()
            self.release = threading.Event()
            self.pause = True

        def next_call(self, offered):
            if self.position == 1 and self.pause:
                self.arrived.set()
                assert self.release.wait(scaled(120)), "test did not release paused provider"
            return super().next_call(offered)

    repo = fixture_repo(tmp_path)
    run = tmp_path / "run"
    script = PausedScript()
    server = serve(script)
    url = f"http://127.0.0.1:{server.server_port}/v1"
    process = start_monitored(
        command(repo, run, url), env=environment(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, run_dir=run, expected_seconds=scaled(120),
        step="interrupted shell run",
    )
    try:
        while not script.arrived.wait(0.1):
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"first attempt stopped before the provider pause: {stdout}{stderr}")
        assert script.arrived.is_set()
        tape = [json.loads(line) for line in (run / "tool-tape.jsonl").read_text().splitlines()]
        assert [row["kind"] for row in tape] == ["intent", "complete"]
        assert tape[0]["call_id"] == tape[1]["call_id"]
        signal_run_owned(run, process, signal.SIGTERM, deadline=time.monotonic() + scaled(120),
                         what="shell completion replay")
        process.wait(timeout=scaled(20))
        original = {name: read_request_record(run, name) for name in list_request_records(run)}
        assert original and (run / "tool-tape.jsonl").exists()
        script.pause = False
        script.release.set()
        # The interrupted response can finish in the provider thread. Reset only
        # the endpoint boundary's cursor; the engine must replay the shell itself.
        deadline = time.monotonic() + scaled(5)
        while script.position == 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        script.position = 1
        result = run_monitored([sys.executable, "-m", "stateframe", "resume", str(run),
                                 "--endpoint", url, "--model", "scripted", "--quiet"],
                                env=environment(), capture_output=True, text=True, expected_seconds=scaled(240))
        assert result.returncode == 0, result.stdout + result.stderr
        rebuilt = {name: read_request_record(run, name) for name in list_request_records(run)}
        assert original.keys() <= rebuilt.keys()
        for name, body in original.items():
            assert rebuilt[name] == body, name
        assert not (run / "resume-divergence.json").exists()
    finally:
        script.release.set()
        if process.poll() is None:
            # Failure cleanup may precede the first checkpoint.
            signal_run_owned(run, process, signal.SIGTERM, deadline=time.monotonic(),
                             what="shell replay cleanup", require_checkpoint_saved=False)
            process.wait(timeout=scaled(20))
        server.shutdown()
        server.server_close()
def test_kill_resume_preserves_repeated_read_request_bytes(tmp_path):
    """The resumed third request must retain the second read's unchanged notice."""
    import threading

    class PausedScript(Script):
        def __init__(self):
            super().__init__({"model": "scripted", "steps": [
                step("read", path="pkg/value.py"),
                step("read", path="pkg/value.py"),
                *finish(),
            ], "tokenize": {"mode": "ok"}})
            self.arrived = threading.Event()
            self.release = threading.Event()
            self.pause = True

        def next_call(self, offered):
            if self.position == 2 and self.pause:
                self.arrived.set()
                assert self.release.wait(scaled(120)), "test did not release paused provider"
            return super().next_call(offered)

    repo = fixture_repo(tmp_path)
    run = tmp_path / "run"
    script = PausedScript()
    server = serve(script)
    url = f"http://127.0.0.1:{server.server_port}/v1"
    env = environment()  # Both switches use their default engine values.
    process = start_monitored(
        command(repo, run, url), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, run_dir=run, expected_seconds=scaled(120),
        step="interrupted repeated-read run",
    )
    try:
        while not script.arrived.wait(0.1):
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"first attempt stopped before the provider pause: {stdout}{stderr}")
        assert script.arrived.is_set()
        original = {name: read_request_record(run, name) for name in list_request_records(run)}
        assert len(original) >= 3
        assert any(b"unchanged since request" in body for body in original.values())
        signal_run_owned(run, process, signal.SIGTERM, deadline=time.monotonic() + scaled(120),
                         what="repeated read replay")
        process.wait(timeout=scaled(20))
        script.pause = False
        script.release.set()
        deadline = time.monotonic() + scaled(5)
        while script.position == 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        script.position = 2
        result = run_monitored([sys.executable, "-m", "stateframe", "resume", str(run),
                                 "--endpoint", url, "--model", "scripted", "--quiet"],
                                env=env, capture_output=True, text=True, expected_seconds=scaled(240))
        assert result.returncode == 0, result.stdout + result.stderr
        rebuilt = {name: read_request_record(run, name) for name in list_request_records(run)}
        assert original.keys() <= rebuilt.keys()
        for name, body in original.items():
            assert rebuilt[name] == body, name
        assert not (run / "resume-divergence.json").exists()
    finally:
        script.release.set()
        if process.poll() is None:
            # Failure cleanup may precede the first checkpoint.
            signal_run_owned(run, process, signal.SIGTERM, deadline=time.monotonic(),
                             what="repeated read cleanup", require_checkpoint_saved=False)
            process.wait(timeout=scaled(20))
        server.shutdown()
        server.server_close()


_JOURNAL_VOLATILE_FIELDS = {
    "event_id", "previous_hash", "record_hash", "run_id", "task_id", "sequence",
    "timestamp_ns", "tokenize_latency_ms", "tokenize_route", "tokenizer_identity",
    "result_sha256",
}


def _without_timing(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            name = str(key).lower()
            if (key in _JOURNAL_VOLATILE_FIELDS
                    or any(part in name for part in ("latency", "duration", "elapsed", "wall_time"))
                    or name.endswith(("_ns", "_ms", "_seconds"))):
                continue
            normalized[key] = _without_timing(item)
        return normalized
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    if isinstance(value, str):
        normalized = re.sub(r"(?<=\| )\d+(?:\.\d+)?s(?= \|)", "<timing>s", value)
        # Action-ledger hashes cover shell observations, including measured duration text.
        return re.sub(r'("result_sha256"\s*:\s*")[0-9a-f]{64}(")',
                      r"\1<timing-derived>\2", normalized)
    return value


def _normalized_requests(run):
    requests = [read_request_record(run, name) for name in list_request_records(run)]
    return tuple(json.dumps(_without_timing(json.loads(raw)), sort_keys=True,
                            separators=(",", ":")).encode() for raw in requests)


def _normalized_tool_tape(run):
    path = run / "tool-tape.jsonl"
    if not path.is_file():
        return ()
    return tuple(json.dumps(_without_timing(json.loads(line)), sort_keys=True,
                            separators=(",", ":"))
                 for line in path.read_text().splitlines() if line.strip())


def test_environment_identity_epoch_keeps_request_and_journal_bytes(tmp_path):
    repo = fixture_repo(tmp_path)
    steps = [step("shell", command="echo hi"), *finish()]
    enabled, _ = run_script(tmp_path, steps, repo=repo, run_name="epoch-on",
                            identity_epoch="on")
    disabled, _ = run_script(tmp_path, steps, repo=repo, run_name="epoch-off",
                             identity_epoch="off")

    assert _normalized_requests(enabled) == _normalized_requests(disabled)
    # The tool tape is the durable shell command journal; its outcome must be unchanged too.
    assert _normalized_tool_tape(enabled) == _normalized_tool_tape(disabled)
