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

from stateframe.scripted_endpoint import Script, serve

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


def environment():
    return {**os.environ, "PYTHONPATH": os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "src/stateframe/engine")))}


def command(repo, run, url, flags=()):
    return [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Inspect the requested shell results, fix VALUE if needed, and verify with run_tests.",
            "--endpoint", url, "--model", "scripted", "--run-dir", str(run),
            "--test-python", sys.executable, "--max-turns", "16", "--quiet", *flags]


def run_script(tmp_path, steps, *, failing=False, flags=(), prepare=None):
    repo = fixture_repo(tmp_path, failing=failing)
    if prepare:
        prepare(repo)
    script = Script({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}})
    server = serve(script)
    run = tmp_path / "run"
    try:
        result = subprocess.run(command(repo, run, f"http://127.0.0.1:{server.server_port}/v1", flags),
                                env=environment(), capture_output=True, text=True, timeout=240)
        summary = json.loads((run / "summary.json").read_text())
        assert result.returncode == 0, result.stdout + result.stderr + str(summary)
        assert summary["succeeded"], summary
        script.assert_consumed()
        requests = "\n".join(p.read_text() for p in sorted((run / "requests").glob("*.json")))
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


def test_protected_test_restored(tmp_path):
    _, requests = run_script(tmp_path, [step("shell", command="printf 'assert False\\n' > tests/test_value.py"),
                                      *finish()])
    assert "restored (not allowed): tests/test_value.py" in requests


def test_shell_product_edit_refreshes_read_and_final_diff(tmp_path):
    run, requests = run_script(tmp_path, [step("read", path="pkg/value.py"),
        step("shell", command="printf 'VALUE = 2\\n' > pkg/value.py"),
        step("read", path="pkg/value.py"), *finish()], failing=True)
    assert "+VALUE = 2" in (run / "final.diff").read_text()
    assert "files changed by this command: modified: pkg/value.py" in requests
    events = "\n".join(p.read_text() for p in (run / "worker").rglob("events.jsonl"))
    assert '"shell_effect"' in events
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
                assert self.release.wait(120), "test did not release paused provider"
            return super().next_call(offered)

    repo = fixture_repo(tmp_path)
    run = tmp_path / "run"
    script = PausedScript()
    server = serve(script)
    url = f"http://127.0.0.1:{server.server_port}/v1"
    output = (tmp_path / "first.log").open("w")
    process = subprocess.Popen(command(repo, run, url), env=environment(), stdout=output,
                               stderr=subprocess.STDOUT)
    try:
        assert script.arrived.wait(120), (tmp_path / "first.log").read_text()
        tape = [json.loads(line) for line in (run / "tool-tape.jsonl").read_text().splitlines()]
        assert [row["kind"] for row in tape] == ["intent", "complete"]
        assert tape[0]["call_id"] == tape[1]["call_id"]
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=20)
        original = {p.name: p.read_bytes() for p in (run / "requests").glob("*.json")}
        assert original and (run / "tool-tape.jsonl").exists()
        script.pause = False
        script.release.set()
        # The interrupted response can finish in the provider thread. Reset only
        # the endpoint boundary's cursor; the engine must replay the shell itself.
        deadline = time.monotonic() + 5
        while script.position == 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        script.position = 1
        result = subprocess.run([sys.executable, "-m", "stateframe", "resume", str(run),
                                 "--endpoint", url, "--model", "scripted", "--quiet"],
                                env=environment(), capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr
        rebuilt = {p.name: p.read_bytes() for p in (run / "requests").glob("*.json")}
        assert original.keys() <= rebuilt.keys()
        for name, body in original.items():
            assert rebuilt[name] == body, name
        assert not (run / "resume-divergence.json").exists()
    finally:
        script.release.set()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        output.close()
        server.shutdown()
        server.server_close()
