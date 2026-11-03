"""Exercise collection guards through the scripted model and real dispatch."""

import json
import shutil
import subprocess
import sys
import zlib

import pytest

from _portable import TRUE
from test_shell_e2e import finish, run_script, step


pytestmark = pytest.mark.host_sandbox


def _journal_events(run):
    return [json.loads(line) for file in (run / "worker").rglob("events.jsonl")
            for line in file.read_text().splitlines()]


@pytest.fixture(autouse=True)
def require_nested_sandbox():
    if sys.platform != "darwin":
        return
    executable = shutil.which("sandbox-exec")
    if executable is None:
        pytest.skip("macOS sandbox-exec is unavailable")
    probe = subprocess.run(
        [executable, "-p", "(version 1) (allow default)", TRUE],
        capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip("host blocks nested sandbox-exec")


@pytest.mark.parametrize("path,content", [
    ("tests/__init__.py", "import summarize\n"),
    ("tests/conftest.py", "# collection hook\n"),
    ("conftest.py", "# collection hook\n"),
])
def test_model_create_collection_file_is_refused(tmp_path, path, content):
    run, requests = run_script(
        tmp_path, [step("file_create", path=path, content=content), *finish()],
        flags=("--protect-tests", "all"),
        prepare=lambda repo: (repo / "summarize.py").write_text("VALUE = 1\n"),
    )
    assert "REFUSED PROTECTED_TEST" in requests
    assert f"new {path.rsplit('/', 1)[-1]} would run at test collection" in requests
    assert not (run / "workspace" / path).exists()
    checks = [json.loads(zlib.decompress(file.read_bytes()))
              for file in (run / "checks").glob("*.json.z")]
    assert checks and all(check["returncode"] == 0 for check in checks)
    details = [event["receipt"]["detail"] for event in _journal_events(run)
               if event.get("kind") == "action_receipt"
               and isinstance(event.get("receipt"), dict)
               and isinstance(event["receipt"].get("detail"), str)]
    assert any("REFUSED PROTECTED_TEST" in detail and path in detail
               for detail in details)


def test_product_create_allowed_under_protection(tmp_path):
    run, requests = run_script(
        tmp_path, [step("file_create", path="pkg/new_module.py", content="VALUE = 2\n"),
                   *finish()], flags=("--protect-tests", "all"))
    assert (run / "workspace/pkg/new_module.py").read_text() == "VALUE = 2\n"
    assert "created pkg/new_module.py" in requests


def test_off_keeps_collection_create_allowed(tmp_path):
    run, requests = run_script(
        tmp_path, [step("file_create", path="tests/__init__.py", content="import summarize\n"),
                   *finish()], flags=("--protect-tests", "off"),
        prepare=lambda repo: (repo / "summarize.py").write_text("VALUE = 1\n"))
    assert "REFUSED PROTECTED_TEST" not in requests
    assert any(event.get("kind") == "action_receipt"
               and isinstance(event.get("receipt"), dict)
               and isinstance(event["receipt"].get("detail"), str)
               and event["receipt"]["detail"].startswith("created")
               and "tests/__init__.py" in event["receipt"]["detail"]
               for event in _journal_events(run))


def test_shell_collection_create_is_restored_and_refused(tmp_path):
    run, requests = run_script(
        tmp_path, [step("shell", command="printf '# hook\\n' > tests/conftest.py"),
                   *finish()], flags=("--protect-tests", "all"))
    assert "REFUSED PROTECTED_TEST" in requests
    assert "restored (not allowed): tests/conftest.py" in requests
    assert not (run / "workspace/tests/conftest.py").exists()
