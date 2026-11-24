"""Full scripted run through the explicitly selected current-container boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys

from stateframe import runner
from stateframe.endpoint import Endpoint
from stateframe.engine import kernel_execution_boundary as boundary
from stateframe.engine.kernel_tool_settings import ToolSettings
from stateframe.scripted_endpoint import Script, serve


DETECTED = boundary.Detection("container", ("mount:docker",), {"pid": "4026533000"}, 1)


def test_container_run_delivers_diff_and_passes_independent_check(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    repo, run = tmp_path / "repo", tmp_path / "run"
    import shutil
    shutil.copytree(root / "examples" / "discount_bug", repo)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "bwrap-called"
    fake = fake_bin / "bwrap"
    fake.write_text(f"#!/bin/sh\nprintf called >> {str(marker)!r}\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    monkeypatch.setattr(boundary, "_active_boundary", None)
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: DETECTED)
    engine_dir = str(root / "src" / "stateframe" / "engine")
    helper = str(root / "src" / "stateframe" / "outer_checks.py")
    original_popen = subprocess.Popen

    def inject(command, *args, **kwargs):
        argv = list(command) if isinstance(command, (list, tuple)) else command
        if isinstance(argv, list) and argv and argv[-1] == "stateframe._child":
            index = argv.index("-c") + 1
            argv[index] = (
                "import os,sys;os.geteuid=lambda:54321;"
                f"sys.path.insert(0,{engine_dir!r});import kernel_execution_boundary as b;"
                "b.detect=lambda **kw:b.Detection('container',('mount:docker',),"
                "{'pid':'4026533000'},1);" + argv[index]
            )
        elif isinstance(argv, list) and len(argv) >= 5 and argv[-1] == "--helper" \
                and str(argv[-2]).endswith("outer_checks.py"):
            argv = [argv[0], "-I", "-B", "-c",
                    "import os,runpy,sys;os.geteuid=lambda:54321;"
                    f"sys.path.insert(0,{engine_dir!r});import kernel_execution_boundary as b;"
                    "b.detect=lambda **kw:b.Detection('container',('mount:docker',),"
                    "{'pid':'4026533000'},1);sys.argv=["
                    f"{helper!r},'--helper'];runpy.run_path({helper!r},run_name='__main__')",
                    "--helper"]
        return original_popen(argv, *args[1:], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", inject)
    script_data = json.loads((root / "examples" / "discount_bug.script.json").read_text())
    script_data["steps"].insert(2, {"tool": "shell", "arguments": {"command": "echo checked"}})
    script = Script(script_data)
    server = serve(script)
    try:
        options = runner.RunOptions(
            repo=repo, task="Fix the pricing code so the tests pass.",
            endpoint=Endpoint(f"http://127.0.0.1:{server.server_port}/v1", "scripted"),
            run_dir=run, test_python=sys.executable, max_turns=8, quiet=True,
            tool_settings=ToolSettings(shell="unrestricted", shell_network="full", run_env=False),
            instructions="off", no_project_hooks=True, no_powerups=True,
        )
        options._execution_boundary = boundary.Boundary(boundary.BACKEND_CONTAINER, DETECTED)
        summary = runner.run(options, printer=lambda *_args: None)
    finally:
        server.shutdown()
        server.server_close()

    summary = json.loads((run / "summary.json").read_text())
    assert summary["exit_code"] == 0, summary.get("engine_error_reason", summary)
    assert summary["succeeded"] is True
    assert summary["execution_boundary"]["backend"] == "container"
    assert "(100 - percent)" in (run / "final.diff").read_text()
    assert not marker.exists()
    assert options._capabilities["schema_validation"]["enabled"] is True
    script.assert_consumed()
    check = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_pricing.py"],
                           cwd=run / "workspace", capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stdout + check.stderr
