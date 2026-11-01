"""CLI-level hook proofs using the real runner, child, sandbox and scripted endpoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stateframe.profiles import DEFAULT as DEFAULT_PROFILE
from stateframe.scripted_endpoint import Script, serve


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.host_sandbox


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg/value.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests/test_value.py").write_text(
        "from pkg.value import VALUE\ndef test_value():\n    assert VALUE in (1, 3)\n",
        encoding="utf-8",
    )
    return repo


def _hook_file(path: Path, hooks: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"schema": "stateframe-hooks/1", "hooks": hooks}),
        encoding="utf-8",
    )
    return path


def _hook(name: str, event: str, code: str) -> dict[str, object]:
    return {
        "name": name,
        "event": event,
        "argv": [sys.executable, "-c", code],
    }


def _environment(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return {
        **os.environ,
        "HOME": str(home),
        "PYTHONPATH": str(ROOT / "src"),
    }


def _step(tool: str, **arguments: object) -> dict[str, object]:
    return {"tool": tool, "required": True, "arguments": arguments}


def _steps() -> list[dict[str, object]]:
    return [
        _step(
            "str_replace",
            path="pkg/value.py",
            old_str="VALUE = 1",
            new_str="VALUE = 3",
        ),
        _step("run_tests", target="tests/test_value.py"),
        _step("done", summary="VALUE is now three and the test passes."),
    ]


def _run(
    repo: Path,
    run_dir: Path,
    *,
    environment: dict[str, str],
    hooks: Path | None = None,
    in_place: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Script]:
    script = Script(
        {"model": "scripted", "steps": _steps(), "tokenize": {"mode": "ok"}}
    )
    server = serve(script)
    command = [
        sys.executable,
        "-m",
        "stateframe",
        "run",
        "--repo",
        str(repo),
        "--task",
        "Change VALUE from one to three and verify the test.",
        "--endpoint",
        f"http://127.0.0.1:{server.server_port}/v1",
        "--model",
        "scripted",
        "--run-dir",
        str(run_dir),
        "--test-python",
        sys.executable,
        "--max-turns",
        "8",
    ]
    if hooks is not None:
        command.extend(("--hooks", str(hooks)))
    if in_place:
        command.append("--in-place")
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=240,
        )
        script.assert_consumed()
        return result, run_dir, script
    finally:
        server.shutdown()
        server.server_close()


def _records(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "hooks/records.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prompt_submit_refusal_exits_two_without_starting_child(tmp_path):
    repo = _repo(tmp_path)
    config = _hook_file(
        tmp_path / "hooks.json",
        [
            _hook(
                "prompt-gate",
                "prompt_submit",
                "import sys;sys.stderr.write('review required');sys.exit(2)",
            )
        ],
    )
    script = Script({"model": "scripted", "steps": [], "tokenize": {"mode": "ok"}})
    server = serve(script)
    run_dir = tmp_path / "prompt-refused-run"
    child_started = tmp_path / "child-started"
    interpreter = tmp_path / "python-probe"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"sentinel = {str(child_started)!r}\n"
        "if any(argument.endswith('_child.py') for argument in sys.argv[1:]):\n"
        "    with open(sentinel, 'w', encoding='utf-8') as marker:\n"
        "        marker.write('started')\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    environment = _environment(tmp_path)
    command = [
        sys.executable,
        "-m",
        "stateframe",
        "run",
        "--repo",
        str(repo),
        "--task",
        "Do not start until the review is complete.",
        "--endpoint",
        f"http://127.0.0.1:{server.server_port}/v1",
        "--model",
        "scripted",
        "--run-dir",
        str(run_dir),
        "--hooks",
        str(config),
        "--test-python",
        sys.executable,
        "--python",
        str(interpreter),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 2, result.stdout + result.stderr
    assert "HOOK_DENIED: prompt-gate: review required" in result.stdout + result.stderr
    records = _records(run_dir)
    prompt = [record for record in records if record["event"] == "prompt_submit"]
    assert len(prompt) == 1
    assert prompt[0]["decision"] == "deny"
    assert prompt[0]["exit_code"] == 2
    assert not (run_dir / "engine-result.json").exists()
    assert not (run_dir / "engine.log").exists()
    assert not (run_dir / "worker").exists()
    assert not child_started.exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["engine_result"] == "HOOK_DENIED"


def test_session_end_reports_in_place_apply_and_final_outcome(tmp_path):
    repo = _repo(tmp_path)
    config = _hook_file(
        tmp_path / "hooks.json",
        [
            _hook(
                "end-report",
                "session_end",
                "import json,sys;d=json.load(sys.stdin)['data'];"
                "print('END outcome=%s applied=%s' % (d['outcome'],str(d['applied']).lower()))",
            )
        ],
    )
    environment = _environment(tmp_path)
    result, run_dir, _ = _run(
        repo,
        tmp_path / "in-place-run",
        environment=environment,
        hooks=config,
        in_place=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["succeeded"] is True
    assert summary["applied_in_place"] == ["pkg/value.py"]
    assert (repo / "pkg/value.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    terminal_line = 'hooks: end-report session_end output: "END outcome=completed applied=true"'
    assert terminal_line in result.stdout
    records = [record for record in _records(run_dir) if record["event"] == "session_end"]
    assert len(records) == 1
    assert records[0]["hook"] == "end-report"
    assert records[0]["decision"] == "continue"
    assert records[0]["exit_code"] == 0
    assert records[0]["replayed"] is False


def test_project_hooks_trust_cli_flow_and_changed_file_is_untrusted(tmp_path):
    repo = _repo(tmp_path)
    project_dir = repo / ".stateframe"
    project_dir.mkdir()
    script_path = project_dir / "context.py"
    script_path.write_text("print('PROJECT_CONTEXT_ONE')\n", encoding="utf-8")
    project_config = _hook_file(
        project_dir / "hooks.json",
        [
            {
                "name": "project-context",
                "event": "session_start",
                "argv": [sys.executable, "context.py"],
                "files": ["context.py"],
            }
        ],
    )
    environment = _environment(tmp_path)

    untrusted, untrusted_run, _ = _run(
        repo, tmp_path / "untrusted-run", environment=environment
    )
    assert untrusted.returncode == 0, untrusted.stdout + untrusted.stderr
    assert "project hooks not loaded:" in untrusted.stdout
    assert "is not trusted (or changed)" in untrusted.stdout
    assert "hooks" not in json.loads(
        (untrusted_run / "spec.json").read_text(encoding="utf-8")
    )
    assert b"PROJECT_CONTEXT_ONE" not in (
        untrusted_run / "wire/0001.request.json"
    ).read_bytes()
    assert not _records(untrusted_run)

    trusted = subprocess.run(
        [
            sys.executable,
            "-m",
            "stateframe",
            "hooks",
            "trust",
            "--repo",
            str(repo),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert trusted.returncode == 0, trusted.stdout + trusted.stderr
    assert "project-context" in trusted.stdout
    assert "trusted project hooks" in trusted.stdout

    loaded, loaded_run, _ = _run(
        repo, tmp_path / "trusted-run", environment=environment
    )
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    assert "hooks: project file trusted (digest " in loaded.stdout
    loaded_spec = json.loads((loaded_run / "spec.json").read_text(encoding="utf-8"))
    assert "hooks" in loaded_spec
    first_request = (loaded_run / "wire/0001.request.json").read_bytes()
    assert b"[USER HOOKS]" in first_request
    assert b"PROJECT_CONTEXT_ONE" in first_request
    records = [record for record in _records(loaded_run) if record["event"] == "session_start"]
    assert len(records) == 1
    assert records[0]["hook"] == "project-context"
    assert records[0]["decision"] == "continue"
    assert records[0]["exit_code"] == 0

    script_path.write_text("print('PROJECT_CONTEXT_TWO')\n", encoding="utf-8")
    changed, changed_run, _ = _run(
        repo, tmp_path / "changed-run", environment=environment
    )
    assert changed.returncode == 0, changed.stdout + changed.stderr
    assert "project hooks not loaded:" in changed.stdout
    assert "is not trusted (or changed)" in changed.stdout
    assert "hooks" not in json.loads(
        (changed_run / "spec.json").read_text(encoding="utf-8")
    )
    changed_wire = (changed_run / "wire/0001.request.json").read_bytes()
    assert b"PROJECT_CONTEXT_ONE" not in changed_wire
    assert b"PROJECT_CONTEXT_TWO" not in changed_wire
    assert not _records(changed_run)
    assert project_config.is_file()


def test_shipped_profile_puts_hook_context_in_first_wire_request(tmp_path):
    repo = _repo(tmp_path)
    (repo / "pkg/value.py").write_text("VALUE = 1   \n", encoding="utf-8")
    config = ROOT / "examples/hooks/hooks.json"
    result, run_dir, _ = _run(
        repo,
        tmp_path / "profile-run",
        environment=_environment(tmp_path),
        hooks=config,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (run_dir / "workspace/pkg/value.py").read_text(encoding="utf-8") == "VALUE = 3\n"
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    assert spec["profile"] == DEFAULT_PROFILE
    request_path = run_dir / "wire/0001.request.json"
    request = request_path.read_bytes()
    assert b"[USER HOOKS]" in request
    assert b"Project context is ready. Inspect the task before editing." in request
    document = json.loads(request)
    panels = [
        message
        for message in document["messages"]
        if str(message.get("content", "")).startswith("[USER HOOKS]")
    ]
    assert len(panels) == 1
    assert "| Project context is ready. Inspect the task before editing." in panels[0]["content"]
    assert "_request_budget_section" not in panels[0]
    records = _records(run_dir)
    start = [record for record in records if record["event"] == "session_start"]
    assert len(start) == 1
    assert start[0]["hook"] == "session-context"
    assert start[0]["decision"] == "continue"
    assert start[0]["exit_code"] == 0
    sample_hooks = {
        "format-changed-python",
        "lint-changed-python",
        "require-passing-check",
    }
    executed = {
        record["hook"]: record
        for record in records
        if record["hook"] in sample_hooks
    }
    assert set(executed) == {
        "format-changed-python",
        "lint-changed-python",
        "require-passing-check",
    }
    assert all(record["decision"] == "continue" for record in executed.values())
    assert all(record["exit_code"] == 0 for record in executed.values())
