"""The public CLI binds endpoint credentials before request token measurement."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _journal import measurement_events
from _stateframe_home import isolated_stateframe_home
from stateframe.scripted_endpoint import Script, serve
from engine_run import run_monitored


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.host_sandbox

_KEY = "sf-e2e-tokenizer-secret"
_WRONG_KEY = "sf-e2e-wrong-tokenizer-secret"
_TOKENIZER_ENV = {
    "LLAMA_CPP_TOKENIZE_URL", "LLAMA_CPP_BASE_URL", "A3B_ENDPOINT", "A3B_BASE_URL",
    "OPENAI_API_BASE", "OPENAI_BASE_URL", "DS_URL",
}


def _run(tmp_path: Path, *, key: str) -> tuple[subprocess.CompletedProcess[str], Path, Script]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    script = Script({
        "model": "scripted",
        "steps": [
            {"tool": "read", "arguments": {"path": "app.py"}},
            {"tool": "done", "arguments": {"summary": "VALUE is seven."}},
        ],
        "chat": {"require_key": _KEY},
        "tokenize": {"mode": "ok", "require_key": _KEY},
    })
    server = serve(script)
    environment = {
        **{
            name: value for name, value in os.environ.items()
            if not name.startswith("STATEFRAME_") and name not in _TOKENIZER_ENV
        },
        "STATEFRAME_HOME": isolated_stateframe_home(),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SF_TOKENIZER_E2E_KEY": key,
    }
    try:
        process = run_monitored(
            [
                sys.executable, "-m", "stateframe", "run",
                "--repo", str(repo),
                "--task", "Inspect app.py and report VALUE.",
                "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                "--model", "scripted",
                "--api-key-env", "SF_TOKENIZER_E2E_KEY",
                "--run-dir", str(run_dir),
                "--max-turns", "8",
                "--quiet",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            expected_seconds=240,
            check=False,
        )
        output = process.stdout + process.stderr
        if ("could not start in the isolated startup probe" in output
                and "sandbox_apply: Operation not permitted" in output):
            pytest.skip("sandbox-exec blocks engine startup; rerun this test on the host")
    finally:
        server.shutdown()
        server.server_close()
    return process, run_dir, script


def _request_diet_events(run_dir: Path) -> list[dict[str, object]]:
    return [event for event in measurement_events(run_dir)
            if event.get("kind") == "request_diet_applied"]


def _unavailable_events(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "engine.log"
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if '"kind":"token_counting_unavailable"' not in line:
            continue
        payload = line.partition(" ")[2]
        events.append(json.loads(payload))
    return events


def _assert_run_files_do_not_contain_key(run_dir: Path, key: str) -> None:
    for path in run_dir.rglob("*"):
        if path.is_file():
            assert key.encode() not in path.read_bytes(), f"credential persisted in {path.relative_to(run_dir)}"


def test_public_cli_uses_bound_key_for_every_tokenizer_measurement(tmp_path):
    process, run_dir, script = _run(tmp_path, key=_KEY)

    assert process.returncode == 0, process.stdout + process.stderr
    assert not _unavailable_events(run_dir)
    assert script.chat_authorized and all(script.chat_authorized)
    assert script.tokenize_calls > 0
    assert script.tokenize_authorized and all(script.tokenize_authorized)
    measures = _request_diet_events(run_dir)
    assert measures
    assert all(event.get("measure_method") == "tokenize" for event in measures)
    assert all(int(event.get("tokenizer_calls", 0)) > 0 for event in measures)
    _assert_run_files_do_not_contain_key(run_dir, _KEY)


def test_public_cli_wrong_key_reports_unauthorized_tokenizer_fallback(tmp_path):
    process, run_dir, script = _run(tmp_path, key=_WRONG_KEY)

    events = _unavailable_events(run_dir)
    assert events and all(event.get("reason") == "unauthorized" for event in events), (
        process.stdout + process.stderr
    )
    assert script.chat_authorized and not any(script.chat_authorized)
    assert script.tokenize_authorized and not any(script.tokenize_authorized)
    _assert_run_files_do_not_contain_key(run_dir, _WRONG_KEY)
