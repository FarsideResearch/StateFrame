"""Task file scope stays intact through parallel projection and host delivery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace

import pytest

from barrier_endpoint import BarrierEndpoint
from stateframe import cli, runner
from stateframe.scripted_endpoint import Script, serve


pytestmark = pytest.mark.host_sandbox


def _repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "target.py").write_text(
        "def target_value():\n    return 1\n", encoding="utf-8")
    (repo / "pkg" / "quiet.py").write_text("VALUE = 0\n", encoding="utf-8")
    (repo / "tests" / "test_target.py").write_text(
        "from pkg.target import target_value\n\n"
        "def test_target_value():\n    assert target_value() == 2\n", encoding="utf-8")
    (repo / "tests" / "test_quiet.py").write_text(
        "from pkg.quiet import VALUE\n\n"
        "def test_quiet_value():\n    assert VALUE == 0\n", encoding="utf-8")
    return repo


def _snapshot(repo: Path) -> dict[str, bytes]:
    return {path.relative_to(repo).as_posix(): path.read_bytes()
            for path in repo.rglob("*") if path.is_file()}


def _pin_run_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "secrets", SimpleNamespace(
        token_hex=lambda size: "0123456789abcdef" * 2 if size == 16 else secrets.token_hex(size)))


def test_scoped_parallel_cli_submits_only_the_target(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    before = _snapshot(repo)
    run_dir = tmp_path / "run"
    units = [
        {"id": "target", "goal": "Make target_value return 2.",
         "owns": ["pkg/target.py"], "creates": [], "exports": [], "consumes": [],
         "reads": [], "requirements": [], "check": {"tests": ["tests/test_target.py"]}},
        {"id": "quiet", "goal": "Leave the quiet module unchanged.",
         "owns": ["pkg/quiet.py"], "creates": [], "exports": [], "consumes": [],
         "reads": [], "requirements": [], "check": {"tests": ["tests/test_quiet.py"]}},
    ]
    parent = [
        {"tool": "parallel", "arguments": {"subframes": units}},
        {"tool": "done", "arguments": {"summary": "Updated target_value."}},
    ]
    subframes = {
        "target": [{"tool": "str_replace", "arguments": {
            "path": "pkg/target.py", "old_str": "return 1", "new_str": "return 2"}}],
        "quiet": [{"tool": "done", "arguments": {"summary": "No change needed."}}],
    }

    for name in tuple(os.environ):
        if name.startswith("STATEFRAME_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    _pin_run_identity(monkeypatch)

    with BarrierEndpoint(parent, subframes, slots=2) as endpoint:
        exit_code = cli.main([
            "run", "--repo", str(repo), "--task",
            "Fix pkg/target.py to return 2. Only edit pkg/target.py and pkg/quiet.py.",
            "--endpoint", endpoint.url, "--model", "scripted", "--profile", "r2hc3",
            "--run-dir", str(run_dir), "--max-turns", "20", "--parallel", "2",
            "--python", sys.executable, "--test-python", sys.executable, "--shell", "off",
            "--target", "pkg/target.py", "--edit", "pkg/quiet.py", "--check", "tests/test_target.py",
            "--protect-tests", "off", "--in-place", "--quiet", "--json",
        ])
        assert exit_code == 0, capsys.readouterr().out
        endpoint.assert_consumed()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    after = _snapshot(repo)
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))

    assert spec["public_row"]["task_file_scope"] == ["pkg/target.py", "pkg/quiet.py"]
    assert summary["engine_result"] == "SUBMITTED"
    assert summary["candidate_files_written"] == ["pkg/target.py"]
    assert changed == ["pkg/target.py"]
    assert (repo / "pkg" / "target.py").read_text(encoding="utf-8") == (
        "def target_value():\n    return 2\n")


def test_unscoped_cli_request_bytes_match_0_8_23_base(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("A small unscoped task input.\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    for name in tuple(os.environ):
        if name.startswith("STATEFRAME_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    _pin_run_identity(monkeypatch)
    script = Script({"steps": [
        {"tool": "read", "arguments": {"path": "README.md"}},
        {"tool": "done", "arguments": {"summary": "Read the task input."}},
    ]})
    server = serve(script)
    try:
        exit_code = cli.main([
            "run", "--repo", str(repo), "--task", "Inspect README.md and report its contents.",
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            "--profile", "r2hc3", "--run-dir", str(run_dir), "--max-turns", "4", "--parallel", "off",
            "--python", sys.executable, "--test-python", sys.executable, "--shell", "off",
            "--check-cmd", f"{sys.executable} -c pass", "--quiet", "--json",
        ])
    finally:
        server.shutdown()
        server.server_close()
    script.assert_consumed()
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((run_dir / "wire").glob("*.request.json"))]

    assert exit_code == 0, capsys.readouterr().out
    spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    assert "task_file_scope" not in spec["public_row"]
    assert hashes == [
        "d349c7c3ee5edf6ce972b8c9de4a8465d7d463d4bef942f5475463b6abb51a0b",
        "f4955a79421704ffd6527177dc9ac865d35d8fef3936c0ee2f920b5159931b3b",
        "f8cfaed17b7ddbaad181238bc10feef926bc4bacb845844952618c88306f77ae",
        "dde75a76a31a01eac4975e86cab0f74a12c35e1fb7ef46718843d7f9585c4347",
    ]
