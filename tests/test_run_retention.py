"""Finish eligibility and run environment retention follow resume's terminal policy."""
from __future__ import annotations

import json
import hashlib
import sys
from types import SimpleNamespace

import pytest

from stateframe import resume, runner, storage_gc


@pytest.mark.parametrize(("result", "finished"), [
    ("SUBMITTED", True),
    ("HARNESS_CONTRACT", True),
    ("FINALIZATION_ERROR", True),
    ("STOPPED_INCOMPLETE", False),
    ("CANCELLED", False),
    ("KILLED", False),
    ("CRASHED", False),
    ("RESPONSE_BUDGET_EXHAUSTED", False),
    ("INTERRUPTED", False),
])
def test_run_finished_for_good_requires_a_nonresumable_terminal_result(tmp_path, result, finished):
    run = tmp_path / result
    run.mkdir()
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": result}}))

    assert resume.run_finished_for_good(run) is finished


def test_finished_run_env_manifest_is_written_before_env_is_removed(tmp_path):
    run = tmp_path / "run"
    site = run / "env/lib/python3.11/site-packages/demo_pkg-1.2.dist-info"
    site.mkdir(parents=True)
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": "SUBMITTED"}}))
    (run / "env-sites.json").write_text('["/kept/site-packages"]')
    (run / "env/pyvenv.cfg").write_text(
        "home = /usr/bin\nimplementation = CPython\nversion_info = 3.11.8\n")
    (site / "METADATA").write_text("Name: Demo_Pkg\nVersion: 1.2\n\n")
    (site / "RECORD").write_bytes(b"demo_pkg/__init__.py,,\n")
    (site / "direct_url.json").write_text(json.dumps({
        "url": "file:///workspace/demo_pkg", "dir_info": {"editable": True}}))
    (site.parent / "demo_pkg").mkdir()
    (site.parent / "demo_pkg/__init__.py").write_text("VALUE = 1\n")

    removed = runner.clean_finished_run_env(run)

    manifest = json.loads((run / "env-manifest.json").read_text())
    assert removed > 0
    assert not (run / "env").exists()
    assert (run / "env-sites.json").read_text() == '["/kept/site-packages"]'
    assert manifest["python_version"] == "3.11.8"
    assert manifest["python_implementation"] == "CPython"
    assert manifest["platform"] == sys.platform and manifest["machine"]
    assert manifest["interpreter_path"] == str(run / "env/bin/python")
    assert manifest["distributions"] == [{
        "name": "demo-pkg", "version": "1.2",
        "record_sha256": hashlib.sha256(b"demo_pkg/__init__.py,,\n").hexdigest(),
        "record_payload_sha256": hashlib.sha256(b"").hexdigest(),
        "direct_url": {"url": "file:///workspace/demo_pkg", "dir_info": {"editable": True}},
        "editable": True,
    }]
    assert manifest["file_count"] >= 3
    assert manifest["byte_count"] > 0


def test_env_manifest_redacts_direct_url_credentials(tmp_path):
    run = tmp_path / "run"
    site = run / "env/lib/python3.11/site-packages/private_pkg-1.0.dist-info"
    site.mkdir(parents=True)
    (run / "env/pyvenv.cfg").write_text("implementation = CPython\nversion_info = 3.11.8\n")
    (site / "METADATA").write_text("Name: private-pkg\nVersion: 1.0\n\n")
    (site / "RECORD").write_bytes(b"private_pkg/__init__.py,,\n")
    (site / "direct_url.json").write_text(json.dumps({
        "url": "https://user:password@example.test/pkg?access_token=secret&"
               "X-Amz-Signature=signed-secret&ref=main#access_token=fragment-secret",
        "vcs_info": {"commit_id": "abc123"},
    }))

    from stateframe.storage_gc import build_env_manifest
    manifest = build_env_manifest(run)

    url = manifest["distributions"][0]["direct_url"]["url"]
    assert "user" not in url and "password" not in url and "secret" not in url
    assert "redacted" in url.lower() and "ref=main" in url

    from stateframe.storage_gc import _safe_direct_url
    malformed = _safe_direct_url("https://user:secret@example.test/pkg")
    assert malformed == {"_unparsed": True}


def test_env_manifest_uses_run_interpreter_when_venv_config_omits_implementation(tmp_path, monkeypatch):
    run = tmp_path / "run"
    env = run / "env"
    env.mkdir(parents=True)
    (env / "pyvenv.cfg").write_text(f"version = {'.'.join(map(str, sys.version_info[:3]))}\n")
    (run / "spec.json").write_text(json.dumps({"run_env_source_python": sys.executable}))
    monkeypatch.setattr(storage_gc, "sys", SimpleNamespace(
        implementation=SimpleNamespace(name="wrong-interpreter"), version_info=(9, 9, 9),
        platform=sys.platform))

    manifest = storage_gc.build_env_manifest(run)

    assert manifest["python_implementation"] == sys.implementation.name


def test_keep_run_env_switch_and_flag_preserve_environment(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "env").mkdir(parents=True)
    (run / "env/file").write_text("retain")
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": "SUBMITTED"}}))

    assert runner.clean_finished_run_env(run, keep=True) is None
    assert (run / "env/file").read_text() == "retain"

    monkeypatch.setenv("STATEFRAME_KEEP_RUN_ENV", "on")
    assert runner.clean_finished_run_env(run) is None
    assert (run / "env/file").read_text() == "retain"


def test_env_switches_are_host_settings_and_cli_exposes_retention(tmp_path, monkeypatch):
    from stateframe.cli import build_parser

    assert runner.OPT_IN_SWITCHES["STATEFRAME_KEEP_RUN_ENV"] == ("on", "off")
    assert runner.OPT_IN_SWITCHES["STATEFRAME_ENV_SEED"] == ("auto", "copy")
    assert {"STATEFRAME_KEEP_RUN_ENV", "STATEFRAME_ENV_SEED"} <= runner.SHELL_SETTINGS
    monkeypatch.setenv("STATEFRAME_KEEP_RUN_ENV", "on")
    monkeypatch.setenv("STATEFRAME_ENV_SEED", "copy")
    assert not ({"STATEFRAME_KEEP_RUN_ENV", "STATEFRAME_ENV_SEED"} & set(runner.resolve_opt_in_switches()))
    run_args = build_parser().parse_args([
        "run", "--repo", str(tmp_path), "--task", "task", "--endpoint", "http://localhost:8080/v1",
        "--model", "local", "--keep-run-env"])
    assert run_args.keep_run_env is True
    resume_args = build_parser().parse_args(["resume", str(tmp_path), "--keep-run-env"])
    assert resume_args.keep_run_env is True


def test_finish_run_storage_calls_all_sealers_and_cleanup_is_best_effort(tmp_path, monkeypatch, capsys):
    from stateframe import storage_gc

    calls = []
    monkeypatch.setattr(resume, "run_finished_for_good", lambda _run: True)
    monkeypatch.setattr(runner, "clean_finished_checkpoints",
                        lambda *_args, **_kwargs: calls.append("checkpoints") or 11)
    monkeypatch.setattr(runner, "clean_finished_run_env",
                        lambda *_args, **_kwargs: calls.append("env") or 22)
    monkeypatch.setattr(storage_gc, "seal_measurements",
                        lambda *_args, **_kwargs: calls.append("measurements"))
    monkeypatch.setattr(storage_gc, "seal_requests",
                        lambda *_args, **_kwargs: calls.append("requests"))

    result = runner.finish_run_storage(tmp_path)

    assert result == {"checkpoints": 11, "env": 22}
    assert calls == ["checkpoints", "env", "measurements", "requests"]

    def fail(*_args, **_kwargs):
        raise OSError("injected storage failure")

    monkeypatch.setattr(runner, "clean_finished_checkpoints", fail)
    monkeypatch.setattr(runner, "clean_finished_run_env", fail)
    monkeypatch.setattr(storage_gc, "seal_measurements", fail)
    monkeypatch.setattr(storage_gc, "seal_requests", fail)
    result = runner.finish_run_storage(tmp_path)

    assert result == {"checkpoints": None, "env": None}
    assert capsys.readouterr().out.count("warning:") >= 4


def test_checkpoint_cleanup_failure_is_reported_and_does_not_change_result(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/0001/state.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": "SUBMITTED"}}))

    def fail(*_args, ignore_errors=False, **_kwargs):
        if ignore_errors:
            return None
        raise PermissionError("injected removal failure")

    monkeypatch.setattr(runner.shutil, "rmtree", fail)
    result = runner.finish_run_storage(run)

    assert result["checkpoints"] is None
    assert checkpoint.is_file()
    assert "checkpoints were not removed" in capsys.readouterr().out
