"""Run environments rehydrate from an exact seed or report every distribution mismatch."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from stateframe import rehydrate


def _distribution(root, name="demo-pkg", version="1.0", record=b"demo_pkg/__init__.py,,\n"):
    dist = root / "lib/python3.11/site-packages" / f"{name}-{version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n\n")
    (dist / "RECORD").write_bytes(record)
    (dist.parent / "demo_pkg").mkdir(exist_ok=True)
    (dist.parent / "demo_pkg/__init__.py").write_text("VALUE = 1\n")
    return dist


def _manifest(run, record):
    manifest = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_implementation": sys.implementation.name,
        "platform": sys.platform,
        "machine": platform.machine(),
        "interpreter_path": str(run / "env/bin/python"),
        "source_interpreter_path": sys.executable,
        "seed_key": "seed-1",
        "system_site_packages": False,
        "distributions": [{"name": "demo-pkg", "version": "1.0",
                            "record_sha256": hashlib.sha256(record).hexdigest()}],
    }
    (run / "env-manifest.json").write_text(json.dumps(manifest))
    return manifest


def _exact_seed_run(tmp_path, monkeypatch):
    home = tmp_path / "home"
    seed = home / "env-seeds/seed-1"
    seed.mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    record = b"demo_pkg/__init__.py,,\n"
    _manifest(run, record)
    (seed / "bin").mkdir()
    (seed / "bin/python").symlink_to(sys.executable)
    (seed / "bin/pip").write_text(f"#!/bin/sh\nexec {seed / 'bin/python'} \"$@\"\n")
    (seed / "pyvenv.cfg").write_text(
        f"home = {sys.prefix}\nimplementation = {sys.implementation.name}\n"
        f"version = {'.'.join(map(str, sys.version_info[:3]))}\n")
    _distribution(seed, record=record)
    from stateframe.engine import kernel_env_seed
    files, directories = kernel_env_seed._inventory(seed)
    for relative, row in files.items():
        if "mode" in row:
            (seed / relative).chmod(row["mode"] & ~0o222)
    (seed / "seed.json").write_text(json.dumps({"key": "seed-1", "files": files,
                                                "directories": directories,
                                                "inputs": {"relocation": {
                                                    "source": str(seed), "source_repo": str(run)}}}))
    (run / "env-seed.json").write_text(json.dumps({"path": str(seed)}))
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    return run, record


def test_rehydrate_reuses_exact_matching_seed(tmp_path, monkeypatch):
    run, record = _exact_seed_run(tmp_path, monkeypatch)

    result = rehydrate.rehydrate_run(run)

    assert result.exact is True
    assert result.used_seed is True
    assert (run / "env/lib/python3.11/site-packages/demo-pkg-1.0.dist-info/RECORD").read_bytes() == record
    assert f"exec {run / 'env'}/bin/python" in (run / "env/bin/pip").read_text()
    assert not (run / "env/seed.json").exists()


def test_inexact_rehydrate_prints_all_mismatches_and_returns_nonzero_result(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "env-seeds/seed-1").mkdir(parents=True)
    run = tmp_path / "run"
    run.mkdir()
    expected_record = b"demo_pkg/__init__.py,,\n"
    _manifest(run, expected_record)
    (run / "env-seed.json").write_text(json.dumps({"path": str(home / "env-seeds/seed-1")}))
    monkeypatch.setenv("STATEFRAME_HOME", str(home))

    def fake_install(destination, _manifest, **_kwargs):
        _distribution(destination, version="2.0", record=b"changed\n")
        (destination / "bin").mkdir()
        (destination / "bin/python").write_text("python\n")

    monkeypatch.setattr(rehydrate, "_build_environment", fake_install)
    lines = []
    result = rehydrate.rehydrate_run(run, offline=True, printer=lines.append)

    assert result.exact is False
    assert result.exit_code != 0
    assert any("INEXACT" in line for line in lines)
    assert any("demo-pkg" in line and "version" in line for line in lines)
    assert any("demo-pkg" in line and "RECORD" in line for line in lines)
    assert not (run / "env-seed.json").exists()


def test_rehydrate_rewrites_staging_interpreter_paths_before_publish(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    record = b"demo_pkg/__init__.py,,\n"
    _manifest(run, record)

    def fake_install(destination, _manifest, **_kwargs):
        _distribution(destination, record=record)
        executable = destination / "bin/demo-tool"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text(f"#!{destination}/bin/python\nprint('ok')\n")

    monkeypatch.setattr(rehydrate, "_build_environment", fake_install)

    result = rehydrate.rehydrate_run(run)

    script = (run / "env/bin/demo-tool").read_text()
    assert result.exact is True
    assert f"#!{run / 'env'}/bin/python" in script
    assert ".env.rehydrate-" not in script


def test_rehydrate_dry_run_does_not_create_env(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _manifest(run, b"")
    lines = []

    result = rehydrate.rehydrate_run(run, dry_run=True, printer=lines.append)

    assert result.dry_run is True
    assert not (run / "env").exists()
    assert not (run / ".run.lock").exists()
    assert any("would rebuild" in line.lower() for line in lines)


def test_rehydrate_requires_the_recorded_python_version(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    manifest = _manifest(run, b"")
    monkeypatch.setattr(rehydrate, "_probe_python", lambda _interpreter: ("0.0.0", "cpython"))

    with pytest.raises(rehydrate.RehydrateError, match="Python version mismatch"):
        rehydrate._build_environment(tmp_path / "env", manifest, run_dir=run, offline=True)

    assert not (tmp_path / "env").exists()


def test_rehydrate_refuses_a_live_run_lock(tmp_path):
    from stateframe.resume import hold_run_lock

    run = tmp_path / "run"
    run.mkdir()
    _manifest(run, b"")

    with hold_run_lock(run), pytest.raises(rehydrate.RehydrateError, match="in use"):
        rehydrate.rehydrate_run(run)

    assert not (run / "env").exists()


def test_offline_rehydrate_uses_only_local_wheelhouse_links(tmp_path, monkeypatch):
    run = tmp_path / "run"
    wheelhouse = run / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    (wheelhouse / "demo_pkg-1.0-py3-none-any.whl").write_bytes(b"wheel")
    interpreter = tmp_path / "python"
    interpreter.write_text("python")
    interpreter.chmod(0o755)
    version = ".".join(map(str, sys.version_info[:3]))
    manifest = {
        "python_version": version,
        "python_implementation": sys.implementation.name,
        "source_interpreter_path": str(interpreter),
        "system_site_packages": False,
        "distributions": [{"name": "demo-pkg", "version": "1.0"}],
    }
    monkeypatch.setattr(rehydrate, "_probe_python", lambda _path: (version, sys.implementation.name))
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(rehydrate.subprocess, "run", fake_run)

    rehydrate._build_environment(tmp_path / "env", manifest, run_dir=run, offline=True)

    install = calls[1][calls[1].index("install") + 1:]
    links = [Path(install[index + 1]) for index, value in enumerate(install[:-1])
             if value == "--find-links"]
    assert install[0] == "--no-index"
    assert wheelhouse in links
    assert install[-1] == "demo-pkg==1.0"


def test_rehydrate_cli_accepts_offline_dry_run(tmp_path, capsys):
    from stateframe.cli import main

    run = tmp_path / "run"
    run.mkdir()
    _manifest(run, b"")

    assert main(["rehydrate", str(run), "--offline", "--dry-run"]) == 0
    assert "offline" in capsys.readouterr().out.lower()


def test_rehydrate_reuses_the_seed_after_a_kernel_update(tmp_path, monkeypatch):
    """A kernel or OS patch release changes platform.platform(), not the seed's compatibility."""
    run, _ = _exact_seed_run(tmp_path, monkeypatch)
    monkeypatch.setattr(platform, "platform", lambda *args, **kwargs: "Linux-9.9.9-other-kernel")

    result = rehydrate.rehydrate_run(run)

    assert result.exact is True
    assert result.used_seed is True


def test_rebuild_installs_local_projects_from_their_path_and_never_asks_the_index(tmp_path, monkeypatch):
    """A run's editable StateFrame (or any direct-URL project) is not on the index under that version."""
    run = tmp_path / "run"
    run.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    interpreter = tmp_path / "python"
    interpreter.write_text("python")
    interpreter.chmod(0o755)
    version = ".".join(map(str, sys.version_info[:3]))
    manifest = {
        "python_version": version,
        "python_implementation": sys.implementation.name,
        "source_interpreter_path": str(interpreter),
        "system_site_packages": False,
        "distributions": [
            {"name": "demo-pkg", "version": "1.0", "direct_url": None, "editable": False},
            {"name": "local-app", "version": "0.3.11", "editable": True,
             "direct_url": {"url": project.as_uri(), "dir_info": {"editable": True}}},
            {"name": "gone-app", "version": "2.0", "editable": False,
             "direct_url": {"url": (tmp_path / "missing").as_uri(), "dir_info": {}}},
        ],
    }
    monkeypatch.setattr(rehydrate, "_probe_python", lambda _path: (version, sys.implementation.name))
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(rehydrate.subprocess, "run", fake_run)
    lines = []

    rehydrate._build_environment(tmp_path / "env", manifest, run_dir=run, offline=False,
                                 printer=lines.append)

    installs = [command[command.index("install") + 1:] for command in calls if "install" in command]
    flat = [argument for command in installs for argument in command]
    assert "demo-pkg==1.0" in flat
    assert not any(argument.startswith(("local-app==", "gone-app==")) for argument in flat)
    assert ["-e", str(project)] in [flat[index:index + 2] for index in range(len(flat) - 1)]
    assert any("gone-app" in line and "not available" in line for line in lines)


def test_comparison_uses_payload_digest_when_both_sides_record_it():
    expected = [{"name": "pkg", "version": "1.0", "record_sha256": "raw-a", "record_payload_sha256": "p"}]
    same_payload = [{"name": "pkg", "version": "1.0", "record_sha256": "raw-b", "record_payload_sha256": "p"}]
    other_payload = [{"name": "pkg", "version": "1.0", "record_sha256": "raw-b", "record_payload_sha256": "q"}]
    legacy = [{"name": "pkg", "version": "1.0", "record_sha256": "raw-b"}]

    assert rehydrate.compare_distributions(expected, same_payload) == []
    assert rehydrate.compare_distributions(expected, other_payload)
    assert rehydrate.compare_distributions(expected, legacy)


def test_editable_distribution_compares_by_version_and_source_not_shim_record():
    """An editable install's code lives in its project; RECORD lists only rebuildable shim files."""
    source = {"url": "file:///src/app", "dir_info": {"editable": True}}
    expected = [{"name": "app", "version": "1.0", "record_sha256": "a", "record_payload_sha256": "x",
                 "editable": True, "direct_url": source}]
    rebuilt = [{"name": "app", "version": "1.0", "record_sha256": "b", "record_payload_sha256": "y",
                "editable": True, "direct_url": source}]
    moved = [{**rebuilt[0], "direct_url": {"url": "file:///elsewhere", "dir_info": {"editable": True}}}]

    assert rehydrate.compare_distributions(expected, rebuilt) == []
    assert rehydrate.compare_distributions(expected, moved)


def test_rebuild_removes_pip_when_the_recorded_environment_had_none(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    interpreter = tmp_path / "python"
    interpreter.write_text("python")
    interpreter.chmod(0o755)
    version = ".".join(map(str, sys.version_info[:3]))
    manifest = {"python_version": version, "python_implementation": sys.implementation.name,
                "source_interpreter_path": str(interpreter), "system_site_packages": False,
                "distributions": [{"name": "demo-pkg", "version": "1.0"}]}
    monkeypatch.setattr(rehydrate, "_probe_python", lambda _path: (version, sys.implementation.name))
    calls = []
    monkeypatch.setattr(rehydrate.subprocess, "run", lambda command, **_kwargs: calls.append(command)
                        or subprocess.CompletedProcess(command, 0, stdout="", stderr=""))

    rehydrate._build_environment(tmp_path / "env", manifest, run_dir=run, offline=False)

    assert calls[-1][-4:] == ["pip", "uninstall", "-y", "pip"]


def test_rehydrate_reuses_a_seed_whose_venv_config_says_version_info(tmp_path, monkeypatch):
    """uv writes version_info in pyvenv.cfg where venv writes version."""
    run, _ = _exact_seed_run(tmp_path, monkeypatch)
    seed = tmp_path / "home/env-seeds/seed-1"
    config = seed / "pyvenv.cfg"
    config.chmod(0o644)
    config.write_text(config.read_text().replace("version = ", "version_info = "))
    from stateframe.engine import kernel_env_seed
    files, directories = kernel_env_seed._inventory(seed)
    files.pop("seed.json", None)
    manifest = json.loads((seed / "seed.json").read_text())
    for relative, row in files.items():
        if "mode" in row:
            (seed / relative).chmod(row["mode"] & ~0o222)
            row["mode"] = row["mode"] & ~0o222
    (seed / "seed.json").write_text(json.dumps({**manifest, "files": files, "directories": directories}))

    result = rehydrate.rehydrate_run(run)

    assert result.used_seed is True
    assert result.exact is True


def test_seed_that_cannot_be_restored_falls_back_to_a_loud_rebuild(tmp_path, monkeypatch):
    run, record = _exact_seed_run(tmp_path, monkeypatch)
    from stateframe.engine.kernel_run_env import RunEnvError

    def refused(*_args, **_kwargs):
        raise RunEnvError("external import path lacks host read authorization: /elsewhere")

    monkeypatch.setattr("stateframe.engine.kernel_run_env._relocate", refused)

    def rebuilt(destination, _manifest, **_kwargs):
        _distribution(destination, record=record)
        (destination / "bin").mkdir()
        (destination / "bin/python").write_text("python\n")

    monkeypatch.setattr(rehydrate, "_build_environment", rebuilt)
    lines = []

    result = rehydrate.rehydrate_run(run, printer=lines.append)

    assert result.used_seed is False
    assert result.exact is True
    assert any("seed" in line and "authorization" in line for line in lines)
    assert not list(run.glob(".env.rehydrate-*"))


@pytest.mark.parametrize("has_env", [True, False])
def test_run_without_manifest_explains_why_instead_of_an_errno(tmp_path, has_env):
    """Runs made before env manifests existed (0.3.16 and older) say so plainly."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "engine-result.json").write_text('{"result": {"result": "SUBMITTED"}}')
    if has_env:
        (run / "env").mkdir()

    with pytest.raises(rehydrate.RehydrateError) as caught:
        rehydrate.rehydrate_run(run, dry_run=True)

    message = str(caught.value)
    assert "Errno" not in message
    assert "no env-manifest.json" in message
    assert ("still has its environment" in message) is has_env
