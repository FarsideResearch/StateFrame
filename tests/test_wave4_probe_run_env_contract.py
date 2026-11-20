"""A probe receipt leaves run environment verification with the provisioner."""

import importlib
import json
import sys
from pathlib import Path

import pytest

from stateframe import runner
from stateframe.engine import kernel_run_env
from stateframe.engine.kernel_request_auth_identity import protected_home_identity
from test_run_env_local import command, wheel


_VERSION_PROBE = "import sys, json; print(json.dumps(list(sys.version_info[:3])))"


@pytest.fixture
def source_python(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_ENV_SEED", "auto")
    source = tmp_path / "source"
    command(sys.executable, "-m", "venv", str(source))
    return source / "bin" / "python"


def _probe_cache(monkeypatch):
    name = "stateframe.send1_probe_cache"
    try:
        cache = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        pytest.skip("the optional interpreter probe cache is absent")
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "on")
    protected_home_identity(initialize=True)
    return cache


def _receipt_key(cache, python):
    return cache.engine_fingerprint(str(python), _VERSION_PROBE, runner._startup_probe_seconds())


def _publish_receipt(cache, python):
    key = _receipt_key(cache, python)
    if key is None:
        pytest.skip("the host interpreter or sandbox is ineligible for a verified probe receipt")
    assert runner._check_python(str(python), require_pytest=False) == str(python)
    assert cache.lookup(cache.ENGINE_VERSION, key) == {"version": list(sys.version_info[:3])}
    return key


def _seed(run_dir):
    return Path(json.loads((run_dir / "env-seed.json").read_text())["path"])


@pytest.mark.host_sandbox
@pytest.mark.parametrize("receipt_available", [False, True], ids=["unavailable", "positive"])
def test_seed_consumer_receipt_is_advisory(tmp_path, monkeypatch, source_python, receipt_available):
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "off")
    cache = _probe_cache(monkeypatch) if receipt_available else None
    source = source_python.parent.parent
    original_wheel = wheel(tmp_path, "seed_demo")
    command(str(source_python), "-m", "pip", "install", "--no-index", "--no-deps", str(original_wheel))
    package = next(source.glob("lib/python*/site-packages/seed_demo/__init__.py"))
    relative_package = package.relative_to(source)
    if cache is not None:
        receipt_key = _publish_receipt(cache, source_python)

    provision = kernel_run_env._run
    pip_checks = []

    def run(argv, **kwargs):
        if argv[-3:] == ["-m", "pip", "--version"]:
            pip_checks.append(Path(argv[0]))
        return provision(argv, **kwargs)

    monkeypatch.setattr(kernel_run_env, "_run", run)
    first_dir = tmp_path / "first"
    first = kernel_run_env.create_run_env(first_dir, source_python)
    seed = _seed(first_dir)

    # Replace the cached wheel payload without writing through a clone's links.
    damaged = seed / relative_package
    replacement = damaged.with_suffix(".damaged")
    replacement.write_text("damaged seed\n")
    replacement.replace(damaged)
    if cache is not None:
        assert _receipt_key(cache, source_python) == receipt_key
        assert cache.lookup(cache.ENGINE_VERSION, receipt_key) is not None
    second_dir = tmp_path / "second"
    second = kernel_run_env.create_run_env(second_dir, source_python)
    assert (second.root / relative_package).read_bytes() == package.read_bytes()
    assert _seed(second_dir) == seed

    first_script = first.bin_dir / "pip"
    second_script = second.bin_dir / "pip"
    second_script_before = second_script.read_bytes()
    first_script.write_text("modified in the first run\n")
    assert second_script.read_bytes() == second_script_before
    assert (source / "bin" / "pip").read_bytes() != first_script.read_bytes()

    added_wheel = wheel(tmp_path, "seed_added")
    command(str(source_python), "-m", "pip", "install", "--no-index", "--no-deps", str(added_wheel))
    third_dir = tmp_path / "third"
    third = kernel_run_env.create_run_env(third_dir, source_python)
    assert _seed(third_dir) != seed
    added = next(source.glob("lib/python*/site-packages/seed_added/__init__.py")).relative_to(source)
    assert (third.root / added).is_file()
    assert not (second.root / added).exists()

    next(source.glob("lib/python*/site-packages/pip/__main__.py")).write_text("raise SystemExit(23)\n")
    rejected = tmp_path / "rejected"
    with pytest.raises(kernel_run_env.RunEnvError, match=r"command failed \(23,"):
        kernel_run_env.create_run_env(rejected, source_python)
    assert not (rejected / "env").exists()
    assert pip_checks == [first.python, second.python, third.python, rejected / "env" / "bin" / "python"]


@pytest.mark.host_sandbox
def test_seed_consumer_interpreter_change_misses(tmp_path, monkeypatch, source_python):
    cache = _probe_cache(monkeypatch)
    receipt_key = _publish_receipt(cache, source_python)
    before = cache.interpreter_fingerprint(source_python)
    assert before is not None
    assert before != cache.interpreter_fingerprint(sys.executable)
    engine_key = _receipt_key(cache, sys.executable)
    assert engine_key is not None
    assert cache.lookup(cache.ENGINE_VERSION, engine_key) is None

    alternate = source_python.with_name("python-alternate")
    alternate.symlink_to(source_python.readlink())
    source_python.unlink()
    source_python.symlink_to(alternate.name)
    after = cache.interpreter_fingerprint(source_python)
    assert after is not None
    assert after != before
    changed_key = _receipt_key(cache, source_python)
    assert changed_key is not None
    assert cache.lookup(cache.ENGINE_VERSION, changed_key) is None
    assert cache.lookup(cache.ENGINE_VERSION, receipt_key) is not None

    run = kernel_run_env.create_run_env(tmp_path / "after", source_python)
    assert run.python.readlink() == Path(alternate.name)
    assert command(str(run.python), "-I", "-B", "-c", "import sys; print(sys.prefix)") == str(run.root)
