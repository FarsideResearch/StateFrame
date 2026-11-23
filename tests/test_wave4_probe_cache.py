"""Cross-process startup receipts and the live fallbacks that bound their use."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from _child_env import child_env

from stateframe import runner, send1_probe_cache as cache
from stateframe.startup_cache_writes import writer_scope
from stateframe.engine import kernel_request_auth_identity, kernel_safe_probe


@pytest.fixture
def install(tmp_path, monkeypatch):
    home = tmp_path / "install"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv(cache.SWITCH, "on")
    kernel_request_auth_identity.protected_home_identity(home, initialize=True)
    return home


def _entries(home):
    return list((home / cache._DIRECTORY).glob("*.json"))


def test_recorded_child_selection_reaches_transport_worker(monkeypatch):
    monkeypatch.delenv(cache.SWITCH, raising=False)
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert not worker.submit(cache.enabled).result()
        with cache.child_scope():
            assert worker.submit(cache.enabled).result()
        assert not worker.submit(cache.enabled).result()


def test_store_checks_full_key_schema_time_and_integrity(install, monkeypatch):
    monkeypatch.setattr(cache, "_SLOTS", 1)
    version = {"version": [3, 12, 0]}
    assert not cache.publish(cache.ENGINE_VERSION, "a", {"version": [3, True, 0]})
    assert not cache.publish(cache.ENGINE_VERSION, "a", {"version": [3, 12, 0], "output": "log"})
    assert cache.publish(cache.ENGINE_VERSION, "a", version)
    assert cache.lookup(cache.ENGINE_VERSION, "a") == version
    assert cache.lookup(cache.ENGINE_VERSION, "b") is None
    assert cache.lookup(cache.ENGINE_VERSION, "a", max_age=0) is None
    clock = cache.time.monotonic()
    with monkeypatch.context() as patch:
        patch.setattr(cache.time, "monotonic", lambda: clock + 30)
        assert cache.lookup(cache.ENGINE_VERSION, "a", max_age=1) is None
    boot = cache._boot_id()
    with monkeypatch.context() as patch:
        patch.setattr(cache, "_boot_id", lambda: boot + "changed")
        assert cache.lookup(cache.ENGINE_VERSION, "a") is None
    assert cache.publish(cache.ENGINE_VERSION, "b", version)
    assert cache.lookup(cache.ENGINE_VERSION, "a") is None
    path, = _entries(install)
    original = path.read_bytes()
    path.write_bytes(original.replace(b"[3,12,0]", b"[3,13,0]"))
    assert cache.lookup(cache.ENGINE_VERSION, "b") is None
    path.write_bytes(original + b"x" * cache._MAX_ENTRY)
    assert cache.lookup(cache.ENGINE_VERSION, "b") is None
    path.unlink()
    path.symlink_to(install / "identity.key")
    assert cache.lookup(cache.ENGINE_VERSION, "b") is None
    assert cache.publish(cache.ENGINE_VERSION, "b", version)
    assert not path.is_symlink()
    assert cache.lookup(cache.ENGINE_VERSION, "b") == version


def test_installation_binding_and_private_modes(install, tmp_path, monkeypatch):
    facts = {"version": [3, 12, 0]}
    identity = kernel_request_auth_identity.protected_home_identity(install)
    other = tmp_path / "other"
    other.mkdir(mode=0o700)
    kernel_request_auth_identity.protected_home_identity(other, initialize=True)
    assert cache.publish(cache.ENGINE_VERSION, "sample", facts)
    entry, = _entries(install)
    destination = other / cache._DIRECTORY
    destination.mkdir(mode=0o700)
    (destination / entry.name).write_bytes(entry.read_bytes())
    (destination / entry.name).chmod(0o600)
    monkeypatch.setenv("STATEFRAME_HOME", str(other))
    assert cache.lookup(cache.ENGINE_VERSION, "sample") is None
    with cache.launch_scope(True, identity):
        assert cache.lookup(cache.ENGINE_VERSION, "sample") == facts
        for path, mode in ((entry, 0o644), (entry.parent, 0o755), (install, 0o755)):
            original = path.stat().st_mode & 0o777
            path.chmod(mode)
            assert cache.lookup(cache.ENGINE_VERSION, "sample") is None
            path.chmod(original)


def test_cross_process_publishers(install):
    code = """
from stateframe import send1_probe_cache as cache
import sys
assert cache.publish(cache.ENGINE_VERSION, "race", {"version": [3, 12, int(sys.argv[1])]})
"""
    writers = [subprocess.Popen([sys.executable, "-c", code, str(patch)], env=child_env()) for patch in (0, 1)]
    assert [process.wait(timeout=20) for process in writers] == [0, 0]
    assert cache.lookup(cache.ENGINE_VERSION, "race") in ({"version": [3, 12, 0]},
                                                         {"version": [3, 12, 1]})
    assert sorted((install / cache._DIRECTORY).iterdir()) == _entries(install)


@pytest.mark.parametrize("changed", ["key", "directory"])
def test_open_cache_reader_refuses_replaced_installation_authority(install, changed):
    facts = {"version": [3, 12, 0]}
    assert cache.publish(cache.ENGINE_VERSION, "sample", facts)
    identity = kernel_request_auth_identity.protected_home_identity(install)
    with cache.launch_scope(True, identity), writer_scope(lambda *_args, **_kwargs: None):
        assert cache.lookup(cache.ENGINE_VERSION, "sample") == facts
        if changed == "key":
            replacement = install / "replacement"
            replacement.write_bytes(b"x" * 32)
            replacement.chmod(0o600)
            replacement.replace(install / "identity.key")
        else:
            directory = install / cache._DIRECTORY
            directory.rename(install / "old-receipts")
            directory.mkdir(mode=0o700)
        assert cache.lookup(cache.ENGINE_VERSION, "sample") is None


@pytest.mark.parametrize("retain_reader", [False, True])
def test_cache_rejects_key_replacement_during_receipt_validation(install, monkeypatch, retain_reader):
    facts = {"version": [3, 12, 0]}
    assert cache.publish(cache.ENGINE_VERSION, "sample", facts)
    boot = cache._boot_id()
    replacement = install / "replacement"
    replacement.write_bytes(b"x" * 32)
    replacement.chmod(0o600)

    def boot_after_key_rotation():
        replacement.replace(install / "identity.key")
        return boot

    reader_scope = writer_scope(lambda *_args, **_kwargs: None) if retain_reader else nullcontext()
    with reader_scope:
        assert cache.lookup(cache.ENGINE_VERSION, "sample") == facts
        monkeypatch.setattr(cache, "_boot_id", boot_after_key_rotation)
        assert cache.lookup(cache.ENGINE_VERSION, "sample") is None
    assert not replacement.exists()


def test_off_and_replay_keep_the_probe_cache_closed(install, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("the cache touched storage")

    monkeypatch.setattr(cache, "_store", unexpected)
    with cache.launch_scope(False, None):
        assert not cache.enabled()
        assert cache.lookup(cache.ENGINE_VERSION, "a") is None
        assert not cache.publish(cache.ENGINE_VERSION, "a", {"version": [3, 12, 0]})
    options = runner.RunOptions(repo=install.parent, endpoint=None, task="test")
    options._send1_wave4 = True
    assert runner._provider_probe(options, cache.enabled) is True
    options._resuming = True
    assert runner._provider_probe(options, cache.enabled) is False


@pytest.mark.parametrize("scan", ["site-packages", "sitecustomize"])
def test_failed_second_dependency_scan_invalidates_interpreter_fingerprint(tmp_path, monkeypatch, scan):
    # Path.iterdir lists through os.scandir or os.listdir depending on the Python version.
    real = {"scandir": os.scandir, "listdir": os.listdir}
    attempted = []

    def refuse(name):
        def listing(path):
            if path in {tmp_path, str(tmp_path)}:
                attempted.append(path)
                raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path)
            return real[name](path)

        return listing

    def interpreter_dependencies(_python):
        dependencies = SimpleNamespace(inputs={"prior-tree-scan": "complete"})
        if scan == "site-packages":
            cache._site_dependencies(tmp_path, dependencies)
        else:
            cache._no_sitecustomize(tmp_path)
        return dependencies

    for name in real:
        monkeypatch.setattr(os, name, refuse(name))
    monkeypatch.setattr(cache, "sys", SimpleNamespace(
        platform="linux", implementation=SimpleNamespace(name="cpython"), version_info=sys.version_info))
    monkeypatch.setattr(cache, "host_fingerprint", lambda: "host")
    monkeypatch.setattr(cache, "_interpreter_dependencies", interpreter_dependencies)
    for name in tuple(os.environ):
        if name.startswith("LD_"):
            monkeypatch.delenv(name)
    assert cache.interpreter_fingerprint(sys.executable) is None
    assert len(attempted) == 1


def test_hit_rechecks_version_and_loses_stale_sandbox_authority(install, monkeypatch):
    monkeypatch.setattr(cache, "engine_fingerprint", lambda *_args: "stable")
    monkeypatch.setattr(cache, "check_sandbox", lambda *_args: True)
    assert cache.publish(cache.ENGINE_VERSION, "stable", {"version": [3, 14, 1]})
    with pytest.raises(runner.RunError, match="3.14.1 is not supported"):
        runner._check_python(sys.executable, require_pytest=False)

    monkeypatch.setattr(cache, "check_sandbox", lambda *_args: False)
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise runner.RunError("current sandbox refused")

    monkeypatch.setattr(runner, "_startup_probe", refuse)
    for choice in ("on", "off"):
        monkeypatch.setenv(cache.SWITCH, choice)
        with pytest.raises(runner.RunError, match="current sandbox refused"):
            runner._check_python(sys.executable, require_pytest=False)
    assert len(calls) == 2


def test_interpreter_change_during_sandbox_check_requires_live_version(install, monkeypatch):
    fingerprint = "original"
    assert cache.publish(cache.ENGINE_VERSION, fingerprint, {"version": [3, 12, 0]})
    monkeypatch.setattr(cache, "engine_fingerprint", lambda *_args: fingerprint)

    def witness(*_args):
        nonlocal fingerprint
        fingerprint = "replaced"
        return True

    monkeypatch.setattr(cache, "check_sandbox", witness)
    monkeypatch.setattr(runner, "_startup_probe", lambda *_args, **_kwargs: b"[3,14,1]")
    with pytest.raises(runner.RunError, match="3.14.1 is not supported"):
        runner._check_python(sys.executable, require_pytest=False)


def test_retried_probe_is_live_on_next_launch(install, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(cache, "engine_fingerprint", lambda *_args: "stable")
    calls = []

    def probe(*_args, **_kwargs):
        calls.append(None)
        return SimpleNamespace(exit_code=0, output_truncated=False, timed_out=False,
                               output=b"[3, 12, 0]", probe_failures=("retried",))

    monkeypatch.setattr(kernel_safe_probe, "run_probe", probe)
    for _ in range(2):
        assert runner._check_python(sys.executable, require_pytest=False) == sys.executable
    assert len(calls) == 2
    assert not _entries(install)


_CHECK = """
import json
import os
import sys
from stateframe import runner, send1_probe_cache as cache
from stateframe.engine import kernel_safe_probe

calls = {"probe": 0, "sandbox": 0}
real_probe = kernel_safe_probe.run_probe
real_witness = cache.check_sandbox

def count(probe, key):
    def wrapped(*args, **kwargs):
        calls[key] += 1
        return probe(*args, **kwargs)
    return wrapped

kernel_safe_probe.run_probe = count(real_probe, "probe")
cache.check_sandbox = count(real_witness, "sandbox")
try:
    selected = runner._check_python(os.environ["PROBE_PYTHON"], require_pytest=False)
except runner.RunError as error:
    result = {"error": str(error)}
else:
    result = {"python": selected, "version": list(sys.version_info[:3])}
print(json.dumps({"calls": calls, "result": result}))
"""


def test_separate_launches_reuse_success_and_notice_interpreter_mutations(install, tmp_path):
    if sys.platform != "linux":
        pytest.skip("persistent interpreter startup currently uses Linux stamps and namespaces")
    base_python = Path(sys.executable).resolve()
    prefix = tmp_path / "environment"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.symlink_to(base_python)
    config = prefix / "pyvenv.cfg"
    config.write_text(f"home = {base_python.parent}\ninclude-system-site-packages = false\n")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = prefix / "lib" / version / "site-packages"
    site.mkdir(parents=True)
    environment = dict(child_env(), PROBE_PYTHON=str(python))

    def launch(choice="on", *, launcher=python, path=None):
        env = dict(environment, **{cache.SWITCH: choice})
        if path is not None:
            env["PATH"] = path
        result = subprocess.check_output([str(launcher), "-B", "-S", "-c", _CHECK], env=env, timeout=90)
        return json.loads(result)

    first = launch()
    assert first["calls"]["probe"] == 1
    if not _entries(install):
        pytest.skip("this host did not establish the interpreter or sandbox provenance")
    second = launch()
    assert second == {"calls": {"probe": 0, "sandbox": 1}, "result": first["result"]}
    assert launch("off") == first

    entry, = _entries(install)
    entry.write_text("corrupted receipt")
    assert launch() == first
    assert launch() == second

    config.write_text(config.read_text() + "test = changed\n")
    assert launch() == first
    assert launch() == second

    library = site / "sitecustomize.py"
    library.write_text("print('startup changed')\n")
    changed = launch()
    assert changed == launch("off")
    assert "invalid Python version" in changed["result"]["error"]
    library.unlink()
    assert launch() == first

    alternate = prefix / "bin" / "alternate"
    alternate.symlink_to(base_python)
    python.unlink()
    python.symlink_to(alternate)
    assert launch() == first
    assert launch() == second

    sandbox_path = tmp_path / "sandbox"
    sandbox_path.mkdir()
    sandbox = sandbox_path / "bwrap"
    sandbox.symlink_to("/usr/bin/bwrap")
    path = str(sandbox_path) + os.pathsep + os.environ["PATH"]
    assert launch(path=path) == first
    sandbox.unlink()
    sandbox.write_text("#!/bin/sh\nexit 3\n")
    sandbox.chmod(0o700)
    on, off = launch(path=path), launch("off", path=path)
    assert on["calls"] == off["calls"] == {"probe": 1, "sandbox": 0}
    def sandbox_refusal(result):
        first_line = result["error"].splitlines()[0]
        assert "startup probe could not run safely" in first_line
        # How the broken launcher closes the channel can change the socket errno.
        return first_line.partition("execution supervisor communication failed:")[0]

    assert sandbox_refusal(on["result"]) == sandbox_refusal(off["result"])

    python.unlink()
    python.write_text("#!/bin/sh\nexit 7\n")
    python.chmod(0o700)
    assert launch(launcher=alternate) == launch("off", launcher=alternate)
    python.chmod(0o600)
    denied = launch(launcher=alternate)
    assert denied == launch("off", launcher=alternate)
    assert denied["calls"]["probe"] == 0
    assert "not executable" in denied["result"]["error"]
