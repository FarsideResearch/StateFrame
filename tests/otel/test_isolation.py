import json
import signal
import threading
import time
from pathlib import Path

import pytest

from stateframe.otel import authority, companion, config
from tests.otlp_receiver import OTLPReceiver
from tests.otel.support import synthetic_batch


def test_sender_environment_drops_credentials_and_proxy_settings():
    clean = companion.worker_environment({
        "PATH": "/usr/bin",
        "OTEL_EXPORTER_OTLP_HEADERS": "x-auth=private-canary",
        "MODEL_API_KEY": "private-canary",
        "HTTPS_PROXY": "private-proxy",
        "PYTHONPATH": "/private/foreign/code",
    })
    assert set(clean) == {"PATH", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONSAFEPATH"}
    assert clean["PYTHONPATH"] != "/private/foreign/code"


def test_application_credential_alias_is_refused():
    assert authority.has_application_alias({
        "MODEL_API_KEY": "private-canary",
        "OTEL_EXPORTER_OTLP_HEADERS": "x-auth=private%2Dcanary",
    })


def test_operator_paths_cannot_be_inside_source_or_tool_grants(private_root):
    run = private_root / "run"
    repo = private_root / "repo"
    grant = private_root / "grant"
    for directory in (run, repo, grant):
        directory.mkdir(mode=0o700)
    (run / "spec.json").write_text(json.dumps({"repo": str(repo)}))
    for directory in (run, repo, grant, private_root):
        path = directory / "operator.toml"
        path.touch(mode=0o600)
        assert authority.config_path_allowed(path, run, (grant,)) == (directory == private_root)


def test_disposable_sender_is_reaped_after_acknowledgement():
    with OTLPReceiver() as receiver:
        batch = synthetic_batch()
        settings = config.resolve(config.Settings(protocol="http/json"), {
            "OTEL_EXPORTER_OTLP_ENDPOINT": receiver.url,
        }).traces
        sender = companion._spawn_sender(batch, b"d" * 16, settings, time.monotonic_ns() + 5_000_000_000)
        assert sender is not None
        try:
            assert sender.child.wait(timeout=5) == 0
            done, result = companion._poll_sender(sender)
            assert done and result.accepted == 1
        finally:
            companion._close_sender(sender)
            companion._reap_owned_child(sender.child)
        assert sender.child.returncode == 0
        assert len(receiver.requests) == 1


def test_sender_setup_failure_closes_lease_and_child_exits(monkeypatch):
    children = []
    original_popen = companion.subprocess.Popen

    def capture_child(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    def fail_frame(*_args, **_kwargs):
        raise OSError("synthetic setup failure")

    settings = config.resolve(config.Settings(), {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
    }).traces
    assert settings is not None
    monkeypatch.setattr(companion.subprocess, "Popen", capture_child)
    monkeypatch.setattr(companion, "_ipc_frame", fail_frame)

    sender = companion._spawn_sender(
        synthetic_batch(), b"d" * 16, settings, time.monotonic_ns() + 5_000_000_000,
    )

    assert sender is None
    assert len(children) == 1
    assert children[0].wait(timeout=5) is not None


def test_export_kills_sender_that_ignores_its_lease(monkeypatch, tmp_path):
    executable = tmp_path / "stalled-python"
    executable.write_text("#!/bin/sh\nexec /bin/sleep 60\n", encoding="ascii")
    executable.chmod(0o700)
    monkeypatch.setattr(companion.sys, "executable", str(executable))

    children = []
    original_popen = companion.subprocess.Popen

    def capture_child(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(companion.subprocess, "Popen", capture_child)
    settings = config.resolve(config.Settings(timeout_ms=100), {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
    }).traces
    assert settings is not None
    result = []
    worker = threading.Thread(
        target=lambda: result.append(companion._send(synthetic_batch(), b"d" * 16, settings)),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=3)
    returned_before_test_cleanup = not worker.is_alive()
    child = children[0] if children else None
    if child is not None and child.poll() is None:
        from stateframe.engine.kernel_hook_driver import _stop_group

        _stop_group(child)
        child.wait()
    worker.join(timeout=1)

    assert returned_before_test_cleanup
    assert child is not None and child.returncode == -signal.SIGKILL
    assert result and result[0].kind.value == "unknown"


@pytest.mark.skipif(not Path("/proc/self/status").is_file(), reason="requires Linux signal disposition view")
def test_spawned_sender_resets_an_inherited_ignored_sigint():
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    sender = None
    try:
        with OTLPReceiver(["hang_before_headers"]) as receiver:
            settings = config.resolve(config.Settings(timeout_ms=30_000, protocol="http/json"), {
                "OTEL_EXPORTER_OTLP_ENDPOINT": receiver.url,
            }).traces
            assert settings is not None
            sender = companion._spawn_sender(
                synthetic_batch(), b"d" * 16, settings, time.monotonic_ns() + 30_000_000_000,
            )
            assert sender is not None
            assert receiver.wait_for_scripted_mode(timeout=2)
            status = Path(f"/proc/{sender.child.pid}/status").read_text(encoding="ascii")
            ignored = next(line.split()[1] for line in status.splitlines() if line.startswith("SigIgn:"))
            assert int(ignored, 16) & (1 << (signal.SIGINT - 1)) == 0
    finally:
        if sender is not None:
            companion._close_sender(sender)
            companion._reap_owned_child(sender.child)
        signal.signal(signal.SIGINT, previous)


def test_state_root_is_unavailable_without_safe_home(monkeypatch):
    monkeypatch.setattr(companion, "safe_home", lambda: None)

    assert companion._state_root(create=True) is None
