from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import pickle
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import kernel_checkpoint
import kernel_request_auth_identity
import kernel_request_diet
import pytest
from kernel_frame_allocation_capture import final_request_tokens
from kernel_request_diet import RequestTokenMeter


# Use the exact module that supplied the function imported by kernel_request_diet.
# A checkout can expose the same source under both its engine path and package path.
assert kernel_request_auth_identity is sys.modules[
    kernel_request_diet._auth_binding_for_key.__module__
]
assert (kernel_request_auth_identity.__dict__
        is kernel_request_diet._auth_binding_for_key.__globals__)
assert (kernel_request_auth_identity.auth_binding_for_key
        is kernel_request_diet._auth_binding_for_key)


BODY = {
    "model": "test/model",
    "messages": [{"role": "user", "content": "probe"}],
    "tools": [],
}
GENERATION_ENDPOINT = "https://provider.example/v1/chat/completions"


def _clear_auth_binding_cache() -> None:
    with kernel_request_auth_identity._AUTH_BINDING_SECRET_LOCK:
        kernel_request_auth_identity._AUTH_BINDING_SECRET_CACHE.clear()
        kernel_request_auth_identity._AUTH_BINDING_FALLBACK_SECRET = None


@pytest.fixture(autouse=True)
def _isolated_auth_binding_cache():
    _clear_auth_binding_cache()
    try:
        yield
    finally:
        _clear_auth_binding_cache()


def _set_isolated_home(monkeypatch, tmp_path: Path, name: str = "install") -> Path:
    home = tmp_path / f"{name}-home"
    state_home = tmp_path / f"{name}-state"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("STATEFRAME_HOME", str(state_home))
    return state_home


def _journaled_identity(auth_key: str):
    meter = RequestTokenMeter(disabled=True)
    meter.bind_tokenizer_auth(auth_key, GENERATION_ENDPOINT)
    tokens, method = meter.measure(BODY)
    return meter, tokens, method, meter.journal_fields(method)["tokenizer_identity"]


def _carried_measurement(auth_key: str, generation_endpoint: str):
    body = dict(BODY)
    meter = RequestTokenMeter(disabled=True)
    meter.bind_tokenizer_auth(auth_key, generation_endpoint)
    tokens, method = meter.measure(body)
    carried = meter.bind_measurement(
        body, tokens=tokens, method=method, request_id="r",
        max_request_bytes=100_000, max_prompt_tokens=33_344,
    )
    capture = SimpleNamespace(request_id="r", _frame_candidate_measurement=carried)
    return body, meter, capture, (tokens, method)


def test_carried_measurement_invalidates_after_auth_route_change(tmp_path, monkeypatch):
    _set_isolated_home(monkeypatch, tmp_path)
    body = dict(BODY)
    meter = RequestTokenMeter(disabled=True)
    meter.bind_tokenizer_auth("tenant-A", "https://provider-a.example/v1/chat/completions")
    tokens, method = meter.measure(body)
    carried = meter.bind_measurement(
        body, tokens=tokens, method=method, request_id="r",
        max_request_bytes=100_000, max_prompt_tokens=33_344,
    )
    meter.bind_tokenizer_auth("tenant-B", "https://provider-b.example/v1/chat/completions")
    meter.measure = lambda _body: (999, "calibrated_bytes")
    capture = SimpleNamespace(request_id="r", _frame_candidate_measurement=carried)
    assert final_request_tokens(body, meter, capture) == (999, "calibrated_bytes")


def test_carried_measurement_invalidates_after_auth_only_change(tmp_path, monkeypatch):
    _set_isolated_home(monkeypatch, tmp_path)
    body, meter, capture, _original = _carried_measurement(
        "tenant-A", GENERATION_ENDPOINT,
    )
    meter.bind_tokenizer_auth("tenant-B", GENERATION_ENDPOINT)
    meter.measure = lambda _body: (999, "calibrated_bytes")

    assert final_request_tokens(body, meter, capture) == (999, "calibrated_bytes")
    assert "tenant-B" not in meter.tokenizer_identity(body)


def test_carried_measurement_invalidates_after_generation_endpoint_only_change(
        tmp_path, monkeypatch):
    _set_isolated_home(monkeypatch, tmp_path)
    body, meter, capture, _original = _carried_measurement(
        "tenant-A", "https://provider-a.example/v1/chat/completions",
    )
    meter.bind_tokenizer_auth(
        "tenant-A", "https://provider-b.example/v1/chat/completions",
    )
    meter.measure = lambda _body: (999, "calibrated_bytes")

    assert final_request_tokens(body, meter, capture) == (999, "calibrated_bytes")


def test_carried_measurement_reuses_unchanged_auth_and_generation_endpoint(
        tmp_path, monkeypatch):
    _set_isolated_home(monkeypatch, tmp_path)
    body, meter, capture, original = _carried_measurement(
        "tenant-A", "https://provider-a.example/v1/chat/completions",
    )
    meter.measure = lambda _body: (999, "calibrated_bytes")

    assert final_request_tokens(body, meter, capture) == original


@pytest.mark.parametrize("auth_key", ["sk-no-key", "EMPTY"])
def test_auth_key_and_binding_secret_never_reach_run_artifacts(
        auth_key, tmp_path, monkeypatch):
    state_home = _set_isolated_home(monkeypatch, tmp_path)
    meter, tokens, method, tokenizer_identity = _journaled_identity(auth_key)
    measurement = meter.bind_measurement(
        BODY, tokens=tokens, method=method, request_id="run-1",
        max_request_bytes=100_000, max_prompt_tokens=33_344,
    )

    # Persist the same identity-bearing surfaces a run writes: its journal,
    # measurement receipt, cache entry and checkpoint state. identity.key is
    # the one intentional secret-bearing file, so it is checked separately.
    run = tmp_path / "run"
    journal = run / "worker" / "measurement-runs" / "one" / "events.jsonl"
    receipt = run / "receipts" / "request.json"
    cache = run / "caches" / "request-measurement.json"
    checkpoint = run / "checkpoints" / "request-meter.pkl"
    for path in (journal, receipt, cache, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)

    journal.write_text(json.dumps({"event": "request_measurement",
                                   **meter.journal_fields(method)}) + "\n")
    receipt.write_text(json.dumps(measurement.receipt_fields(), sort_keys=True))
    cache.write_text(json.dumps({"tokenizer_identity": tokenizer_identity,
                                 "method": method, "tokens": tokens}))

    meter_state = kernel_checkpoint._runtime_state(
        meter,
        kernel_checkpoint.DROP_ATTRS[("kernel_request_diet", "RequestTokenMeter")],
    )
    module_state = kernel_checkpoint._module_state(
        "kernel_request_auth_identity", kernel_request_auth_identity,
    )
    assert "_AUTH_BINDING_SECRET_CACHE" not in module_state
    assert "_AUTH_BINDING_FALLBACK_SECRET" not in module_state
    assert "_AUTH_BINDING_SECRET_LOCK" not in module_state
    checkpoint.write_bytes(pickle.dumps({"meter": meter_state,
                                         "module": module_state}))

    key_path = state_home / "identity.key"
    secret = key_path.read_bytes()
    assert len(secret) == 32
    markers = (
        auth_key.encode("utf-8"),
        hashlib.sha256(auth_key.encode("utf-8")).hexdigest().encode("ascii"),
        secret,
        secret.hex().encode("ascii"),
    )
    artifact_files = [
        path for path in tmp_path.rglob("*")
        if path.is_file() and path != key_path
    ]
    artifact_realpaths = {path.resolve() for path in artifact_files}
    assert {journal.resolve(), receipt.resolve(), cache.resolve(), checkpoint.resolve()}.issubset(
        artifact_realpaths
    )
    for path in artifact_files:
        if path.resolve() == key_path.resolve():
            continue
        data = path.read_bytes()
        for marker in markers:
            assert marker not in data, f"sensitive marker found in {path.relative_to(tmp_path)}"


def test_identity_key_mode_is_exact_under_zero_umask(tmp_path, monkeypatch):
    state_home = _set_isolated_home(monkeypatch, tmp_path)
    previous_umask = os.umask(0)
    try:
        _journaled_identity("mode-test-key")
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((state_home / "identity.key").stat().st_mode) == 0o600


def test_different_installs_journal_different_identity_for_same_key(
        tmp_path, monkeypatch):
    first_home = _set_isolated_home(monkeypatch, tmp_path, "first")
    first_identity = _journaled_identity("same-low-entropy-key")[3]
    first_secret = (first_home / "identity.key").read_bytes()

    second_home = _set_isolated_home(monkeypatch, tmp_path, "second")
    second_identity = _journaled_identity("same-low-entropy-key")[3]
    second_secret = (second_home / "identity.key").read_bytes()

    assert first_secret != second_secret
    assert first_identity != second_identity


def _spawn_auth_binding_worker(state_home: str, barrier, results) -> None:
    if os.environ.get("STATEFRAME_HOME") != state_home:
        raise RuntimeError("spawned worker did not inherit the isolated StateFrame home")
    barrier.wait(timeout=30)
    binding = kernel_request_auth_identity.auth_binding_for_key("concurrent-first-use")
    meter = RequestTokenMeter(disabled=True)
    meter.bind_tokenizer_auth("concurrent-first-use", GENERATION_ENDPOINT)
    _, method = meter.measure(BODY)
    results.put((binding, meter.journal_fields(method)["tokenizer_identity"]))


def test_concurrent_first_use_publishes_one_private_key(tmp_path, monkeypatch):
    state_home = _set_isolated_home(monkeypatch, tmp_path, "concurrent")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(target=_spawn_auth_binding_worker,
                        args=(str(state_home), barrier, results))
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        bindings = [results.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)

        assert all(process.exitcode == 0 for process in processes)
        assert bindings[0][0] == bindings[1][0]
        assert bindings[0][1] == bindings[1][1]
        key_path = state_home / "identity.key"
        assert key_path.is_file()
        assert len(key_path.read_bytes()) == 32
        assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
        assert list(state_home.glob("identity.key")) == [key_path]
        assert not list(state_home.glob(".identity.key.*.tmp"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()
        results.join_thread()


@pytest.mark.parametrize(
    "failure",
    ["invalid-length", "invalid-mode", "symlink", "wrong-owner", "unreadable"],
)
def test_invalid_or_unreadable_identity_key_falls_back_and_remeasures(
        failure, tmp_path, monkeypatch):
    valid_home = _set_isolated_home(monkeypatch, tmp_path, "valid")
    valid_identity = _journaled_identity("same-low-entropy-key")[3]

    fallback_home = _set_isolated_home(monkeypatch, tmp_path, failure)
    key_path = fallback_home / "identity.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    contents = (b"not-a-32-byte-installation-key"
                if failure == "invalid-length" else b"u" * 32)
    if failure == "symlink":
        target = fallback_home / "target.key"
        target.write_bytes(contents)
        target.chmod(0o600)
        key_path.symlink_to(target.name)
    else:
        key_path.write_bytes(contents)
        key_path.chmod(0o644 if failure == "invalid-mode" else 0o600)

    if failure == "unreadable":
        real_open = os.open

        def deny_identity_read(path, flags, *args, **kwargs):
            if (os.fspath(path) == os.fspath(key_path)
                    and flags & os.O_WRONLY == 0):
                raise PermissionError("identity key is unreadable")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", deny_identity_read)
    elif failure == "wrong-owner":
        current_uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: current_uid + 1)

    fallback_identity = _journaled_identity("same-low-entropy-key")[3]

    assert fallback_identity != valid_identity
    assert key_path.is_file()
    if failure == "symlink":
        assert key_path.is_symlink()
        assert (fallback_home / "target.key").read_bytes() == contents
    else:
        assert key_path.read_bytes() == contents
    assert (valid_home / "identity.key").is_file()
