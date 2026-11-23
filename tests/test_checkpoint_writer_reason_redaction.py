"""Checkpoint writer reasons stay safe across the child-to-parent boundary."""

from __future__ import annotations

import contextvars
import errno
import io
import json
import os
import pickle
import signal
import time
from pathlib import Path

import pytest

import kernel_checkpoint


pytestmark = pytest.mark.host_sandbox

_KEY = 'ckpt-writer\\key"\'雪-credential'
_SECRET_GUARD_REASON = "checkpoint refused: registered secret in checkpoint"


def _secret_forms(value: str) -> set[str]:
    return {
        value,
        repr(value),
        json.dumps(value, ensure_ascii=True),
        json.dumps(value, ensure_ascii=False),
        value.encode("unicode_escape").decode("ascii"),
        str(KeyError(value)),
        repr(KeyError(value)),
    }


def _assert_secret_free(text: object, secret: str) -> None:
    if not isinstance(text, str):
        raise AssertionError("checkpoint writer reason is missing or not text")
    if any(form and form in text for form in _secret_forms(secret)):
        raise AssertionError("checkpoint writer diagnostic contains a registered secret")


def _parent_reason(run: Path, step: int, identity: str, writer_pid: int) -> str:
    journal = run / "checkpoints" / "events.jsonl"
    try:
        with journal.open("rb") as stream:
            info = os.fstat(stream.fileno())
    except FileNotFoundError:
        return kernel_checkpoint._writer_failure_reason(
            run, step, identity, writer_pid, 0, None, journal_missing=True,
        )
    except OSError:
        return kernel_checkpoint._writer_failure_reason(run, step, identity, writer_pid, 0, None)
    return kernel_checkpoint._writer_failure_reason(
        run, step, identity, writer_pid, 0, (info.st_dev, info.st_ino),
    )


def _run_writer_failure(
        tmp_path: Path, capsys, secret: str, fail_write,
        expected_reason: str) -> tuple[str, dict[str, object]]:
    previous_secrets = kernel_checkpoint._SECRETS.copy()
    previous_state = kernel_checkpoint._STATE.copy()
    kernel_checkpoint.register_secret(secret)

    run = tmp_path / "run"
    pending = run / "checkpoints" / ".0001-1.tmp"
    pending.mkdir(parents=True)
    kernel_checkpoint._STATE.clear()
    prior_failures = kernel_checkpoint.MAX_CONSECUTIVE_FAILURES - 1
    kernel_checkpoint._STATE.update(
        failed=prior_failures, consecutive_failed=prior_failures, disabled=False,
    )

    try:
        kernel_checkpoint._fork_writer(
            run, 1, run / "checkpoints" / "0001-1", fail_write, pending,
        )
        deadline = time.monotonic() + 10
        while not kernel_checkpoint._reap():
            if time.monotonic() >= deadline:
                child = kernel_checkpoint._STATE.get("child")
                if child:
                    os.kill(child, signal.SIGKILL)
                raise AssertionError("background checkpoint writer was not reaped")
            time.sleep(0.01)

        warning = capsys.readouterr().err
        events = [json.loads(line) for line in
                  (run / "checkpoints" / "events.jsonl").read_text().splitlines()]
        event = next(row for row in reversed(events)
                     if row.get("event") == "checkpoint_failed")
        reason = event.get("failure_reason")

        _assert_secret_free(warning, secret)
        _assert_secret_free(reason, secret)
        _assert_secret_free(event.get("error"), secret)
        if expected_reason not in warning:
            raise AssertionError("parent warning omitted its fixed failure code")
        if reason != expected_reason:
            raise AssertionError("child recorded a non-allowlisted failure reason")
        if event.get("error") != expected_reason:
            raise AssertionError("child event carried exception text instead of the fixed code")
        return warning, event
    finally:
        child = kernel_checkpoint._STATE.get("child")
        if child:
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass
        kernel_checkpoint._SECRETS[:] = previous_secrets
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE.update(previous_state)


def _run_synchronous_writer_failure(
        tmp_path: Path, capsys, monkeypatch, secret: str, fail_save,
        expected_reason: str) -> tuple[str, dict[str, object]]:
    previous_secrets = kernel_checkpoint._SECRETS.copy()
    previous_state = kernel_checkpoint._STATE.copy()
    kernel_checkpoint.register_secret(secret)

    run = tmp_path / "run"
    (run / "wire").mkdir(parents=True)
    (run / "wire" / "0001.request.json").write_text("{}")
    kernel_checkpoint._STATE.clear()
    prior_failures = kernel_checkpoint.MAX_CONSECUTIVE_FAILURES - 1
    kernel_checkpoint._STATE.update(
        failed=prior_failures, consecutive_failed=prior_failures, disabled=False,
        last_sends=None, restored_step=None,
    )
    monkeypatch.setattr(kernel_checkpoint, "mode", lambda: "save")
    monkeypatch.setattr(kernel_checkpoint, "_every", lambda: 1)
    monkeypatch.setattr(kernel_checkpoint, "_async_enabled", lambda: False)
    monkeypatch.setattr(kernel_checkpoint, "_low_disk_space", lambda _run_dir: None)
    monkeypatch.setattr(kernel_checkpoint, "_sends", lambda _run_dir: 1)
    monkeypatch.setattr(kernel_checkpoint, "save", lambda *_args, **_kwargs: fail_save())

    try:
        escaped = False
        try:
            kernel_checkpoint.save_point(object(), 1, run / "checkpoint-target")
        except BaseException:
            escaped = True
        if escaped:
            raise AssertionError("synchronous checkpoint failure escaped its save boundary")

        output = capsys.readouterr()
        events = [json.loads(line) for line in
                  (run / "checkpoints" / "events.jsonl").read_text().splitlines()]
        event = next(row for row in reversed(events)
                     if row.get("event") == "checkpoint_failed")
        reason = event.get("failure_reason")
        child_output = output.out + output.err

        _assert_secret_free(child_output, secret)
        _assert_secret_free(reason, secret)
        _assert_secret_free(event.get("error"), secret)
        if expected_reason not in child_output:
            raise AssertionError("synchronous child warning omitted its fixed failure code")
        if reason != expected_reason or event.get("error") != expected_reason:
            raise AssertionError("synchronous child event carried a non-allowlisted reason")
        if "engine_error" in child_output:
            raise AssertionError("contained checkpoint failure became a child engine error")

        # The host copies child stdout/stderr verbatim into engine.log.
        engine_log = tmp_path / "engine.log"
        engine_log.write_text(child_output)
        _assert_secret_free(engine_log.read_text(), secret)
        return child_output, event
    finally:
        kernel_checkpoint._SECRETS[:] = previous_secrets
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE.update(previous_state)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="async writer requires fork")
@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        ("key_error", "unexpected writer error"),
        ("dynamic_exception", "unexpected writer error"),
        ("os_error_filename", "checkpoint write failed: ENOENT"),
        ("os_error_errno_returns_secret", "checkpoint write failed: os error"),
        ("os_error_errno_raises_secret", "checkpoint write failed: os error"),
        ("os_error_table_secret", "checkpoint write failed: EIO"),
        ("checkpoint_error", "checkpoint refused"),
        ("memory_error", "checkpoint write failed: memory error"),
        ("pickling_error", "checkpoint serialization failed"),
    ],
)
def test_writer_warning_and_child_reason_never_expose_secret(
        tmp_path: Path, capsys, monkeypatch, failure: str, expected_reason: str) -> None:
    if failure == "key_error":
        def fail_write():
            raise KeyError(_KEY)
    elif failure == "dynamic_exception":
        dynamic_error = type(f"WriterFailure_{_KEY}", (RuntimeError,), {})

        def fail_write():
            raise dynamic_error("writer failed")
    elif failure == "checkpoint_error":
        def fail_write():
            raise kernel_checkpoint.CheckpointError(f"refused credential {_KEY}")
    elif failure == "memory_error":
        def fail_write():
            raise MemoryError(_KEY)
    elif failure == "pickling_error":
        def fail_write():
            raise pickle.PicklingError(_KEY)
    elif failure in {"os_error_errno_returns_secret", "os_error_errno_raises_secret"}:
        class SecretErrnoOSError(OSError):
            @property
            def errno(self):
                if failure == "os_error_errno_raises_secret":
                    raise RuntimeError(_KEY)
                return _KEY

        def fail_write():
            raise SecretErrnoOSError(_KEY)
    elif failure == "os_error_table_secret":
        monkeypatch.setattr(errno, "errorcode", {errno.EIO: _KEY})

        def fail_write():
            raise OSError(errno.EIO, "checkpoint writer failed")
    else:
        def fail_write():
            raise OSError(errno.ENOENT, "checkpoint missing", _KEY)

    _run_writer_failure(tmp_path, capsys, _KEY, fail_write, expected_reason)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="async writer requires fork")
def test_secret_guard_records_its_fixed_reason(tmp_path: Path, capsys) -> None:
    def fail_write():
        kernel_checkpoint._StateWriter(io.BytesIO()).write(_KEY.encode("utf-8"))

    _run_writer_failure(tmp_path, capsys, _KEY, fail_write, _SECRET_GUARD_REASON)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="async writer requires fork")
def test_chunk_secret_guard_records_its_fixed_reason(tmp_path: Path, capsys) -> None:
    def fail_write():
        kernel_checkpoint._SAVE.clear()
        kernel_checkpoint._SAVE.update(root=tmp_path / "run", index={})
        kernel_checkpoint._chunk_list("registered-secret", [_KEY])

    _run_writer_failure(tmp_path, capsys, _KEY, fail_write, _SECRET_GUARD_REASON)
    cache = tmp_path / "run" / kernel_checkpoint.CACHE_DIR
    if cache.exists() and any(cache.iterdir()):
        raise AssertionError("chunk secret guard wrote a checkpoint chunk")


def test_parent_ignores_unallowlisted_failure_reason(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    event = {
        "event": "checkpoint_failed",
        "step": 7,
        "background": True,
        "writer_generation": "writer-7",
        "writer_pid": 773,
        "failure_reason": _KEY,
        "error": _KEY,
    }
    (checkpoints / "events.jsonl").write_text(json.dumps(event) + "\n")

    if _parent_reason(run, 7, "writer-7", 773) != "unexpected writer error":
        raise AssertionError("parent accepted a non-allowlisted writer reason")


def test_parent_does_not_reuse_same_step_failure_from_prior_writer(
        tmp_path: Path, capsys) -> None:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    old_event = {
        "event": "checkpoint_failed",
        "step": 7,
        "background": True,
        "writer_generation": "prior-generation",
        "writer_pid": 774,
        "failure_reason": _SECRET_GUARD_REASON,
    }
    (checkpoints / "events.jsonl").write_text(json.dumps(old_event) + "\n")

    previous_state = kernel_checkpoint._STATE.copy()
    kernel_checkpoint._STATE.clear()
    kernel_checkpoint._STATE.update(
        failed=kernel_checkpoint.MAX_CONSECUTIVE_FAILURES - 1,
        consecutive_failed=kernel_checkpoint.MAX_CONSECUTIVE_FAILURES - 1,
        disabled=False,
    )
    try:
        reason = _parent_reason(run, 7, "current-generation", 774)
        kernel_checkpoint._note_failure(run, reason)
        warning = capsys.readouterr().err
        assert reason == "unexpected writer error"
        assert "unexpected writer error" in warning
        assert _SECRET_GUARD_REASON not in warning
    finally:
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE.update(previous_state)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="async writer requires fork")
def test_async_writer_reports_safe_error_type(tmp_path: Path, capsys) -> None:
    def fail_write():
        raise RuntimeError(_KEY)

    warning, event = _run_writer_failure(
        tmp_path, capsys, _KEY, fail_write, "unexpected writer error")

    assert "unexpected writer error: RuntimeError" in warning
    assert event.get("writer_error_type") == "RuntimeError"
    _assert_secret_free(warning, _KEY)
    _assert_secret_free(event.get("writer_error_type"), _KEY)


def test_parent_uses_fixed_reason_when_child_event_is_missing(tmp_path: Path) -> None:
    if _parent_reason(tmp_path / "run", 7, "writer-7", 775) != "unexpected writer error":
        raise AssertionError("parent exposed a dynamic writer exit status")


@pytest.mark.parametrize("errno_behavior", ["returns", "raises"])
def test_synchronous_failure_is_contained_and_redacted(
        tmp_path: Path, capsys, monkeypatch, errno_behavior: str) -> None:
    class SecretErrnoOSError(OSError):
        @property
        def errno(self):
            if errno_behavior == "raises":
                raise RuntimeError(_KEY)
            return _KEY

    def fail_save():
        raise SecretErrnoOSError(_KEY)

    _run_synchronous_writer_failure(
        tmp_path, capsys, monkeypatch, _KEY, fail_save,
        "checkpoint write failed: os error",
    )


def test_parent_rejects_errno_table_injected_reason(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(errno, "errorcode", {errno.EIO: _KEY})
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    event = {
        "event": "checkpoint_failed",
        "step": 7,
        "background": True,
        "writer_generation": "writer-7",
        "writer_pid": 776,
        "failure_reason": f"checkpoint write failed: {_KEY}",
        "error": _KEY,
    }
    (checkpoints / "events.jsonl").write_text(json.dumps(event) + "\n")

    if _parent_reason(run, 7, "writer-7", 776) != "unexpected writer error":
        raise AssertionError("parent accepted a reason injected through errno.errorcode")


def test_synchronous_writer_reports_safe_error_type(
        tmp_path: Path, capsys, monkeypatch) -> None:
    def fail_save():
        raise RuntimeError(_KEY)

    warning, event = _run_synchronous_writer_failure(
        tmp_path, capsys, monkeypatch, _KEY, fail_save, "unexpected writer error")

    assert "unexpected writer error: RuntimeError" in warning
    assert event.get("writer_error_type") == "RuntimeError"
    _assert_secret_free(warning, _KEY)
    _assert_secret_free(event.get("writer_error_type"), _KEY)


def test_serialization_failure_logs_error_type_and_value_free_field_path(
        tmp_path: Path, capsys, monkeypatch) -> None:
    previous_state = kernel_checkpoint._STATE.copy()
    run = tmp_path / "run"
    (run / "wire").mkdir(parents=True)
    (run / "wire" / "0001.request.json").write_text("{}")
    kernel_checkpoint._STATE.clear()
    kernel_checkpoint._STATE.update(
        failed=kernel_checkpoint.MAX_CONSECUTIVE_FAILURES - 1,
        consecutive_failed=kernel_checkpoint.MAX_CONSECUTIVE_FAILURES - 1,
        disabled=False, last_sends=None, restored_step=None,
    )
    monkeypatch.setattr(kernel_checkpoint, "mode", lambda: "save")
    monkeypatch.setattr(kernel_checkpoint, "_every", lambda: 1)
    monkeypatch.setattr(kernel_checkpoint, "_async_enabled", lambda: False)
    monkeypatch.setattr(kernel_checkpoint, "_low_disk_space", lambda _run_dir: None)
    monkeypatch.setattr(kernel_checkpoint, "_sends", lambda _run_dir: 1)

    error = pickle.PicklingError(_KEY)
    kernel_checkpoint._mark_serialization_failure(error, {_KEY: contextvars.copy_context()})

    def fail_save(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(kernel_checkpoint, "save", fail_save)
    try:
        kernel_checkpoint.save_point(object(), 1, run / "checkpoint-target")
        output = capsys.readouterr().err
        events = [json.loads(line) for line in
                  (run / "checkpoints" / "events.jsonl").read_text().splitlines()]
        event = next(row for row in reversed(events) if row.get("event") == "checkpoint_failed")

        assert event["serialization_error_type"] == "PicklingError"
        assert event["serialization_field_path"] == "payload.mapping_value[0]"
        assert "PicklingError at payload.mapping_value[0]" in output
        _assert_secret_free(output, _KEY)
        _assert_secret_free(event["serialization_field_path"], _KEY)
        assert event["error"] == "checkpoint serialization failed"
    finally:
        kernel_checkpoint._STATE.clear()
        kernel_checkpoint._STATE.update(previous_state)
