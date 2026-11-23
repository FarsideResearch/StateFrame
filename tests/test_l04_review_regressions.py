from __future__ import annotations

import errno
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

import kernel_reason_log as reason_log
import kernel_request_auth_identity as auth
import kernel_run_vitals as vitals
from kernel_reason_log import builtin_exception_name

ROOT = Path(__file__).resolve().parents[1]
BASE = "6063b3c499"


def _git_show(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)


def _engine_module_at(ref: str, name: str) -> ModuleType:
    path = f"src/stateframe/engine/{name}.py"
    module = ModuleType(name)
    exec(compile(_git_show(ref, path), f"{ref}:{path}", "exec"), vars(module))
    return module


@pytest.fixture
def diagnostics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[StringIO, Path]:
    stream = StringIO()
    path = tmp_path / "problems.jsonl"
    monkeypatch.setattr(reason_log, "_STATE", reason_log._fresh())
    reason_log.configure(level="WARN", stream=stream, problems_path=path)
    return stream, path


class Journal:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.events: list[tuple[str, dict[str, object]]] = []
        self._run_vitals_accumulator = vitals.RunVitalsAccumulator()

    def emit(self, kind: str, **fields: object) -> dict[str, object]:
        self.events.append((kind, fields))
        if kind == "run_vitals" and self.error is not None:
            raise self.error
        return {"kind": kind, **fields}


def _deny_key_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = tmp_path / "identity.key"
    key.write_bytes(b"r" * 32)
    key.chmod(0o600)
    original_open = os.open

    def deny_read(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if os.fspath(path) == str(key) and (flags & os.O_ACCMODE) == os.O_RDONLY:
            raise PermissionError(
                errno.EACCES, "synthetic identity read failure", str(key),
            )
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_read)
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path))
    monkeypatch.setattr(auth, "_AUTH_BINDING_SECRET_CACHE", {})


def test_denied_identity_read_reports_builtin_failure_class_without_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    diagnostics: tuple[StringIO, Path],
) -> None:
    _deny_key_read(monkeypatch, tmp_path)
    stream, _ = diagnostics

    assert auth.auth_binding_for_key("synthetic authorization") is not None

    output = stream.getvalue()
    assert str(tmp_path) not in output
    assert "synthetic identity read failure" not in output
    assert "PermissionError" in output


def test_identity_journal_false_allows_only_the_documented_final_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    diagnostics: tuple[StringIO, Path],
) -> None:
    _deny_key_read(monkeypatch, tmp_path)
    journal = Journal()
    reason_log.bind_journal(journal)

    assert auth.auth_binding_for_key("synthetic authorization") is not None
    assert journal.events == []
    summary = reason_log.flush_summary(journal)

    assert summary is not None
    assert [kind for kind, _ in journal.events] == ["reason_log_summary"]
    _, fields = journal.events[0]
    assert fields["dispatch"] is False
    assert fields["total"] == 1
    assert set(fields["by_cause"]) == {
        "fallback:installation_identity_process_local",
    }


def test_vitals_journal_false_keeps_only_the_run_end_summary(
    monkeypatch: pytest.MonkeyPatch,
    diagnostics: tuple[StringIO, Path],
) -> None:
    monkeypatch.setenv(vitals.FLAG_ENV, "on")
    journal = Journal(OSError("synthetic vitals write failure"))
    reason_log.bind_journal(journal)

    assert vitals.emit_run_vitals(journal) is None
    assert vitals.emit_run_vitals(journal) is None
    reason_log.flush_summary(journal)

    assert [kind for kind, _ in journal.events] == [
        "run_vitals", "reason_log_summary",
    ]
    assert journal.events[-1][1]["dispatch"] is False
    assert "fallback:run_vitals_emit_failed" in journal.events[-1][1]["by_cause"]


def test_custom_vitals_exception_name_is_reduced_to_other(
    monkeypatch: pytest.MonkeyPatch,
    diagnostics: tuple[StringIO, Path],
) -> None:
    monkeypatch.setenv(vitals.FLAG_ENV, "on")
    marker = "/synthetic/private/terminal-storage"
    error_type = type(f"WriteError at {marker}", (OSError,), {})
    stream, path = diagnostics

    assert vitals.emit_run_vitals(Journal(error_type("private error text"))) is None

    output = stream.getvalue() + path.read_text(encoding="utf-8")
    assert "run_vitals_emit_failed" in output
    assert "other" in output
    assert "private error text" not in output
    assert marker not in output


def test_exception_class_diagnostics_use_only_the_fixed_builtin_allowlist() -> None:
    class CustomPermissionError(PermissionError):
        pass

    assert builtin_exception_name(PermissionError()) == "PermissionError"
    assert builtin_exception_name(ValueError()) == "ValueError"
    assert builtin_exception_name(KeyError()) == "other"
    assert builtin_exception_name(CustomPermissionError()) == "other"


def test_exception_class_diagnostics_never_hash_or_compare_custom_classes() -> None:
    calls: list[str] = []

    class HostileExceptionMeta(type):
        def __hash__(cls) -> int:
            calls.append("hash")
            raise AssertionError("custom exception class was hashed")

        def __eq__(cls, other: object) -> bool:
            calls.append("eq")
            raise AssertionError("custom exception class was compared")

    class HostileStorageError(OSError, metaclass=HostileExceptionMeta):
        pass

    calls.clear()
    assert builtin_exception_name(HostileStorageError()) == "other"
    assert calls == []


def test_unhashable_storage_exception_does_not_break_auth_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    diagnostics: tuple[StringIO, Path],
) -> None:
    class UnhashableExceptionMeta(type):
        __hash__ = None

    class WrappedStorageError(OSError, metaclass=UnhashableExceptionMeta):
        pass

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path))
    monkeypatch.setattr(auth, "_AUTH_BINDING_SECRET_CACHE", {})
    real_open = os.open

    def deny_key_creation(path: object, flags: int, *args: object, **kwargs: object) -> int:
        key = Path(path)
        if key.parent == tmp_path and key.name.startswith(".identity.key."):
            raise WrappedStorageError("synthetic storage-layer failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_key_creation)

    assert auth.auth_binding_for_key("synthetic-review-key") is not None
    assert "persistent identity key unavailable (other)" in diagnostics[0].getvalue()


def test_reason_summary_does_not_return_a_custom_exception_name(
    monkeypatch: pytest.MonkeyPatch,
    diagnostics: tuple[StringIO, Path],
) -> None:
    marker = "/synthetic/private/summary-state"
    custom_error = type(f"Failure at {marker}", (RuntimeError,), {})

    class BrokenCauses:
        def items(self) -> object:
            raise custom_error("private state")

    monkeypatch.setitem(reason_log._STATE, "causes", BrokenCauses())
    summary = reason_log.run_summary()

    assert summary["summary_error"] == "other"
    assert marker not in repr(summary)


def test_auth_fallback_request_measurement_bytes_match_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    diagnostics: tuple[StringIO, Path],
) -> None:
    from kernel_request_diet import RequestTokenMeter
    import kernel_request_diet as diet

    _deny_key_read(monkeypatch, tmp_path)
    old = _engine_module_at(BASE, "kernel_request_auth_identity")
    fixed_fallback = b"f" * 32
    monkeypatch.setattr(auth, "_AUTH_BINDING_FALLBACK_SECRET", fixed_fallback)
    old._AUTH_BINDING_FALLBACK_SECRET = fixed_fallback
    body = {"model": "scripted", "messages": [{"role": "user", "content": "review"}]}
    boundaries = []
    for binding in (old.auth_binding_for_key, auth.auth_binding_for_key):
        with monkeypatch.context() as patch_context:
            patch_context.setattr(diet, "_auth_binding_for_key", binding)
            meter = RequestTokenMeter(disabled=True)
            meter.bind_tokenizer_auth("synthetic authorization", "https://example.invalid/v1")
            tokens, method = meter.measure(body)
            measurement = meter.bind_measurement(
                body, tokens=tokens, method=method, request_id="review",
                max_request_bytes=100_000, max_prompt_tokens=33_344,
            )
            boundaries.append(measurement)

    assert boundaries[0] == boundaries[1]
    assert boundaries[0].body_sha256 == "sha256:" + hashlib.sha256(
        json.dumps(body).encode(),
    ).hexdigest()
