"""Fault rows never carry a registered secret, for every registered handler.

record_fault stores the caught error's message in problems.jsonl and the reason line. A message that
contains a secret registered with the checkpoint byte guard (raw or JSON-escaped) is withheld, and an
error whose __str__ raises is recorded with a fixed text instead of breaking the recording path.
"""
from __future__ import annotations

import json

import pytest

import kernel_checkpoint
import kernel_fault_disposition as disposition
from _fault_contract import isolated_reason_log
from kernel_fault_registry import HANDLERS

SECRET = "sk-test-0123456789abcdef-secret"


@pytest.fixture
def registered_secret():
    previous = kernel_checkpoint._SECRETS.copy()
    kernel_checkpoint.register_secret(SECRET)
    try:
        yield SECRET
    finally:
        kernel_checkpoint._SECRETS[:] = previous


class _HostileStr(RuntimeError):
    def __str__(self) -> str:
        raise ValueError("no text for you")


def _record(handler_id: str, error: BaseException, monkeypatch, tmp_path) -> tuple[list[dict], str]:
    spec = HANDLERS[handler_id]
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    with isolated_reason_log(monkeypatch, log_dir) as harness:
        disposition.record_fault(handler_id, error, disposition=spec.dispositions[0],
                                 observed_at=handler_id, fallback_id=spec.fallback_id, journal=False)
        rows = harness.rows()
    persisted = "".join(path.read_text() for path in log_dir.rglob("*") if path.is_file())
    return rows, persisted


@pytest.mark.parametrize("handler_id", sorted(HANDLERS))
@pytest.mark.parametrize("form", ["raw", "json_escaped"])
def test_secret_in_error_message_never_reaches_a_fault_row(handler_id, form, registered_secret,
                                                           monkeypatch, tmp_path, capsys):
    text = registered_secret if form == "raw" else json.dumps(registered_secret)[1:-1].replace("-", "\\u002d")
    rows, persisted = _record(handler_id, RuntimeError(f"failed with token {text}"), monkeypatch, tmp_path)

    assert len(rows) == 1, rows
    assert rows[0]["fields"]["exception_message"] == disposition.SECRET_WITHHELD_DETAIL
    captured = capsys.readouterr()
    for blob in (json.dumps(rows), persisted, captured.out, captured.err):
        assert registered_secret not in blob


@pytest.mark.parametrize("handler_id", sorted(HANDLERS)[:3])
def test_hostile_str_is_recorded_without_breaking_the_path(handler_id, monkeypatch, tmp_path):
    rows, _ = _record(handler_id, _HostileStr(), monkeypatch, tmp_path)

    assert len(rows) == 1, rows
    assert rows[0]["fields"]["exception_class"] == "_HostileStr"
    assert rows[0]["fields"]["exception_message"] == disposition.UNRENDERABLE_DETAIL


def test_plain_message_is_kept_when_no_secret_is_registered(monkeypatch, tmp_path):
    previous = kernel_checkpoint._SECRETS.copy()
    kernel_checkpoint._SECRETS[:] = []
    try:
        handler_id = sorted(HANDLERS)[0]
        rows, _ = _record(handler_id, RuntimeError("disk full"), monkeypatch, tmp_path)
    finally:
        kernel_checkpoint._SECRETS[:] = previous
    assert rows[0]["fields"]["exception_message"] == "disk full"
