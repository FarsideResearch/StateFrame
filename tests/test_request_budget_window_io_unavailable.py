"""Request builders retain typed evidence gaps and journal them on live paths."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import kernel_request_budget as budget
import kernel_request_source_window as window
from kernel_obligation_assess import UnavailableEvidence
import pytest


def _journal():
    events = []
    return SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))), events


def _gap(value, events, reason, producer):
    assert isinstance(value, UnavailableEvidence)
    assert value.reason_code == reason
    assert events == [("host_obligation_evidence_failure", {
        "producer": producer, "dispatch": False, "reason_code": reason,
    })]


def _capture(journal):
    return SimpleNamespace(journal=journal, request_id="r1", step=1)


def test_owner_resolver_gap_on_live_budget_entry():
    journal, events = _journal()

    def broken():
        raise RuntimeError("detached owner")

    runtime = SimpleNamespace(_idealpath_owner=broken,
                              declared_environment={budget.MODE_ENV: "on"})
    capture = _capture(journal)
    budget.apply_runtime_budget(runtime, capture, [], [])
    _gap(capture.request_budget_unavailable, events, "REQUEST_BUDGET_OWNER_UNAVAILABLE",
         "kernel_request_budget")
    assert capture.request_budget_final_wire["status"] == budget.HOST_CONTRACT_LIMIT
    assert capture.request_budget_final_wire["final_validation_failures"] == (
        "REQUEST_BUDGET_OWNER_UNAVAILABLE",)


def test_stage_liveness_gap_on_live_budget_entry():
    journal, events = _journal()

    def broken():
        raise ValueError("stage unavailable")

    owner = SimpleNamespace(stage="S1", stage_exit_liveness=broken)
    runtime = SimpleNamespace(ideal_path_owner=owner,
                              declared_environment={budget.MODE_ENV: "on"})
    capture = _capture(journal)
    budget.apply_runtime_budget(runtime, capture, [], [])
    _gap(capture.request_budget_unavailable, events, "REQUEST_BUDGET_STAGE_LIVENESS_UNAVAILABLE",
         "kernel_request_budget")


def test_active_runtime_lookup_gap_on_capture_entry(monkeypatch):
    import kernel_pilot

    journal, events = _journal()

    class BrokenContext:
        def get(self):
            raise LookupError("context detached")

    monkeypatch.setattr(kernel_pilot, "ACTIVE_RUNTIME", BrokenContext())
    capture = _capture(journal)
    budget.apply_capture_budget(capture, {"messages": [], "tools": []})
    _gap(capture.request_budget_unavailable, events, "REQUEST_BUDGET_RUNTIME_UNAVAILABLE",
         "kernel_request_budget")


def test_path_resolution_gap_on_live_source_projection(tmp_path, monkeypatch):
    journal, events = _journal()
    target = tmp_path / "app.py"
    target.write_text("value = 1\n")
    runtime = SimpleNamespace(target=target, owner=SimpleNamespace(root=tmp_path), journal=journal)
    original = Path.resolve

    def broken(self, *args, **kwargs):
        if self == target:
            raise OSError("target vanished")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", broken)
    result = window.bound_runtime_sources([], runtime, max_request_bytes=4096)
    _gap(result, events, "REQUEST_SOURCE_TARGET_UNAVAILABLE",
         "kernel_request_source_window")


def test_product_file_gap_on_live_literal_inventory(tmp_path, monkeypatch):
    journal, events = _journal()
    (tmp_path / "app.py").write_text("token = 'needle'\n")
    original = Path.read_text

    def broken(self, *args, **kwargs):
        if self.name == "app.py":
            raise OSError("read failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", broken)
    result = window.literal_site_inventory(tmp_path, ("app.py",), ("needle",), journal=journal)
    _gap(result, events, "REQUEST_SOURCE_FILE_UNAVAILABLE",
         "kernel_request_source_window")


def test_product_file_gap_on_live_entry_verification(tmp_path, monkeypatch):
    journal, events = _journal()
    (tmp_path / "app.py").write_text("def value():\n    return 1\n")
    owner = SimpleNamespace(root=tmp_path, fn="value", task_row={"editable_paths": ["app.py"]})
    runtime = SimpleNamespace(owner=owner, target=tmp_path / "app.py", journal=journal)
    original = Path.read_text

    def broken(self, *args, **kwargs):
        if self.name == "app.py":
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", broken)
    _gap(window.entry_verification(runtime), events, "REQUEST_SOURCE_FILE_UNAVAILABLE",
         "kernel_request_source_window")


def test_declared_outline_gap_on_live_entry_verification(tmp_path, monkeypatch):
    journal, events = _journal()
    (tmp_path / "app.ts").write_text("export function value() { return 1; }\n")
    owner = SimpleNamespace(root=tmp_path, fn="value", task_row={"editable_paths": ["app.ts"]})
    runtime = SimpleNamespace(owner=owner, target=tmp_path / "app.ts", journal=journal)

    def broken(_path, _source):
        raise OSError("outline parser unavailable")

    monkeypatch.setattr(window, "_outline_for", broken)
    result = window.entry_verification(runtime)
    _gap(result, events, "REQUEST_SOURCE_OUTLINE_UNAVAILABLE",
         "kernel_request_source_window")


@pytest.mark.parametrize("failure", [TypeError, KeyError])
def test_unexpected_owner_resolver_failure_propagates(failure):
    journal, events = _journal()

    def broken():
        raise failure("unexpected")

    runtime = SimpleNamespace(_idealpath_owner=broken)
    with pytest.raises(failure, match="unexpected"):
        budget._runtime_context(runtime, _capture(journal), [])
    assert events == []


@pytest.mark.parametrize("failure", [TypeError, KeyError])
def test_unexpected_outline_failure_propagates(tmp_path, monkeypatch, failure):
    journal, events = _journal()
    (tmp_path / "app.ts").write_text("export function value() { return 1; }\n")

    def broken(_path, _source):
        raise failure("unexpected")

    monkeypatch.setattr(window, "_outline_for", broken)
    with pytest.raises(failure, match="unexpected"):
        window._declared_definitions(tmp_path, ("app.ts",), "value", journal=journal)
    assert events == []


def test_unexpected_file_failure_propagates(tmp_path, monkeypatch):
    journal, events = _journal()
    (tmp_path / "app.py").write_text("x = 1\n")

    def broken(*_args, **_kwargs):
        raise TypeError("unexpected read failure")

    monkeypatch.setattr(Path, "read_text", broken)
    with pytest.raises(TypeError, match="unexpected read failure"):
        window._read_product_file(tmp_path, "app.py", journal=journal)
    assert events == []
