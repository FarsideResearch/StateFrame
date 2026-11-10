"""I/O failures leave typed, journaled gaps in host obligation evidence."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import kernel_obligation_assess as obligations
import pytest


def _journal():
    events = []
    return SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))), events


def _assert_unavailable(value, events, code):
    assert isinstance(value, obligations.UnavailableEvidence)
    assert value.reason_code == code
    assert events == [("host_obligation_evidence_failure", {
        "producer": "kernel_obligation_assess", "dispatch": False,
        "reason_code": code,
    })]


@pytest.mark.parametrize("failure", [OSError, UnicodeError, SyntaxError])
def test_definition_scan_returns_typed_gap_once(tmp_path, monkeypatch, failure):
    (tmp_path / "app.py").write_text("def value(arg):\n    return arg\n")
    journal, events = _journal()
    original = Path.read_text

    def broken(self, *args, **kwargs):
        if self.name == "app.py":
            raise failure("unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", broken)
    _assert_unavailable(obligations.mutation_sites(tmp_path, "app:value", journal=journal),
                        events, "DEFINITION_SOURCE_UNAVAILABLE")


@pytest.mark.parametrize("failure", [OSError, UnicodeError])
def test_mutation_rescan_returns_typed_gap_once(tmp_path, monkeypatch, failure):
    (tmp_path / "app.py").write_text("def value(arg):\n    arg.append(1)\n")
    journal, events = _journal()
    original = Path.read_text
    reads = 0

    def broken(self, *args, **kwargs):
        nonlocal reads
        if self.name == "app.py":
            reads += 1
            if reads == 2:
                raise failure("unavailable")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", broken)
    _assert_unavailable(obligations.mutation_sites(tmp_path, "app:value", journal=journal),
                        events, "MUTATION_SOURCE_UNAVAILABLE")


@pytest.mark.parametrize("failure", [OSError, UnicodeError])
def test_public_test_read_returns_typed_gap_once(tmp_path, monkeypatch, failure):
    path = tmp_path / "public.py"
    path.write_text("value([1])\n")
    journal, events = _journal()

    def broken(*args, **kwargs):
        raise failure("unavailable")

    monkeypatch.setattr(Path, "read_text", broken)
    _assert_unavailable(obligations.fixture_from_public_test(path, "app:value", journal=journal),
                        events, "PUBLIC_TEST_SOURCE_UNAVAILABLE")


def test_snapshot_read_returns_typed_gap_once(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("x = 1\n")
    journal, events = _journal()

    def broken(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(Path, "read_bytes", broken)
    _assert_unavailable(obligations.workspace_snapshot(tmp_path, journal=journal),
                        events, "WORKSPACE_SNAPSHOT_UNAVAILABLE")


def test_live_check_entry_emits_snapshot_event(tmp_path, monkeypatch):
    """Planted control: the run_check entry reaches the real snapshot handler."""
    (tmp_path / "app.py").write_text("def value():\n    return 1\n")
    journal, events = _journal()

    def broken(*args, **kwargs):
        raise OSError("planted live-path failure")

    monkeypatch.setattr(Path, "read_bytes", broken)
    check = obligations.CheckDefinition("check", "obligation", "OUTPUT_EQUALS",
                                        entry_point="app:value", expected=1)
    _assert_unavailable(obligations.run_check(tmp_path, check, journal=journal),
                        events, "WORKSPACE_SNAPSHOT_UNAVAILABLE")


def test_planted_runtime_host_closure_emits_snapshot_event(tmp_path, monkeypatch):
    """The live runtime call site forwards its journal into the snapshot read."""
    from kernel_stateframe_runtime import StateFrameRuntime

    (tmp_path / "app.py").write_text("x = 1\n")
    journal, events = _journal()

    def broken(*args, **kwargs):
        raise OSError("planted runtime failure")

    monkeypatch.setattr(Path, "read_bytes", broken)
    runtime = SimpleNamespace(
        stateframe_host_closure_mode="on", owner=SimpleNamespace(root=tmp_path, task_id="task"),
        _host_check_bundle=lambda: None, binding=lambda: {}, journal=journal,
    )
    assert StateFrameRuntime._host_closure_for_done(runtime) == ((), None, None)
    assert runtime.host_closure_snapshot is None
    assert events == [("host_obligation_evidence_failure", {
        "producer": "kernel_obligation_assess", "dispatch": False,
        "reason_code": "WORKSPACE_SNAPSHOT_UNAVAILABLE",
    })]
