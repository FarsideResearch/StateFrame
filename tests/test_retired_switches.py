"""Retired STATEFRAME switches are ignored with a journal row, never refused.

Covers the removed read lockout switch: it must not register as a reader, must not be reported
as unknown, must not stop a run, and its old "[Read lockout] " host panel must not be trusted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
sys.path.insert(0, str(ENGINE))
import kernel_release_switch_registry as registry  # noqa: E402

RETIRED = "STATEFRAME_READ_LOCKOUT"


class _FakeJournal:
    """Minimal journal double: records emit calls and exposes the path enforce() reports."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[tuple[str, dict[str, object]]] = []

    def emit(self, kind: str, **fields: object) -> None:
        self.rows.append((kind, fields))


@pytest.fixture
def journal(tmp_path: Path) -> _FakeJournal:
    return _FakeJournal(tmp_path / "journal.jsonl")


def test_retired_switch_is_not_unknown_and_is_listed_as_retired():
    detail = registry.unconsumed_stateframe_switches({RETIRED: "on", "PATH": "/bin"})
    assert detail["unknown_switches"] == []
    assert detail["retired_switches"] == [RETIRED]


def test_retired_switch_reason_is_published_in_the_module_table():
    assert "never hides read routes" in registry.RETIRED_SWITCHES[RETIRED]


def test_enforce_with_only_a_retired_switch_does_not_refuse_and_journals_one_row(journal):
    detail = registry.enforce_stateframe_switch_registry({RETIRED: "on"}, journal=journal)
    assert detail["retired_switches"] == [RETIRED]
    assert [kind for kind, _ in journal.rows] == ["retired_switch_ignored"]
    _, fields = journal.rows[0]
    assert fields["retired_switches"] == [RETIRED]
    assert fields["reasons"] == {RETIRED: registry.RETIRED_SWITCHES[RETIRED]}
    assert fields["dispatch"] is False


def test_enforce_with_no_retired_switch_journals_nothing(journal):
    registry.enforce_stateframe_switch_registry({"PATH": "/bin"}, journal=journal)
    assert journal.rows == []


def test_unknown_switch_is_still_refused_and_retired_row_is_still_journaled(journal):
    with pytest.raises(registry.StateFrameSwitchPreflightError) as raised:
        registry.enforce_stateframe_switch_registry(
            {"STATEFRAME_FOO_BAR": "1", RETIRED: "on"}, journal=journal)
    assert raised.value.detail["unknown_switches"] == ["STATEFRAME_FOO_BAR"]
    kinds = [kind for kind, _ in journal.rows]
    assert kinds == ["retired_switch_ignored", "stateframe_switch_preflight_refused"]


def test_unknown_switch_alone_is_refused(journal):
    with pytest.raises(registry.StateFrameSwitchPreflightError):
        registry.enforce_stateframe_switch_registry({"STATEFRAME_FOO_BAR": "1"}, journal=journal)
    assert [kind for kind, _ in journal.rows] == ["stateframe_switch_preflight_refused"]


def test_shipped_registry_does_not_list_the_retired_switch():
    switches, _ = registry.release_stateframe_switches(ENGINE)
    assert RETIRED not in switches
    shipped = json.loads((ENGINE / registry.REGISTRY_FILE).read_text(encoding="utf-8"))
    assert RETIRED not in shipped["switches"]


def test_retired_table_literal_does_not_register_its_keys_as_readers(tmp_path: Path):
    # The scanner must skip the RETIRED_SWITCHES table even though its dict literal
    # holds STATEFRAME_ keys and the name is loaded elsewhere in the module.
    root = tmp_path / "engine"
    root.mkdir()
    (root / "table.py").write_text(
        'RETIRED_SWITCHES = {"STATEFRAME_GONE": "removed"}\n'
        "def reason():\n"
        "    return RETIRED_SWITCHES\n",
        encoding="utf-8")
    (root / "release-manifest.json").write_text(
        json.dumps({"files": {"table.py": ""}}), encoding="utf-8")
    document = registry.build_registry_document(root)
    assert "STATEFRAME_GONE" not in document["switches"]


def test_framefit5_does_not_treat_a_read_lockout_message_as_a_host_panel():
    from kernel_stateframe_framefit5 import _is_framefit5_host_context_panel

    message = {
        "role": "user",
        "content": "[Read lockout] pkg/app.py was re-read unchanged",
        "_request_budget_section": "stage_evidence",
    }
    assert _is_framefit5_host_context_panel(message) is False
