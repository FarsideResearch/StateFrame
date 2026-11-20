from types import SimpleNamespace

import pytest

import kernel_coordinated_change_ledger as coordination
from kernel_green_lock import GreenLock


@pytest.fixture
def scanned_files(monkeypatch):
    scanned = []
    file_scan = coordination._file_scan
    indexed_scan = coordination._indexed_scan

    def count_scan(rows, source, relative):
        if not relative.endswith(".py"):
            scanned.append(relative)
        return file_scan(rows, source, relative)

    def count_indexed_scan(index, source, relative, module):
        scanned.append(relative)
        return indexed_scan(index, source, relative, module)

    monkeypatch.setattr(coordination, "_file_scan", count_scan)
    monkeypatch.setattr(coordination, "_indexed_scan", count_indexed_scan)
    return scanned


def test_readiness_skips_coordination_scan_without_proposed_changes(tmp_path, monkeypatch):
    (tmp_path / "service.py").write_text("def subscribe():\n    pass\n")
    monkeypatch.setenv(coordination.LEDGER_ENV, "on")
    monkeypatch.setattr(coordination, "guidance_enabled", lambda: True)

    def unexpected_scan(*args, **kwargs):
        pytest.fail("Readiness started a coordination scan")

    monkeypatch.setattr(coordination, "_scan_repository", unexpected_scan)
    decision = object()
    runtime = SimpleNamespace(
        stateframe_readiness_authority_mode="on",
        owner=SimpleNamespace(root=tmp_path, task_row={"brief": "Find the subscription model."}),
    )
    runtime.readiness_decision = lambda frame: coordination.apply_readiness_to_decision(
        runtime, decision)

    assert GreenLock(runtime, mode="off")._readiness_snapshot() is decision
    assert runtime.coordinated_change_ledger.items == []


def test_ticket_facts_scan_only_definitions_and_direct_references(tmp_path, scanned_files):
    (tmp_path / "service.py").write_text("def subscribe():\n    pass\n")
    (tmp_path / "caller.py").write_text("from service import subscribe\n")
    (tmp_path / "notes.md").write_text("Call subscribe during setup.\n")
    (tmp_path / "other.py").write_text("import caller\n")
    ticket = "Rename `subscribe` to `enroll` and keep old callers working."
    ledger = coordination.CoordinatedChangeLedger.from_ticket(tmp_path, ticket)
    rename = ledger.item("rename:subscribe->enroll")
    alias = ledger.item("alias:subscribe->enroll")

    assert rename.defining_file == "service.py"
    assert {site.path for site in rename.usage_sites} == {"caller.py", "notes.md", "service.py"}
    assert {site.path for site in alias.importers} == {"caller.py"}
    assert alias.checkable
    assert ledger.coordination_evidence()["holds"]
    assert scanned_files == ["caller.py", "notes.md", "service.py"]

    coordination.CoordinatedChangeLedger.from_ticket(tmp_path, ticket)
    assert scanned_files == ["caller.py", "notes.md", "service.py"]


def test_inferred_input_target_keeps_definition_order(tmp_path, scanned_files):
    (tmp_path / "a_handler.py").write_text("def submit(record):\n    pass\n")
    (tmp_path / "b_handler.py").write_text("def submit(record):\n    return record['campus_id']\n")
    (tmp_path / "other.py").write_text("def unrelated():\n    pass\n")
    ledger = coordination.CoordinatedChangeLedger.from_ticket(
        tmp_path, "The word school becomes campus. Clients still send school_id.")

    item = ledger.item("input:school_id->campus_id")
    assert item.compatibility["symbol"] == "submit"
    assert item.defining_file == "a_handler.py"
    assert scanned_files == ["b_handler.py", "a_handler.py"]


def test_batch_refresh_defers_scan_and_reuses_unchanged_file_results(tmp_path, scanned_files):
    service = tmp_path / "registration.py"
    service.write_text("def subscribe():\n    pass\n")
    (tmp_path / "caller.py").write_text("from registration import subscribe\n")
    ledger = coordination.CoordinatedChangeLedger.from_ticket(
        tmp_path, "Rename `subscribe` to `enroll`.")
    assert ledger.scan.defining_file("subscribe") == "registration.py"
    scanned_files.clear()

    service.write_text("def enroll():\n    pass\n")
    revision = coordination.tree_revision(tmp_path)
    ledger.record_batch(coordination.BatchResult("APPLIED", "batch", revision, ("registration.py",)))
    assert ledger.item("rename:subscribe->enroll").status == coordination.STATUS_APPLIED
    assert scanned_files == []

    scan = ledger.scan
    assert scan.defining_file("enroll") == "registration.py"
    assert scan.defining_file("subscribe") == ""
    assert {site.path for site in scan.importers["subscribe"]} == {"caller.py"}
    assert ledger.scan is scan
    assert scanned_files == ["registration.py"]
