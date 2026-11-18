"""MR-8 read-evidence producer: a read receipt is flagged as new evidence only when it returns new content.

Under STATEFRAME_ROUTING_SCHEMA=2 the live read path sets ``new_evidence=True`` on the receipt row
when the (path, content revision, line range) identity was not yet seen in this run. The router
projection already resets ``reads_without_discriminating_action`` on that flag under v2. Schema 1
never adds the key and never allocates the seen set.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

import kernel_read_exec as reads
import kernel_read_evidence as evidence
from kernel_stateframe_model_router import host_signal_projection

SCHEMA_ENV = "STATEFRAME_ROUTING_SCHEMA"


class Journal:
    def emit(self, *_args, **_fields):
        return None


class Bolt:
    """Minimal stand-in for KernelBolt: appends read rows shaped like its receipt()."""

    def __init__(self):
        self._receipts: list[dict] = []

    def receipt(self, kind, intent, material, result_class, detail="", **_kwargs):
        self._receipts.append({
            "kind": kind,
            "result_class": result_class,
            "fingerprint": material,
            "receipt_id": f"rcpt-{len(self._receipts) + 1}",
        })


def _rev(tag: str) -> str:
    return "sha256:" + hashlib.sha256(tag.encode()).hexdigest()


@pytest.fixture
def content(monkeypatch):
    """Per-path content revision the fake read executor reports; mutate to simulate an edit."""
    revs: dict[str, str] = {}

    def fake_execute_read(root, p, raw, display, start, *, delivery, **_kwargs):
        delivery["read_dedup"] = {
            "status": "REEXECUTED",
            "revision": revs[str(p)],
            "effective_range": {"start_line": 1, "end_line": 2},
        }
        return "OK", f"[read] {display}", b"alpha\nbeta\n"

    monkeypatch.setattr(reads, "execute_read", fake_execute_read)
    return revs


@pytest.fixture
def v1(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "1")


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")


def _runtime():
    return SimpleNamespace(journal=Journal(), _frame_budget_request_count=0)


def _read(rt, bolt, path):
    return reads.dispatch_live_read(
        path.parent, path, path.name, path.name, 0,
        args={"path": path.name}, measurement=rt, step=1, bolt=bolt,
        policy=None, resolved_path=path.name)


def _signals(rows):
    return host_signal_projection(None, {"committed_receipts": rows})["signals"]


def test_v2_first_read_is_flagged_and_resets_reads_counter(tmp_path, v2, content):
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    rt, bolt = _runtime(), Bolt()
    _read(rt, bolt, path)
    assert bolt._receipts[-1]["new_evidence"] is True
    assert _signals(bolt._receipts)["reads_without_discriminating_action"] == 0


def test_v2_unchanged_reread_is_not_flagged_and_counts(tmp_path, v2, content):
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    rt, bolt = _runtime(), Bolt()
    _read(rt, bolt, path)
    _read(rt, bolt, path)
    assert "new_evidence" not in bolt._receipts[1]
    assert _signals(bolt._receipts)["reads_without_discriminating_action"] == 1


def test_v2_repeated_unchanged_reads_accumulate_until_new_evidence(tmp_path, v2, content):
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    rt, bolt = _runtime(), Bolt()
    for _ in range(3):
        _read(rt, bolt, path)
    assert _signals(bolt._receipts)["reads_without_discriminating_action"] == 2


def test_v2_content_changed_by_edit_is_new_evidence_again(tmp_path, v2, content):
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    rt, bolt = _runtime(), Bolt()
    _read(rt, bolt, path)
    _read(rt, bolt, path)
    content[str(path)] = _rev("a-2")  # an edit changed the file revision
    _read(rt, bolt, path)
    assert bolt._receipts[2]["new_evidence"] is True
    assert _signals(bolt._receipts)["reads_without_discriminating_action"] == 0


def test_v2_first_read_of_new_path_is_flagged_even_after_other_reads(tmp_path, v2, content):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    content[str(a)] = _rev("a-1")
    content[str(b)] = _rev("b-1")
    rt, bolt = _runtime(), Bolt()
    _read(rt, bolt, a)
    _read(rt, bolt, a)
    _read(rt, bolt, b)
    assert bolt._receipts[2]["new_evidence"] is True


def test_v2_seen_set_is_bounded_and_oldest_entries_drop(tmp_path, v2, content, monkeypatch):
    monkeypatch.setattr(evidence, "SEEN_CAP", 2)
    a, b, c = (tmp_path / f"{name}.txt" for name in "abc")
    for path in (a, b, c):
        content[str(path)] = _rev(path.name)
    rt, bolt = _runtime(), Bolt()
    for path in (a, b, c):
        _read(rt, bolt, path)
    # A was evicted when C arrived, so its repeat now counts as unseen evidence.
    _read(rt, bolt, a)
    assert bolt._receipts[-1]["new_evidence"] is True
    # B and C are still in the set, so an unchanged repeat of C is not new.
    _read(rt, bolt, c)
    assert "new_evidence" not in bolt._receipts[-1]


def test_v2_seen_set_is_per_run(tmp_path, v2, content):
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    first_bolt, second_bolt = Bolt(), Bolt()
    _read(_runtime(), first_bolt, path)
    _read(_runtime(), second_bolt, path)
    assert second_bolt._receipts[0]["new_evidence"] is True


def test_v1_read_producer_never_flags_rows(tmp_path, v1, content):
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    rt, bolt = _runtime(), Bolt()
    for _ in range(3):
        _read(rt, bolt, path)
    assert all("new_evidence" not in row for row in bolt._receipts)
    assert _signals(bolt._receipts)["reads_without_discriminating_action"] == 3


def test_v1_does_not_allocate_seen_state(tmp_path, v1, content):
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    rt, bolt = _runtime(), Bolt()
    _read(rt, bolt, path)
    assert not hasattr(rt, evidence.SEEN_ATTR)


def test_schema_unset_behaves_as_v1(tmp_path, monkeypatch, content):
    monkeypatch.delenv(SCHEMA_ENV, raising=False)
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    rt, bolt = _runtime(), Bolt()
    _read(rt, bolt, path)
    assert "new_evidence" not in bolt._receipts[0]


def test_v2_new_evidence_read_is_never_committed_or_completed_action(tmp_path, v2, content):
    # A flagged read is evidence, not an action: it must not become the committed or completed
    # action receipt, even when it is the newest row in the stream.
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    rt, bolt = _runtime(), Bolt()
    _read(rt, bolt, path)
    assert bolt._receipts[0]["new_evidence"] is True
    projection = host_signal_projection(None, {"committed_receipts": bolt._receipts})
    assert projection["committed_action_receipt_id"] is None
    assert projection["completed_action_receipt_id"] is None
    assert projection["committed_action_meaningful"] is False


def test_v2_new_evidence_read_does_not_displace_prior_action(tmp_path, v2, content):
    path = tmp_path / "a.txt"
    content[str(path)] = _rev("a-1")
    edit = {"receipt_id": "edit-1", "action": "EDIT", "status": "APPLIED"}
    rt, bolt = _runtime(), Bolt()
    _read(rt, bolt, path)
    projection = host_signal_projection(None, {"committed_receipts": [edit, *bolt._receipts]})
    assert projection["committed_action_receipt_id"] == "edit-1"
    assert projection["completed_action_receipt_id"] == "edit-1"
