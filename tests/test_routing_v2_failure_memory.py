"""MR-8: an applied-but-wrong edit must not reset failure memory under STATEFRAME_ROUTING_SCHEMA=2.

Only verifying success (a passing test/check receipt) or a terminal DONE/COMMITTED clears the
counterexample/hypothesis counters under v2. Schema 1 (default) keeps the legacy clearing.
"""
import pytest

from kernel_stateframe_model_router import host_signal_projection

SCHEMA_ENV = "STATEFRAME_ROUTING_SCHEMA"


@pytest.fixture
def v1(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "1")
    return monkeypatch


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setenv(SCHEMA_ENV, "2")
    return monkeypatch


def _test(receipt_id, status, fingerprint="fp-A", hypothesis_id=None):
    row = {
        "receipt_id": receipt_id,
        "action": "RUN_TESTS",
        "status": status,
        "fingerprint": fingerprint,
        "binding_stable": True,
        "binding": {"rev": "r1"},
        "after_binding": {"rev": "r1"},
    }
    if hypothesis_id is not None:
        row["hypothesis_id"] = hypothesis_id
    return row


def _edit(receipt_id, status="APPLIED"):
    return {"receipt_id": receipt_id, "action": "EDIT", "status": status}


def _read(receipt_id, new_evidence=None):
    row = {"receipt_id": receipt_id, "action": "READ", "status": "OK"}
    if new_evidence is not None:
        row["new_evidence"] = new_evidence
    return row


def _signals(rows):
    return host_signal_projection(None, {"committed_receipts": rows})["signals"]


def test_v2_applied_edit_keeps_first_failure(v2):
    rows = [_test("t1", "FAIL"), _edit("e1"), _test("t2", "FAIL")]
    assert _signals(rows)["same_valid_counterexample"] == 2


def test_v1_applied_edit_clears_failure_history(v1):
    rows = [_test("t1", "FAIL"), _edit("e1"), _test("t2", "FAIL")]
    assert _signals(rows)["same_valid_counterexample"] == 1


def test_v2_applied_edit_keeps_hypothesis_repeats(v2):
    rows = [
        _test("t1", "FAIL", hypothesis_id="h1"),
        _edit("e1"),
        _test("t2", "FAIL", hypothesis_id="h1"),
    ]
    assert _signals(rows)["rejected_hypothesis_repeats"] == 2


def test_v2_passing_test_clears_failure_history(v2):
    rows = [_test("t1", "FAIL"), _test("t2", "PASSED")]
    signals = _signals(rows)
    assert signals["same_valid_counterexample"] == 0
    assert signals["rejected_hypothesis_repeats"] == 0


def test_v2_terminal_done_clears_failure_history(v2):
    rows = [_test("t1", "FAIL"), {"receipt_id": "d1", "action": "DONE", "status": "DONE"}]
    assert _signals(rows)["same_valid_counterexample"] == 0


def test_v2_applied_edit_is_not_verifying_success(v2):
    # An APPLIED edit must not satisfy the verifying-success rule even though it still
    # counts as a successful host action for evidence relevance.
    rows = [_test("t1", "FAIL"), _edit("e1", status="OK")]
    assert _signals(rows)["same_valid_counterexample"] == 1


def test_v2_read_with_new_receipt_resets_reads_since_action(v2):
    rows = [_read("r1"), _read("r2"), _read("r3", new_evidence=True)]
    assert _signals(rows)["reads_without_discriminating_action"] == 0


def test_v2_plain_reads_still_accumulate(v2):
    rows = [_read("r1"), _read("r2"), _read("r3")]
    assert _signals(rows)["reads_without_discriminating_action"] == 3


def test_v1_read_with_new_receipt_is_unchanged(v1):
    rows = [_read("r1"), _read("r2"), _read("r3", new_evidence=True)]
    assert _signals(rows)["reads_without_discriminating_action"] == 3


def test_schema_unset_behaves_as_v1(monkeypatch):
    monkeypatch.delenv(SCHEMA_ENV, raising=False)
    rows = [_test("t1", "FAIL"), _edit("e1"), _test("t2", "FAIL")]
    assert _signals(rows)["same_valid_counterexample"] == 1
