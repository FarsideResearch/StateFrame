"""A replacement capture shares the step's host-authorship ledger; it never forks it."""
from __future__ import annotations

import kernel_host_authorship as authorship
import kernel_measurement
import kernel_request_reprepare as reprepare


def test_replacement_capture_shares_ledger_by_reference(tmp_path, monkeypatch):
    monkeypatch.setenv(authorship.MODE_ENV, "shadow")
    journal = kernel_measurement.AuditJournal(tmp_path, "host-authorship-reprepare", {})
    original = kernel_measurement.RequestCapture(journal, 2, "snapshot")
    ledger = authorship.HostAuthorship(2, original.request_id)
    row = authorship.author_row(ledger, "user", authorship.builder().host("[Panel]"), kind="panel",
                                site="t.panel")
    original._host_authorship = ledger
    original._framefit5_newest_user_suffix = ({"role": "user", "content": "newest"},)

    fresh = reprepare._replacement_capture(original)

    assert fresh is not original
    assert fresh._host_authorship is ledger  # same object: not a deep copy, not dropped
    # Ordinary stable fields are still copied, so only the ledger is shared.
    assert fresh._framefit5_newest_user_suffix == original._framefit5_newest_user_suffix
    assert fresh._framefit5_newest_user_suffix[0] is not original._framefit5_newest_user_suffix[0]
    # Rows the replacement's passes author are visible through the original's reference.
    authorship.author_row(fresh._host_authorship, "user", authorship.builder().host("[Later]"),
                          kind="panel", site="t.later")
    assert authorship.row_kind(original._host_authorship, {"role": "user", "content": "[Later]"}) == "panel"
    assert authorship.row_kind(fresh._host_authorship, row) == "panel"
    assert "_host_authorship" in reprepare._STABLE and "_host_authorship" in reprepare._OPAQUE
    # Off: there is nothing to share and nothing is created.
    plain = kernel_measurement.RequestCapture(journal, 2, "snapshot")
    assert reprepare._replacement_capture(plain)._host_authorship is None
