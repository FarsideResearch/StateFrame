"""Every request capture has a host-authorship slot, empty until a ledger is bound."""
from __future__ import annotations

import kernel_host_authorship as authorship
import kernel_measurement
import kernel_request_reprepare as reprepare


def test_new_capture_has_empty_slot_and_reprepare_accepts_it(tmp_path, monkeypatch):
    monkeypatch.delenv(authorship.MODE_ENV, raising=False)
    journal = kernel_measurement.AuditJournal(tmp_path, "host-authorship-slot", {})
    capture = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    assert "_host_authorship" in vars(capture)
    assert capture._host_authorship is None
    assert authorship.ledger_of(capture) is None
    # A capture that later holds a ledger is not an unknown ownership field for a
    # replacement capture: the slot exists on every fresh capture.
    capture._host_authorship = authorship.HostAuthorship(1, capture.request_id)
    fresh = kernel_measurement.RequestCapture(journal, 1, "snapshot")
    unknown = (set(vars(capture)) - set(vars(fresh)) - set(reprepare._STABLE)
               - reprepare._REGENERATED)
    assert "_host_authorship" not in unknown
    assert fresh._host_authorship is None
