from __future__ import annotations

from types import SimpleNamespace

import kernel_post_green_menu
from kernel_stateframe_runtime import StateFrameRuntime


SELECTORS = ("tests/test_r1.py", "tests/test_r2.py")


def _runtime(tmp_path, *, blockers=(), advisories=()):
    root = tmp_path / "workspace"
    root.mkdir()
    binding = {"product_revision": "current"}
    receipts = [
        {
            "producer": "test_subprocess",
            "receipt_id": f"receipt-{index}",
            "selectors": [selector],
            "binding": binding,
            "after_binding": binding,
            "binding_stable": True,
            "exit_code": 0,
            "sequence": index,
        }
        for index, selector in enumerate(SELECTORS, start=1)
    ]
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(
        root=root, f2p_ids=list(SELECTORS), task_row={}, current_step=1)
    runtime.progress_guard = False
    runtime.disposition_v2 = False
    runtime.stateframe_host_closure_mode = "on"
    runtime.stateframe_first_green_disclosure_mode = "off"
    runtime.stateframe_readiness_authority_mode = "on"
    runtime.stateframe_get_sure_mode = "off"
    runtime.stateframe_green_coherence_mode = "off"
    runtime.conclusion_cache_mode = "off"
    runtime.continuation = None
    runtime.test_receipts = receipts
    runtime.verdict_identity_provider = None
    runtime.binding = lambda: binding
    runtime.journal = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    runtime._readiness_request_decision = SimpleNamespace(
        blockers=tuple(blockers),
        advisory=tuple(advisories),
        repairable_advisories=tuple(
            row for row in advisories if row.get("repairable")),
    )
    return runtime


def _stabilize_post_green_state(monkeypatch):
    monkeypatch.setattr(kernel_post_green_menu, "pending_verification", lambda _runtime: False)
    monkeypatch.setattr(kernel_post_green_menu, "note_reused_verification",
                        lambda *_args: False)


def test_current_r1_proof_reuses_while_r2_blocker_stays_open(tmp_path, monkeypatch):
    _stabilize_post_green_state(monkeypatch)
    runtime = _runtime(tmp_path, blockers=({"check_id": SELECTORS[1]},))

    r1_result = runtime.redundant_verification(SELECTORS[0])
    r2_result = runtime.redundant_verification(SELECTORS[1])

    assert r1_result is not None
    assert r2_result is None


def test_current_proof_does_not_reuse_for_its_own_open_blocker(tmp_path, monkeypatch):
    _stabilize_post_green_state(monkeypatch)
    runtime = _runtime(tmp_path, blockers=({"check_id": SELECTORS[0]},))

    assert runtime.redundant_verification(SELECTORS[0]) is None


def test_current_proof_does_not_reuse_for_its_own_open_advisory(tmp_path, monkeypatch):
    _stabilize_post_green_state(monkeypatch)
    runtime = _runtime(
        tmp_path,
        advisories=({"check_id": SELECTORS[0], "repairable": False},),
    )

    assert runtime.redundant_verification(SELECTORS[0]) is None


def test_current_r1_proof_reuses_while_r2_advisory_stays_open(tmp_path, monkeypatch):
    _stabilize_post_green_state(monkeypatch)
    runtime = _runtime(
        tmp_path,
        advisories=({"check_id": SELECTORS[1], "repairable": True},),
    )

    assert runtime.redundant_verification(SELECTORS[0]) is not None
