"""A source-read detour child must be stored with its own acceptance, never the parent's."""
from __future__ import annotations

from types import SimpleNamespace

from kernel_continuation_evidence import binding_versions
from kernel_continuation_executive import ContinuationExecutive
from kernel_continuation_journal import ContinuationJournal
from kernel_continuation_runtime import LiveContinuation

PARENT_ACCEPTANCE = {
    "kind": "no_public_checks",
    "selectors": ["tests/test_value.py::test_value"],
}


def _binding() -> dict:
    return {
        "product_files": {"pkg/value.py": "sha256:source"},
        "harness_files": {},
    }


def _live_continuation(tmp_path, engine: ContinuationExecutive, binding: dict) -> LiveContinuation:
    """Build the production LiveContinuation around a real engine and journal."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "value.py").write_text("VALUE = 1\n")

    live = LiveContinuation.__new__(LiveContinuation)
    live.engine = engine
    live.journal = engine.journal
    live.guarded_effects = None
    live.dispatch = {
        "tool": "continuation_detour",
        "accepted": True,
        "event_id": "dispatch-1",
        "continuation_id": "root",
    }
    live.runtime = SimpleNamespace(
        owner=SimpleNamespace(
            root=str(root),
            current_step=1,
            receipt=lambda *args, **kwargs: None,
        ),
        binding=lambda: binding,
        step_limit=None,
        current_request_id="request-1",
        native_batch=None,
    )
    return live


def test_detour_child_gets_its_own_acceptance_and_parent_keeps_its_own(tmp_path):
    journal = ContinuationJournal(tmp_path, "detour-acceptance", {"origin": "test"})
    try:
        engine = ContinuationExecutive(journal)
        binding = _binding()
        engine.apply(
            "create", new_id="root", objective="finish value",
            pending_decision="decide value", dependencies=binding_versions(binding),
            turns=3, acceptance=dict(PARENT_ACCEPTANCE),
        )
        assert engine.choose_next(binding) == "root"

        live = _live_continuation(tmp_path, engine, binding)
        feedback = live.operation("continuation_detour", {
            "path": "pkg/value.py", "start_line": 1, "end_line": 1, "reason": "need source",
            "pending_decision": "read the value source",
        })

        assert feedback is not None
        assert "Detour rejected" not in feedback
        children = [item for item in engine.state.intentions.values()
                    if item.objective.startswith("Acquire source:")]
        assert len(children) == 1
        assert children[0].acceptance == {
            "kind": "source_read", "path": "pkg/value.py",
            "start_line": 1, "end_line": 1,
        }
        assert engine.state.intentions["root"].acceptance == PARENT_ACCEPTANCE
    finally:
        journal.close()
