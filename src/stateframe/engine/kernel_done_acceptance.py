"""ONE accessor for "has this run's done been accepted" (b1426).

Two authorities record acceptance on the runtime/measurement: the completion reducer sets
``completion_accepted``, the legacy disposition-v2 path sets ``v2_accepted_done``. Readers that consulted
only one of them dropped a sealed, accepted done (b1426: reducer ACCEPTED+sealed, terminal relabelled it
"step limit without accepted done"). Every reader goes through done_accepted(); the source scan
done_acceptance_scan.py (guard probes/test_done_acceptance_single_accessor_unittest.py) fails on any raw read
elsewhere. Writers keep setting their own flag; withdrawal (repair_blocks_done) clears both.
"""
from __future__ import annotations


def done_accepted(runtime: object) -> bool:
    """True when the immutable done seal or a legacy authority accepted this run."""
    from kernel_done_is_submit_authority import done_sealed, enabled as done_submit_enabled

    if (done_sealed(runtime) is not None
            or getattr(runtime, "completion_accepted", False)
            or getattr(runtime, "v2_accepted_done", False)):
        return True
    # Under STATEFRAME_DONE_IS_SUBMIT the seal is written in finish (after pending mutations settle),
    # but the v2 terminal is decided before that, so a MODEL done read as "step limit without accepted done"
    # (r2GB DONE-env smoke: done_submit_requested model seq 932, v2_terminal seq 940, done_sealed seq 953).
    # An accepted, pending MODEL submission is an accepted done. Host-sourced pending never counts.
    try:
        if not done_submit_enabled(runtime):
            return False
    except ValueError:
        return False
    pending = getattr(runtime, "_done_submit_pending", None)
    return bool(pending is not None
                and getattr(pending, "source", None) == "model"
                and getattr(pending, "accepted", False))


__all__ = ["done_accepted"]
