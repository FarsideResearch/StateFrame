"""STATEFRAME_HY_STEP_FIX: give the hy (H+state+ hybrid compaction) arm a live loop guard.

r46 item (f3_r44d_review.md #2c): hy hits REACT_STEPS=60 in 11/28 r44d runs (8 scored, all
FAIL) while f4 hits it 6/28 and hs 3/28.  ``kernel_tool_call_loop_breaker.ToolCallLoopBreaker``
already exists and is already wired into ``StateFrameRuntime.observe()`` (it fires whenever
``self._tool_call_loop_breaker`` is not None) -- but nothing in kernel_stateframe_runtime.py
ever *constructs* that instance and attaches it, on any arm.  ``self.stateframe_tool_call_
loop_breaker_mode`` only gates a post-hoc audit-summary journal event (_emit_tool_call_cap_
audit), never a live breaker.  r45's README confirms this was a deliberate, arm-wide decision
("R45_HISTORY is observational only; loop breaker pinned off") that the pod spec's
STATEFRAME_TOOL_CALL_LOOP_BREAKER env var enforces above the lane and that r46 must not
override (rule: do not change controls 381-383, i.e. do not touch what the shared breaker
switch does for any arm).

This module is the smallest fix that is still real: it attaches a *live* ToolCallLoopBreaker
to the runtime, but only when BOTH are true --

  * STATEFRAME_HY_STEP_FIX=on (this switch, default off, owned by r46 alone)
  * the runtime is actually the hy arm (STATEFRAME_HYBRID_COMPACTION=on -- per kernel_
    stateframe_hybrid_compaction's own docstring, "the H+state+ history request" is exactly
    what hy renders while shallow)

so f4 and hs (and hy itself with the switch off) are STATEFRAME_HY_STEP_FIX-off byte-identical:
``self._tool_call_loop_breaker`` stays unset, ``getattr(self, "_tool_call_loop_breaker", None)``
still returns None everywhere it is read, and every already-tested off-parity path is
untouched.  With the switch on for hy, ``observe()`` starts trip-detecting identical
non-productive tool-call streaks the same way the (pinned-off) shared breaker would have,
using the same threshold env (STATEFRAME_TOOL_CALL_LOOP_BREAKER_STREAK, default 5) so the cap
consistent-with-other-arms shape holds without inventing a second threshold knob.  The existing
render/consume sites (kernel_stateframe_runtime.py ~13987-13990, ~15186-15195) already turn a
tripped breaker into an early, receipted ``system_outcome = "tool_call_loop_breaker"`` stop
instead of grinding to REACT_STEPS=60 -- this module only supplies the missing live instance.
"""
from __future__ import annotations

import os

SWITCH = "STATEFRAME_HY_STEP_FIX"


def mode(environ: "os._Environ[str] | dict[str, str] | None" = None) -> str:
    """Return the strict switch mode; an invalid value is a configuration error."""
    source = environ if environ is not None else os.environ
    value = str(source.get(SWITCH, "off")).strip().lower()
    if value in {"off", "0", "false", "no", ""}:
        return "off"
    if value in {"on", "1", "true", "yes"}:
        return "on"
    raise ValueError(f"{SWITCH} must be on or off")


def enabled(environ: "os._Environ[str] | dict[str, str] | None" = None) -> bool:
    return mode(environ) == "on"


__all__ = ["SWITCH", "mode", "enabled"]
