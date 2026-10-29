"""Hybrid "frame as compaction" arm (STATEFRAME_HYBRID_COMPACTION), rebuilt on frame epochs.

Behaviour
---------
* While shallow the request is exactly the H+state+ history request (this module does
  nothing but measure it).
* The crossing test is the FINAL rendered request estimate -- after every presentation
  transform, sidecar and the H+state+ tail -- against ``T = context - output_reserve``.
  A history request above T is never dispatched.
* On crossing, that same prepare() renders the F4 frame through the cache-stable epoch
  renderer (kernel_stateframe_epoch): stable block + one immutable snapshot + a
  chronological append-only log of real exchanges and state deltas + volatile tail.
  The runtime latches the frame for the rest of the episode, so the log keeps appending
  and the cached prefix is stable again; ordinary epoch rollovers handle later growth.
* If even a fresh frame does not fit T the request is refused with the TERMINAL overflow
  kind ``HYBRID_CONTEXT_OVERFLOW`` (the runner breaks on any non-RENDER_OVERFLOW kind),
  overriding any recoverable render overflow.

Why rebuilt: five hostile reviews rejected a bespoke compact-frame renderer (grounding,
receipts, source-span certification, get-sure scrub ordering, sidecar persistence, OFF
parity).  After crossing, the F4 renderer is the single owner of all of that and it is
already reviewed; this module only owns configuration, the crossing decision and its
typed event.
"""
from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping

MODE_ENV = "STATEFRAME_HYBRID_COMPACTION"
CONTEXT_ENV = "STATEFRAME_HYBRID_CONTEXT_TOKENS"
RESERVE_ENV = "STATEFRAME_HYBRID_OUTPUT_RESERVE"
SCHEMA = "stateframe-hybrid-compaction/2"
OVERFLOW_KIND = "HYBRID_CONTEXT_OVERFLOW"
DEFAULT_CONTEXT_TOKENS = 32768
DEFAULT_OUTPUT_RESERVE = 4096


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def _positive(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name, str(default))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclasses.dataclass(frozen=True)
class HybridConfig:
    context_tokens: int
    output_reserve: int

    @property
    def threshold(self) -> int:
        return self.context_tokens - self.output_reserve


def configuration(environ: Mapping[str, str] | None = None) -> HybridConfig:
    source = os.environ if environ is None else environ
    config = HybridConfig(_positive(source, CONTEXT_ENV, DEFAULT_CONTEXT_TOKENS),
                          _positive(source, RESERVE_ENV, DEFAULT_OUTPUT_RESERVE))
    if config.output_reserve >= config.context_tokens:
        raise ValueError(f"{RESERVE_ENV} must be smaller than {CONTEXT_ENV}")
    return config


def crossing_ceiling(*, frame_ceiling: int, config: HybridConfig) -> int:
    """Return the frame budget that owns rendering after the hybrid crossing.

    ``frame_ceiling`` is the shared pre-crossing budget.  A hybrid arm must be
    comparable with its H+state+ control using that same budget, while its
    post-crossing frame must fit the crossing threshold.  Keeping this clamp in
    the hybrid module gives the crossing rule one owner and leaves OFF/history
    paths untouched.
    """
    if frame_ceiling <= 0:
        raise ValueError("STATEFRAME_CEILING must be positive")
    return min(frame_ceiling, config.threshold)


def validate(*, config: HybridConfig, f4_mode: str, hstate_plus_history: bool,
             history_cap_enforced: bool, restart_enabled: bool, epoch_mode: str,
             frame_ceiling: int, runner_compaction: str = "off",
             think_prefill: str = "off") -> None:
    """Fail closed on every combination whose ownership would be ambiguous."""
    if f4_mode == "on":
        raise ValueError(f"{MODE_ENV}=on is a history arm; STATEFRAME_F4=on already owns the frame")
    if epoch_mode == "on":
        raise ValueError(f"{MODE_ENV}=on manages its own epochs; leave STATEFRAME_FRAME_EPOCH off")
    if not hstate_plus_history:
        raise ValueError(f"{MODE_ENV}=on requires the H+state+ history arm "
                         "(STATEFRAME_HSTATE_PLUS=on, render_selection not applied)")
    if history_cap_enforced:
        raise ValueError(f"{MODE_ENV}=on cannot share history with an enforcing history cap")
    if restart_enabled:
        raise ValueError(f"{MODE_ENV}=on does not support worker restart")
    if str(runner_compaction).strip().lower() in {"on", "1", "true", "yes"}:
        raise ValueError(f"{MODE_ENV}=on owns compaction; REACT_HISTORY_COMPACTION must be off")
    if str(think_prefill).strip().lower() != "off":
        # self_plan builds bodies AFTER measurement (review13 #2); on/user arm an assistant
        # prefill + continue_final_message transport metadata that the crossing frame (ending in
        # a user handoff) would silently invalidate (review15 #1).
        raise ValueError(f"{MODE_ENV}=on requires STATEFRAME_THINK_PREFILL=off")
    if frame_ceiling <= 0:
        raise ValueError("STATEFRAME_CEILING must be positive")


def crosses(estimate: int, config: HybridConfig) -> bool:
    return estimate > config.threshold


def overflow(estimate: int, config: HybridConfig, previous: Mapping[str, object] | None) -> dict:
    """Terminal overflow that always overrides a recoverable render overflow."""
    result: dict[str, object] = {"kind": OVERFLOW_KIND, "required_tokens": estimate,
                                 "available_tokens": config.threshold}
    if previous:
        result["superseded_overflow"] = dict(previous)
    return result


__all__ = [
    "CONTEXT_ENV", "DEFAULT_CONTEXT_TOKENS", "DEFAULT_OUTPUT_RESERVE", "HybridConfig", "MODE_ENV",
    "OVERFLOW_KIND", "RESERVE_ENV", "SCHEMA", "configuration", "crossing_ceiling", "crosses",
    "mode", "overflow", "validate",
]
