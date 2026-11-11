"""Host-only diagnostics for mutation hook timing."""
from __future__ import annotations

import os
import sys
import time


def _hook_timing(event: str, *, at: float | None = None) -> None:
    if os.environ.get("STATEFRAME_DEBUG_HOOK_TIMING") == "1":
        try:
            print(f"[hook-timing] {time.monotonic() if at is None else at:.9f} {event}",
                  file=sys.stderr, flush=True)
        except OSError:
            pass  # Diagnostics must never change hook execution or containment.
