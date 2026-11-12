"""Name exactly what ended a check process, from evidence, never by guessing.

Of the limits StateFrame sets on a check, only RLIMIT_CPU can end it by signal: the soft
limit sits one second under the hard one, so a runaway first dies by SIGXCPU and only a
process that ignores SIGXCPU reaches the hard SIGKILL. RLIMIT_DATA makes allocation fail
(ENOMEM) and RLIMIT_FSIZE sends SIGXFSZ. A SIGKILL that none of our limits explains came
from outside the check's own policy and is reported as such, with the CPU it had used.
"""
from __future__ import annotations

import signal
from collections.abc import Mapping
from dataclasses import dataclass

# Initiators whose occurrence depends on the host (load, other engines), not on the check.
HOST_DEPENDENT_INITIATORS = frozenset({
    "aggregate_rss_supervisor", "resident_memory_supervisor_error",
    "external_sigkill_unattributed",
})


@dataclass(frozen=True)
class Termination:
    initiator: str
    # The exact limit that ended the process ("RLIMIT_CPU=3600s"), when one did.
    limit: str | None
    detail: str

    def as_fields(self) -> dict[str, object]:
        return {"likely_initiator": self.initiator, "termination_limit": self.limit,
                "termination_detail": self.detail}


def signal_of(exit_code: object) -> int | None:
    """The signal number behind an exit status (negative, or 128+N from a wrapper)."""
    if type(exit_code) is not int:
        return None
    if exit_code < 0:
        return -exit_code
    if 128 < exit_code < 128 + 65:
        return exit_code - 128
    return None


def _seconds(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and value >= 0 else None


def attribute_termination(exit_code: object, *, memory_limit_killed: bool = False,
                          memory_limit_reason: object = None, stop_cause: object = None,
                          cpu_seconds_used: object = None,
                          cpu_limit_seconds: object = None) -> Termination | None:
    """Return what ended the check, or None for an ordinary exit."""
    if exit_code == 124:
        return Termination("wall_timeout", None, "the check exceeded its wall-clock bound")
    number = signal_of(exit_code)
    cpu_limit = _seconds(cpu_limit_seconds)
    cpu_used = _seconds(cpu_seconds_used)
    limit_name = f"RLIMIT_CPU={cpu_limit:g}s" if cpu_limit is not None else "RLIMIT_CPU"
    if memory_limit_killed:
        reason = memory_limit_reason or "aggregate resident-memory limit exceeded"
        return Termination("aggregate_rss_supervisor", "aggregate resident memory", str(reason))
    if isinstance(stop_cause, str) and stop_cause.startswith("supervisor_error"):
        return Termination("resident_memory_supervisor_error", None,
                           f"the resident-memory supervisor stopped the check: {stop_cause}")
    if number == signal.SIGXCPU:
        return Termination("rlimit_cpu_soft", limit_name,
                           f"CPU time reached the soft limit ({_cpu_text(cpu_used, cpu_limit)})")
    if number == signal.SIGXFSZ:
        return Termination("rlimit_fsize", "RLIMIT_FSIZE",
                           "a file write exceeded the check's file-size limit")
    if number != signal.SIGKILL:
        return None
    if cpu_used is not None and cpu_limit is not None and cpu_used >= cpu_limit - 1:
        return Termination("rlimit_cpu_hard", limit_name,
                           f"CPU time reached the hard limit after SIGXCPU was ignored "
                           f"({_cpu_text(cpu_used, cpu_limit)})")
    return Termination("external_sigkill_unattributed", None,
                       f"no StateFrame limit reached: {_cpu_text(cpu_used, cpu_limit)}")


_EVIDENCE_KEYS = frozenset({"memory_limit_reason", "stop_cause", "cpu_seconds_used",
                            "cpu_limit_seconds"})


def receipt_fields(exit_code: object, *, memory_limit_killed: bool = False,
                   evidence: Mapping[str, object] | None = None) -> dict[str, object]:
    """Fields for a test-completion receipt: the initiator, and the exact limit when one hit."""
    known = {key: value for key, value in (evidence or {}).items() if key in _EVIDENCE_KEYS}
    termination = attribute_termination(
        exit_code, memory_limit_killed=memory_limit_killed, **known)
    if termination is None:
        return {"likely_initiator": None}
    return termination.as_fields()


def _cpu_text(used: float | None, limit: float | None) -> str:
    used_text = f"{used:.2f}s" if used is not None else "unknown"
    limit_text = f"{limit:g}s" if limit is not None else "unknown"
    return f"cpu {used_text} of {limit_text}"


def model_facing_kill_line(termination: Termination | None, signal_number: int) -> str:
    """What the model is told about a signal exit. Only a named limit is called one."""
    if termination is not None and termination.limit is not None:
        return (f"CHECK PROCESS STOPPED BY {termination.limit} (signal {signal_number}): "
                f"{termination.detail}. Not a test result; nothing about your patch was "
                "observed. Narrow the selector or remove the runaway work.")
    return (f"CHECK PROCESS KILLED FROM OUTSIDE THE CHECK (signal {signal_number}); no limit of "
            "this check was reached. Not a test result; the output above is what it printed "
            "before the kill. Re-run the same check.")
