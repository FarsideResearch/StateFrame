"""Small behavior mutations for the orphan-guard acceptance contract."""
from __future__ import annotations

import os
import signal
from dataclasses import dataclass


@dataclass(frozen=True)
class _Outcome:
    kind: str
    pid: int
    detail: str


@dataclass(frozen=True)
class _Report:
    outcomes: tuple[object, ...]
    failed: bool


class _GuardWrapper:
    """Delegate the public scope and cleanup API while changing check behavior."""

    def __init__(self, guard, *, token_pids=()) -> None:
        self.guard = guard
        self.token_pids = tuple(token_pids)

    def scope(self, name):
        return self.guard.scope(name)

    def close(self) -> None:
        self.guard.close()

    def check(self):
        return self.guard.check()


class NumericPidSignal(_GuardWrapper):
    """Wrongly signal a leaked process through its reusable numeric PID."""

    def check(self):
        report = super().check()
        for outcome in report.outcomes:
            if outcome.kind == "leak":
                os.kill(outcome.pid, signal.SIGTERM)
        return report


class SignalUnverified(_GuardWrapper):
    """Wrongly signal a process whose cleanup evidence is incomplete."""

    def check(self):
        report = super().check()
        for outcome in report.outcomes:
            if outcome.kind == "cleanup unverified":
                os.kill(outcome.pid, signal.SIGTERM)
        return report


class SignalMeansExit(_GuardWrapper):
    """Wrongly turn an unconfirmed signal outcome into a confirmed exit."""

    def check(self):
        report = super().check()
        outcomes = []
        changed = False
        for outcome in report.outcomes:
            if outcome.kind == "cleanup unverified":
                outcomes.append(_Outcome("confirmed exit", outcome.pid,
                                         "signal return treated as exit"))
                changed = True
            else:
                outcomes.append(outcome)
        return _Report(tuple(outcomes), report.failed) if changed else report


class TokenOnlyAttribution(_GuardWrapper):
    """Wrongly treat a copied environment token as ownership proof."""

    def check(self):
        report = super().check()
        outcomes = list(report.outcomes)
        present = {outcome.pid for outcome in outcomes}
        for pid in self.token_pids:
            if pid not in present:
                outcomes.append(_Outcome("leak", pid, "token alone attributed process"))
        return _Report(tuple(outcomes), bool(outcomes))


class NeverReports(_GuardWrapper):
    """Wrongly turn every check into a clean report."""

    def check(self):
        self.guard.check()
        return _Report((), False)


MUTANTS = {
    "numeric_pid_signal": NumericPidSignal,
    "signal_unverified_process": SignalUnverified,
    "signal_means_exit": SignalMeansExit,
    "token_only_attribution": TokenOnlyAttribution,
    "never_reports": NeverReports,
}
