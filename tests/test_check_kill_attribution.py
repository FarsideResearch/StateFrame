"""A killed check names the exact limit that killed it, or says no limit of ours did.

A public check was SIGKILLed 0.9 s after it started (rc 137) and its receipt guessed
"sandbox_rlimit_cpu_or_memory". The sender was another process, not any limit of the check.
Only RLIMIT_CPU can end a check by signal, and with the soft limit one second under the hard
one it does so by SIGXCPU. RLIMIT_CPU is a generous runaway backstop, not a budget.
"""
from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest

import kernel_execution_policy as policy
from kernel_kill_attribution import (
    attribute_termination, model_facing_kill_line, receipt_fields)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("exit_code", [137, -9])
def test_sigkill_under_the_cpu_limit_is_external_with_cpu_used(exit_code):
    fields = receipt_fields(exit_code, evidence={"cpu_seconds_used": 0.41,
                                                 "cpu_limit_seconds": 3600.0})
    assert fields == {"likely_initiator": "external_sigkill_unattributed",
                      "termination_limit": None,
                      "termination_detail": "no StateFrame limit reached: cpu 0.41s of 3600s"}


def test_sigkill_with_unknown_cpu_is_still_never_called_an_rlimit():
    termination = attribute_termination(137)
    assert termination.initiator == "external_sigkill_unattributed"
    assert termination.limit is None
    assert termination.detail == "no StateFrame limit reached: cpu unknown of unknown"


def test_sigkill_at_the_cpu_limit_is_the_hard_rlimit():
    termination = attribute_termination(137, cpu_seconds_used=3599.6, cpu_limit_seconds=3600)
    assert termination.initiator == "rlimit_cpu_hard"
    assert termination.limit == "RLIMIT_CPU=3600s"


def test_sigxcpu_names_the_soft_cpu_limit_exactly():
    termination = attribute_termination(152, cpu_seconds_used=1.0, cpu_limit_seconds=2)
    assert (termination.initiator, termination.limit) == ("rlimit_cpu_soft", "RLIMIT_CPU=2s")
    assert termination.detail == "CPU time reached the soft limit (cpu 1.00s of 2s)"


def test_sigxfsz_names_the_file_size_limit():
    termination = attribute_termination(-signal.SIGXFSZ)
    assert (termination.initiator, termination.limit) == ("rlimit_fsize", "RLIMIT_FSIZE")


def test_rss_kill_and_supervisor_error_are_named_from_their_evidence():
    rss = attribute_termination(137, memory_limit_killed=True,
                                memory_limit_reason="resident 900 MB over 512 MB")
    assert rss.as_fields() == {"likely_initiator": "aggregate_rss_supervisor",
                               "termination_limit": "aggregate resident memory",
                               "termination_detail": "resident 900 MB over 512 MB"}
    error = attribute_termination(137, stop_cause="supervisor_error: procfs unreadable")
    assert error.initiator == "resident_memory_supervisor_error"
    assert "supervisor_error: procfs unreadable" in error.detail


def test_wall_timeout_and_ordinary_exits():
    assert attribute_termination(124).initiator == "wall_timeout"
    for code in (0, 1, 2, 5, None, "137"):
        assert attribute_termination(code) is None
    assert receipt_fields(1) == {"likely_initiator": None}


def test_receipt_fields_ignore_unknown_evidence_keys():
    fields = receipt_fields(137, evidence={"cpu_seconds_used": 0.2, "cpu_limit_seconds": 10,
                                           "injected": "x"})
    assert fields["likely_initiator"] == "external_sigkill_unattributed"


def test_model_is_told_a_limit_only_when_one_was_reached():
    external = model_facing_kill_line(attribute_termination(137, cpu_seconds_used=0.4,
                                                            cpu_limit_seconds=3600), 9)
    assert "KILLED FROM OUTSIDE THE CHECK" in external
    assert "no limit of this check was reached" in external
    assert "resource limit or timeout" not in external
    named = model_facing_kill_line(attribute_termination(152, cpu_limit_seconds=3600), 24)
    assert named.startswith("CHECK PROCESS STOPPED BY RLIMIT_CPU=3600s (signal 24)")


def test_cpu_backstop_defaults_to_an_hour_and_follows_the_policy(monkeypatch):
    monkeypatch.delenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", raising=False)
    assert policy.check_cpu_backstop_seconds() == 3600.0
    monkeypatch.setenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", "120")
    assert policy.check_cpu_backstop_seconds() == 120.0
    for bad in ("nope", "-5", "0", "nan", "inf"):
        monkeypatch.setenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", bad)
        assert policy.check_cpu_backstop_seconds() == 3600.0
    monkeypatch.setenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", "1e9")
    assert policy.check_cpu_backstop_seconds() == policy.MAX_CPU_SECONDS


def test_shipped_default_profile_uses_the_backstop_not_a_budget():
    profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
    assert profile["switches"]["STATEFRAME_PUBLIC_CHECK_CPU_SECONDS"] == "3600"


def test_host_dependent_kills_do_not_change_wire_identity():
    from kernel_wire_identity import WIRE_AUDIT_ONLY_FIELDS, wire_identity_field_value

    for initiator in ("external_sigkill_unattributed", "resident_memory_supervisor_error"):
        assert wire_identity_field_value("likely_initiator", initiator) is None
    assert wire_identity_field_value("likely_initiator", "rlimit_cpu_soft") == "rlimit_cpu_soft"
    assert {"termination_limit", "termination_detail"} <= WIRE_AUDIT_ONLY_FIELDS


def _sandboxed(source: str, tmp_path: Path, **limits):
    try:
        return policy.run_sandboxed((sys.executable, "-c", source), tmp_path,
                                    timeout=60, **limits)
    except policy.ExecutionPolicyError as refusal:
        pytest.skip(f"sandbox unavailable here: {refusal}")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux RLIMIT_CPU signals")
@pytest.mark.parametrize(("limit", "signal_number", "initiator"), [
    # At 1 s soft and hard coincide, so the kernel's signal is SIGKILL; the CPU spent names it.
    (1, signal.SIGKILL, "rlimit_cpu_hard"),
    (2, signal.SIGXCPU, "rlimit_cpu_soft"),
])
def test_real_runaway_is_stopped_by_rlimit_cpu_and_named_exactly(
        tmp_path, limit, signal_number, initiator):
    result = _sandboxed("while True:\n    pass\n", tmp_path, cpu_seconds=limit)
    code = result.completed.returncode
    assert code in (-signal_number, 128 + signal_number), code
    assert result.cpu_limit_seconds == float(limit)
    # Sampled every 20 ms, so the last reading can trail the kill by a tick or two.
    assert result.cpu_seconds_used is not None and result.cpu_seconds_used >= limit - 1.2
    termination = attribute_termination(code, cpu_seconds_used=result.cpu_seconds_used,
                                        cpu_limit_seconds=result.cpu_limit_seconds)
    assert (termination.initiator, termination.limit) == (initiator, f"RLIMIT_CPU={limit}s")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux procfs CPU times")
def test_real_external_sigkill_is_reported_unattributed_with_cpu_used(tmp_path):
    result = _sandboxed("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n", tmp_path,
                        cpu_seconds=policy.check_cpu_backstop_seconds())
    code = result.completed.returncode
    assert code in (-signal.SIGKILL, 128 + signal.SIGKILL), code
    assert isinstance(result.cpu_seconds_used, float) and result.cpu_seconds_used < 30
    fields = receipt_fields(code, evidence={"cpu_seconds_used": result.cpu_seconds_used,
                                            "cpu_limit_seconds": result.cpu_limit_seconds})
    assert fields["likely_initiator"] == "external_sigkill_unattributed"
    assert fields["termination_detail"].endswith(" of 3600s")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux RLIMIT_CPU")
def test_real_check_sees_soft_limit_one_second_under_hard(tmp_path):
    result = _sandboxed("import resource\nprint(*resource.getrlimit(resource.RLIMIT_CPU))\n",
                        tmp_path, cpu_seconds=3600)
    assert result.completed.returncode == 0, result.completed.stderr
    soft, hard = map(int, result.completed.stdout.split())
    assert (soft, hard) == (3599, 3600) or (hard < 3600 and soft == max(1, hard - 1))


def test_ignored_sigxcpu_still_dies_at_the_hard_limit_by_name():
    # A check that ignores SIGXCPU reaches the hard limit and is SIGKILLed with its CPU spent.
    termination = attribute_termination(-9, cpu_seconds_used=2.4, cpu_limit_seconds=3)
    assert termination.limit == "RLIMIT_CPU=3s"
