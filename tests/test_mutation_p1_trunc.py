"""Regression for mutation output truncation reported by the execution session."""
from dataclasses import replace

import pytest

from mutate_leftovers import make_launch_spec, mutation_root, stage_test_unit, sweep_and_assert


@pytest.mark.host_sandbox
def test_mutation_reports_session_output_truncation_as_stdout_overflow(
        mutation_root, monkeypatch):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    staged = stage_test_unit(build, mutation_root,
                             "def handle(input, ctx):\n    return {'ok': True}\n")
    executor = execution.MutationExecutor(mutation_root, network_envelope="none")
    spec = make_launch_spec(execution, wire, staged)
    collect = executor._session._collect

    def truncated_collect(handle, cap):
        result = collect(handle, cap)
        assert result.cleanup.ok
        assert result.output_total_bytes < spec.limits.output_bytes
        return replace(result, output_truncated=True)

    monkeypatch.setattr(executor._session, "_collect", truncated_collect)
    try:
        outcome = executor.run(spec)
        assert outcome.exit_code == 0
        assert outcome.stdout_overflow is True
    finally:
        assert executor.close().ok
        sweep_and_assert(mutation_root)
