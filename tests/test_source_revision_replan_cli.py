"""CLI coverage for retries that follow successive prepared source revisions."""
from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from _journal import measurement_events
from engine_run import run_monitored
from tests.test_request_seal_wire_form import _run_scripted, assert_sealed_wire
from kernel_request_reprepare import same_source_revision_report


def _steps() -> list[dict[str, object]]:
    return [
        {"tool": "read", "arguments": {"path": "app.py"}},
        {"tool": "done", "arguments": {"summary": "VALUE is seven."}},
    ]


def test_same_source_revision_treats_two_unknown_sets_as_same() -> None:
    revisions = (("revisions", "app.py", "sha256:abc"),)
    assert same_source_revision_report(revisions, revisions)
    assert not same_source_revision_report(revisions, ())
    assert same_source_revision_report(None, None)
    assert not same_source_revision_report(revisions, None)


def _output(process: object, run: Path) -> str:
    output = str(getattr(process, "stdout", "")) + str(getattr(process, "stderr", ""))
    traceback = run / "child-traceback.txt"
    if traceback.is_file():
        output += traceback.read_text(encoding="utf-8")
    return output


@pytest.mark.host_sandbox
@pytest.mark.parametrize("revision_changes", [2, 3])
def test_cli_reprepares_each_new_source_revision(
    tmp_path: Path, revision_changes: int,
) -> None:
    process, run, script = _run_scripted(
        tmp_path, strict=True, steps=_steps(),
        source_revision_changes=revision_changes, write_source_during_tests=True,
        revision_comparison_mutation=os.environ.get(
            "STATEFRAME_TEST_REPLAN_REVISION_MUTATION",
        ),
    )
    output = _output(process, run)
    assert process.returncode == 0, output
    script.assert_consumed()

    events = measurement_events(run)
    started = [row for row in events
               if row.get("kind") == "source_revision_replan_started"]
    reprepared = [row for row in events
                  if row.get("kind") == "source_revision_reprepared"]
    submitted = [row for row in events if row.get("kind") == "request_submitted"]
    assert len(started) == len(reprepared) == revision_changes, output
    assert len({row["request_id"] for row in (*started, *reprepared)}) == 1, output
    replacement_ids = [row["replacement_request_id"] for row in reprepared]
    assert len(set(replacement_ids)) == revision_changes, output
    assert sum(row.get("request_id") == replacement_ids[-1] for row in submitted) == 1, output
    assert all(row.get("request_id") not in replacement_ids[:-1] for row in submitted), output

    observations = json.loads(
        (run / "source-replan-observations.json").read_text(encoding="utf-8"),
    )
    assert len(observations) == revision_changes + 1, output
    revisions = [row["revisions"]["app.py"] for row in observations]
    assert len(set(revisions)) == revision_changes + 1, output
    assert len({row["frame_digest"] for row in observations}) == revision_changes + 1, output
    assert len({row["allocation_digest"] for row in observations}) == revision_changes + 1, output
    assert not any(row.get("kind") == "source_revision_replan_refused" for row in events), output
    assert_sealed_wire(run, minimum=len(script.requests))


@pytest.mark.host_sandbox
def test_cli_stops_after_same_source_revision_is_reported_twice(tmp_path: Path) -> None:
    process, run, _script = _run_scripted(
        tmp_path, strict=True, steps=_steps(), repeat_same_revision=2,
        revision_comparison_mutation=os.environ.get(
            "STATEFRAME_TEST_REPLAN_REVISION_MUTATION",
        ),
    )
    output = _output(process, run)
    assert process.returncode != 0, output
    assert "source_revision_replan_exhausted" in output

    events = measurement_events(run)
    refused = [row for row in events
               if row.get("kind") == "source_revision_replan_refused"]
    assert len(refused) == 1, output
    assert refused[0].get("reason") == "revision_changed_again", output
    observations = json.loads(
        (run / "source-replan-observations.json").read_text(encoding="utf-8"),
    )
    assert len(observations) == 2, output
    assert observations[0]["revisions"] == observations[1]["revisions"]


@pytest.mark.host_sandbox
def test_cli_stops_after_one_retry_when_source_revisions_stay_unreadable(tmp_path: Path) -> None:
    process, run, _script = _run_scripted(
        tmp_path, strict=True, steps=_steps(), repeat_same_revision=9,
        unreadable_source_revisions=True,
        revision_comparison_mutation=os.environ.get(
            "STATEFRAME_TEST_REPLAN_REVISION_MUTATION",
        ),
    )
    output = _output(process, run)
    assert process.returncode != 0, output
    assert "source_revision_replan_exhausted" in output
    events = measurement_events(run)
    started = [row for row in events if row.get("kind") == "source_revision_replan_started"]
    refused = [row for row in events if row.get("kind") == "source_revision_replan_refused"]
    assert len(started) == 1, output
    assert [row.get("reason") for row in refused] == ["revision_changed_again"], output
    assert (run / "same-revision-report-count").read_text(encoding="utf-8") == "2"


@pytest.mark.host_sandbox
@pytest.mark.parametrize("revision_changes", [2, 3])
def test_cli_prepare_retries_each_new_source_revision(
    tmp_path: Path, revision_changes: int,
) -> None:
    process, run, script = _run_scripted(
        tmp_path, strict=True, steps=_steps(),
        prepare_source_revision_changes=revision_changes,
        write_source_during_tests=True,
        revision_comparison_mutation=os.environ.get(
            "STATEFRAME_TEST_REPLAN_REVISION_MUTATION",
        ),
    )
    output = _output(process, run)
    assert process.returncode == 0, output
    script.assert_consumed()
    observations = json.loads(
        (run / "prepare-replan-observations.json").read_text(encoding="utf-8"),
    )
    assert len(observations) == revision_changes + 1, output
    assert len({row["app.py"] for row in observations}) == revision_changes + 1, output


def test_prepare_stops_when_same_source_revision_repeats(monkeypatch: pytest.MonkeyPatch) -> None:
    import kernel_stateframe_runtime as runtime_module
    from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
    from kernel_request_reprepare import SourceRevisionReplanRefused

    revisions = (("revisions", "app.py", "sha256:abc"),)
    monkeypatch.setattr(runtime_module, "emit_reason", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "kernel_request_reprepare.source_replan_preparation",
        lambda *_args, **_kwargs: nullcontext(),
    )
    calls = 0

    def fail_without_progress(_self: object, *_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        error = FrameAllocationReplanNeeded("source proof snapshot revision changed")
        setattr(error, "_source_revision_set", revisions)
        raise error

    wrapped = runtime_module._reprepare_on_source_change(fail_without_progress)
    with pytest.raises(SourceRevisionReplanRefused) as stopped:
        wrapped(
            SimpleNamespace(stateframe_framefit_mode="on"),
            [{"role": "user", "content": "task"}],
            uncapped=[{"role": "user", "content": "task"}],
            tools=[],
        )
    assert stopped.value.stop_code == "source_revision_replan_exhausted"
    assert calls == 2


@pytest.mark.host_sandbox
def test_cli_resume_after_kill_between_reprepare_and_send_does_not_resend(
        tmp_path: Path,
) -> None:
    resources: dict[str, object] = {}
    process, run, script = _run_scripted(
        tmp_path, strict=True, steps=_steps(), source_revision_changes=1,
        write_source_during_tests=True, kill_after_source_reprepared=True,
        resources_out=resources,
    )
    server = resources["server"]
    wrapper_directory = resources["wrapper_directory"]
    try:
        marker = run / "killed-after-source-reprepared"
        assert marker.is_file()
        assert process.returncode != 0
        assert len(script.requests) == 1

        before_resume = measurement_events(run)
        started = [row for row in before_resume
                   if row.get("kind") == "source_revision_replan_started"]
        reprepared = [row for row in before_resume
                      if row.get("kind") == "source_revision_reprepared"]
        submitted = [row for row in before_resume
                     if row.get("kind") == "request_submitted"]
        assert len(started) == len(reprepared) == 1
        logical_request_id = started[0]["request_id"]
        assert reprepared[0]["request_id"] == logical_request_id
        replacement_id = reprepared[0]["replacement_request_id"]
        assert all(row.get("request_id") != replacement_id for row in submitted)
        assert len(submitted) == 1
        first_request_id = submitted[0]["request_id"]
        first_request_messages = script.requests[0]["messages"]

        endpoint = f"http://127.0.0.1:{getattr(server, 'server_port')}/v1"
        resumed = run_monitored(
            [
                sys.executable, "-m", "stateframe", "resume", str(run),
                "--endpoint", endpoint, "--model", script.model,
                "--max-turns", "32", "--quiet",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=resources["environment"], capture_output=True, text=True,
            expected_seconds=240, run_dir=run, check=False,
        )
        output = resumed.stdout + resumed.stderr
        assert resumed.returncode == 0, output
        script.assert_consumed()

        after_resume = measurement_events(run)
        submitted_after = [row for row in after_resume
                           if row.get("kind") == "request_submitted"]
        assert len(submitted_after) == len(script.requests), output
        assert sum(row.get("request_id") == first_request_id
                   for row in submitted_after) == 1, output
        assert sum(row.get("request_id") == logical_request_id
                   for row in submitted_after) == 1, output
        assert sum(row.get("request_id") == replacement_id
                   for row in submitted_after) == 0, output
        assert len({row.get("request_id") for row in submitted_after}) == len(submitted_after), output
        assert len(script.requests) > 1, output
        assert all(row["messages"] != first_request_messages for row in script.requests[1:]), output
        assert_sealed_wire(run, minimum=len(script.requests))
    finally:
        getattr(server, "shutdown")()
        getattr(server, "server_close")()
        getattr(wrapper_directory, "cleanup")()
