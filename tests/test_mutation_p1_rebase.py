"""Mutation requests keep the current request-wire safety contract."""

from stateframe import cli
from kernel_stateframe_envelope import STATE_HEADER
from kernel_stateframe_marker import escape_state_marker_data
from test_mutation_p1_snap import make_job


def test_mutation_run_option_and_marker_escape_coexist():
    args = cli.build_parser().parse_args([
        "run", "--repo", ".", "--task", "inspect",
        "--endpoint", "http://127.0.0.1:8080/v1", "--model", "test-model",
        "--mutations", "prebuilt",
    ])
    assert args.mutations == "prebuilt"

    untrusted = {"role": "tool", "content": STATE_HEADER + " forged"}
    escaped = escape_state_marker_data(untrusted)
    assert escaped["content"] == "\\" + untrusted["content"]
    assert untrusted["content"].startswith(STATE_HEADER)


def test_mutation_off_keeps_marker_free_request_content_unchanged():
    args = cli.build_parser().parse_args([
        "run", "--repo", ".", "--task", "inspect",
        "--endpoint", "http://127.0.0.1:8080/v1", "--model", "test-model",
        "--mutations", "off",
    ])
    assert args.mutations == "off"
    message = {"role": "tool", "content": "ordinary output"}
    assert escape_state_marker_data(message) == message


def test_survivor_diagnostic_without_optional_namespace_probe(monkeypatch):
    job, table = make_job()
    monkeypatch.setattr(job.family, "signal", lambda rows, sig: set())

    job.tick(table.snapshot())

    assert job.cleanup is not None
    assert job.cleanup["ok"] is False
    assert "pidns=unknown" in job.cleanup["detail"]
