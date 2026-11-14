"""Subframe usage is derived from durable send and frame records."""
from __future__ import annotations

import json

import pytest

from stateframe.engine.kernel_parallel import ParallelError, _usage


def test_initial_frame_tokens_come_from_the_first_request_journal(tmp_path):
    run = tmp_path / "unit"
    journal = run / "worker" / "measurement-runs" / "journal-id" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({"kind": "unit_frame_tokens", "request": "first",
                                   "tokens": 1134, "counter": "tokenize"}) + "\n")
    (run / "sends.jsonl").write_text(json.dumps({"send": 1, "prompt_tokens": 1134,
                                                  "completion_tokens": 8}) + "\n")
    actual = _usage(run)
    assert actual.requests == 1
    assert actual.prompt_tokens == 1134
    assert actual.initial_frame_tokens == 1134


def test_resume_keeps_original_initial_frame_measurement(tmp_path):
    run = tmp_path / "unit"
    original = run / "attempts" / "1" / "worker" / "measurement-runs" / "first" / "events.jsonl"
    resumed = run / "worker" / "measurement-runs" / "second" / "events.jsonl"
    for path, tokens in ((original, 1007), (resumed, 1007)):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"kind": "unit_frame_tokens", "request": "first",
                                    "tokens": tokens, "counter": "tokenize"}) + "\n")
    assert _usage(run).initial_frame_tokens == 1007


def test_missing_first_request_measurement_refuses_usage_summary(tmp_path):
    run = tmp_path / "unit"
    run.mkdir()
    (run / "sends.jsonl").write_text(json.dumps({"send": 1, "prompt_tokens": 8,
                                                  "completion_tokens": 2}) + "\n")
    with pytest.raises(ParallelError, match="lack an initial frame-token measurement"):
        _usage(run)
