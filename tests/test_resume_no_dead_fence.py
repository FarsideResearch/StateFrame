"""A final continuation fence cannot be resumed as if it were an incomplete run."""
import json

import pytest

from stateframe import resume


def test_continuation_blocked_refuses_resume(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps({"schema": "stateframe-run/1",
        **{key: "x" for key in ("environment", "public_row", "parent_snapshot", "chat_url",
                               "model", "max_turns", "repo")}}))
    (run / "engine-result.json").write_text(json.dumps({
        "result": {"result": "CONTINUATION_BLOCKED"}}))
    with pytest.raises(resume.ResumeRefused, match="CONTINUATION_BLOCKED run cannot be resumed"):
        resume.assess(run)
