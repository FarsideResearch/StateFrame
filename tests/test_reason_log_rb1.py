"""RB1 refusal/eviction/fallback log sites: one stderr line per cause per run, repeats stay silent."""
import io
import re

import pytest

import kernel_reason_log as rl
import kernel_request_diet as request_diet
import kernel_prompt_compact as prompt_compact
import kernel_skills as skills

LINE = re.compile(r"^\[stateframe\] (WARN|ERROR|INFO) (?P<component>[a-z_]+): .+", re.M)


@pytest.fixture(autouse=True)
def stream(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    out = io.StringIO()
    rl.configure(level="WARN", stream=out)
    yield out
    rl.reset_for_tests()


def _components(out):
    return [m.group("component") for m in LINE.finditer(out.getvalue())]


class _Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **payload):
        self.events.append((kind, payload))


def test_tokenizer_deadline_logs_once_per_run(stream):
    meter = request_diet.RequestTokenMeter(tokenizer=lambda _text: 1)
    meter._deadline = 0.0  # already expired: every measure raises before any tokenizer call
    for _ in range(2):
        with pytest.raises(TimeoutError):
            meter.measure({"messages": [], "tools": []})
    assert _components(stream) == ["request_diet"]
    assert "tokenizer request (deadline expired)" in stream.getvalue()
    assert rl.run_summary()["by_cause"]["timeout:tokenizer_deadline"]["count"] == 2


def test_calibrated_fallback_logs_schema_measure_once(stream):
    # No tokenizer and no endpoint: byte measuring is by design, so INFO (silent at the WARN default).
    from kernel_agent_schemas import measure_schemas
    measure_schemas()
    measure_schemas()
    assert "agent_schemas" not in stream.getvalue()
    summary = rl.run_summary()["by_cause"]["fallback:schema_measured_unconfigured"]
    assert summary["level"] == "INFO" and summary["count"] > 2
    rl.configure(level="INFO")
    rl.reset_for_tests()
    rl.configure(level="INFO", stream=stream)
    measure_schemas()
    lines = [line for line in stream.getvalue().splitlines() if "agent_schemas" in line]
    assert len(lines) == 1 and "INFO" in lines[0]
    assert "measured without tokenizer" in lines[0] and "calibrated_bytes" in lines[0]


def test_configured_tokenizer_failure_fallback_is_warn(stream):
    from kernel_agent_schemas import measure_schemas

    def broken(_text):
        raise OSError("down")
    measure_schemas(tokenizer=broken)
    lines = [line for line in stream.getvalue().splitlines() if "agent_schemas" in line]
    assert len(lines) == 1 and "WARN" in lines[0]


def test_skill_omission_names_the_dropped_skill(stream):
    removed = {"_request_budget_section": "skill_bodies",
               "content": "USER-PROVIDED SKILL INSTRUCTIONS: demo-skill (body text)"}
    messages = [{"role": "user", "content": "[StateFrame]\nstate"}]
    skills.note_budget_omission(messages, dict(removed))
    skills.note_budget_omission(messages, dict(removed))
    assert _components(stream) == ["skill_catalog"]
    assert "skill demo-skill body omitted from request" in stream.getvalue()
    assert rl.run_summary()["by_cause"]["eviction:skill_omitted"]["count"] == 2


def test_passive_withheld_drop_logs_once_with_count(stream, monkeypatch):
    monkeypatch.setattr(prompt_compact, "_PASSIVE_DROP_PENDING", [3])
    journal = _Journal()
    assert prompt_compact.emit_passive_withheld_dropped(journal, request_id="r1") is True
    prompt_compact._PASSIVE_DROP_PENDING[0] = 2
    assert prompt_compact.emit_passive_withheld_dropped(journal, request_id="r2") is True
    assert _components(stream) == []  # opt-in experiment mode: INFO, silent at the WARN default
    assert rl.run_summary()["by_cause"]["eviction:passive_withheld_dropped"]["level"] == "INFO"
    assert "withheld 3 passive rows" in rl.run_summary()["by_cause"]["eviction:passive_withheld_dropped"]["what"]
    assert len(journal.events) == 2


def test_dropped_samples_formats_path_and_range_and_caps_at_five():
    records = [{"path": f"src/f{i}.py", "start_line": i, "end_line": i + 9} for i in range(7)]
    text = request_diet._dropped_samples(records)
    assert text.startswith("src/f0.py:0-9, src/f1.py:1-10")
    assert "src/f4.py:4-13" in text and "src/f5.py" not in text
    assert request_diet._dropped_samples([]) is None


def test_diet_trim_logs_eviction_once_per_run(stream, monkeypatch):
    # Force the trimmed branch on a request that otherwise fits: the site under test is the log call.
    real_audit = request_diet._audit

    def trimmed_audit(*args, **kwargs):
        audit = real_audit(*args, **kwargs)
        audit["reason"] = "trimmed"
        return audit

    monkeypatch.setattr(request_diet, "_audit", trimmed_audit)
    for _ in range(2):
        body = {"messages": [{"role": "user", "content": "brief"}], "tools": []}
        request_diet.apply_diet(body, stage="FREE", max_request_bytes=100_000, token_budget=100_000,
                                token_meter=request_diet.RequestTokenMeter(tokenizer=lambda _text: 1))
    assert _components(stream) == ["request_diet"]
    assert rl.run_summary()["by_cause"]["eviction:request_diet_trim"]["count"] == 2


def test_diet_cap_infeasible_logs_refusal_with_reason(stream):
    body = {"messages": [{"role": "system", "content": "sys " * 50}] + [
        {"role": "tool", "content": "result %d " % i * 400} for i in range(6)] + [
        {"role": "user", "content": "go"}], "tools": []}
    with pytest.raises(request_diet.RequestDietDeferred):
        request_diet.apply_diet(body, stage="FREE", max_request_bytes=100_000, token_budget=300,
                                token_meter=request_diet.RequestTokenMeter(tokenizer=lambda t: max(1, len(t) // 4)))
    assert _components(stream) == ["request_diet"]
    assert "cap infeasible: non_dietable_request_content_exceeds_token_budget" in stream.getvalue()
    cause = "refusal:request_diet_cap_infeasible:non_dietable_request_content_exceeds_token_budget"
    assert rl.run_summary()["by_cause"][cause]["count"] == 1
