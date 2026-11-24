"""Optional engine settings recover at launch and keep request-time reads pure."""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from kernel_continuation_state import ContinuationError
from kernel_empirical_fixes import read_dedup_observation
from kernel_native_prefix import delivered_source
from kernel_reread_fix import enabled as reread_fix_enabled, reread_fix_mode
from kernel_source_presentation import configuration, scope_labels_enabled
from stateframe import runner
from stateframe.endpoint import Endpoint


@pytest.mark.parametrize("view, expected", [("operational", "blocks"), ("existing", "json")])
def test_unknown_source_presentation_is_read_only_and_compatible(monkeypatch, capsys,
                                                                  view, expected) -> None:
    monkeypatch.setenv("STATEFRAME_SOURCE_PRESENTATION", "typo")
    monkeypatch.setenv("STATEFRAME_VIEW", view)

    assert configuration() == expected
    assert os.environ["STATEFRAME_SOURCE_PRESENTATION"] == "typo"
    assert capsys.readouterr().err == ""


def test_unknown_working_source_scope_is_read_only(monkeypatch, capsys) -> None:
    monkeypatch.setenv("STATEFRAME_WORKING_SOURCE_SCOPE", "typo")

    assert scope_labels_enabled() is False
    assert os.environ["STATEFRAME_WORKING_SOURCE_SCOPE"] == "typo"
    assert capsys.readouterr().err == ""


def test_unknown_reread_fix_is_read_only_and_uses_auto(monkeypatch, capsys) -> None:
    monkeypatch.setenv("STATEFRAME_REREAD_FIX", "typo")
    monkeypatch.setenv("STATEFRAME_SOURCE_LABELS", "plain")

    assert reread_fix_enabled() is True
    assert reread_fix_mode() == "auto"
    assert os.environ["STATEFRAME_REREAD_FIX"] == "typo"
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("setting", "fallback"), [
    ("STATEFRAME_SOURCE_PRESENTATION", "blocks"),
    ("STATEFRAME_WORKING_SOURCE_SCOPE", "0"),
    ("STATEFRAME_REREAD_FIX", "auto"),
])
def test_prepare_persists_and_reports_effective_optional_setting(
        monkeypatch, tmp_path, setting: str, fallback: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n", encoding="utf-8")
    notices: list[str] = []
    options = runner.RunOptions(
        repo=repo, task="Inspect sample.py", profile="r2hc3", max_turns=1,
        run_dir=tmp_path / "run",
        endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "scripted", None),
        extra_env={setting: "typo"},
        tool_settings=runner.ToolSettings(run_env=False),
    )

    run, _work, spec = runner.prepare(options, printer=notices.append)
    saved = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    resumed = runner._resume_spec(run, options.endpoint)

    assert spec["environment"][setting] == fallback
    assert saved["environment"][setting] == fallback
    assert resumed["environment"][setting] == fallback
    assert spec["environment_sources"][setting] == "runner:optional-setting-recovery"
    assert saved["environment_sources"][setting] == "runner:optional-setting-recovery"
    recovery_notes = [line for line in notices
                      if "recovered optional engine settings" in line.lower()]
    assert len(recovery_notes) == 1
    assert f"{setting}={fallback}" in recovery_notes[0]

    from kernel_model_policy import journal_effective_policy
    monkeypatch.setenv(setting, fallback)
    events = []
    policy_runtime = SimpleNamespace(
        model_name="scripted", served_model="scripted",
        declared_environment=spec["environment"],
        environment_sources=spec["environment_sources"],
        journal=SimpleNamespace(records=lambda: [],
                                emit=lambda kind, **fields: events.append((kind, fields))),
    )
    journal_effective_policy(policy_runtime)
    config = events[0][1]["effective_config"]
    assert config[setting] == {"value": fallback,
                               "source": "runner:optional-setting-recovery"}


def test_runtime_constructor_uses_optional_fallbacks_without_mutating_environment(
        tmp_path, monkeypatch, capsys, shipped_runtime_profile) -> None:
    from kernel_bolt import KernelBolt
    from kernel_stateframe_runtime import StateFrameRuntime
    from detreason.world_knowledge import WorldKnowledgeStore

    root = tmp_path / "repo"
    root.mkdir()
    target = root / "sample.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    owner = KernelBolt(root, "", "optional-fallbacks")
    owner.world = WorldKnowledgeStore(sandbox_id="optional-fallbacks")
    owner.task_row = {"msg": "Inspect sample.py", "files": {"sample.py": "VALUE = 1\n"}}
    owner.current_step = 1
    for name in ("STATEFRAME_SOURCE_PRESENTATION", "STATEFRAME_WORKING_SOURCE_SCOPE",
                 "STATEFRAME_REREAD_FIX"):
        monkeypatch.setenv(name, "typo")

    runtime = StateFrameRuntime(owner, output=tmp_path / "journal", target=target,
                                model="scripted", tools=[], temperature=0,
                                source_resolver=lambda _path, _symbol: None)

    assert runtime.source_presentation == "blocks"
    assert runtime.working_source_scope is False
    assert reread_fix_mode() == "auto"
    assert all(os.environ[name] == "typo" for name in (
        "STATEFRAME_SOURCE_PRESENTATION", "STATEFRAME_WORKING_SOURCE_SCOPE",
        "STATEFRAME_REREAD_FIX"))
    error_output = capsys.readouterr().err
    assert "invalid STATEFRAME_" not in error_output


def test_native_prefix_auto_fallback_verifies_source_revision(monkeypatch, capsys) -> None:
    path, source = "sample.py", "VALUE = 1\n"
    revision = "sha256:" + hashlib.sha256(source.encode()).hexdigest()
    messages = [{"role": "user", "content": f"[Current source: {path}]\n{source}"}]
    monkeypatch.setenv("STATEFRAME_SOURCE_LABELS", "plain")
    monkeypatch.setenv("STATEFRAME_REREAD_FIX", "typo")

    assert delivered_source(messages, path, revision) == source
    assert os.environ["STATEFRAME_REREAD_FIX"] == "typo"
    assert capsys.readouterr().err == ""
    with pytest.raises(ContinuationError, match="complete target source was not delivered"):
        delivered_source(messages, path, "sha256:" + "0" * 64)


@pytest.mark.parametrize(("mode", "labels", "enriched"), [
    ("typo", "plain", True),
    ("auto", "legacy", False),
    ("on", "legacy", True),
])
def test_read_dedup_caller_applies_auto_recovery_and_legacy_semantics(
        monkeypatch, capsys, mode: str, labels: str, enriched: bool) -> None:
    path, revision = "sample.py", "sha256:" + "a" * 64
    monkeypatch.setenv("STATEFRAME_READ_RECEIPT", "on")
    monkeypatch.setenv("STATEFRAME_TOKENLAB_SERVE_READS", "off")
    monkeypatch.setenv("STATEFRAME_TOKENLAB_STALL_BREAKER", "off")
    monkeypatch.setenv("STATEFRAME_SOURCE_LABELS", labels)
    monkeypatch.setenv("STATEFRAME_REREAD_FIX", mode)
    runtime = SimpleNamespace(
        frame_series=object(), empirical_fixes_mode="on",
        stateframe_tokenlab_stall_breaker_mode="off",
        _empirical_serialized_sources={path: {
            "full": True, "revision": revision, "request_hash": "request",
        }},
        _empirical_current_revisions={path: revision},
    )

    receipt = read_dedup_observation(runtime, path, 0)

    assert receipt["status"] == "READ_UNCHANGED"
    assert (receipt.get("edit_precondition_source") == "IN_FRAME") is enriched
    if mode == "typo":
        assert os.environ["STATEFRAME_REREAD_FIX"] == "typo"
        assert capsys.readouterr().err == ""
