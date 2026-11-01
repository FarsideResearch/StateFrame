import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_hooks import PanelEntry, create_bundle, event_id, load_hook_files
from stateframe import hook_host, runner, workcopy
from stateframe.endpoint import Endpoint


@pytest.fixture(autouse=True)
def _unlock_hook_bundles_for_tmp_cleanup(tmp_path):
    yield
    for bundle in tmp_path.rglob("hook-bundle"):
        if bundle.is_symlink():
            continue
        for path in bundle.rglob("*"):
            if path.is_symlink():
                continue
            path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o600))
        bundle.chmod(bundle.stat().st_mode | 0o700)


def _repo(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "sample.py").write_text("value = 1\n", encoding="utf-8")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text("def test_sample():\n    assert True\n", encoding="utf-8")
    return root


def _hook_config(root: Path) -> Path:
    folder = root / "hook-source"
    folder.mkdir()
    config = folder / "hooks.json"
    config.write_text(json.dumps({
        "schema": "stateframe-hooks/1",
        "hooks": [
            {"name": "start", "event": "session_start",
             "argv": [sys.executable, "-c", "print('welcome')"]},
            {"name": "guard", "event": "prompt_submit",
             "argv": [sys.executable, "-c", "pass"]},
        ],
    }), encoding="utf-8")
    return config


def _options(repo: Path, run_dir: Path, *, hooks=(), in_place=False) -> runner.RunOptions:
    return runner.RunOptions(
        repo=repo,
        task="update sample",
        endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        max_turns=1,
        test_cmd="true",
        checks=["tests/test_sample.py"],
        target="sample.py",
        run_dir=run_dir,
        hooks_files=tuple(hooks),
        in_place=in_place,
    )


class _FakeHookRunner:
    deny_prompt = False

    def __init__(self, run_dir, hooks, **kwargs):
        self.run_dir = Path(run_dir)
        self.hooks = hooks
        self.run_seed = kwargs["run_seed"]
        self.tape = hook_host.HookTape(self.run_dir)
        self.closed = False

    def run_event(self, event, *, boundary, **_kwargs):
        indexed = next(((index, hook) for index, hook in enumerate(self.hooks.definitions)
                        if hook.event == event), None)
        if indexed is None:
            return SimpleNamespace(runs=(), entries=(), decision="continue", deny_code=None,
                                   deny_hook=None, reason="")
        index, definition = indexed
        hook = definition.name
        record_id = event_id(self.run_seed, boundary, index)
        identity = {"event_id": record_id, "hook": hook, "event": event}
        self.tape.intent(record_id, identity)
        denied = event == "prompt_submit" and self.deny_prompt
        context = ("welcome" if event == "session_start" else
                   "prompt context" if event == "prompt_submit" else "")
        outcome = {"decision": "deny" if denied else "continue", "reason": "blocked" if denied else "",
                   "context": context}
        self.tape.complete(record_id, {"outcome": outcome})
        records = self.run_dir / "hooks" / "records.jsonl"
        records.parent.mkdir(parents=True, exist_ok=True)
        with records.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"hook": hook, "event": event,
                                     "decision": outcome["decision"], "error_code": None}) + "\n")
        entries = ([PanelEntry("start", hook, event, "", context)]
                   if event in ("session_start", "prompt_submit") and context else [])
        return SimpleNamespace(runs=(), entries=entries,
                               decision=outcome["decision"], deny_code="HOOK_DENIED" if denied else None,
                               deny_hook=hook if denied else None, reason=outcome["reason"])

    def close(self):
        self.closed = True


def test_prepare_load_errors_are_run_errors(tmp_path):
    repo = _repo(tmp_path / "repo")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    options = _options(repo, tmp_path / "run", hooks=(invalid,))
    with pytest.raises(runner.RunError, match="JSON|json|invalid"):
        runner.prepare(options, printer=lambda *_: None)
    assert not options.run_dir.exists()


def test_prepare_refuses_run_directory_inside_hook_read_roots(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    config = _hook_config(tmp_path)
    protected = tmp_path / "protected"
    protected.mkdir()
    run_dir = protected / "run"
    monkeypatch.setattr(hook_host.HookSet, "read_roots", lambda _self: (protected,))
    with pytest.raises(runner.RunError, match="under hook read root"):
        runner.prepare(_options(repo, run_dir, hooks=(config,)), printer=lambda *_: None)
    assert not run_dir.exists()


def test_no_hooks_prepare_has_no_hook_spec_or_files(tmp_path):
    repo = _repo(tmp_path / "repo")
    run_dir = tmp_path / "run"
    run, work, spec = runner.prepare(_options(repo, run_dir), printer=lambda *_: None)
    assert "hooks" not in spec
    assert json.loads((run / "spec.json").read_text(encoding="utf-8")) == spec
    assert set(spec) == {
        "schema", "created", "engine_dir", "engine_source", "profile", "capabilities", "chat_url",
        "model", "api_key_env", "max_turns", "parent_snapshot", "public_row", "tool_settings",
        "custom_tools_digest", "custom_executables", "environment", "environment_sources",
        "prompt_cache_slot", "test_cmd", "run_identity_seed", "repo", "in_place",
        # Since 0.2.4 every run pins its (possibly empty) skill selection.
        "skills", "skills_digest",
    }
    assert json.loads((run / "tool-settings.json").read_text(encoding="utf-8")) == spec["tool_settings"]
    assert not any((run / name).exists() for name in ("hooks", "hook-bundle", "hook-tape.jsonl"))
    files = {path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file()}
    expected = {"spec.json", "tool-settings.json", "skill-settings.json"}
    for base in ("workspace", "parent"):
        expected.update(f"{base}/{rel}" for rel in ("sample.py", "tests/test_sample.py"))
    assert files == expected
    assert work.copy == run / "workspace"


def test_untrusted_project_file_only_prints_notice_and_does_not_enable_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = _repo(tmp_path / "repo")
    project = repo / ".stateframe"
    project.mkdir()
    (project / "context.py").write_text("print('must not run')\n", encoding="utf-8")
    (project / "hooks.json").write_text(json.dumps({
        "schema": "stateframe-hooks/1",
        "hooks": [{"name": "project", "event": "session_start",
                   "argv": [sys.executable, "context.py"], "files": ["context.py"]}],
    }), encoding="utf-8")
    output = []
    run_dir, _, spec = runner.prepare(_options(repo, tmp_path / "run"), printer=output.append)
    assert "hooks" not in spec
    assert len([line for line in output if "project hooks not loaded" in line]) == 1
    assert not any((run_dir / name).exists() for name in ("hooks", "hook-bundle", "hook-tape.jsonl"))
    files = {path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()}
    expected = {"spec.json", "tool-settings.json", "skill-settings.json"}
    for base in ("workspace", "parent"):
        expected.update(f"{base}/{rel}" for rel in ("sample.py", "tests/test_sample.py"))
    assert files == expected


def test_in_place_apply_ignores_agent_created_stateframe_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = _repo(tmp_path / "repo")
    project_dir = repo / ".stateframe"
    project_dir.mkdir()
    project_hooks = project_dir / "hooks.json"
    trusted_contents = json.dumps({"schema": "stateframe-hooks/1", "hooks": []})
    project_hooks.write_text(trusted_contents, encoding="utf-8")
    trusted = hook_host.trust_project_hooks(repo, printer=lambda *_: None)
    work = workcopy.create(repo, tmp_path / "workspace")
    assert not (work.copy / ".stateframe").exists()
    agent_file = work.copy / ".stateframe" / "hooks.json"
    agent_file.parent.mkdir()
    agent_file.write_text('{"hooks": []}\n', encoding="utf-8")
    assert workcopy.apply_in_place(work) == []
    assert project_hooks.read_text(encoding="utf-8") == trusted_contents
    selection = hook_host.resolve_project_hooks(repo, shell_network="none")
    assert selection.hooks is not None and selection.digest == trusted


def test_prepare_publishes_completed_runner_context_and_bundle(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    config = _hook_config(tmp_path)
    monkeypatch.setattr(hook_host, "HookRunner", _FakeHookRunner)
    output = []
    run_dir = tmp_path / "run"
    run, _, spec = runner.prepare(_options(repo, run_dir, hooks=(config,)), printer=output.append)
    pinned = spec["hooks"]
    assert set(pinned["bundle_manifest"]) == {"files", "directories"}
    assert [entry["event"] for entry in pinned["start_context"]] == ["session_start", "prompt_submit"]
    assert pinned["start_context"][0] == asdict(PanelEntry("start", "start", "session_start", "", "welcome"))
    assert pinned["start_context"][1] == asdict(
        PanelEntry("start", "guard", "prompt_submit", "", "prompt context"))
    assert (run / "hook-bundle").is_dir()
    assert "hooks: 2 loaded from 1 files" in output
    assert "start_context" in json.loads((run / "spec.json").read_text(encoding="utf-8"))["hooks"]
    records = [json.loads(line) for line in (run / "hook-tape.jsonl").read_text().splitlines()]
    assert [item["identity"]["event"] for item in records if item["kind"] == "intent"] == [
        "session_start", "prompt_submit"
    ]
    assert [item["kind"] for item in records] == ["intent", "complete", "intent", "complete"]


def test_prompt_submit_denial_precedes_baseline_and_child(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    config = _hook_config(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["hooks"].append({"name": "notify", "event": "session_end",
                              "argv": [sys.executable, "-c", "print('finished')"]})
    config.write_text(json.dumps(document), encoding="utf-8")
    class DenyingRunner(_FakeHookRunner):
        deny_prompt = True

    monkeypatch.setattr(hook_host, "HookRunner", DenyingRunner)
    monkeypatch.setattr(Endpoint, "check_locality", lambda _self: None)
    monkeypatch.setattr(Endpoint, "probe", lambda _self: {"model_found": True})
    monkeypatch.setattr(runner, "_check_python", lambda value: value)
    monkeypatch.setattr(runner, "_probe_capabilities", lambda _options: {"display": "caps"})
    monkeypatch.setattr(runner, "_select_test_python", lambda _options, _printer: sys.executable)
    monkeypatch.setattr(runner, "check_slot_context", lambda *_args: None)
    monkeypatch.setattr(runner, "_select_prompt_cache_slot", lambda *_args: None)
    calls = []
    monkeypatch.setattr(runner, "run_test_cmd", lambda *_args, **_kwargs: calls.append("baseline"))
    monkeypatch.setattr(runner, "drive", lambda *_args, **_kwargs: calls.append("child"))
    options = _options(repo, tmp_path / "denied-run", hooks=(config,))
    with pytest.raises(runner.RunError, match="hook refused run: HOOK_DENIED: guard: blocked"):
        runner._run(options, 0.0, lambda *_: None)
    assert calls == []
    records = tmp_path / "denied-run" / "hooks" / "records.jsonl"
    recorded = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    assert recorded[-2]["decision"] == "deny"
    assert recorded[-1]["event"] == "session_end"
    summary = json.loads((tmp_path / "denied-run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["runtime_outcome"] == "hook_refused"
    assert summary["engine_result"] == "HOOK_DENIED"
    assert summary["exit_code"] == 2
    assert summary["hooks"]["denied"] == 1
    with pytest.raises(runner.resume_module.ResumeRefused,
                       match="hook refused prompt submission; start a new run"):
        runner.resume_module.assess(tmp_path / "denied-run")


@pytest.mark.parametrize(
    ("records", "expected_counts", "terminal_line"),
    [
        ([{"hook": "gate", "event": "stop", "decision": "deny", "error_code": None}],
         {"runs": 1, "denied": 0, "vetoes": 1, "errors": 0, "last_error": None},
         "hooks: 1 runs, 0 denied, 1 vetoes, 0 errors"),
        ([{"hook": "prompt", "event": "prompt_submit", "decision": "deny", "error_code": None},
          {"hook": "tool", "event": "pre_tool", "decision": "deny", "error_code": None},
          {"hook": "gate", "event": "stop", "decision": "deny", "error_code": None},
          {"hook": "broken", "event": "stop", "decision": "error", "error_code": "timeout"},
          {"hook": "audit", "event": "session_end", "decision": "continue", "error_code": None}],
         {"runs": 5, "denied": 2, "vetoes": 1, "errors": 1,
          "last_error": "broken: timeout"},
         "hooks: 5 runs, 2 denied, 1 vetoes, 1 errors"),
    ],
)
def test_hook_terminal_counts_keep_denials_and_vetoes_separate(
    tmp_path, records, expected_counts, terminal_line
):
    run_dir = tmp_path / "run"
    records_path = run_dir / "hooks" / "records.jsonl"
    records_path.parent.mkdir(parents=True)
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output = []
    runner._write_hook_terminal_summary(
        run_dir, runtime_outcome="hook_blocked", engine_result="HOOK_BLOCKED",
        printer=output.append,
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["hooks"] == expected_counts
    assert output == [terminal_line]


def test_start_hook_containment_persists_terminal_summary(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    config = _hook_config(tmp_path)

    class ContainingRunner(_FakeHookRunner):
        def run_event(self, *_args, **_kwargs):
            raise hook_host.HookContainmentError("workspace changed")

    monkeypatch.setattr(hook_host, "HookRunner", ContainingRunner)
    run_dir = tmp_path / "contained-run"
    with pytest.raises(runner.HookContainmentRunError, match="HOOK_CONTAINMENT: workspace changed"):
        runner.prepare(_options(repo, run_dir, hooks=(config,)), printer=lambda *_: None)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["runtime_outcome"] == "hook_containment"
    assert summary["engine_result"] == "HOOK_CONTAINMENT"
    assert summary["exit_code"] == 2
    assert summary["hooks"]["errors"] == 1


def test_session_end_runs_after_in_place_apply_and_containment_keeps_changes(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    config = _hook_config(tmp_path)
    monkeypatch.setattr(hook_host, "HookRunner", _FakeHookRunner)
    run, work, spec = runner.prepare(_options(repo, tmp_path / "run", hooks=(config,), in_place=True),
                                     printer=lambda *_: None)
    digest = "sha256:" + "a" * 64
    terminal = {"result": "SUBMITTED", "submission": {"stored": True, "seal_id": digest,
                                                           "artifact_digest": digest}}

    def drive(_run, _options, _printer):
        (_run / "engine-result.json").write_text(json.dumps({"result": terminal}), encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "drive", drive)
    monkeypatch.setattr(runner, "summarize", lambda *_args: {
        "engine_terminal": terminal, "engine_result": "SUBMITTED", "cached_share_display": "cache"
    })
    monkeypatch.setattr(runner, "final_candidate", lambda _run: {"sample.py": b"value = 2\n"})
    monkeypatch.setattr(runner, "_artifact_digest", lambda _candidate: digest)
    monkeypatch.setattr(runner, "_tests_python", lambda _options: sys.executable)
    monkeypatch.setattr(runner, "_outer_test_python", lambda _options: sys.executable)
    monkeypatch.setattr(runner, "run_test_cmd", lambda *args, **kwargs: {
        "command": "true", "exit_code": 0, "passed": True, "tail": "ok"
    })
    event = {}

    def end(_run, _hooks, workspace, *, data, **_kwargs):
        event.update(data)
        assert (repo / "sample.py").read_text(encoding="utf-8") == "value = 2\n"
        assert workspace == work.copy
        raise hook_host.HookContainmentError("descendant survived")

    monkeypatch.setattr(hook_host, "run_session_end", end)
    output = []
    summary = runner._execute(_options(repo, run, hooks=(config,), in_place=True), 0.0,
                              output.append, run, work, spec)
    assert event["outcome"] == "completed"
    assert event["applied"] is True
    assert event["changed_paths"] == ["sample.py"]
    assert summary["runtime_outcome"] == "hook_containment"
    assert summary["exit_code"] == 2 and not summary["succeeded"]
    assert summary["applied_in_place"] == ["sample.py"]
    assert summary["hooks"]["errors"] == 1
    assert summary["hooks"]["last_error"] == "descendant survived"
    assert "hooks: 2 runs, 0 denied, 0 vetoes, 1 errors" in output


def test_session_end_output_is_terminal_escaped_text(tmp_path, monkeypatch):
    entry = PanelEntry("start", "notify", "session_end", "", "done\nREFUSED fake")

    class Runner:
        def run_event(self, *_args, **_kwargs):
            return SimpleNamespace(runs=(), entries=(entry,))

        def close(self):
            pass

    monkeypatch.setattr(hook_host, "_new_runner", lambda *_args, **_kwargs: Runner())
    output = []
    hook_host.run_session_end(tmp_path, {}, tmp_path, data={}, shell_network="none",
                              run_seed="seed", run_id="run", version="0.2.1", printer=output.append)
    assert output == ['hooks: notify session_end output: "done\\nREFUSED fake"']


def test_hook_terminal_outcomes_are_nonzero_and_session_end_waits_for_final_attempt():
    blocked = runner.outcomes.project({
        "engine_terminal": {"result": "HOOK_BLOCKED"}, "child_exit_code": 0,
        "candidate_captured": False,
    })
    containment = runner.outcomes.project({
        "engine_terminal": {"result": "HOOK_CONTAINMENT"}, "child_exit_code": 0,
        "candidate_captured": False,
    })
    assert blocked["runtime_outcome"] == "hook_blocked"
    assert runner.outcomes.exit_code(blocked) == 1
    assert containment["runtime_outcome"] == "hook_containment"
    assert runner.outcomes.exit_code(containment) == 2
    assert runner._session_end_is_final({"engine_terminal": {"result": "SUBMITTED"}})
    assert runner._session_end_is_final({"engine_terminal": {"result": "HOOK_BLOCKED"}})
    assert not runner._session_end_is_final({"engine_terminal": {"result": "BUDGET_STOP"}})


@pytest.mark.host_sandbox
def test_real_runner_start_hook_publishes_sanitized_context(tmp_path):
    repo = _repo(tmp_path / "repo")
    folder = tmp_path / "hook-source"
    folder.mkdir()
    config = folder / "hooks.json"
    config.write_text(json.dumps({
        "schema": "stateframe-hooks/1",
        "hooks": [{"name": "start", "event": "session_start",
                   "argv": [sys.executable, "-c", "print('welcome from hook')"]}],
    }), encoding="utf-8")
    hooks = load_hook_files((config,), shell_network="none", repo=repo)
    run = tmp_path / "run"
    run.mkdir()
    (run / "workspace").mkdir()
    manifest = create_bundle(hooks, run / "hook-bundle")
    spec = hooks.as_spec(manifest)
    entries = hook_host.run_start_hooks(run, spec, run / "workspace", task="work", repo=repo,
                                        shell_network="none", run_seed="seed", run_id="run",
                                        version="0.2.1", printer=lambda *_: None)
    assert entries == [asdict(PanelEntry("start", "start", "session_start", "", "welcome from hook"))]
