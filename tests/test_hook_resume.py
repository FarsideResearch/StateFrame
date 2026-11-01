import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_hook_exec
from kernel_hooks import PanelEntry, event_id
from stateframe import hook_host, resume, runner, shell_host
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


def _config(root: Path) -> tuple[Path, Path, Path]:
    folder = root / "hook-source"
    folder.mkdir()
    script = folder / "hook.py"
    script.write_text("print('resume context')\n", encoding="utf-8")
    resource = folder / "resource.txt"
    resource.write_text("pinned resource", encoding="utf-8")
    config = folder / "hooks.json"
    config.write_text(json.dumps({
        "schema": "stateframe-hooks/1",
        "hooks": [{"name": "start", "event": "session_start",
                   "argv": [sys.executable, "hook.py"], "files": ["hook.py", "resource.txt"]}],
    }), encoding="utf-8")
    return config, script, resource


class _RecordingHookRunner:
    instances = 0

    def __init__(self, run_dir, hooks, **kwargs):
        type(self).instances += 1
        self.run_dir = Path(run_dir)
        self.hooks = hooks
        self.run_seed = kwargs["run_seed"]
        self.tape = kernel_hook_exec.HookTape(self.run_dir)

    def run_event(self, event, *, boundary, **_kwargs):
        indexed = next(((index, hook) for index, hook in enumerate(self.hooks.definitions)
                        if hook.event == event), None)
        if indexed is None:
            return SimpleNamespace(runs=(), entries=(), decision="continue", deny_code=None,
                                   deny_hook=None, reason="")
        index, definition = indexed
        hook = definition.name
        record_id = event_id(self.run_seed, boundary, index)
        self.tape.intent(record_id, {"event_id": record_id, "hook": hook, "event": event})
        context = "resume context" if event == "session_start" else ""
        self.tape.complete(record_id, {"outcome": {
            "decision": "continue", "reason": "", "context": context,
        }})
        records = self.run_dir / "hooks" / "records.jsonl"
        records.parent.mkdir(parents=True, exist_ok=True)
        with records.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"hook": hook, "event": event, "decision": "continue",
                                     "error_code": None}) + "\n")
        entries = [PanelEntry("start", hook, event, "", context)] if context else []
        return SimpleNamespace(runs=(), entries=entries, decision="continue", deny_code=None,
                               deny_hook=None, reason="")

    def close(self):
        pass


def _prepared(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict, Path, Path, Path]:
    repo = _repo(tmp_path / "repo")
    config, script, resource = _config(tmp_path)
    _RecordingHookRunner.instances = 0
    monkeypatch.setattr(hook_host, "HookRunner", _RecordingHookRunner)
    options = runner.RunOptions(
        repo=repo, task="update sample", endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        max_turns=1, test_cmd="true", checks=["tests/test_sample.py"], target="sample.py",
        run_dir=tmp_path / "run", hooks_files=(config,),
    )
    run_dir, _, spec = runner.prepare(options, printer=lambda *_: None)
    (run_dir / "wire").mkdir()
    (run_dir / "sends.jsonl").write_text("", encoding="utf-8")
    return repo, run_dir, spec, config, script, resource


@pytest.mark.parametrize("target", ["definition", "executable", "file", "bundle"])
def test_resume_refuses_changed_hook_inputs(tmp_path, monkeypatch, target):
    _repo_path, run_dir, spec, config, script, resource = _prepared(tmp_path, monkeypatch)
    if target == "definition":
        config.write_text(config.read_text(encoding="utf-8").replace('"name": "start"', '"name": "other"'),
                          encoding="utf-8")
    elif target == "executable":
        script.write_text("print('changed executable')\n", encoding="utf-8")
    elif target == "file":
        resource.write_text("changed resource", encoding="utf-8")
    else:
        bundled = run_dir / "hook-bundle" / "0" / "resource.txt"
        bundled.chmod(0o600)
        bundled.write_text("changed bundle", encoding="utf-8")
    with pytest.raises(resume.ResumeRefused, match="hook definition, executable or file changed"):
        resume.assess(run_dir)


def test_resume_refuses_changed_external_hook_executable(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    executable = tmp_path / "hook-command"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    config_dir = tmp_path / "hook-source"
    config_dir.mkdir()
    config = config_dir / "hooks.json"
    config.write_text(json.dumps({
        "schema": "stateframe-hooks/1",
        "hooks": [{"name": "start", "event": "session_start", "argv": [str(executable)]}],
    }), encoding="utf-8")
    _RecordingHookRunner.instances = 0
    monkeypatch.setattr(hook_host, "HookRunner", _RecordingHookRunner)
    options = runner.RunOptions(
        repo=repo, task="update sample", endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        max_turns=1, test_cmd="true", checks=["tests/test_sample.py"], target="sample.py",
        run_dir=tmp_path / "run", hooks_files=(config,),
    )
    run_dir, _, _spec = runner.prepare(options, printer=lambda *_: None)
    executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    executable.chmod(0o755)
    with pytest.raises(resume.ResumeRefused, match="hook definition, executable or file changed"):
        resume.assess(run_dir)


def test_resume_uses_pinned_project_hooks_without_rechecking_trust_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = _repo(tmp_path / "repo")
    project = repo / ".stateframe"
    project.mkdir()
    (project / "context.py").write_text("print('trusted')\n", encoding="utf-8")
    (project / "hooks.json").write_text(json.dumps({
        "schema": "stateframe-hooks/1",
        "hooks": [{"name": "start", "event": "session_start",
                   "argv": [sys.executable, "context.py"], "files": ["context.py"]}],
    }), encoding="utf-8")
    hook_host.trust_project_hooks(repo, printer=lambda *_: None)
    _RecordingHookRunner.instances = 0
    monkeypatch.setattr(hook_host, "HookRunner", _RecordingHookRunner)
    options = runner.RunOptions(
        repo=repo, task="update sample", endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        max_turns=1, test_cmd="true", checks=["tests/test_sample.py"], target="sample.py",
        run_dir=tmp_path / "run",
    )
    run_dir, _, spec = runner.prepare(options, printer=lambda *_: None)
    (run_dir / "wire").mkdir()
    (run_dir / "sends.jsonl").write_text("", encoding="utf-8")
    trust_store = tmp_path / "home" / ".stateframe" / "trusted-hooks.json"
    trust_store.unlink()
    state = resume.assess(run_dir)
    assert state["spec"]["hooks"]["digest"] == spec["hooks"]["digest"]


def test_pending_runner_intent_refuses_before_archive(tmp_path, monkeypatch):
    _repo_path, run_dir, _spec, _config_path, _script, _resource = _prepared(tmp_path, monkeypatch)
    kernel_hook_exec.HookTape(run_dir).intent("unresolved", {
        "event_id": "unresolved", "hook": "guard", "event": "session_end",
    })
    with pytest.raises(resume.ResumeRefused, match="hook guard may or may not have run; start a new run"):
        runner.resume_run(run_dir, endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None))
    assert not (run_dir / "attempts").exists()


def test_complete_unpublished_context_rebuilds_without_rerunning(tmp_path, monkeypatch):
    _repo_path, run_dir, spec, _config_path, _script, _resource = _prepared(tmp_path, monkeypatch)
    del spec["hooks"]["start_context"]
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True), encoding="utf-8")

    class StopAfterRebuild(Exception):
        pass

    def stop_endpoint(*_args, **_kwargs):
        raise StopAfterRebuild

    monkeypatch.setattr(runner.Endpoint, "from_args", stop_endpoint)
    with pytest.raises(StopAfterRebuild):
        runner.resume_run(run_dir)
    repaired = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    assert repaired["hooks"]["start_context"] == [
        {"kind": "start", "hook": "start", "event": "session_start", "label": "", "text": "resume context"}
    ]
    assert _RecordingHookRunner.instances == 1
    assert not (run_dir / "attempts").exists()


def test_resume_rejects_published_start_context_that_differs_from_runner_tape(tmp_path, monkeypatch):
    _repo_path, run_dir, spec, _config_path, _script, _resource = _prepared(tmp_path, monkeypatch)
    spec["hooks"]["start_context"][0]["text"] = "edited context"
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True), encoding="utf-8")
    with pytest.raises(resume.ResumeRefused,
                       match="published hook start_context does not match its runner records"):
        resume.assess(run_dir)


def test_unstarted_start_hooks_run_during_resume_before_engine(tmp_path, monkeypatch):
    _repo_path, run_dir, spec, _config_path, _script, _resource = _prepared(tmp_path, monkeypatch)
    (run_dir / "hook-tape.jsonl").unlink()
    (run_dir / "hooks" / "records.jsonl").unlink()
    (run_dir / "hooks").rmdir()
    del spec["hooks"]["start_context"]
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True), encoding="utf-8")
    _RecordingHookRunner.instances = 0
    captured = {}

    monkeypatch.setattr(runner.Endpoint, "check_locality", lambda _self: None)
    monkeypatch.setattr(runner.Endpoint, "probe", lambda _self: {"model_found": True})
    monkeypatch.setattr(runner, "_check_python", lambda value: value)
    monkeypatch.setattr(runner, "check_slot_context", lambda *_args: None)
    monkeypatch.setattr(shell_host, "provision", lambda *_args: None)

    def execute(_options, _started, _printer, _run, _work, current_spec):
        captured.update(current_spec)
        return {"runtime_outcome": "budget_stop", "exit_code": 1}

    monkeypatch.setattr(runner, "_execute", execute)
    result = runner.resume_run(
        run_dir,
        endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        printer=lambda *_: None,
    )
    assert result["runtime_outcome"] == "budget_stop"
    assert captured["hooks"]["start_context"] == [
        {"kind": "start", "hook": "start", "event": "session_start", "label": "", "text": "resume context"}
    ]
    assert _RecordingHookRunner.instances == 1
    records = [json.loads(line) for line in (run_dir / "hook-tape.jsonl").read_text().splitlines()]
    assert [item["identity"]["event"] for item in records if item["kind"] == "intent"] == ["session_start"]


def test_containment_run_refuses_resume(tmp_path, monkeypatch):
    _repo_path, run_dir, _spec, _config_path, _script, _resource = _prepared(tmp_path, monkeypatch)
    (run_dir / "summary.json").write_text(json.dumps({"runtime_outcome": "hook_containment"}),
                                           encoding="utf-8")
    with pytest.raises(resume.ResumeRefused, match="HOOK_CONTAINMENT run cannot be resumed"):
        runner.resume_run(run_dir)
    assert not (run_dir / "attempts").exists()


def test_hook_blocked_run_is_terminal_for_resume(tmp_path, monkeypatch):
    _repo_path, run_dir, _spec, _config_path, _script, _resource = _prepared(tmp_path, monkeypatch)
    terminal = {"result": "HOOK_BLOCKED"}
    (run_dir / "engine-result.json").write_text(json.dumps({"result": terminal}), encoding="utf-8")
    with pytest.raises(resume.ResumeRefused, match="HOOK_BLOCKED run cannot be resumed"):
        runner.resume_run(run_dir)
    assert not (run_dir / "attempts").exists()


def test_two_archives_keep_pinned_bundle_and_attempt_scope_only_hooks(tmp_path, monkeypatch):
    _repo_path, run_dir, spec, _config_path, _script, _resource = _prepared(tmp_path, monkeypatch)
    bundle = run_dir / "hook-bundle"
    before = {path.relative_to(bundle).as_posix(): path.read_bytes()
              for path in bundle.rglob("*") if path.is_file()}
    assert "hooks" in resume.ATTEMPT_FILES
    assert "hook-bundle" not in resume.ATTEMPT_FILES
    assert "hook-tape.jsonl" not in resume.ATTEMPT_FILES
    assert "hook-effects" not in resume.ATTEMPT_FILES

    state = resume.assess(run_dir)
    resume.archive_attempt(state, Path(spec["engine_dir"]), use_checkpoint=False)
    (run_dir / "wire").mkdir()
    (run_dir / "sends.jsonl").write_text("", encoding="utf-8")
    resume.fresh_workcopy(run_dir, Path(spec["repo"]))
    state = resume.assess(run_dir)
    resume.archive_attempt(state, Path(spec["engine_dir"]), use_checkpoint=False)

    after = {path.relative_to(bundle).as_posix(): path.read_bytes()
             for path in bundle.rglob("*") if path.is_file()}
    assert after == before
    assert (run_dir / "attempts" / "1" / "hooks" / "records.jsonl").is_file()
    assert (run_dir / "attempts" / "2" / "spec.json").is_file()
