import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_hooks import PanelEntry, event_id
from stateframe import chat, hook_host, runner, workcopy
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


def _project_hook_file(repo: Path) -> Path:
    project = repo / ".stateframe"
    project.mkdir()
    (project / "context.py").write_text("print('trusted')\n", encoding="utf-8")
    config = project / "hooks.json"
    config.write_text(json.dumps({
        "schema": "stateframe-hooks/1",
        "hooks": [
            {"name": "start", "event": "session_start",
             "argv": [sys.executable, "context.py"], "files": ["context.py"]},
            {"name": "guard", "event": "prompt_submit", "argv": [sys.executable, "-c", "pass"]},
        ],
    }), encoding="utf-8")
    return config


class _FakeHookRunner:
    deny_prompt = False

    def __init__(self, run_dir, hooks, **kwargs):
        self.run_dir = Path(run_dir)
        self.hooks = hooks
        self.run_seed = kwargs["run_seed"]
        self.tape = hook_host.HookTape(self.run_dir)

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
        denied = event == "prompt_submit" and self.deny_prompt
        context = "trusted context" if event == "session_start" else ""
        outcome = {"decision": "deny" if denied else "continue", "reason": "blocked" if denied else "",
                   "context": context}
        self.tape.complete(record_id, {"outcome": outcome})
        records = self.run_dir / "hooks" / "records.jsonl"
        records.parent.mkdir(parents=True, exist_ok=True)
        with records.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"hook": hook, "event": event, "decision": outcome["decision"],
                                     "error_code": None}) + "\n")
        entries = [PanelEntry("start", hook, event, "", context)] if context else []
        return SimpleNamespace(runs=(), entries=entries, decision=outcome["decision"],
                               deny_code="HOOK_DENIED" if denied else None,
                               deny_hook=hook if denied else None, reason=outcome["reason"])

    def close(self):
        pass


def _session(repo: Path, home: Path, *, no_project_hooks=False) -> chat.Session:
    base = runner.RunOptions(
        repo=repo,
        task="",
        endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "test", None),
        max_turns=1,
        test_cmd="true",
        checks=["tests/test_sample.py"],
        target="sample.py",
        no_project_hooks=no_project_hooks,
    )
    return chat.Session(repo=repo, home=home, base=base)


def _success_summary(run_dir: Path, work, spec: dict, value: int) -> dict:
    workcopy.apply_files(work, {"sample.py": f"value = {value}\n".encode()},
                         declared=set(spec["public_row"]["files"]))
    (run_dir / "final.diff").write_text(workcopy.unified_diff(work), encoding="utf-8")
    candidate = runner.candidate_digest(work.copy)
    artifact = "sha256:" + "a" * 64
    terminal = {"result": "SUBMITTED", "submission": {
        "stored": True, "seal_id": artifact, "artifact_digest": artifact,
    }}
    return {
        "runtime_outcome": "completed", "engine_result": "SUBMITTED", "engine_terminal": terminal,
        "child_exit_code": 0, "candidate_captured": True, "candidate_sha256": candidate,
        "candidate_artifact_digest": artifact, "candidate_error": None,
        "tests_after": {"passed": True, "exit_code": 0, "candidate_sha256": candidate},
        "check_outcome": "pass", "artifact": "saved", "exit_code": 0, "succeeded": True,
        "files_changed": workcopy.changed_files(work), "run_dir": str(run_dir),
    }


def test_trusted_project_hooks_pass_to_each_task_using_original_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = _repo(tmp_path / "repo")
    config = _project_hook_file(repo)
    hook_host.trust_project_hooks(repo, printer=lambda *_: None)
    monkeypatch.setattr(hook_host, "HookRunner", _FakeHookRunner)
    observed = []

    def fake_run(options, printer):
        run_dir, work, spec = runner.prepare(options, printer=printer)
        observed.append((options.hooks_repo, options.repo, spec.get("hooks")))
        return _success_summary(run_dir, work, spec, len(observed) + 1)

    monkeypatch.setattr(chat, "run", fake_run)
    session = _session(repo, tmp_path / "session")
    session.start(printer=lambda *_: None)
    first = session.run_task("first", printer=lambda *_: None)
    second = session.run_task("second", printer=lambda *_: None)

    assert first.kept and second.kept
    assert [item[0] for item in observed] == [repo.resolve(), repo.resolve()]
    assert all(item[1] != repo.resolve() for item in observed)
    assert all(item[2] and item[2]["definitions"][0]["name"] == "start" for item in observed)
    assert config.is_file()


def test_same_project_hook_bytes_are_untrusted_in_another_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    trusted_repo = _repo(tmp_path / "trusted")
    trusted_file = _project_hook_file(trusted_repo)
    hook_host.trust_project_hooks(trusted_repo, printer=lambda *_: None)
    untrusted_repo = _repo(tmp_path / "untrusted")
    untrusted_file = _project_hook_file(untrusted_repo)
    assert trusted_file.read_bytes() == untrusted_file.read_bytes()
    monkeypatch.setattr(hook_host, "HookRunner", _FakeHookRunner)
    observed = []

    def fake_run(options, printer):
        run_dir, work, spec = runner.prepare(options, printer=printer)
        observed.append(spec.get("hooks"))
        return _success_summary(run_dir, work, spec, 2)

    monkeypatch.setattr(chat, "run", fake_run)
    session = _session(untrusted_repo, tmp_path / "untrusted-session")
    messages = []
    session.start(printer=messages.append)
    record = session.run_task("task", printer=messages.append)
    assert record.kept
    assert observed == [None]
    assert any("project hooks not loaded" in message for message in messages)


def test_prompt_submit_refusal_and_hook_blocked_task_are_not_promoted(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = _repo(tmp_path / "repo")
    _project_hook_file(repo)
    hook_host.trust_project_hooks(repo, printer=lambda *_: None)
    monkeypatch.setattr(hook_host, "HookRunner", type("DenyingRunner", (_FakeHookRunner,), {"deny_prompt": True}))

    def refusing_run(options, printer):
        runner.prepare(options, printer=printer)
        raise AssertionError("prompt_submit refusal must stop prepare")

    monkeypatch.setattr(chat, "run", refusing_run)
    session = _session(repo, tmp_path / "session")
    session.start(printer=lambda *_: None)
    refused = session.run_task("refused", printer=lambda *_: None)
    assert refused.error and "hook refused run" in refused.error
    assert refused.runtime_outcome == "hook_refused" and refused.result == "HOOK_DENIED"
    assert not refused.kept
    assert session.copy.joinpath("sample.py").read_text(encoding="utf-8") == "value = 1\n"

    monkeypatch.setattr(hook_host, "HookRunner", _FakeHookRunner)

    def blocked_run(options, printer):
        run_dir, work, spec = runner.prepare(options, printer=printer)
        workcopy.apply_files(work, {"sample.py": b"value = 9\n"}, declared=set(spec["public_row"]["files"]))
        (run_dir / "final.diff").write_text(workcopy.unified_diff(work), encoding="utf-8")
        return {"runtime_outcome": "hook_blocked", "engine_result": "HOOK_BLOCKED",
                "engine_terminal": {"result": "HOOK_BLOCKED"}, "child_exit_code": 1,
                "files_changed": workcopy.changed_files(work), "run_dir": str(run_dir)}

    monkeypatch.setattr(chat, "run", blocked_run)
    blocked = session.run_task("blocked", printer=lambda *_: None)
    assert blocked.runtime_outcome == "hook_blocked"
    assert not blocked.kept
    assert session.copy.joinpath("sample.py").read_text(encoding="utf-8") == "value = 1\n"


def test_session_end_containment_stops_chat_and_agent_hook_file_cannot_apply(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    session = _session(repo, tmp_path / "session")
    assert session.base is not None
    session.base.hooks_files = (tmp_path / "explicit-hooks.json",)
    session.start(printer=lambda *_: None)

    def contained_run(options, printer):
        options.run_dir.mkdir(parents=True)
        (options.run_dir / "final.diff").write_text("", encoding="utf-8")
        return {"runtime_outcome": "hook_containment", "hook_containment": "cleanup failed",
                "engine_result": "SUBMITTED", "engine_terminal": {"result": "SUBMITTED"},
                "child_exit_code": 0, "run_dir": str(options.run_dir)}

    monkeypatch.setattr(chat, "run", contained_run)
    record = session.run_task("stop", printer=lambda *_: None)
    assert record.runtime_outcome == "hook_containment"
    assert not record.kept and session._stopped
    with pytest.raises(runner.RunError, match="stopped after HOOK_CONTAINMENT"):
        session.run_task("later", printer=lambda *_: None)

    fake_config = session.copy / ".stateframe" / "hooks.json"
    fake_config.parent.mkdir(parents=True)
    fake_config.write_text("agent-created", encoding="utf-8")
    assert session.apply() == []
    assert not (repo / ".stateframe" / "hooks.json").exists()


def test_no_project_hooks_and_apply_do_not_run_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = _repo(tmp_path / "repo")
    _project_hook_file(repo)
    hook_host.trust_project_hooks(repo, printer=lambda *_: None)
    monkeypatch.setattr(hook_host, "HookRunner", _FakeHookRunner)
    calls = []

    def fake_run(options, printer):
        calls.append("run")
        run_dir, work, spec = runner.prepare(options, printer=printer)
        return _success_summary(run_dir, work, spec, 2)

    monkeypatch.setattr(chat, "run", fake_run)
    session = _session(repo, tmp_path / "session", no_project_hooks=True)
    session.start(printer=lambda *_: None)
    assert session.run_task("task", printer=lambda *_: None).kept
    assert calls == ["run"]
    (session.copy / "sample.py").write_text("value = 3\n", encoding="utf-8")
    assert session.apply() == ["sample.py"]
    assert calls == ["run"]
    assert (repo / "sample.py").read_text(encoding="utf-8") == "value = 3\n"


def test_chat_prepare_reverifies_pinned_project_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = _repo(tmp_path / "repo")
    config = _project_hook_file(repo)
    hook_host.trust_project_hooks(repo, printer=lambda *_: None)
    monkeypatch.setattr(hook_host, "HookRunner", _FakeHookRunner)
    session = _session(repo, tmp_path / "session")
    session.start(printer=lambda *_: None)
    config.write_text(config.read_text(encoding="utf-8").replace('"name": "start"', '"name": "changed"'),
                      encoding="utf-8")

    def prepare_run(options, printer):
        return runner.prepare(options, printer=printer)

    monkeypatch.setattr(chat, "run", prepare_run)
    record = session.run_task("task", printer=lambda *_: None)
    assert record.error
    assert "hook definition, executable or file changed" in record.error
    assert not record.kept


def test_hook_enabled_apply_never_copies_agent_created_project_hook_file(tmp_path):
    repo = _repo(tmp_path / "repo")
    session = _session(repo, tmp_path / "session")
    assert session.base is not None
    session.base.hooks_files = (tmp_path / "explicit-hooks.json",)
    session.start(printer=lambda *_: None)
    agent_file = session.copy / ".stateframe" / "hooks.json"
    agent_file.parent.mkdir()
    agent_file.write_text('{"schema": "stateframe-hooks/1", "hooks": []}\n', encoding="utf-8")
    assert session.apply() == []
    assert not (repo / ".stateframe").exists()
    selection = hook_host.resolve_project_hooks(repo, shell_network="none")
    assert selection.hooks is None and selection.message is None


def test_no_hooks_apply_preserves_021_ignored_path_behavior(tmp_path):
    repo = _repo(tmp_path / "repo")
    session = _session(repo, tmp_path / "session")
    session.start(printer=lambda *_: None)
    agent_file = session.copy / ".stateframe" / "hooks.json"
    agent_file.parent.mkdir()
    agent_file.write_text("ordinary data", encoding="utf-8")
    assert session.apply() == [".stateframe/hooks.json"]
    assert (repo / ".stateframe" / "hooks.json").read_text() == "ordinary data"


def test_chat_without_hooks_starts_with_regular_stateframe_file(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / ".stateframe").write_text("ordinary repository data", encoding="utf-8")
    session = _session(repo, tmp_path / "session")
    session.start(printer=lambda *_: None)
    assert session._project_hooks is None
    assert session.copy.is_dir()


def test_session_without_run_options_can_start_and_apply_existing_chat_state(tmp_path):
    repo = _repo(tmp_path / "repo")
    session = chat.Session(repo=repo, home=tmp_path / "session", base=None)
    session.start(printer=lambda *_: None)
    (session.copy / "sample.py").write_text("value = 5\n", encoding="utf-8")

    assert session.apply() == ["sample.py"]
    assert (repo / "sample.py").read_text(encoding="utf-8") == "value = 5\n"
    with pytest.raises(runner.RunError, match="chat run options are required"):
        session.run_task("cannot run without options", printer=lambda *_: None)


def test_repl_returns_containment_status_and_stops_after_terminal_hook_failure(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    session = _session(repo, tmp_path / "session")

    def contained_task(text, printer=print):
        session._stopped = True
        return chat.TaskRecord(
            task=text, result="HOOK_CONTAINMENT", tests_passed=False, files=[], diff="",
            run_dir=str(tmp_path / "run"), kept=False, attempt=1,
            runtime_outcome="hook_containment", error="task error: cleanup failed",
        )

    monkeypatch.setattr(session, "run_task", contained_task)
    lines = iter(["task", "/apply"])
    output = []
    assert chat.repl(session, read=lambda _prompt: next(lines), printer=output.append) == 2
    assert any("chat stopped after HOOK_CONTAINMENT" in line for line in output)
    assert not any(line.startswith("applied to your repo:") for line in output)
