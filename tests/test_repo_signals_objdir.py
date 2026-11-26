"""Repository signals are optional at startup and first-request preparation."""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_repo_signals as signals


pytestmark = pytest.mark.usefixtures("isolated_pilot_context")


def _git(root: Path, *args: str) -> str:
    env = {"PATH": os.defpath, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    result = subprocess.run(["git", *args], cwd=root, env=env, capture_output=True,
                            text=True, check=True, timeout=10)
    return result.stdout.strip()


@pytest.fixture
def git_probe(monkeypatch):
    import kernel_safe_probe

    def run(argv, workspace, *, timeout, env, read_roots):
        # Exercise real Git inside the private view without requiring a host sandbox.
        assert Path(env["GIT_DIR"]) == read_roots[0]
        assert env["GIT_DIR"] != str(workspace / ".git")
        result = subprocess.run(argv, cwd=workspace, env=env, capture_output=True,
                                check=False, timeout=timeout)
        return SimpleNamespace(timed_out=False, exit_code=result.returncode,
                               output_truncated=False, output=result.stdout)

    monkeypatch.setattr(kernel_safe_probe, "run_probe", run)


def _repo(root: Path) -> tuple[Path, str]:
    root.mkdir()
    (root / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "-q", "--template=")
    _git(root, "add", "value.py")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "Initial value")
    return root, _git(root, "rev-parse", "HEAD")


def _first_request(root: Path, output: Path, *, base: str | None = None,
                   before_prepare: Callable[[], None] | None = None):
    from detreason.world_knowledge import WorldKnowledgeStore
    from kernel_bolt import KernelBolt
    from kernel_stateframe_runtime import StateFrameRuntime
    from probes.react_repair import _tools_for_probe_modes

    source = root / "value.py"
    owner = KernelBolt(root, "", "repo-signals-objdir")
    owner.world = WorldKnowledgeStore(sandbox_id="repo-signals-objdir")
    owner.task_row = {"msg": "Inspect value.py", "files": {"value.py": source.read_text()},
                      "entry_path": "value.py", "base_commit": base}
    owner.current_step = 1
    tools = _tools_for_probe_modes(probe_gate_on=False, observation_probe_on=False)
    runtime = StateFrameRuntime(
        owner, output=output, target=source, model="scripted", tools=tools,
        temperature=0, source_resolver=lambda _path, _symbol: None,
    )
    if before_prepare is not None:
        before_prepare()
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "Task: Inspect value.py"}]
    request = runtime.prepare(messages, uncapped=messages, tools=tools)
    assert request.overflow is None
    return runtime, request


@pytest.mark.parametrize("layout", ["alternates", "worktree"])
def test_first_request_uses_external_git_objects(tmp_path, monkeypatch, git_probe, layout):
    monkeypatch.setenv(signals.MODE_ENV, "on")
    monkeypatch.setenv("STATEFRAME_F4", "off")
    source, base = _repo(tmp_path / "source")
    root = tmp_path / "workspace"
    if layout == "alternates":
        _git(tmp_path, "clone", "-q", "--shared", str(source), str(root))
        objects = root / ".git" / "objects"
        # A relative alternate is relative to the object directory.
        (objects / "info" / "alternates").write_text(
            os.path.relpath(source / ".git" / "objects", objects) + "\n", encoding="utf-8")
        assert not (objects / base[:2] / base[2:]).exists()
    else:
        _git(source, "worktree", "add", "-q", "--detach", str(root))
        assert (root / ".git").is_file()
        common = _git(root, "rev-parse", "--git-common-dir")
        assert (root / common).resolve() == source / ".git"

    runtime, request = _first_request(root, tmp_path / "journal", base=base)
    assert runtime.repo_signals["history"]["allowed_commits"] == [base]
    assert any(message["content"].startswith("[Repository signals ")
               for message in request.messages)


@pytest.mark.parametrize("layout", ["missing-objects", "dangling-alternate", "non-git",
                                    "objects-lost-after-start", "metadata-lost-after-start"])
def test_first_request_skips_unavailable_repo_signals(tmp_path, monkeypatch, git_probe, layout):
    monkeypatch.setenv(signals.MODE_ENV, "on")
    monkeypatch.setenv("STATEFRAME_F4", "off")
    reasons = []
    monkeypatch.setattr(signals, "emit_reason", lambda *args, **kwargs: reasons.append(args))
    base = None
    before_prepare = None
    if layout == "non-git":
        import kernel_safe_probe

        parent, base = _repo(tmp_path / "parent")
        root = parent / "snapshot"
        root.mkdir()
        (root / "value.py").write_text("VALUE = 1\n", encoding="utf-8")

        def unexpected_probe(*args, **kwargs):
            pytest.fail("Git-free workspace queried its enclosing repository")

        monkeypatch.setattr(kernel_safe_probe, "run_probe", unexpected_probe)
    elif layout == "dangling-alternate":
        source, base = _repo(tmp_path / "source")
        root = tmp_path / "workspace"
        _git(tmp_path, "clone", "-q", "--shared", str(source), str(root))
        (source / ".git" / "objects").rename(tmp_path / "unavailable-objects")
    else:
        root, base = _repo(tmp_path / "workspace")
        metadata = root / ".git"
        unavailable = metadata if layout == "metadata-lost-after-start" else metadata / "objects"

        def move_git_path() -> None:
            unavailable.rename(tmp_path / "unavailable-git-path")

        if layout == "missing-objects":
            move_git_path()
        else:
            before_prepare = move_git_path

    runtime, request = _first_request(root, tmp_path / "journal", base=base,
                                      before_prepare=before_prepare)
    if layout == "non-git":
        assert runtime.repo_signals is not None
        history = runtime.repo_signals["history"]
        assert history["reason"] == "no_git_repository"
        assert history["repository"] is None
        assert history["allowed_commits"] == []
        assert any(message["content"].startswith("[Repository signals ")
                   for message in request.messages)
        assert not reasons
        return
    assert (runtime.repo_signals_w0 is not None) == (before_prepare is not None)
    assert runtime.repo_signals is None
    assert not any(message["content"].startswith("[Repository signals ")
                   for message in request.messages)
    assert any(reason[:3] == ("degrade", "repo_signals_git_unavailable", "repo_signals")
               for reason in reasons)


def test_first_request_keeps_workspace_signals_without_git(tmp_path, monkeypatch):
    from stateframe.workcopy import create

    monkeypatch.setenv(signals.MODE_ENV, "on")
    monkeypatch.setenv("STATEFRAME_F4", "off")
    original, base = _repo(tmp_path / "source")
    (original / "test_value.py").write_text(
        "from value import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    root = create(original, tmp_path / "workspace", admit_snapshot=False).copy
    assert not (root / ".git").exists()

    def edit_source() -> None:
        (root / "value.py").write_text("VALUE = 2\n", encoding="utf-8")

    runtime, request = _first_request(root, tmp_path / "journal", base=base,
                                      before_prepare=edit_source)

    assert runtime.repo_signals is not None
    assert runtime.repo_signals_w0["git_repository"] is None
    delta = runtime.repo_signals["task_delta"]
    assert delta["modified"] == ["value.py"]
    assert delta["pre_existing_dirty"] == []
    assert runtime.repo_signals["localization"]["shortlist"][0]["path"] == "value.py"
    assert runtime.repo_signals["history"]["status"] == "unavailable"
    assert runtime.repo_signals["history"]["reason"] == "no_git_repository"
    blocks = [message["content"] for message in request.messages
              if message["content"].startswith("[Repository signals ")]
    assert len(blocks) == 1
    assert "task delta W0->Wt: +0, ~1, -0" in blocks[0]
    assert "test_value.py -> value.py basis=import" in blocks[0]
    assert "co-change: unavailable (no_git_repository)" in blocks[0]


def test_signals_off_preserves_first_request_bytes(tmp_path, monkeypatch):
    root, base = _repo(tmp_path / "workspace")
    (root / ".git" / "objects").rename(tmp_path / "unavailable-objects")

    def unexpected_query(*args, **kwargs):
        pytest.fail("signals off attempted a Git query")

    monkeypatch.setattr(signals, "_run_git", unexpected_query)
    requests = []
    for mode in (None, "off"):
        if mode is None:
            monkeypatch.delenv(signals.MODE_ENV, raising=False)
        else:
            monkeypatch.setenv(signals.MODE_ENV, mode)
        runtime, request = _first_request(root, tmp_path / f"journal-{mode}", base=base)
        assert runtime.repo_signals is None
        requests.append(json.dumps({"messages": request.messages, "tools": request.tools}).encode())
    assert requests[0] == requests[1]
