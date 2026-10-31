from pathlib import Path
import subprocess

import pytest

from stateframe.engine.kernel_git_view import GitView, engine_diff, handle_git_call
from stateframe.workcopy import create


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def runner(argv, env, read_roots):
    assert all(root.is_dir() for root in read_roots)
    result = subprocess.run(argv, env=env, capture_output=True, check=False)
    return result.returncode, result.stdout + result.stderr


@pytest.fixture
def repo(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-q")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.com")
    (source / "public.txt").write_text("before\n")
    (source / "deleted.txt").write_text("gone\n")
    (source / ".env").write_text("API_SECRET=never-expose\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "initial")
    return source


def view_for(repo, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    create(repo, run / "workspace")
    import shutil
    shutil.copytree(run / "workspace", run / "parent")
    view = GitView.build(repo, run, secret_filter=lambda path: path.endswith(".env"))
    assert view is not None
    return view


def test_pinned_view_excludes_secrets_and_preserves_deletions(repo, tmp_path):
    view = view_for(repo, tmp_path)
    (view.workspace / "deleted.txt").unlink()
    (view.workspace / "added.txt").write_text("added\n")
    status, receipt = handle_git_call(view, runner, {"action": "status"})
    assert receipt["ok"]
    assert "D deleted.txt" in status and "A added.txt" in status
    diff, receipt = handle_git_call(view, runner, {"action": "diff"})
    assert receipt["ok"] and "--- a/deleted.txt" in diff and "+++ /dev/null" in diff
    assert "added.txt" in diff
    for args in ({"action": "show", "rev": "HEAD:.env"},
                 {"action": "show", "rev": "HEAD", "path": ".env"},
                 {"action": "blame", "path": ".env"}):
        text, receipt = handle_git_call(view, runner, args)
        assert not receipt["ok"] and "never-expose" not in text
    shown, receipt = handle_git_call(view, runner, {"action": "show", "rev": "HEAD"})
    assert receipt["ok"] and "never-expose" not in shown
    output, receipt = handle_git_call(view, runner, {"action": "diff", "against": "HEAD"})
    assert receipt["ok"] and "deleted.txt" in output
    assert "added.txt" in output
    shown, receipt = handle_git_call(view, runner, {"action": "diff", "paths": ["added.txt"]})
    assert receipt["ok"] and "added.txt" in shown and "deleted.txt" not in shown


def test_git_rejects_option_injection(repo, tmp_path):
    view = view_for(repo, tmp_path)
    for args in ({"action": "show", "rev": "--output=/tmp/x"},
                 {"action": "log", "since_rev": "-c"},
                 {"action": "diff", "paths": ["--exec"]},
                 {"action": "blame", "path": "-c"}):
        _, receipt = handle_git_call(view, runner, args)
        assert receipt["refusal"] == "invalid_request"


def test_malicious_repository_configuration_is_ignored(repo, tmp_path):
    marker = tmp_path / "marker"
    script = tmp_path / "malicious.sh"
    script.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    script.chmod(0o755)
    config = repo / ".git" / "config"
    with config.open("a") as output:
        output.write(f"\n[core]\n\tsshCommand = {script}\n\tfsmonitor = {script}\n\thooksPath = {script}\n")
        output.write(f"[include]\n\tpath = {script}\n[alias]\n\tshow = !{script}\n")
        output.write(f"[diff \"external\"]\n\ttextconv = {script}\n")
    view = view_for(repo, tmp_path)
    for action in ("status", "diff", "show", "log", "blame"):
        args = {"action": action}
        if action == "blame":
            args["path"] = "public.txt"
        text, receipt = handle_git_call(view, runner, args)
        assert receipt["ok"], text
    assert not marker.exists()


def test_worktree_and_shallow_clone(repo, tmp_path):
    worktree = tmp_path / "worktree"
    git(repo, "worktree", "add", "-q", "--detach", str(worktree))
    view = view_for(worktree, tmp_path)
    _, receipt = handle_git_call(view, runner, {"action": "log"})
    assert receipt["ok"]
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth=1", repo.as_uri(), str(shallow)], check=True)
    run = tmp_path / "other-run"
    run.mkdir()
    create(shallow, run / "workspace")
    import shutil
    shutil.copytree(run / "workspace", run / "parent")
    second = GitView.build(shallow, run, secret_filter=lambda path: path == ".env")
    assert second is not None
    _, receipt = handle_git_call(second, runner, {"action": "log"})
    assert receipt["ok"]


def test_engine_diff_union_and_missing_repo(tmp_path):
    base = tmp_path / "base"
    work = tmp_path / "work"
    base.mkdir()
    work.mkdir()
    (base / "deleted.txt").write_text("gone\n")
    (work / "new.txt").write_text("new\n")
    (base / ".env").write_text("secret\n")
    patch = engine_diff(base, work)
    assert "deleted.txt" in patch and "new.txt" in patch and "secret" not in patch
    assert GitView.build(base, tmp_path / "run", secret_filter=lambda _: False) is None


def test_missing_pinned_object_is_typed_failure(repo, tmp_path):
    view = view_for(repo, tmp_path)
    blob = git(repo, "rev-parse", "HEAD:public.txt")
    object_file = repo / ".git" / "objects" / blob[:2] / blob[2:]
    object_file.unlink()
    text, receipt = handle_git_call(view, runner, {"action": "diff", "against": "HEAD"})
    assert receipt["refusal"] == "object_unavailable"
    assert "pinned object unavailable" in text


def test_partial_clone_does_not_lazy_fetch(repo, tmp_path):
    git(repo, "config", "uploadpack.allowFilter", "true")
    partial = tmp_path / "partial"
    subprocess.run(["git", "clone", "-q", "--no-local", "--no-checkout",
                    "--filter=blob:none", repo.as_uri(), str(partial)], check=True)
    (partial / "public.txt").write_text("before\n")
    run = tmp_path / "partial-run"
    run.mkdir()
    create(partial, run / "workspace")
    import shutil
    shutil.copytree(run / "workspace", run / "parent")
    view = GitView.build(partial, run, secret_filter=lambda _: False)
    assert view is not None
    text, receipt = handle_git_call(view, runner, {"action": "diff", "against": "HEAD"})
    assert receipt["refusal"] == "object_unavailable"
    assert "unavailable" in text


def test_custom_secret_filter_applies_before_history_read(repo, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    create(repo, run / "workspace")
    import shutil
    shutil.copytree(run / "workspace", run / "parent")
    view = GitView.build(repo, run, secret_filter=lambda path: path == "public.txt")
    assert view is not None
    _, receipt = handle_git_call(view, runner, {"action": "show", "path": "public.txt"})
    assert receipt["refusal"] == "invalid_request"


def test_head_and_refs_are_pinned_after_source_moves(repo, tmp_path):
    view = view_for(repo, tmp_path)
    (repo / "public.txt").write_text("later\n")
    git(repo, "add", "public.txt")
    git(repo, "commit", "-qm", "later change")
    shown, receipt = handle_git_call(view, runner, {"action": "show", "path": "public.txt"})
    assert receipt["ok"] and shown == "before\n"
    output, receipt = handle_git_call(view, runner, {"action": "log"})
    assert receipt["ok"] and "later change" not in output


def test_git_observation_is_bounded(repo, tmp_path):
    (repo / "large.txt").write_text("x" * 90_000 + "\n")
    git(repo, "add", "large.txt")
    git(repo, "commit", "-qm", "large file")
    view = view_for(repo, tmp_path)
    shown, receipt = handle_git_call(view, runner, {"action": "show", "path": "large.txt"})
    assert receipt["ok"] and receipt["truncated"]
    assert "git output truncated" in shown and receipt["total_bytes"] > 90_000


def test_quoted_secret_path_and_oversized_workspace_file(repo, tmp_path):
    (repo / "private note.txt").write_text("hidden token\n")
    git(repo, "add", "private note.txt")
    git(repo, "commit", "-qm", "add note")
    run = tmp_path / "run"
    run.mkdir()
    create(repo, run / "workspace")
    import shutil
    shutil.copytree(run / "workspace", run / "parent")
    view = GitView.build(repo, run, secret_filter=lambda path: path == "private note.txt")
    assert view is not None
    shown, receipt = handle_git_call(view, runner, {"action": "show"})
    assert receipt["ok"] and "hidden token" not in shown
    (view.workspace / "huge.txt").write_bytes(b"x" * ((1 << 20) + 1))
    text, receipt = handle_git_call(view, runner, {"action": "status"})
    assert receipt["refusal"] == "invalid_request" and "size limit" in text
