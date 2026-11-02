"""STATEFRAME_GIT_COMMIT: git_commit / git_branch record snapshots; the host exports patch series."""
import json
import shutil
import subprocess

import pytest

import kernel_git_commit as commits
from kernel_effects import ContentStore, scan
from kernel_git_view import excluded_path
from stateframe.commit_export import export_commits

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture
def run(tmp_path):
    workspace = tmp_path / "run" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "app.py").write_text("VALUE = 1\n")
    parent = tmp_path / "run" / "parent"
    shutil.copytree(workspace, parent)
    return tmp_path / "run", workspace


def _commit(run_dir, workspace, call_id, message):
    log = commits.CommitLog(run_dir)
    manifest = scan(workspace, exclude=lambda _: False)
    return commits.commit(log, ContentStore(run_dir), manifest, workspace, call_id, message, excluded_path)


def test_commit_records_a_snapshot_and_refuses_an_unchanged_tree(run):
    run_dir, workspace = run
    (workspace / "app.py").write_text("VALUE = 2\n")
    assert _commit(run_dir, workspace, "c1", "Raise value") == "commit 1 on main: Raise value"
    assert _commit(run_dir, workspace, "c2", "Again").startswith("REFUSED NOTHING_TO_COMMIT")
    entries = commits.CommitLog(run_dir).entries()
    assert [entry["call_id"] for entry in entries] == ["c1"]
    tree = json.loads(ContentStore(run_dir).get(entries[0]["tree"]))
    assert set(tree) == {"app.py"}


def test_commit_never_records_secret_paths(run):
    run_dir, workspace = run
    (workspace / ".env").write_text("TOKEN=secret\n")
    (workspace / "app.py").write_text("VALUE = 3\n")
    _commit(run_dir, workspace, "c1", "Change")
    tree = json.loads(ContentStore(run_dir).get(commits.CommitLog(run_dir).entries()[0]["tree"]))
    assert ".env" not in tree and "app.py" in tree


@pytest.mark.parametrize("message", ["", "   ", None, "x" * (commits.MAX_MESSAGE + 1)])
def test_commit_refuses_bad_messages(run, message):
    run_dir, workspace = run
    assert _commit(run_dir, workspace, "c1", message).startswith("REFUSED BAD_ARGUMENT")
    assert commits.CommitLog(run_dir).entries() == []


@pytest.mark.parametrize("name", ["", "-x", "a b", "a/../b", "main", "x.lock", "a" * 64, None])
def test_branch_refuses_bad_or_existing_names(run, name):
    run_dir, _ = run
    reply = commits.branch(commits.CommitLog(run_dir), "b1", name)
    assert reply.startswith("REFUSED"), reply


def test_branch_moves_later_commits(run):
    run_dir, workspace = run
    (workspace / "app.py").write_text("VALUE = 2\n")
    _commit(run_dir, workspace, "c1", "On main")
    assert commits.branch(commits.CommitLog(run_dir), "b1", "feature").startswith("branch feature")
    assert commits.branch(commits.CommitLog(run_dir), "b2", "feature").startswith("REFUSED BRANCH_EXISTS")
    assert _commit(run_dir, workspace, "c2", "Same tree").startswith("REFUSED NOTHING_TO_COMMIT")
    (workspace / "app.py").write_text("VALUE = 3\n")
    assert _commit(run_dir, workspace, "c3", "On feature") == "commit 2 on feature: On feature"


def test_invalid_mode_is_refused(monkeypatch):
    monkeypatch.setenv(commits.MODE_ENV, "yes")
    with pytest.raises(ValueError, match="off or on"):
        commits.enabled()


@needs_git
def test_export_writes_a_series_per_branch_that_git_am_applies(run, tmp_path):
    run_dir, workspace = run
    (workspace / "app.py").write_text("VALUE = 2\n")
    _commit(run_dir, workspace, "c1", "Raise value\n\nBody line.")
    (workspace / "lib.py").write_text("def f():\n    return 1\n")
    _commit(run_dir, workspace, "c2", "Add lib")
    commits.branch(commits.CommitLog(run_dir), "b1", "experiment")
    (workspace / "app.py").unlink()
    _commit(run_dir, workspace, "c3", "Drop app")

    summary = export_commits(run_dir)
    assert summary["count"] == 3
    assert summary["branches"]["main"]["commits"] == 2
    assert summary["branches"]["experiment"]["commits"] == 1
    assert export_commits(run_dir) == summary  # deterministic re-export

    target = tmp_path / "user-repo"
    shutil.copytree(run_dir / "parent", target)
    env = {"GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(target)]
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "base"]):
        subprocess.run(git + args, check=True, env=env)
    patches = sorted((run_dir / "commits" / "main").glob("*.patch"))
    subprocess.run(git + ["am", "-q", *map(str, patches)], check=True, env=env)
    assert (target / "app.py").read_text() == "VALUE = 2\n"
    assert (target / "lib.py").exists()
    log = subprocess.run(git + ["log", "--format=%s"], check=True, env=env, capture_output=True, text=True).stdout
    assert log.splitlines() == ["Add lib", "Raise value", "base"]
    experiment = sorted((run_dir / "commits" / "experiment").glob("*.patch"))
    subprocess.run(git + ["am", "-q", *map(str, experiment)], check=True, env=env)
    assert not (target / "app.py").exists()


def test_completion_route_alignment_keeps_offered_commit_tools(monkeypatch):
    import kernel_completion_owner as owner
    from types import SimpleNamespace
    monkeypatch.setenv(commits.MODE_ENV, "on")
    route = SimpleNamespace(route="ACTION", tool_schemas=({"type": "function", "function": {"name": "done"}},))
    monkeypatch.setattr(owner, "completion_route_decision", lambda *a, **k: route)
    monkeypatch.setattr(owner, "_green_open_work", lambda runtime: False)
    monkeypatch.setattr(owner, "_deferred_source_read_tool", lambda runtime, tools: None)
    tool = lambda name: {"type": "function", "function": {"name": name}}  # noqa: E731
    runtime = SimpleNamespace(journal=None)
    with_commits = owner.align_tools_to_completion_route(
        runtime, [tool("edit"), tool("git_commit"), tool("git_branch"), tool("done")])
    assert [item["function"]["name"] for item in with_commits] == ["done", "git_commit", "git_branch"]
    without = owner.align_tools_to_completion_route(SimpleNamespace(journal=None), [tool("edit"), tool("done")])
    assert [item["function"]["name"] for item in without] == ["done"]


def test_static_registries_name_commit_tools_only_when_switched_on(monkeypatch):
    monkeypatch.delenv(commits.MODE_ENV, raising=False)
    assert commits.registered_names() == frozenset() and commits.registered_schemas() == ()
    monkeypatch.setenv(commits.MODE_ENV, "on")
    assert commits.registered_names() == frozenset(commits.TOOL_NAMES)
    assert commits.registered_schemas() == commits.SCHEMAS


def test_every_static_tool_registry_unions_the_switched_commit_names():
    """Each registry a built-in observation tool lives in must include the commit tools when on."""
    from pathlib import Path
    engine = Path(commits.__file__).resolve().parent
    for name in ("kernel_lean.py", "kernel_frame_routes.py", "kernel_bounded_locate_code.py",
                 "kernel_continuation_evidence.py", "kernel_completion_owner.py", "kernel_tool_surface.py",
                 "probes/react_repair.py"):
        text = (engine / name).read_text(encoding="utf-8")
        assert "registered_names" in text or "registered_schemas" in text, name


def test_registries_include_commit_tools_in_an_engine_started_with_the_switch(tmp_path):
    import sys
    from pathlib import Path
    engine = Path(commits.__file__).resolve().parent
    probe = ("import kernel_lean, kernel_bounded_locate_code, kernel_continuation_evidence, kernel_frame_routes\n"
             "from kernel_completion_owner import TOOL_SPECS\n"
             "names = {'git_commit', 'git_branch'}\n"
             "assert names <= set(TOOL_SPECS)\n"
             "assert names <= kernel_lean._KNOWN_TOOL_NAMES\n"
             "assert names <= kernel_bounded_locate_code.OBSERVATION_TOOLS\n"
             "assert names <= kernel_continuation_evidence.READ_ONLY\n"
             "assert names <= set(kernel_continuation_evidence.READ_RESULT_CLASSES)\n"
             "assert names <= kernel_frame_routes.READ\n")
    for mode, expect in (("on", 0), ("off", 1)):
        env = {"PATH": "/usr/bin:/bin", "STATEFRAME_GIT_COMMIT": mode, "PYTHONPATH": str(engine)}
        done = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, cwd=tmp_path)
        assert (done.returncode != 0) == bool(expect), (mode, done.stderr.decode()[-600:])


def test_export_without_commits_writes_nothing(run):
    run_dir, _ = run
    assert export_commits(run_dir) is None
    commits.branch(commits.CommitLog(run_dir), "b1", "only-branch")
    assert export_commits(run_dir) is None
    assert not (run_dir / "commits").exists()
