"""A startup hint must be a public source, even when tests are editable."""

from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import subprocess
import sys

import pytest

from stateframe.task_row import TaskRowError, _public_hint, build_row
from stateframe.workcopy import create
from acceptance._support import assert_tree_imports, tree_python


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", (
    "shop/pricing.py", "site/README.md", "tests/test_site.py", "test_site.py",
    "suite/site_test.py", "fixtures/source.py", ".hidden/source.py",
    "private/source.py", "oracle/source.py", "conftest.py", "__init__.py",
    "/tmp/source.py", "shop/../source.py",
))
def test_start_hint_public_policy_matches_engine(path):
    from stateframe.engine.kernel_aha11_native_honesty import _public_path

    try:
        _public_path(path)
    except ValueError:
        assert not _public_hint(path)
    else:
        assert _public_hint(path)


@pytest.mark.parametrize(
    ("example", "task", "expected"),
    [
        ("tiny_site", "Build a shop in site/ with site/index.html. Make tests/test_site.py pass.",
         "site/README.md"),
        ("brief_to_report", "Read input/brief.md and write report/summary.json and "
         "report/report.md. Make tests/test_report.py pass.", "report/README.md"),
    ],
)
def test_create_examples_start_at_declared_output(tmp_path, example, task, expected):
    work = create(ROOT / "examples" / example, tmp_path / "copy")
    row = build_row(work, task=task, protect_tests="off")
    assert row["entry"]["path"] == expected
    assert row["path"] == expected


@pytest.mark.parametrize("protection", ("off", "all"))
@pytest.mark.parametrize("runner_kind", ("pytest", "generic"))
def test_tests_only_repo_starts_at_task_declared_output(tmp_path, protection, runner_kind):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests/test_output.py").write_text(
        "from pathlib import Path\n"
        "def test_output():\n"
        "    assert Path('output/result.txt').is_file()\n",
        encoding="utf-8",
    )
    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Create output/result.txt. Make tests/test_output.py pass.",
                    protect_tests=protection, runner_kind=runner_kind,
                    test_command=("python", "-m", "unittest", "discover")
                    if runner_kind == "generic" else ())
    assert row["entry"]["path"] == "output/result.txt"
    assert row["path"] == "output/result.txt"
    assert "output/result.txt" not in row["files"]
    assert row["new_file_roots"] == ["output"]
    from stateframe.engine.capability_study_support import public_parent_row
    from stateframe.engine.kernel_aha11_native_honesty import public_solver_row
    from stateframe.engine.probes.react_repair import FileSet

    assert public_parent_row(row)["parent_src"] == ""
    solver = public_solver_row(public_parent_row(row))
    assert solver["path"] == "output/result.txt"
    assert FileSet.from_row(solver).primary_path == "output/result.txt"


def test_read_inputs_do_not_steal_output_hint(tmp_path):
    repo = tmp_path / "repo"
    (repo / "input").mkdir(parents=True)
    (repo / "output").mkdir()
    (repo / "tests").mkdir()
    (repo / "input/README.md").write_text("Inputs\n", encoding="utf-8")
    (repo / "output/README.md").write_text("Outputs\n", encoding="utf-8")
    (repo / "tests/test_output.py").write_text(
        "def test_output(): assert False\n", encoding="utf-8"
    )
    work = create(repo, tmp_path / "copy")
    row = build_row(work, task="Read input/data.csv and write output/result.csv.",
                    protect_tests="off")
    assert row["entry"]["path"] == "output/README.md"


def test_tests_only_repo_without_output_has_no_starting_hint(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests/test_output.py").write_text(
        "def test_output(): assert False\n", encoding="utf-8"
    )
    work = create(repo, tmp_path / "copy")
    with pytest.raises(TaskRowError, match="could not pick a public starting file"):
        build_row(work, task="Make the checks pass.", protect_tests="off")


@pytest.mark.parametrize("relative", (
    "tests/test_output.py", "test_output.py", ".hidden/source.py",
))
def test_source_window_ignores_nonpublic_target(tmp_path, relative):
    from stateframe.engine.kernel_request_source_window import bound_runtime_sources

    root = tmp_path / "repo"
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_text("def test_output(): pass\n", encoding="utf-8")
    runtime = SimpleNamespace(target=target, owner=SimpleNamespace(root=root))
    assert bound_runtime_sources([], runtime, max_request_bytes=4096) == {
        "changed": False, "windowed": 0, "paths": (),
    }


def test_new_output_has_verified_empty_source_scope(tmp_path):
    from stateframe.engine.kernel_request_source_window import (
        bound_runtime_sources, verify_final_source_delivery,
    )

    root = tmp_path / "repo"
    root.mkdir()
    owner = SimpleNamespace(root=root, task_row={
        "msg": "Create output/result.txt.", "new_file_roots": ["output"],
    })
    runtime = SimpleNamespace(target=root / "output/result.txt", owner=owner,
                              product_revision="0")
    assert bound_runtime_sources([], runtime, max_request_bytes=4096)["changed"] is False
    plan = runtime._request_source_closure
    assert plan["new_output_target"] == "output/result.txt"
    receipt = verify_final_source_delivery(plan, [], payload={"messages": []}, sources={})
    assert receipt["status"] == "COMPLETE"
    assert receipt["new_output_without_source"] is True


@pytest.mark.host_sandbox
@pytest.mark.parametrize("protection", ("off", "all"))
def test_tests_only_repo_can_submit_declared_output(tmp_path, protection):
    from stateframe.scripted_endpoint import Script, serve

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests/test_output.py").write_text(
        "from pathlib import Path\n"
        "def test_output():\n"
        "    assert Path('output/result.txt').read_text() == 'ready'\n",
        encoding="utf-8",
    )
    script = Script({"model": "scripted", "steps": [
        {"tool": "file_create", "required": True,
         "arguments": {"path": "output/result.txt", "content": "ready"}},
        {"tool": "run_tests", "required": True,
         "arguments": {"target": "tests/test_output.py"}},
        {"tool": "done", "required": True,
         "arguments": {"summary": "Created the requested output."}},
    ], "tokenize": {"mode": "ok"}})
    run = tmp_path / "run"
    python = tree_python(ROOT / "src", tmp_path / "current-python")
    assert_tree_imports(python, ROOT / "src", tmp_path)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    server = serve(script)
    try:
        result = subprocess.run(
            [python, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Create output/result.txt. Make tests/test_output.py pass.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
             "--test-python", sys.executable, "--run-dir", str(run),
             *(["--protect-tests", "all"] if protection == "all" else [])],
            capture_output=True, text=True, timeout=240, env=environment,
        )
        summary = json.loads((run / "summary.json").read_text())
        assert result.returncode == 0, result.stdout + result.stderr
        assert summary["provider_sends"] >= 1
        assert summary["engine_result"] == "SUBMITTED"
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.host_sandbox
def test_existing_discount_requests_match_untouched_0_2_8(tmp_path):
    _assert_discount_requests_match_0_2_8(tmp_path)


@pytest.mark.host_sandbox
def test_0_2_8_parity_ignores_a_powerup_enabled_in_the_inherited_home(tmp_path, monkeypatch):
    # Planted: the home this process hands to subprocesses has a powerup enabled, as when an earlier test left one in
    # the session home. The current tree would apply it and 0.2.8 cannot, so each capture must get its own home.
    # The planted home is private: the real session home is shared by every xdist worker of a gated run.
    from stateframe import powerups

    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "polluted-home"))
    installed = powerups.install(ROOT / "examples/powerups/py-format")
    powerups.enable(installed.name, mode="on")
    _assert_discount_requests_match_0_2_8(tmp_path)


def _assert_discount_requests_match_0_2_8(tmp_path):
    from tools.compare_no_mcp_requests import capture, write_launcher

    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "--format=tar", "f848c2e"],
        capture_output=True, check=True,
    ).stdout
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    subprocess.run(["tar", "-xf", "-", "-C", str(baseline)], input=archive, check=True)
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    launcher = tmp_path / "launcher.py"
    write_launcher(launcher)
    run_dir = tmp_path / "run"
    script = ROOT / "examples/discount_bug.script.json"
    baseline_python = tree_python(baseline / "src", tmp_path / "baseline-python")
    assert_tree_imports(baseline_python, baseline / "src", tmp_path)
    before = capture(baseline, baseline_python, launcher, script, repo, run_dir, stable_tool_head=None,
                     source_on_pythonpath=False, test_python=sys.executable)
    run_dir.rename(tmp_path / "baseline-run")
    current_python = tree_python(ROOT / "src", tmp_path / "current-python")
    assert_tree_imports(current_python, ROOT / "src", tmp_path)
    # 0.3.13 turned the stable tool head on by default; switched off, request bytes stay those of 0.2.8.
    after = capture(ROOT, current_python, launcher, script, repo, run_dir, stable_tool_head="off",
                    source_on_pythonpath=False, test_python=sys.executable)
    assert before == after
