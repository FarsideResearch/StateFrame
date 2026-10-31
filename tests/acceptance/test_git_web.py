"""Offline Git history and web policy cases through public tool surfaces."""
from __future__ import annotations

import subprocess

import pytest

from ._support import copy_repo, run_case


pytestmark = pytest.mark.host_sandbox


def _git(tmp_path):
    repo = copy_repo(tmp_path)
    for args in (("init", "-q"), ("config", "user.name", "Acceptance"),
                 ("config", "user.email", "acceptance@example.invalid")):
        subprocess.run(("git", *args), cwd=repo, check=True, capture_output=True)
    for index in range(3):
        (repo / "app.py").write_text(f'VALUE = "before"  # commit-{index}\n')
        (repo / ".env").write_text(f"SECRET=history-{index}\n")
        subprocess.run(("git", "add", "app.py", ".env", "Makefile", "tests/test_app.py"),
                       cwd=repo, check=True, capture_output=True)
        subprocess.run(("git", "commit", "-qm", f"commit {index}"),
                       cwd=repo, check=True, capture_output=True)
    return repo


def test_u9_git_history_and_excluded_secret(tmp_path):
    repo = _git(tmp_path)
    case = run_case(tmp_path, "git.script.json", repo=repo)
    case.assert_offered("git")
    case.assert_reply("commit 2")
    case.assert_reply("commit-1")
    case.assert_reply("REFUSED EXCLUDED_PATH:")
    assert "SECRET=history" not in case.transcript
    assert not (case.run_dir / "workspace" / ".env").exists()


def test_u9_git_in_non_git_repo_is_typed_refusal(tmp_path):
    case = run_case(tmp_path, "git_nongit.script.json", force_missing_tool=True)
    case.assert_not_offered("git")
    case.assert_reply("REFUSED NOT_A_GIT_REPO:")


def test_u11_private_address_refused_when_web_enabled(tmp_path):
    case = run_case(tmp_path, "web.script.json", flags=("--web",))
    case.assert_offered("web_fetch")
    case.assert_reply("REFUSED WEB_REFUSED_ADDRESS:")


def test_u11_web_not_offered_by_default(tmp_path):
    case = run_case(tmp_path, "web.script.json", force_missing_tool=True)
    case.assert_not_offered("web_fetch")
    assert "WEB_REFUSED_ADDRESS" not in case.transcript
