"""Corpus/record/CLI values must never reach a git argv as an option.

Four sites build git argv from external data (a mined-fix ``sha``, a task
row ``sha``, a dataset ``repo_url`` and an operator-supplied revision).  Each
must reject option-like values on its existing failure path -- never run git
and never crash.  All git calls are faked: no network, no real clones.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import blame_channel
import realtask_benchmark
from build_tasks import swebench_adapter

ROOT = Path(__file__).resolve().parents[1]

BAD_SHAS = ["--output=/tmp/x", "-1", "--all", "abc", "zzzzzzz", "HEAD", "abc123; rm", "", "deadbeef^^"]
GOOD_SHAS = ["deadbeef", "0123456789abcdef0123456789abcdef01234567"]
BAD_URLS = [
    "--upload-pack=touch /tmp/pwned",
    "-c core.sshCommand=touch /tmp/pwned",
    "-",
    "file://-c",
    "ext::sh -c id",
    "ftp://example.invalid/repo.git",
    "/srv/git/repo.git",
    "example.invalid/owner/repo.git",
    "https://",
]
GOOD_URLS = [
    "https://github.com/owner/repo.git",
    "http://git.example.invalid/owner/repo",
    "ssh://git@example.invalid/owner/repo.git",
    "git://example.invalid/owner/repo.git",
    "file:///srv/mirrors/repo.git",
    "git@example.invalid:owner/repo.git",
]


class _GitRan(AssertionError):
    """A faked git call was reached with an unsafe argv."""


def _recording_run(calls):
    def fake_run(argv, *args, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    return fake_run


# ----------------------------------------------------------- blame_channel
@pytest.mark.parametrize("bad", BAD_SHAS)
def test_calibration_cases_drop_non_hex_sha(monkeypatch, tmp_path, bad):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(blame_channel, "mined_fixes",
                        lambda _p=None: [{"repo": "r", "sha": bad, "path": "backend/app/x.py"},
                                         {"repo": "r", "sha": "deadbeef", "path": "backend/app/y.py"}])
    cases = blame_channel.calibration_cases(repo_map={"r": str(tmp_path)})
    assert [c.sha for c in cases] == ["deadbeef"]


@pytest.mark.parametrize("anchor", ["--output=/tmp/x", "-1", "-n5", "--all^"])
def test_walk_commits_from_refuses_option_like_anchor(monkeypatch, tmp_path, anchor):
    calls: list = []
    monkeypatch.setattr(blame_channel.subprocess, "run", _recording_run(calls))
    assert blame_channel.walk_commits_from(str(tmp_path), anchor, "backend", 5) == []
    assert calls == []


@pytest.mark.parametrize("anchor", ["HEAD", "deadbeef^", "deadbeef"])
def test_walk_commits_from_keeps_plain_anchors(monkeypatch, tmp_path, anchor):
    calls: list = []
    monkeypatch.setattr(blame_channel.subprocess, "run", _recording_run(calls))
    blame_channel.walk_commits_from(str(tmp_path), anchor, "backend", 5)
    assert len(calls) == 1 and calls[0][3:5] == ["log", anchor]


# -------------------------------------------------------------- converge
def _load_converge(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["converge"])  # the script parses N/STEPS at import
    path = ROOT / "bench" / "lab" / "converge.py"
    spec = importlib.util.spec_from_file_location("lab_converge_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


@pytest.mark.parametrize("bad", BAD_SHAS + [f"{b}^" for b in BAD_SHAS if b])
def test_converge_archive_refuses_non_hex_revision(monkeypatch, tmp_path, bad):
    converge = _load_converge(monkeypatch)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _recording_run(calls))
    assert converge._extract_git_archive(tmp_path, bad, tmp_path, ("backend",)) is False
    assert calls == []


@pytest.mark.parametrize("good", GOOD_SHAS + [f"{g}^" for g in GOOD_SHAS])
def test_converge_archive_runs_for_hex_revision(monkeypatch, tmp_path, good):
    converge = _load_converge(monkeypatch)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", _recording_run(calls))
    converge._extract_git_archive(tmp_path, good, tmp_path, ("backend",))
    assert calls and calls[0][3:6] == ["archive", good, "backend"]


# ------------------------------------------------------ swebench_adapter
def _record(**overrides):
    base = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "0123456789abcdef0123456789abcdef01234567",
        "patch": "diff --git a/x b/x\n",
        "problem_statement": "it breaks",
        "FAIL_TO_PASS": ["tests/test_x.py::test_x"],
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("bad", BAD_URLS)
def test_swebench_record_rejects_option_like_repo_url(bad):
    with pytest.raises(swebench_adapter.AdapterError, match="repo_url"):
        swebench_adapter.SWEInstance.from_mapping(_record(repo_url=bad), line_no=3)


@pytest.mark.parametrize("bad", BAD_URLS)
def test_swebench_record_rejects_option_like_repository_url_alias(bad):
    with pytest.raises(swebench_adapter.AdapterError, match="repo_url"):
        swebench_adapter.SWEInstance.from_mapping(_record(repository_url=bad))


@pytest.mark.parametrize("good", GOOD_URLS)
def test_swebench_record_keeps_allowlisted_repo_url(good):
    instance = swebench_adapter.SWEInstance.from_mapping(_record(repo_url=good))
    assert instance.repo_url == good
    assert not instance.source_url.startswith("-")


def test_swebench_record_without_repo_url_uses_github():
    instance = swebench_adapter.SWEInstance.from_mapping(_record())
    assert instance.source_url == "https://github.com/owner/repo.git"


# ----------------------------------------------------- realtask_benchmark
@pytest.mark.parametrize("bad", ["-n", "--all", "-", "--output=/tmp/x", ""])
def test_commit_details_refuses_option_like_revision(monkeypatch, tmp_path, bad):
    calls: list = []

    def fake_git_bytes(_repo, *args):
        calls.append(list(args))
        return b""

    monkeypatch.setattr(realtask_benchmark, "_git_bytes", fake_git_bytes)
    with pytest.raises(realtask_benchmark.BenchmarkError):
        realtask_benchmark._commit_details(tmp_path, bad)
    assert calls == []


def test_extract_dev_tasks_skips_option_like_revision(monkeypatch, tmp_path):
    calls: list = []

    def fake_git_bytes(_repo, *args):
        calls.append(list(args))
        return b""

    monkeypatch.setattr(realtask_benchmark, "_git_bytes", fake_git_bytes)
    assert realtask_benchmark.extract_dev_tasks(tmp_path, revisions=["--all"]) == ()
    assert calls == []


def test_commit_details_keeps_plain_revision(monkeypatch, tmp_path):
    calls: list = []

    def fake_git_bytes(_repo, *args):
        calls.append(list(args))
        if args[0] == "rev-list":
            return b"a" * 40 + b" " + b"b" * 40 + b"\n"
        if args[0] == "diff":
            return b"src/x.py\x00tests/test_x.py\x00"
        return b"subject\n"

    monkeypatch.setattr(realtask_benchmark, "_git_bytes", fake_git_bytes)
    try:
        realtask_benchmark._commit_details(tmp_path, "HEAD~1")
    except realtask_benchmark.BenchmarkError:
        pass  # later stages may reject the fake diff; argv was already built
    assert calls and calls[0] == ["rev-list", "--parents", "-n", "1", "HEAD~1"]
