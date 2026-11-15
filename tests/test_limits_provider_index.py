"""The corpus provider index file cap: default, env override, clamp, and reported truncation."""
import pytest

import edit_sets

ENV = edit_sets.PROVIDER_INDEX_MAX_FILES_ENV


def _corpus(count):
    return {f"pkg/mod_{index:05d}.py": "VALUE = 1\n" for index in range(count)}


@pytest.fixture(autouse=True)
def _no_inherited_override(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


def test_default_and_hard_max_constants():
    assert edit_sets.DEFAULT_PROVIDER_INDEX_MAX_FILES == 5000
    assert edit_sets.MAX_PROVIDER_INDEX_MAX_FILES == 200000
    assert edit_sets.provider_index_max_files() == 5000


def test_601_files_are_all_scanned_by_default():
    index = edit_sets.build_provider_index(_corpus(601))
    assert index.files_scanned == 601
    assert index.truncated is False


def test_boundary_at_default_cap():
    at_cap = edit_sets.build_provider_index(_corpus(5000))
    assert (at_cap.files_scanned, at_cap.truncated) == (5000, False)
    over_cap = edit_sets.build_provider_index(_corpus(5001))
    assert (over_cap.files_scanned, over_cap.truncated) == (5000, True)


def test_env_override_truncates_and_reports(monkeypatch):
    monkeypatch.setenv(ENV, "10")
    index = edit_sets.build_provider_index(_corpus(601))
    assert index.truncated is True
    assert index.files_scanned == 10


def test_env_is_read_at_call_time(monkeypatch):
    corpus = _corpus(30)
    assert edit_sets.build_provider_index(corpus).files_scanned == 30
    monkeypatch.setenv(ENV, "7")
    assert edit_sets.build_provider_index(corpus).files_scanned == 7
    monkeypatch.delenv(ENV)
    assert edit_sets.build_provider_index(corpus).files_scanned == 30


def test_whitespace_around_value_is_stripped(monkeypatch):
    monkeypatch.setenv(ENV, "  12  ")
    assert edit_sets.provider_index_max_files() == 12


def test_explicit_max_files_argument_still_wins(monkeypatch):
    monkeypatch.setenv(ENV, "10")
    index = edit_sets.build_provider_index(_corpus(601), max_files=3)
    assert index.files_scanned == 3


@pytest.mark.parametrize("raw", ["", "   ", "abc", "1e3", "2.5", "nan", "inf", "0", "-5"])
def test_invalid_or_non_positive_env_falls_back_to_default(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    assert edit_sets.provider_index_max_files() == 5000


def test_env_value_above_hard_max_is_clamped(monkeypatch):
    monkeypatch.setenv(ENV, "999999999")
    assert edit_sets.provider_index_max_files() == 200000


def test_env_value_at_hard_max_is_kept(monkeypatch):
    monkeypatch.setenv(ENV, "200000")
    assert edit_sets.provider_index_max_files() == 200000


def test_explicit_max_files_argument_is_clamped_to_hard_max(monkeypatch):
    monkeypatch.setattr(edit_sets, "MAX_PROVIDER_INDEX_MAX_FILES", 4)
    index = edit_sets.build_provider_index(_corpus(6), max_files=10**9)
    assert index.files_scanned == 4
    assert index.truncated is True
