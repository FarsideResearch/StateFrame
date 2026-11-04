"""The expected_installer_leak marker: a marked test must leak (and gets restored); unmarked leaks fail."""
from __future__ import annotations

import os
from pathlib import Path

pytest_plugins = ["pytester"]

TESTS = Path(__file__).resolve().parent

INNER = '''
import pytest
import kernel_bounded_http

REAL = kernel_bounded_http.request_bytes


@pytest.mark.expected_installer_leak
def test_marked_and_leaks():
    kernel_bounded_http.request_bytes = lambda *args, **kwargs: b"leaked"


def test_sender_was_restored_after_the_marked_leak():
    assert kernel_bounded_http.request_bytes is REAL


@pytest.mark.expected_installer_leak
def test_marked_but_leaks_nothing():
    pass


def test_unmarked_leak():
    kernel_bounded_http.request_bytes = lambda *args, **kwargs: b"leaked"
'''


def test_marker_requires_a_leak_and_restores_it(pytester):
    pytester.makepyfile(test_inner=INNER)
    # Load the real tests/conftest.py as a plugin so its guard runs with its own paths.
    pytester._monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(TESTS), os.environ.get("PYTHONPATH", "")]))
    result = pytester.runpytest_subprocess("-p", "conftest", "-p", "no:cacheprovider", "-q", "-rA")
    # Teardown failures are reported as errors on top of the passed call phase.
    result.assert_outcomes(passed=4, errors=2)
    result.stdout.fnmatch_lines([
        "*test_marked_but_leaks_nothing*is marked expected_installer_leak but leaked nothing*",
        "*test_unmarked_leak*leaked child installer state: kernel_bounded_http.request_bytes*",
    ])
    assert "PASSED test_inner.py::test_marked_and_leaks" in result.stdout.str()
    assert "PASSED test_inner.py::test_sender_was_restored_after_the_marked_leak" in result.stdout.str()
    assert "ERROR test_inner.py::test_marked_and_leaks" not in result.stdout.str()
