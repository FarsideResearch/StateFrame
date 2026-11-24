"""Test helpers for isolating the process-wide reason log around journal tests."""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def fresh_reason_log():
    import kernel_reason_log

    kernel_reason_log.reset_for_tests()
    yield
    kernel_reason_log.reset_for_tests()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Emit one WARN before a test when exercising journal isolation."""
    if os.environ.get("TEST_WARN_REASON_PLUGIN") != "1":
        return
    import kernel_reason_log

    kernel_reason_log.reset_for_tests()
    kernel_reason_log.emit_reason(
        "fallback", "test_warning_plugin", "test_warning_plugin",
        "test warning", "warning before test",
    )
