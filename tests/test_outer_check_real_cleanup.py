"""Real outer checks must acknowledge containment after ordinary completion."""

import sys

import pytest

from _timeouts import scaled
from stateframe import outer_checks


@pytest.mark.host_sandbox
@pytest.mark.parametrize("exit_code", [0, 7])
def test_outer_check_reports_normal_completion_and_cleanup(tmp_path, exit_code):
    (tmp_path / "check.py").write_text(
        f"import sys\nprint('check-value={exit_code}')\nsys.exit({exit_code})\n",
        encoding="utf-8",
    )

    result = outer_checks.run(
        f"{sys.executable} check.py", tmp_path, python=sys.executable,
        test_python=sys.executable, runner_kind="generic", capture_output=True,
        timeout_s=scaled(20),
    )

    assert result["exit_code"] == exit_code
    assert result["passed"] is (exit_code == 0)
    assert result["output"] == f"check-value={exit_code}\n"
    assert result["process_group_killed"] is True
