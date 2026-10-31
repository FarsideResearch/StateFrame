"""A check command that is not installed is reported as missing, not as a broken sandbox."""
import sys

import pytest

from stateframe import outer_checks

pytestmark = [pytest.mark.host_sandbox,
              pytest.mark.skipif(not sys.platform.startswith("linux"), reason="bwrap execvp diagnostics")]


def test_missing_check_program_names_the_program(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README").write_text("x\n")
    with pytest.raises(outer_checks.OuterCheckError) as caught:
        outer_checks.run("sf-no-such-check-program test", repo, python=sys.executable)
    message = str(caught.value)
    assert "check command not found: sf-no-such-check-program" in message
    assert "sandbox unavailable" not in message
