"""Probe code runs like a script: functions see names defined beside them."""
from __future__ import annotations

import sys

import pytest

from kernel_probe import ProbeRunner, ProbeStatus


@pytest.mark.host_sandbox
@pytest.mark.parametrize("code", [
    "def helper():\n    return 7\n\ndef main():\n    return helper()\n\nprint(main())\n",
    "import math\n\ndef main():\n    return int(math.sqrt(49))\n\nprint(main())\n",
    "LIMIT = 7\n\nclass Box:\n    def value(self):\n        return LIMIT\n\nprint(Box().value())\n",
    "base = 6\nprint(sum(base + extra for extra in (1,)))\n",
], ids=["function", "import", "class", "generator"])
def test_probe_functions_see_names_defined_in_the_same_probe(tmp_path, code):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    receipt = ProbeRunner(repo, executable=sys.executable).run(
        code, "names defined in the probe", feature_enabled=True)
    assert receipt.status == ProbeStatus.COMPLETED, receipt
    assert receipt.exit_code == 0, receipt
    assert "7" in receipt.captured_output, receipt
