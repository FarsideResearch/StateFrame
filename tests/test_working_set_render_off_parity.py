"""Whole-request OFF parity against the assigned cutover base."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.pinned_tree import pinned_tree


@pytest.mark.parametrize("reference", ['56af0dc6d6', '144281ff7c'])
def test_off_wire_bytes_match_1de34aa4(tmp_path, tmp_path_factory, reference):
    root = Path(__file__).resolve().parents[1]
    base = pinned_tree(reference, tmp_path_factory)
    driver = '''
import json, sys
from pathlib import Path
import pytest
from test_request_determinism import _run_scripted_example, EXAMPLES
with pytest.MonkeyPatch.context() as patch:
    patch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    patch.setenv("SF_LAB_DETFIX3", "on")
    hashes, _, _ = _run_scripted_example(patch, "discount_bug", EXAMPLES["discount_bug"], Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(hashes))
'''
    hashes = []
    for label, tree in (("base", base), ("current", root)):
        destination = tmp_path / f"{label}.json"
        environment = {k: v for k, v in os.environ.items() if not k.startswith("STATEFRAME_")}
        environment["PYTHONPATH"] = os.pathsep.join(str(tree / path) for path in
            ("src", "src/stateframe/engine", "tests"))
        result = subprocess.run([sys.executable, "-c", driver, str(tmp_path / label),
            str(destination)], cwd=tree, env=environment, capture_output=True,
            text=True, timeout=600)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        hashes.append(json.loads(destination.read_text()))
    assert hashes[0] and hashes[0] == hashes[1]
