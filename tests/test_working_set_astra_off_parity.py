"""Unset allocator preserves provider request bytes at each 0.6.23 ship point."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tools.pinned_tree import pinned_tree


@pytest.mark.parametrize("reference", ['56af0dc6d6'])
def test_off_provider_bytes_match_cutover_base(tmp_path, tmp_path_factory, reference):
    base = pinned_tree(reference, tmp_path_factory)
    current = Path(__file__).resolve().parents[1]
    driver = '''
import json, sys
from pathlib import Path
import pytest
from test_request_determinism import _run_scripted_example, EXAMPLES
with pytest.MonkeyPatch.context() as patch:
    patch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    patch.setenv("SF_LAB_DETFIX3", "on")
    from stateframe import runner
    runner.test_interpreter.choose = lambda *args, **kwargs: (sys.executable, "parity test interpreter")
    hashes, _, _ = _run_scripted_example(patch, "discount_bug", EXAMPLES["discount_bug"], Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(hashes))
'''
    results = []
    for label, tree in (("base", base), ("current", current)):
        destination = tmp_path / f"{label}.json"
        env = {key: value for key, value in os.environ.items() if not key.startswith("STATEFRAME_")}
        env["PYTHONPATH"] = os.pathsep.join(str(tree / path) for path in
            ("src", "src/stateframe/engine", "tests"))
        result = subprocess.run([sys.executable, "-c", driver, str(tmp_path / label), str(destination)],
            cwd=tree, env=env, capture_output=True, text=True, timeout=600, check=False)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        results.append(json.loads(destination.read_text()))
    assert results[0] and results[0] == results[1]
