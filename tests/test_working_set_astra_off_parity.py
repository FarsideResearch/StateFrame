"""Unset allocator preserves provider request bytes at each 0.6.23 ship point."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tools.pinned_tree import pinned_tree


def _provider_hashes(tree, work):
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
    work.mkdir(parents=True, exist_ok=True)
    destination = work / "hashes.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith("STATEFRAME_")}
    env["PYTHONPATH"] = os.pathsep.join(str(tree / path) for path in
        ("src", "src/stateframe/engine", "tests"))
    result = subprocess.run([sys.executable, "-c", driver, str(work / "run"), str(destination)],
        cwd=tree, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    hashes = json.loads(destination.read_text())
    assert hashes
    return hashes


@pytest.fixture(scope="module")
def current_provider_hashes(tmp_path_factory):
    return _provider_hashes(Path(__file__).resolve().parents[1],
                            tmp_path_factory.mktemp("current-provider-parity"))


@pytest.mark.parametrize("reference", ['56af0dc6d6', '76a38627ca'])
def test_off_provider_bytes_match_cutover_base(tmp_path, tmp_path_factory, reference,
                                              current_provider_hashes):
    base = pinned_tree(reference, tmp_path_factory)
    assert _provider_hashes(base, tmp_path) == current_provider_hashes
