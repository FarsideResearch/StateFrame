"""An identity-key cache entry from one test must not leak into the next one.

The release gate runs pytest with tmp_path_retention_policy=failed, which deletes a passing
test's tmp_path and lets the next test reuse that name. A cached key for the reused path then
skipped home creation and broke protected_home_identity(initialize=True).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import kernel_request_auth_identity as auth

ROOT = Path(__file__).resolve().parents[1]
_LEAKED = Path("/nonexistent-stateframe-cache-isolation/home/identity.key")


def test_a_leaves_an_entry_in_the_identity_cache():
    auth._AUTH_BINDING_SECRET_CACHE[os.fspath(_LEAKED)] = b"x" * 32
    auth._AUTH_BINDING_FALLBACK_SECRET = b"y" * 32


def test_b_starts_with_an_empty_identity_cache():
    # Runs after test_a in file order; the autouse fixture in conftest.py cleared the cache.
    assert os.fspath(_LEAKED) not in auth._AUTH_BINDING_SECRET_CACHE
    assert auth._AUTH_BINDING_FALLBACK_SECRET is None


def test_identity_tests_pass_when_pytest_reuses_deleted_tmp_paths(tmp_path):
    home = tmp_path / "gate-home"
    home.mkdir()
    env = {key: value for key, value in os.environ.items()
           if key not in {"STATEFRAME_HOME", "PYTEST_XDIST_WORKER", "PYTEST_ADDOPTS"}}
    env.update(HOME=str(home), TMPDIR=str(home),
               PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "src/stateframe/engine")]))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
         "-n", "0", f"--basetemp={tmp_path / 'inner'}",
         "-o", "tmp_path_retention_count=1", "-o", "tmp_path_retention_policy=failed",
         "tests/test_child_home_identity_authority.py"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
