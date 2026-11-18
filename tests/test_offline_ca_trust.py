"""Offline sandboxed commands must see the public CA trust store.

Debian's bundled pip wheel points certifi at /etc/ssl/certs/ca-certificates.crt and loads it when
imported, so `ensurepip` inside an offline probe failed and no run env was built (checks then ran a
bare `python` that does not exist).
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import kernel_exec_capability
from kernel_exec_capability import shell_capability

_CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")
linux_ca = pytest.mark.skipif(sys.platform == "darwin" or not _CA_BUNDLE.is_file(),
                              reason="needs a Linux system CA bundle at /etc/ssl/certs")


def _cap(tmp_path, network):
    for name in ("workspace", "env", "home"):
        (tmp_path / name).mkdir(exist_ok=True)
    return shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                            scratch_home=tmp_path / "home", network=network, writable=False)


@linux_ca
@pytest.mark.parametrize("network", ["none", "full"])
def test_offline_and_online_capabilities_admit_public_ca_store(tmp_path, network):
    roots = set(_cap(tmp_path, network).read_roots)
    # The bundle must be readable: some admitted root is the bundle or one of its ancestors.
    assert any(root == _CA_BUNDLE or root in _CA_BUNDLE.parents for root in roots)


@linux_ca
def test_private_keys_and_rest_of_etc_stay_hidden_offline(tmp_path):
    roots = _cap(tmp_path, "none").read_roots
    assert Path("/etc") not in roots and Path("/etc/ssl") not in roots
    assert not any(root == Path("/etc/ssl/private") or Path("/etc/ssl/private") in root.parents
                   for root in roots)
    helper = getattr(kernel_exec_capability, "public_ca_read_roots", tuple)
    assert all(Path("/etc/ssl/private") not in path.parents for path in helper())


@linux_ca
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs the bwrap sandbox")
def test_ensurepip_succeeds_inside_offline_probe():
    from kernel_safe_probe import run_probe

    base = Path(tempfile.mkdtemp(prefix="ensurepip-ca-"))
    try:
        venv = base / "v"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages",
                        str(venv)], check=True)
        with tempfile.TemporaryDirectory(dir=str(base)) as workspace:
            result = run_probe([str(venv / "bin" / "python3"), "-m", "ensurepip", "--default-pip"],
                               Path(workspace), timeout=120, write_roots=(venv,))
        output = result.output.decode(errors="replace")
        assert result.exit_code == 0, output[-2000:]
        assert "FileNotFoundError" not in output
    finally:
        shutil.rmtree(base, ignore_errors=True)
