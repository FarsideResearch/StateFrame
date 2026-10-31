"""Fixture scripts stay parseable and never enter the outer test collection."""
import json
import subprocess
import sys

from ._support import FIXTURES, script_path


def test_all_offline_scripts_are_valid_and_have_steps():
    scripts = sorted((FIXTURES / "scripts").glob("*.script.json"))
    assert scripts
    for script in scripts:
        document = json.loads(script.read_text())
        assert document["model"] == "scripted", script
        assert document["steps"] and all("tool" in step for step in document["steps"]), script


def test_job_script_template_preserves_port_number(tmp_path):
    script = script_path(tmp_path, "jobs.script.json", {"PORT": "8765"})
    steps = json.loads(script.read_text())["steps"]
    assert steps[1]["arguments"]["port"] == 8765
    assert "8765" in steps[2]["arguments"]["command"]


def test_e25_older_run_spec_is_refused_cleanly(tmp_path):
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(json.dumps({"schema": "stateframe-run/0"}))
    result = subprocess.run((sys.executable, "-m", "stateframe", "resume", str(run_dir)),
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "not a stateframe-run/1 run spec" in result.stderr + result.stdout


def test_install_fixture_builds_valid_wheel_without_site_packages(tmp_path):
    import csv
    import io
    import base64
    import hashlib
    import zipfile
    backend = FIXTURES / 'install_repo/localpkg/fixture_backend.py'
    script = tmp_path / 'build_fixture.py'
    script.write_text(
        'import runpy\n'
        f'backend = runpy.run_path({str(backend)!r})\n'
        f'print(backend["build_wheel"]({str(tmp_path)!r}))\n')
    result = subprocess.run([sys.executable, '-I', '-S', str(script)],
                            text=True, capture_output=True, check=True)
    with zipfile.ZipFile(tmp_path / result.stdout.strip()) as wheel:
        rows = csv.reader(io.StringIO(wheel.read('localpkg-0.0.1.dist-info/RECORD').decode()))
        for name, digest, size in rows:
            if not digest:
                continue
            data = wheel.read(name)
            assert int(size) == len(data)
            assert digest == 'sha256=' + base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()).rstrip(b'=').decode()
        assert b'installed' in wheel.read('localpkg/__init__.py')
