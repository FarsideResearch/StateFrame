"""Distribution checks for prebuilt mutation fixture files."""

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path, PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "stateframe"
PREBUILT = PACKAGE / "mutations_prebuilt"
BACKEND = ROOT / "_build" / "sf_backend.py"


def _build_backend():
    spec = importlib.util.spec_from_file_location("stateframe_test_build_backend", BACKEND)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_files():
    return {
        path.relative_to(PACKAGE).as_posix(): path.read_bytes()
        for fixture_root in sorted(path for path in PREBUILT.rglob("fixtures") if path.is_dir())
        for path in sorted(fixture_root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _is_fixture_path(relative):
    parts = PurePosixPath(relative).parts
    return len(parts) >= 3 and parts[0] == "mutations_prebuilt" and "fixtures" in parts[1:-1]


def _assert_fixture_payload(actual, expected):
    missing = sorted(expected.keys() - actual.keys())
    assert not missing, f"missing mutation fixture files: {missing}"
    extra = sorted(actual.keys() - expected.keys())
    assert not extra, f"unexpected mutation fixture files: {extra}"
    mismatched = sorted(name for name, data in expected.items() if actual[name] != data)
    assert not mismatched, f"mutation fixture bytes differ: {mismatched}"


def _wheel_fixtures(archive):
    prefix = "stateframe/"
    return {
        name[len(prefix):]: archive.read(name)
        for name in archive.namelist()
        if name.startswith(prefix) and _is_fixture_path(name[len(prefix):])
    }


def _sdist_fixtures(archive, sdist_root):
    prefix = f"{sdist_root}/src/stateframe/"
    fixtures = {}
    for member in archive.getmembers():
        if not member.isfile() or not member.name.startswith(prefix):
            continue
        relative = member.name[len(prefix):]
        if _is_fixture_path(relative):
            stream = archive.extractfile(member)
            assert stream is not None
            fixtures[relative] = stream.read()
    return fixtures


def _assert_fixture_record(archive, expected):
    record_name = next(name for name in archive.namelist() if name.endswith(".dist-info/RECORD"))
    rows = {
        row[0]: row[1:]
        for row in csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
    }
    for relative, data in expected.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        assert rows[f"stateframe/{relative}"] == [f"sha256={digest}", str(len(data))]


def test_mutation_fixtures_are_in_wheel_and_sdist(tmp_path):
    expected = _fixture_files()
    assert expected
    backend = _build_backend()
    wheel_name = backend.build_wheel(str(tmp_path))
    sdist_name = backend.build_sdist(str(tmp_path))

    with zipfile.ZipFile(tmp_path / wheel_name) as wheel:
        _assert_fixture_payload(_wheel_fixtures(wheel), expected)
        _assert_fixture_record(wheel, expected)

    sdist_root = sdist_name.removesuffix(".tar.gz")
    with tarfile.open(tmp_path / sdist_name, "r:gz") as sdist:
        _assert_fixture_payload(_sdist_fixtures(sdist, sdist_root), expected)


def test_mutation_fixture_content_guard_detects_dropped_file():
    expected = _fixture_files()
    assert expected
    actual = dict(expected)
    actual.pop(sorted(expected)[0])

    with pytest.raises(AssertionError, match="missing mutation fixture files"):
        _assert_fixture_payload(actual, expected)


@pytest.mark.skipif(sys.platform != "linux", reason="mutation sandbox selftest is Linux-only")
def test_installed_wheel_mutation_selftest_is_ok(tmp_path):
    backend = _build_backend()
    wheel_path = tmp_path / backend.build_wheel(str(tmp_path))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update({
        "HOME": str(tmp_path / "home"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONNOUSERSITE": "1",
        "STATEFRAME_HOME": str(tmp_path / "stateframe-home"),
    })
    (tmp_path / "home").mkdir()

    environment_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(environment_dir)
    python = environment_dir / "bin" / "python"
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--no-deps",
         "--disable-pip-version-check", str(wheel_path)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=180,
    )
    assert install.returncode == 0, f"wheel install failed:\n{install.stdout}\n{install.stderr}"

    result = subprocess.run(
        [str(environment_dir / "bin" / "stateframe"), "mutations", "selftest", "--json"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"installed selftest failed:\n{result.stdout}\n{result.stderr}"
    report = json.loads(result.stdout)
    assert report.get("ok") is True, json.dumps(report, sort_keys=True)
