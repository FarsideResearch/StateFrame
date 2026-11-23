from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "9.0.3"
WHEEL_NAME = f"pytest-{VERSION}-py3-none-any.whl"
WHEEL_SHA256 = "2c5efc453d45394fdd706ade797c0a81091eccd1d6e4bccfcd476e2b8e0ab5d9"
SDIST_SHA256 = "b86ada508af81d19edeb213c681b1d48246c1a91d304c6c81a427674c17eb91c"


def _pinned_requirement(path: Path, name: str) -> str | None:
    prefix = f"{name.lower()}=="
    matches = [
        line.strip().removesuffix("\\").strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().lower().startswith(prefix)
    ]
    return matches[0] if len(matches) == 1 else None


def _requirement_hashes(path: Path, name: str) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"{name.lower()}=="
    start = next(
        (index for index, line in enumerate(lines) if line.strip().lower().startswith(prefix)),
        None,
    )
    if start is None:
        return set()

    hashes: set[str] = set()
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        if not line[:1].isspace():
            break
        value = line.strip()
        if value.endswith("\\"):
            value = value[:-1].strip()
        if value.startswith("--hash=sha256:"):
            hashes.add(value.removeprefix("--hash=sha256:"))
    return hashes


def test_fixed_release_matches_runtime_and_offline_install_pins():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    runtime_pins = [
        item for item in project["dependencies"]
        if item.split("==", 1)[0].lower() == "pytest"
    ]
    assert runtime_pins == [f"pytest=={VERSION}"]

    online_requirements = ROOT / "requirements.txt"
    assert _pinned_requirement(online_requirements, "pytest") == f"pytest=={VERSION}"
    assert _requirement_hashes(online_requirements, "pytest") == {WHEEL_SHA256, SDIST_SHA256}

    offline_requirements = ROOT / "wheelhouse" / "requirements.txt"
    assert _pinned_requirement(offline_requirements, "pytest") == f"pytest=={VERSION}"


def test_build_inventory_reports_the_fixed_release():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_wheelhouse.py"), "--print-requirements"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    pytest_pins = [line for line in result.stdout.splitlines() if line.lower().startswith("pytest==")]
    assert pytest_pins == [f"pytest=={VERSION}"]


def test_bundled_license_inventories_match_the_fixed_release():
    for path in (ROOT / "THIRD_PARTY_LICENSES.md", ROOT / "src/stateframe/THIRD_PARTY_LICENSES.md"):
        content = path.read_text(encoding="utf-8")
        assert "| pytest | 9.0.3 | MIT |" in content
        assert "pytest-9.0.3-py3-none-any.whl" in content
        assert "### pytest 9.0.3" in content
        assert "pytest 9.0.2" not in content


def test_fixed_wheel_matches_package_metadata_and_offline_hashes():
    wheel_path = ROOT / "wheelhouse" / WHEEL_NAME
    assert wheel_path.is_file(), f"fixed wheel is missing: {WHEEL_NAME}"
    assert not (ROOT / "wheelhouse" / "pytest-9.0.2-py3-none-any.whl").exists()

    actual_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    assert actual_digest == WHEEL_SHA256
    manifest_lines = (ROOT / "wheelhouse" / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    matching_entries = [line.split(maxsplit=1) for line in manifest_lines if line.endswith(WHEEL_NAME)]
    assert matching_entries == [[WHEEL_SHA256, WHEEL_NAME]]

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_names = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_names) == 1
        metadata = BytesParser().parsebytes(wheel.read(metadata_names[0]))
    assert metadata.get("Name", "").lower() == "pytest"
    assert metadata.get("Version") == VERSION


def test_built_source_archive_contains_the_fixed_wheel(tmp_path: Path):
    backend_path = ROOT / "_build" / "sf_backend.py"
    spec = importlib.util.spec_from_file_location("source_archive_backend", backend_path)
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)

    archive_path = tmp_path / backend.build_sdist(str(tmp_path))
    with tarfile.open(archive_path) as archive:
        members = archive.getnames()
        package_root = next(
            name.partition("/")[0] for name in members if name.endswith("/PKG-INFO")
        )
        wheel_member = f"{package_root}/wheelhouse/{WHEEL_NAME}"
        assert wheel_member in members
        assert f"{package_root}/wheelhouse/requirements.txt" in members
        assert f"{package_root}/wheelhouse/SHA256SUMS" in members
        assert f"{package_root}/docs/guide/contracts/offline-pytest.md" in members
        payload = archive.extractfile(wheel_member)
        assert payload is not None
        assert hashlib.sha256(payload.read()).hexdigest() == WHEEL_SHA256
        assert f"{package_root}/wheelhouse/pytest-9.0.2-py3-none-any.whl" not in members
