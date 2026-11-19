"""Dependency traversal and platform coverage regressions for the offline builder."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wheelhouse_builder", ROOT / "scripts/build_wheelhouse.py")
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def wheel(directory, name, *, version="1.0", tag="py3-none-any", dependencies=(), python=">=3.11"):
    normalized = name.replace("-", "_")
    info = f"{normalized}-{version}.dist-info"
    payload = {
        f"{info}/METADATA": (
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\nLicense-Expression: MIT\n"
            f"Requires-Python: {python}\n" + "".join(f"Requires-Dist: {d}\n" for d in dependencies)
        ).encode(),
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n".encode(),
    }
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for path, data in payload.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((path, f"sha256={digest}", len(data)))
    writer.writerow((f"{info}/RECORD", "", ""))
    payload[f"{info}/RECORD"] = record.getvalue().encode()
    path = directory / f"{normalized}-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for filename, data in payload.items():
            archive.writestr(filename, data)
    return path


def index(directory, pins):
    (directory / "requirements.txt").write_text("".join(f"{p}\n" for p in pins))
    (directory / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(directory.glob("*.whl"))
    ))


def resolve(paths, roots):
    return builder._runtime_closure(lambda name: paths.get(name, []), roots)


def test_transitive_markers_use_each_target_instead_of_build_host(tmp_path):
    root = wheel(tmp_path, "root", dependencies=(
        'old-python; python_version < "3.13"',
        'linux-arm; sys_platform == "linux" and platform_machine == "aarch64"',
        'mac-intel; platform_system == "Darwin" and platform_machine == "x86_64"',
        'never; os_name == "nt"',
        'never-extra; extra == "test"',
    ))
    paths = {name: [wheel(tmp_path, name)] for name in ("old-python", "linux-arm", "mac-intel")}
    paths["root"] = [root]
    assert set(resolve(paths, ["root==1.0"])) == set(paths)


def test_transitive_requirement_missing_only_on_older_python_is_named(tmp_path):
    root = wheel(tmp_path, "root", dependencies=('missing; python_version < "3.13"',))
    with pytest.raises(builder.WheelhouseError, match=r"missing.*CPython 3.11"):
        resolve({"root": [root]}, ["root==1.0"])


def test_extra_activated_transitively_is_revisited_and_cycles_terminate(tmp_path):
    paths = {
        "root": [wheel(tmp_path, "root", dependencies=("child", "child[feature]"))],
        "child": [wheel(tmp_path, "child", dependencies=('leaf; extra == "feature"', "root"))],
        "leaf": [wheel(tmp_path, "leaf")],
    }
    assert set(resolve(paths, ["root"])) == set(paths)


def test_conflicting_version_and_requires_python_are_rejected(tmp_path):
    paths = {
        "root": [wheel(tmp_path, "root", dependencies=("child<2",))],
        "child": [wheel(tmp_path, "child", version="2.0")],
    }
    with pytest.raises(builder.WheelhouseError, match="child<2"):
        resolve(paths, ["root"])
    with pytest.raises(builder.WheelhouseError, match="root.*CPython 3.11"):
        resolve({"root": [wheel(tmp_path, "root", python=">=3.12")]}, ["root"])


def test_abi3_wheels_cover_all_versions_and_are_preferred(tmp_path):
    stable = [wheel(tmp_path, "root", tag=f"cp39-abi3-{platform}")
              for platform in builder.SUPPORTED_PLATFORMS]
    specific = wheel(tmp_path, "root", tag="cp311-cp311-manylinux_2_17_x86_64")
    closure = resolve({"root": [specific, *stable]}, ["root"])
    assert closure["root"][1] == set(stable)


@pytest.mark.parametrize("missing_platform", builder.SUPPORTED_PLATFORMS)
@pytest.mark.parametrize("missing_python", builder.SUPPORTED_PYTHON_VERSIONS)
def test_missing_platform_or_python_wheel_is_not_hidden_by_other_targets(
    tmp_path, missing_platform, missing_python,
):
    paths = [wheel(tmp_path, "root", tag=f"cp{major}{minor}-cp{major}{minor}-{platform}")
             for major, minor in builder.SUPPORTED_PYTHON_VERSIONS
             for platform in builder.SUPPORTED_PLATFORMS
             if ((major, minor), platform) != (missing_python, missing_platform)]
    with pytest.raises(builder.WheelhouseError, match="missing required wheel: root"):
        resolve({"root": paths}, ["root"])


def test_verifier_checks_missing_closure_and_checksums(tmp_path, monkeypatch):
    root = wheel(tmp_path, "root", dependencies=("missing",))
    index(tmp_path, ["root==1.0"])
    monkeypatch.setattr(builder, "_runtime_roots", lambda: ["root==1.0"])
    with pytest.raises(builder.WheelhouseError, match="missing required wheel: missing"):
        builder.verify_wheelhouse(tmp_path)
    root.write_bytes(root.read_bytes() + b"tampered")
    with pytest.raises(builder.WheelhouseError, match="checksum mismatch"):
        builder.verify_wheelhouse(tmp_path)


def test_builder_failure_preserves_existing_files(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    wheel(output, "root", dependencies=("missing",))
    index(output, ["root==1.0"])
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    monkeypatch.setattr(builder, "_runtime_roots", lambda: ["root==1.0"])
    with pytest.raises(builder.WheelhouseError, match="missing"):
        builder.build_wheelhouse(tmp_path / "no-installed-packages", output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_builder_reads_declared_roots_and_discovers_new_transitive_packages(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    wheel(source, "novel-root", dependencies=("transitive-child",))
    wheel(source, "transitive-child")
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["novel-root==1.0"]\n')
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    output = tmp_path / "output"
    builder.build_wheelhouse(tmp_path / "no-installed-packages", output, wheel_dir=source)
    assert {name for name, _, _ in builder.verify_wheelhouse(output)} == {"novel-root", "transitive-child"}


def test_missing_indexed_file_names_the_distribution(tmp_path):
    path = wheel(tmp_path, "missing-dist")
    index(tmp_path, ["missing-dist==1.0"])
    path.rename(tmp_path / "held.whl")
    with pytest.raises(builder.WheelhouseError, match="missing pinned wheel: missing-dist==1.0"):
        builder._indexed_wheels(tmp_path)


def test_duplicate_zip_members_are_rejected(tmp_path):
    path = wheel(tmp_path, "root")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("root-1.0.dist-info/WHEEL", "ignored duplicate")
    with pytest.raises(builder.WheelhouseError, match="duplicate ZIP members"):
        builder._verify_wheel(path, "root", "1.0")


def test_filename_tags_must_match_archive_tags(tmp_path):
    path = wheel(tmp_path, "root")
    renamed = path.with_name(path.name.replace("py3-none-any", "cp311-cp311-manylinux_2_17_x86_64"))
    path.rename(renamed)
    with pytest.raises(builder.WheelhouseError, match="tags do not match"):
        builder._verify_wheel(renamed, "root", "1.0")


def test_optional_vendor_gate_remains_disabled(tmp_path):
    assert builder.PINNED_VENDOR_SHA256 == {}
    sources, missing = builder._verified_vendor_sources(tmp_path)
    assert sources == []
    assert missing == [f"{name}=={version}" for name, version in builder.OPTIONAL_VENDOR_DISTRIBUTIONS]


def test_verifier_bootstraps_packaging_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts/build_wheelhouse.py"), "--verify"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_rejects_a_corrupted_packaging_wheel(tmp_path):
    source = next((ROOT / "wheelhouse").glob("packaging-*-py3-none-any.whl"))
    copied = tmp_path / source.name
    copied.write_bytes(source.read_bytes())
    index(tmp_path, ["packaging==26.0"])
    copied.write_bytes(copied.read_bytes() + b"tampered")
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts/build_wheelhouse.py"), "--verify", "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "checksum mismatch for packaging" in result.stderr


def test_cli_missing_required_wheel_exits_nonzero_and_names_distribution(tmp_path):
    index(tmp_path, ["jsonschema==4.26.0"])
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_wheelhouse.py"), "--verify", "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "missing pinned wheel: jsonschema==4.26.0" in result.stderr


def test_source_archive_includes_every_indexed_wheel(tmp_path):
    spec = importlib.util.spec_from_file_location("wheelhouse_build_backend", ROOT / "_build/sf_backend.py")
    assert spec and spec.loader
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    filename = backend.build_sdist(str(tmp_path))
    with tarfile.open(tmp_path / filename) as archive:
        names = {name.partition("/")[2] for name in archive.getnames()}
    expected = {f"wheelhouse/{path.name}" for path in (ROOT / "wheelhouse").glob("*.whl")}
    assert expected <= names
    assert {"wheelhouse/SHA256SUMS", "wheelhouse/requirements.txt"} <= names
