from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import re
import tarfile
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
# Pinned from the root LICENSE at base commit 95ef93cc.
EXPECTED_LICENSE_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
HEADER_MARKER = re.compile(rb"^\s*#\s*(?:SPDX-License-Identifier\s*:|Copyright\b)", re.IGNORECASE)

# Every non-code package member shipped in the current wheel and source archive has
# an origin and license record here. Paths are relative to the package directory.
ASSET_PROVENANCE: dict[str, tuple[str, str]] = {
    "engine/ENGINE_SOURCE.json": ("Generated source-file manifest", "Root license"),
    "engine/release-manifest.json": ("Generated release manifest", "Root license"),
    "engine/switch-registry.json": ("Generated configuration registry", "Root license"),
    "mutations_prebuilt/archive_list/fixtures/sample.zip": (
        "Repository-authored fixture archive", "Root license"
    ),
    "mutations_prebuilt/archive_list/unit.json": (
        "Repository-authored unit metadata", "Root license"
    ),
    "mutations_prebuilt/file_facts/fixtures/sample.txt": (
        "Repository-authored fixture text", "Root license"
    ),
    "mutations_prebuilt/file_facts/unit.json": (
        "Repository-authored unit metadata", "Root license"
    ),
    "otel/manifest.json": ("Generated exporter file manifest", "Root license"),
    "otel/switch-registry.json": ("Repository-authored exporter settings registry", "Root license"),
    "mutations_prebuilt/html_extract/fixtures/sample.html": (
        "Repository-authored fixture markup", "Root license"
    ),
    "mutations_prebuilt/html_extract/unit.json": (
        "Repository-authored unit metadata", "Root license"
    ),
    "mutations_prebuilt/log_watch/fixtures/sample.log": (
        "Repository-authored fixture log", "Root license"
    ),
    "mutations_prebuilt/log_watch/unit.json": (
        "Repository-authored unit metadata", "Root license"
    ),
    "mutations_prebuilt/manifest.json": (
        "Generated fixture manifest", "Root license"
    ),
    "mutations_prebuilt/snapshot_diff/fixtures/after.txt": (
        "Repository-authored fixture text", "Root license"
    ),
    "mutations_prebuilt/snapshot_diff/fixtures/before.txt": (
        "Repository-authored fixture text", "Root license"
    ),
    "mutations_prebuilt/snapshot_diff/unit.json": (
        "Repository-authored unit metadata", "Root license"
    ),
    "mutations_prebuilt/sqlite_peek/fixtures/sample.sqlite": (
        "Repository-authored fixture database", "Root license"
    ),
    "mutations_prebuilt/sqlite_peek/unit.json": (
        "Repository-authored unit metadata", "Root license"
    ),
    "mutations_prebuilt/structured_peek/fixtures/sample.csv": (
        "Repository-authored fixture table", "Root license"
    ),
    "mutations_prebuilt/structured_peek/unit.json": (
        "Repository-authored unit metadata", "Root license"
    ),
    "mutations_prebuilt/text_grep_struct/fixtures/sample.log": (
        "Repository-authored fixture log", "Root license"
    ),
    "mutations_prebuilt/text_grep_struct/unit.json": (
        "Repository-authored unit metadata", "Root license"
    ),
    "profiles/r2hc3.json": ("Repository-authored runtime profile", "Root license"),
}


@dataclass(frozen=True)
class _BuiltArtifacts:
    wheel: Path
    sdist: Path
    package: str
    sdist_root: str


def _inventory_path(root: Path) -> Path:
    candidates = sorted((root / "src").glob("*/THIRD_PARTY_LICENSES.md"))
    assert len(candidates) == 1, f"expected one packaged third-party inventory, found {candidates}"
    return candidates[0]


def _load_build_backend(root: Path) -> ModuleType:
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    build_system = config["build-system"]
    module_name = build_system["build-backend"].split(".", 1)[0]
    backend_file = next(
        root / backend_path / f"{module_name}.py"
        for backend_path in build_system["backend-path"]
        if (root / backend_path / f"{module_name}.py").is_file()
    )
    spec = importlib.util.spec_from_file_location("release_test_backend", backend_file)
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def _sdist_root(names: list[str]) -> str:
    pkg_info = [name for name in names if PurePosixPath(name).name == "PKG-INFO"]
    assert len(pkg_info) == 1, f"expected one PKG-INFO, found {pkg_info}"
    return PurePosixPath(pkg_info[0]).parts[0]


@pytest.fixture(scope="module")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> _BuiltArtifacts:
    output = tmp_path_factory.mktemp("release-artifacts")
    backend = _load_build_backend(ROOT)
    wheel = output / backend.build_wheel(str(output))
    sdist = output / backend.build_sdist(str(output))
    with tarfile.open(sdist, "r:gz") as archive:
        archive_root = _sdist_root(archive.getnames())
    package = _inventory_path(ROOT).parent.relative_to(ROOT / "src").as_posix()
    return _BuiltArtifacts(wheel, sdist, package, archive_root)


def _license_digest_problem(payload: bytes, label: str) -> str | None:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != EXPECTED_LICENSE_SHA256:
        return f"{label}: expected pinned license SHA-256 {EXPECTED_LICENSE_SHA256}, got {actual}"
    return None


def _required_legal_members(artifacts: _BuiltArtifacts) -> tuple[list[str], list[str]]:
    with zipfile.ZipFile(artifacts.wheel) as wheel:
        metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(metadata) == 1, f"expected one wheel metadata member, found {metadata}"
    dist_info = metadata[0].rsplit("/", 1)[0]
    wheel_members = [
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/licenses/NOTICE",
        f"{artifacts.package}/THIRD_PARTY_LICENSES.md",
    ]
    sdist_members = [
        f"{artifacts.sdist_root}/LICENSE",
        f"{artifacts.sdist_root}/NOTICE",
        f"{artifacts.sdist_root}/src/{artifacts.package}/THIRD_PARTY_LICENSES.md",
    ]
    return wheel_members, sdist_members


def _wheel_metadata_member(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(metadata) == 1, f"expected one wheel metadata member, found {metadata}"
    return metadata[0]


def _legal_file_problems(root: Path, artifacts: _BuiltArtifacts) -> list[str]:
    inventory = _inventory_path(root)
    expected_payloads = {
        "LICENSE": root / "LICENSE",
        "NOTICE": root / "NOTICE",
        "THIRD_PARTY_LICENSES.md": inventory,
    }
    problems: list[str] = []
    wheel_expected, sdist_expected = _required_legal_members(artifacts)

    with zipfile.ZipFile(artifacts.wheel) as wheel:
        wheel_names = set(wheel.namelist())
        for member, filename in zip(wheel_expected, expected_payloads):
            if member not in wheel_names:
                problems.append(f"wheel: missing member {member}")
                continue
            payload = wheel.read(member)
            if filename == "LICENSE":
                problem = _license_digest_problem(payload, f"wheel {member}")
                if problem:
                    problems.append(problem)
            if payload != expected_payloads[filename].read_bytes():
                problems.append(f"wheel: {member} differs from its source file")

    with tarfile.open(artifacts.sdist, "r:gz") as sdist:
        sdist_names = set(sdist.getnames())
        for member, filename in zip(sdist_expected, expected_payloads):
            if member not in sdist_names:
                problems.append(f"sdist: missing member {member}")
                continue
            stream = sdist.extractfile(member)
            if stream is None:
                problems.append(f"sdist: member {member} is not a regular file")
                continue
            payload = stream.read()
            if filename == "LICENSE":
                problem = _license_digest_problem(payload, f"sdist {member}")
                if problem:
                    problems.append(problem)
            if payload != expected_payloads[filename].read_bytes():
                problems.append(f"sdist: {member} differs from its source file")
    return problems


def _package_asset_problems(names: list[str], prefix: str, label: str) -> list[str]:
    problems: list[str] = []
    for name in names:
        if name.endswith("/") or not name.startswith(prefix):
            continue
        relative = name.removeprefix(prefix)
        if relative == "THIRD_PARTY_LICENSES.md" or relative.endswith(".py"):
            continue
        provenance = ASSET_PROVENANCE.get(relative)
        if provenance is None:
            problems.append(f"{label}: no asset provenance entry for {relative}")
        elif not all(part.strip() for part in provenance):
            problems.append(f"{label}: incomplete asset provenance entry for {relative}")
    return problems


def _asset_problems(artifacts: _BuiltArtifacts) -> list[str]:
    problems: list[str] = []
    wheel_prefix = f"{artifacts.package}/"
    with zipfile.ZipFile(artifacts.wheel) as wheel:
        problems.extend(_package_asset_problems(wheel.namelist(), wheel_prefix, "wheel"))

    sdist_prefix = f"{artifacts.sdist_root}/src/{artifacts.package}/"
    with tarfile.open(artifacts.sdist, "r:gz") as sdist:
        problems.extend(_package_asset_problems(sdist.getnames(), sdist_prefix, "sdist"))
    return problems


def _marker_problems_for_members(
    names: list[str],
    prefix: str,
    record_prefix: str,
    inventory_member: str,
    read_member: Callable[[str], bytes],
    label: str,
) -> list[str]:
    problems: list[str] = []
    if inventory_member not in names:
        return [f"{label}: missing third-party inventory {inventory_member}"]
    inventory_payload = read_member(inventory_member)
    if not inventory_payload.strip():
        return [f"{label}: empty third-party inventory {inventory_member}"]
    inventory_text = inventory_payload.decode("utf-8", errors="replace")
    for member in names:
        if not member.startswith(prefix) or not member.endswith(".py"):
            continue
        stream = io.BytesIO(read_member(member))
        for line_number, line in enumerate(stream, start=1):
            if HEADER_MARKER.search(line):
                source_path = member.removeprefix(prefix)
                record_path = f"{record_prefix}/{source_path}"
                if record_path not in inventory_text:
                    problems.append(
                        f"{label}: unrecorded SPDX/copyright marker in {member}:{line_number}"
                    )
    return problems


def _source_marker_problems(artifacts: _BuiltArtifacts) -> list[str]:
    problems: list[str] = []
    wheel_prefix = f"{artifacts.package}/"
    wheel_inventory = f"{wheel_prefix}THIRD_PARTY_LICENSES.md"
    with zipfile.ZipFile(artifacts.wheel) as wheel:
        names = wheel.namelist()

        def read_wheel(member: str) -> bytes:
            return wheel.read(member)

        problems.extend(
            _marker_problems_for_members(
                names, wheel_prefix, artifacts.package, wheel_inventory, read_wheel, "wheel"
            )
        )

    sdist_prefix = f"{artifacts.sdist_root}/src/{artifacts.package}/"
    sdist_inventory = f"{sdist_prefix}THIRD_PARTY_LICENSES.md"
    with tarfile.open(artifacts.sdist, "r:gz") as sdist:
        names = sdist.getnames()

        def read_sdist(member: str) -> bytes:
            stream = sdist.extractfile(member)
            assert stream is not None, f"sdist: member {member} is not a regular file"
            return stream.read()

        problems.extend(
            _marker_problems_for_members(
                names, sdist_prefix, artifacts.package, sdist_inventory, read_sdist, "sdist"
            )
        )
    return problems


def _write_wheel_copy(
    source: Path,
    target: Path,
    *,
    keep: set[str],
    additions: dict[str, bytes] | None = None,
) -> None:
    added = additions or {}
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as copy,
    ):
        for info in original.infolist():
            if info.filename in keep:
                copy.writestr(info, original.read(info.filename))
        for member, payload in added.items():
            copy.writestr(member, payload)


def _write_sdist_copy(
    source: Path,
    target: Path,
    *,
    keep: set[str],
    additions: dict[str, bytes] | None = None,
) -> None:
    added = additions or {}
    with tarfile.open(source, "r:gz") as original, tarfile.open(target, "w:gz") as copy_tar:
        for member in keep:
            info = original.getmember(member)
            payload = original.extractfile(info) if info.isfile() else None
            copy_tar.addfile(copy.copy(info), payload)
        for member, payload in added.items():
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            copy_tar.addfile(info, io.BytesIO(payload))


def test_root_license_identity_is_pinned_and_rejects_a_swapped_copy(tmp_path: Path) -> None:
    root_license = ROOT / "LICENSE"
    root_payload = root_license.read_bytes()
    assert _license_digest_problem(root_payload, "root LICENSE") is None

    inventory = _inventory_path(ROOT).read_bytes().decode("utf-8")
    start = inventory.index("The MIT License (MIT)")
    end = inventory.index("\n### ", start)
    planted_mit = tmp_path / "LICENSE"
    planted_mit.write_bytes(inventory[start:end].strip().encode("utf-8") + b"\n")
    problem = _license_digest_problem(planted_mit.read_bytes(), "temporary LICENSE")
    assert problem is not None and "pinned license SHA-256" in problem


def test_built_wheel_and_sdist_physically_ship_the_legal_files(
    built_artifacts: _BuiltArtifacts, tmp_path: Path
) -> None:
    assert _legal_file_problems(ROOT, built_artifacts) == []

    wheel_members, sdist_members = _required_legal_members(built_artifacts)
    broken_wheel = tmp_path / built_artifacts.wheel.name
    broken_sdist = tmp_path / built_artifacts.sdist.name
    _write_wheel_copy(
        built_artifacts.wheel,
        broken_wheel,
        keep={_wheel_metadata_member(built_artifacts.wheel)},
    )
    _write_sdist_copy(
        built_artifacts.sdist,
        broken_sdist,
        keep={f"{built_artifacts.sdist_root}/PKG-INFO"},
    )
    broken = _BuiltArtifacts(
        broken_wheel,
        broken_sdist,
        built_artifacts.package,
        built_artifacts.sdist_root,
    )
    problems = _legal_file_problems(ROOT, broken)
    assert {f"wheel: missing member {member}" for member in wheel_members} <= set(problems)
    assert {f"sdist: missing member {member}" for member in sdist_members} <= set(problems)


def test_source_markers_need_inventory_records_and_reject_an_unlisted_header(
    built_artifacts: _BuiltArtifacts, tmp_path: Path
) -> None:
    assert _source_marker_problems(built_artifacts) == []

    source_probe = f"{built_artifacts.package}/_unlisted_source_probe.py"
    wheel_inventory = f"{built_artifacts.package}/THIRD_PARTY_LICENSES.md"
    source_bytes = (
        b"# SPDX-License-Identifier: GPL-2.0-only\n"
        b"# Copyright (c) Unlisted Source\n"
        b"VALUE = 1\n"
    )
    planted_wheel = tmp_path / built_artifacts.wheel.name
    _write_wheel_copy(
        built_artifacts.wheel,
        planted_wheel,
        keep={wheel_inventory},
        additions={source_probe: source_bytes},
    )
    planted = _BuiltArtifacts(
        planted_wheel,
        built_artifacts.sdist,
        built_artifacts.package,
        built_artifacts.sdist_root,
    )
    problems = _source_marker_problems(planted)
    assert any(
        source_probe in problem and "unrecorded SPDX/copyright marker" in problem
        for problem in problems
    )

    sdist_source_probe = (
        f"{built_artifacts.sdist_root}/src/{built_artifacts.package}/"
        "_unlisted_sdist_source_probe.py"
    )
    sdist_inventory = (
        f"{built_artifacts.sdist_root}/src/{built_artifacts.package}/THIRD_PARTY_LICENSES.md"
    )
    planted_sdist = tmp_path / built_artifacts.sdist.name
    _write_sdist_copy(
        built_artifacts.sdist,
        planted_sdist,
        keep={sdist_inventory},
        additions={sdist_source_probe: source_bytes},
    )
    sdist_planted = _BuiltArtifacts(
        built_artifacts.wheel,
        planted_sdist,
        built_artifacts.package,
        built_artifacts.sdist_root,
    )
    problems = _source_marker_problems(sdist_planted)
    assert any(
        sdist_source_probe in problem and "unrecorded SPDX/copyright marker" in problem
        for problem in problems
    )


def test_every_non_code_package_asset_has_provenance_and_rejects_an_unknown_asset(
    built_artifacts: _BuiltArtifacts, tmp_path: Path
) -> None:
    assert _asset_problems(built_artifacts) == []

    asset_probe = "_unlisted_asset_probe.json"
    asset_bytes = b'{"origin":"unknown","license":"unknown"}\n'
    wheel_member = f"{built_artifacts.package}/{asset_probe}"
    sdist_member = f"{built_artifacts.sdist_root}/src/{built_artifacts.package}/{asset_probe}"
    planted_wheel = tmp_path / built_artifacts.wheel.name
    planted_sdist = tmp_path / built_artifacts.sdist.name
    _write_wheel_copy(
        built_artifacts.wheel,
        planted_wheel,
        keep={_wheel_metadata_member(built_artifacts.wheel)},
        additions={wheel_member: asset_bytes},
    )
    _write_sdist_copy(
        built_artifacts.sdist,
        planted_sdist,
        keep={f"{built_artifacts.sdist_root}/PKG-INFO"},
        additions={sdist_member: asset_bytes},
    )
    planted = _BuiltArtifacts(
        planted_wheel,
        planted_sdist,
        built_artifacts.package,
        built_artifacts.sdist_root,
    )
    problems = _asset_problems(planted)
    assert f"wheel: no asset provenance entry for {asset_probe}" in problems
    assert f"sdist: no asset provenance entry for {asset_probe}" in problems


# Full texts of the licenses for code adapted into repo_intel_core, pinned by digest of the
# exact upstream text (trailing whitespace stripped).
ADAPTED_CODE_LICENSES = {
    "CodeGraph": "69ba01d8aad8c9f7fa721245eda05fdb36d82bbae1c43f9751f13ab8cf3cc34b",
    "CPython": "35936f8ff79198c68a38a9bb1912fa131fc7b840ca5928feea09dc96e5c66b8d",
}


def _adapted_license_text(inventory: str, project: str) -> str:
    start = inventory.index(f"### {project}")
    block = inventory[start:].split("```", 2)
    assert len(block) == 3, f"{project}: license text block missing"
    return block[1].strip("\n").rstrip()


def test_adapted_code_notices_ship_in_wheel_and_sdist(built_artifacts: _BuiltArtifacts) -> None:
    with zipfile.ZipFile(built_artifacts.wheel) as wheel:
        members = {name: wheel.read(name).decode() for name in wheel.namelist()
                   if name == f"{built_artifacts.package}/THIRD_PARTY_LICENSES.md" or name.endswith("/NOTICE")}
    with tarfile.open(built_artifacts.sdist, "r:gz") as sdist:
        for path in ("NOTICE", f"src/{built_artifacts.package}/THIRD_PARTY_LICENSES.md"):
            handle = sdist.extractfile(f"{built_artifacts.sdist_root}/{path}")
            assert handle is not None, path
            members[f"sdist:{path}"] = handle.read().decode()
    notices = {name: text for name, text in members.items() if name.endswith("NOTICE")}
    inventories = {name: text for name, text in members.items() if name.endswith("THIRD_PARTY_LICENSES.md")}
    assert len(notices) == 2 and len(inventories) == 2, sorted(members)
    for name, text in notices.items():
        assert "CodeGraph (MIT)" in text and "CPython (PSF-2.0)" in text, name
        assert "call_graph.py" in text and "ts_extract.py" in text and "ts_dispatch.py" in text, name
    for name, text in inventories.items():
        assert "https://github.com/colbymchenry/codegraph" in text, name
        assert "Copyright (c) 2026 Colby Mchenry" in text, name
        for project, digest in ADAPTED_CODE_LICENSES.items():
            body = _adapted_license_text(text, project)
            assert hashlib.sha256(body.encode()).hexdigest() == digest, f"{name}: {project}"
