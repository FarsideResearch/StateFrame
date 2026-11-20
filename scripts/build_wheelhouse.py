#!/usr/bin/env python3
"""Build and verify an offline wheelhouse with universal and pinned platform wheels.

Runtime roots and default-installed extras come from pyproject.toml. Supply downloaded
wheels with --wheel-dir; installed pure-Python packages can also be repacked with
--site-packages. Resolution
and verification cover CPython 3.11-3.14 on manylinux x86_64/aarch64 (glibc 2.17+)
and macOS x86_64/arm64 (11+). Neither mode downloads anything.
"""
from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from email.message import Message
from email.parser import BytesParser
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path, PurePosixPath
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE_PACKAGES = Path("/opt/homebrew/lib/python3.14/site-packages")
SUPPORTED_PYTHON_VERSIONS = ((3, 11), (3, 12), (3, 13), (3, 14))
SUPPORTED_PLATFORMS = (
    "manylinux_2_17_x86_64", "manylinux_2_17_aarch64",
    "macosx_11_0_arm64", "macosx_11_0_x86_64",
)
PINNED_PLATFORM_VERSIONS = {"rpds-py": "2026.6.3"}
# Keep in sync with install.sh's default-enabled extras (covered by a regression test).
DEFAULT_INSTALL_EXTRAS = ("repomap",)
PINNED_LOCAL_VERSIONS = {
    "attrs": "26.1.0",
    "PyYAML": "6.0.3",
    "jedi": "0.20.0",
    "parso": "0.8.7",
    "networkx": "3.6.1",
}
OPTIONAL_VENDOR_DISTRIBUTIONS = (("jsonschema", "4.17.3"), ("pyrsistent", "0.20.0"))
# Optional vendor packages stay disabled until exact filenames and SHA-256 digests are reviewed.
PINNED_VENDOR_SHA256: dict[str, str] = {}
EXCLUDED_FILES_BY_DISTRIBUTION: dict[str, tuple[str, ...]] = {
    # PyYAML's pure Python implementation remains complete without its optional libyaml accelerator.
    "pyyaml": (
        "_yaml", "_yaml/*", "_yaml/**", "yaml/_yaml*", "yaml/_yaml/*", "yaml/_yaml/**",
        "yaml/*.so", "yaml/*.pyd", "yaml/*.dll", "yaml/*.dylib",
    ),
}
INSTALL_ONLY_DIST_INFO_FILES = {
    "INSTALLER", "REQUESTED", "RECORD", "RECORD.jws", "RECORD.p7s", "direct_url.json",
}
COMPILED_SUFFIXES = tuple(EXTENSION_SUFFIXES) + (".so", ".pyd", ".dll", ".dylib", ".o", ".a", ".lib")
VENDOR_SOURCE_SUFFIXES = (".whl", ".tar.gz", ".tgz", ".zip")


class WheelhouseError(Exception):
    """The local distributions or wheelhouse do not meet the offline contract."""


def _canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_distribution_name(name: str) -> str:
    return _canonicalize(name).replace("-", "_")


def _wheel_version(version: str) -> str:
    return version.replace("-", "_")


def _wheel_filename(name: str, version: str) -> str:
    return f"{_wheel_distribution_name(name)}-{_wheel_version(version)}-py3-none-any.whl"


def _is_compiled_file(record_path: str) -> bool:
    lowered = record_path.lower()
    return any(lowered.endswith(suffix.lower()) for suffix in COMPILED_SUFFIXES)


def _explicitly_excluded(distribution: str, record_path: str) -> bool:
    patterns = EXCLUDED_FILES_BY_DISTRIBUTION.get(_canonicalize(distribution), ())
    return any(fnmatch.fnmatchcase(record_path, pattern) for pattern in patterns)


def _find_distribution(site_packages: Path, wanted: str) -> tuple[Path, Message, bytes]:
    matches: list[tuple[Path, Message, bytes]] = []
    for dist_info in sorted(site_packages.glob("*.dist-info")):
        metadata_file = dist_info / "METADATA"
        if not metadata_file.is_file():
            continue
        metadata_bytes = metadata_file.read_bytes()
        metadata = BytesParser().parsebytes(metadata_bytes)
        if _canonicalize(metadata.get("Name", "")) == _canonicalize(wanted):
            matches.append((dist_info, metadata, metadata_bytes))
    if not matches:
        raise WheelhouseError(f"required installed distribution is missing: {wanted} in {site_packages}")
    if len(matches) != 1:
        names = ", ".join(str(path.name) for path, _, _ in matches)
        raise WheelhouseError(f"multiple installed distributions match {wanted}: {names}")
    return matches[0]


def _license(metadata: Message) -> str:
    expressions = metadata.get_all("License-Expression") or []
    if expressions:
        return " / ".join(expressions)
    licenses = metadata.get_all("License") or []
    if licenses:
        return " / ".join(licenses)
    classifiers = [
        item.removeprefix("License :: ")
        for item in metadata.get_all("Classifier") or []
        if item.startswith("License :: ")
    ]
    if classifiers:
        return " / ".join(classifiers)
    raise WheelhouseError(f"license is absent from METADATA for {metadata.get('Name', 'unknown')}")


def _validate_record_entry(
    distribution: str,
    record_path: str,
    data: bytes,
    recorded_hash: str,
    recorded_size: str,
) -> None:
    if recorded_size:
        try:
            size = int(recorded_size)
        except ValueError as error:
            raise WheelhouseError(f"invalid RECORD size for {distribution}: {record_path}") from error
        if size != len(data):
            raise WheelhouseError(f"RECORD size mismatch for {distribution}: {record_path}")
    if recorded_hash:
        algorithm, separator, expected = recorded_hash.partition("=")
        if not separator or algorithm not in hashlib.algorithms_available:
            raise WheelhouseError(f"unsupported RECORD digest for {distribution}: {recorded_hash}")
        actual = base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest()).rstrip(b"=").decode()
        if actual != expected:
            raise WheelhouseError(f"RECORD hash mismatch for {distribution}: {record_path}")


def _record_payload(site_packages: Path, dist_info: Path, metadata: Message) -> dict[str, bytes]:
    record_file = dist_info / "RECORD"
    if not record_file.is_file():
        raise WheelhouseError(f"installed distribution has no RECORD: {dist_info.name}")

    site_root = site_packages.resolve()
    wheel_info = f"{_wheel_distribution_name(metadata['Name'])}-{_wheel_version(metadata['Version'])}.dist-info"
    payload: dict[str, bytes] = {}
    with record_file.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) != 3:
                raise WheelhouseError(f"malformed RECORD row for {metadata['Name']}: {row!r}")
            record_path, recorded_hash, recorded_size = row
            relative = PurePosixPath(record_path)
            if relative.is_absolute():
                continue

            source = site_packages.joinpath(*relative.parts)
            resolved = source.resolve(strict=False)
            if not resolved.is_relative_to(site_root):
                # pip-generated console scripts live outside purelib; entry_points.txt recreates them.
                continue
            if "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue
            if not source.is_file():
                raise WheelhouseError(f"RECORD file is missing for {metadata['Name']}: {record_path}")

            data = source.read_bytes()
            _validate_record_entry(metadata["Name"], record_path, data, recorded_hash, recorded_size)
            if _explicitly_excluded(metadata["Name"], record_path):
                continue
            if _is_compiled_file(record_path):
                raise WheelhouseError(
                    f"{metadata['Name']} is not pure Python: compiled file recorded at {record_path}"
                )

            archive_path = PurePosixPath(*resolved.relative_to(site_root).parts)
            if archive_path.parts and archive_path.parts[0] == dist_info.name:
                info_relative = archive_path.relative_to(dist_info.name)
                if info_relative.name in INSTALL_ONLY_DIST_INFO_FILES:
                    continue
                if info_relative.as_posix() == "WHEEL":
                    continue
                archive_path = PurePosixPath(wheel_info, *info_relative.parts)
            archive_name = archive_path.as_posix()
            if archive_name in payload and payload[archive_name] != data:
                raise WheelhouseError(f"duplicate RECORD path with different content: {archive_name}")
            payload[archive_name] = data

    metadata_path = f"{wheel_info}/METADATA"
    if metadata_path not in payload:
        raise WheelhouseError(f"RECORD did not include METADATA for {metadata['Name']}")
    return payload


def _record_has_excluded_compiled_file(dist_info: Path, metadata: Message) -> bool:
    with (dist_info / "RECORD").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 3 and _is_compiled_file(row[0]) and _explicitly_excluded(metadata["Name"], row[0]):
                return True
    return False


def _build_wheel(site_packages: Path, wanted: str) -> tuple[str, bytes, str, str]:
    dist_info, metadata, _ = _find_distribution(site_packages, wanted)
    name = metadata.get("Name", "")
    version = metadata.get("Version", "")
    if not name or not version:
        raise WheelhouseError(f"installed distribution has incomplete METADATA: {dist_info.name}")
    pinned_version = next(
        (expected for package, expected in PINNED_LOCAL_VERSIONS.items()
         if _canonicalize(package) == _canonicalize(name)),
        None,
    )
    if pinned_version is not None and version != pinned_version:
        raise WheelhouseError(f"{name} must be version {pinned_version}; found {version}")

    wheel_metadata = dist_info / "WHEEL"
    if not wheel_metadata.is_file():
        raise WheelhouseError(f"installed distribution has no WHEEL metadata: {dist_info.name}")
    wheel_text = wheel_metadata.read_text(encoding="utf-8")
    tags = [line.removeprefix("Tag: ") for line in wheel_text.splitlines() if line.startswith("Tag: ")]
    # Jedi's universal wheel carries both py2-none-any and py3-none-any tags.
    # Keep only the Python 3 tag in our deterministic, runtime-specific output.
    source_is_pure = (
        "Root-Is-Purelib: true" in wheel_text
        and "py3-none-any" in tags
        and all(tag in {"py2-none-any", "py3-none-any"} for tag in tags)
    )
    allowed_pyyaml_repack = (
        _canonicalize(name) == "pyyaml"
        and _record_has_excluded_compiled_file(dist_info, metadata)
    )
    if not source_is_pure and not allowed_pyyaml_repack:
        raise WheelhouseError(f"{name} is not installed as a pure-Python wheel: tags={tags!r}")

    license_name = _license(metadata)
    payload = _record_payload(site_packages, dist_info, metadata)
    if _canonicalize(name) == "pyyaml":
        wheel_info = f"{_wheel_distribution_name(name)}-{_wheel_version(version)}.dist-info"
        unexpected = [
            path for path in payload
            if not (path.startswith("yaml/") or path.startswith(f"{wheel_info}/"))
        ]
        if unexpected:
            raise WheelhouseError(f"PyYAML pure repack contains unexpected paths: {unexpected[:5]}")

    wheel_info = f"{_wheel_distribution_name(name)}-{_wheel_version(version)}.dist-info"
    payload[f"{wheel_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: stateframe build_wheelhouse.py\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode("utf-8")

    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for path, data in sorted(payload.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((path, f"sha256={digest}", str(len(data))))
    writer.writerow((f"{wheel_info}/RECORD", "", ""))
    payload[f"{wheel_info}/RECORD"] = record_buffer.getvalue().encode("utf-8")

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return _wheel_filename(name, version), output.getvalue(), license_name, f"{name}=={version}"


def _parse_requirements(path: Path) -> list[tuple[str, str, str]]:
    if not path.is_file():
        raise WheelhouseError(f"pinned requirements file is missing: {path}")
    parsed: list[tuple[str, str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9.!+_-]+)", line.strip())
        if not match:
            raise WheelhouseError(f"invalid pinned requirement at {path}:{line_number}: {line!r}")
        name, version = match.groups()
        parsed.append((_canonicalize(name), version, line.strip()))

    names = [name for name, _, _ in parsed]
    if len(names) != len(set(names)):
        raise WheelhouseError(f"duplicate distribution pin in {path}")
    if not parsed:
        raise WheelhouseError(f"pinned requirements file is empty: {path}")
    versions = {name: version for name, version, _ in parsed}
    for name, version in PINNED_LOCAL_VERSIONS.items():
        if _canonicalize(name) in versions and versions[_canonicalize(name)] != version:
            raise WheelhouseError(f"requirements.txt must pin {name}=={version}")
    return parsed


def _verify_wheel(path: Path, requirement_name: str, version: str) -> None:
    from packaging.tags import parse_tag
    from packaging.utils import parse_wheel_filename

    try:
        filename_name, filename_version, _, filename_tags = parse_wheel_filename(path.name)
        if filename_name != requirement_name or str(filename_version) != version:
            raise WheelhouseError(f"wheel filename does not match its pin: {path.name}")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise WheelhouseError(f"wheel has a corrupt ZIP member: {path.name}")
            names = set(archive.namelist())
            if len(names) != len(archive.namelist()):
                raise WheelhouseError(f"wheel contains duplicate ZIP members: {path.name}")
            platform_versions = {**PINNED_PLATFORM_VERSIONS, **_default_extra_pins()}
            platform_wheel = requirement_name in platform_versions
            if platform_wheel and platform_versions[requirement_name] != version:
                raise WheelhouseError(f"unreviewed platform wheel version: {requirement_name}=={version}")
            if not platform_wheel and any(_is_compiled_file(name) for name in names):
                raise WheelhouseError(f"wheel contains compiled files: {path.name}")
            info_candidates = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(info_candidates) != 1:
                raise WheelhouseError(f"wheel must contain one METADATA file: {path.name}")
            info_root = info_candidates[0].removesuffix("/METADATA")
            metadata = BytesParser().parsebytes(archive.read(info_candidates[0]))
            if _canonicalize(metadata.get("Name", "")) != requirement_name or metadata.get("Version") != version:
                raise WheelhouseError(f"wheel metadata does not match its pin: {path.name}")
            _license(metadata)
            wheel = archive.read(f"{info_root}/WHEEL").decode("utf-8")
            wheel_tags = set().union(*(
                parse_tag(line.removeprefix("Tag: "))
                for line in wheel.splitlines() if line.startswith("Tag: ")
            ))
            if wheel_tags != filename_tags:
                raise WheelhouseError(f"wheel tags do not match its filename: {path.name}")
            if not platform_wheel and (
                "Root-Is-Purelib: true" not in wheel
                or any(tag.abi != "none" or tag.platform != "any" for tag in wheel_tags)
            ):
                raise WheelhouseError(f"wheel is not tagged pure Python: {path.name}")
            record_name = f"{info_root}/RECORD"
            rows = list(csv.reader(archive.read(record_name).decode("utf-8").splitlines()))
            recorded: set[str] = set()
            for row in rows:
                if len(row) != 3:
                    raise WheelhouseError(f"malformed wheel RECORD: {path.name}")
                record_path, digest, size = row
                if record_path in recorded:
                    raise WheelhouseError(f"duplicate wheel RECORD row: {path.name}: {record_path}")
                recorded.add(record_path)
                if not digest:
                    if record_path != record_name or size:
                        raise WheelhouseError(f"invalid unhashed RECORD row: {path.name}")
                    continue
                if record_path not in names:
                    raise WheelhouseError(f"wheel RECORD references a missing file: {path.name}: {record_path}")
                data = archive.read(record_path)
                algorithm, separator, expected = digest.partition("=")
                if not separator or algorithm != "sha256":
                    raise WheelhouseError(f"unsupported wheel RECORD digest: {path.name}: {digest}")
                actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                if actual != expected or int(size) != len(data):
                    raise WheelhouseError(f"wheel RECORD integrity check failed: {path.name}: {record_path}")
            # Upstream wheels may include structural ZIP directory entries outside RECORD.
            # Only empty directories are exempt; every byte of file payload stays covered.
            empty_directories = {
                entry.filename for entry in archive.infolist() if entry.is_dir() and entry.file_size == 0
            }
            if recorded - names or names - recorded - empty_directories:
                raise WheelhouseError(f"wheel RECORD does not cover every file: {path.name}")
    except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile, csv.Error, ValueError) as error:
        raise WheelhouseError(f"cannot read wheel {path}: {error}") from error


def _read_checksums(directory: Path) -> dict[str, str]:
    path = directory / "SHA256SUMS"
    if not path.is_file():
        raise WheelhouseError(f"wheelhouse checksum file is missing: {path}")
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+\.whl)", line)
        if not match:
            raise WheelhouseError(f"malformed checksum line in {path}: {line!r}")
        digest, filename = match.groups()
        if filename in checksums:
            raise WheelhouseError(f"duplicate checksum entry for {filename}")
        checksums[filename] = digest
    return checksums


def _prepare_packaging(directory: Path) -> None:
    # The installer verifies before installing dependencies. Bootstrap only from
    # a checksum-verified universal wheel when packaging is not already present.
    try:
        import packaging.requirements  # noqa: F401
        return
    except ImportError:
        pass
    checksums = _read_checksums(directory)
    candidates = sorted(directory.glob("packaging-*-py3-none-any.whl"))
    if len(candidates) != 1:
        raise WheelhouseError("missing or ambiguous required wheel: packaging")
    wheel = candidates[0]
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != checksums.get(wheel.name):
        raise WheelhouseError(f"checksum mismatch for {wheel.name}")
    sys.path.insert(0, str(wheel.resolve()))
    try:
        import packaging.requirements  # noqa: F401
    except ImportError as error:
        raise WheelhouseError("cannot load required wheel: packaging") from error


def _wheel_metadata(path: Path) -> Message:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(entries) != 1:
                raise WheelhouseError(f"wheel must contain one METADATA file: {path.name}")
            return BytesParser().parsebytes(archive.read(entries[0]))
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise WheelhouseError(f"cannot read wheel metadata: {path.name}: {error}") from error


def _default_extra_roots() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    try:
        return [requirement for extra in DEFAULT_INSTALL_EXTRAS
                for requirement in project["optional-dependencies"][extra]]
    except KeyError as error:
        raise WheelhouseError(f"default-installed extra is missing from pyproject.toml: {error}") from error


def _default_extra_pins() -> dict[str, str]:
    """Native extra wheels are allowed only at the exact versions declared by the project."""
    from packaging.requirements import Requirement

    pins: dict[str, str] = {}
    for text in _default_extra_roots():
        requirement = Requirement(text)
        specifiers = list(requirement.specifier)
        if requirement.url or len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
            raise WheelhouseError(f"default-installed extra must use an exact version pin: {text}")
        pins[_canonicalize(requirement.name)] = specifiers[0].version
    return pins


def _runtime_roots(*, include_default_extras: bool = True) -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return [*project.get("dependencies", []),
            *(_default_extra_roots() if include_default_extras else [])]


def _target_platform_tags(platform: str) -> list[str]:
    from packaging.tags import mac_platforms

    if platform.startswith("macosx_"):
        arch = "arm64" if platform.endswith("arm64") else "x86_64"
        return list(mac_platforms(version=(11, 0), arch=arch))
    arch = "aarch64" if platform.endswith("aarch64") else "x86_64"
    tags = [f"manylinux_2_{minor}_{arch}" for minor in range(17, 4, -1)]
    tags.append(f"manylinux2014_{arch}")
    if arch == "x86_64":
        tags.extend(("manylinux2010_x86_64", "manylinux1_x86_64"))
    return tags


def _target_environment(version: tuple[int, int], platform: str) -> dict[str, str]:
    linux = platform.startswith("manylinux")
    arch = ("aarch64" if linux else "arm64") if platform.endswith(("aarch64", "arm64")) else "x86_64"
    python = ".".join(map(str, version))
    return {
        "implementation_name": "cpython", "implementation_version": f"{python}.0",
        "os_name": "posix", "platform_machine": arch,
        "platform_python_implementation": "CPython",
        "platform_release": "", "platform_system": "Linux" if linux else "Darwin",
        "platform_version": "", "python_full_version": f"{python}.0",
        "python_version": python, "sys_platform": "linux" if linux else "darwin",
        "extra": "",
    }


def _runtime_closure(
    candidates: Callable[[str], list[Path]], roots: list[str],
) -> dict[str, tuple[str, set[Path]]]:
    """Resolve one pinned version per distribution, evaluating every target's metadata.

    Explicit target tags avoid inspecting the build machine's ABI or architecture.
    Revisit a distribution for newly requested extras, including cycles through extras.
    """
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.tags import compatible_tags, cpython_tags
    from packaging.utils import parse_wheel_filename

    selected: dict[str, tuple[str, set[Path]]] = {}
    for version in SUPPORTED_PYTHON_VERSIONS:
        for platform in SUPPORTED_PLATFORMS:
            env = _target_environment(version, platform)
            target = f"CPython {env['python_version']} / {platform}"
            interpreter = f"cp{version[0]}{version[1]}"
            platforms = _target_platform_tags(platform)
            tags = set(cpython_tags(version, abis=[interpreter], platforms=platforms))
            tags.update(compatible_tags(version, interpreter=interpreter, platforms=platforms))
            pending = [(root, ("",), "project.dependencies") for root in roots]
            visited: set[tuple[str, tuple[str, ...]]] = set()
            while pending:
                text, extras, parent = pending.pop()
                requirement = Requirement(text)
                if requirement.marker and not any(
                    requirement.marker.evaluate({**env, "extra": extra}) for extra in extras
                ):
                    continue
                name = _canonicalize(requirement.name)
                if requirement.url:
                    raise WheelhouseError(f"{name}: direct URL dependencies cannot be resolved offline")
                paths = candidates(name)
                matching: list[tuple[Path, str, Message]] = []
                for path in paths:
                    _, candidate_version, _, wheel_tags = parse_wheel_filename(path.name)
                    if candidate_version not in requirement.specifier or not wheel_tags & tags:
                        continue
                    metadata = _wheel_metadata(path)
                    python_constraint = SpecifierSet(metadata.get("Requires-Python", ""))
                    if env["python_full_version"] in python_constraint:
                        matching.append((path, str(candidate_version), metadata))
                if not matching:
                    raise WheelhouseError(f"missing required wheel: {requirement} for {target} (required by {parent})")
                versions = {item[1] for item in matching}
                if len(versions) != 1:
                    raise WheelhouseError(f"ambiguous versions for {name}: {sorted(versions)}; supply exact pins")
                # Prefer a universal or stable-ABI wheel over per-interpreter wheels.
                matching.sort(key=lambda item: ("-none-any.whl" not in item[0].name,
                                                "-abi3-" not in item[0].name, item[0].name))
                path, resolved_version, metadata = matching[0]
                if name in selected and selected[name][0] != resolved_version:
                    raise WheelhouseError(f"conflicting runtime requirements for {name}: {requirement}")
                selected.setdefault(name, (resolved_version, set()))[1].add(path)
                active_extras = tuple(sorted({"", *requirement.extras}))
                key = (name, active_extras)
                if key in visited:
                    continue
                visited.add(key)
                pending.extend((dependency, active_extras, name)
                               for dependency in metadata.get_all("Requires-Dist", []))
    return selected


def _indexed_wheels(directory: Path) -> tuple[list[tuple[str, str, str]], dict[str, list[Path]]]:
    """Check pins, archive integrity, and checksums without assuming closure completeness."""
    from packaging.utils import parse_wheel_filename

    requirements = _parse_requirements(directory / "requirements.txt")
    pins = {name: version for name, version, _ in requirements}
    checksums = _read_checksums(directory)
    wheels = {path.name: path for path in directory.glob("*.whl")}
    for filename in sorted(set(checksums) - set(wheels)):
        name, version, _, _ = parse_wheel_filename(filename)
        raise WheelhouseError(f"missing pinned wheel: {name}=={version} ({filename})")
    if set(wheels) - set(checksums):
        raise WheelhouseError(f"SHA256SUMS omits wheel files: {sorted(set(wheels) - set(checksums))}")
    by_name: dict[str, list[Path]] = {}
    for filename, path in sorted(wheels.items()):
        name, version, _, _ = parse_wheel_filename(filename)
        if pins.get(name) != str(version):
            raise WheelhouseError(f"wheel has no matching requirement pin: {filename}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksums[filename]:
            raise WheelhouseError(f"checksum mismatch for {filename}")
        _verify_wheel(path, name, str(version))
        by_name.setdefault(name, []).append(path)
    for name in pins:
        if name not in by_name:
            raise WheelhouseError(f"missing pinned wheel: {name}=={pins[name]}")
    return requirements, by_name


def verify_wheelhouse(directory: Path) -> list[tuple[str, str, str]]:
    if not directory.is_dir():
        raise WheelhouseError(f"wheelhouse directory is missing: {directory}")
    _prepare_packaging(directory)
    requirements, by_name = _indexed_wheels(directory)
    closure = _runtime_closure(lambda name: by_name.get(name, []), _runtime_roots())
    pinned = {name for name, _, _ in requirements}
    if pinned != set(closure):
        raise WheelhouseError(f"requirements.txt contains distributions outside runtime closure: {sorted(pinned - set(closure))}")
    return requirements


def _vendor_candidate_paths(vendor_dir: Path, name: str, version: str) -> list[Path]:
    prefix = f"{name}-{version}".lower()
    candidates: list[Path] = []
    for path in vendor_dir.iterdir():
        lowered = path.name.lower()
        if not path.is_file() or not lowered.startswith(prefix):
            continue
        remainder = lowered[len(prefix):]
        if not (
            remainder.startswith("-")
            or remainder in {".tar.gz", ".tgz", ".zip"}
        ):
            continue
        if any(lowered.endswith(suffix) for suffix in VENDOR_SOURCE_SUFFIXES):
            candidates.append(path)
    return sorted(candidates)


def _build_vendor_sdist(source: Path, name: str, version: str) -> tuple[str, bytes, str, str]:
    with tempfile.TemporaryDirectory(prefix="stateframe-vendor-build-") as temporary:
        wheel_dir = Path(temporary)
        command = [
            sys.executable, "-m", "pip", "wheel", "--disable-pip-version-check", "--no-index",
            "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheel_dir), str(source),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env={**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        )
        if completed.returncode != 0:
            raise WheelhouseError(
                f"could not build pinned vendor sdist {source.name} offline: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        wheels = sorted(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise WheelhouseError(f"vendor sdist did not produce exactly one wheel: {source.name}")
        wheel = wheels[0]
        _verify_wheel(wheel, _canonicalize(name), version)
        target_name = _wheel_filename(name, version)
        if wheel.name != target_name:
            raise WheelhouseError(f"vendor sdist did not produce a py3-none-any wheel: {wheel.name}")
        with zipfile.ZipFile(wheel) as archive:
            metadata_entry = next(item for item in archive.namelist() if item.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(metadata_entry))
        return target_name, wheel.read_bytes(), _license(metadata), f"{metadata['Name']}=={version}"


def _verified_vendor_sources(vendor_dir: Path | None) -> tuple[list[Path], list[str]]:
    chosen: list[Path] = []
    missing: list[str] = []
    if not PINNED_VENDOR_SHA256:
        return [], [f"{name}=={version}" for name, version in OPTIONAL_VENDOR_DISTRIBUTIONS]
    for filename, digest in PINNED_VENDOR_SHA256.items():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise WheelhouseError(f"invalid pinned vendor SHA-256 for {filename}")
    if vendor_dir is None or not vendor_dir.is_dir():
        return [], [f"{name}=={version}" for name, version in OPTIONAL_VENDOR_DISTRIBUTIONS]

    for name, version in OPTIONAL_VENDOR_DISTRIBUTIONS:
        candidates = _vendor_candidate_paths(vendor_dir, name, version)
        verified = [
            path for path in candidates
            if path.name in PINNED_VENDOR_SHA256
            and hashlib.sha256(path.read_bytes()).hexdigest() == PINNED_VENDOR_SHA256[path.name]
        ]
        if not verified:
            missing.append(f"{name}=={version}")
            continue
        verified.sort(key=lambda path: (0 if path.name.endswith(".whl") else 1, path.name))
        chosen.append(verified[0])

    if missing:
        return [], missing
    return chosen, []


def _existing_wheel_bytes(output: Path) -> dict[str, bytes]:
    if not output.is_dir() or not list(output.glob("*.whl")):
        return {}
    _indexed_wheels(output)
    return {path.name: path.read_bytes() for path in output.glob("*.whl")}


def build_wheelhouse(
    site_packages: Path, output: Path, vendor_dir: Path | None = None,
    wheel_dir: Path | None = None,
) -> None:
    _prepare_packaging(output if (output / "SHA256SUMS").is_file() else ROOT / "wheelhouse")
    from packaging.utils import parse_wheel_filename

    existing = _existing_wheel_bytes(output)
    vendor_sources, missing_vendor = _verified_vendor_sources(vendor_dir)
    if missing_vendor:
        print("Optional vendor distributions missing: " + ", ".join(missing_vendor))
    with tempfile.TemporaryDirectory(prefix="wheelhouse-") as temporary:
        staging = Path(temporary)
        by_name: dict[str, list[Path]] = {}

        def add_wheel(filename: str, data: bytes) -> None:
            path = staging / filename
            if path.exists():
                # Existing verified release wheels retain their exact bytes.
                return
            name, version, _, _ = parse_wheel_filename(filename)
            path.write_bytes(data)
            _verify_wheel(path, name, str(version))
            by_name.setdefault(name, []).append(path)

        for filename, data in existing.items():
            add_wheel(filename, data)
        if wheel_dir is not None:
            if not wheel_dir.is_dir():
                raise WheelhouseError(f"wheel source directory is missing: {wheel_dir}")
            for path in sorted(wheel_dir.glob("*.whl")):
                add_wheel(path.name, path.read_bytes())
        for (name, version), source in zip(OPTIONAL_VENDOR_DISTRIBUTIONS, vendor_sources):
            if source.suffix == ".whl":
                _verify_wheel(source, _canonicalize(name), version)
                target_name = _wheel_filename(name, version)
                if source.name != target_name:
                    raise WheelhouseError(f"pinned vendor wheel must be named {target_name}: {source.name}")
                add_wheel(source.name, source.read_bytes())
            else:
                filename, data, _, _ = _build_vendor_sdist(source, name, version)
                add_wheel(filename, data)

        def candidates(name: str) -> list[Path]:
            if name not in by_name:
                filename, data, _, _ = _build_wheel(site_packages, name)
                add_wheel(filename, data)
            return by_name[name]

        closure = _runtime_closure(candidates, _runtime_roots())
        selected = {path.name: path for _, paths in closure.values() for path in paths}
        extra_existing = set(existing) - set(selected)
        if extra_existing:
            raise WheelhouseError(
                "refusing to remove existing wheelhouse wheels no longer selected: " + ", ".join(sorted(extra_existing))
            )
        # Index and verify the full candidate before changing any release artifacts.
        # Sources not used by this target matrix are kept outside the final candidate.
        candidate = staging / "selected"
        candidate.mkdir()
        for filename, source in selected.items():
            (candidate / filename).write_bytes(source.read_bytes())
        (candidate / "requirements.txt").write_text(
            "".join(f"{name}=={version}\n" for name, (version, _) in sorted(closure.items())), encoding="utf-8",
        )
        (candidate / "SHA256SUMS").write_text(
            "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {filename}\n"
                    for filename, path in sorted(selected.items())), encoding="utf-8",
        )
        verify_wheelhouse(candidate)
        output.mkdir(parents=True, exist_ok=True)
        for filename in selected:
            if filename not in existing:
                with (output / filename).open("xb") as handle:
                    handle.write((candidate / filename).read_bytes())
        for index in ("requirements.txt", "SHA256SUMS"):
            (output / index).write_bytes((candidate / index).read_bytes())
        verify_wheelhouse(output)

        print(f"Built and verified offline wheelhouse at {output.resolve()}")
        for filename, path in sorted(selected.items()):
            metadata = _wheel_metadata(path)
            print(f"  {metadata['Name']}=={metadata['Version']}: {filename} (license: {_license(metadata)})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "wheelhouse", help="wheelhouse output directory")
    parser.add_argument(
        "--site-packages", type=Path, default=DEFAULT_SITE_PACKAGES,
        help="installed pure-Python distributions to package",
    )
    parser.add_argument("--vendor-dir", type=Path, help="local directory with SHA-256-pinned vendor files")
    parser.add_argument("--wheel-dir", type=Path, help="downloaded universal and pinned platform wheels")
    parser.add_argument("--verify", action="store_true", help="verify an existing wheelhouse without rebuilding")
    parser.add_argument("--print-requirements", action="store_true",
                        help="verify the complete wheelhouse and print only installation pins")
    parser.add_argument("--no-default-extras", action="store_true",
                        help="omit default extras from --print-requirements (still verify the complete wheelhouse)")
    arguments = parser.parse_args(argv)
    if arguments.no_default_extras and not arguments.print_requirements:
        parser.error("--no-default-extras requires --print-requirements")
    try:
        if arguments.print_requirements:
            requirements = verify_wheelhouse(arguments.output)
            if arguments.no_default_extras:
                _, by_name = _indexed_wheels(arguments.output)
                closure = _runtime_closure(lambda name: by_name.get(name, []),
                                           _runtime_roots(include_default_extras=False))
                requirements = [pin for pin in requirements if pin[0] in closure]
            for _, _, pin in requirements:
                print(pin)
        elif arguments.verify:
            requirements = verify_wheelhouse(arguments.output)
            print(f"Verified offline wheelhouse at {arguments.output.resolve()}")
            for _, _, pin in requirements:
                print(f"  {pin}")
        else:
            build_wheelhouse(arguments.site_packages, arguments.output, arguments.vendor_dir, arguments.wheel_dir)
    except (WheelhouseError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"build_wheelhouse.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
