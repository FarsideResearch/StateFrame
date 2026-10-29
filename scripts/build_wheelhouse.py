#!/usr/bin/env python3
"""Build and verify StateFrame's offline, pure-Python wheelhouse."""
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
import zipfile
from email.message import Message
from email.parser import BytesParser
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE_PACKAGES = Path("/opt/homebrew/lib/python3.14/site-packages")
REQUIRED_DISTRIBUTIONS = (
    "pytest", "pluggy", "iniconfig", "packaging", "pygments", "attrs", "PyYAML",
    "jedi", "parso", "networkx",
)
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
    expected = [_canonicalize(name) for name in REQUIRED_DISTRIBUTIONS]
    optional = [_canonicalize(name) for name, _ in OPTIONAL_VENDOR_DISTRIBUTIONS]
    if names[:len(expected)] != expected or names[len(expected):] not in ([], optional):
        raise WheelhouseError(
            "requirements.txt must pin the required distributions in order, followed by the optional vendor pair"
        )
    versions = {name: version for name, version, _ in parsed}
    for name, version in PINNED_LOCAL_VERSIONS.items():
        if versions.get(_canonicalize(name)) != version:
            raise WheelhouseError(f"requirements.txt must pin {name}=={version}")
    for name, version in OPTIONAL_VENDOR_DISTRIBUTIONS:
        if _canonicalize(name) in versions and versions[_canonicalize(name)] != version:
            raise WheelhouseError(f"requirements.txt must pin {name}=={version}")
    return parsed


def _verify_wheel(path: Path, requirement_name: str, version: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise WheelhouseError(f"wheel has a corrupt ZIP member: {path.name}")
            names = set(archive.namelist())
            if any(_is_compiled_file(name) for name in names):
                raise WheelhouseError(f"wheel contains compiled files: {path.name}")
            info_candidates = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(info_candidates) != 1:
                raise WheelhouseError(f"wheel must contain one METADATA file: {path.name}")
            info_root = info_candidates[0].removesuffix("/METADATA")
            metadata = BytesParser().parsebytes(archive.read(info_candidates[0]))
            if _canonicalize(metadata.get("Name", "")) != requirement_name or metadata.get("Version") != version:
                raise WheelhouseError(f"wheel metadata does not match its pin: {path.name}")
            wheel = archive.read(f"{info_root}/WHEEL").decode("utf-8")
            if "Root-Is-Purelib: true" not in wheel or "Tag: py3-none-any" not in wheel:
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
            if recorded != names:
                raise WheelhouseError(f"wheel RECORD does not cover every file: {path.name}")
    except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile, csv.Error, ValueError) as error:
        raise WheelhouseError(f"cannot read wheel {path}: {error}") from error


def verify_wheelhouse(directory: Path) -> list[tuple[str, str, str]]:
    if not directory.is_dir():
        raise WheelhouseError(f"wheelhouse directory is missing: {directory}")
    requirements = _parse_requirements(directory / "requirements.txt")
    expected_files: set[str] = set()
    for name, version, _ in requirements:
        filename = _wheel_filename(name, version)
        expected_files.add(filename)
        wheel_path = directory / filename
        if not wheel_path.is_file():
            raise WheelhouseError(f"missing pinned wheel: {wheel_path}")
        _verify_wheel(wheel_path, name, version)

    actual_wheels = {path.name for path in directory.glob("*.whl")}
    if actual_wheels != expected_files:
        extra = sorted(actual_wheels - expected_files)
        missing = sorted(expected_files - actual_wheels)
        raise WheelhouseError(f"wheelhouse wheel set mismatch (extra={extra}, missing={missing})")

    checksums_path = directory / "SHA256SUMS"
    if not checksums_path.is_file():
        raise WheelhouseError(f"wheelhouse checksum file is missing: {checksums_path}")
    checksums: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+\.whl)", line)
        if not match:
            raise WheelhouseError(f"malformed checksum line in {checksums_path}: {line!r}")
        digest, filename = match.groups()
        if filename in checksums:
            raise WheelhouseError(f"duplicate checksum entry for {filename}")
        checksums[filename] = digest
    if set(checksums) != expected_files:
        raise WheelhouseError("SHA256SUMS does not list exactly the pinned wheel files")
    for filename, expected in checksums.items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise WheelhouseError(f"checksum mismatch for {directory / filename}")
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
    wheels = sorted(output.glob("*.whl")) if output.is_dir() else []
    if not wheels:
        return {}
    if not (output / "requirements.txt").is_file() or not (output / "SHA256SUMS").is_file():
        raise WheelhouseError(f"refusing to update a nonempty unindexed wheelhouse: {output}")
    pins: dict[str, tuple[str, str]] = {}
    for line_number, line in enumerate((output / "requirements.txt").read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9.!+_-]+)", line.strip())
        if not match:
            raise WheelhouseError(f"invalid existing wheelhouse pin at {output / 'requirements.txt'}:{line_number}")
        name, version = match.groups()
        filename = _wheel_filename(name, version)
        if filename in pins:
            raise WheelhouseError(f"duplicate existing wheelhouse pin: {line}")
        pins[filename] = (_canonicalize(name), version)
    wheel_filenames = {path.name for path in wheels}
    if not set(pins).issubset(wheel_filenames):
        raise WheelhouseError(f"existing wheelhouse requirements reference missing wheels: {output}")
    checksums: dict[str, str] = {}
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+\.whl)", line)
        if not match:
            raise WheelhouseError(f"malformed existing wheelhouse checksum line: {line!r}")
        digest, filename = match.groups()
        if filename in checksums:
            raise WheelhouseError(f"duplicate existing checksum entry: {filename}")
        checksums[filename] = digest
    if set(checksums) != set(pins):
        raise WheelhouseError(f"existing wheelhouse checksums do not match its wheel files: {output}")
    for filename, (name, version) in pins.items():
        wheel = output / filename
        _verify_wheel(wheel, name, version)
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != checksums[filename]:
            raise WheelhouseError(f"existing wheelhouse checksum mismatch: {wheel}")
    for wheel in wheels:
        if wheel.name in pins:
            continue
        try:
            with zipfile.ZipFile(wheel) as archive:
                metadata_entries = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if len(metadata_entries) != 1:
                    raise WheelhouseError(f"unindexed wheel must contain one METADATA file: {wheel.name}")
                message = BytesParser().parsebytes(archive.read(metadata_entries[0]))
        except (OSError, KeyError, UnicodeDecodeError, zipfile.BadZipFile) as error:
            raise WheelhouseError(f"cannot inspect unindexed wheel {wheel}: {error}") from error
        name = message.get("Name", "")
        version = message.get("Version", "")
        if not name or not version or wheel.name != _wheel_filename(name, version):
            raise WheelhouseError(f"unindexed wheel has an unexpected filename or identity: {wheel.name}")
        _verify_wheel(wheel, _canonicalize(name), version)
    return {path.name: path.read_bytes() for path in wheels}


def build_wheelhouse(site_packages: Path, output: Path, vendor_dir: Path | None = None) -> None:
    if not site_packages.is_dir():
        raise WheelhouseError(f"site-packages directory is missing: {site_packages}")

    existing = _existing_wheel_bytes(output)
    built = [_build_wheel(site_packages, name) for name in REQUIRED_DISTRIBUTIONS]
    vendor_sources, missing_vendor = _verified_vendor_sources(vendor_dir)
    if missing_vendor:
        print("Optional vendor distributions missing: " + ", ".join(missing_vendor))
    else:
        for (name, version), source in zip(OPTIONAL_VENDOR_DISTRIBUTIONS, vendor_sources):
            if source.suffix == ".whl":
                _verify_wheel(source, _canonicalize(name), version)
                target_name = _wheel_filename(name, version)
                if source.name != target_name:
                    raise WheelhouseError(f"pinned vendor wheel must be named {target_name}: {source.name}")
                with zipfile.ZipFile(source) as archive:
                    metadata_entry = next(item for item in archive.namelist() if item.endswith(".dist-info/METADATA"))
                    metadata = BytesParser().parsebytes(archive.read(metadata_entry))
                built.append((target_name, source.read_bytes(), _license(metadata), f"{name}=={version}"))
            else:
                built.append(_build_vendor_sdist(source, name, version))

    output.mkdir(parents=True, exist_ok=True)
    by_filename = {filename: (data, license_name, pin) for filename, data, license_name, pin in built}
    extra_existing = set(existing) - set(by_filename)
    if extra_existing:
        raise WheelhouseError(
            "refusing to remove existing wheelhouse wheels no longer selected: " + ", ".join(sorted(extra_existing))
        )
    for filename, data, _, pin in built:
        prior = existing.get(filename)
        if prior is not None:
            # Keep verified existing wheels intact; generated indexes are refreshed from their bytes below.
            name, version = pin.split("==", 1)
            _verify_wheel(output / filename, _canonicalize(name), version)
            continue
        else:
            (output / filename).write_bytes(data)

    selected: dict[str, tuple[bytes, str, str]] = {}
    for filename, (data, license_name, pin) in by_filename.items():
        selected[filename] = (existing.get(filename, data), license_name, pin)
    requirements_path = output / "requirements.txt"
    checksums_path = output / "SHA256SUMS"
    requirements_path.write_text("".join(f"{pin}\n" for _, _, _, pin in built), encoding="utf-8")
    checksum_lines = [
        f"{hashlib.sha256(data).hexdigest()}  {filename}"
        for filename, (data, _, _) in sorted(selected.items())
    ]
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    verify_wheelhouse(output)

    print(f"Built and verified offline wheelhouse at {output.resolve()}")
    for filename, (_, license_name, pin) in sorted(selected.items()):
        print(f"  {pin}: {filename} (license: {license_name})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "wheelhouse", help="wheelhouse output directory")
    parser.add_argument(
        "--site-packages", type=Path, default=DEFAULT_SITE_PACKAGES,
        help="installed pure-Python distributions to package",
    )
    parser.add_argument("--vendor-dir", type=Path, help="local directory with SHA-256-pinned vendor files")
    parser.add_argument("--verify", action="store_true", help="verify an existing wheelhouse without rebuilding")
    arguments = parser.parse_args(argv)
    try:
        if arguments.verify:
            requirements = verify_wheelhouse(arguments.output)
            print(f"Verified offline wheelhouse at {arguments.output.resolve()}")
            for _, _, pin in requirements:
                print(f"  {pin}")
        else:
            build_wheelhouse(arguments.site_packages, arguments.output, arguments.vendor_dir)
    except (WheelhouseError, OSError, subprocess.SubprocessError) as error:
        print(f"build_wheelhouse.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
