"""Explicit study-only immutable Python releases; no import-time activation."""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys

TREES = (
    "probes", "detreason", "build_tasks", "graphs", "architecture", "swarm",
    "mc_repo_intel_vendor", "sys1",
    "tools",  # 10-08: shipped tools/sciledger subset (final-product manifest writer)
)
FORMAT = "stateframe-runtime-release/1"
ADMISSION_ENV = "STATEFRAME_MECHANISM_ADMISSION"
ADMISSION_FIELD = "mechanism_admission"
_ADMISSION_UNSET = object()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def admission_gate_enabled() -> bool:
    """Read the opt-in gate without importing the admission module when off."""
    if os.environ.get(ADMISSION_ENV, "off") == "off":
        return False
    from kernel_mechanism_admission import admission_enabled

    return admission_enabled()


def admission_required_switches() -> list[str]:
    from kernel_mechanism_admission import required_switches

    return required_switches()


def inventory(root: Path) -> dict[str, bytes]:
    """The explicit Python-only universe includes the old freeze universe."""
    root = root.resolve(strict=True)
    paths = list(root.glob("*.py"))
    for name in TREES:
        directory = root / name
        if directory.is_symlink():
            raise ValueError(f"release tree is a symlink: {name}")
        if directory.exists():
            paths += list(directory.rglob("*.py"))
    platform_directory = root / "platform"
    if platform_directory.exists():
        paths += list(platform_directory.rglob("*.py"))
    result = {}
    for path in sorted(paths):
        relative = path.relative_to(root)
        if any(part.is_symlink() for part in [path, *path.parents] if part != root):
            raise ValueError(f"release source is a symlink: {relative}")
        result[relative.as_posix()] = path.read_bytes()
    if not result:
        raise ValueError("empty runtime release")
    return result


def create_release(
    source: Path,
    destination: Path,
    *,
    required_switches: object = _ADMISSION_UNSET,
) -> Path:
    """Create exclusively; a partial or different existing release is rejected."""
    source = source.resolve(strict=True)
    data = inventory(source)
    admission = None
    requested_switches = None
    admission_mode = os.environ.get(ADMISSION_ENV, "off")
    if admission_mode != "off":
        from kernel_mechanism_admission import (
            load_manifest,
            require_declaration,
            required_switches as parse_required_switches,
            resolve_policy,
        )

        policy = resolve_policy()
    else:
        policy = None
    if policy is not None and policy.enabled:
        if required_switches is _ADMISSION_UNSET:
            require_declaration(policy)
        explicit = (
            []
            if required_switches is _ADMISSION_UNSET
            else parse_required_switches(required_switches)
        )
        requested_switches = sorted(set(policy.required_switches) | set(explicit))
        admission = load_manifest(source, required_switches=requested_switches)
    manifest = {"format": FORMAT, "files": {name: digest(raw) for name, raw in data.items()}}
    if admission is not None:
        manifest[ADMISSION_FIELD] = admission
    identity = digest(canonical(manifest))
    destination.mkdir(parents=True, exist_ok=True)
    release = destination.resolve() / identity
    if release.exists():
        verify_release(release)
        if admission is not None:
            check_release_admission(release, required_switches=requested_switches)
        return release
    release.mkdir()
    for name, raw in data.items():
        path = release / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    # Guard implementation resolves these relative to its own source. They are
    # empty placeholders, never copied evidence or a replacement corpus.
    for name in ("results", "results.SNAPSHOT-20260920"):
        (release / name).mkdir()
    with (release / "release-manifest.json").open("xb") as output:
        output.write(canonical(manifest))
        output.flush()
        os.fsync(output.fileno())
    if inventory(source) != data:
        raise RuntimeError("source changed during release creation; release not accepted")
    verify_release(release)
    if admission is not None:
        check_release_admission(release, required_switches=requested_switches)
    return release


def verify_release(release: Path) -> dict:
    release = release.resolve(strict=True)
    path = release / "release-manifest.json"
    if path.is_symlink():
        raise ValueError("symlink release manifest")
    manifest = json.loads(path.read_bytes())
    if manifest.get("format") != FORMAT or not isinstance(manifest.get("files"), dict):
        raise ValueError("invalid release manifest")
    if ADMISSION_FIELD in manifest:
        from kernel_mechanism_admission import validate_manifest

        validate_manifest(manifest[ADMISSION_FIELD])
    if digest(canonical(manifest)) != release.name:
        raise ValueError("release identity mismatch")
    actual = {name: digest(raw) for name, raw in inventory(release).items()}
    if actual != manifest["files"]:
        raise ValueError("release source hash mismatch")
    # No data or unrecorded resources silently join the runtime package.
    permitted = {*actual, "release-manifest.json"}
    for path in release.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink within runtime release")
        if path.is_file() and path.relative_to(release).as_posix() not in permitted:
            raise ValueError(f"unrecorded release resource: {path}")
    return manifest


def check_release_admission(
    release: Path,
    *,
    required_switches: object | None = None,
) -> dict:
    """R4x freeze/runner check: reject a release without complete evidence."""
    manifest = verify_release(release)
    from kernel_mechanism_admission import validate_manifest

    block = manifest.get(ADMISSION_FIELD)
    if block is None:
        raise ValueError("release missing mechanism admission manifest")
    validate_manifest(block, required_switches=required_switches)
    return manifest


class ReleaseImports(importlib.abc.MetaPathFinder):
    """Reject known runtime names outside release, before executing their code."""

    def __init__(self, release: Path, source: Path):
        self.release = release.resolve(strict=True)
        self.source = source.resolve(strict=True)
        manifest = verify_release(self.release)
        self.files = manifest["files"]
        self.names = {Path(name).stem for name in self.files
                      if "/" not in name or Path(name).parent.as_posix() == "probes"}
        self.names.update(TREES)

    def check(self, name: str, origin: str | None, locations=()) -> None:
        runtime_name = name.split(".")[0] in self.names
        paths = list(locations or ())
        if origin and origin not in {"built-in", "frozen"}:
            paths.append(origin)
        for raw in paths:
            path = Path(raw).resolve()
            if path.is_relative_to(self.release):
                if path.is_file() and path.relative_to(self.release).as_posix() not in self.files:
                    raise ImportError(f"unrecorded runtime import: {name}")
            elif runtime_name or path.is_relative_to(self.source):
                raise ImportError(f"mutable or foreign runtime import: {name}: {path}")

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None:
            self.check(fullname, spec.origin, spec.submodule_search_locations)
        return spec

    def snapshot(self) -> dict[str, str]:
        result = {}
        for name, module in list(sys.modules.items()):
            origin = getattr(module, "__file__", None)
            self.check(name, origin, getattr(module, "__path__", ()))
            if origin and Path(origin).resolve().is_relative_to(self.release):
                result[name] = str(Path(origin).resolve())
        return result


def activate_release(release: Path, source: Path) -> ReleaseImports:
    fence = ReleaseImports(release, source)
    sys.path[:] = [str(fence.release), *[
        value for value in sys.path
        if value and not Path(value).resolve().is_relative_to(fence.source)
        and Path(value).resolve() != fence.release
    ]]
    # Loaded source modules must not survive a late activation.
    fence.snapshot()
    sys.meta_path.insert(0, fence)
    return fence
