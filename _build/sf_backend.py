"""Stdlib-only PEP 517 build backend for StateFrame (no setuptools, no downloads).

Install offline with:  pip install --no-index --no-build-isolation [-e] <product dir>
It builds a pure-Python wheel from src/ (package code, the vendored engine, profiles), an
editable wheel (a .pth file pointing at src/), or an sdist. Metadata comes from pyproject.toml.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import os
import tarfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "stateframe"
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SDIST_TOP_LEVEL = frozenset({"LICENSE", "NOTICE", "README.md", "install.sh", "pyproject.toml",
                             "requirements.txt", "requirements-repomap.txt", "COMMERCIAL-LICENSE.md",
                             "CONTRIBUTING.md", "CLA.md", "CHANGELOG.md", "ARCHITECTURE.md", "QUICKSTART.md",
                             "KNOWN_LIMITATIONS.md", "AGENTS.md", "llms.txt"})
# Onboarding guide for developers and AI coding assistants (AGENTS.md and llms.txt point here).
SDIST_DOCS = frozenset({f"docs/guide/{name}.md" for name in (
    "concepts", "request-lifecycle", "code-map", "running", "extending", "troubleshooting", "design")})
SDIST_DOCS = SDIST_DOCS | {"docs/assets/stateframe-banner.webp"}  # README banner (relative link)
# The installer verifies the bundled wheels with this standalone helper.
SDIST_SCRIPTS = frozenset({"scripts/build_wheelhouse.py"})
# Product regression tests belong to the maintainer tree, not an installed release.
SDIST_TESTS: frozenset[str] = frozenset()
SDIST_WHEELHOUSE = frozenset({
    "wheelhouse/requirements.txt", "wheelhouse/SHA256SUMS",
    "wheelhouse/pytest-9.0.2-py3-none-any.whl", "wheelhouse/pluggy-1.6.0-py3-none-any.whl",
    "wheelhouse/iniconfig-2.3.0-py3-none-any.whl", "wheelhouse/packaging-26.0-py3-none-any.whl",
    "wheelhouse/pygments-2.20.0-py3-none-any.whl", "wheelhouse/attrs-26.1.0-py3-none-any.whl",
    "wheelhouse/pyyaml-6.0.3-py3-none-any.whl", "wheelhouse/jedi-0.20.0-py3-none-any.whl",
    "wheelhouse/parso-0.8.7-py3-none-any.whl", "wheelhouse/networkx-3.6.1-py3-none-any.whl",
})
SDIST_EXAMPLES = frozenset({
    "examples/discount_bug.script.json", "examples/discount_bug/shop/__init__.py",
    "examples/discount_bug/shop/pricing.py", "examples/discount_bug/tests/test_pricing.py",
    "examples/tiny_site.script.json", "examples/tiny_site/README.md",
    "examples/tiny_site/site/README.md", "examples/tiny_site/tests/test_site.py",
    "examples/brief_to_report.script.json", "examples/brief_to_report/README.md",
    "examples/brief_to_report/input/brief.md", "examples/brief_to_report/input/tickets.csv",
    "examples/brief_to_report/report/README.md", "examples/brief_to_report/tests/test_report.py",
})
SDIST_FILES = (SDIST_TOP_LEVEL | SDIST_DOCS | SDIST_SCRIPTS | SDIST_TESTS | SDIST_EXAMPLES | SDIST_WHEELHOUSE
               | {"_build/sf_backend.py", "src/stateframe/THIRD_PARTY_LICENSES.md"})


def _sdist_includes(relative: Path) -> bool:
    if SKIP_PARTS & set(relative.parts):
        return False
    name = relative.as_posix()
    if name in SDIST_FILES:
        return True
    if relative.parts[:2] == ("src", "stateframe"):
        return relative.suffix in {".py", ".json"}
    return False


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def _dist_name() -> str:
    project = _project()
    return f"{project['name'].replace('-', '_')}-{project['version']}"


def _metadata() -> str:
    project = _project()
    lines = ["Metadata-Version: 2.4", f"Name: {project['name']}", f"Version: {project['version']}",
             f"Summary: {project['description']}", f"License-Expression: {project['license']}",
             f"Requires-Python: {project['requires-python']}"]
    lines += [f"License-File: {name}" for name in project.get("license-files", [])]
    lines += [f"Classifier: {c}" for c in project.get("classifiers", [])]
    lines += [f"Requires-Dist: {d}" for d in project.get("dependencies", [])]
    for extra, deps in project.get("optional-dependencies", {}).items():
        lines.append(f"Provides-Extra: {extra}")
        lines += [f'Requires-Dist: {d}; extra == "{extra}"' for d in deps]
    readme = ROOT / project.get("readme", "README.md")
    lines.append("Description-Content-Type: text/markdown")
    return "\n".join(lines) + "\n\n" + (readme.read_text() if readme.is_file() else "")


def _entry_points() -> str:
    scripts = _project().get("scripts", {})
    return "[console_scripts]\n" + "".join(f"{k} = {v}\n" for k, v in scripts.items())


def _record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def _package_files() -> list[tuple[str, bytes]]:
    out = []
    for path in sorted(PACKAGE.rglob("*")):
        # THIRD_PARTY_LICENSES.md ships in the package so NOTICE's pointer is true for a plain wheel install.
        if path.is_file() and not SKIP_PARTS & set(path.parts) and (
                path.suffix in {".py", ".json"} or path.relative_to(PACKAGE).as_posix() == "THIRD_PARTY_LICENSES.md"):
            out.append((path.relative_to(SRC).as_posix(), path.read_bytes()))
    return out


def _write_wheel(directory: str, payload: list[tuple[str, bytes]]) -> str:
    name = _dist_name()
    info = f"{name}.dist-info"
    files = list(payload)
    files.append((f"{info}/METADATA", _metadata().encode()))
    files.append((f"{info}/WHEEL", b"Wheel-Version: 1.0\nGenerator: sf_backend\n"
                                   b"Root-Is-Purelib: true\nTag: py3-none-any\n"))
    files.append((f"{info}/entry_points.txt", _entry_points().encode()))
    for license_name in _project().get("license-files", []):
        files.append((f"{info}/licenses/{license_name}", (ROOT / license_name).read_bytes()))
    record = [_record_line(n, d) for n, d in files] + [f"{info}/RECORD,,"]
    files.append((f"{info}/RECORD", ("\n".join(record) + "\n").encode()))
    wheel_name = f"{name}-py3-none-any.whl"
    with zipfile.ZipFile(Path(directory) / wheel_name, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_name, data in files:
            info_entry = zipfile.ZipInfo(file_name, date_time=(2026, 1, 1, 0, 0, 0))
            info_entry.external_attr = 0o644 << 16
            archive.writestr(info_entry, data, zipfile.ZIP_DEFLATED)
    return wheel_name


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _write_wheel(wheel_directory, _package_files())


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    pth = (str(SRC) + "\n").encode()
    return _write_wheel(wheel_directory, [("_stateframe_editable.pth", pth)])


# 2026-01-01T00:00:00Z, the same fixed time the wheel's zip entries carry.
SDIST_MTIME = 1767225600


def _normalize_tar_member(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = SDIST_MTIME
    info.mode = 0o755 if info.mode & 0o111 else 0o644
    info.pax_headers = {}
    return info


def build_sdist(sdist_directory, config_settings=None):
    name = _dist_name()
    target = Path(sdist_directory) / f"{name}.tar.gz"
    # gzip header mtime 0 and anonymous, fixed-time tar members: no builder account or build time leaks.
    with open(target, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
        # Never walk maintainer directories or previous build/test outputs.
        candidates = {ROOT / relative for relative in SDIST_FILES}
        candidates.update(PACKAGE.rglob("*"))
        for path in sorted(candidates):
            relative = path.relative_to(ROOT)
            if not path.is_file() or not _sdist_includes(relative) or path.is_symlink():
                continue
            archive.add(path, arcname=f"{name}/{relative.as_posix()}", recursive=False,
                        filter=_normalize_tar_member)
        pkg_info = _metadata().encode()
        entry = _normalize_tar_member(tarfile.TarInfo(f"{name}/PKG-INFO"))
        entry.size = len(pkg_info)
        archive.addfile(entry, io.BytesIO(pkg_info))
    return target.name


__all__ = ["build_editable", "build_sdist", "build_wheel", "get_requires_for_build_editable",
           "get_requires_for_build_sdist", "get_requires_for_build_wheel"]
_ = os
