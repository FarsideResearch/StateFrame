from __future__ import annotations

import os
import time
from pathlib import Path

import kernel_ship_tree_smoke as smoke


def test_source_roots_preserves_unique_ordered_roots(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    project = root / "project"
    configured = root / "configured"
    backend = root / "backend"
    extra = root / "extra"
    ambiguous_a = root / "ambiguous_a"
    ambiguous_b = root / "ambiguous_b"
    stdlib_shadow = root / "stdlib_shadow"
    outside = tmp_path / "outside"
    for directory in (
        project,
        configured,
        backend,
        extra,
        ambiguous_a,
        ambiguous_b,
        stdlib_shadow,
        outside,
    ):
        directory.mkdir(parents=True)

    (project / "module.py").write_text(
        "import local_dep\n"
        "import sibling_dep\n"
        "import configured_dep\n"
        "import env_dep\n"
        "import ambiguous_dep\n"
        "import json\n",
        encoding="utf-8",
    )
    (project / "local_dep.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "pytest.ini").write_text("[pytest]\npythonpath = .\n", encoding="utf-8")
    (project / "setup.cfg").write_text(
        "[tool:pytest]\npythonpath = ../configured\n", encoding="utf-8"
    )
    (configured / "configured_dep.py").write_text("VALUE = 1\n", encoding="utf-8")
    (backend / "sibling_dep.py").write_text("VALUE = 1\n", encoding="utf-8")
    (extra / "env_dep.py").write_text("VALUE = 1\n", encoding="utf-8")
    (ambiguous_a / "ambiguous_dep.py").write_text("VALUE = 1\n", encoding="utf-8")
    (ambiguous_b / "ambiguous_dep.py").write_text("VALUE = 1\n", encoding="utf-8")
    (stdlib_shadow / "json.py").write_text("VALUE = 1\n", encoding="utf-8")
    (outside / "outside.py").write_text("VALUE = 1\n", encoding="utf-8")
    # `configured` is already discovered through pytest config and must not be
    # appended a second time when the run's import path is added.
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(configured), str(extra), str(outside)))
    )

    result = smoke.source_roots(
        root,
        ("project/module.py", "../outside/outside.py"),
    )

    assert result == tuple(map(str, (root, project, configured, backend, extra)))


def test_source_roots_large_workspace_has_bounded_provider_work(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    paths: list[str] = []
    source_files: set[Path] = set()
    first_level_dirs: list[Path] = []
    for package_number in range(20):
        package = root / f"package_{package_number:04d}"
        package.mkdir()
        first_level_dirs.append(package)
        (package / "local_api.py").write_text("VALUE = 1\n", encoding="utf-8")
        for file_number in range(249):
            source = package / f"checked_{file_number:03d}.py"
            missing_name = f"missing_api_{package_number:04d}_{file_number:03d}"
            source.write_text(
                f"import local_api\nimport {missing_name}\n", encoding="utf-8"
            )
            paths.append(source.relative_to(root).as_posix())
            source_files.add(source)

    provider_files = sum(
        (directory / "local_api.py").is_file() for directory in first_level_dirs
    )
    assert len(paths) + provider_files == 5_000
    monkeypatch.setenv("PYTHONPATH", "")

    counts = {
        "source_reads": 0,
        "top_level_parse_calls": 0,
        "provider_comparisons": 0,
        "first_level_directory_scans": 0,
        "provider_directory_scans": 0,
    }
    original_read_text = Path.read_text
    original_scandir = os.scandir
    original_top_level_names = smoke._top_level_import_names
    original_first_level_dirs = smoke._first_level_dirs
    provider_directories = {root, *first_level_dirs}

    def counted_read_text(path, *args, **kwargs):
        if path in source_files:
            counts["source_reads"] += 1
        return original_read_text(path, *args, **kwargs)

    def counted_top_level_names(source):
        counts["top_level_parse_calls"] += 1
        return original_top_level_names(source)

    def counted_provides(directory, name):
        counts["provider_comparisons"] += 1
        return name == "local_api" and directory.name.startswith("package_")

    def counted_first_level_dirs(_root):
        counts["first_level_directory_scans"] += 1
        return original_first_level_dirs(_root)

    def counted_scandir(directory):
        if Path(directory) in provider_directories:
            counts["provider_directory_scans"] += 1
        return original_scandir(directory)

    monkeypatch.setattr(Path, "read_text", counted_read_text)
    monkeypatch.setattr(smoke, "_top_level_import_names", counted_top_level_names)
    monkeypatch.setattr(smoke, "_provides", counted_provides)
    monkeypatch.setattr(smoke, "_first_level_dirs", counted_first_level_dirs)
    monkeypatch.setattr(smoke.os, "scandir", counted_scandir)

    started = time.perf_counter()
    result = smoke.source_roots(root, paths)
    elapsed = time.perf_counter() - started

    expected_roots = (root, *first_level_dirs)
    assert result == tuple(map(str, expected_roots))
    violations = []
    if counts["source_reads"] != len(paths):
        violations.append("each checked source must be read once")
    if counts["top_level_parse_calls"] != len(paths):
        violations.append("each checked source must be parsed once")
    if counts["provider_comparisons"] > 5_000:
        violations.append("provider comparisons must stay bounded by indexed roots")
    if counts["first_level_directory_scans"] > 1:
        violations.append("first-level provider directories must be indexed once")
    if counts["provider_directory_scans"] > len(first_level_dirs) + 2:
        violations.append("provider directories must be enumerated once each")
    if elapsed >= 10.0:
        violations.append("source root discovery must finish under 10 seconds")
    assert not violations, (
        f"5,000-file source-root work exceeded its budget: {violations}; "
        f"counts={counts}; elapsed={elapsed:.3f}s"
    )
