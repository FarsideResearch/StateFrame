"""Canonical identities for the actual public checks, including selection filters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path



def requested_check_targets(root: Path, explicit: str) -> list[str]:
    """Parse a tool target list without executing shell syntax or guessing a suffix.

    Existing exact paths/node IDs win, including spaces in paths or parameters.
    Otherwise shell-style quoting separates selectors; options are not this API.
    """
    import shlex
    if not explicit or not explicit.strip():
        return []
    if (root / explicit.split("::", 1)[0]).exists():
        return [explicit] if _contained(root, explicit) else []
    try:
        targets = shlex.split(explicit)
    except ValueError:
        return []
    if any(not target or target.startswith("-") or target in {";", "|", "&&", "||", "&"}
           or not _contained(root, target) for target in targets):
        return []
    return targets


def _contained(root: Path, target: str) -> bool:
    """A target's file part must resolve inside root (no absolute-outside or '..' escape)."""
    try:
        return (root / target.split("::", 1)[0]).resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False

def check_selectors(root: Path, targets: list[str], kexpr: str | None = None) -> list[str]:
    result = []
    for target in targets:
        base, separator, node = str(target).partition('::')
        path = Path(base)
        resolved = (root / path).resolve()
        if resolved.is_relative_to(root.resolve()):
            base = resolved.relative_to(root.resolve()).as_posix()
        selector = base + (separator + node if separator else '')
        # -k can select an arbitrary subset. It cannot prove an unfiltered file.
        if kexpr:
            selector += f'::filtered[-k={kexpr}]'
        result.append(selector)
    return result


def public_check_files(root: Path, selectors: list[str]) -> list[str]:
    """Only existing in-workspace files; private node names/parameters are not copied."""
    files = set()
    for selector in selectors:
        path = (root / str(selector).split('::', 1)[0]).resolve()
        if path.is_relative_to(root.resolve()) and path.is_file():
            files.add(path.relative_to(root.resolve()).as_posix())
    return sorted(files)


def declared_public_check_paths(row: dict) -> list[str] | None:
    """Validate explicitly public file checks without consulting oracle metadata.

    Sources must be declared public workspace parents or separate public-test
    files. Node selectors and absent sources cannot gain public status by
    appearing in this declaration.
    None preserves the legacy selector-discovery path.
    """
    from build_tasks.multifile_row import normalize_repo_path
    from kernel_live_projection import is_harness_path

    if 'public_check_paths' not in row:
        return None
    paths = row['public_check_paths']
    files = row.get('files')
    if not isinstance(paths, list) or not paths or not isinstance(files, (Mapping, Sequence)) \
            or isinstance(files, (str, bytes)):
        raise ValueError('public_check_paths requires nonempty paths and public files')
    result = []
    for path in paths:
        if not isinstance(path, str) or '::' in path or not path.endswith('.py'):
            raise ValueError('public checks must be Python file paths, not node selectors')
        normalized = normalize_repo_path(path, field='public check path')
        if normalized != path or path in result:
            raise ValueError('public check paths must be unique and normalized')
        path_parts = Path(path).parts
        if any(part.lower() in {
                'grader', 'graders', 'grading', 'gold', 'oracle', 'private', 'hidden',
        } for part in path_parts):
            raise ValueError('public check paths cannot enter harness/grader data')
        if not is_harness_path(Path(path)):
            raise ValueError('public check paths must identify declared test files')
        if isinstance(files, Mapping):
            payload = files.get(path)
        else:
            payload = next((item for item in files
                            if isinstance(item, Mapping) and item.get("path") == path), None)
        if isinstance(payload, Mapping) and isinstance(payload.get('parent_src'), str):
            result.append(path)
            continue
        if isinstance(payload, Mapping) and isinstance(
                payload.get("source", payload.get("content")), str):
            result.append(path)
            continue
        if _public_test_source(row, path) is None:
            raise ValueError('public check source must already exist in declared public files')
        result.append(path)
    return result


def _public_test_source(row: Mapping[str, object], path: str) -> str | None:
    """Return one explicitly public test source without consulting oracle fields.

    Multi-file repair rows historically put acceptance files in ``files``.  New
    task manifests keep them in ``public_tests`` instead, so the worker must be
    able to consume both forms.  This helper deliberately does not inspect
    ``oracle``/``private`` data.
    """
    for key in ("public_tests", "public_test_files"):
        value = row.get(key)
        if isinstance(value, Mapping):
            item = value.get(path)
            if isinstance(item, str):
                return item
            if isinstance(item, Mapping):
                source = item.get("source", item.get("content", item.get("parent_src")))
                if isinstance(source, str):
                    return source
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                if not isinstance(item, Mapping) or item.get("path") != path:
                    continue
                source = item.get("source", item.get("content", item.get("parent_src")))
                if isinstance(source, str):
                    return source
    files = row.get("files")
    if isinstance(files, Mapping):
        item = files.get(path)
        if isinstance(item, Mapping):
            source = item.get("parent_src", item.get("source", item.get("content")))
            if isinstance(source, str):
                return source
    elif isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
        for item in files:
            if isinstance(item, Mapping) and item.get("path") == path:
                source = item.get("parent_src", item.get("source", item.get("content")))
                if isinstance(source, str):
                    return source
    for key in ("public_test_source", "public_test_src", "test_source", "test_src"):
        source = row.get(key)
        declared_path = row.get("public_test_path", row.get("test_path"))
        if isinstance(source, str) and (declared_path == path or declared_path is None):
            return source
    return None


def public_test_sources(row: Mapping[str, object]) -> dict[str, str]:
    """Return all declared public test sources keyed by normalized workspace path."""
    declared = declared_public_check_paths(dict(row))
    if declared is None:
        declared = []
        for key in ("public_tests", "public_test_files"):
            value = row.get(key)
            if isinstance(value, Mapping):
                declared.extend(str(path) for path in value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                declared.extend(str(item.get("path")) for item in value
                                if isinstance(item, Mapping) and isinstance(item.get("path"), str))
        if not declared:
            path = row.get("public_test_path", row.get("test_path"))
            if isinstance(path, str):
                declared.append(path)
        declared = list(dict.fromkeys(declared))
    sources: dict[str, str] = {}
    for path in declared:
        source = _public_test_source(row, path)
        if source is None:
            payload = row.get("files")
            if isinstance(payload, Mapping):
                item = payload.get(path)
            elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
                item = next((candidate for candidate in payload
                             if isinstance(candidate, Mapping)
                             and candidate.get("path") == path), None)
            else:
                item = None
            source = item.get("parent_src") if isinstance(item, Mapping) else None
            if not isinstance(source, str) and isinstance(item, Mapping):
                source = item.get("source", item.get("content"))
        if not isinstance(source, str):
            raise ValueError(f"public check source is unavailable: {path}")
        sources[path] = source
    return sources


def materialized_public_check_paths(root: Path, row: dict) -> list[str] | None:
    declared = declared_public_check_paths(row)
    if declared is None:
        return None
    root = root.resolve()
    for path in declared:
        source = root / path
        if source.is_symlink() or not source.resolve().is_relative_to(root) or not source.is_file():
            raise ValueError(f'declared public check is absent or unsafe in workspace: {path}')
    return declared
