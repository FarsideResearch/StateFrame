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


# --- scoped public-check truth -------------------------------------------------
# A run_tests call that covers only some of the declared public selectors must not read
# as "ALL TESTS PASSED" or invite `done`: the host admits done only when every declared
# selector is green.  These helpers are pure so every renderer (model-facing result, action
# ledger, frame, automatic verification) shares one definition, and the definition of
# "covers" is done admission's own (kernel_verdict._covers): exact selector identity, a
# file never stands in for a node and a node never stands in for its file.

PARTIAL_PASS = "PARTIAL_PASS"


def _selector_covers(ran: str, declared: str) -> bool:
    from kernel_verdict import _covers
    return _covers(ran.strip(), declared.strip())


def declared_task_selectors(row: Mapping[str, object] | None, f2p_ids: Sequence[object],
                            fallback: Sequence[object] = ()) -> list[str]:
    """The task's declared public selectors: the denominator of every coverage statement.

    A generic runner declares exactly its `@check` command.  Otherwise the task row's
    failing-to-passing ids are the declaration; ``fallback`` (the active acceptance scope)
    is used only when the task declares nothing, never to shrink the denominator.
    """
    if isinstance(row, Mapping) and row.get("runner_kind") == "generic":
        if not row.get("test_command"):
            return []
        from kernel_generic_check import CHECK_ID
        return [CHECK_ID]
    out: list[str] = []
    for source in (f2p_ids, fallback):
        for item in source or ():
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
        if out:
            break
    return out


def executed_selectors(selectors: Sequence[object], codes: Sequence[object],
                       row: Mapping[str, object] | None = None) -> list[str]:
    """Selectors whose subprocess actually completed with an exit code.

    A refused or unrun selector (``None`` / negative code) never counts as executed, so it
    can never establish completeness.  A generic runner's identity is its `@check` command,
    whatever target text the model requested.
    """
    if isinstance(row, Mapping) and row.get("runner_kind") == "generic":
        from kernel_generic_check import CHECK_ID
        return [CHECK_ID] if any(type(c) is int and c >= 0 for c in codes) else []
    out: list[str] = []
    for selector, code in zip(selectors, codes):
        text = str(selector).strip()
        if text and type(code) is int and code >= 0 and text not in out:
            out.append(text)
    return out


def uncovered_public_selectors(declared: Sequence[str], ran: Sequence[str]) -> list[str]:
    """Declared public selectors that no selector in ``ran`` covers, in declared order."""
    ran_list = [str(item) for item in ran]
    out: list[str] = []
    for item in declared:
        text = str(item).strip()
        if text and text not in out and not any(_selector_covers(r, text) for r in ran_list):
            out.append(text)
    return out


def scoped_check_complete(declared: Sequence[str], ran: Sequence[str]) -> bool:
    """True when nothing is declared or the run covered every declared selector."""
    return not uncovered_public_selectors(declared, ran)


def scoped_result_class(result_class: str, complete: bool) -> str:
    """An exit-0 run that covered only part of the declaration is PARTIAL_PASS, not GREEN."""
    return PARTIAL_PASS if result_class == "GREEN" and not complete else result_class


def _clip(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_scoped_check_truth(declared: Sequence[str], ran: Sequence[str],
                              outcomes: Mapping[str, int | None] | None = None,
                              exit_code: int | None = None, max_listed: int = 5) -> str | None:
    """One line saying how much of the required check set this run covered.

    Returns None when the run covered every declared selector (callers then keep the
    original text).  ``outcomes`` maps ran selector -> its exit code when known.
    """
    missing = uncovered_public_selectors(declared, ran)
    if not missing:
        return None
    total = len({str(item).strip() for item in declared if str(item).strip()})
    ran_list = [str(item) for item in ran if str(item).strip()]
    covered = total - len(missing)
    codes = dict(outcomes or {})
    if not codes and ran_list and exit_code is not None:
        codes = {item: exit_code for item in ran_list}
    failed = [s for s in ran_list if codes.get(s) not in (0, None)]
    passed = [s for s in ran_list if codes.get(s) == 0]
    listed = ", ".join(_clip(item) for item in missing[:max_listed])
    if len(missing) > max_listed:
        listed += f", +{len(missing) - max_listed} more"
    head: str
    if failed:
        head = (f"FAIL {', '.join(_clip(s) for s in failed[:max_listed])}"
                + (f"; PASS {len(passed)}" if passed else ""))
    elif passed:
        head = f"PASS {', '.join(_clip(s) for s in passed[:max_listed])}"
    elif ran_list:
        head = f"RAN {', '.join(_clip(s) for s in ran_list[:max_listed])} (no pass verdict)"
    else:
        head = "NO required selector completed"
    noun = "selector" if len(missing) == 1 else "selectors"
    return (f"[kernel] {head} ({covered} of {total} required selectors); "
            f"{len(missing)} required {noun} not run: {listed}. "
            "This is a partial check, not a full pass.")


def scoped_pass_ratio(declared: Sequence[str], ran: Sequence[str]) -> tuple[int, int] | None:
    """(covered, total) required selectors when ``ran`` is a partial cover, else None."""
    total = len({str(item).strip() for item in declared if str(item).strip()})
    missing = uncovered_public_selectors(declared, ran)
    return (total - len(missing), total) if missing and total else None


def project_scoped_run(declared: Sequence[str], selectors: Sequence[object],
                       codes: Sequence[object], row: Mapping[str, object] | None = None,
                       ) -> tuple[str | None, bool]:
    """(truth line, scope_complete) for one run from its structured per-selector exit codes.

    Complete (None, True) when nothing is declared or every declared selector executed.
    Refused selectors are not executed, so a refusal can never complete the scope.
    """
    declared_list = [str(item).strip() for item in declared if str(item).strip()]
    if not declared_list:
        return None, True
    ran = executed_selectors(selectors, codes, row)
    if scoped_check_complete(declared_list, ran):
        return None, True
    outcomes = {s: c for s, c in zip((str(x).strip() for x in selectors), codes)
                if type(c) is int and c >= 0}
    if isinstance(row, Mapping) and row.get("runner_kind") == "generic" and ran:
        outcomes = {ran[0]: next(c for c in codes if type(c) is int and c >= 0)}
    return render_scoped_check_truth(declared_list, ran, outcomes), False


def render_scoped_delivery(observation: str, truth_line: str) -> str:
    """Model-facing text for a partial run: no all-pass claim, plus the covered/required line."""
    return observation.replace("ALL TESTS PASSED.", "PASSED (partial check).") + "\n" + truth_line
