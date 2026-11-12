"""Shared public check selection and pristine test overlay for shipped grading."""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
from collections.abc import Mapping

_ENGINE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ENGINE))
from kernel_safe_paths import safe_expanduser
from build_test_path import resolve_build_test_path
from probes import react_repair as rr

def _swebench_workspace(d):
    """Find the cached base/gold worktree for one SWE-bench row."""
    from build_tasks.swebench_env import WorkspaceLookupError, find_workspace

    roots = []
    configured = os.environ.get("SWEBENCH_WORKSPACE_CACHE_DIR")
    if configured:
        roots.append(pathlib.Path(configured))
    python = d.get("python")
    if isinstance(python, str) and python:
        try:
            env_cache = safe_expanduser(pathlib.Path(python)).parents[3]
            roots.extend((env_cache, *sorted(env_cache.parent.glob("swebench*"))))
        except IndexError:
            pass
    roots.append(pathlib.Path(tempfile.gettempdir()) / "swebench-lite-cache")

    seen = set()
    last_error = None
    for cache in roots:
        cache = safe_expanduser(cache).resolve()
        if cache in seen:
            continue
        seen.add(cache)
        try:
            return find_workspace(cache, d)
        except WorkspaceLookupError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise WorkspaceLookupError("SWE-bench workspace cache is not configured")

def _swebench_path(root: pathlib.Path, value, label: str) -> pathlib.Path:
    relative = pathlib.PurePosixPath(str(value))
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts \
            or root.resolve() not in path.resolve().parents:
        raise ValueError(f"SWE-bench {label} path escapes checkout: {value}")
    return path

def _write_swebench_runner(root: pathlib.Path, d) -> None:
    runner = _swebench_path(root, d["test_path"], "runner")
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(d["test_src"], encoding="utf-8")

def _is_multifile_row(d) -> bool:
    """Return whether ``d`` uses the H10 whole-workspace contract."""
    return isinstance(d.get("files"), Mapping)

def _declared_build_test_path(row) -> str:
    """Return the build oracle path from the shared task contract.

    The function name is not a reliable filename.  The resolver follows the
    legacy declarations and the path exposed in the task text, and never
    recreates a historical ``test_<fn>`` fallback.
    """
    return resolve_build_test_path(row, row.get("fn") if isinstance(row.get("fn"), str) else None)

def overlay_fix_tests(root: pathlib.Path, d) -> int:
    """Materialise the FIX COMMIT'S test files into the sandbox. Returns files written.

    Measured: `article_publishable_projection`'s discriminating test was ADDED by the fix
    commit (+16 lines in test_publication_decisions.py), so in the era-coherent parent
    sandbox it does not exist and the preflight said "no usable oracle". That verdict is
    correct FOR THE AGENT'S WORLD — and wrong for GRADING. This is exactly SWE-bench's
    fail-to-pass design: the agent never sees the fix-era tests (no leak, they are absent
    from its sandbox), and the grader overlays them afterwards as the hidden oracle.
    """
    if _is_multifile_row(d):
        oracle = d.get("oracle")
        test_files = oracle.get("test_files") if isinstance(oracle, Mapping) else {}
        if not isinstance(test_files, Mapping):
            return 0
        for path in sorted(test_files):
            source = test_files[path]
            if not isinstance(source, str):
                raise ValueError(f"multi-file oracle test source is not text: {path}")
            destination = _swebench_path(root, path, "oracle test")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source, encoding="utf-8")
        return len(test_files)
    if d.get("source_dataset") and d.get("base_commit"):
        _write_swebench_runner(root, d)
        return 1
    if d.get("repo") == "build":
        # E380: the pristine spec IS the oracle — rewriting it is also the anti-weakening
        # restore (an agent that gutted the spec tests gets graded on the real spec).
        tdir = root / "tests"
        tdir.mkdir(parents=True, exist_ok=True)
        destination = root / _declared_build_test_path(d)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(d["test_src"])
        return 1
    # E387: a REGISTERED corpus fix is the AUTHORITATIVE oracle — the fix commit either
    # shipped no usable test or shipped a dishonest one (inspect.getsource, inverted
    # rewards); replacement is two-sided-verified. Applied after nothing (before
    # the git overlay would be fine too — apply_corpus_fixes writes its own filename),
    # and fail_to_pass_ids() prefers its selector so preflight+grading ask ITS question.
    try:
        from apply_corpus_fixes import apply_corpus_fixes as _acf
        _n = _acf(root, d)
    except Exception as _e:                                            # noqa: BLE001
        print(f"  [corpus-fix] overlay failed for {d.get('fn')}: {type(_e).__name__}",
              flush=True)
        _n = 0
    if _n and d.get("repo") != "mc":
        return _n
    if d.get("repo") != "mc" or not d.get("sha"):
        return 0
    import subprocess
    mc = rr.REPOS["mc"].parent
    ls = subprocess.run(["git", "-C", str(mc), "show", "--name-only", "--format=",
                         d["sha"], "--", "backend/tests/"],
                        capture_output=True, text=True, timeout=60)
    n = 0
    for relpath in ls.stdout.splitlines():
        relpath = relpath.strip()
        if not relpath:
            continue
        show = subprocess.run(["git", "-C", str(mc), "show", f"{d['sha']}:{relpath}"],
                              capture_output=True, text=True, timeout=60)
        if show.returncode != 0:
            continue                      # deleted by the fix; nothing to overlay
        dest = root / pathlib.Path(relpath).relative_to("backend")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(show.stdout)
        n += 1
    return n

def fail_to_pass_ids(d) -> list[str]:
    """The EXACT tests the fix commit added or modified — the true fail-to-pass set.

    Whole-file grading proved too strict: `_ack_decision`'s gold failed on OTHER tests in the
    same file (its commit subject literally says "the test was masking it" — the file was not
    green at sha either). SWE-bench's answer is precise node ids: only the tests the fix
    touched are the oracle. Extract added/modified `def test_...` names from the commit's diff
    of each test file.
    """
    from kernel_public_evidence import declared_public_check_paths
    public_paths = declared_public_check_paths(d)
    if public_paths is not None:
        return public_paths
    if _is_multifile_row(d):
        oracle = d.get("oracle")
        selectors = oracle.get("selectors") if isinstance(oracle, Mapping) else ()
        if not isinstance(selectors, (list, tuple)):
            return []
        seen = set()
        return [selector for selector in selectors
                if isinstance(selector, str) and selector
                and not (selector in seen or seen.add(selector))]
    if d.get("source_dataset") and d.get("base_commit"):
        test_path = d.get("test_path")
        return [test_path] if isinstance(test_path, str) and test_path else []
    if d.get("repo") == "build":
        # E380: ALL spec tests are the oracle (fail-to-pass by construction), so the
        # selector is the whole file — callers use ids[0] as the single pytest target,
        # and per-node ids here would silently grade only the first spec test.
        return [_declared_build_test_path(d)]
    # E387: a registered replacement oracle outranks the fix commit's own tests
    # (which are absent or dishonest for exactly these tasks).
    try:
        from apply_corpus_fixes import corpus_fix_for as _cff
        _fix = _cff(d)
        if _fix is not None:
            return [_fix.selector]
    except Exception:                                                  # noqa: BLE001
        pass
    if d.get("repo") != "mc" or not d.get("sha"):
        return []
    import re as _re
    import subprocess
    mc = rr.REPOS["mc"].parent
    diff = subprocess.run(["git", "-C", str(mc), "diff", f"{d['sha']}^", d["sha"],
                           "--", "backend/tests/"],
                          capture_output=True, text=True, timeout=60)
    ids, cur_file = [], None
    for line in diff.stdout.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:].removeprefix("backend/")
        # ANY `def test_...` appearing in the diff — added ('+'), context (' '), or named in a
        # hunk header — marks a touched test. Measured on `_ack_decision`: the hunk header
        # names the PREVIOUS function (`@@ ... def _ledger_verdict_rows`), and the modified
        # test's own `def` line appears as an unchanged CONTEXT line inside the hunk. Only
        # matching added defs returned []; only matching headers named a helper. Cast wide —
        # a touched-but-unchanged test id in the set is harmless (it just runs), while a
        # missed one silently deletes the oracle.
        m = _re.match(r"[+ ]\s*(?:async\s+)?def (test_\w+)", line)
        if m and cur_file:
            ids.append(f"{cur_file}::{m.group(1)}")
        h = _re.match(r"@@ .+ @@\s*(?:async\s+)?def (test_\w+)", line)
        if h and cur_file:
            ids.append(f"{cur_file}::{h.group(1)}")
    # de-dup, preserve order
    seen = set()
    return [i for i in ids if not (i in seen or seen.add(i))]
