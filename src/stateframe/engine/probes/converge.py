"""Run iterative repairs with behavioral and regression checks.

A repair must pass a discriminating test that fails on the original source,
together with the remaining regression checks. AST equality is diagnostic only.
Tasks without a verified discriminator are reported and excluded.
"""
from __future__ import annotations

import json
import hashlib
import os
import pathlib
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping

POC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
sys.path.insert(0, str(POC / "probes"))

import react_repair as rr                                        # noqa: E402
from build_tasks.noncoding_harness import grade_artifact        # noqa: E402
from build_test_path import resolve_build_test_path              # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 21
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
OUT = POC / "results" / ("converge" + os.environ.get("REACT_TAG", "") + ".jsonl")


def _extract_git_archive(
    git_root: pathlib.Path,
    revision: str,
    destination: pathlib.Path,
    paths: tuple[str, ...],
) -> bool:
    """Extract one revision archive without making optional paths all-or-nothing."""
    import subprocess

    archive = subprocess.run(
        ["git", "-C", str(git_root), "archive", revision, *paths],
        capture_output=True,
        timeout=180,
    )
    if archive.returncode != 0:
        return False
    extracted = subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=archive.stdout,
        capture_output=True,
        timeout=180,
    )
    return extracted.returncode == 0


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
            env_cache = pathlib.Path(python).expanduser().parents[3]
            roots.extend((env_cache, *sorted(env_cache.parent.glob("swebench*"))))
        except IndexError:
            pass
    roots.append(pathlib.Path(tempfile.gettempdir()) / "swebench-lite-cache")

    seen = set()
    last_error = None
    for cache in roots:
        cache = cache.expanduser().resolve()
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


def _row_label(d) -> str:
    """Return a stable display/receipt label for legacy and multi-file rows."""
    if _is_multifile_row(d):
        return str(d.get("task_id") or d.get("instance_id") or "multi_file")
    return str(d.get("fn") or d.get("task_id") or d.get("id") or "unknown")


def _persist_noncoding_receipt(receipt: dict[str, object]) -> None:
    """Persist one converge-compatible noncoding receipt for regime joins."""
    transcript_dir = POC / "results" / (
        "react_transcripts" + os.environ.get("REACT_TAG", "")
    )
    transcript_dir.mkdir(parents=True, exist_ok=True)
    fn = str(receipt["fn"])
    (transcript_dir / f"{fn}.jsonl").write_text(
        json.dumps(receipt, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )



def _strip_json_fence(text: str) -> str:
    """Return the JSON body of a ```json ... ``` (or ```) fenced block; else text unchanged."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _noncoding_loop_retries() -> int:
    """Return the bounded non-coding retry count; zero is the explicit kill switch."""
    raw = os.environ.get("NONCODING_LOOP", "2")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        # A malformed experiment flag must not turn a row into a task error; retain the
        # default-on posture while keeping the loop bounded.
        return 2


def _noncoding_failing_adapters(receipt: Mapping[str, object]) -> list[str]:
    """Return adapter names whose deterministic receipts rejected the artifact."""
    results = receipt.get("oracle_results")
    if not isinstance(results, Mapping):
        return []
    return [
        str(name)
        for name, result in results.items()
        if isinstance(result, Mapping) and result.get("pass") is not True
    ]


def _noncoding_evidence_items(evidence: object) -> list[object]:
    """Extract at most three failing claims/fields from one adapter evidence mapping."""
    if not isinstance(evidence, Mapping):
        return []
    items: list[object] = []
    for key in ("claims", "checks", "errors", "invalid", "values"):
        values = evidence.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, Mapping):
                if key == "claims" and value.get("grounded") is True:
                    continue
                if key == "checks" and value.get("pass") is True:
                    continue
                candidate = value.get("quote") or value.get("field")
                if key == "checks" and candidate:
                    candidate = {
                        "field": candidate,
                        "actual": value.get("actual"),
                        "expected": value.get("expected"),
                    }
                if candidate is not None:
                    items.append(candidate)
            else:
                items.append(value)
            if len(items) >= 3:
                return items
    return items


def _noncoding_evidence_text(value: object) -> str:
    """Serialize one evidence item as one bounded line without inventing content."""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text.replace("\r", "\\r").replace("\n", "\\n")[:240]


def _noncoding_contract_text(row: Mapping[str, object]) -> str:
    """Render the row's deterministic output contract for both model arms."""
    spec = row.get("oracle_spec")
    if not isinstance(spec, Mapping):
        return ""

    grounding = spec.get("citation_span_grounding")
    declarations = grounding.get("claim_paths", []) if isinstance(grounding, Mapping) else []
    span_fields: list[str] = []
    atom_fields: list[str] = []
    if isinstance(declarations, list):
        for declaration in declarations:
            if isinstance(declaration, str):
                span_fields.append(declaration)
                continue
            if not isinstance(declaration, Mapping):
                continue
            field = declaration.get("path")
            mode = declaration.get("mode", "span")
            if not isinstance(field, str):
                continue
            if mode == "atoms":
                atom_fields.append(field)
            elif mode == "span":
                span_fields.append(field)

    lines: list[str] = []
    if span_fields:
        lines.append(
            "For each of these fields, copy ONE contiguous input substring "
            "character-for-character, including case and punctuation: "
            + ", ".join(span_fields)
            + "."
        )
    if atom_fields:
        lines.append(
            "These fields may be written in your own words, but every identifier, "
            "number, date and enum value in them must appear verbatim in the input: "
            + ", ".join(atom_fields)
            + "."
        )

    closed_specs = spec.get("closed_set_membership", [])
    if isinstance(closed_specs, list):
        for closed_spec in closed_specs:
            if not isinstance(closed_spec, Mapping):
                continue
            field = closed_spec.get("field")
            allowed = closed_spec.get("allowed")
            if not isinstance(field, str) or not isinstance(allowed, list):
                continue
            allowed_text = ", ".join(str(value) for value in allowed)
            lines.append(
                f"{field}: use only one of {allowed_text} — an allowed value beats "
                "the source's own wording."
            )

    schema_spec = spec.get("schema_validity")
    schema = schema_spec.get("schema") if isinstance(schema_spec, Mapping) else None
    required = schema.get("required") if isinstance(schema, Mapping) else None
    if isinstance(required, list):
        required_fields = [field for field in required if isinstance(field, str)]
        if required_fields:
            lines.append(
                "Required fields (all must be present): "
                + ", ".join(required_fields)
                + "."
            )
    return " ".join(lines)


def _noncoding_contract_feedback(result_name: str, result: Mapping[str, object]) -> list[str]:
    """Return bounded, mode-specific contract reminders for one failed adapter."""
    evidence = result.get("evidence")
    if not isinstance(evidence, Mapping):
        return []

    clauses: list[str] = []
    if result_name == "citation_span_grounding":
        claims = evidence.get("claims")
        if isinstance(claims, list):
            for claim in claims:
                if not isinstance(claim, Mapping) or claim.get("grounded") is True:
                    continue
                field = str(claim.get("field") or "field")
                mode = claim.get("mode", "span")
                if mode == "atoms":
                    clauses.append(
                        "These fields may be written in your own words, but every "
                        "identifier, number, date and enum value in them must appear "
                        f"verbatim in the input: {field}."
                    )
                else:
                    quote = _noncoding_evidence_text(claim.get("quote") or "")
                    clauses.append(
                        "Copy the sentence from the input exactly, character-for-"
                        "character, including case and punctuation — your text "
                        f"`{quote}` differs from the source."
                    )
                if len(clauses) >= 3:
                    break
    elif result_name.startswith("closed_set_membership"):
        allowed = evidence.get("allowed")
        invalid = evidence.get("invalid")
        allowed_text = (
            ", ".join(str(value) for value in allowed)
            if isinstance(allowed, list)
            else _noncoding_evidence_text(allowed)
        )
        invalid_values = invalid if isinstance(invalid, list) else []
        for value in invalid_values[:3]:
            clauses.append(
                f"Allowed values: {allowed_text}; replace "
                f"`{_noncoding_evidence_text(value)}`."
            )
        if not clauses and allowed_text:
            clauses.append(f"Allowed values: {allowed_text}; provide one exactly.")
    elif result_name == "schema_validity":
        errors = evidence.get("errors")
        if isinstance(errors, list):
            for error in errors:
                if not isinstance(error, str) or "missing required field " not in error:
                    continue
                field = error.split("missing required field ", 1)[1].strip().strip("'")
                clauses.append(f"Required fields (all must be present): {field}.")
                if len(clauses) >= 3:
                    break
    return clauses


def _noncoding_feedback(receipt: Mapping[str, object]) -> str:
    """Render deterministic adapter failures for the next turn in the same conversation."""
    results = receipt.get("oracle_results")
    lines: list[str] = []
    if isinstance(results, Mapping):
        for name, result in results.items():
            if not isinstance(result, Mapping) or result.get("pass") is True:
                continue
            reason = str(result.get("reason") or "failed")
            evidence = _noncoding_evidence_items(result.get("evidence"))
            bounded = "; ".join(_noncoding_evidence_text(item) for item in evidence[:3])
            contract = _noncoding_contract_feedback(str(name), result)
            if (
                str(name) == "citation_span_grounding"
                and reason == "no_factual_atoms"
            ):
                data_keys: list[str] = []
                artifact = receipt.get("artifact")
                data = artifact.get("data") if isinstance(artifact, Mapping) else None
                if isinstance(data, Mapping):
                    data_keys = [str(key) for key in data]
                if not data_keys:
                    spec = receipt.get("oracle_spec")
                    schema_spec = (
                        spec.get("schema_validity")
                        if isinstance(spec, Mapping)
                        else None
                    )
                    schema = (
                        schema_spec.get("schema")
                        if isinstance(schema_spec, Mapping)
                        else None
                    )
                    properties = (
                        schema.get("properties") if isinstance(schema, Mapping) else None
                    )
                    data_schema = (
                        properties.get("data") if isinstance(properties, Mapping) else None
                    )
                    data_properties = (
                        data_schema.get("properties")
                        if isinstance(data_schema, Mapping)
                        else None
                    )
                    if isinstance(data_properties, Mapping):
                        data_keys = [str(key) for key in data_properties]
                result_evidence = result.get("evidence")
                claims = (
                    result_evidence.get("claims")
                    if isinstance(result_evidence, Mapping)
                    else None
                )
                if isinstance(claims, list) and data_keys:
                    for claim in claims:
                        if (
                            not isinstance(claim, Mapping)
                            or claim.get("mode") != "atoms"
                            or claim.get("atoms") != []
                        ):
                            continue
                        field = str(claim.get("field") or "field")
                        if field.startswith("$."):
                            field = field[2:]
                        contract.append(
                            f"Your `{field}` contains no facts. Write one sentence "
                            "that states the values in `data` ("
                            + ", ".join(data_keys)
                            + ") in your own words."
                        )
            suffix = " " + " ".join(contract) if contract else ""
            lines.append(f"- {name}: {reason}; evidence: {bounded or 'none'}.{suffix}")
    if not lines:
        lines.append("- not_attempted: empty content; evidence: none")
    return (
        "[kernel] Your answer was checked by deterministic adapters. Failures:\n"
        + "\n".join(lines)
        + "\nReturn the corrected JSON only. Every value must be copied verbatim from the input; "
          "do not invent fields, units, or numbers."
    )


def _grade_noncoding_attempt(row, content: object, *, tokens: int, latency_ms: int):
    """Fence-strip and grade one raw model response, preserving parse-failure evidence."""
    content = content if isinstance(content, str) else ""
    if not content.strip():
        return grade_artifact(row, None, steps=1, tokens=tokens, latency_ms=latency_ms)
    try:
        # Defensive fence stripping — the 9B wraps
        # JSON in ```json fences often enough that it cost 1 of 3 smoke rows.
        artifact = json.loads(_strip_json_fence(content))
    except json.JSONDecodeError:
        receipt = grade_artifact(row, {}, steps=1, tokens=tokens, latency_ms=latency_ms)
        receipt["reason"] = "unparseable"
        # Keep the raw text (bounded) so an unparseable row can be autopsied.
        receipt["raw_content"] = content[:2000]
        return receipt
    return grade_artifact(row, artifact, steps=1, tokens=tokens, latency_ms=latency_ms)


def _run_noncoding(row):
    """Run bounded report-only JSON extraction and persist its final grading receipt."""
    schema = row["oracle_spec"]["schema_validity"]["schema"]
    prompt = {
        "task": (
            "Return JSON only for this non-coding extraction task.\n\n"
            + _noncoding_contract_text(row)
        ),
        "input": row["input"],
        "schema": schema,
    }
    temperature = float(os.environ.get("REACT_TEMP", str(rr.TEMPERATURE)))
    messages: list[dict[str, object]] = [
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
    ]
    attempt_receipts: list[dict[str, object]] = []
    total_tokens = 0
    total_latency_ms = 0
    empty_retry_used = False
    max_attempts = 1 + _noncoding_loop_retries()
    attempt = 0

    while True:
        attempt += 1
        started = time.perf_counter()
        content, _calls = rr.ask_tools(
            messages,
            mt=4000,
            temp=temperature,
            tools=[],
        )
        content = content if isinstance(content, str) else ""
        latency_ms = round((time.perf_counter() - started) * 1000)
        tokens = int(rr.approx_tokens(content)) if content else 0
        receipt = _grade_noncoding_attempt(
            row, content, tokens=tokens, latency_ms=latency_ms
        )
        failing_adapters = _noncoding_failing_adapters(receipt)
        attempt_receipts.append({
            "behaviour": receipt.get("behaviour"),
            "failing_adapters": failing_adapters,
            "tokens": tokens,
            "latency_ms": latency_ms,
        })
        total_tokens += tokens
        total_latency_ms += latency_ms

        if receipt.get("behaviour") == "PASS_behavioural":
            break
        if attempt >= max_attempts:
            break
        # Empty responses receive at most one follow-up, even if the global loop budget is
        # larger; this prevents a dead endpoint from consuming the whole row budget.
        if empty_retry_used:
            break
        if receipt.get("behaviour") == "not_attempted":
            empty_retry_used = True
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": _noncoding_feedback(receipt)})

    receipt = dict(receipt)
    receipt["attempts"] = len(attempt_receipts)
    receipt["attempt_receipts"] = attempt_receipts
    receipt["tokens"] = total_tokens
    receipt["latency_ms"] = total_latency_ms
    receipt["steps"] = len(attempt_receipts)
    _persist_noncoding_receipt(receipt)
    return receipt


def _dispatch_row(d, oracle_symptom):
    """Keep coding dispatch byte-for-byte in behavior while adding noncoding routing."""
    if d.get("task_kind") == "noncoding":
        return _run_noncoding(d)
    _ids = fail_to_pass_ids(d)
    _sym = str(oracle_symptom or "")
    if _ids:
        _names = ", ".join(i.split("::")[-1] for i in _ids[:6])
        _sym += f"\nThe failing regression tests are named: {_names}"
    return rr.work({
        **d,
        "_oracle_symptom": _sym,
        "_oracle_preflight": {
            "parent_red": True,
            "gold_green": True,
            "oracle_ids": tuple(_ids),
        },
    })


def _multifile_paths(d) -> tuple[str, ...]:
    """Return declared code paths in deterministic order."""
    return tuple(sorted(str(path) for path in d["files"]))


def _multifile_primary_path(d) -> str | None:
    """Choose the explicit entry path, falling back to the first declared file."""
    paths = _multifile_paths(d)
    entry = d.get("entry")
    if isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
        candidate = entry["path"]
        if candidate in paths:
            return candidate
    return paths[0] if paths else None


def _multifile_cache_key(d) -> str:
    """Identify a multi-file oracle by its complete workspace inputs."""
    files = d.get("files") if isinstance(d.get("files"), Mapping) else {}
    paths = _multifile_paths(d)
    parents = {path: files[path].get("parent_src", "") for path in paths
               if isinstance(files[path], Mapping)}
    fixed = {path: files[path].get("fixed_src", "") for path in paths
             if isinstance(files[path], Mapping)}
    oracle = d.get("oracle") if isinstance(d.get("oracle"), Mapping) else {}

    def digest(value) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return ("multi_file@paths=" + ",".join(paths) +
            f"@parent={digest(parents)}@fixed={digest(fixed)}@oracle={digest(oracle)}")


def _write_multifile_side(root: pathlib.Path, d, side: str) -> None:
    """Write every declared source file for one oracle leg."""
    files = d["files"]
    for path in _multifile_paths(d):
        payload = files[path]
        if not isinstance(payload, Mapping) or not isinstance(payload.get(side), str):
            raise ValueError(f"multi-file row has no {side} source for {path}")
        destination = _swebench_path(root, path, f"multi-file {side}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload[side], encoding="utf-8")


def _declared_build_test_path(row) -> str:
    """Return the build oracle path from the shared task contract.

    The function name is not a reliable filename.  The resolver follows the
    legacy declarations and the path exposed in the task text, and never
    recreates a historical ``test_<fn>`` fallback.
    """
    return resolve_build_test_path(row, row.get("fn") if isinstance(row.get("fn"), str) else None)


def _sandbox(d):
    """A faithful, ERA-COHERENT copy. Must match react_repair's sandbox exactly.

    The first version copied TODAY'S repo — so the preflight judged oracles against tests
    from a different era than the bug. A test that
    discriminates in the parent-commit world can look non-discriminating in today's world and
    vice versa, which silently mis-classifies tasks as unscoreable. For MC tasks, materialise
    the fix's parent commit via `git archive`, the same as the loop does.
    """
    sbox = pathlib.Path(tempfile.mkdtemp(prefix="conv_"))
    if _is_multifile_row(d):
        root = None
        src = rr.REPOS.get(d.get("repo")) if d.get("repo") else None
        if d.get("source_dataset") and d.get("base_commit"):
            root = sbox / "repo"
            shutil.copytree(
                _swebench_workspace(d) / "base",
                root,
                ignore=shutil.ignore_patterns("__pycache__", ".git", ".*"),
            )
        elif d.get("repo") == "mc" and d.get("sha") and src is not None:
            (sbox / "repo").mkdir()
            backend_archived = _extract_git_archive(
                src.parent, f"{d['sha']}^", sbox / "repo", ("backend",))
            if backend_archived and (sbox / "repo" / "backend").is_dir():
                root = sbox / "repo" / "backend"
                _extract_git_archive(
                    src.parent, f"{d['sha']}^", sbox / "repo", ("test_support",))
                support = sbox / "repo" / "test_support"
                if support.is_dir() and not (root / "test_support").exists():
                    shutil.copytree(support, root / "test_support")
        elif d.get("repo") == "build" or src is None:
            root = sbox / "repo"
            root.mkdir(parents=True, exist_ok=True)
        if root is None:
            if (sbox / "repo").exists():
                shutil.rmtree(sbox / "repo", ignore_errors=True)
            shutil.copytree(
                src, sbox / "repo",
                ignore=shutil.ignore_patterns("__pycache__", ".git", ".*"),
            )
            root = sbox / "repo"
            support = src.parent / "test_support"
            if support.is_dir() and not (root / "test_support").exists():
                shutil.copytree(support, root / "test_support",
                                ignore=shutil.ignore_patterns("__pycache__", ".*"))
        _write_multifile_side(root, d, "parent_src")
        if d.get("source_dataset") and d.get("base_commit") \
                and d.get("test_path") and d.get("test_src") is not None:
            _write_swebench_runner(root, d)
        primary = _multifile_primary_path(d)
        target = _swebench_path(root, primary, "source") if primary else None
        return sbox, root, target
    if d.get("source_dataset") and d.get("base_commit"):
        root = sbox / "repo"
        shutil.copytree(
            _swebench_workspace(d) / "base",
            root,
            ignore=shutil.ignore_patterns("__pycache__", ".git", ".*"),
        )
        target = _swebench_path(root, d["path"], "source")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(d["parent_src"], encoding="utf-8")
        _write_swebench_runner(root, d)
        return sbox, root, target

    src = rr.REPOS.get(d.get("repo") or "aha")
    # BUILD TASKS (E380): synthetic greenfield world — no git era, the row IS the world.
    # target starts as the stub (parent_src), the spec tests are VISIBLE by design (build-to-
    # spec is TDD, not a leak); the same two-sided preflight law applies (stub FAILS the
    # spec, reference PASSES it) so a broken spec can never mint a free resolution.
    if d.get("repo") == "build":
        root = sbox / "repo"
        target = root / "app" / d["rel"]
        target.parent.mkdir(parents=True, exist_ok=True)
        for pkg in [p for p in (target.parent, root / "app") if p.is_dir()]:
            (pkg / "__init__.py").touch()
        target.write_text(d["parent_src"])
        tdir = root / "tests"
        tdir.mkdir(parents=True, exist_ok=True)
        destination = root / _declared_build_test_path(d)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(d["test_src"])
        return sbox, root, target
    root = None
    if d.get("repo") == "mc" and d.get("sha"):
        mc_root = src.parent
        (sbox / "repo").mkdir()
        # `test_support` was introduced after some registered fix parents. Including it in
        # the same archive command makes git reject the entire archive, then silently drops
        # into a copy of today's dirty checkout. Archive the required backend first and add
        # the helper tree only when that historical path exists.
        backend_archived = _extract_git_archive(
            mc_root,
            f"{d['sha']}^",
            sbox / "repo",
            ("backend",),
        )
        if backend_archived and (sbox / "repo" / "backend").is_dir():
            root = sbox / "repo" / "backend"
            _extract_git_archive(
                mc_root,
                f"{d['sha']}^",
                sbox / "repo",
                ("test_support",),
            )
            ts = sbox / "repo" / "test_support"
            if ts.is_dir() and not (root / "test_support").exists():
                shutil.copytree(ts, root / "test_support")
    if root is None:
        # The archive branch pre-creates sbox/"repo"; if the archive then fails (e.g. a sha
        # not reachable from this checkout) the fallback copytree collided with that empty
        # dir and CRASHED the whole preflight (FileExistsError — took v7 down mid-run).
        # Clear it first; an empty leftover directory must never abort the run.
        if (sbox / "repo").exists():
            shutil.rmtree(sbox / "repo", ignore_errors=True)
        shutil.copytree(src, sbox / "repo",
                        ignore=shutil.ignore_patterns("__pycache__", ".git", ".*"))
        root = sbox / "repo"
        helper = src.parent / "test_support"
        if helper.is_dir() and not (root / "test_support").exists():
            shutil.copytree(helper, root / "test_support",
                            ignore=shutil.ignore_patterns("__pycache__", ".*"))
    target = None
    for cand in (root / "app" / d["rel"], root / d["rel"]):
        if cand.exists():
            target = cand
            break
    return sbox, root, target


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


def verify_oracle(d):
    """Preflight: the test must FAIL on parent and PASS on gold, or the task is unscoreable.

    This is the E256 lesson generalised — verify the RULER before measuring with it. A test
    that passes on the buggy code proves nothing; a test that fails on gold is broken.

    anti-gaming addendum (solve_differentapproach), adopted as doctrine for any future
    grader change: per task, the shipped patch must be ACCEPTED, a no-op must be REJECTED, a
    wrong-function patch must be REJECTED, an assertion-WEAKENING patch must be REJECTED, and
    a broken oracle must yield UNKNOWN — never PASS. The parent/gold two-sided check here
    covers the first two; the weakening control lives with the grader, because the agent
    can edit test files in its sandbox and "make it green by gutting the assert" must never
    count as resolution.
    """
    if _is_multifile_row(d):
        sbox, root, target = _sandbox(d)
        try:
            if target is None:
                return None
            overlay_fix_tests(root, d)
            selectors = fail_to_pass_ids(d)
            if not selectors:
                return None
            _write_multifile_side(root, d, "parent_src")
            row = {**d, "_skip_db": True}
            on_parent, parent_codes = rr.run_tests_for_selectors(
                root, "__multi_file__", selectors, _multifile_primary_path(d), row=row)
            _write_multifile_side(root, d, "fixed_src")
            on_gold, gold_codes = rr.run_tests_for_selectors(
                root, "__multi_file__", selectors, _multifile_primary_path(d), row=row)
            if (parent_codes and all(code == 1 for code in parent_codes)
                    and gold_codes and all(code == 0 for code in gold_codes)):
                lines = [line for line in on_parent.splitlines()
                         if line.startswith(("E ", "assert", "FAILED", "> "))][:14]
                return "\n".join(lines) or "(multi-file oracle fails on the buggy workspace)"
            return None
        except Exception:
            return None
        finally:
            shutil.rmtree(sbox, ignore_errors=True)
    sbox, root, target = _sandbox(d)
    try:
        if target is None:
            return None
        # Grade in the GRADING world: parent-era code + the fix commit's tests, and run ONLY
        # the fail-to-pass node ids (whole-file was too strict — sibling tests in the same
        # file were not green at sha either).
        overlay_fix_tests(root, d)
        ids = fail_to_pass_ids(d)
        selector = ids[0] if ids else None
        target.write_text(d["parent_src"])
        on_parent = rr.run_tests(root, d["fn"], selector, d["rel"], row=d)
        target.write_text(d["fixed_src"])
        on_gold = rr.run_tests(root, d["fn"], selector, d["rel"], row=d)
        fails_on_parent = "pytest exit 1" in on_parent
        passes_on_gold = "pytest exit 0" in on_gold
        if fails_on_parent and passes_on_gold:
            lines = [line for line in on_parent.splitlines()
                     if line.startswith(("E ", "assert", "FAILED", "> "))][:14]
            return "\n".join(lines) or "(test fails on the buggy code)"
        return None
    except Exception:
        return None
    finally:
        shutil.rmtree(sbox, ignore_errors=True)


def main() -> int:
    if not rr.KEY:
        print("DEEPINFRA_API_KEY unset — refusing to report a vacuous zero.")
        return 2
    rr.MAX_STEPS = STEPS
    tasks = [json.loads(line) for line in rr.CORPUS.read_text().splitlines() if line.strip()][:N]

    print(f"CONVERGENCE — {rr.MODEL}, up to {STEPS} steps, success = BEHAVIOURAL\n")
    # E293: killed runs LEAK their tmp sandboxes (rmtree lives in a finally the kill never
    # reaches). Measured: 89 orphans, 16GB, disk at 99% — the second disk-full of the day.
    # Purge sandboxes older than 3h at startup; live lanes touch theirs constantly.
    try:
        import subprocess as _sp2
        _sp2.run("find ${TMPDIR:-/tmp} -maxdepth 1 \\( -name 'react_*' -o -name 'conv_*' \\)"
                 " -mmin +180 -exec rm -rf {} + 2>/dev/null", shell=True, timeout=120)
    except Exception:                                            # noqa: BLE001
        pass
    # E279 SWEEP: drop every leftover react_* test database before starting. One night of
    # uuid-per-invocation names left 118 full MC schemas and filled the Docker volume until
    # postgres could not write its own pid file. Names are now per-(run, task) so a run
    # leaks at most ~21; this sweep makes even that bounded at one run's worth.
    try:
        import subprocess as _sp
        # Concurrent workers may already be using the database.
        # its own react_* DBs — dropping them under it injects phantom infra errors into its
        # agent's world. Sweep only when this process is the only converge alive.
        _others = _sp.run(["pgrep", "-f", "probes/converge.py"], capture_output=True,
                          text=True).stdout.split()
        if len([p for p in _others if p.strip() and int(p) != os.getpid()]) > 0:
            raise RuntimeError("another converge work stream is live; skipping the DB sweep")
        _sp.run(["docker", "exec", "mission-control-db", "bash", "-c",
                 'for db in $(psql -U mission -d postgres -t -A -c '
                 '"select datname from pg_database where datname like '
                 "'mission_control_test_react_%'\"); do "
                 'psql -U mission -d postgres -q -c "DROP DATABASE IF EXISTS \\"$db\\""; '
                 "done"], capture_output=True, timeout=120)
        print("  (swept leftover react_* test databases)")
    except Exception as _e:                                          # noqa: BLE001
        print(f"  (react-db sweep skipped: {_e})")
    print("  preflighting oracles (test must FAIL on parent and PASS on gold)...")
    # PREFLIGHT CACHE. Tonight's harness fixes forced four relaunches, each paying ~30 min
    # of identical two-sided preflights before the graded phase could start — the verdicts
    # never changed because they depend only on (sha, corpus), not on the worker code being
    # iterated. Cache per (fn, sha); PREFLIGHT_CACHE=0 forces a full re-verify, and any task
    # missing from the cache is verified live. The cache holds the verdict plus the parent-leg
    # failure lines verify_oracle already returns; converge never feeds either into a brief
    # (the worker's brief is built independently in react_repair from the commit subject).
    cache_path = POC / "results" / "preflight_cache.json"
    use_cache = os.environ.get("PREFLIGHT_CACHE", "1") == "1"
    try:
        cache = json.loads(cache_path.read_text()) if use_cache else {}
    except Exception:                                            # noqa: BLE001
        cache = {}
    scoreable, excluded = [], []
    swebench_django_seen = 0
    for d in tasks:
        if d.get("source_dataset") and d.get("base_commit"):
            # Keep converge aligned with swebench_env.run_proof: its persisted corpus proof
            # covers setup blockers plus the first three Django rows only.
            _is_swebench_django = d.get("repo") == "django/django"
            if _is_swebench_django:
                swebench_django_seen += 1
            _swebench_blocker = d.get("env_blocker")
            if not _swebench_blocker and not _is_swebench_django:
                _swebench_blocker = (
                    "ENV_VERIFY_NOT_REQUESTED: proof slice is scoped to Django"
                )
            elif not _swebench_blocker and swebench_django_seen > 3:
                _swebench_blocker = (
                    "ENV_VERIFY_NOT_REQUESTED: proof slice limited to first 3 Django rows"
                )
            if _swebench_blocker:
                d = {**d, "env_blocker": str(_swebench_blocker)}
                excluded.append((d, None))
                label = _row_label(d)
                print(f"    {label[:32]:34s} excluded "
                      f"({str(_swebench_blocker).split(':', 1)[0]})", flush=True)
                continue
        if d.get("task_kind") == "noncoding":
            scoreable.append((d, None))
            print(f"    {_row_label(d)[:32]:34s} ORACLE OK (adapter)", flush=True)
            continue
        ck = (_multifile_cache_key(d) if _is_multifile_row(d)
              else f"{d['fn']}@{d.get('sha', '')[:12]}")
        if use_cache and ck in cache:
            rep = cache[ck]["rep"]
            tag = "ORACLE OK (cached)" if rep else "no usable oracle (cached)"
        else:
            rep = verify_oracle(d)
            cache[ck] = {"rep": rep}
            tag = "ORACLE OK" if rep else "no usable oracle"
        (scoreable if rep else excluded).append((d, rep))
        label = (_row_label(d) if d.get("source_dataset") or _is_multifile_row(d)
                 else d["fn"])
        print(f"    {label[:32]:34s} {tag}", flush=True)
    if use_cache:
        cache_path.write_text(json.dumps(cache, indent=1))

    print(f"\n  scoreable: {len(scoreable)}   excluded: {len(excluded)}")
    _excluded = []
    for d, _rep in excluded:
        blocker = d.get("env_blocker")
        reason = (str(blocker).split(":", 1)[0]
                  if blocker else
                  "parent-red/gold-green oracle not proven")
        label = (_row_label(d) if d.get("source_dataset") or _is_multifile_row(d)
                 else d["fn"])
        _excluded.append(f"{label} ({reason})")
    print(f"  excluded (named, never silently counted): {_excluded}\n")

    solved, rows = [], []
    with OUT.open("a") as sink:
        for d, _rep in scoreable:
            try:
                # E277/E309: coding rows retain the existing f2p symptom and oracle-id
                # payload; noncoding rows take the one-shot report-only adapter path.
                row = _dispatch_row(d, _rep)
            except Exception as e:                               # noqa: BLE001
                # E400: a swallowed stack made collectN's terminal FileNotFoundError
                # undiagnosable after 190 turns of paid work — print the traceback
                # (snitch law: an ERR row without its stack is a body without a name).
                import traceback as _tb
                print(f"  [task-error] {_row_label(d)}: {type(e).__name__}: {e}", flush=True)
                _tb.print_exc()
                row = {"fn": _row_label(d), "result": f"ERR:{type(e).__name__}"}
            beh = row.get("behaviour")
            # BEHAVIOURAL success is the objective; AST is diagnostic only.
            won = beh == "PASS_behavioural"
            rows.append({**row, "behavioural_win": won})
            sink.write(json.dumps(rows[-1]) + "\n")
            sink.flush()
            if won:
                solved.append(_row_label(d))
            else:
                # E434: at the cross-run attempt ceiling, emit ONE escalation packet
                # (what was tried / proven / unknown + the decider question) for the
                # decider lane. Converged tasks never emit (planted control).
                try:
                    from kernel_escalation import build_escalation_packet, canonical_json
                    from detreason.attempt_memory import discover_tags, distill_attempt
                    _task_key = _row_label(d)
                    _tags = sorted(discover_tags(str(POC / "results"), _task_key))
                    _ceil = int(os.environ.get("ESCALATION_CEILING", "4"))
                    if len(_tags) >= _ceil:
                        _recs = [distill_attempt(str(POC / "results"), _task_key, tag=_t,
                                                 attempt_no=_i, campaign="")
                                 for _i, _t in enumerate(_tags, 1)]
                        _pkt = build_escalation_packet(
                            _recs, attempt_ceiling=_ceil, task_id=_task_key)
                        if _pkt is not None:
                            _ed = POC / "results" / "escalations"
                            _ed.mkdir(exist_ok=True)
                            (_ed / f"{_task_key}.json").write_text(
                                canonical_json(_pkt.to_dict()))
                            (_ed / f"{_task_key}.txt").write_text(_pkt.render())
                            print(f"    [escalation] packet -> results/escalations/"
                                  f"{_task_key}.json", flush=True)
                except Exception as _ee:                             # noqa: BLE001
                    print(f"    [escalation] packet failed (non-fatal): "
                          f"{type(_ee).__name__}: {_ee}", flush=True)
            print(f"  {_row_label(d)[:32]:34s} ast={row.get('result'):6s} "
                  f"behaviour={str(beh)[:22]:24s} steps={row.get('steps')} "
                  f"edits={row.get('edits')} tests={row.get('tests')}", flush=True)

    print("\n" + "=" * 74)
    print(f"  RESOLVED (behavioural) : {len(solved)}/{len(scoreable)} scoreable tasks")
    ast_only_unverified = sum(
        1
        for r in rows
        if r.get("result") == "PASS"
        and r.get("behaviour") not in {"PASS_behavioural", "tests_fail"}
    )
    failed = len(rows) - len(solved) - ast_only_unverified
    print(f"  AST-ONLY-UNVERIFIED     : {ast_only_unverified}/{len(scoreable)} scoreable tasks")
    print(f"  FAILED                  : {failed}/{len(scoreable)} scoreable tasks")
    print(f"  solved: {solved}")
    print("\n  Convergence check: if 200 steps yields NO additional resolutions over the")
    print("  36-step baseline across two fresh runs, persistence is not the bottleneck and")
    print("  localization/hypothesis generation is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
