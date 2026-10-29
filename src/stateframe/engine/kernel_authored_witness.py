"""Host-owned authored-witness receipts for NEW_BEHAVIOR rows (IDEALPATH-off path).

``kernel_row_role.certification_blocked`` refuses ``done`` for a NEW_BEHAVIOR row
until ``runtime._row_role_authored_receipts`` holds an AUTHORED_CHECK receipt with
``outcome == BEHAVIOR_FAIL``, ``reached_observation`` and ``integrity``.  With
IDEALPATH off nothing else writes that field, so this module is its native writer.

A receipt is recorded only from HOST observations, never from model text, and only
when ALL of these hold for one model-authored test file (a test-discovery path whose
content differs from the parent binding):

(a) the test source references the row's entry symbol (or a declared entrypoint);
(c) the host's own test run of that file on the CURRENT candidate passed (exit 0,
    the file was a target of the run, >=1 test reached, no ``-k`` filter);
(b) the host replays the SAME command and the SAME file against a reconstructed
    PARENT tree (current workspace copy, every changed product file restored from
    the task row's ``parent_src`` and verified hash-equal to the parent binding,
    every added file removed) and it FAILS for a behaviour reason.  An
    ImportError/AttributeError/NameError naming the entry symbol is accepted as
    the witness ``ABSENCE_ON_PARENT`` (NEW_BEHAVIOR rows only); any other
    environment/collection failure is not a witness.

Cost: the parent replay runs once per (test path, content hash); every other test
run pays a dict comparison.  A later failing run of the same authored file on a
new candidate retires its receipt.  ``STATEFRAME_AUTHORED_WITNESS`` (default ``on``).
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import tempfile
import tokenize
from collections.abc import Mapping, Sequence
from pathlib import Path

ENV = "STATEFRAME_AUTHORED_WITNESS"
SCHEMA = "stateframe-authored-witness/1"
_PARENT_TIMEOUT_SECONDS = 300.0
_MAX_COPY_BYTES = 512 * 1024 * 1024
_MAX_SOURCE_BYTES = 1024 * 1024
_COPY_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".test-tmp",
                                      ".mypy_cache", ".ruff_cache")
_ABSENCE_ERROR = re.compile(
    r"(?:cannot import name ['\"](?P<a>[\w.]+)['\"]"
    r"|has no attribute ['\"](?P<b>[\w.]+)['\"]"
    r"|name ['\"](?P<c>[\w.]+)['\"] is not defined)")


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    value = str(values.get(ENV, "on") or "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value == "on"


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def entry_names(runtime: object) -> tuple[str, ...]:
    """Entry symbol + any public entrypoint the row names (leaf identifiers)."""
    names: list[str] = []
    try:
        from kernel_request_source_window import entry_symbol
        names.append(entry_symbol(runtime))
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    entry = getattr(runtime, "_site_facts_entry", None)
    if isinstance(entry, Mapping):
        names.extend(str(entry.get(key) or "") for key in ("symbol", "name", "fn"))
    row = getattr(getattr(runtime, "owner", None), "task_row", {})
    if isinstance(row, Mapping):
        for key in ("entrypoints", "public_entrypoints"):
            value = row.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                names.extend(str(item) for item in value)
    result: list[str] = []
    for name in names:
        leaf = str(name or "").strip().rsplit(".", 1)[-1]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", leaf) and leaf not in result:
            result.append(leaf)
    return tuple(result)


def references_entry(source: str, names: Sequence[str]) -> bool:
    wanted = {str(name).strip() for name in names if str(name).strip()}
    if not wanted:
        return False
    # Python's tokenizer gives us a reliable comment/string boundary.  A
    # comment saying ``new_entry`` is not an executable witness reference.
    try:
        tokens = {
            token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type == tokenize.NAME
        }
        if wanted & tokens:
            return True
        if "def " in source or "import " in source or "from " in source:
            return False
    except (IndentationError, tokenize.TokenError, UnicodeDecodeError):
        pass
    # Opaque non-Python adapters use the same conservative lexical boundary;
    # remove comments and quoted literals before looking for an identifier.
    code: list[str] = []
    index = 0
    quote = ""
    while index < len(source):
        char = source[index]
        pair = source[index:index + 2]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if pair in {"//", "/*"} or char == "#":
            if pair == "/*":
                end = source.find("*/", index + 2)
                index = len(source) if end < 0 else end + 2
            else:
                end = source.find("\n", index + 1)
                index = len(source) if end < 0 else end
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        code.append(char)
        index += 1
    identifiers = set(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", "".join(code)))
    return bool(wanted & identifiers)


def _is_test_path(relative: str) -> bool:
    from kernel_request_source_window import _TEST_PATH
    return bool(_TEST_PATH.search(relative)) and not relative.endswith("conftest.py")


def _files(binding: object, key: str) -> dict[str, str]:
    value = binding.get(key) if isinstance(binding, Mapping) else None
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, Mapping) else {}


def authored_test_files(parent: object, current: object) -> dict[str, str]:
    """Test-discovery paths whose content differs from the parent binding."""
    parent_files = {**_files(parent, "product_files"), **_files(parent, "harness_files")}
    current_files = {**_files(current, "product_files"), **_files(current, "harness_files")}
    return {path: sha for path, sha in current_files.items()
            if parent_files.get(path) != sha and _is_test_path(path)}


def _parent_sources(runtime: object) -> dict[str, str]:
    row = getattr(getattr(runtime, "owner", None), "task_row", {})
    files = row.get("files") if isinstance(row, Mapping) else None
    result: dict[str, str] = {}
    if isinstance(files, Mapping):
        for path, payload in files.items():
            if isinstance(payload, Mapping) and isinstance(payload.get("parent_src"), str):
                result[str(path)] = payload["parent_src"]
    return result


def _tree_bytes(root: Path) -> int:
    total = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache", ".test-tmp"}]
        for name in files:
            try:
                total += (Path(current) / name).lstat().st_size
            except OSError:
                continue
            if total > _MAX_COPY_BYTES:
                return total
    return total


def build_parent_tree(root: Path, destination: Path, *, parent: Mapping[str, object],
                      current: Mapping[str, object], parent_sources: Mapping[str, str],
                      keep: str) -> str | None:
    """Reconstruct the parent workspace plus ``keep``; None on success, else a reason."""
    if _tree_bytes(root) > _MAX_COPY_BYTES:
        return "WORKSPACE_TOO_LARGE"
    shutil.copytree(root, destination, symlinks=True, ignore=_COPY_IGNORE)
    parent_files = {**_files(parent, "product_files"), **_files(parent, "harness_files")}
    current_files = {**_files(current, "product_files"), **_files(current, "harness_files")}
    for path in sorted(set(parent_files) | set(current_files)):
        if path == keep or parent_files.get(path) == current_files.get(path):
            continue
        target = destination / path
        if path not in parent_files:
            target.unlink(missing_ok=True)
            continue
        source = parent_sources.get(path)
        if source is None or _sha256(source.encode("utf-8")) != parent_files[path]:
            if _is_test_path(path) and not path.endswith("conftest.py"):
                target.unlink(missing_ok=True)   # another edited test: not a target here
                continue
            return f"PARENT_UNRECONSTRUCTABLE:{path}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.encode("utf-8"))
    # Clean-copy proof: the reconstructed product tree hashes exactly to the parent's.
    from kernel_live_projection import workspace_binding
    rebuilt = workspace_binding(destination, "parent-replay")
    rebuilt_product = {k: v for k, v in _files(rebuilt, "product_files").items() if k != keep}
    parent_product = {k: v for k, v in _files(parent, "product_files").items() if k != keep}
    if rebuilt_product != parent_product:
        return "PARENT_HASH_MISMATCH"
    return None


def _retarget(value: str, root: str, replacement: str) -> str:
    return value.replace(root, replacement) if root and root in value else value


def replay_on_parent(runtime: object, *, root: Path, command: Sequence[object],
                     environment: Mapping[str, str] | None, test_path: str,
                     parent: Mapping[str, object], current: Mapping[str, object]
                     ) -> dict[str, object]:
    """Run the host's own test command for ``test_path`` against the parent tree."""
    from kernel_execution_policy import run_sandboxed
    from kernel_reprospec import _run_result
    if not command:
        return {"outcome": "ENVIRONMENT", "reason": "NO_HOST_COMMAND"}
    resolved_root = str(root.resolve())
    with tempfile.TemporaryDirectory(prefix="authored-witness-") as scratch:
        destination = Path(scratch).resolve() / "parent"
        try:
            reason = build_parent_tree(root.resolve(), destination, parent=parent,
                                       current=current, parent_sources=_parent_sources(runtime),
                                       keep=test_path)
        except (OSError, shutil.Error, ValueError) as error:
            reason = f"PARENT_COPY_FAILED:{type(error).__name__}"
        if reason is not None:
            return {"outcome": "ENVIRONMENT", "reason": reason}
        replacement = str(destination)
        argv = [_retarget(str(item), resolved_root, replacement) for item in command]
        argv.append(test_path)   # cwd is the parent tree; relative works for every runner
        env = ({str(k): _retarget(str(v), resolved_root, replacement)
                for k, v in environment.items()} if environment is not None else None)
        try:
            result = run_sandboxed(argv, destination, env=env, timeout=_PARENT_TIMEOUT_SECONDS,
                                   output_limit=256 * 1024)
            completed = result.completed
            output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
            if getattr(result, "timed_out", False):
                return {"outcome": "RESOURCE_LIMIT", "reason": "PARENT_TIMEOUT"}
            exit_code = completed.returncode
        except (OSError, RuntimeError, ValueError) as error:
            return {"outcome": "ENVIRONMENT", "reason": f"PARENT_RUN_FAILED:{type(error).__name__}"}
    output = output.replace(replacement, "<parent>")
    outcome, exit_value, _text, tests, phase = _run_result(
        {"observation": output, "exit_code": exit_code})
    if exit_code not in (None, 0) and outcome == "INVALID_CHECK" and re.search(
            r"\b(?:\d+ failed|FAILED|AssertionError)\b", output):
        outcome, phase = "BEHAVIOR_FAIL", "ASSERTION"   # pytest summary has no "Ran N"
    return {"outcome": outcome, "exit_code": exit_value, "tests": tests,
            "failure_phase": phase, "output_tail": output[-2000:]}


_MISSING_MODULE = re.compile(r"No module named ['\"](?P<m>[\w.]+)['\"]")


def added_modules(parent: object, current: object) -> frozenset[str]:
    """Dotted names of product modules the candidate added (absent on the parent)."""
    parent_files = _files(parent, "product_files")
    result: set[str] = set()
    for path in _files(current, "product_files"):
        if path in parent_files or not path.endswith(".py"):
            continue
        dotted = path[:-3].replace("/", ".")
        result.add(dotted[: -len(".__init__")] if dotted.endswith(".__init__") else dotted)
    return frozenset(result)


def classify_parent(replay: Mapping[str, object], names: Sequence[str],
                    new_modules: frozenset[str] = frozenset()) -> str | None:
    """Witness kind for a parent replay, or None when it is not a behaviour failure.

    Absence counts only when the error names the entry symbol, or a module the
    candidate itself added; any other import/collection error is environment."""
    outcome = replay.get("outcome")
    if replay.get("exit_code") == 0 or outcome == "PASS":
        return None
    text = str(replay.get("output_tail") or "")
    for match in _ABSENCE_ERROR.finditer(text):
        symbol = next(group for group in match.groups() if group)
        if symbol.rsplit(".", 1)[-1] in names:
            from kernel_row_role import ABSENCE_ON_PARENT
            return ABSENCE_ON_PARENT
    for match in _MISSING_MODULE.finditer(text):
        module = match.group("m")
        if any(module == item or item.endswith("." + module) for item in new_modules):
            from kernel_row_role import ABSENCE_ON_PARENT
            return ABSENCE_ON_PARENT
    if outcome == "BEHAVIOR_FAIL":
        return "BEHAVIOR_FAIL"
    return None


def _clause_id(runtime: object, names: Sequence[str]) -> str:
    """The row the witness certifies: the task row id, else its entry symbol."""
    row = getattr(getattr(runtime, "owner", None), "task_row", {})
    if isinstance(row, Mapping):
        for key in ("requirement_id", "row_id", "task_id", "id"):
            if row.get(key):
                return str(row[key])
    return "entry:" + (names[0] if names else "unknown")


def _store(runtime: object) -> dict[str, dict[str, object]]:
    store = getattr(runtime, "_row_role_authored_receipts", None)
    if not isinstance(store, dict):
        store = {}
        runtime._row_role_authored_receipts = store
    return store


def _emit(runtime: object, event: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        try:
            emit(event, producer="kernel_authored_witness", dispatch=False, **fields)
        except (TypeError, ValueError, KeyError, OSError):
            pass


def note_test_run(runtime: object, *, selectors: Sequence[object], exit_code: int,
                  binding: Mapping[str, object], observation: object,
                  command: Sequence[object] = (),
                  environment: Mapping[str, str] | None = None) -> dict[str, object] | None:
    """Fold one host test run; record a receipt when (a)+(b)+(c) all hold."""
    if not enabled():
        return None
    from kernel_row_role import NEW_BEHAVIOR, enabled as row_role_enabled, runtime_role
    if not row_role_enabled():
        return None
    parent = getattr(runtime, "_row_role_parent_binding", None)
    if not isinstance(parent, Mapping) or not _files(parent, "product_files"):
        return None
    authored = authored_test_files(parent, binding)
    if not authored:
        return None
    root = Path(getattr(getattr(runtime, "owner", None), "root", ""))
    from kernel_public_evidence import public_check_files
    targeted = set(public_check_files(root, [str(item) for item in selectors]))
    filtered = any("::filtered[" in str(item) for item in selectors)
    store = _store(runtime)
    candidate_revision = str(binding.get("product_revision") or "")
    binding_stable = binding.get("binding_stable") is True
    host_execution_id = str(
        binding.get("host_execution_id") or binding.get("execution_id") or ""
    ).strip()
    if exit_code != 0:
        # A failing run of an authored file retires its receipts (stale candidate).
        for receipt_id in [rid for rid, rec in store.items()
                           if rec.get("test_path") in targeted]:
            store.pop(receipt_id, None)
            _emit(runtime, "authored_witness_retired", receipt_id=receipt_id)
        return None
    if filtered:
        return None
    tests_seen = re.search(r"\b(\d+) passed\b|Ran\s+([1-9]\d*)\s+tests?", str(observation or ""))
    if tests_seen is None or tests_seen.group(0).startswith("0 "):
        return None
    attempted = getattr(runtime, "_authored_witness_attempts", None)
    if not isinstance(attempted, dict):
        attempted = {}
        runtime._authored_witness_attempts = attempted
    pending = [(path, sha) for path, sha in sorted(authored.items())
               if path in targeted and (path, sha) not in attempted]
    if not pending:
        return None
    if runtime_role(runtime).get("role") != NEW_BEHAVIOR:
        return None
    names = entry_names(runtime)
    recorded: dict[str, object] | None = None
    for path, sha in pending:
        verdict: dict[str, object] = {"test_path": path, "content_sha256": sha}
        attempted[(path, sha)] = verdict
        try:
            data = (root / path).read_bytes()
        except OSError:
            verdict["reason"] = "UNREADABLE"
            continue
        if len(data) > _MAX_SOURCE_BYTES or _sha256(data) != sha:
            verdict["reason"] = "CONTENT_CHANGED_SINCE_RUN"
            attempted.pop((path, sha), None)
            continue
        if not names or not references_entry(data.decode("utf-8", "replace"), names):
            verdict["reason"] = "NO_ENTRY_REFERENCE"
            _emit(runtime, "authored_witness_rejected", verdict=dict(verdict))
            continue
        replay = replay_on_parent(runtime, root=root, command=command, environment=environment,
                                  test_path=path, parent=parent, current=binding)
        witness_kind = classify_parent(replay, names, added_modules(parent, binding))
        verdict["parent"] = {key: replay.get(key) for key in (
            "outcome", "exit_code", "tests", "failure_phase", "reason")}
        if witness_kind is None:
            verdict["reason"] = ("PARENT_PASSED" if replay.get("outcome") == "PASS"
                                 else str(replay.get("reason") or "PARENT_NOT_BEHAVIOR_FAIL"))
            if str(replay.get("outcome", "")).upper() in {
                    "ENVIRONMENT", "RESOURCE_LIMIT", "UNRUNNABLE", "TIMEOUT"}:
                # Setup/runner incapacity is retryable after the host changes;
                # it must not consume the only authored-check route forever.
                attempted.pop((path, sha), None)
                verdict["reason"] = "UNRUNNABLE"
            _emit(runtime, "authored_witness_rejected", verdict=dict(verdict))
            continue
        if not binding_stable:
            verdict["reason"] = "UNSTABLE_BINDING"
            _emit(runtime, "authored_witness_rejected", verdict=dict(verdict))
            continue
        receipt_id = "authored:" + hashlib.sha256(
            f"{path}\0{sha}\0{candidate_revision}".encode("utf-8")).hexdigest()[:24]
        parent_result = {**verdict["parent"], "raw_outcome": verdict["parent"].get("outcome"),
                         "outcome": "BEHAVIOR_FAIL", "witness_kind": witness_kind,
                         "revision": str(parent.get("product_revision") or ""),
                         "output_tail": str(replay.get("output_tail") or "")[-600:]}
        candidate_result = {"outcome": "PASS", "exit_code": exit_code,
                            "revision": candidate_revision,
                            "observation": str(observation or "")[-2000:],
                            "binding_stable": binding_stable,
                            "host_execution_id": host_execution_id or None}
        receipt = {
            "schema": SCHEMA, "receipt_id": receipt_id,
            "provenance": {"kind": "AUTHORED_CHECK", "source": "host_parent_replay"},
            # Identity fields kernel_row_role._authored_witness pairs on.
            "clause_id": _clause_id(runtime, names),
            "check_identity": f"{path}@{sha}",
            "oracle_hash": sha,
            "test_path": path, "content_sha256": sha,
            "outcome": "BEHAVIOR_FAIL", "witness_kind": witness_kind,
            "reached_observation": True, "integrity": True,
            "binding_stable": binding_stable,
            "host_execution_id": host_execution_id or None,
            "execution_id": host_execution_id or None,
            "revision": candidate_revision,
            "entry_names": list(names),
            "parent_result": parent_result,
            "candidate_result": candidate_result,
        }
        store[receipt_id] = receipt
        verdict["receipt_id"] = receipt_id
        _emit(runtime, "authored_witness_recorded", receipt={
            key: value for key, value in receipt.items() if key != "parent_result"},
            parent_outcome=verdict["parent"])
        recorded = receipt
    return recorded


__all__ = ["ENV", "SCHEMA", "authored_test_files", "build_parent_tree", "classify_parent",
           "added_modules", "enabled", "entry_names", "note_test_run", "references_entry", "replay_on_parent"]
