"""Guard request bytes and identities against run-to-run entropy leaks."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import re
import secrets
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "discount_bug": "Fix the pricing code so the tests pass.",
    "tiny_site": "Build the site described in site/README.md so the tests pass.",
    "brief_to_report": (
        "Read the brief and the support-ticket CSV in input/ and write report/summary.json "
        "(counts by category) and report/report.md (a short written report with the same "
        "numbers). Make tests/test_report.py pass."
    ),
}

# Scan every shipped package module, including opt-in request-path modules. This
# deliberately exceeds the default runner path so optional model-visible IDs are
# reviewed too. Probe files are absent from the release manifest. New source
# modules join automatically; new entropy calls need an allowlist reason.
REQUEST_PATH_MODULES = (
    *(path.relative_to(ROOT).as_posix() for path in sorted(
        (ROOT / "src/stateframe").rglob("*.py"))
      if "probes" not in path.relative_to(ROOT / "src/stateframe").parts),
)
RANDOM_CALL_ALLOWLIST = {
    "src/stateframe/runner.py": {
        "secrets.token_hex": (
            1, "Creates the persisted run seed; same-seed runs and resume reuse it."
        ),
    },
    "src/stateframe/_child.py": {
        "secrets.token_hex": (
            1, "Names a private temporary checkpoint-key file; the name is never sent to the model."
        ),
        "secrets.token_bytes": (
            1, "Creates the private checkpoint signing key, not a request or receipt identity."
        ),
    },
    "src/stateframe/engine/kernel_bolt.py": {
        "uuid.uuid4": (
            1, "Creates a host run label; applied-edit wire rows use the stable journal projection."
        ),
    },
    "src/stateframe/engine/kernel_continuation_journal.py": {
        "uuid.uuid4": (
            1, "Names the private journal directory; public run IDs are aliased before rendering."
        ),
    },
    "src/stateframe/engine/kernel_continuation_runtime.py": {
        "uuid.uuid4": (
            1, "Creates a private scheduler wake token that is not included in model continuation context."
        ),
    },
    "src/stateframe/engine/kernel_continuation_workspace.py": {
        "uuid.uuid4": (
            2, "Creates host workspace and grading paths; public workspace paths are normalized."
        ),
    },
    "src/stateframe/engine/kernel_exec_session.py": {
        "uuid.uuid4": (
            1, "Names a private temporary execution directory, not a request identity."
        ),
    },
    "src/stateframe/engine/kernel_file_ops.py": {
        "secrets.token_hex": (
            1, "Names a private intermediate move path that is not returned to the model."
        ),
    },
    "src/stateframe/engine/kernel_history_spool.py": {
        "os.urandom": (
            1, "Creates private spool lineage and deliberately stays outside the wire-ID stream."
        ),
    },
    "src/stateframe/engine/kernel_measurement.py": {
        "uuid.uuid4": (
            2, "Creates host profiling/runtime IDs; provider request IDs use content_wire_id."
        ),
    },
    "src/stateframe/engine/kernel_request_auth_identity.py": {
        "secrets.token_bytes": (
            2, "Creates installation HMAC secrets (identity.key and fallback); only the HMAC enters tokenizer_identity metadata, never request bytes."
        ),
        "secrets.token_hex": (
            1, "Names the atomic identity.key publication temp file; never stored or sent."
        ),
    },
    "src/stateframe/engine/kernel_stateframe_resume.py": {
        "uuid.uuid4": (
            1, "Creates journal correlation metadata; the resumed model message uses user text only."
        ),
    },
    "src/stateframe/engine/kernel_stateframe_call_coverage.py": {
        "uuid.uuid4": (
            1, "Fallback for synthetic callers without a journal; live coverage IDs bind to the journal."
        ),
    },
    "src/stateframe/engine/kernel_test_run_spans.py": {
        "uuid.uuid4": (
            1, "Fallback for standalone spans without a journal; live spans use content_wire_id."
        ),
    },
    "src/stateframe/engine/kernel_bind_quantity.py": {
        "uuid.uuid4": (
            1, "Correlates an auxiliary provider call in journal metadata; it is not in that request body."
        ),
    },
    "src/stateframe/engine/kernel_continuation_retention.py": {
        "uuid.uuid4": (
            1, "Names a host-side retention batch; the seeded run UUID stream makes replay stable."
        ),
    },
    "src/stateframe/engine/kernel_decide2_ladder.py": {
        "random.Random": (
            1, "Uses the explicit vote seed to make option ordering reproducible."
        ),
        "shuffle": (
            1, "Shuffles only the seeded local option list; no entropy enters an ID."
        ),
    },
    "src/stateframe/engine/kernel_decision_api.py": {
        "random.Random": (
            1, "Uses a SHA-256-derived seed for deterministic candidate ordering."
        ),
        "shuffle": (
            1, "Shuffles only the locally seeded candidate list."
        ),
    },
    "src/stateframe/engine/kernel_execution_policy.py": {
        "uuid.uuid4": (
            1, "Names a private cgroup; the host sandbox does not serialize this path or ID."
        ),
    },
    "src/stateframe/engine/kernel_instrument_validation.py": {
        "secrets.token_urlsafe": (
            1, "Creates an unguessable validation nonce for worker sink verification, not a request ID."
        ),
    },
    "src/stateframe/engine/kernel_pilot.py": {
        "uuid.uuid4": (
            2, "Creates parent/permit accounting IDs; run-seeded UUIDs keep same-seed records stable."
        ),
    },
    "src/stateframe/engine/kernel_plain_settlement.py": {
        "uuid.uuid4": (
            2, "Creates terminal settlement journal IDs after the model request loop."
        ),
    },
    "src/stateframe/engine/kernel_repro_first_runtime.py": {
        "uuid.uuid4": (
            1, "Names a bounded host scratch file; the run-seeded UUID stream stabilizes replay."
        ),
    },
    "src/stateframe/engine/kernel_reprospec.py": {
        "uuid.uuid4": (
            2, "Names bounded host scratch files; the run-seeded UUID stream stabilizes replay."
        ),
    },
    "src/stateframe/engine/kernel_stateframe_fork_a.py": {
        "uuid.uuid4": (
            1, "Names a private worker attempt workspace; public worker request IDs use seeded content."
        ),
    },
    "src/stateframe/engine/kernel_stateframe_model_router.py": {
        "uuid.uuid4": (
            2, "Creates routing lease IDs for journal/control state; they are not provider request IDs."
        ),
    },
    "src/stateframe/engine/kernel_stateframe_portfolio.py": {
        "uuid.uuid4": (
            4, "Names standalone journal fallback and private staging/grading paths, not provider request IDs."
        ),
    },
    "src/stateframe/hook_host.py": {
        "secrets.token_hex": (
            1, "Names a private temporary trust-store file during atomic host configuration writes."
        ),
    },
    "src/stateframe/workcopy.py": {
        "secrets.token_hex": (
            1, "Names a private staging file while applying a completed run to its destination."
        ),
    },
    "src/stateframe/engine/capability_release_runner.py": {
        "uuid.uuid4": (
            9, "Creates release-harness temporary names and follow-up journal correlation, outside base request bodies."
        ),
    },
    "src/stateframe/engine/detreason/decision_event.py": {
        "uuid.uuid4": (
            1, "Creates decision-event identity from the run-seeded UUID stream."
        ),
    },
    "src/stateframe/engine/detreason/skeptic.py": {
        "random.Random": (
            1, "Uses the caller-provided seed for reproducible stratified selection."
        ),
        "rng.sample": (
            1, "Samples only from the explicitly seeded local generator."
        ),
    },
    "src/stateframe/engine/detreason/snapshot.py": {
        "_random.seed": (
            1, "Seeds the local random snapshot operation from its stable snapshot input."
        ),
    },
    "src/stateframe/engine/kernel_stable_tool_head.py": {
        "registry.seed": (
            1, "Seeds the tool menu order with the run's offerable tools; no randomness is involved."
        ),
    },
    "src/stateframe/storage_gc.py": {
        "uuid.uuid4": (1, "Names the temporary file for an atomic storage-GC state write; never sent."),
    },
    "src/stateframe/engine/kernel_journal_store.py": {
        "uuid.uuid4": (1, "Names the temporary file while a finished journal is sealed to .xz; never sent."),
    },
    "src/stateframe/rehydrate.py": {
        "uuid.uuid4": (1, "Names the staging directory while a run environment is rehydrated; never sent."),
    },
    "src/stateframe/request_records.py": {
        "uuid.uuid4": (1, "Names the temporary file while request records are packed; never sent."),
    },
    "src/stateframe/powerup_state.py": {
        "secrets.token_hex": (
            1, "Names the temporary file for an atomic powerup-state write; it is renamed into place and never sent."
        ),
    },
    "src/stateframe/powerup_management.py": {
        "secrets.token_hex": (
            1, "Fallback suffix for the backup directory of a replaced powerup when its tree cannot be hashed; host-only."
        ),
    },
    "src/stateframe/engine/prompt_ledger.py": {
        "uuid.uuid4": (
            2, "Adds entropy to offline prompt-log row IDs only; those IDs do not alter the captured request body."
        ),
    },
    "src/stateframe/engine/sys1/prompt.py": {
        "random.Random": (
            1, "Uses a SHA-256-derived seed for deterministic prompt candidate ordering."
        ),
        "shuffle": (
            1, "Shuffles only the locally seeded prompt candidate list."
        ),
    },
    "src/stateframe/engine/task_runners.py": {
        "uuid.uuid4": (
            1, "Names an isolated test database when configured; it is not a model request identity."
        ),
    },
    "src/stateframe/engine/witness_runner.py": {
        "uuid.uuid4": (
            1, "Names a host-side witness execution; the run-seeded UUID stream stabilizes same-seed runs."
        ),
    },
}
RANDOM_API_NAMES = {
    "betavariate", "choice", "choices", "expovariate", "gammavariate",
    "gauss", "getrandbits", "lognormvariate", "normalvariate",
    "paretovariate", "randbytes", "randint", "random", "randrange", "sample",
    "seed", "shuffle", "SystemRandom", "token_bytes", "token_hex",
    "token_urlsafe", "triangular", "uniform", "urandom", "uuid4", "vonmisesvariate",
    "weibullvariate",
}
WIRE_ID_FIELDS = frozenset({
    "request_id", "receipt_id", "edit_receipt", "check_receipt_id",
    "execution_id", "check_execution_ref", "root_budget_id", "handle",
    "worker_id", "branch_id", "journal_id",
    "effect_id", "continuation_effect_id", "unsettled_effect_ids",
    "supporting_evidence", "open_requirement_ids", "evidence_ids",
    "source_event_id", "journal_event_id", "check_receipt_ids", "execution_ids",
    "intention_id", "continuation_id", "child_id",
})
WIRE_ID_FIELDS_BY_SCOPE = {
    ("src/stateframe/engine/kernel_continuation_runtime.py", "LiveContinuation.context"): {
        "id", "event_id", "receipt_ref", "return_receipts", "run_id", "workspace_id", "grant_id",
    },
    ("src/stateframe/engine/kernel_continuation_runtime.py", "LiveContinuation.returned_source"): {
        "receipt_ref",
    },
    ("src/stateframe/engine/kernel_continuation_runtime.py", "LiveContinuation.submission_result"): {
        "event_id",
    },
    ("src/stateframe/engine/kernel_continuation_runtime.py", "LiveContinuation._accept_done_keyed"): {
        "event_id",
    },
}
WIRE_ID_SOURCE_ALLOWLIST = {
    ("src/stateframe/engine/kernel_test_run_spans.py", "TestRunProfile.__init__",
     "execution_id", "uuid.uuid4"): (
        1, "Only standalone profiles without an audit journal use this fallback; live request IDs use content_wire_id."
    ),
    ("src/stateframe/engine/kernel_completion_authority.py", "RootFuelBudget.__init__",
     "root_budget_id", "time.time_ns"): (
        1, "Only standalone budgets use this time-based fallback; live request frames pass the seeded budget ID."
    ),
    ("src/stateframe/engine/capability_release_runner.py", "_resume_descriptor",
     "request_id", "uuid.uuid4"): (
        1, "Release-harness resume correlation only; resumed provider messages append follow-up text without this ID."
    ),
    ("src/stateframe/engine/detreason/juggler.py", "Juggler.submit_human_message",
     "source_event_id", "event_id"): (
        1, "Mission event-bus source linkage; this bus is not serialized by the StateFrame provider renderer."
    ),
    ("src/stateframe/engine/kernel_bind_quantity.py", "run_operator",
     "request_id", "uuid.uuid4"): (
        7, "Auxiliary operator journal correlation; its provider body contains no request ID and capture IDs are stable."
    ),
    ("src/stateframe/engine/kernel_check_truth_adapter.py", "handoff_automatic_result",
     "execution_id", "sequence"): (
        2, "Sequence is only forwarded as receipt cursor metadata; canonical execution IDs use run seed, step and selector content."
    ),
    ("src/stateframe/engine/kernel_continuation_journal.py", "ContinuationJournal._index_framefit_receipt",
     "source_event_id", "source_event_id"): (
        1, "Private journal-origin witness for replay validation; typed provider projections omit this source identity."
    ),
    ("src/stateframe/engine/kernel_control_plane_receipts.py", "HostReceipt.create",
     "receipt_id", "sequence"): (
        2, "Standalone control-plane store IDs bind its local audit sequence and never enter StateFrame provider requests."
    ),
    ("src/stateframe/engine/kernel_measurement.py", "HostSpanRecorder.measure",
     "handle", "sequence"): (
        1, "Profiler span handle is local tracing metadata and is not serialized into request messages."
    ),
    ("src/stateframe/engine/kernel_stateframe_portfolio.py", "PortfolioJournal.__init__",
     "journal_id", "uuid.uuid4"): (
        1, "Standalone portfolio journal fallback; branch callback IDs are explicitly seeded from run, step and content."
    ),
    ("src/stateframe/engine/kernel_stateframe_portfolio.py", "PortfolioJournal.emit",
     "journal_id", "sequence"): (
        1, "Journal record metadata only; branch callback request IDs use seeded content identities."
    ),
    ("src/stateframe/engine/kernel_stateframe_portfolio.py", "PortfolioJournal.emit",
     "journal_id", "time.time_ns"): (
        1, "Timestamp is audit metadata only; no provider request field is built from it."
    ),
    ("src/stateframe/engine/kernel_stateframe_resume.py", "user_request_event",
     "request_id", "uuid.uuid4"): (
        2, "Follow-up journal correlation only; the restored provider conversation appends user text without this ID."
    ),
    ("src/stateframe/engine/kernel_timeline.py", "TimelineEvent.from_mapping",
     "source_event_id", "source_event_id"): (
        1, "Source identity is retained for timeline deduplication; provider frame text renders seeded content references."
    ),
    ("src/stateframe/engine/kernel_timeline.py", "TimelineIndex._ingest_journal_record",
     "source_event_id", "event_id"): (
        2, "Journal source IDs key the optional host timeline index; frame_slice renders content-derived references instead."
    ),
    ("src/stateframe/engine/kernel_timeline.py", "TimelineIndex.append",
     "source_event_id", "event_id"): (
        1, "Append return metadata is for optional timeline indexing; its provider frame renderer uses stable references."
    ),
    ("src/stateframe/engine/kernel_tokenlab_stall_breaker.py", "_closed_action",
     "receipt_id", "event_id"): (
        2, "Fallback key deduplicates internal stall observations; the ID is not included in the rendered stall signal."
    ),
    ("src/stateframe/engine/kernel_verdict.py", "_event_rows",
     "receipt_id", "sequence"): (
        1, "Journal sequence orders candidate rows; public evidence references are re-derived from content and local duplicate ordinal."
    ),
}
UNSTABLE_ID_INPUT_NAMES = frozenset({
    "sequence", "_sequence", "event_id", "event_sequence", "_next_event_number",
})
UNSTABLE_ID_CALLS = frozenset({
    "uuid.uuid4", "uuid.uuid1", "uuid.uuid7", "uuid4", "token_hex",
    "secrets.token_hex", "urandom", "os.urandom", "random", "random.random",
    "random.randrange", "random.randint", "randrange", "randint", "randbytes",
    "getrandbits", "time", "time.time", "time.time_ns", "time.monotonic",
    "time.monotonic_ns", "time.perf_counter", "time.perf_counter_ns",
    "datetime.now", "datetime.utcnow", "next_wire_id", "_new_id",
})


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_random_call(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return (leaf in RANDOM_API_NAMES or leaf == "Random"
            or (name.startswith("random.") and leaf not in {"getstate", "setstate"}))


def _scan_request_path_random_calls() -> Counter[tuple[str, str]]:
    calls: Counter[tuple[str, str]] = Counter()
    for relative in REQUEST_PATH_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _dotted_name(node.func)
                if _is_random_call(name):
                    calls[(relative, name)] += 1
    return calls


def test_request_path_randomness_is_reviewed():
    expected: Counter[tuple[str, str]] = Counter()
    for path, callsites in RANDOM_CALL_ALLOWLIST.items():
        assert path in REQUEST_PATH_MODULES
        for call, (count, reason) in callsites.items():
            assert count > 0 and reason.strip()
            expected[(path, call)] = count

    scanned = _scan_request_path_random_calls()
    violations = {
        key: {"expected": expected.get(key, 0), "scanned": scanned.get(key, 0)}
        for key in set(expected) | set(scanned)
        if expected.get(key, 0) != scanned.get(key, 0)
    }
    expected_count = sum(expected.values())
    scanned_count = sum(scanned.values())
    print(
        f"request-path id scan: expected={expected_count} "
        f"scanned={scanned_count} violated={sum(abs(v['expected'] - v['scanned']) for v in violations.values())}"
    )
    assert not violations, f"unreviewed or changed request-path random calls: {violations}"


def test_request_path_randomness_rejects_a_new_unreviewed_secrets_call(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "src/stateframe", tmp_path / "src/stateframe",
                    ignore=shutil.ignore_patterns("__pycache__"))
    planted_relative = "src/stateframe/engine/kernel_unreviewed_randomness.py"
    (tmp_path / planted_relative).write_text(
        "import secrets\nsecrets.token_bytes(8)\n", encoding="utf-8"
    )

    original_modules = REQUEST_PATH_MODULES
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    monkeypatch.setitem(globals(), "REQUEST_PATH_MODULES", (*original_modules, planted_relative))
    with pytest.raises(AssertionError, match="unreviewed or changed") as unreviewed:
        test_request_path_randomness_is_reviewed()
    assert planted_relative in str(unreviewed.value)
    assert "secrets.token_bytes" in str(unreviewed.value)
    assert "'expected': 0, 'scanned': 1" in str(unreviewed.value)

    identity_relative = "src/stateframe/engine/kernel_request_auth_identity.py"
    with (tmp_path / identity_relative).open("a", encoding="utf-8") as identity_source:
        identity_source.write("\nsecrets.token_bytes(8)\n")
    monkeypatch.setitem(globals(), "REQUEST_PATH_MODULES", original_modules)
    with pytest.raises(AssertionError, match="unreviewed or changed") as changed_count:
        test_request_path_randomness_is_reviewed()
    assert identity_relative in str(changed_count.value)
    assert "secrets.token_bytes" in str(changed_count.value)
    assert "'expected': 2, 'scanned': 3" in str(changed_count.value)


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in node.elts))
    return set()


def _function_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names = []
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
    return ".".join(reversed(names)) or "<module>"


def _wire_id_field(relative: str, scope: str, field: str) -> bool:
    return (field in WIRE_ID_FIELDS
            or field in WIRE_ID_FIELDS_BY_SCOPE.get((relative, scope), set()))


def _unstable_id_sources(expression: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.Name) and node.id in UNSTABLE_ID_INPUT_NAMES:
            found.add(node.id)
        if (isinstance(node, ast.Attribute)
                and node.attr in {"sequence", "_sequence", "event_id", "_next_event_number"}):
            found.add(node.attr)
        if isinstance(node, ast.Constant) and node.value in {
                "sequence", "event_id", "source_event_id", "journal_event_id"}:
            found.add(str(node.value))
        if isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if (name in UNSTABLE_ID_CALLS or name.startswith("uuid.")
                    or name.endswith(".token_hex") or name.endswith(".urandom")
                    or name.startswith("random.") or name.startswith("time.")):
                found.add(name)
    return found


def _scan_wire_id_sources(sources: dict[str, str] | None = None
                          ) -> Counter[tuple[str, str, str, str]]:
    scanned: Counter[tuple[str, str, str, str]] = Counter()
    inputs = sources or {
        relative: (ROOT / relative).read_text(encoding="utf-8")
        for relative in REQUEST_PATH_MODULES
    }
    for relative, source in inputs.items():
        tree = ast.parse(source, filename=relative)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        aliases: dict[tuple[str, str], set[str]] = {}

        def expression_sources(expression: ast.AST, scope: str) -> set[str]:
            found = _unstable_id_sources(expression)
            for child in ast.walk(expression):
                if isinstance(child, ast.Name):
                    found.update(aliases.get((scope, child.id), ()))
            return found

        def record(node: ast.AST, field: str, expression: ast.AST) -> None:
            scope = _function_scope(node, parents)
            for unstable in expression_sources(expression, scope):
                scanned[(relative, scope, field, unstable)] += 1

        for node in sorted(ast.walk(tree), key=lambda item: (
                getattr(item, "lineno", 0), getattr(item, "col_offset", 0))):
            scope = _function_scope(node, parents)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.AugAssign)):
                expression = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if expression is None:
                    for target in targets:
                        for name in _target_names(target):
                            aliases[(scope, name)] = set()
                    continue
                sources_for_value = expression_sources(expression, scope)
                for target in targets:
                    names = _target_names(target)
                    for field in names:
                        if _wire_id_field(relative, scope, field):
                            record(node, field, expression)
                    for name in names:
                        aliases[(scope, name)] = set(sources_for_value)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                            and _wire_id_field(relative, scope, key.value)):
                        record(node, str(key.value), value)
            elif isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg and _wire_id_field(relative, scope, keyword.arg):
                        record(node, keyword.arg, keyword.value)
    return scanned


def _wire_id_source_violations(
        scanned: Counter[tuple[str, str, str, str]],
) -> dict[tuple[str, str, str, str], dict[str, int]]:
    expected = Counter({key: value[0] for key, value in WIRE_ID_SOURCE_ALLOWLIST.items()})
    return {
        key: {"expected": expected.get(key, 0), "scanned": scanned.get(key, 0)}
        for key in set(expected) | set(scanned)
        if expected.get(key, 0) != scanned.get(key, 0)
    }


def test_wire_rendered_id_sources_are_reviewed():
    for _key, (count, reason) in WIRE_ID_SOURCE_ALLOWLIST.items():
        assert count > 0 and reason.strip()
    scanned = _scan_wire_id_sources()
    expected_count = sum(item[0] for item in WIRE_ID_SOURCE_ALLOWLIST.values())
    scanned_count = sum(scanned.values())
    violations = _wire_id_source_violations(scanned)
    violated_count = sum(abs(value["expected"] - value["scanned"])
                         for value in violations.values())
    print(
        f"wire-rendered id source scan: expected={expected_count} "
        f"scanned={scanned_count} violated={violated_count}"
    )
    assert not violations, f"unreviewed unstable wire ID sources: {violations}"


def test_wire_id_source_scan_rejects_sequence_derivation():
    planted = {
        "planted.py": (
            "def build(journal):\n"
            "    request_id = f'request:{journal._sequence}'\n"
            "    return {'request_id': request_id}\n"
        ),
    }
    scanned = _scan_wire_id_sources(planted)
    assert scanned[("planted.py", "build", "request_id", "_sequence")] >= 1
    assert _wire_id_source_violations(scanned)


def test_wire_id_source_scan_tracks_unstable_aliases():
    planted = {
        "planted.py": (
            "import time, uuid\n"
            "from secrets import token_hex\n"
            "from os import urandom\n"
            "from random import random\n"
            "def build(journal):\n"
            "    position = journal._sequence\n"
            "    request_id = f'request:{position}'\n"
            "    now = time.time_ns()\n"
            "    receipt_id = f'receipt:{now}'\n"
            "    nonce = uuid.uuid4().hex\n"
            "    edit_receipt = f'edit:{nonce}'\n"
            "    effect = uuid.uuid4().hex\n"
            "    journal.apply(effect_id=effect)\n"
            "    branch_id = 'fork-'\n"
            "    branch_id += uuid.uuid4().hex\n"
            "    worker_id = uuid.uuid4().hex\n"
            "    journal_id = f'journal:{journal._sequence}'\n"
            "    token = token_hex(10)\n"
            "    token_id = f'token:{token}'\n"
            "    raw = urandom(10).hex()\n"
            "    raw_id = f'raw:{raw}'\n"
            "    random_id = f'random:{random()}'\n"
            "    return {'request_id': request_id, 'receipt_id': receipt_id, "
            "'edit_receipt': edit_receipt, 'unsettled_effect_ids': [effect], "
            "'source_event_id': journal.event_id, 'branch_id': branch_id, "
            "'worker_id': worker_id, 'journal_id': journal_id, 'execution_id': token_id, "
            "'intention_id': raw_id, 'continuation_id': random_id}\n"
        ),
    }
    scanned = _scan_wire_id_sources(planted)
    assert scanned[("planted.py", "build", "request_id", "_sequence")] >= 1
    assert scanned[("planted.py", "build", "receipt_id", "time.time_ns")] >= 1
    assert scanned[("planted.py", "build", "edit_receipt", "uuid.uuid4")] >= 1
    assert scanned[("planted.py", "build", "effect_id", "uuid.uuid4")] >= 1
    assert scanned[("planted.py", "build", "unsettled_effect_ids", "uuid.uuid4")] >= 1
    assert scanned[("planted.py", "build", "source_event_id", "event_id")] >= 1
    assert scanned[("planted.py", "build", "branch_id", "uuid.uuid4")] >= 1
    assert scanned[("planted.py", "build", "worker_id", "uuid.uuid4")] >= 1
    assert scanned[("planted.py", "build", "journal_id", "_sequence")] >= 1
    assert scanned[("planted.py", "build", "execution_id", "token_hex")] >= 1
    assert scanned[("planted.py", "build", "intention_id", "urandom")] >= 1
    assert scanned[("planted.py", "build", "continuation_id", "random")] >= 1
    assert _wire_id_source_violations(scanned)


def test_action_receipt_ids_ignore_journal_noise_and_replay_position(monkeypatch):
    import kernel_wire_identity

    monkeypatch.setenv("SF_LAB_DETFIX3", "on")

    class Journal:
        def __init__(self, sequence: int, *, seed: str = "46" * 16,
                     task_id: str = "task-46c84e9d"):
            self.task_id = task_id
            self._sequence = sequence
            self.manifest = {"run_identity_seed": seed}

    action = {
        "request_id": "674ce0515b391b5f235b",
        "step": 7,
        "tool_call_id": "call_7",
        "action_kind": "edit",
        "fingerprint": "b7d036c6b1d5129d",
        "target_path": "shop/pricing.py",
    }
    first_journal = Journal(42)
    original = kernel_wire_identity.action_receipt_wire_id(first_journal, **action)
    provider_call_id_changed = kernel_wire_identity.action_receipt_wire_id(
        Journal(42), **{**action, "tool_call_id": "call_randomized"})
    duplicate = kernel_wire_identity.action_receipt_wire_id(
        first_journal, **{**action, "tool_call_id": "another_provider_call_id"},
        ordinal=2)
    # The per-run random seed differs between a control run and a resumed or
    # repeated run of the same task; wire IDs must not depend on it.
    resumed = kernel_wire_identity.action_receipt_wire_id(
        Journal(107, seed="47" * 16), **action)
    next_step = kernel_wire_identity.action_receipt_wire_id(
        Journal(108), **{**action, "step": 8}
    )
    other_task = kernel_wire_identity.action_receipt_wire_id(
        Journal(42, task_id="different-task-id"), **action)

    assert original == resumed
    assert original == provider_call_id_changed
    assert duplicate != original
    assert re.fullmatch(r"[0-9a-f]{20}", original)
    assert next_step != original
    assert other_task != original

    prior_receipts = [
        {"request_id": action["request_id"], "turn": action["step"],
         "kind": "edit", "target_path": action["target_path"]},
    ]
    assert kernel_wire_identity.action_receipt_ordinal(
        prior_receipts, request_id=action["request_id"], step=action["step"],
        action_kind="edit", target_path=action["target_path"]) == 2


def test_content_wire_ids_use_seed_step_content_and_same_content_ordinal():
    import kernel_wire_identity

    class Journal:
        def __init__(self, sequence: int, seed: str = "46" * 16,
                     task_id: str = "task-46c84e9d"):
            self.task_id = task_id
            self._sequence = sequence
            self.manifest = {"run_identity_seed": seed}

    first_journal = Journal(11)
    first = kernel_wire_identity.content_wire_id(
        first_journal, "check-execution", 7, {"selector": "tests/test_shop.py"},
        ordinal=1)
    duplicate = kernel_wire_identity.content_wire_id(
        first_journal, "check-execution", 7, {"selector": "tests/test_shop.py"},
        ordinal=2)
    replay = kernel_wire_identity.content_wire_id(
        Journal(91), "check-execution", 7, {"selector": "tests/test_shop.py"},
        ordinal=1)
    other_step = kernel_wire_identity.content_wire_id(
        Journal(91), "check-execution", 8, {"selector": "tests/test_shop.py"})
    other_seed = kernel_wire_identity.content_wire_id(
        Journal(91, "47" * 16), "check-execution", 7,
        {"selector": "tests/test_shop.py"})
    other_task = kernel_wire_identity.content_wire_id(
        Journal(91, task_id="other-task"), "check-execution", 7,
        {"selector": "tests/test_shop.py"})

    assert first == replay
    assert duplicate != first
    assert other_step != first
    assert other_seed == first, "the per-run random seed must not reach a wire ID"
    assert other_task != first


def test_wire_ids_rederive_from_disk_replayed_action_receipt_history(tmp_path):
    from kernel_continuation_journal import ContinuationJournal
    from kernel_continuation_runtime import _wire_effect_id, _wire_effect_ordinal
    from kernel_wire_identity import action_receipt_ordinal, action_receipt_wire_id

    parent = tmp_path / "journal-parent"
    parent.mkdir()
    seed = "46" * 16
    journal = ContinuationJournal(parent, "task-wire-replay", {
        "run_identity_seed": seed,
    })
    action = {
        "request_id": "request-live-7", "step": 7, "tool_call_id": "call-1",
        "action_kind": "edit", "fingerprint": "same-edit", "target_path": "shop/a.py",
    }
    effect = {
        "request_id": "request-live-7", "step": 7, "tool": "str_replace",
        "arguments_hash": "same-edit",
    }
    first_ordinal = action_receipt_ordinal(
        journal, request_id=action["request_id"], step=action["step"],
        action_kind=action["action_kind"], target_path=action["target_path"],
    )
    first_id = action_receipt_wire_id(journal, **action, ordinal=first_ordinal)
    first = {
        "receipt_id": first_id, "request_id": action["request_id"],
        "turn": action["step"], "kind": action["action_kind"],
        "target_path": action["target_path"],
    }
    journal.emit("action_receipt", receipt=first)
    second_ordinal = action_receipt_ordinal(
        journal, request_id=action["request_id"], step=action["step"],
        action_kind=action["action_kind"], target_path=action["target_path"],
    )
    second_id = action_receipt_wire_id(journal, **action, ordinal=second_ordinal)
    second = {**first, "receipt_id": second_id}
    journal.emit("action_receipt", receipt=second)
    first_effect_ordinal = _wire_effect_ordinal(journal, **effect)
    first_effect_id = _wire_effect_id(journal, **effect, ordinal=first_effect_ordinal)
    journal.emit(
        "step_dispatch_binding", **effect,
        continuation_effect_id=first_effect_id,
    )
    second_effect_ordinal = _wire_effect_ordinal(journal, **effect)
    second_effect_id = _wire_effect_id(journal, **effect, ordinal=second_effect_ordinal)
    journal.emit(
        "step_dispatch_binding", **effect,
        continuation_effect_id=second_effect_id,
    )
    live_next_ordinal = action_receipt_ordinal(
        journal, request_id=action["request_id"], step=action["step"],
        action_kind=action["action_kind"], target_path=action["target_path"],
    )
    live_next = action_receipt_wire_id(journal, **action, ordinal=live_next_ordinal)
    live_next_effect_ordinal = _wire_effect_ordinal(journal, **effect)
    live_next_effect_id = _wire_effect_id(
        journal, **effect, ordinal=live_next_effect_ordinal)
    path = journal.path
    head = journal.head
    manifest = dict(journal.manifest)
    journal.close()

    replayed = ContinuationJournal.resume(
        path, task_id="task-wire-replay", manifest=manifest, expected_head=head)
    try:
        replayed_ordinal = action_receipt_ordinal(
            replayed, request_id=action["request_id"], step=action["step"],
            action_kind=action["action_kind"], target_path=action["target_path"],
        )
        replayed_next = action_receipt_wire_id(
            replayed, **action, ordinal=replayed_ordinal)
        replayed_effect_ordinal = _wire_effect_ordinal(replayed, **effect)
        replayed_effect_id = _wire_effect_id(
            replayed, **effect, ordinal=replayed_effect_ordinal)
        assert replayed_ordinal == live_next_ordinal == 3
        assert replayed_effect_ordinal == live_next_effect_ordinal == 3
        assert action_receipt_wire_id(replayed, **action, ordinal=2) == second_id
        assert _wire_effect_id(replayed, **effect, ordinal=2) == second_effect_id
        assert replayed_next == live_next
        assert replayed_effect_id == live_next_effect_id
    finally:
        replayed.close()


def test_request_wire_id_ignores_journal_sequence_and_binds_request_content():
    from kernel_wire_identity import request_wire_id

    class Journal:
        def __init__(self, sequence: int, seed: str = "46" * 16):
            self._sequence = sequence
            self.manifest = {"run_identity_seed": seed}

    body = {"messages": [{"role": "user", "content": "same frame"}]}
    first = request_wire_id(Journal(12), 7, body, purpose="decision")
    observer_shifted = request_wire_id(Journal(39), 7, body, purpose="decision")
    next_step = request_wire_id(Journal(39), 8, body, purpose="decision")
    other_content = request_wire_id(
        Journal(39), 7, {"messages": [{"role": "user", "content": "changed"}]},
        purpose="decision",
    )

    assert first == observer_shifted
    assert next_step != first
    assert other_content != first


def test_request_capture_gives_a_second_send_in_one_step_its_own_id():
    from kernel_measurement import RequestCapture

    class Journal:
        task_id = "task-a"
        manifest = {"run_identity_seed": "46" * 16}

        def __init__(self):
            self.submitted: set[str] = set()

        def request_records(self, request_id):
            return ({"kind": "request_submitted"},) if request_id in self.submitted else ()

    journal = Journal()
    first = RequestCapture(journal, 4, "snap", call_purpose="decision").request_id
    # Built but never sent (a crash before submit): rebuilding keeps the same ID.
    assert RequestCapture(journal, 4, "snap", call_purpose="decision").request_id == first
    # Once it was submitted, the next send in that step (a route escalation or a
    # discarded reply asked again) must not reuse it: the two sends would share
    # one request contract, and the second was refused before any check ran.
    journal.submitted.add(first)
    second = RequestCapture(journal, 4, "snap", call_purpose="decision").request_id
    assert second != first
    journal.submitted.add(second)
    assert RequestCapture(journal, 4, "snap", call_purpose="decision").request_id not in {first, second}
    # A fresh journal with the same task allocates the same sequence of IDs.
    replayed = Journal()
    assert RequestCapture(replayed, 4, "snap", call_purpose="decision").request_id == first


def test_checkpoint_capture_observer_settings_reach_engine_spec(monkeypatch):
    from stateframe import runner

    monkeypatch.setenv("STATEFRAME_FL_CHECKPOINT_CAPTURE", "on")
    monkeypatch.setenv("STATEFRAME_FL_CAPTURE_EVERY_K", "1")
    resolved = runner.resolve_opt_in_switches()
    assert resolved["STATEFRAME_FL_CHECKPOINT_CAPTURE"] == "on"
    assert resolved["STATEFRAME_FL_CAPTURE_EVERY_K"] == "1"

    monkeypatch.setenv("STATEFRAME_FL_CAPTURE_EVERY_K", "1001")
    with pytest.raises(runner.RunError, match="between 1 and 1000"):
        runner.resolve_opt_in_switches()


def test_check_receipt_execution_id_ignores_journal_sequence():
    from kernel_check_receipts import _execution_receipt_id, canonical_execution_id

    class Journal:
        def __init__(self, sequence: int):
            self._sequence = sequence
            self.manifest = {"run_identity_seed": "46" * 16}

    class Runtime:
        def __init__(self, sequence: int):
            self.journal = Journal(sequence)
            self.owner = type("Owner", (), {"current_step": 7})()

    original = canonical_execution_id(
        Runtime(12), selector="tests/test_shop.py", selector_scope=["tests/test_shop.py"],
        producer="pytest")
    observer_shifted = canonical_execution_id(
        Runtime(39), selector="tests/test_shop.py", selector_scope=["tests/test_shop.py"],
        producer="pytest")

    assert original == observer_shifted
    assert _execution_receipt_id(original) == _execution_receipt_id(observer_shifted)
    first_sequence = canonical_execution_id(
        Runtime(12), selector="tests/test_shop.py", sequence=3,
        selector_scope=["tests/test_shop.py"], producer="pytest")
    same_sequence_after_noise = canonical_execution_id(
        Runtime(39), selector="tests/test_shop.py", sequence=3,
        selector_scope=["tests/test_shop.py"], producer="pytest")
    next_sequence = canonical_execution_id(
        Runtime(39), selector="tests/test_shop.py", sequence=4,
        selector_scope=["tests/test_shop.py"], producer="pytest")
    assert first_sequence == same_sequence_after_noise
    assert next_sequence != first_sequence


def test_test_run_profile_id_ignores_journal_sequence():
    from kernel_test_run_spans import TestRunProfile

    class Journal:
        def __init__(self, sequence: int):
            self._sequence = sequence
            self.manifest = {"run_identity_seed": "46" * 16}

    class Runtime:
        def __init__(self, sequence: int):
            self.journal = Journal(sequence)
            self.owner = type("Owner", (), {"current_step": 7})()

    first = TestRunProfile("on", runtime=Runtime(11))
    observer_shifted = TestRunProfile("on", runtime=Runtime(38))
    assert first.execution_id == observer_shifted.execution_id

    # A second test run in the same step gets its own ID; sharing one made it
    # inherit the first run's pinned binding and its older revision (0.2.3 comparison).
    runtime = Runtime(11)
    again = (TestRunProfile("on", runtime=runtime), TestRunProfile("on", runtime=runtime))
    assert again[0].execution_id == first.execution_id
    assert again[0].execution_id != again[1].execution_id


def test_pending_effect_id_uses_seed_step_content_and_duplicate_ordinal():
    from kernel_continuation_runtime import _wire_effect_id

    class Journal:
        def __init__(self, sequence: int, seed: str = "46" * 16):
            self._sequence = sequence
            self.manifest = {"run_identity_seed": seed}

    action = {
        "request_id": "674ce0515b391b5f235b",
        "step": 7,
        "tool": "str_replace",
        "arguments_hash": "b7d036c6b1d5129d",
    }
    original_journal = Journal(12)
    first = _wire_effect_id(
        original_journal, request_id=action["request_id"], step=7,
        tool=action["tool"], arguments_hash=action["arguments_hash"],
    )
    duplicate = _wire_effect_id(
        original_journal, request_id=action["request_id"], step=7,
        tool=action["tool"], arguments_hash=action["arguments_hash"],
        ordinal=2,
    )
    observer_shifted = _wire_effect_id(
        Journal(39), request_id=action["request_id"], step=7,
        tool=action["tool"], arguments_hash=action["arguments_hash"],
    )
    next_step = _wire_effect_id(
        Journal(39), request_id=action["request_id"], step=8,
        tool=action["tool"], arguments_hash=action["arguments_hash"],
    )

    assert first == observer_shifted
    assert duplicate != first
    assert next_step != first


def test_continuation_intention_ids_ignore_journal_position_and_keep_local_ordinals():
    from kernel_continuation_runtime import _wire_intention_id

    class Journal:
        def __init__(self, sequence: int, seed: str = "46" * 16,
                     task_id: str = "task-46c84e9d"):
            self.task_id = task_id
            self._sequence = sequence
            self.manifest = {"run_identity_seed": seed}

    content = {
        "role": "source_detour", "parent_id": "intention:parent",
        "objective": "Acquire source: shop/pricing.py",
        "acceptance": {"kind": "source_read", "start_line": 4, "end_line": 8},
    }
    first_journal = Journal(8)
    first = _wire_intention_id(first_journal, step=7, content=content)
    duplicate = _wire_intention_id(first_journal, step=7, content=content, ordinal=2)
    observer_shifted = _wire_intention_id(Journal(19), step=7, content=content)
    next_step = _wire_intention_id(Journal(19), step=8, content=content)
    other_seed = _wire_intention_id(Journal(19, "47" * 16), step=7, content=content)
    other_task = _wire_intention_id(Journal(19, task_id="other-task"), step=7, content=content)

    assert first == observer_shifted
    assert duplicate != first
    assert next_step != first
    assert other_seed == first, "the per-run random seed must not reach a wire ID"
    assert other_task != first


def test_continuation_context_omits_raw_journal_references():
    from kernel_continuation_runtime import _public_continuation_context

    class Journal:
        run_id = "host-run"
        manifest = {"run_identity_seed": "46" * 16}

    check_receipt = "check-receipt:" + "a" * 24
    public = _public_continuation_context({
        "identity": {
            "run_id": "host-run", "workspace_id": "workspace-host",
            "grant_id": "grant-host",
        },
        "active": {
            "id": "intention:" + "b" * 20,
            "event_id": "host-run:19",
            "receipt_ref": "79a763a2-80d9-43c1-8d86-836b7e692816",
            "check_receipt_id": check_receipt,
        },
        "return_receipts": ["host-run:18", "79a763a2-80d9-43c1-8d86-836b7e692816"],
    }, Journal())
    rendered = json.dumps(public, sort_keys=True)

    assert "host-run" not in rendered
    assert "79a763a2-80d9-43c1-8d86-836b7e692816" not in rendered
    assert "event_id" not in public["active"]
    assert "receipt_ref" not in public["active"]
    assert "return_receipts" not in public
    assert public["active"]["id"] == "intention:" + "b" * 20
    assert public["active"]["check_receipt_id"] == check_receipt


def test_router_replay_handoff_omits_internal_lease_uuid():
    from kernel_stateframe_model_router_replay import public_projection

    projection = public_projection({
        "lease": {
            "lease_id": "79a763a2-80d9-43c1-8d86-836b7e692816",
            "endpoint": "frontier", "actions_remaining": 2,
        },
        "consumed_receipts": (),
    })

    assert projection["status"] == "replayed"
    assert projection["endpoint"] == "frontier"
    assert "lease_id" not in json.dumps(projection)


def test_keyed_completion_tool_result_omits_host_event_id():
    from kernel_continuation_runtime import LiveContinuation

    class Journal:
        def cached_records_since(self, cursor):
            if cursor >= 19:
                return ()
            return ({
                "sequence": 19,
                "kind": "continuation_transition", "operation": "complete",
                "event_id": "host-run:19",
                "data": {"submission_key": ["task", "submission"]},
            },)

    continuation = object.__new__(LiveContinuation)
    continuation.journal = Journal()

    assert continuation.submission_result(("task", "submission")) == {
        "accepted": True, "submission_key": ["task", "submission"],
    }
    assert continuation._view().last_sync_count == 0


def test_verdict_evidence_references_ignore_event_sequence_and_id():
    from kernel_verdict import _wire_event_refs
    from kernel_wire_truth_facts import _public_evidence_refs

    first = (0, {
        "event_id": "run-a:8", "sequence": 8, "selector": "tests/test_shop.py",
        "status": "PASS", "producer": "test_subprocess",
    })
    observer_shifted = (1, {
        "event_id": "run-a:19", "sequence": 19, "selector": "tests/test_shop.py",
        "status": "PASS", "producer": "test_subprocess",
    })
    original = _wire_event_refs([first], wire_seed="46" * 16, wire_step=7)
    shifted = _wire_event_refs(
        [(0, {"event_id": "run-a:1", "sequence": 1, "kind": "observer"}),
         observer_shifted],
        wire_seed="46" * 16, wire_step=7,
    )

    assert original[0] == shifted[1]
    duplicate = _wire_event_refs(
        [first, (1, dict(first[1]))], wire_seed="46" * 16, wire_step=7)
    assert duplicate[0] != duplicate[1]

    class Journal:
        run_id = "host-run"
        manifest = {"run_identity_seed": "46" * 16}

    check_receipt = "check-receipt:" + "a" * 24
    assert _public_evidence_refs(
        ["host-run:8", check_receipt], Journal()) == check_receipt


def test_host_fact_evidence_omits_sequence_bearing_event_ids():
    from kernel_host_facts import _evidence_ids

    fact_content = {"kind": "test_result", "subject": "tests/test_shop.py"}
    first = _evidence_ids({"event_id": "host-run:8"}, fact_content)
    observer_shifted = _evidence_ids({"event_id": "host-run:19"}, fact_content)

    assert first == observer_shifted
    assert all("host-run:" not in item for item in first)


def test_frame_continuity_check_rows_omit_journal_event_identity_and_order():
    from kernel_stateframe_frame_continuity import journal_check_receipts

    receipt_id = "check-receipt:" + "a" * 24
    base = {
        "kind": "check_receipt", "status": "PASS", "selector": "tests/test_shop.py",
        "revision": "sha256:" + "b" * 64, "receipt_id": receipt_id,
        "execution_id": "check-execution:" + "c" * 20,
        "request_id": "d" * 20,
    }
    first = journal_check_receipts({"journal_receipt_records": [
        {**base, "event_id": "run:8", "sequence": 8, "ordinal": 1},
    ]})
    observer_shifted = journal_check_receipts({"journal_receipt_records": [
        {**base, "event_id": "run:19", "source_event_id": "run:19",
         "sequence": 19, "ordinal": 2},
    ]})

    assert first == observer_shifted
    assert first[0]["receipt_id"] == receipt_id
    assert not {"event_id", "source_event_id", "sequence", "ordinal"}.intersection(first[0])
    legacy = journal_check_receipts({"journal_receipt_records": [{
        "kind": "check_receipt", "status": "PASS", "event_id": "run:20", "sequence": 20,
    }]})
    assert "receipt_id" not in legacy[0]


def test_frame_journal_edit_projection_never_uses_sequence_as_receipt_id():
    from kernel_frame_journal_gate import _edit_row

    assert "receipt_id" not in _edit_row({"kind": "edit"}, 11)
    assert "receipt_id" not in _edit_row({"kind": "edit", "receipt_id": "run:11"}, 11)
    receipt_id = "a" * 20
    assert _edit_row({"kind": "edit", "receipt_id": receipt_id}, 91)["receipt_id"] == receipt_id


def test_check_result_projection_drops_host_event_ids_and_sequences():
    from kernel_frame_edit_integrity import collapse_check_results

    stable_receipt_id = "check-receipt:" + "e" * 24
    rows = collapse_check_results([
        {"kind": "pytest", "selector": "tests/test_shop.py", "status": "FAIL",
         "event_id": "run:8", "sequence": 8, "revision": "old"},
        {"kind": "pytest", "selector": "tests/test_shop.py", "status": "PASS",
         "receipt_id": stable_receipt_id, "event_id": "run:19", "sequence": 19,
         "revision": "current"},
    ])

    assert len(rows) == 1
    assert rows[0]["receipt_id"] == stable_receipt_id
    assert not {"event_id", "source_event_id", "sequence", "event_sequence"}.intersection(rows[0])
    assert "receipt_id" not in rows[0]["superseded"][0]


def test_hstate_full_suite_projection_omits_event_id_fallback():
    from kernel_stateframe_hstate_plus import full_suite_status

    runtime = type("Runtime", (), {"test_receipts": []})()
    frame = {"test_receipts": [{
        "kind": "pytest", "scope": "full suite", "status": "PASS",
        "revision": "r1", "event_id": "run:8",
    }]}
    status = full_suite_status(runtime, frame, revision="r1")

    assert status["status"] == "PASS"
    assert "receipt_id" not in status


def test_runtime_test_completion_receipt_ignores_journal_event_identity(monkeypatch):
    from types import SimpleNamespace

    import kernel_readiness_authority
    from kernel_stateframe_runtime import StateFrameRuntime

    monkeypatch.setattr(kernel_readiness_authority, "runtime_enabled", lambda _runtime: False)

    class Journal:
        manifest = {"run_identity_seed": "46" * 16}

        def __init__(self, sequence: int):
            self.sequence = sequence

    def build(sequence: int):
        runtime = SimpleNamespace(
            journal=Journal(sequence),
            owner=SimpleNamespace(current_step=7),
            test_receipts=[],
            micrograph_wire=None,
            _test_invalidation_receipt_fields=lambda _event, _owner, _runtime: {},
        )
        StateFrameRuntime._record_test_completion_receipt(runtime, {
            "kind": "test_subprocess_completed", "event_id": f"run:{sequence}",
            "sequence": sequence, "status": "PASS", "selector": "tests/test_shop.py",
            "revision": "rev-a",
        }, [])
        return runtime.test_receipts[0]

    first = build(8)
    shifted = build(19)
    assert first["receipt_id"] == shifted["receipt_id"]
    # INVERTED (11-03): the receipt row keeps the event's own fields, event_id and
    # sequence included, because dropping them changed request bytes against 0.2.8
    # (test_existing_discount_requests_match_untouched_0_2_8). Only the receipt ID
    # must ignore them, which the line above checks; the same-seed and observer
    # engine runs in this file prove they never reach the wire.
    assert first["event_id"] == "run:8" and shifted["event_id"] == "run:19"


def test_runtime_probe_receipt_uses_seed_step_and_observation_content():
    from types import SimpleNamespace

    from kernel_stateframe_runtime import StateFrameRuntime

    class Journal:
        manifest = {"run_identity_seed": "46" * 16}

        def __init__(self, sequence: int):
            self.sequence = sequence

        def emit(self, _kind: str, **_fields):
            self.sequence += 1
            return {"event_id": f"run:{self.sequence}", "sequence": self.sequence}

    class Gate:
        def record_probe(self, *_args, **_kwargs):
            return {"accepted": True, "status": "PASS",
                    "record": {"result": "observed", "revision": "rev-a"}}

    def build(sequence: int):
        runtime = SimpleNamespace(
            probe_gate_enabled=True, probe_gate=Gate(), journal=Journal(sequence),
            owner=SimpleNamespace(current_step=7), _probe_revision="rev-a", probe_receipts=[],
        )
        StateFrameRuntime.record_probe(runtime, "requirement-1", {"host": "observed"})
        return runtime

    first = build(8)
    shifted = build(19)
    assert first.probe_receipts[0]["receipt_id"] == shifted.probe_receipts[0]["receipt_id"]
    StateFrameRuntime.record_probe(first, "requirement-1", {"host": "observed"})
    assert first.probe_receipts[0]["receipt_id"] != first.probe_receipts[1]["receipt_id"]


def test_runtime_harness_fault_receipt_ignores_journal_sequence():
    from types import SimpleNamespace

    from kernel_stateframe_runtime import StateFrameRuntime

    class Journal:
        manifest = {"run_identity_seed": "46" * 16}

        def __init__(self, sequence: int):
            self.sequence = sequence

        def emit(self, _kind: str, **_fields):
            self.sequence += 1
            return {"event_id": f"run:{self.sequence}", "sequence": self.sequence}

    class Actions:
        def append(self, **row):
            self.row = row

    def build(sequence: int):
        runtime = SimpleNamespace(
            journal=Journal(sequence), owner=SimpleNamespace(current_step=7),
            fault_receipts=[], actions=Actions(), binding=lambda: {"revision": "rev-a"},
            lifecycle=lambda _kind: None,
        )
        StateFrameRuntime.harness_fault(runtime, "transport_timeout", "tests/test_shop.py")
        return runtime

    first = build(8)
    shifted = build(19)
    first_id = first.fault_receipts[0]["receipt_id"]
    assert first_id == shifted.fault_receipts[0]["receipt_id"]
    assert "event_id" not in first.fault_receipts[0]
    assert first.actions.row["observation"] == first_id


def test_conclusion_cache_receipt_line_drops_event_id_fallback():
    from kernel_conclusion_cache import normalize_receipt, receipt_line

    current = {}
    first = normalize_receipt({
        "kind": "pytest", "selector": "tests/test_shop.py", "status": "PASS",
        "event_id": "run:8",
    }, current)
    shifted = normalize_receipt({
        "kind": "pytest", "selector": "tests/test_shop.py", "status": "PASS",
        "event_id": "run:19",
    }, current)
    stable = normalize_receipt({
        "kind": "pytest", "selector": "tests/test_shop.py", "status": "PASS",
        "receipt_id": "check-receipt:" + "f" * 24,
    }, current)

    assert first["receipt_id"] is None
    assert shifted["receipt_id"] is None
    assert receipt_line(first) == receipt_line(shifted)
    assert stable["receipt_id"] == "check-receipt:" + "f" * 24


def test_environment_receipt_projection_ignores_event_id_and_sequence(tmp_path):
    from kernel_environment_state import EnvironmentState

    class Journal:
        manifest = {"run_identity_seed": "46" * 16}

    common = {
        "kind": "test", "result_class": "PASS", "revision": "rev-a",
        "workspace_revision": 4, "fingerprint": "suite:shop", "path": "tests/test_shop.py",
    }
    first_state = EnvironmentState(tmp_path, journal=Journal())
    shifted_state = EnvironmentState(tmp_path, journal=Journal())
    first = first_state.ingest_receipts([
        {**common, "event_id": "run:8", "sequence": 8},
        {**common, "event_id": "run:9", "sequence": 9},
    ])
    shifted = shifted_state.ingest_receipts([
        {**common, "event_id": "run:19", "sequence": 19},
        {**common, "event_id": "run:20", "sequence": 20},
    ])

    assert [row.receipt_id for row in first] == [row.receipt_id for row in shifted]
    assert len({row.receipt_id for row in first}) == 2
    assert all(row.receipt_id.startswith("environment-receipt:") for row in first)


def test_expectation_receipt_ids_ignore_prior_observations_and_keep_local_ordinals():
    from expectation_ledger import (
        EnumerationKind, EnumerationObservation, ExpectationKey,
        ExpectationLedger, ObservationProvenance,
    )

    key = ExpectationKey(
        scope="repo", environment="test", runner="pytest", subject="shop",
        selector="tests/test_shop.py", metric="test_passed", unit="tests",
    )
    observation = EnumerationObservation.count(
        kind=EnumerationKind.TEST_PASSED,
        key=key,
        value=4,
        provenance=ObservationProvenance(
            turn=7, workspace_revision=3, action_id="provider-call-random-a"),
    )
    replay_observation = EnumerationObservation.count(
        kind=EnumerationKind.TEST_PASSED,
        key=key,
        value=4,
        provenance=ObservationProvenance(
            turn=7, workspace_revision=3, action_id="provider-call-random-b"),
    )
    shifted = ExpectationLedger(run_seed="46" * 16)
    shifted.observe(EnumerationObservation.count(
        kind=EnumerationKind.FILE_COUNT,
        key=ExpectationKey(
            scope="repo", environment="test", runner="files", subject="shop",
            selector="src", metric="file_count", unit="files",
        ),
        value=12,
        provenance=ObservationProvenance(turn=6),
    ))
    first = shifted.observe(observation).observation.provenance.receipt_id
    duplicate = shifted.observe(observation).observation.provenance.receipt_id
    replay = ExpectationLedger(run_seed="46" * 16).observe(replay_observation)
    other_seed = ExpectationLedger(run_seed="47" * 16).observe(observation)

    assert first == replay.observation.provenance.receipt_id
    assert duplicate != first
    assert other_seed.observation.provenance.receipt_id != first
    assert re.fullmatch(r"expectation-receipt:[0-9a-f]{24}", first)


def test_effect_receipt_fallback_ids_use_seed_step_content_and_ordinal(monkeypatch):
    from kernel_effect_liveness import EffectReceipt

    monkeypatch.setenv("STATEFRAME_RUN_IDENTITY_SEED", "46" * 16)

    class Journal:
        manifest = {"run_identity_seed": "46" * 16}

    value = {
        "step": 7, "obligation_id": "obl-1", "control": "positive",
        "effect_observed": True, "event_id": "run:8", "sequence": 8,
    }
    first = EffectReceipt.from_mapping(value, ordinal=1, journal=Journal())
    replay = EffectReceipt.from_mapping(
        {**value, "event_id": "run:19", "sequence": 19}, ordinal=1,
        journal=Journal())
    duplicate = EffectReceipt.from_mapping(value, ordinal=2, journal=Journal())

    assert first.receipt_id == replay.receipt_id
    assert first.experiment_id == replay.experiment_id
    assert first.stimulus_correlation_id == replay.stimulus_correlation_id
    assert duplicate.receipt_id != first.receipt_id
    assert duplicate.experiment_id != first.experiment_id


def test_effect_liveness_ingest_ids_use_journal_seed_and_same_content_ordinal():
    from kernel_effect_liveness import EffectLivenessProbe

    class Journal:
        manifest = {"run_identity_seed": "46" * 16}

        def __init__(self, sequence: int):
            self.sequence = sequence

        def emit(self, _kind: str, **_fields):
            self.sequence += 1

    value = {
        "step": 7, "obligation_id": "obl-1", "control": "positive",
        "effect_observed": True, "event_id": "run:8", "sequence": 8,
    }
    first_probe = EffectLivenessProbe(journal=Journal(8), enabled=True)
    first = first_probe.ingest_receipt(value)
    duplicate = first_probe.ingest_receipt(value)
    shifted_probe = EffectLivenessProbe(journal=Journal(19), enabled=True)
    shifted = shifted_probe.ingest_receipt(
        {**value, "event_id": "run:19", "sequence": 19})

    assert first is not None and duplicate is not None and shifted is not None
    assert first.receipt_id == shifted.receipt_id
    assert duplicate.receipt_id != first.receipt_id


def test_portfolio_request_context_ids_bind_seed_step_content_and_ordinal(tmp_path):
    from kernel_stateframe_portfolio import PortfolioManager

    base = tmp_path / "base"
    base.mkdir()
    (base / "README.md").write_text("stable base\n", encoding="utf-8")
    request = {
        "step": 7,
        "messages": [{"role": "user", "content": "repair the same issue"}],
        "journal_id": "journal:observer-shifted",
        "sequence": 41,
    }

    def branch(store_name: str, root_index: int = 0):
        manager = PortfolioManager(
            base, store_root=tmp_path / store_name, ticket_id="ticket-1",
            run_seed="46" * 16,
        )
        return manager._new_branch(
            base, root_index=root_index, fork_index=None, temperature=0.0,
            request=request,
        )

    first = branch("portfolio-a")
    replay = branch("portfolio-b")
    second_ordinal = branch("portfolio-c", root_index=1)

    assert (first.branch_id, first.journal_id, first.request_id) == (
        replay.branch_id, replay.journal_id, replay.request_id)
    assert first.request_id != second_ordinal.request_id
    assert first.branch_id != second_ordinal.branch_id


def test_fork_a_request_context_ids_ignore_sequence_and_randomness(tmp_path):
    from kernel_stateframe_fork_a import ForkAController

    base = tmp_path / "base"
    base.mkdir()
    (base / "README.md").write_text("stable base\n", encoding="utf-8")
    request = {
        "step": 7,
        "messages": [{"role": "user", "content": "repair the same issue"}],
        "journal_id": "journal:observer-shifted",
        "sequence": 41,
    }

    def worker(store_name: str, index: int = 0):
        controller = ForkAController(
            base, store_root=tmp_path / store_name, ticket_id="ticket-1",
            run_seed="46" * 16,
        )
        return controller._new_worker(index, request=request)

    first = worker("fork-a")
    replay = worker("fork-b")
    second_ordinal = worker("fork-c", index=1)

    assert (first.worker_id, first.journal_id, first.request_id) == (
        replay.worker_id, replay.journal_id, replay.request_id)
    assert first.request_id != second_ordinal.request_id
    assert first.worker_id != second_ordinal.worker_id


def test_timeline_frame_ids_and_recent_turns_ignore_observer_events(monkeypatch):
    from kernel_timeline import TimelineIndex

    monkeypatch.setenv("STATEFRAME_RUN_IDENTITY_SEED", "46" * 16)

    def frame(*, observer: bool):
        timeline = TimelineIndex("run:random-host-label")
        timeline.append_message(
            "user", "update shop pricing", turn=7, message_id="message:random-host-id",
            ask_id="ask:random-host-id", persist=False,
        )
        timeline.append(
            "edit", {"path": "shop/pricing.py", "event_id": "run:8", "sequence": 8},
            actor="host", turn=7, content="updated shop pricing", entities=("shop/pricing.py",),
            persist=False,
        )
        timeline.append(
            "edit", {"path": "shop/pricing.py", "event_id": "run:9", "sequence": 9},
            actor="host", turn=7, content="updated shop pricing", entities=("shop/pricing.py",),
            persist=False,
        )
        if observer:
            timeline.append(
                "fl_checkpoint_captured", {"event_id": "run:19", "sequence": 19},
                actor="host", turn=7, content="checkpoint captured", persist=False,
            )
        return timeline.frame_slice(
            focus={"path": "shop/pricing.py"}, budget_tokens=900,
        ).rendered

    plain = frame(observer=False)
    observed = frame(observer=True)

    assert plain == observed
    assert "message:random-host-id" not in plain
    assert "ask:random-host-id" not in plain
    assert "run:8" not in plain and "run:9" not in plain
    assert "checkpoint captured" not in plain


def test_micrograph_does_not_use_event_id_as_wire_receipt_identity():
    from kernel_micrograph_wire import MicroGraphWire

    wire = MicroGraphWire("task-1")
    receipt = {
        "kind": "edit", "result_class": "APPLIED", "path": "shop/pricing.py",
        "event_id": "run:8",
    }

    assert wire.observe(receipt) == ()
    assert wire.skipped == 1
    assert wire.observe({**receipt, "receipt_id": "a" * 20})


def test_where_used_report_identity_ignores_journal_event_id(monkeypatch, tmp_path):
    from kernel_where_used import analyze_accepted_edit

    monkeypatch.setenv("STATEFRAME_WHERE_USED_CLOSURE", "on")
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "pricing.py"

    class Journal:
        manifest = {"run_identity_seed": "46" * 16}

        def __init__(self, sequence: int):
            self.sequence = sequence

        def emit(self, _kind: str, **_fields):
            self.sequence += 1
            return {"event_id": f"run:{self.sequence}", "sequence": self.sequence}

    args = {
        "root": root, "path": path,
        "before": "def old_price():\n    return 1\n",
        "after": "def new_price():\n    return 2\n",
        "edit_receipt_id": "a" * 20, "step": 7,
    }
    first = analyze_accepted_edit(**args, journal=Journal(8))
    shifted = analyze_accepted_edit(**args, journal=Journal(19))

    assert first.event_id == shifted.event_id
    assert first.event_id is not None and not first.event_id.startswith("run:")
    assert first.journaled and shifted.journaled


def test_where_used_read_receipt_id_ignores_event_id_and_keeps_duplicate_ordinal():
    from kernel_where_used import SiteObligation, WhereUsedLedger

    class Journal:
        manifest = {"run_identity_seed": "46" * 16}

    def record(sequence: int, duplicate: bool = False):
        ledger = WhereUsedLedger(journal=Journal())
        ledger._obligations["site-1"] = SiteObligation(
            obligation_id="site-1", identity="price", path="shop/pricing.py",
            span={"start_line": 1, "end_line": 2}, kind="call",
            site_id="site-1",
        )
        receipt = {
            "kind": "read", "path": "shop/pricing.py", "turn": 7,
            "content_sha256": "b" * 64,
            "event_id": f"run:{sequence}", "sequence": sequence,
        }
        ledger.record_read_receipt("site-1", receipt)
        first = ledger._read_receipts["site-1"]["receipt_id"]
        if duplicate:
            ledger.record_read_receipt("site-1", receipt)
            return first, ledger._read_receipts["site-1"]["receipt_id"]
        return first

    first, duplicate = record(8, duplicate=True)
    observer_shifted = record(19)
    assert first == observer_shifted
    assert duplicate != first
    assert not first.startswith("run:")


def test_completion_root_budget_wire_id_is_seeded_and_resumes(monkeypatch):
    import kernel_wire_identity
    from kernel_completion_authority import RootFuelBudget

    monkeypatch.setenv("SF_LAB_DETFIX3", "on")

    class Journal:
        task_id = "task-46c84e9d"
        run_id = "seeded-run-1"
        manifest = {"run_identity_seed": "46" * 16}

    same_run = kernel_wire_identity.completion_root_budget_wire_id(Journal())
    replay = kernel_wire_identity.completion_root_budget_wire_id(Journal())

    class OtherRun(Journal):
        run_id = "seeded-run-2"
        manifest = {"run_identity_seed": "47" * 16}

    class OtherTask(Journal):
        task_id = "task-other"

    other_run = kernel_wire_identity.completion_root_budget_wire_id(OtherRun())
    other_task = kernel_wire_identity.completion_root_budget_wire_id(OtherTask())
    budget = RootFuelBudget(total=8, root_budget_id=same_run)
    restored = RootFuelBudget.from_snapshot(budget.snapshot())

    assert same_run == replay == restored.root_budget_id
    # A second run of the same task (new random seed and run id) sends the same bytes.
    assert same_run == other_run
    assert same_run != other_task
    assert re.fullmatch(r"[0-9a-f]{20}", same_run)


def _assert_same_send_hashes(first: list[str], second: list[str]) -> None:
    assert len(first) == len(second), f"send counts differ: {len(first)} != {len(second)}"
    assert first == second, "request hashes differ between repeated same-seed runs"


def _request_hash_for_receipt(receipt_id: str) -> str:
    request = {
        "messages": [{
            "role": "user",
            "content": json.dumps({"action_ledger": {"applied_edits": [{
                "action_kind": "edit", "edit_receipt": receipt_id,
                "result_class": "APPLIED",
            }]}}, sort_keys=True, separators=(",", ":")),
        }],
    }
    wire = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def test_repeated_run_guard_rejects_random_receipt_mint(monkeypatch):
    import kernel_wire_identity

    monkeypatch.setenv("SF_LAB_DETFIX3", "on")
    monkeypatch.setattr(
        kernel_wire_identity,
        "stable_wire_id",
        lambda *_parts: secrets.SystemRandom().getrandbits(80).to_bytes(10, "big").hex(),
    )

    class Journal:
        task_id = "task-46c84e9d"
        _sequence = 42
        manifest = {"run_identity_seed": "46" * 16}

    action = {
        "request_id": "674ce0515b391b5f235b",
        "step": 7,
        "tool_call_id": "call_7",
        "action_kind": "edit",
        "fingerprint": "b7d036c6b1d5129d",
        "target_path": "shop/pricing.py",
    }
    first_id = kernel_wire_identity.action_receipt_wire_id(Journal(), **action)
    second_id = kernel_wire_identity.action_receipt_wire_id(Journal(), **action)
    first = [_request_hash_for_receipt(first_id)]
    second = [_request_hash_for_receipt(second_id)]

    with pytest.raises(AssertionError, match="request hashes differ"):
        _assert_same_send_hashes(first, second)


def test_repeated_run_guard_rejects_sequence_based_receipt_mint(monkeypatch):
    import kernel_wire_identity

    monkeypatch.setenv("SF_LAB_DETFIX3", "on")
    monkeypatch.setattr(
        kernel_wire_identity, "action_receipt_wire_id",
        lambda journal, **_action: f"{journal._sequence:020x}",
    )

    class Journal:
        def __init__(self, sequence: int):
            self._sequence = sequence

    action = {
        "request_id": "674ce0515b391b5f235b",
        "step": 7,
        "tool_call_id": "call_7",
        "action_kind": "edit",
        "fingerprint": "b7d036c6b1d5129d",
        "target_path": "shop/pricing.py",
    }
    first_id = kernel_wire_identity.action_receipt_wire_id(Journal(42), **action)
    observer_shifted_id = kernel_wire_identity.action_receipt_wire_id(Journal(44), **action)
    first = [_request_hash_for_receipt(first_id)]
    observer_shifted = [_request_hash_for_receipt(observer_shifted_id)]

    with pytest.raises(AssertionError, match="request hashes differ"):
        _assert_same_send_hashes(first, observer_shifted)


def _walk_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _array_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _has_applied_edit_in_frame(request: object) -> bool:
    marker = re.compile(r'"applied_edits"\s*:\s*\[')
    edit_receipt = re.compile(r'"edit_receipt"\s*:\s*"[0-9a-f]{20}"')
    action_kind = re.compile(r'"action_kind"\s*:\s*"edit"')
    result_class = re.compile(r'"result_class"\s*:\s*"APPLIED"')
    for text in _walk_strings(request):
        for match in marker.finditer(text):
            start = match.end() - 1
            end = _array_end(text, start)
            if end is None:
                continue
            rows = text[start:end]
            if edit_receipt.search(rows) and action_kind.search(rows) and result_class.search(rows):
                return True
    return False


def test_applied_edit_frame_detector_requires_a_real_edit_row():
    request = {"messages": [{"content": json.dumps({"action_ledger": {"applied_edits": [{
        "action_kind": "edit", "edit_receipt": "a" * 20, "result_class": "APPLIED",
    }]}})}]}
    assert _has_applied_edit_in_frame(request)
    assert not _has_applied_edit_in_frame({"messages": [{"content": '{"applied_edits": []}'}]})


def _run_scripted_example(monkeypatch, example: str, task: str, work: Path,
                          *, observer: bool = False, package_root: Path | None = None):
    import os
    from engine_run import run_monitored
    import sys

    from stateframe import cli, runner
    from stateframe.scripted_endpoint import Script, serve

    work.mkdir(parents=True, exist_ok=True)
    repo = work / "repo"
    shutil.copytree(ROOT / "examples" / example, repo)
    run = work / "run"
    script = Script(json.loads((ROOT / "examples" / f"{example}.script.json").read_text()))
    server = serve(script)
    monkeypatch.setenv("STATEFRAME_FL_CHECKPOINT_CAPTURE", "on" if observer else "off")
    for name in (
            "STATEFRAME_FL_CAPTURE_EVERY_K", "STATEFRAME_FL_CAPTURE_MAX_COUNT",
            "STATEFRAME_FL_CAPTURE_MAX_FILES", "STATEFRAME_FL_CAPTURE_MAX_BYTES"):
        monkeypatch.delenv(name, raising=False)
    if observer:
        monkeypatch.setenv("STATEFRAME_FL_CAPTURE_EVERY_K", "1")
    monkeypatch.setattr(
        runner.secrets, "token_hex",
        lambda n=16: "0123456789abcdef0123456789abcdef"[: 2 * n],
    )
    arguments = [
        "run", "--repo", str(repo), "--task", task,
        "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
        "--model", "scripted", "--run-dir", str(run), "--quiet", "--json",
    ]
    try:
        if package_root is None:
            cli.main(arguments)
        else:
            # A separate interpreter imports the (planted) package copy, not this tree.
            environment = dict(os.environ, PYTHONPATH=str(package_root))
            completed = run_monitored(
                [sys.executable, "-m", "stateframe", *arguments], env=environment,
                capture_output=True, text=True, expected_seconds=600, check=False)
            assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    finally:
        server.shutdown()
        server.server_close()
    script.assert_consumed()

    sends_path = run / "sends.jsonl"
    sends = [json.loads(line) for line in sends_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    # request_sha256 is the exact serialized request hash recorded by the endpoint;
    # deliberately do not normalize IDs or volatile fields here.
    hashes = [record["request_sha256"] for record in sends]
    assert hashes, f"{example} produced no recorded sends"
    assert all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
               for value in hashes)
    import lzma

    capture_events = 0
    # Finished runs seal their journals as events.jsonl.xz; read live and sealed forms.
    journal_paths = [*run.rglob("*.jsonl"), *run.rglob("*.jsonl.xz")]
    assert journal_paths, f"{example} left no journal to inspect"
    for journal_path in journal_paths:
        raw = (lzma.decompress(journal_path.read_bytes()) if journal_path.suffix == ".xz"
               else journal_path.read_bytes())
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Any checkpoint-capture event is an extra journal record the plain run
            # lacks, which shifts every later journal position. Where capture cannot
            # anchor yet it still journals fl_checkpoint_capture_disabled early in the run.
            if isinstance(record, dict) and str(record.get("kind", "")).startswith("fl_checkpoint_"):
                capture_events += 1
    return hashes, list(script.requests), capture_events


def _cached_prefix_problems(requests: list) -> list[str]:
    """The system message must be byte-identical on every send of a run, and each send's tool
    list must begin with the previous send's list (the stable tool head only appends), so the
    cacheable prefix never changes between sends. Per-request IDs live in the last message."""
    def encoded(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)

    problems = []
    systems = [encoded(request["messages"][0]) for request in requests]
    if any(system != systems[0] for system in systems):
        problems.append(f"system message differs across sends: {len(set(systems))} versions")
    for index in range(1, len(requests)):
        before, after = requests[index - 1].get("tools") or [], requests[index].get("tools") or []
        if [encoded(tool) for tool in after[:len(before)]] != [encoded(tool) for tool in before]:
            problems.append(f"send {index + 1}: tool list is not an append-only extension of send {index}")
    return problems


def test_cached_prefix_guard_rejects_a_changed_system_message_or_reordered_tools():
    base = {"messages": [{"role": "system", "content": "fixed"}, {"role": "user", "content": "a"}],
            "tools": [{"name": "read"}, {"name": "edit"}]}
    grown = {**base, "messages": [base["messages"][0], {"role": "user", "content": "b"}],
             "tools": [*base["tools"], {"name": "run_tests"}]}
    assert _cached_prefix_problems([base, grown]) == []
    changed_system = {**grown, "messages": [{"role": "system", "content": "fixed!"}, *grown["messages"][1:]]}
    assert _cached_prefix_problems([base, changed_system])
    reordered = {**grown, "tools": [{"name": "edit"}, {"name": "read"}, {"name": "run_tests"}]}
    assert _cached_prefix_problems([base, reordered])


@pytest.mark.host_sandbox
def test_bundled_examples_send_identical_bytes_with_fixed_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("SF_LAB_DETFIX3", "on")
    saw_applied_edit = False
    for example, task in EXAMPLES.items():
        work = tmp_path / example
        first_hashes, first_requests, first_capture_events = _run_scripted_example(
            monkeypatch, example, task, work / "run-a")
        second_hashes, second_requests, second_capture_events = _run_scripted_example(
            monkeypatch, example, task, work / "run-b")
        observer_hashes, observer_requests, observer_capture_events = _run_scripted_example(
            monkeypatch, example, task, work / "run-o", observer=True)
        _assert_same_send_hashes(first_hashes, second_hashes)
        _assert_same_send_hashes(first_hashes, observer_hashes)
        assert len(first_requests) == len(first_hashes)
        assert len(second_requests) == len(second_hashes)
        assert len(observer_requests) == len(observer_hashes)
        # Two sends of one run share the cacheable prefix: same system message, and a tool
        # list that only grows. A second send in one step (its own request_id) included.
        assert len(first_requests) > 1
        assert _cached_prefix_problems(first_requests) == [], example
        assert first_capture_events == second_capture_events == 0
        assert observer_capture_events > 0, (
            f"{example} did not emit the observer checkpoint journal event"
        )
        saw_applied_edit = saw_applied_edit or any(
            _has_applied_edit_in_frame(request) for request in first_requests
        )
    assert saw_applied_edit, "no send rendered an APPLIED edit receipt in its request frame"


@pytest.mark.host_sandbox
def test_bundled_example_guard_rejects_sequence_based_receipt_derivation(tmp_path, monkeypatch):
    monkeypatch.setenv("SF_LAB_DETFIX3", "on")
    example, task = "discount_bug", EXAMPLES["discount_bug"]
    plain_hashes, plain_requests, _plain_events = _run_scripted_example(
        monkeypatch, example, task, tmp_path / "run-a")
    observer_hashes, observer_requests, observer_events = _run_scripted_example(
        monkeypatch, example, task, tmp_path / "run-o", observer=True)

    assert observer_events > 0, "the checkpoint observer injection did not run"
    assert _has_applied_edit_in_frame(plain_requests)
    assert _has_applied_edit_in_frame(observer_requests)
    _assert_same_send_hashes(plain_hashes, observer_hashes)

    # Planted control on the real engine path: a package copy whose action
    # receipt IDs also hash the global journal sequence (the old derivation).
    # The same plain-vs-observer comparison must then fail, or the guard above
    # cannot see the bug class it exists for.
    planted = _planted_sequence_receipt_package(tmp_path / "planted-pkg")
    seq_plain, seq_plain_requests, _ = _run_scripted_example(
        monkeypatch, example, task, tmp_path / "pln-a", package_root=planted)
    seq_observer, _, seq_observer_events = _run_scripted_example(
        monkeypatch, example, task, tmp_path / "pln-o", observer=True,
        package_root=planted)
    assert seq_observer_events > 0, "the planted observer injection did not run"
    assert _has_applied_edit_in_frame(seq_plain_requests), "planted run made no edit to receipt"
    with pytest.raises(AssertionError, match="request hashes differ"):
        _assert_same_send_hashes(seq_plain, seq_observer)


_PLANT_ANCHOR = '            "fingerprint": fingerprint,\n        }, ordinal=ordinal,'


def _planted_sequence_receipt_package(destination: Path) -> Path:
    """Copy the package and make action receipt IDs depend on the journal sequence."""
    package = destination / "stateframe"
    shutil.copytree(ROOT / "src" / "stateframe", package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    target = package / "engine" / "kernel_wire_identity.py"
    source = target.read_text(encoding="utf-8")
    assert source.count(_PLANT_ANCHOR) == 1, "planted-control anchor moved; update the plant"
    target.write_text(source.replace(_PLANT_ANCHOR, (
        '            "fingerprint": fingerprint,\n'
        '            "journal_sequence": getattr(journal, "_sequence", 0),\n'
        '        }, ordinal=ordinal,')), encoding="utf-8")
    return destination


def test_bundled_example_planted_control_anchor_exists():
    """The planted control patches a real line; a moved line must fail loudly, not skip."""
    source = (ROOT / "src/stateframe/engine/kernel_wire_identity.py").read_text(encoding="utf-8")
    assert source.count(_PLANT_ANCHOR) == 1
