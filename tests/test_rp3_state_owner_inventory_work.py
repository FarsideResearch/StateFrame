"""State-owner inventory work and behavior on a fenced workspace."""
from __future__ import annotations

import sys
from collections import Counter
from time import perf_counter
from types import SimpleNamespace

import pytest

import kernel_state_owner_inventory as owners
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_feed import WorkspaceFeed
from kernel_workspace_state import WorkspaceFence, WorkspaceState

FIELD = "publication_state"
OWNER_SOURCE = """class Publisher:
    publication_state: int = 0

    def publish(self):
        self.publication_state = 1
        return self.publication_state
"""
TURN_SWITCHES = (
    "ATOMIC_PUBLICATION", "BOUNDED_QUALIFICATION", "DELIVERY_PROOFS",
    "GENERATION_OVERLAP", "INDEXED_READS", "MINIMAL_LAUNCH", "PANELS", "REPO_INTEL",
    "TIMING",
)


def _switches(monkeypatch, value):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", value)
    monkeypatch.setenv(owners.ENV, "on")
    for switch in TURN_SWITCHES:
        monkeypatch.setenv(f"STATEFRAME_TURN_{switch}", value)


def _workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"),
                           feed=WorkspaceFeed(root, enabled=True))
    consumers = WorkspaceConsumers(state)
    bind_consumers(root, consumers)
    return root, consumers


def _runtime(root, parent):
    return SimpleNamespace(owner=SimpleNamespace(root=root, task_row={
        "problem_statement": f"Keep `{FIELD}` current.",
        "files": {path: {"parent_src": source} for path, source in parent.items()},
    }))


@pytest.mark.parametrize("pinned", [False, True])
def test_initial_rulebook_defers_owner_context_and_full_outline(tmp_path, monkeypatch, pinned):
    import kernel_parse_cache
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_unknowns_ledger import observe
    from repo_intel_core import ts_dispatch

    _switches(monkeypatch, "on" if pinned else "off")
    monkeypatch.setenv("STATEFRAME_UNKNOWNS_LEDGER", "on")
    monkeypatch.setattr(ts_dispatch, "_parse_source", lambda *_args: None)
    root, _consumers = _workspace(tmp_path)
    source = "CONSTANT = 1\n" + OWNER_SOURCE.replace(
        "    def publish", "    publication_state_meta: int = 0\n\n    def publish")
    other = "CONSTANT = 2\n"
    (root / "publisher.py").write_text(source)
    (root / "other.py").write_text(other)
    runtime = _runtime(root, {})
    runtime.owner.task_row["problem_statement"] = (
        f"Keep `{FIELD}` current. Refresh it on failure.")
    runtime.stateframe_site_facts_mode = "on"
    runtime._site_fact_blocks = list
    runtime._frame_budget_request_count = 0

    work = Counter()
    kernel_parse_cache.clear()
    kernel_parse_cache.parse(source)
    real_parse = kernel_parse_cache.ast.parse

    def counted_parse(text, *args, **kwargs):
        if text in (source, other):
            work["parse"] += 1
        return real_parse(text, *args, **kwargs)

    def count(module, name):
        original = getattr(module, name)

        def counted(*args, **kwargs):
            work[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)

    monkeypatch.setattr(kernel_parse_cache.ast, "parse", counted_parse)
    count(owners, "_field_contexts")
    count(ts_dispatch, "_ast_python_outline")
    count(ts_dispatch, "_python_ast_module_record")
    first = StateFrameRuntime._apply_rulebook_mechanisms(runtime, [], provider_request=True)
    assert work == {"parse": 1, "_ast_python_outline": 2}
    inventory = runtime._state_owner_inventory
    assert inventory.writer_count == 1
    assert not inventory.fields[0].contexts
    assert any(message["content"].startswith(owners.HEADER) for message in first)
    assert any(message["content"].startswith("[LIFECYCLE STATE TABLE") for message in first)
    receipts = observe(runtime, task_id="task", scope="parent")
    assert [(row["predicate"], row["subject"]["locator"])
            for receipt in receipts for row in receipt["unknowns"]] == [
        ("state.writer", "publication_state_meta"),
    ]

    runtime._frame_budget_request_count = 1
    later = StateFrameRuntime._apply_rulebook_mechanisms(runtime, [], provider_request=True)
    assert first == later
    assert work == {"parse": 1, "_ast_python_outline": 2, "_field_contexts": 1}
    assert [site.text for site in runtime._state_owner_inventory.fields[0].contexts] == [
        "def publish(self):", "return self.publication_state",
    ]

    outline = ts_dispatch.outline_source(source, "publisher.py")
    assert work["parse"] == 1
    assert outline["functions"][0]["signature"] == "def publish(self):"
    assert outline["module_statements"][0]["source"] == "CONSTANT = 1"


@pytest.mark.parametrize("objects", ["off", "on"])
def test_empty_owner_candidates_report_parser_before_verdict(tmp_path, monkeypatch, objects):
    from kernel_continuation_journal import ContinuationJournal

    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", objects)
    monkeypatch.setenv(owners.ENV, "on")
    root, _consumers = _workspace(tmp_path)
    (root / "pricing.py").write_text("price = 10\n")
    runtime = _runtime(root, {})
    ticket = "Fix the pricing code so the tests pass."
    runtime.owner.task_row["problem_statement"] = ticket
    assert not owners.owner_candidates(ticket)

    with ContinuationJournal(tmp_path, "pricing", {}) as journal:
        start = journal.emit("import_smoke")["sequence"]
        assert owners.runtime_inventory(runtime) is None
        assert owners.runtime_inventory(runtime) is None  # warm observation
        journal.emit("public_test_verdict_fact_normalized")
        events = [row for row in journal.records() if row["sequence"] >= start]

    assert [row["kind"] for row in events] == [
        "import_smoke", "outline_parser_capability", "public_test_verdict_fact_normalized",
    ]
    assert events[1]["language"] == "python"


def test_inventory_work_on_598_sources(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    root, consumers = _workspace(tmp_path)
    parent = {"publisher.py": OWNER_SOURCE + "# parent publisher\n"}
    (root / "publisher.py").write_text(OWNER_SOURCE, encoding="utf-8")
    for index in range(597):
        path = f"module_{index:04d}.py"
        (root / path).write_text(f"VALUE = {index}\n# " + "x" * 12_000 + "\n",
                                 encoding="utf-8")
        parent[path] = f"VALUE = {index}\n# parent {index}\n"
    for index in range(3902):
        (root / f"asset_{index:04d}.txt").touch()
    runtime = _runtime(root, parent)
    started = perf_counter()
    with consumers.state.pin("admitted-workspace"):
        pass
    print(f"shared workspace admission: {perf_counter() - started:.6f}s")
    counts = Counter()
    real_read = owners._read
    real_outline = owners._outline
    real_sha256 = owners.hashlib.sha256
    parent_bytes = {source.encode() for source in parent.values()}

    def counted_read(*args, **kwargs):
        counts["source_reads"] += 1
        return real_read(*args, **kwargs)

    def counted_outline(*args, **kwargs):
        counts["outlines"] += 1
        return real_outline(*args, **kwargs)

    def counted_hash(data=b"", *args, **kwargs):
        if data in parent_bytes:
            counts["parent_hashes"] += 1
        return real_sha256(data, *args, **kwargs)

    monkeypatch.setattr(owners, "_read", counted_read)
    monkeypatch.setattr(owners, "_outline", counted_outline)
    monkeypatch.setattr(owners, "hashlib", SimpleNamespace(sha256=counted_hash))

    def observe(label):
        counts.clear()
        started = perf_counter()
        inventory = owners.runtime_inventory(runtime)
        elapsed = perf_counter() - started
        work = counts.copy()
        print(f"{label}: {elapsed:.6f}s {dict(work)}")
        return inventory, work, elapsed

    cold, cold_work, _cold_seconds = observe("cold")
    warm_a, warm_a_work, warm_a_seconds = observe("warm 1")
    warm_b, warm_b_work, warm_b_seconds = observe("warm 2")
    with consumers.state.begin_write(("module_0000.py",)):
        (root / "module_0000.py").write_text(
            "def update(publisher):\n    publisher.publication_state = 2\n", encoding="utf-8")
    edited, edit_work, edit_seconds = observe("one edit")
    assert cold is not None and cold.files_scanned == 598 and cold.writer_count == 1
    assert warm_a == warm_b == cold
    assert edited is not None and edited.writer_count == 2
    assert cold_work["source_reads"] == cold_work["outlines"] == 598
    assert cold_work["parent_hashes"] == 598
    assert not warm_a_work and not warm_b_work
    assert edit_work["source_reads"] <= 1
    assert edit_work["outlines"] <= 1
    assert edit_work["parent_hashes"] == 0
    assert max(edit_seconds, warm_a_seconds, warm_b_seconds) < 2


def _assert_parity(runtime, monkeypatch):
    actual = owners.runtime_inventory(runtime)
    with monkeypatch.context() as context:
        _switches(context, "off")
        legacy = SimpleNamespace(owner=runtime.owner)
        expected = owners.runtime_inventory(legacy)
        assert owners.owner_writer_count(legacy) == (
            expected.writer_count if expected is not None and expected.complete else None
        )
    assert actual == expected
    if actual is not None:
        assert owners.render(actual).encode() == owners.render(expected).encode()
    return actual


def test_inventory_tracks_membership_ticket_and_parent(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    root, consumers = _workspace(tmp_path)
    # Also check that pinned byte hashes preserve text-mode newline normalization.
    (root / "publisher.py").write_bytes(OWNER_SOURCE.replace("\n", "\r\n").encode())
    runtime = _runtime(root, {"publisher.py": OWNER_SOURCE})
    before = _assert_parity(runtime, monkeypatch)
    assert before.identities["publisher.py"]["status"] == "UNCHANGED"
    assert owners.owner_writer_count(runtime) == 1

    with consumers.state.begin_write(None):
        (root / "writers").mkdir()
        (root / "writers" / "update.py").write_text(
            "def update(publisher):\n    publisher.publication_state = 2\n")
        (root / "tests").mkdir()
        (root / "tests" / "test_update.py").write_text("publisher.publication_state = 3\n")
    # The authorization consumer must see the new generation before a panel refresh.
    assert owners.owner_writer_count(runtime) == 2
    added = _assert_parity(runtime, monkeypatch)
    assert added.files_scanned == 2
    assert added.identities["writers/update.py"]["status"] == "UNKNOWN"

    runtime.owner.task_row["files"]["publisher.py"]["parent_src"] = OWNER_SOURCE + "# parent\n"
    rebound = _assert_parity(runtime, monkeypatch)
    assert rebound.dependency_revision == added.dependency_revision
    assert rebound.identities["publisher.py"]["status"] == "CHANGED"
    assert rebound.identities["publisher.py"]["parent_revision"] != (
        added.identities["publisher.py"]["parent_revision"]
    )

    runtime.owner.task_row["problem_statement"] = "Keep `other_state` current."
    assert _assert_parity(runtime, monkeypatch) is None
    runtime.owner.task_row["problem_statement"] = f"Keep `{FIELD}` current."
    with consumers.state.begin_write(("writers/update.py",)):
        (root / "writers" / "update.py").unlink()
    deleted = _assert_parity(runtime, monkeypatch)
    assert deleted.writer_count == 1
    assert set(deleted.identities) == {"publisher.py"}


def test_parser_policy_change_rechecks_writer_authority(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    root, _consumers = _workspace(tmp_path)
    (root / "publisher.py").write_text(OWNER_SOURCE)
    (root / "reader.ts").write_text('const TAG = "publication_state";\n')
    runtime = _runtime(root, {})
    grammar_available = False

    class Parser:
        def parse(self, _source):
            return SimpleNamespace(root_node=SimpleNamespace(type="program", children=()))

    def get_parser(language):
        if language == "typescript" and grammar_available:
            return Parser()
        raise LookupError(language)

    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", SimpleNamespace(
        get_parser=get_parser, available_languages=lambda: [],
    ))
    incomplete = _assert_parity(runtime, monkeypatch)
    assert incomplete.parser_capabilities["reader.ts"] == "DEGRADED"
    assert incomplete.writer_count == 1
    assert owners.owner_writer_count(runtime) is None

    grammar_available = True
    complete = _assert_parity(runtime, monkeypatch)
    assert complete.complete and complete.parser_capabilities["reader.ts"] == "AVAILABLE"
    assert owners.owner_writer_count(runtime) == 1
    grammar_available = False
    assert owners.owner_writer_count(runtime) is None


def test_inventory_marks_truncated_or_unavailable_scope_unknown(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    root, consumers = _workspace(tmp_path)
    (root / "a.py").write_text(OWNER_SOURCE)
    (root / "b.py").write_text("publisher.publication_state = 2\n")
    runtime = _runtime(root, {})
    monkeypatch.setattr(owners, "_MAX_FILES", 1)
    partial = owners.runtime_inventory(runtime)
    legacy_scan = owners.scan_product_files(root, limit=1)
    assert partial == owners.build_inventory(root, partial.candidates, scan=legacy_scan)
    assert partial.writer_count == 1
    assert owners.owner_writer_count(runtime) is None

    with consumers.state.begin_write(("b.py",)):
        (root / "b.py").unlink()
    assert owners.owner_writer_count(runtime) == 1
    root.rename(tmp_path / "displaced")
    root.mkdir()
    assert owners.owner_writer_count(runtime) is None
    unavailable = owners.runtime_inventory(runtime)
    assert not unavailable.complete and not unavailable.fields
    runtime.owner.task_row["problem_statement"] = "Fix the pricing code so the tests pass."
    assert owners.runtime_inventory(runtime) is None
