"""Owner inventory reuse preserves the facts consumed by request sidecars."""
import copy
import hashlib
import sys
from types import SimpleNamespace

import pytest

import kernel_state_owner_inventory as owners
import kernel_tool_tape
import kernel_unknowns_ledger as ledger
from kernel_catalog import replay
from kernel_stateframe_runtime import StateFrameRuntime
from repo_intel_core import ts_dispatch
import stateframe


FIELD = "answer_meta"
SOURCE = """class Answer:
    answer_meta: dict = {}
    pending_meta: dict = {}

    def read(self):
        return self.answer_meta
"""
WRITER = "# sibling outside the editable scope\n\nanswer.answer_meta = {}\n"


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "on")
    monkeypatch.setenv(owners.ENV, "on")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_UNKNOWNS_LEDGER", "on")
    monkeypatch.setenv("STATEFRAME_ROW_ROLE", "on")
    monkeypatch.setenv("STATEFRAME_LIFECYCLE_TABLE", "on")
    monkeypatch.setenv("STATEFRAME_TIMING_TIMELINE", "off")
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    monkeypatch.setattr(owners, "_MEMO", owners.OrderedDict())
    records = {}
    accesses = []

    def lookup(kind, fingerprint):
        accesses.append(("lookup", kind))
        return copy.deepcopy(records.get(fingerprint))

    def publish(kind, fingerprint, facts):
        accesses.append(("publish", kind))
        records[fingerprint] = copy.deepcopy(facts)

    store = SimpleNamespace(active=True, records=records, accesses=accesses,
                            lookup=lookup, publish=publish)
    store.enabled = lambda: store.active
    monkeypatch.setitem(sys.modules, "stateframe.send1_probe_cache", store)
    monkeypatch.setattr(stateframe, "send1_probe_cache", store, raising=False)
    monkeypatch.setitem(sys.modules, "stateframe.workspace_start_snapshot",
                        SimpleNamespace(startup_observation=lambda *_: None))
    return store


def _runtime(root, *, contexts=False):
    runtime = object.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=root, task_row={
        "problem_statement": "Keep `answer_meta` current. Retry on failure.",
        "public_baseline": "PASS",
        "files": {"answer.py": {"parent_src": SOURCE}},
    })
    runtime.events = []
    runtime.journal = SimpleNamespace(emit=lambda kind, **row: runtime.events.append((kind, row)))
    runtime._site_fact_blocks = lambda: []
    runtime.stateframe_site_facts_mode = "on"
    runtime._frame_budget_request_count = int(contexts)
    runtime.binding = lambda: None
    runtime._catalog_task_scope = lambda: ("task", "required")
    runtime._catalog_sync = lambda: replay(ledger.observe(runtime, task_id="task", scope="required"))
    return runtime


def _observe_files(root, purpose):
    assert purpose == "state-owner-inventory"
    members = sorted(path for path in root.rglob("*") if path.is_file())
    digests = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in members}
    return SimpleNamespace(
        files=digests,
        modes={path.relative_to(root).as_posix(): path.stat().st_mode for path in members},
        exclusion_digest="exclusions", overlay_digest="overlays",
        validate_for_use=lambda selected: selected == purpose and _observe_files(root, purpose).files == digests,
    )


def _prepare(runtime):
    messages = runtime._apply_rulebook_mechanisms([], provider_request=True)
    receipts = ledger.observe(runtime, task_id="task", scope="required")
    return messages, runtime._state_owner_inventory, receipts, copy.deepcopy(runtime.events)


def _count(monkeypatch, module, name, calls):
    original = getattr(module, name)

    def counted(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)


@pytest.mark.parametrize("option_a", ["off", "on"])
def test_inventory_hit_skips_build_and_preserves_first_request_and_closure(
        tmp_path, monkeypatch, pilot, option_a):
    monkeypatch.setenv("STATEFRAME_SEND1_DEFER_TOOL_PREP", option_a)
    (tmp_path / "answer.py").write_text(SOURCE)
    calls = []
    for name in ("_read", "_outline", "build_inventory", "_field_contexts"):
        _count(monkeypatch, owners, name, calls)

    monkeypatch.setattr(sys.modules["stateframe.workspace_start_snapshot"],
                        "startup_observation", _observe_files)

    eager = _runtime(tmp_path)
    pilot.active = False
    expected = _prepare(eager)
    assert not pilot.accesses
    pilot.active = True
    calls.clear()
    cold = _prepare(_runtime(tmp_path))
    assert cold == expected
    assert calls == ["_read", "_outline", "build_inventory"]
    assert len(pilot.records) == 1
    messages, inventory, receipts, _events = cold
    assert inventory.complete and not inventory.fields[0].contexts
    unknown = next(message["content"] for message in messages
                   if message["content"].startswith("[OPEN UNKNOWNS]"))
    assert "advisory state.writer `answer_meta`: 0 writers in product code (answer.py:2)" in unknown
    assert receipts[0]["revision"] == inventory.revision

    # Drop process-local data to exercise the same detached lookup as another launch.
    owners._MEMO.clear()
    warm = _runtime(tmp_path)
    calls.clear()
    first = _prepare(warm)
    assert first == cold
    assert not calls
    first[1].fields[0].readers.clear()
    assert _prepare(_runtime(tmp_path)) == cold

    (tmp_path / "sibling.py").write_text(WRITER)
    warm._frame_budget_request_count = eager._frame_budget_request_count = 1
    after_edit = _prepare(warm)
    pilot.active = False
    assert after_edit == _prepare(eager)
    updated = after_edit[1]
    assert updated.complete and updated.fields[0].writers == [owners.Site(
        "sibling.py", 3, "answer.answer_meta = {}",
    )]
    assert updated.fields[0].contexts == [
        owners.Site("answer.py", 5, "def read(self):"),
        owners.Site("answer.py", 6, "return self.answer_meta"),
    ]
    assert all("state.writer `answer_meta`" not in row["content"] for row in after_edit[0])
    closure = after_edit[2][-1]
    assert closure["resolution_scope"]["supersedes_obsolete"]
    assert closure["revision"] == updated.revision

    # Replaying the old send rebuilds its inputs, then the replayed edit closes
    # the same advisory with the same revision and receipt binding.
    (tmp_path / "sibling.py").unlink()
    pilot.active = True
    resumed = _runtime(tmp_path)
    monkeypatch.setattr(kernel_tool_tape, "replay_mode_active", lambda: True)
    pilot.accesses.clear()
    assert _prepare(resumed) == cold
    (tmp_path / "sibling.py").write_text(WRITER)
    resumed._frame_budget_request_count = 1
    assert _prepare(resumed) == after_edit
    assert not pilot.accesses


def test_mutations_revalidate_membership_candidates_parser_and_failures(tmp_path, monkeypatch, pilot):
    (tmp_path / "answer.py").write_text(SOURCE)
    extra = tmp_path / "sibling.py"
    calls = []
    _count(monkeypatch, owners, "build_inventory", calls)

    def compare(*, symbol="", status="FOUND", contexts=False, ticket=None, parent=SOURCE):
        active = _runtime(tmp_path, contexts=contexts)
        active.owner.fn = symbol
        active._site_facts_entry = {"symbol": symbol, "status": status, "revision": "entry"}
        row = active.owner.task_row
        if ticket:
            row["problem_statement"] = ticket
        row["files"]["answer.py"]["parent_src"] = parent
        expected_runtime = _runtime(tmp_path, contexts=contexts)
        expected_runtime.owner.fn = symbol
        expected_runtime.owner.task_row = row
        expected_runtime._site_facts_entry = active._site_facts_entry
        result = owners.runtime_inventory(active)
        with monkeypatch.context() as disabled:
            disabled.setattr(pilot, "active", False)
            assert result == owners.runtime_inventory(expected_runtime)
        return result

    before = compare()
    assert compare() == before
    assert len(calls) == 3  # one build on the pilot path and two eager references
    extra.write_text("class External:\n    answer_meta: dict = {}\n\n" + WRITER)
    added = compare()
    assert [(field.path, field.line) for field in added.fields] == [
        ("answer.py", 2), ("sibling.py", 2),
    ]
    assert all(field.writers == [owners.Site("sibling.py", 6, "answer.answer_meta = {}")]
               for field in added.fields)
    extra.unlink()
    assert compare() == before

    assert compare(symbol=FIELD) is None
    reassigned = compare(symbol="Answer.read", status="NOT_FOUND", contexts=True)
    assert reassigned.fields[0].contexts
    assert reassigned.candidates != before.candidates
    one = compare(ticket="Keep `answer_meta` and `pending_meta`.")
    two = compare(ticket="Keep `pending_meta` and `answer_meta`.")
    assert one.candidates == tuple(reversed(two.candidates))
    rebound = compare(parent=SOURCE + "# older parent\n")
    assert rebound.dependency_revision == before.dependency_revision
    assert rebound.identities["answer.py"]["status"] == "CHANGED"

    extra.write_bytes(b"\xff")
    invalid = compare()
    assert not invalid.complete
    assert invalid.parser_capabilities["sibling.py"] == "UNAVAILABLE"
    assert invalid.skipped_files[0]["reason"] == "file is unreadable or exceeds bounded size"
    extra.write_text("def broken(\n")
    invalid = compare()
    assert not invalid.complete and invalid.parser_capabilities["sibling.py"] == "DEGRADED"
    extra.unlink()
    assert compare() == before

    class Parser:
        def parse(self, _source):
            return SimpleNamespace(root_node=SimpleNamespace(type="module", children=()))

    pack = SimpleNamespace(get_parser=lambda _: Parser(), __version__="changed")
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", pack)
    reports = []
    token = ts_dispatch.bind_outline_capability_listener(lambda *args: reports.append(args))
    try:
        assert compare().complete
        assert ("python", "tree-sitter", "AVAILABLE", "KNOWN") in reports
    finally:
        ts_dispatch.reset_outline_capability_listener(token)


@pytest.mark.parametrize("failure", ["malformed", "unavailable"])
def test_bad_store_falls_back_to_current_inventory(tmp_path, monkeypatch, pilot, failure):
    (tmp_path / "answer.py").write_text(SOURCE)
    expected = owners.runtime_inventory(_runtime(tmp_path))
    owners._MEMO.clear()
    calls = []
    _count(monkeypatch, owners, "build_inventory", calls)

    def lookup(*_args):
        if failure == "unavailable":
            raise OSError("cache unavailable")
        return {"inventory": '{"dependency_revision":[],"fields":[],"helpers":[],"identities":{}}'}

    monkeypatch.setattr(pilot, "lookup", lookup)
    assert owners.runtime_inventory(_runtime(tmp_path)) == expected
    assert calls == ["build_inventory"]


def test_owner_inventory_rechecks_workspace_after_warm_lookup(tmp_path, monkeypatch, pilot):
    (tmp_path / "answer.py").write_text(SOURCE)
    monkeypatch.setattr(sys.modules["stateframe.workspace_start_snapshot"],
                        "startup_observation", _observe_files)
    initial = owners.runtime_inventory(_runtime(tmp_path))
    assert not initial.fields[0].writers
    assert pilot.records
    owners._MEMO.clear()

    read = pilot.lookup

    def lookup(*args):
        record = read(*args)
        if record is not None:
            (tmp_path / "sibling.py").write_text(WRITER)
        return record

    monkeypatch.setattr(pilot, "lookup", lookup)
    observed = owners.runtime_inventory(_runtime(tmp_path))
    assert [site.path for site in observed.fields[0].writers] == ["sibling.py"]
    pilot.active = False
    assert observed == owners.runtime_inventory(_runtime(tmp_path))


def test_startup_scan_matches_authoritative_cap_and_filters(tmp_path, monkeypatch, pilot):
    names = ("child.py", "a/deep/child.py", "a/z.py", "b/a.py", "tests/test_child.py",
             ".hidden/child.py", "vendor/child.py", "vendor.py", "z.text")
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True, parents=True)
        target.touch()
    observed = SimpleNamespace(files=dict.fromkeys(reversed(names)))
    monkeypatch.setattr(sys.modules["stateframe.workspace_start_snapshot"], "startup_observation",
                        lambda *_: observed)
    monkeypatch.setattr(owners, "_MAX_FILES", 3)
    actual, proof = owners._startup_source_scan(tmp_path)
    assert proof is observed
    assert actual == owners.scan_product_files(tmp_path, limit=3)


def test_admitted_cutoff_preserves_partial_scan_on_hit(tmp_path, monkeypatch, pilot):
    (tmp_path / "answer.py").write_text(SOURCE)
    huge = tmp_path / "oversized.py"
    huge.write_bytes(b" " * (owners._MAX_FILE_BYTES + 1))
    (tmp_path / "tail.py").write_text(WRITER)

    monkeypatch.setattr(sys.modules["stateframe.workspace_start_snapshot"],
                        "startup_observation", _observe_files)
    original_scan = owners.scan_product_files
    monkeypatch.setattr(owners, "scan_product_files", lambda root: original_scan(root, limit=2))
    monkeypatch.setattr(owners, "_MAX_FILES", 2)
    calls = []
    for name in ("_read", "_outline", "build_inventory"):
        _count(monkeypatch, owners, name, calls)

    cold = owners.runtime_inventory(_runtime(tmp_path))
    assert cold.files_scanned == 1 and not cold.complete
    assert [(row["path"], row["status"]) for row in cold.skipped_files] == [
        ("tail.py", "PARTIAL"), ("oversized.py", "UNKNOWN"),
    ]
    calls.clear()
    assert owners.runtime_inventory(_runtime(tmp_path)) == cold
    assert not calls

    huge.write_text(WRITER)
    changed = owners.runtime_inventory(_runtime(tmp_path))
    assert changed.files_scanned == 2 and not changed.complete
    assert changed.fields[0].writers[0].path == "oversized.py"
    assert calls[-1] == "build_inventory"


@pytest.mark.parametrize("direct_child", [False, True])
def test_audited_adapter_without_python_grammar_can_cache_ast(
        tmp_path, monkeypatch, pilot, direct_child):
    from importlib.util import module_from_spec, spec_from_file_location

    from stateframe import ts_grammars

    (tmp_path / "answer.py").write_text(SOURCE)
    adapter = ts_grammars
    if direct_child:
        spec = spec_from_file_location("ts_grammars", ts_grammars.__file__)
        adapter = module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "ts_grammars", adapter)
        spec.loader.exec_module(adapter)
    monkeypatch.setitem(sys.modules, "tree_sitter_python", None)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", SimpleNamespace(
        __stateframe_grammar_adapter__=True,
        get_parser=adapter.get_parser,
        get_language=adapter.get_language,
    ))
    calls = []
    _count(monkeypatch, owners, "build_inventory", calls)
    expected = owners.runtime_inventory(_runtime(tmp_path))
    assert expected.complete and len(pilot.records) == 1
    owners._MEMO.clear()
    assert owners.runtime_inventory(_runtime(tmp_path)) == expected
    assert calls == ["build_inventory"]
