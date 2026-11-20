"""Selection caching preserves source-store results and skips repeated allocation."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from types import SimpleNamespace

import pytest

from kernel_source_evidence_store import SourceEvidenceStore
from kernel_stateframe_runtime import StateFrameRuntime


def _revision(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_pair(path: str, content: str, call_id: str):
    revision = _revision(content)
    end_line = len(content.splitlines())
    header = (f"[read page] path={path} revision={revision} lines=1-{end_line}/{end_line} "
              f"chars={len(content)} truncated=false decoded_with_replacement=false")
    assistant = {"role": "assistant", "tool_calls": [{
        "id": call_id,
        "type": "function",
        "function": {"name": "read", "arguments": json.dumps({"path": path})},
    }]}
    result = {"role": "tool", "tool_call_id": call_id,
              "content": f"{header}\nnotice\n{content}"}
    return [assistant, result], revision


def _make_store(monkeypatch, page_rows):
    monkeypatch.setenv("STATEFRAME_EVIDENCE_DELIVERY", "off")
    events = []
    store = SourceEvidenceStore(lambda name, **fields: events.append((name, fields)))
    current = {}
    for index, (path, content) in enumerate(page_rows):
        pair, revision = _read_pair(path, content, f"read-{index}")
        assert store.remember_exchange(pair)
        current[path] = revision
    return store, current, events


def _count_select_calls(monkeypatch, store):
    original = store.working_set.select
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store.working_set, "select", counted)
    return lambda: calls


def _uncached_select(store, *args, **kwargs):
    store._selection_key = None
    store._selection_cache = None
    return store.select(*args, **kwargs)


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _source_runtime(store, current, root):
    runtime = object.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=root)
    runtime._f4_exchange_ledger = None
    runtime._f4_source_store = store
    runtime.artifact_hashes = current
    runtime.retained_evidence = ()
    runtime.working_source_scope = False
    runtime.frame_clarity_mode = "off"
    runtime.stateframe_source_obligations_mode = "off"
    runtime.stateframe_evidence_delivery_mode = "off"
    runtime.journal = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    return runtime


def test_repeated_select_reuses_selection_and_keeps_deepcopy_semantics(monkeypatch):
    rows = [(f"pkg/mod_{i}.py", f"def item_{i}():\n    return {i}\n") for i in range(4)]
    store, current, _events = _make_store(monkeypatch, rows)
    calls = _count_select_calls(monkeypatch, store)

    first, first_state = store.select(current, "target.py", "working source")
    expected = copy.deepcopy((first, first_state))
    first[0]["receipt_digests"][0]["receipt_id"] = "mutated"
    first_state["active"].clear()

    cached = store.select(current, "target.py", "working source")

    assert calls() == 1
    assert cached == expected


def test_revision_change_invalidates_and_reselects(monkeypatch):
    store, current, _events = _make_store(
        monkeypatch, [("pkg/changed.py", "def original():\n    return 1\n")])
    calls = _count_select_calls(monkeypatch, store)

    first, _ = store.select(current, "target.py", "")
    changed = dict(current, **{"pkg/changed.py": "sha256:" + "0" * 64})
    second, _ = store.select(changed, "target.py", "")

    assert first
    assert second == []
    assert calls() == 2
    assert store.pages == {}
    assert store.evidence == {}
    assert store.working_set.spans == {}


@pytest.mark.parametrize("stale_area", ("evidence", "archive", "working_set"))
def test_cache_misses_when_invalidation_would_remove_related_state(monkeypatch, stale_area):
    store, current, _events = _make_store(
        monkeypatch, [("pkg/current.py", "def current():\n    return True\n")])
    calls = _count_select_calls(monkeypatch, store)
    store.select(current, "target.py", "")
    stale_path = "pkg/stale.py"
    stale_revision = "sha256:" + "f" * 64
    working_set_events = []
    store.working_set.emit = lambda name, **fields: working_set_events.append((name, fields))

    if stale_area == "evidence":
        key = (stale_path, stale_revision, 1, 1, "content-hash", "receipt")
        store.evidence[key] = {"path": stale_path, "revision": stale_revision,
                              "recency": 0}
    elif stale_area == "archive":
        key = (stale_path, stale_revision, 1, 1)
        archived = {"path": stale_path, "revision": stale_revision,
                    "content": "stale", "demand": 0}
        store.archive[key] = archived
        store._archive_bytes = len(b"stale")
    else:
        span = copy.deepcopy(next(iter(store.working_set.spans.values())))
        span.update(span_id="stale-span", path=stale_path, revision=stale_revision)
        store.working_set.spans["stale-span"] = span

    store.select(current, "target.py", "")

    assert calls() == 2
    if stale_area == "evidence":
        assert key not in store.evidence
    if stale_area == "archive":
        assert store.archive == {}
        assert store._archive_bytes == 0
    if stale_area == "working_set":
        assert all(span["path"] != stale_path for span in store.working_set.spans.values())
        assert any(name == "read_working_set_removed"
                   for name, _fields in working_set_events)


def test_mandatory_path_roles_are_reapplied_on_a_cache_hit(monkeypatch):
    path = "pkg/required.py"
    store, current, _events = _make_store(
        monkeypatch, [(path, "def required():\n    return True\n")])
    store.required_source_paths = (path,)
    select_calls = _count_select_calls(monkeypatch, store)
    original_set_owner = store.working_set.set_owner
    role_updates = 0

    def count_role_updates(*args, **kwargs):
        nonlocal role_updates
        role_updates += 1
        return original_set_owner(*args, **kwargs)

    monkeypatch.setattr(store.working_set, "set_owner", count_role_updates)
    first, _ = store.select(current, "target.py", "")
    second, _ = store.select(current, "target.py", "")

    row = next(iter(store.pages.values()))
    span = next(iter(store.working_set.spans.values()))
    assert select_calls() == 1
    assert role_updates == 2
    assert row["owner_role"] == "required_caller"
    assert row["owner_roles"] == ("required_caller",)
    assert span["owner_role"] == "required_caller"
    assert first == second


def test_cached_source_messages_and_journal_bytes_match_uncached(tmp_path, monkeypatch):
    path = "pkg/retained.py"
    content = "def retained():\n    return 'same bytes'\n"
    source_path = tmp_path / path
    source_path.parent.mkdir(parents=True)
    source_path.write_text(content)
    rows = [(path, content)]
    cached_store, current, cached_events = _make_store(monkeypatch, rows)
    uncached_store, _current, uncached_events = _make_store(monkeypatch, rows)
    cached_runtime = _source_runtime(cached_store, current, tmp_path)
    uncached_runtime = _source_runtime(uncached_store, current, tmp_path)

    cached_first = StateFrameRuntime._f4_source_messages(cached_runtime, {}, "", "")
    uncached_first = StateFrameRuntime._f4_source_messages(uncached_runtime, {}, "", "")
    uncached_store._selection_key = None
    uncached_store._selection_cache = None
    cached_hit = StateFrameRuntime._f4_source_messages(cached_runtime, {}, "", "")
    uncached_miss = StateFrameRuntime._f4_source_messages(uncached_runtime, {}, "", "")

    assert _canonical(cached_first) == _canonical(uncached_first)
    assert _canonical(cached_hit) == _canonical(uncached_miss)
    assert _canonical(cached_events) == _canonical(uncached_events)


def test_cached_results_match_uncached_results_for_randomized_inputs(monkeypatch):
    rng = random.Random(0xC08A14)
    for case in range(24):
        page_rows = [
            (f"pkg/mod_{index}.py", f"def item_{index}():\n    return {case + index}\n")
            for index in range(rng.randint(1, 12))
        ]
        cached_store, base_current, cached_events = _make_store(monkeypatch, page_rows)
        uncached_store, _uncached_current, uncached_events = _make_store(monkeypatch, page_rows)

        current = dict(base_current)
        for path in list(current):
            choice = rng.randrange(5)
            if choice == 0:
                current.pop(path)
            elif choice == 1:
                current[path] = "sha256:" + hashlib.sha256(
                    f"stale:{case}:{path}".encode()).hexdigest()
        for index in range(rng.randrange(4)):
            path = f"pkg/external_{case}_{index}.py"
            current[path] = "sha256:" + hashlib.sha256(path.encode()).hexdigest()

        working_path = rng.choice(("target.py", *(path for path, _ in page_rows)))
        working_content = rng.choice((None, "", "working source", f"revision {case}"))
        owner_symbol = rng.choice(("", "item_0", f"pkg.mod_{case % len(page_rows)}.item_0"))
        caller_path = rng.choice(("", *(path for path, _ in page_rows)))
        caller_symbol = rng.choice(("", "required", "caller"))
        mandatory = tuple(sorted(rng.sample(
            [path for path, _ in page_rows], rng.randrange(len(page_rows) + 1))))
        cached_store.required_source_paths = mandatory
        uncached_store.required_source_paths = mandatory
        args = (current, working_path, working_content)
        kwargs = {"owner_symbol": owner_symbol,
                  "required_caller_path": caller_path,
                  "required_caller_symbol": caller_symbol}

        actual_first = cached_store.select(*args, **kwargs)
        expected_first = _uncached_select(uncached_store, *args, **kwargs)
        assert actual_first == expected_first
        assert _canonical(actual_first) == _canonical(expected_first)

        actual_hit = cached_store.select(*args, **kwargs)
        expected_miss = _uncached_select(uncached_store, *args, **kwargs)
        assert actual_hit == expected_miss
        assert _canonical(actual_hit) == _canonical(expected_miss)

        changed_mandatory = () if mandatory else (page_rows[0][0],)
        cached_store.required_source_paths = changed_mandatory
        uncached_store.required_source_paths = changed_mandatory
        changed_args = (current, "pkg/other_target.py", f"changed work {case}")
        changed_kwargs = {**kwargs, "owner_symbol": "item_0" if owner_symbol != "item_0" else ""}
        actual_changed = cached_store.select(*changed_args, **changed_kwargs)
        expected_changed = _uncached_select(uncached_store, *changed_args, **changed_kwargs)
        assert _canonical(actual_changed) == _canonical(expected_changed)
        actual_changed_hit = cached_store.select(*changed_args, **changed_kwargs)
        expected_changed_miss = _uncached_select(uncached_store, *changed_args, **changed_kwargs)
        assert _canonical(actual_changed_hit) == _canonical(expected_changed_miss)
        assert _canonical(cached_events) == _canonical(uncached_events)
