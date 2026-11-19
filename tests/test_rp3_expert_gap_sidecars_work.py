"""First-request work and parity for the rulebook sidecars."""
from types import SimpleNamespace

import kernel_state_owner_inventory as owners
import kernel_unknowns_ledger as ledger
import pytest
from kernel_catalog import replay
from kernel_stateframe_runtime import StateFrameRuntime, note_provider_request
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_feed import WorkspaceFeed
from kernel_workspace_state import WorkspaceFence, WorkspaceState


@pytest.fixture
def rulebook_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    monkeypatch.setenv(owners.ENV, "on")
    monkeypatch.setenv("STATEFRAME_UNKNOWNS_LEDGER", "on")
    monkeypatch.setenv("STATEFRAME_LIFECYCLE_TABLE", "on")
    monkeypatch.setenv("STATEFRAME_ROW_ROLE", "on")
    monkeypatch.setenv("STATEFRAME_TIMING_TIMELINE", "off")
    root = tmp_path / "workspace"
    root.mkdir()
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"),
                           feed=WorkspaceFeed(root, enabled=True))
    consumers = WorkspaceConsumers(state)
    bind_consumers(root, consumers)
    (root / "publisher.py").write_text(
        "class Publisher:\n"
        "    publication_state: int = 0\n"
        "\n"
        "    def read(self):\n"
        "        return self.publication_state\n",
        encoding="utf-8",
    )
    runtime = object.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=root, task_row={
        "problem_statement": "Keep `publication_state` current. Retry publication on failure.",
        "public_baseline": "PASS",
    })
    runtime.journal = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    runtime._site_fact_blocks = list
    runtime.stateframe_site_facts_mode = "on"
    runtime._frame_budget_request_count = 0
    runtime.binding = lambda: None
    runtime._catalog_task_scope = lambda: ("task", "required")
    runtime._catalog_sync = lambda: replay(ledger.observe(runtime, task_id="task", scope="required"))
    yield runtime, state


def _prepare(runtime):
    return runtime._apply_rulebook_mechanisms([], provider_request=True)


def _assert_wire_facts(messages):
    contents = [message["content"] for message in messages]
    assert any(block.startswith(owners.HEADER) for block in contents)
    assert any(block.startswith("[LIFECYCLE STATE TABLE") for block in contents)
    assert any("[ROW ROLE] NEW_BEHAVIOR" in block for block in contents)
    assert any("state.writer `publication_state`: 0 writers" in block for block in contents)


def test_first_request_uses_one_inventory_observation(rulebook_runtime, monkeypatch):
    runtime, _state = rulebook_runtime
    runtime_inventory = owners.runtime_inventory
    calls = []

    def inventory(runtime):
        observed = runtime_inventory(runtime)
        calls.append(observed)
        return observed

    monkeypatch.setattr(owners, "runtime_inventory", inventory)
    first = _prepare(runtime)
    assert len(calls) == 1 and calls[0].complete
    _assert_wire_facts(first)

    note_provider_request(runtime, first)
    assert _prepare(runtime) == first


def test_first_request_builds_incomplete_inventory_once(rulebook_runtime, monkeypatch):
    runtime, state = rulebook_runtime
    with state.begin_write(("unreadable.py",)):
        (runtime.owner.root / "unreadable.py").write_bytes(b"\xff")

    build_inventory = owners.build_inventory
    builds = []

    def build(*args, **kwargs):
        inventory = build_inventory(*args, **kwargs)
        builds.append(inventory)
        return inventory

    monkeypatch.setattr(owners, "build_inventory", build)
    contents = [message["content"] for message in _prepare(runtime)]
    assert len(builds) == 1 and not builds[0].complete
    assert any("completeness=UNKNOWN" in block for block in contents)
    assert all("[ROW ROLE] NEW_BEHAVIOR" not in block for block in contents)


def test_later_request_uses_current_writer_facts(rulebook_runtime):
    runtime, state = rulebook_runtime
    first = _prepare(runtime)
    _assert_wire_facts(first)
    note_provider_request(runtime, first)

    with state.begin_write(("publisher.py",)):
        path = runtime.owner.root / "publisher.py"
        source = path.read_text(encoding="utf-8")
        path.write_text(source.replace("return self.publication_state",
                                       "self.publication_state = 1"), encoding="utf-8")

    contents = [message["content"] for message in _prepare(runtime)]
    assert any("writer   publication_state: publisher.py:5" in block for block in contents)
    assert any("parent behaviour: writers: publisher.py:5" in block for block in contents)
    assert all("[ROW ROLE] NEW_BEHAVIOR" not in block for block in contents)
