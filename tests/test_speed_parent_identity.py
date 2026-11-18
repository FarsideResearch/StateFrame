"""Parent identity work budgets and generation boundaries for completion."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from time import perf_counter
from types import SimpleNamespace

import pytest

import kernel_completion_owner as completion
import kernel_green_lock as green_lock
from kernel_live_projection import WorkspaceViewStore
from kernel_stateframe_runtime import StateFrameRuntime


def _runtime(root, paths):
    runtime = object.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=root, task_row={"editable_paths": paths}, current_step=1)
    runtime.max_steps = 10
    runtime.projector_kind = "rich"
    runtime._unit_policy = None
    runtime._binding_epoch = 0
    runtime._binding_cache = None
    runtime._workspace_view_store = WorkspaceViewStore(root, "test-environment")
    runtime.journal = SimpleNamespace(manifest={}, emit=lambda *_args, **_kwargs: None)
    runtime.stateframe_completion_owner_mode = "on"
    runtime.pass_gates_vector = None
    runtime.public_test_verdict = lambda: None
    green_lock.GreenLock(runtime, mode="on")
    return runtime


def test_route_menu_and_send_share_one_parent_comparison(tmp_path, monkeypatch):
    file_count = 5_000
    paths = []
    for package_index in range(50):
        package = tmp_path / f"package_{package_index:02}"
        package.mkdir()
        for module_index in range(file_count // 50):
            source = package / f"module_{module_index:03}.py"
            source.write_text(f"VALUE = {module_index}\n")
            paths.append(source.relative_to(tmp_path).as_posix())
    monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "on")
    monkeypatch.setenv("STATEFRAME_HOST_CACHE", "off")
    runtime = _runtime(tmp_path, paths)

    counts = Counter()
    scan = green_lock._workspace_product_file_digests
    read = Path.read_bytes
    compare = green_lock.product_revision_change_from_parent

    def tracked_read(path):
        if path.suffix == ".py" and tmp_path in path.parents:
            counts["source_reads"] += 1
        return read(path)

    def tracked_scan(runtime, paths):
        counts["comparisons"] += 1
        return scan(runtime, paths)

    def tracked_compare(runtime):
        counts["consumers"] += 1
        return compare(runtime)

    monkeypatch.setattr(Path, "read_bytes", tracked_read)
    monkeypatch.setattr(green_lock, "_workspace_product_file_digests", tracked_scan)
    monkeypatch.setattr(green_lock, "product_revision_change_from_parent", tracked_compare)
    tools = [{"type": "function", "function": {
        "name": "read", "parameters": {"type": "object"},
    }}]
    messages = [{"role": "user", "content": "Read the package."}]
    started = perf_counter()
    for _ in range(17):
        assert completion.completion_route_decision(runtime, tools=tools).route == "NONE"
    assert completion.postgreen_done_offer(runtime)[0] is False
    menu = completion.terminal_tool_menu(runtime, tools)
    assert [tool["function"]["name"] for tool in menu] == ["read"]
    prepared = completion.establish_request_action_contract(
        runtime, messages, menu, request_id="request-1", preserve_input_surface=True,
    )
    sent = completion.assert_request_action_contract(
        runtime, messages, menu, request_id="request-1", boundary="provider",
    )
    elapsed = perf_counter() - started
    assert sent == prepared
    # Work counts are the stable performance contract; filesystem and suite load make
    # an absolute wall-clock cutoff noisy, so retain elapsed time as diagnostic only.
    print(f"Parent comparison fixture: {elapsed:.3f}s, {dict(counts)}")
    assert counts == {"consumers": 43, "comparisons": 1, "source_reads": file_count}


@pytest.mark.parametrize("change", ["edit", "create", "delete", "symlink"])
def test_parent_comparison_tracks_generation_and_restored_contents(tmp_path, monkeypatch, change):
    source = tmp_path / "pkg" / "module.py"
    source.parent.mkdir()
    if change != "create":
        source.write_text("VALUE = 1\n")
    monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "on")
    runtime = _runtime(tmp_path, ["pkg/module.py"])
    start = green_lock.product_revision_change_from_parent(runtime)
    assert start == (False, runtime.binding()["product_revision"],
                     "product_file_digests_match_parent")

    if change == "delete":
        source.unlink()
    elif change == "symlink":
        source.unlink()
        source.symlink_to("unavailable.py")
    else:
        source.write_text("VALUE = 2\n")
    runtime._note_applied_mutation(("pkg/module.py",))
    result = green_lock.product_revision_change_from_parent(runtime)
    expected = ((False, runtime.binding()["product_revision"], "parent_product_revision_unavailable")
                if change == "symlink" else
                (True, runtime.binding()["product_revision"], "product_file_digests_differ_from_parent"))
    assert result == expected
    assert result[1] != start[1]

    if change in {"create", "symlink"}:
        source.unlink()
    if change != "create":
        source.write_text("VALUE = 1\n")
    runtime._note_applied_mutation(("pkg/module.py",))
    assert green_lock.product_revision_change_from_parent(runtime) == start


def test_parent_comparison_tracks_workspace_view_and_widened_scope(tmp_path, monkeypatch):
    source = tmp_path / "one.py"
    other = tmp_path / "two.py"
    source.write_text("VALUE = 1\n")
    other.write_text("VALUE = 2\n")
    monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "on")
    runtime = _runtime(tmp_path, ["one.py"])
    scans = []
    scan = green_lock._workspace_product_file_digests

    def tracked_scan(runtime, paths):
        scans.append(tuple(paths))
        return scan(runtime, paths)

    monkeypatch.setattr(green_lock, "_workspace_product_file_digests", tracked_scan)
    for _ in range(2):
        assert green_lock.product_parent_proven_unchanged(runtime)
    assert scans == [("one.py",)]

    source.write_text("VALUE = 3\n")
    binding_epoch = runtime._binding_epoch
    runtime.current_view("external")
    assert runtime._binding_epoch == binding_epoch
    for _ in range(2):
        assert not green_lock.product_parent_proven_unchanged(runtime)
    source.write_text("VALUE = 1\n")
    runtime.note_mutation(("one.py",))
    assert green_lock.product_parent_proven_unchanged(runtime)

    assert green_lock.admit_widened_editable_path(runtime, "two.py")
    assert green_lock.product_parent_proven_unchanged(runtime)
    assert runtime._binding_epoch == binding_epoch
    assert scans == [("one.py",)] * 3 + [("two.py",), ("one.py", "two.py")]
    other.write_text("VALUE = 4\n")
    runtime._invalidate_binding_cache()
    result = green_lock.product_revision_change_from_parent(runtime)
    assert result[0] is True
    assert result[2] == "product_file_digests_differ_from_parent"


def test_concurrent_consumers_share_the_cold_comparison(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "on")
    runtime = _runtime(tmp_path, ["module.py"])
    scan = green_lock._workspace_product_file_digests
    scans = []
    ready = Barrier(2, timeout=10)

    def tracked_scan(runtime, paths):
        scans.append(tuple(paths))
        return scan(runtime, paths)

    def compare():
        ready.wait()
        return green_lock.product_revision_change_from_parent(runtime)

    monkeypatch.setattr(green_lock, "_workspace_product_file_digests", tracked_scan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(compare)
        second = pool.submit(compare)
        assert first.result(timeout=10) == second.result(timeout=10)
    assert scans == [("module.py",)]


def test_green_verdict_and_route_match_uncached_comparison(tmp_path, monkeypatch):
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    runtime = _runtime(tmp_path, ["module.py"])
    runtime.owner.f2p_ids = ("tests/test_module.py::test_value",)

    def green_receipt():
        binding = runtime.binding()
        return {
            "producer": "test_subprocess", "selectors": runtime.owner.f2p_ids,
            "binding": binding, "after_binding": binding, "binding_stable": True,
            "exit_code": 0,
        }

    def observe():
        return (
            green_lock.current_public_verdict_is_green(runtime),
            green_lock.product_revision_change_from_parent(runtime),
            completion._product_parent_unchanged(runtime, reason="test"),
            completion.postgreen_done_offer(runtime),
        )

    runtime.test_receipts = [green_receipt()]
    for expected in ((True, True, False), (False, False, False), (True, False, True)):
        monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "off")
        uncached = observe()
        monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "on")
        assert observe() == observe() == uncached
        verdict, _, unchanged, offer = uncached
        assert (verdict, unchanged, offer[0]) == expected

        if unchanged:
            source.write_text("VALUE = 2\n")
            runtime._note_applied_mutation(("module.py",))
        elif not verdict:
            runtime.test_receipts.append(green_receipt())


def test_parent_comparison_retries_unavailable_evidence(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "on")
    runtime = _runtime(tmp_path, ["module.py"])
    scan = green_lock._workspace_product_file_digests

    def unavailable(*_args):
        return None

    monkeypatch.setattr(green_lock, "_workspace_product_file_digests", unavailable)
    assert green_lock.product_revision_change_from_parent(runtime)[2] == "parent_product_revision_unavailable"
    monkeypatch.setattr(green_lock, "_workspace_product_file_digests", scan)
    assert green_lock.product_parent_proven_unchanged(runtime)


def test_comparison_retries_when_generation_changes_mid_scan(tmp_path, monkeypatch):
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "on")
    runtime = _runtime(tmp_path, ["module.py"])
    scan = green_lock._workspace_product_file_digests
    scans = []

    def compare_then_edit(runtime, paths):
        result = scan(runtime, paths)
        scans.append(runtime._binding_epoch)
        if len(scans) == 1:
            source.write_text("VALUE = 2\n")
            runtime._note_applied_mutation(("module.py",))
        return result

    monkeypatch.setattr(green_lock, "_workspace_product_file_digests", compare_then_edit)
    result = green_lock.product_revision_change_from_parent(runtime)
    assert result == (True, runtime.binding()["product_revision"], "product_file_digests_differ_from_parent")
    assert green_lock.product_revision_change_from_parent(runtime) == result
    assert scans == [0, 1]
