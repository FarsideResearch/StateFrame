"""Parent identity work budgets and generation boundaries for completion."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Barrier
from time import perf_counter
from types import SimpleNamespace

import pytest

import kernel_completion_owner as completion
import kernel_file_digest_memo
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
    events = []

    def emit(*args, **kwargs):
        events.append((args, kwargs))

    runtime.journal = SimpleNamespace(manifest={}, events=events, emit=emit)
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
    kernel_file_digest_memo.clear()

    counts = Counter()
    observations = {}
    generations = {}
    checkpoint = "route"
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
        result = compare(runtime)
        observations.setdefault(checkpoint, []).append(result)
        cached = getattr(runtime, "_parent_product_comparison_cache", None)
        if isinstance(cached, tuple) and cached:
            generations.setdefault(checkpoint, []).append(cached[0])
        return result

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
    checkpoint = "postgreen_done_offer"
    assert completion.postgreen_done_offer(runtime)[0] is False
    checkpoint = "terminal_tool_menu"
    menu = completion.terminal_tool_menu(runtime, tools)
    assert [tool["function"]["name"] for tool in menu] == ["read"]
    checkpoint = "request_preparation"
    prepared = completion.establish_request_action_contract(
        runtime, messages, menu, request_id="request-1", preserve_input_surface=True,
    )
    checkpoint = "provider_checkpoint"
    sent = completion.assert_request_action_contract(
        runtime, messages, menu, request_id="request-1", boundary="provider",
    )
    elapsed = perf_counter() - started
    assert sent == prepared
    # Work counts are the stable performance contract; filesystem and suite load make
    # an absolute wall-clock cutoff noisy, so retain elapsed time as diagnostic only.
    print(f"Parent comparison fixture: {elapsed:.3f}s, {dict(counts)}")
    expected_checkpoints = {
        "route", "postgreen_done_offer", "terminal_tool_menu",
        "request_preparation", "provider_checkpoint",
    }
    assert set(observations) == expected_checkpoints
    observed_results = [result for values in observations.values() for result in values]
    assert counts["consumers"] == len(observed_results) == 43
    assert counts["comparisons"] == 1
    assert counts["source_reads"] == file_count
    assert set(generations) == expected_checkpoints
    observed_generations = [generation for values in generations.values() for generation in values]
    assert observed_results
    assert all(result == observed_results[0] for result in observed_results)
    assert observed_results[0][0] is False
    assert observed_results[0][2] == "product_file_digests_match_parent"
    assert all(generation == observed_generations[0] for generation in observed_generations)


def test_parent_comparison_cache_preserves_completion_wire_and_journal(tmp_path, monkeypatch):
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    from kernel_measurement import ACTIVE_REQUEST
    from stateframe import _child
    import kernel_bounded_http

    wire_state_present = "wire" in _child.HOST_STATE
    previous_wire_state = _child.HOST_STATE.get("wire")

    def render(cache_mode):
        monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", cache_mode)
        monkeypatch.setenv("STATEFRAME_HOST_CACHE", "off")
        monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
        monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
        runtime = _runtime(tmp_path, ["module.py"])
        tools = [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object"},
        }}]
        messages = [{"role": "user", "content": "Read the package."}]
        route = completion.completion_route_decision(runtime, tools=tools)
        offer = completion.postgreen_done_offer(runtime)
        menu = completion.terminal_tool_menu(runtime, tools)
        prepared = completion.establish_request_action_contract(
            runtime, messages, menu, request_id="request-1", preserve_input_surface=True,
        )
        sent = completion.assert_request_action_contract(
            runtime, messages, menu, request_id="request-1", boundary="provider",
        )
        assert sent == prepared

        run = tmp_path.parent / f"wire-{cache_mode}"
        run.mkdir()
        monkeypatch.setattr(
            _child, "time",
            SimpleNamespace(monotonic=lambda: 1.0, monotonic_ns=lambda: 1_000_000_000),
        )
        monkeypatch.setattr(_child, "emit", lambda *_args, **_kwargs: None)

        def offline_send(_wire, _endpoint, _key, **_kwargs):
            return b'{"choices":[{}]}'

        monkeypatch.setattr(kernel_bounded_http, "request_bytes", offline_send)
        _child.install_wire_recorder(run, "http://localhost/chat")
        capture = SimpleNamespace(request_id="request-1")
        request_wire = json.dumps({"messages": messages, "tools": tools}).encode("utf-8")
        active = ACTIVE_REQUEST.set(capture)
        try:
            response = kernel_bounded_http.request_bytes(
                request_wire, "http://localhost/chat", "",
            )
        finally:
            ACTIVE_REQUEST.reset(active)
        assert response == b'{"choices":[{}]}'
        wire_tape = {
            path.relative_to(run).as_posix(): path.read_bytes().hex()
            for path in sorted(run.rglob("*")) if path.is_file()
        }
        wire_hashes = {
            path: hashlib.sha256(bytes.fromhex(content)).hexdigest()
            for path, content in wire_tape.items()
        }
        artifact = {
            "route": route.to_dict(),
            "postgreen_offer": offer,
            "messages": messages,
            "tools": tools,
            "menu": menu,
            "prepared": prepared,
            "sent": sent,
            "journal": runtime.journal.events,
            "wire_tape": wire_tape,
            "wire_hashes": wire_hashes,
        }
        return json.dumps(
            artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")

    try:
        uncached_wire_and_journal = render("off")
        cached_wire_and_journal = render("on")
    finally:
        if wire_state_present:
            _child.HOST_STATE["wire"] = previous_wire_state
        else:
            _child.HOST_STATE.pop("wire", None)
    assert cached_wire_and_journal == uncached_wire_and_journal
    artifact_hash = hashlib.sha256(uncached_wire_and_journal).hexdigest()
    assert hashlib.sha256(cached_wire_and_journal).hexdigest() == artifact_hash
    wire_hashes = json.loads(uncached_wire_and_journal)["wire_hashes"]
    print(f"Completion wire/tape file SHA-256: {wire_hashes}")
    print(f"Completion wire/journal bundle SHA-256: {artifact_hash}")


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
