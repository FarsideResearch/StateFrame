"""Work and authority checks across the editable-path consumers."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_common_sense_governor as governor
import kernel_editready_fact as editready
import kernel_request_source_window as source_window
from capability_request_preflight import prepare_release_host_text
from kernel_request_budget import apply_capture_budget
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_feed import WorkspaceFeed
from kernel_workspace_index import WorkspaceIndex
from kernel_workspace_state import WorkspaceFence, WorkspaceState


def _switches(monkeypatch, mode):
    for switch in (
        "WORKSPACE_OBJECTS", "TURN_ATOMIC_PUBLICATION", "TURN_BOUNDED_QUALIFICATION",
        "TURN_DELIVERY_PROOFS", "TURN_GENERATION_OVERLAP", "TURN_INDEXED_READS",
        "TURN_MINIMAL_LAUNCH", "TURN_PANELS", "TURN_REPO_INTEL", "TURN_TIMING",
    ):
        monkeypatch.setenv(f"STATEFRAME_{switch}", mode)
    monkeypatch.setenv("STATEFRAME_EDITREADY", "on")
    monkeypatch.setenv("STATEFRAME_COMMON_SENSE_GOVERNOR", "on")
    monkeypatch.setenv("STATEFRAME_LEAN", "on")
    monkeypatch.setenv("STATEFRAME_REQUEST_BUDGET", "off")
    monkeypatch.setenv("STATEFRAME_OWNER_FIRST_ALLOC", "off")


def _workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    state = WorkspaceState(
        WorkspaceFence.create(root, tmp_path / "private"),
        feed=WorkspaceFeed(root, enabled=True),
    )
    consumers = WorkspaceConsumers(state)
    bind_consumers(root, consumers)
    return root, consumers


def _read(call_id, path, span):
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "function": {"name": "read", "arguments": json.dumps({"path": path})}},
        ]},
        {"role": "tool", "tool_call_id": call_id,
         "content": f"[read page] path={path} revision=sha256:{'a' * 64} lines={span}\nsource"},
    ]


def _runtime(root, paths, harness=()):
    return SimpleNamespace(
        owner=SimpleNamespace(root=root, task_row={"editable_paths": list(reversed(paths))}),
        target=root / paths[-1],
        binding={"product_files": dict.fromkeys(paths), "harness_files": harness},
        host_closure_assessments=[{
            "id": "behavior", "clause": "Preserve behavior in checks/guard.py.",
            "verdict": "UNKNOWN",
        }],
        done_requested=True,
    )


def _physical_validation(runtime, rendered, page):
    capture = SimpleNamespace(presentation_runtime=runtime)
    body = {"messages": rendered, "tools": []}
    assert apply_capture_budget(capture, body) == body
    assert capture._lean_required_source_paths == page


def test_stable_editable_paths_across_prepare_and_physical_validations(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    root, consumers = _workspace(tmp_path)
    paths = tuple(f"pkg/item_{index:04d}.py" for index in range(4500))
    (root / "pkg").mkdir()
    for path in paths:
        (root / path).write_text("VALUE = 1\n", encoding="utf-8")
    runtime = _runtime(root, paths, harness=("checks/guard.py",))
    messages = _read("read-1", paths[-2], "1-2")
    page = (paths[-1], *paths[:15])
    counts = Counter()

    def count(module, name, bucket, predicate=lambda *args: True):
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            if predicate(*args):
                counts[bucket] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapped)

    count(source_window, "_editable_path_names", "wire_passes")
    count(source_window, "_eligible_editable_path", "wire_probes")
    count(editready, "_normalize_repo_path", "read_normalizations")
    if hasattr(source_window, "_first_editable_page"):
        count(source_window, "_first_editable_page", "eligibility_passes")
        count(source_window, "_path_text", "raw_normalizations")
    count(governor, "_paths", "governor_passes",
          lambda values, *args: values is not None and len(values) == len(paths))
    count(Path, "resolve", "path_resolves", lambda path: root in path.parents)
    count(os, "scandir", "directory_scans", lambda path: isinstance(path, int))
    count(WorkspaceIndex, "_file_digest", "snapshot_hashes")

    started = time.perf_counter()
    assert source_window.editable_product_paths(runtime) == page
    rendered, receipt = prepare_release_host_text(messages, runtime)
    assert receipt["editready_fact"]["path_count"] == 1
    assert paths[-2] in rendered[-1]["content"]
    assert governor.decide(runtime).unmet_scope == ("checks/guard.py",)
    prepare_elapsed = time.perf_counter() - started
    first = counts.copy()

    assert prepare_release_host_text(messages, runtime)[0] == rendered
    assert governor.decide(runtime).unmet_scope == ("checks/guard.py",)
    _physical_validation(runtime, rendered, page)
    _physical_validation(runtime, rendered, page)
    elapsed = time.perf_counter() - started
    repeated = counts - first
    print(f"prepare={prepare_elapsed:.6f}s repeated={elapsed - prepare_elapsed:.6f}s "
          f"total={elapsed:.6f}s first={dict(first)} repeated_work={dict(repeated)}")

    assert counts["wire_passes"] == 1, dict(counts)
    assert counts["wire_probes"] <= len(page)
    assert counts["eligibility_passes"] <= 1
    assert counts["raw_normalizations"] <= len(paths) + 3
    assert counts["governor_passes"] <= 1
    assert counts["read_normalizations"] <= len(paths) + 4
    assert repeated["read_normalizations"] == 1
    assert repeated["path_resolves"] == 0
    assert repeated["directory_scans"] == repeated["snapshot_hashes"] == 0
    assert consumers.state.work.content_reads == 0
    assert elapsed - prepare_elapsed < 10


@pytest.mark.parametrize("mode", ["off", "on"])
def test_scope_order_creates_and_current_read_receipt(tmp_path, monkeypatch, mode):
    _switches(monkeypatch, mode)
    root, consumers = _workspace(tmp_path)
    paths = tuple(f"pkg/part_{index:02d}.py" for index in range(20))
    (root / "pkg").mkdir()
    (root / "Private").mkdir()
    (root / "Tests").mkdir()
    for path in (*paths, "Private/answer.py", "Tests/check.py", "pkg/outside_scope.py"):
        (root / path).touch()
    outside = tmp_path / "external.py"
    outside.touch()
    (root / "pkg/a_invalid.py").symlink_to(outside)
    (root / "pkg/b_alias.py").symlink_to("../pkg/part_00.py")
    runtime = _runtime(root, paths, harness=("pkg/outside_scope.py",))
    runtime.owner.task_row["editable_paths"].extend(
        ["pkg/a_invalid.py", "pkg/b_alias.py", "Private/answer.py"],
    )
    runtime.edit_contract = {"edit_authority": {"editable_paths": ("pkg/0_new.py",)}}
    runtime.owner.edit_contract = {"editable_paths": ["Tests/check.py"]}
    first_page = (paths[-1], "pkg/b_alias.py", *paths[:14])
    assert source_window.editable_product_paths(runtime) == first_page

    messages = (
        _read("owned", paths[-2], "1-2")
        + _read("alias", "pkg/b_alias.py", "3-4")
        + _read("invalid", "pkg/a_invalid.py", "1-1")
        + _read("out-of-scope", "pkg/outside_scope.py", "1-1")
        + _read("excluded", "Tests/check.py", "1-1")
    )
    assert editready.editable_source_ranges(messages, runtime) == {
        paths[-2]: ((1, 2),), "pkg/b_alias.py": ((3, 4),),
    }
    rendered, _ = prepare_release_host_text(messages, runtime)
    assert "1-2" in rendered[-1]["content"]
    assert "outside_scope" not in rendered[-1]["content"]
    assert "pkg/outside_scope.py" in governor._available_paths(runtime, runtime.binding)
    assert "pkg/0_new.py" not in governor._available_paths(runtime, runtime.binding)

    # An external alias may reenter the workspace while its pinned tree stays identical.
    outside.unlink()
    outside.symlink_to(root / paths[0])
    assert source_window.editable_product_paths(runtime) == (
        paths[-1], "pkg/a_invalid.py", "pkg/b_alias.py", *paths[:13],
    )
    outside.unlink()
    outside.touch()

    with consumers.state.begin_write(("pkg/0_new.py", "pkg/b_alias.py")):
        (root / "pkg/0_new.py").touch()
        (root / "pkg/b_alias.py").unlink()
        (root / "pkg/b_alias.py").symlink_to(outside)
    assert source_window.editable_product_paths(runtime) == (paths[-1], "pkg/0_new.py", *paths[:14])
    messages[1] = _read("owned", paths[-2], "8-9")[1]
    assert editready.editable_source_ranges(messages, runtime) == {paths[-2]: ((8, 9),)}
    amended, _ = prepare_release_host_text([*messages, rendered[-1]], runtime)
    assert "8-9" in amended[-1]["content"]
    assert "1-2" not in amended[-1]["content"]
    assert editready.editable_source_ranges(_read("created", "pkg/0_new.py", "1-1"), runtime) == {
        "pkg/0_new.py": ((1, 1),),
    }

    runtime.edit_contract["edit_authority"]["editable_paths"] = ("pkg/outside_scope.py",)
    runtime.owner.task_row["editable_paths"].remove(paths[-2])
    assert editready.editable_source_ranges(messages, runtime) == {
        "pkg/outside_scope.py": ((1, 1),),
    }

    # Keep absolute authority declarations distinct from the relative wire page.
    runtime.owner.edit_contract["editable_paths"].append(str(root / paths[-2]))
    assert paths[-2] not in source_window.editable_product_paths(runtime)
    assert editready.editable_source_ranges(messages, runtime)[paths[-2]] == ((8, 9),)

    harness = tuple(f"checks/guard_{index:02d}.py" for index in range(10))
    runtime.binding["harness_files"] = harness
    runtime.host_closure_assessments = [
        {"id": path, "clause": f"Preserve behavior in {path}.", "verdict": "UNKNOWN"}
        for path in harness
    ]
    assert governor.decide(runtime).unmet_scope == harness[:8]
