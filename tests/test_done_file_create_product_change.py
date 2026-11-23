from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_done_is_submit_authority as submit
import kernel_idealpath
from kernel_delivery_manifest import DeliveryManifestTracker


def _runtime(root: Path, path: str, *, row=None, result="APPLIED", receipts=None):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"created\n")
    receipt = {"kind": "edit", "intent": f"file_create {path}",
               "result_class": result, "target_path": path, "receipt_id": "create-1"}
    owner = SimpleNamespace(
        root=root,
        task_row=row or {"tier": "USER-REPO", "scoped_edits": False},
        _receipts=[receipt] if receipts is None else receipts,
    )
    return SimpleNamespace(
        owner=owner,
        delivery_contract=DeliveryManifestTracker(root, {}, None),
        stateframe_done_is_submit_mode="on",
    )


@pytest.mark.parametrize("path", (
    "answer.json", "nested/answer.json", "notes.md", "nested/notes.md",
    "readme.txt", "nested/readme.txt", "module.py", "nested/module.py",
))
def test_live_applied_file_create_counts_product_file(tmp_path, path):
    runtime = _runtime(tmp_path, path)

    assert submit._applied_file_create_product_changed(runtime)


def test_empty_created_file_still_counts(tmp_path):
    runtime = _runtime(tmp_path, "answer.json")
    (tmp_path / "answer.json").write_bytes(b"")

    assert submit._applied_file_create_product_changed(runtime)


@pytest.mark.parametrize("delete_by_tool", (False, True))
def test_deleted_created_file_does_not_count(tmp_path, delete_by_tool):
    runtime = _runtime(tmp_path, "answer.json")
    if delete_by_tool:
        runtime.owner._receipts.append({
            "kind": "edit", "intent": "file_delete answer.json",
            "result_class": "APPLIED", "target_path": "answer.json",
            "receipt_id": "delete-1",
        })
    (tmp_path / "answer.json").unlink()

    assert not submit._applied_file_create_product_changed(runtime)


def test_refused_create_and_host_written_file_do_not_count(tmp_path):
    refused = _runtime(tmp_path, "refused.json", result="REFUSED")
    host_written = _runtime(tmp_path, "host-check.json", receipts=[])

    assert not submit._applied_file_create_product_changed(refused)
    assert not submit._applied_file_create_product_changed(host_written)


@pytest.mark.parametrize(("path", "row"), (
    ("outside.json", {"tier": "USER-REPO", "scoped_edits": False,
                       "task_file_scope": ["inside.json"]}),
    ("__pycache__/generated.py", {"tier": "USER-REPO", "scoped_edits": False}),
    ("answer.json", {"tier": "USER-REPO", "scoped_edits": False,
                      "allow_new_files": False}),
))
def test_scope_policy_and_product_exclusions_still_apply(tmp_path, path, row):
    runtime = _runtime(tmp_path, path, row=row)

    assert not submit._applied_file_create_product_changed(runtime)


def test_manifest_reader_must_return_the_applied_create(tmp_path):
    runtime = _runtime(tmp_path, "answer.json")
    runtime.delivery_contract._applied_edits = lambda _receipts, *, read_content: []

    assert not submit._applied_file_create_product_changed(runtime)


@pytest.mark.parametrize("replacement", ("fifo", "external_symlink"))
def test_done_does_not_read_nonregular_create_targets(tmp_path, monkeypatch, replacement):
    runtime = _runtime(tmp_path / "repo", "answer.json")
    target = runtime.owner.root / "answer.json"
    target.unlink()
    if replacement == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("requires named pipes")
        os.mkfifo(target)
    else:
        outside = tmp_path / "outside.json"
        outside.write_text("{}\n")
        target.symlink_to(outside)

    read_bytes = Path.read_bytes

    def prevent_unsafe_read(path):
        assert path != target, f"done tried to read {replacement} at the create path"
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", prevent_unsafe_read)
    assert not submit._applied_file_create_product_changed(runtime)


@pytest.mark.skipif(os.sep != "/", reason="requires a literal backslash in a file name")
def test_create_receipt_checks_the_literal_file_path(tmp_path):
    runtime = _runtime(tmp_path, r"notes\answer.json")
    assert submit._applied_file_create_product_changed(runtime)


@pytest.mark.skipif(os.sep != "/", reason="requires a literal backslash in a file name")
def test_live_task_scope_admits_only_the_actual_create_target(tmp_path):
    runtime = _runtime(tmp_path, r"notes\answer.json")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes/answer.json").write_text("{}\n")
    runtime.owner.task_row["task_file_scope"] = ["notes/answer.json"]

    assert not submit._applied_file_create_product_changed(runtime)


@pytest.mark.parametrize("protected_by", ("test", "hook"))
def test_create_target_replaced_with_protected_resource_is_refused(tmp_path, monkeypatch, protected_by):
    from kernel_hook_wiring import HookWiring

    runtime = _runtime(tmp_path, "answer.json")
    protected = tmp_path / "fixture.json"
    protected.write_text("{}\n")
    if protected_by == "test":
        runtime.protected_tests = {protected: "digest"}
    else:
        runtime._hook_wiring = HookWiring.__new__(HookWiring)
        runtime._hook_wiring.runner = SimpleNamespace(workspace=tmp_path)
        runtime._hook_wiring.protected_paths = {"fixture.json"}
    target = tmp_path / "answer.json"
    target.unlink()
    os.link(protected, target)

    _assert_unchanged_done_refusal(runtime, monkeypatch)


def test_create_target_with_unavailable_file_status_is_refused(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, "answer.json")
    target = tmp_path / "answer.json"
    is_file = Path.is_file
    os_stat = os.stat

    def stat_path(path, *args, **kwargs):
        if path == target:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return os_stat(path, *args, **kwargs)

    def check_file(path):
        # Python 3.11 through 3.13 propagate some OS errors from is_file().
        if path == target:
            return stat.S_ISREG(path.stat().st_mode)
        return is_file(path)

    monkeypatch.setattr(os, "stat", stat_path)
    monkeypatch.setattr(Path, "is_file", check_file)
    _assert_unchanged_done_refusal(runtime, monkeypatch)


def _assert_unchanged_done_refusal(runtime, monkeypatch):
    import kernel_green_lock

    monkeypatch.setattr(kernel_idealpath, "enabled", lambda _runtime: False)
    monkeypatch.setattr(submit, "candidate_changed", lambda _runtime: False)
    monkeypatch.setattr(kernel_green_lock, "current_public_verdict_is_green", lambda _runtime: False)
    monkeypatch.setattr(kernel_green_lock, "product_parent_proven_unchanged", lambda _runtime: True)

    refusal = submit.ownerless_early_done_refusal(runtime, available_tools=["file_create"])
    assert refusal is not None
    assert "no product file has been changed yet; no check has passed" in refusal["detail"]
    return refusal


def test_parent_change_short_circuits_receipt_fallback(monkeypatch):
    monkeypatch.setattr(kernel_idealpath, "enabled", lambda _runtime: False)
    monkeypatch.setattr(submit, "candidate_changed", lambda _runtime: True)
    monkeypatch.setattr(submit, "_applied_file_create_product_changed",
                        lambda _runtime: pytest.fail("receipt fallback ran after base change"))

    assert submit._ownerless_gate_preflight(
        SimpleNamespace(stateframe_done_is_submit_mode="on"),
        [{"function": {"name": "file_create"}}],
    ) == ("mutation_present", ("file_create",))


@pytest.mark.parametrize(("failure_site", "error_type"), (
    ("hook", RuntimeError),
    ("scope", LookupError),
    ("path", OSError),
    ("receipts", TypeError),
))
def test_unavailable_created_file_proof_keeps_early_done_refusal(
        tmp_path, monkeypatch, failure_site, error_type):
    import kernel_edit_source_tool
    import kernel_tool_paths

    runtime = _runtime(tmp_path, "answer.json")
    events = []
    runtime.journal = SimpleNamespace(
        emit=lambda kind, **fields: events.append({"kind": kind, **fields}))

    def fail(*_args, **_kwargs):
        raise error_type("created-file proof unavailable")

    if failure_site == "hook":
        runtime._hook_wiring = SimpleNamespace(protected=fail)
    elif failure_site == "scope":
        monkeypatch.setattr(kernel_edit_source_tool, "task_scope_refusal", fail)
    elif failure_site == "path":
        monkeypatch.setattr(kernel_tool_paths, "receipt_target_path", fail)
    else:
        class UnavailableReceipts:
            def __iter__(self):
                return fail()

        runtime.owner._receipts = UnavailableReceipts()

    refusal = _assert_unchanged_done_refusal(runtime, monkeypatch)

    assert [(event["kind"], event.get("error_type")) for event in events] == [
        ("done_submit_ownerless_gate_unavailable", error_type.__name__),
        ("early_done_refused", None),
    ]
    assert refusal["reason_code"] == submit.OWNERLESS_EARLY_DONE_REASON


@pytest.mark.parametrize("failed_check", (
    "candidate_changed", "current_public_verdict_is_green", "product_parent_proven_unchanged",
))
def test_other_preflight_errors_still_skip_early_done_gate(tmp_path, monkeypatch, failed_check):
    import kernel_green_lock

    runtime = _runtime(tmp_path, "answer.json")
    events = []
    runtime.journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))

    def unavailable(_runtime):
        raise LookupError("preflight proof unavailable")

    monkeypatch.setattr(kernel_idealpath, "enabled", lambda _runtime: False)
    monkeypatch.setattr(submit, "candidate_changed", lambda _runtime: False)
    monkeypatch.setattr(kernel_green_lock, "current_public_verdict_is_green", lambda _runtime: False)
    monkeypatch.setattr(kernel_green_lock, "product_parent_proven_unchanged", lambda _runtime: True)
    module = submit if failed_check == "candidate_changed" else kernel_green_lock
    monkeypatch.setattr(module, failed_check, unavailable)
    monkeypatch.setattr(submit, "_applied_file_create_product_changed",
                        lambda _runtime: pytest.fail("receipt fallback ran after a preflight error"))

    assert submit.ownerless_early_done_refusal(runtime, available_tools=["file_create"]) is None
    assert [(kind, fields["reason"]) for kind, fields in events] == [
        ("done_submit_ownerless_gate_unavailable", "ownerless proof preflight failed: LookupError"),
        ("early_done_gate_skipped", "proof_unavailable"),
    ]
    assert events[0][1]["error_type"] == "LookupError"


def test_created_file_proof_diagnostic_failure_keeps_early_done_refusal(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, "answer.json")
    events = []

    def unavailable_policy(_path):
        raise RuntimeError("hook edit policy is unavailable")

    def emit(kind, **fields):
        events.append((kind, fields))
        if kind == "done_submit_ownerless_gate_unavailable" and fields["error_type"] == "RuntimeError":
            raise LookupError("created-file proof diagnostic unavailable")

    runtime._hook_wiring = SimpleNamespace(protected=unavailable_policy)
    runtime.journal = SimpleNamespace(emit=emit)

    refusal = _assert_unchanged_done_refusal(runtime, monkeypatch)

    assert refusal["reason_code"] == submit.OWNERLESS_EARLY_DONE_REASON
    assert [(kind, fields.get("error_type")) for kind, fields in events] == [
        ("done_submit_ownerless_gate_unavailable", "RuntimeError"),
        ("early_done_refused", None),
    ]
