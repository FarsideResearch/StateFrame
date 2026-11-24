"""Workspace settling restores files from the pre-command manifest; asserts no request bytes."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import kernel_shell_tool
from kernel_effects import scan as scan_manifest
from kernel_exec_session import CleanupReport, ExecResult
from kernel_shell_tool import ShellTool
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import configure_replay
from kernel_workspace_consumers import ShellWorkspace, WorkspaceConsumers, bind_consumers
from kernel_workspace_objects import ObjectRef, open_workspace_objects
from kernel_workspace_state import WorkspaceFence, WorkspaceState


class ProcessBoundary:
    def configure(self, capability):
        pass

    def run(self, argv, capability, *, cwd):
        (cwd / "test_app.py").write_bytes(b"assert False\n")
        return ExecResult(0, False, b"changed\n", False, 8, 0.01,
                          CleanupReport(True, 0, (), "settled"))


@pytest.mark.parametrize("container_backend", [False, True], ids=["default", "container"])
def test_workspace_settle_uses_precommand_manifest_and_restores_test(
        tmp_path, monkeypatch, container_backend):
    run = tmp_path / "run"
    root = run / "workspace"
    root.mkdir(parents=True)
    test_file = root / "test_app.py"
    original = b"assert True\n"
    pre_command = b"assert 3\n"
    test_file.write_bytes(original)
    state = WorkspaceState(WorkspaceFence.create(root, run / "fence"))
    store_parent = run / "private"
    store_parent.mkdir(mode=0o700)
    store = open_workspace_objects(store_parent, "f" * 64, enabled=True)
    refs = {}

    class RollbackBlobs:
        def get(self, digest):
            with store.lease() as lease:
                return lease.read(refs[digest])

    class ShellPersistenceFixture(ShellWorkspace):
        def baseline(self):
            manifest = scan_manifest(root, exclude=lambda _: False)
            self._save(manifest)
            return manifest

        def _commit(self, boundary, snapshot, call_id, kind):
            assert snapshot.validate_for_use(boundary)
            self._save(self._manifest(snapshot.tree))

        def _save(self, manifest):
            pending = []
            for path, entry in manifest.items():
                if entry.kind == "file" and entry.digest not in refs:
                    ref = ObjectRef(entry.digest, entry.size)
                    data = (root / path).read_bytes()
                    pending.append((ref, lambda data=data: BytesIO(data)))
                    refs[entry.digest] = ref
            with store.lease() as lease:
                lease.publish_many(pending)

    shell = ShellPersistenceFixture(state, store, parent_key=None)
    shell.blobs = RollbackBlobs()
    consumers = WorkspaceConsumers(state, shell=shell)
    bind_consumers(root, consumers)

    tool = ShellTool(run, ToolSettings(run_env=False), {}, ProcessBoundary())
    tool.bind(root, {"files": {"test_app.py": original.decode()}, "scoped_edits": False,
                     "allowed_product_paths": ["test_app.py"], "new_file_roots": []},
              {str(test_file)})
    assert tool.workspace_effects is not None
    with state.begin_write(("test_app.py",)):
        test_file.write_bytes(pre_command)
    effects = tool.workspace_effects
    original_before = effects.before
    original_settle = effects.settle
    before_manifests = []
    settle_manifests = []

    def before(call_id):
        tree, manifest = original_before(call_id)
        before_manifests.append(manifest)
        return tree, manifest

    def settle(before_tree, manifest, policy, call_id):
        settle_manifests.append(manifest)
        return original_settle(before_tree, manifest, policy, call_id)

    monkeypatch.setattr(effects, "before", before)
    monkeypatch.setattr(effects, "settle", settle)
    detection_bytes = []
    if container_backend:
        original_detection = kernel_shell_tool._container_tampered_protected_tests

        def record_detection(baseline, manifest, protected):
            detection_bytes.append(test_file.read_bytes())
            return original_detection(baseline, manifest, protected)

        monkeypatch.setattr(kernel_shell_tool, "_container_tampered_protected_tests",
                            record_detection)
    monkeypatch.setattr(kernel_shell_tool, "active", lambda: SimpleNamespace(
        backend=(kernel_shell_tool.BACKEND_CONTAINER if container_backend else "host")))
    scan = tool._scan
    scans = []

    def counted_scan():
        scans.append(None)
        return scan()

    monkeypatch.setattr(tool, "_scan", counted_scan)
    configure_replay(run, active=False)
    reply = tool.execute({"command": "python change.py"}, "call-1")

    outcomes = {
        "protected file restored byte-for-byte": test_file.read_bytes() == original,
        "settle received pre-command manifest object": (
            len(before_manifests) == 1 and len(settle_manifests) == 1
            and settle_manifests[0] is before_manifests[0]),
        "manifest scan count matches backend": len(scans) == (1 if container_backend else 0),
        "container detection saw command edit before settlement": (
            not container_backend or detection_bytes == [b"assert False\n"]),
        "container tamper is reported": (
            not container_backend or "DETECTED PROTECTED_TEST_TAMPERED" in reply.observation),
    }
    assert all(outcomes.values()), outcomes
