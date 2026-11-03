"""Exact collision witnesses for the six migrated engine digest sites."""
from __future__ import annotations

from pathlib import Path
import os
from types import SimpleNamespace

import pytest

from kernel_coordinated_change_ledger import _compute_tree_revision
from kernel_request_evidence_closure import dependency_revision, source_revision
from kernel_expert_gap_sidecars import _fresh_closure
from kernel_request_source_window import _revision12
from kernel_stateframe_get_sure import _workspace_fingerprint
from kernel_ship_tree_smoke import tree_digest as smoke_digest
from realtask_benchmark import _tree_digest
# Names introduced by the fix are imported inside the tests that need them, so the
# collision witnesses above also run (and fail) on the unfixed 0.3.18 tree.


def _make(root: Path, files: dict[str, bytes], *, link: str | None = None) -> Path:
    root.mkdir()
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if link is not None:
        (root / "x").symlink_to(link)
    return root


@pytest.mark.parametrize("site,left,right,paths", [
    (lambda r, p: _compute_tree_revision(r, p), {}, {"a.py": b""}, ("a.py",)),
    (dependency_revision, {}, {"a.py": b"<unavailable>"}, ("a.py",)),
    (_revision12, {}, {"a.py": b"<missing>"}, ("a.py",)),
    (lambda r, p: _workspace_fingerprint(r), {"a.py": b"x"}, {"a.pyx": b""}, ()),
])
def test_collision_witnesses_and_determinism(tmp_path, site, left, right, paths):
    first = _make(tmp_path / "first", left)
    second = _make(tmp_path / "second", right)
    same = _make(tmp_path / "same", left)
    assert site(first, paths) != site(second, paths)
    assert site(first, paths) == site(same, paths)


def test_ship_missing_vs_literal_sentinel(tmp_path):
    missing = _make(tmp_path / "missing", {})
    literal = _make(tmp_path / "literal", {"a.py": b"<missing>"})
    assert smoke_digest(missing, ("a.py",)) != smoke_digest(literal, ("a.py",))


def test_ship_old_baseline_is_not_reused():
    from kernel_ship_tree_smoke import ShipTreeSmokeResult, TREE_DIGEST_FORMAT, apply_baseline
    failure = (("a.py", "ModuleNotFoundError: missing"),)
    result = ShipTreeSmokeResult("FAIL", ("a.py",), failure,
                                 digest_format=TREE_DIGEST_FORMAT)
    old = ShipTreeSmokeResult("FAIL", ("a.py",), failure)
    assert apply_baseline(result, old) == result


def test_ship_path_content_boundary_witness_and_determinism(tmp_path):
    first = _make(tmp_path / "first", {"a.py": b"x"})
    second = _make(tmp_path / "second", {"a.pyx": b""})
    same = _make(tmp_path / "same", {"a.py": b"x"})
    assert smoke_digest(first, ("a.py",)) != smoke_digest(second, ("a.pyx",))
    assert smoke_digest(first, ("a.py",)) == smoke_digest(same, ("a.py",))


def test_realtask_symlink_and_file_witness_and_count(tmp_path):
    linked = _make(tmp_path / "linked", {"y": b"target"}, link="y")
    regular = _make(tmp_path / "regular", {"y": b"target", "x": b"L:y"})
    same = _make(tmp_path / "same", {"y": b"target"}, link="y")
    assert _tree_digest(linked)[0] != _tree_digest(regular)[0]
    assert _tree_digest(linked) == _tree_digest(same)
    assert _tree_digest(linked)[1] == _tree_digest(regular)[1] == 2


def test_canonical_types_boundaries_domains_and_order():
    from kernel_tree_digest import (file_entry, missing_entry, symlink_entry,
                                    tree_digest as canonical_digest, unreadable_entry)
    domain = "test.tree/2"
    variants = [missing_entry("a.py"), unreadable_entry("a.py"), file_entry("a.py", b""),
                file_entry("a.py", b"<missing>"), file_entry("a.py", b"<unavailable>")]
    assert len({canonical_digest(domain, [entry]) for entry in variants}) == 5
    assert canonical_digest(domain, [file_entry("a.py", b"x")]) != canonical_digest(
        domain, [file_entry("a.pyx", b"")])
    assert canonical_digest(domain, [symlink_entry("x", "y")]) != canonical_digest(
        domain, [file_entry("x", b"L:y")])
    assert canonical_digest(domain, [variants[0]]) != canonical_digest("other.tree/2", [variants[0]])
    assert canonical_digest(domain, variants) == canonical_digest(domain, reversed(variants))


def test_sorted_sites_ignore_caller_path_order(tmp_path):
    root = _make(tmp_path / "root", {"a.py": b"a", "b.py": b"b"})
    assert dependency_revision(root, ("a.py", "b.py")) == dependency_revision(root, ("b.py", "a.py"))
    assert smoke_digest(root, ("a.py", "b.py")) == smoke_digest(root, ("b.py", "a.py"))


def test_path_reader_preserves_link_type_and_target(tmp_path):
    from kernel_tree_digest import path_entry, symlink_entry
    root = _make(tmp_path / "root", {"y": b"value"}, link="y")
    assert path_entry(root / "x", "x") == symlink_entry("x", "y")
    assert path_entry(root / "y", "x") != path_entry(root / "x", "x")


def test_old_realtask_row_is_refused_before_snapshot_lookup(tmp_path):
    from realtask_benchmark import (BenchmarkError, DIGEST_FORMAT, TaskPrivate,
                                    materialize_task_world, persist_snapshot, validate_public_task)
    row = {"base_digest": "0" * 64, "split": "dev", "symptom": "broken"}
    with pytest.raises(BenchmarkError, match="legacy realtask tree digest format"):
        materialize_task_world(row, tmp_path / "snapshots", tmp_path / "world")
    with pytest.raises(BenchmarkError, match="legacy realtask tree digest format"):
        validate_public_task(row)
    with pytest.raises(BenchmarkError, match="legacy realtask snapshot digest format"):
        materialize_task_world({**row, "digest_format": DIGEST_FORMAT,
                                "snapshot": {"tree_digest": "0" * 64}},
                               tmp_path / "snapshots", tmp_path / "world")
    private = TaskPrivate("parent", "gold", "subject", (), "")
    assert private.controller_record("task", "0" * 64)["digest_format"] == DIGEST_FORMAT
    root = _make(tmp_path / "tree", {"a.py": b"x"})
    with pytest.raises(BenchmarkError, match="old or incorrect format"):
        persist_snapshot(root, tmp_path / "snapshots", "0" * 64)


def _legacy(root: Path, paths, sentinel: bytes | None) -> str:
    """The 0.3.18 encoding (path + NUL + content, a sentinel for unreadable files)."""
    import hashlib
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8") + b"\0")
        try:
            digest.update((root / path).read_bytes())
        except OSError:
            if sentinel is not None:
                digest.update(sentinel)
    return "sha256:" + digest.hexdigest()


@pytest.mark.parametrize("site,sentinel,sort", [
    (_revision12, b"<missing>", True),
    (dependency_revision, b"<unavailable>", True),
    (lambda r, p: _compute_tree_revision(r, p), None, False),
])
def test_request_visible_sites_keep_018_bytes_for_regular_files(tmp_path, site, sentinel, sort):
    # Every NUL-free regular row retains the 0.3.18 preimage.
    root = _make(tmp_path / "root", {"a.py": b"x = 1\n", "pkg/b.py": b"", "c.txt": b"\xff"})
    paths = ("pkg/b.py", "a.py", "c.txt")
    ordered = sorted(paths) if sort else paths
    assert site(root, paths) == _legacy(root, ordered, sentinel)
    assert site(root, ()) == _legacy(root, (), sentinel)


@pytest.mark.parametrize("site,sentinel", [
    (_revision12, b"<missing>"), (dependency_revision, b"<unavailable>"),
    (lambda r, p: _compute_tree_revision(r, p), b""),
])
def test_request_visible_sites_type_missing_files(tmp_path, site, sentinel):
    missing = _make(tmp_path / "missing", {"b.py": b"1"})
    literal = _make(tmp_path / "literal", {"a.py": sentinel, "b.py": b"1"})
    assert site(missing, ("a.py", "b.py")) != site(literal, ("a.py", "b.py"))
    assert site(missing, ("a.py", "b.py")) == site(missing, ("a.py", "b.py"))


def test_compatible_encoding_cannot_meet_the_legacy_preimage():
    from kernel_tree_digest import compatible_revision
    # A legacy preimage starts with a non-empty NUL-free path; the typed form starts with NUL.
    typed = compatible_revision("t/2", [("a.py", None, "missing")])
    assert typed != compatible_revision("t/2", [("a.py", b"", "file")])
    assert compatible_revision("t/2", [("", b"x", "file")]) != compatible_revision(
        "t/2", [("", b"x", "file"), ("b", None, "missing")])
    with pytest.raises(ValueError):
        compatible_revision("unversioned", [("a.py", b"", "file")])


@pytest.mark.parametrize("source", ["VALUE = 1\n", "VALUE = 1\0\n"])
def test_expert_gap_sidecar_accepts_compatible_dependency_revision(tmp_path, source):
    (tmp_path / "a.py").write_text(source)
    revision = source_revision(source)
    dependency = dependency_revision(tmp_path, ("a.py",))
    identity = {"a.py": {"current_revision": revision}}
    plan = {
        "schema": "stateframe-request-evidence-closure/1", "target_path": "a.py",
        "dependency_digest_format": "stateframe.dependency_revision/2",
        "ticket_sha256": "ticket", "paths": ["a.py"],
        "omissions": [], "skipped_files": [],
        "coverage": {"completeness": {"status": "COMPLETE", "selected_files": 1,
                                       "declared_files": 1},
                     "skipped_file": {"status": "COMPLETE", "count": 0},
                     "omission": {"status": "COMPLETE", "count": 0},
                     "lexical_site": {"status": "COMPLETE", "count": 0},
                     "current_parent_identity": {"status": "CURRENT", "values": identity},
                     "parser_capability": {"status": "KNOWN"},
                     "dependency_revision": {"status": "CURRENT", "values": {"a.py": dependency}}},
        "revisions": {"a.py": revision}, "dependency_revisions": {"a.py": dependency},
        "dependency_paths": {"a.py": ["a.py"]},
        "adapters": {"a.py": {"completeness": "KNOWN", "status": "CURRENT"}},
        "source_contents": {"a.py": source}, "identities": identity,
        "workspace_authorization": {"scope": str(tmp_path), "generation": "1"},
    }
    closure = SimpleNamespace(closure_plan_for_runtime=lambda *_: plan,
                              language_adapter_for=lambda *_: object(),
                              _dependency_candidates=lambda *_args, **_kwargs: ())
    runtime = SimpleNamespace(workspace_scope=tmp_path, workspace_revision="1")
    ticket = SimpleNamespace(sha256="ticket")
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert _fresh_closure(runtime, tmp_path, root_fd, "a.py", closure, ticket) is not None
        old_paths = plan.pop("dependency_paths")
        assert _fresh_closure(runtime, tmp_path, root_fd, "a.py", closure, ticket) is None
        plan["dependency_paths"] = old_paths
        old_format = plan.pop("dependency_digest_format")
        assert _fresh_closure(runtime, tmp_path, root_fd, "a.py", closure, ticket) is None
        plan["dependency_digest_format"] = old_format
        plan["dependency_revisions"] = {"a.py": "sha256:" + "0" * 64}
        plan["coverage"]["dependency_revision"]["values"] = plan["dependency_revisions"]
        assert _fresh_closure(runtime, tmp_path, root_fd, "a.py", closure, ticket) is None
    finally:
        os.close(root_fd)


def test_expert_gap_sidecar_dependency_boundary_pair(tmp_path):
    first = _make(tmp_path / "first", {"a.py": b"xb.py\0y"})
    second = _make(tmp_path / "second", {"a.py": b"x", "b.py": b"y"})

    def fresh(root: Path, dependencies: tuple[str, ...]):
        source = (root / "a.py").read_text()
        source_rev = source_revision(source)
        dep_rev = dependency_revision(root, ("a.py", *dependencies))
        identity = {"a.py": {"current_revision": source_rev}}
        plan = {
            "schema": "stateframe-request-evidence-closure/1",
            "dependency_digest_format": "stateframe.dependency_revision/2",
            "target_path": "a.py", "ticket_sha256": "ticket", "paths": ["a.py"],
            "omissions": [], "skipped_files": [],
            "coverage": {"completeness": {"status": "COMPLETE", "selected_files": 1,
                                          "declared_files": 1},
                         **{key: {"status": "COMPLETE", "count": 0}
                            for key in ("skipped_file", "omission", "lexical_site")},
                         "current_parent_identity": {"status": "CURRENT", "values": identity},
                         "parser_capability": {"status": "KNOWN"},
                         "dependency_revision": {"status": "CURRENT", "values": {"a.py": dep_rev}}},
            "revisions": {"a.py": source_rev}, "dependency_revisions": {"a.py": dep_rev},
            "dependency_paths": {"a.py": sorted(("a.py", *dependencies))},
            "adapters": {"a.py": {"completeness": "KNOWN", "status": "CURRENT"}},
            "source_contents": {"a.py": source}, "identities": identity,
            "workspace_authorization": {"scope": str(root), "generation": "1"},
        }
        closure = SimpleNamespace(closure_plan_for_runtime=lambda *_: plan,
                                  language_adapter_for=lambda *_: object(),
                                  _dependency_candidates=lambda *_args, **_kwargs: dependencies)
        runtime = SimpleNamespace(workspace_scope=root, workspace_revision="1")
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            result = _fresh_closure(runtime, root, root_fd, "a.py", closure,
                                    SimpleNamespace(sha256="ticket"))
            assert result is not None
            return result[2]["a.py"]
        finally:
            os.close(root_fd)

    assert fresh(first, ()) != fresh(second, ("b.py",))


@pytest.mark.parametrize("site", [_revision12, dependency_revision,
                                   lambda r, p: _compute_tree_revision(r, p)])
def test_request_visible_nul_content_boundary(tmp_path, site):
    first = _make(tmp_path / "first", {"a.py": b"x\0b.py\0y"})
    second = _make(tmp_path / "second", {"a.py": b"x", "b.py": b"y"})
    assert site(first, ("a.py",)) != site(second, ("a.py", "b.py"))


@pytest.mark.parametrize("site", [_revision12, dependency_revision,
                                   lambda r, p: _compute_tree_revision(r, p)])
def test_request_visible_actual_nul_boundary_collision(tmp_path, site):
    first = _make(tmp_path / "first", {"a.py": b"xb.py\0y"})
    second = _make(tmp_path / "second", {"a.py": b"x", "b.py": b"y"})
    assert site(first, ("a.py",)) != site(second, ("a.py", "b.py"))


@pytest.mark.parametrize("site", [_revision12, dependency_revision,
                                   lambda r, p: _compute_tree_revision(r, p)])
def test_request_visible_symlink_is_not_regular_file(tmp_path, site):
    linked = _make(tmp_path / "linked", {"target.py": b"x"})
    (linked / "a.py").symlink_to("target.py")
    regular = _make(tmp_path / "regular", {"a.py": b"x"})
    assert site(linked, ("a.py",)) != site(regular, ("a.py",))


@pytest.mark.parametrize("site", [_revision12, dependency_revision,
                                   lambda r, p: _compute_tree_revision(r, p)])
def test_request_visible_nul_free_sequence_boundary(tmp_path, site):
    first = _make(tmp_path / "first", {"a": b"bc", "d": b"e"})
    second = _make(tmp_path / "second", {"a": b"b", "cd": b"e"})
    assert site(first, ("a", "d")) == site(second, ("a", "cd"))


def test_chain_hash_nul_witness():
    from chains import tree_hash
    assert tree_hash({"a.py": "x = 1\0b.py\0y = 2"}) != tree_hash(
        {"a.py": "x = 1", "b.py": "y = 2"})


def test_chain_hash_surrogate_witness():
    from chains import tree_hash
    assert tree_hash({"a.py": "x = '\ud800'"}) != tree_hash({"a.py": "x = '\udfff'"})


def test_chain_hash_keeps_legacy_regular_bytes():
    from chains import tree_hash
    import hashlib
    files = {"a.py": "x", "b.py": "y"}
    payload = b"a.py\0x\0b.py\0y\0"
    assert tree_hash(files) == hashlib.sha256(payload).hexdigest()[:16]


def test_behavior_memo_key_nul_witness():
    from behavior_bridge import _files_hash
    assert _files_hash({"a.py": "x = 1\0b.py\0y = 2"}) != _files_hash(
        {"a.py": "x = 1", "b.py": "y = 2"})


def test_owner_revision_vector_nul_witness():
    from kernel_state_owner_inventory import _source_revision_vector
    def revision(files):
        return _source_revision_vector(files, {})[0]
    assert revision({"a.py": "xb.py\0y"}) != revision({"a.py": "x", "b.py": "y"})


def test_owner_revision_vector_unterminated_witness():
    from kernel_state_owner_inventory import _source_revision_vector
    def revision(files):
        return _source_revision_vector(files, {})[0]
    assert revision({"a.py": "xb", "c.py": "y"}) == revision(
        {"a.py": "x", "bc.py": "y"})


def test_owner_single_file_keeps_legacy_bytes():
    from kernel_state_owner_inventory import _source_revision_vector
    import hashlib
    assert _source_revision_vector({"a.py": "x"}, {})[0] == hashlib.sha256(b"a.py\0x").hexdigest()


def test_swebench_manifest_nul_witness(tmp_path):
    from build_tasks.swebench_env import _dependency_manifest_digest
    first = _make(tmp_path / "first", {"requirements.txt": b"x\0requirements2.txt\0y"})
    second = _make(tmp_path / "second", {"requirements.txt": b"x", "requirements2.txt": b"y"})
    assert _dependency_manifest_digest(first)[0] != _dependency_manifest_digest(second)[0]


def test_swebench_manifest_symlink_tracks_target_and_content(tmp_path):
    from build_tasks.swebench_env import _dependency_manifest_digest
    linked = _make(tmp_path / "linked", {"deps.txt": b"x"})
    (linked / "requirements.txt").symlink_to("deps.txt")
    regular = _make(tmp_path / "regular", {"requirements.txt": b"x"})
    before = _dependency_manifest_digest(linked)[0]
    assert before != _dependency_manifest_digest(regular)[0]
    (linked / "deps.txt").write_bytes(b"y")
    assert before != _dependency_manifest_digest(linked)[0]


def test_swebench_manifest_format_refuses_old_cache(tmp_path):
    from build_tasks.swebench_env import (EnvironmentManager, _dependency_manifest_digest,
                                         DEPENDENCY_MANIFEST_FORMAT, _READY_MARKER,
                                         _FAILED_MARKER, _new_repair_contract,
                                         _write_repair_contract, environment_dir)
    import json
    first = _make(tmp_path / "first", {"requirements.txt": b"x"})
    digest, _ = _dependency_manifest_digest(first)
    assert digest.startswith(DEPENDENCY_MANIFEST_FORMAT + ":")
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").write_bytes(b"")
    (env_dir / _READY_MARKER).write_text(json.dumps({"status": "ready", "instance_id": "i",
        "manifest_digest": "sha256:" + "0" * 64}))
    result = EnvironmentManager(tmp_path)._cached_result("i", env_dir, manifest_digest=digest)
    assert result is not None and result.install_ok is False
    (env_dir / _READY_MARKER).write_text(json.dumps({"status": "ready", "instance_id": "i"}))
    result = EnvironmentManager(tmp_path)._cached_result("i", env_dir, manifest_digest=digest)
    assert result is not None and result.install_ok is False
    (env_dir / _FAILED_MARKER).write_text(json.dumps({"instance_id": "i", "blocker": "old",
        "manifest_digest": "sha256:" + "0" * 64}))
    result = EnvironmentManager(tmp_path)._failed_result("i", env_dir, manifest_digest=digest)
    assert result is not None and result.blocker.startswith("ENV_MANIFEST_CHANGED")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    contract = _new_repair_contract("i", manifest_digest="sha256:" + "0" * 64,
                                    manifest_names=("requirements.txt",), timeout=60)
    _write_repair_contract(environment_dir(cache_dir, "i"), contract)
    result = EnvironmentManager(cache_dir).prepare("i", first)
    assert result.install_ok is False
    assert result.blocker.startswith("ENV_REPAIR_CONTRACT_MISMATCH")


def test_remaining_sites_nul_witness(tmp_path, monkeypatch):
    from kernel_remaining_sites import current_workspace_revision
    import kernel_remaining_sites
    first = _make(tmp_path / "first", {"a.py": b"x\0b.py\0y"})
    second = _make(tmp_path / "second", {"a.py": b"x", "b.py": b"y"})
    paths = ["a.py"]
    monkeypatch.setattr(kernel_remaining_sites.source_window, "editable_product_paths",
                        lambda _runtime: paths)
    def revision(root):
        return current_workspace_revision(SimpleNamespace(owner=SimpleNamespace(root=root)))
    left = revision(first)
    paths[:] = ["a.py", "b.py"]
    assert left != revision(second)


def test_remaining_sites_keeps_legacy_regular_bytes(tmp_path, monkeypatch):
    import hashlib
    import kernel_remaining_sites
    from kernel_remaining_sites import current_workspace_revision
    root = _make(tmp_path / "root", {"a.py": b"x", "b.py": b"y"})
    monkeypatch.setattr(kernel_remaining_sites.source_window, "editable_product_paths",
                        lambda _runtime: ("a.py", "b.py"))
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root))
    assert current_workspace_revision(runtime) == hashlib.sha256(
        b"a.py\0x\0b.py\0y\0").hexdigest()[:8]


def test_staged_task_marker_boundary_witness(tmp_path):
    from long_horizon_protocol import _staged_digest
    first = _make(tmp_path / "first", {"manifest": b"x\0long-horizon-brief\0y", "brief": b"z"})
    second = _make(tmp_path / "second", {"manifest": b"x", "brief": b"y\0long-horizon-brief\0z"})
    phases = ({"phase": 1, "ticket": "ticket"},)
    assert _staged_digest(first / "manifest", first / "brief", phases) != _staged_digest(
        second / "manifest", second / "brief", phases)


def test_staged_task_keeps_legacy_unambiguous_bytes(tmp_path):
    from long_horizon_protocol import _staged_digest, STAGED_DIGEST_FORMAT
    import hashlib
    phases = ({"phase": 1, "ticket": "ticket"},)
    normal = _make(tmp_path / "normal", {"manifest": b"x", "brief": b"y"})
    old = b"x\0long-horizon-brief\0y\0phase\0" + b"1\0ticket"
    assert _staged_digest(normal / "manifest", normal / "brief", phases) == hashlib.sha256(old).hexdigest()
    assert STAGED_DIGEST_FORMAT == "stateframe.staged_task/2"


def test_staged_typed_digest_keeps_phase_order(tmp_path):
    from long_horizon_protocol import _staged_digest
    root = _make(tmp_path / "task", {"manifest": b"x\0", "brief": b"y"})
    phases = ({"phase": 1, "ticket": "one"}, {"phase": 2, "ticket": "two"})
    assert _staged_digest(root / "manifest", root / "brief", phases) != _staged_digest(
        root / "manifest", root / "brief", phases[::-1])


def test_staged_protocol_refuses_old_digest_format():
    from long_horizon_protocol import check_protocol, STAGED_DIGEST_FORMAT, STAGED_FAMILY
    tasks = [{"task_id": f"long-horizon2-0{number}-task", "task_digest": "abc",
              "task_digest_format": STAGED_DIGEST_FORMAT, "phases": [{}, {}]}
             for number in range(1, 4)]
    protocol = {"task_family": STAGED_FAMILY, "tasks": tasks}
    current = check_protocol(protocol, require_live_capacity=False)["errors"]
    assert "staged task digest format is missing or obsolete" not in current
    tasks[0].pop("task_digest_format")
    old = check_protocol(protocol, require_live_capacity=False)["errors"]
    assert "staged task digest format is missing or obsolete" in old


@pytest.mark.parametrize("probe", ["index", "literal"])
def test_workspace_truth_separator_witness(tmp_path, monkeypatch, probe):
    import hashlib
    import kernel_workspace_truth as truth
    original = hashlib.sha256

    class FixedDigest:
        def digest(self):
            return b"Q" * 32

        def hexdigest(self):
            return (b"Q" * 32).hex()

    monkeypatch.setattr(truth.hashlib, "sha256",
                        lambda data=b"": FixedDigest() if data == b"x" else original(data))
    first = _make(tmp_path / "first", {"a.py\\0" + "Q" * 32 + "b.py": b"x"})
    second = _make(tmp_path / "second", {"a.py": b"x", "b.py": b"x"})
    if probe == "index":
        assert truth._symbol_index(first)[2] != truth._symbol_index(second)[2]
    else:
        assert truth._current_literal_fact(first, "x", None, None, None).revision != (
            truth._current_literal_fact(second, "x", None, None, None).revision)


def test_workspace_truth_keeps_legacy_regular_bytes(tmp_path):
    import hashlib
    import kernel_workspace_truth as truth
    root = _make(tmp_path / "root", {"a.py": b"x"})
    digest = hashlib.sha256()
    digest.update(b"a.py\\0")
    digest.update(hashlib.sha256(b"x").digest())
    expected = "sha256:" + digest.hexdigest()
    assert truth._symbol_index(root)[2] == expected
    assert truth._current_literal_fact(root, "x", None, None, None).revision == expected


def test_owner_inventory_multifile_018_bytes():
    import hashlib
    from kernel_state_owner_inventory import _source_revision_vector
    assert _source_revision_vector({"a": "bc", "d": "e"}, {})[0] == hashlib.sha256(
        b"a\0bcd\0e").hexdigest()


def test_owner_inventory_surrogate_sources_remain_distinct():
    from kernel_state_owner_inventory import _source_revision_vector
    first = _source_revision_vector({"a.py": "\ud800"}, {})[0]
    second = _source_revision_vector({"a.py": "\udfff"}, {})[0]
    assert first != second


def test_workspace_truth_multifile_018_bytes(tmp_path):
    import hashlib
    import kernel_workspace_truth as truth
    root = _make(tmp_path / "root", {"a.py": b"x", "b.py": b"y"})
    expected = hashlib.sha256(b"a.py\\0" + hashlib.sha256(b"x").digest()
                              + b"b.py\\0" + hashlib.sha256(b"y").digest()).hexdigest()
    assert truth._symbol_index(root)[2] == "sha256:" + expected
    assert truth._current_literal_fact(root, "x", None, None, None).revision == "sha256:" + expected


def test_workspace_truth_fixed_digest_can_contain_legacy_marker(tmp_path, monkeypatch):
    import hashlib
    import kernel_workspace_truth as truth
    original = hashlib.sha256
    raw_digest = b"\\0" + b"Q" * 30

    class FixedDigest:
        def digest(self):
            return raw_digest

    monkeypatch.setattr(truth.hashlib, "sha256",
                        lambda data=b"": FixedDigest() if data == b"x" else original(data))
    root = _make(tmp_path / "root", {"a.py": b"x"})
    expected = "sha256:" + original(b"a.py\\0" + raw_digest).hexdigest()
    assert truth._symbol_index(root)[2] == expected
    assert truth._current_literal_fact(root, "x", None, None, None).revision == expected


def test_negative_lookup_dedupe_binds_path_tuple(tmp_path):
    from kernel_fivefix_control import classify_negative_lookup
    common = dict(status="NOT_FOUND", complete=True, query="thing", product_revision="sha256:" + "a" * 64)
    first = classify_negative_lookup(**common, scope="a|d|x", paths=("a", "d|x"))
    second = classify_negative_lookup(**common, scope="a|d|x", paths=("a", "d", "x"))
    assert first["dedupe_key"] != second["dedupe_key"]
    assert first["product_revision"] == second["product_revision"]
    left = _make(tmp_path / "left", {"a": b"bc", "d": b"e"})
    right = _make(tmp_path / "right", {"a": b"b", "cd": b"e"})
    revision = _revision12(left, ("a", "d"))
    assert revision == _revision12(right, ("a", "cd"))
    common["product_revision"] = revision
    left_fact = classify_negative_lookup(**common, scope="a|d", paths=("a", "d"))
    right_fact = classify_negative_lookup(**common, scope="a|cd", paths=("a", "cd"))
    assert left_fact["dedupe_key"] != right_fact["dedupe_key"]


def test_initialization_cache_binds_dependency_paths(tmp_path, monkeypatch):
    import kernel_request_evidence_closure as closure
    from kernel_request_evidence_closure import RevisionAwareFactCache, dependency_revision, initialization_fact
    root = _make(tmp_path / "root", {"a.py": b"x = 1\n", "b": b"bc", "d": b"e", "cd": b"e"})
    first_revision = dependency_revision(root, ("a.py", "b", "d"))
    (root / "b").write_bytes(b"b")
    assert dependency_revision(root, ("a.py", "b", "cd")) == first_revision
    choices = [("b", "d"), ("b", "cd")]
    monkeypatch.setattr(closure, "_dependency_candidates", lambda *_args, **_kwargs: choices.pop(0))
    cache = RevisionAwareFactCache()
    initialization_fact(root, "a.py", ("x",), cache=cache)
    initialization_fact(root, "a.py", ("x",), cache=cache)
    assert len(cache._values) == 2


def test_cached_closure_rejects_equal_revision_different_paths(tmp_path, monkeypatch):
    import kernel_request_evidence_closure as evidence
    from kernel_request_source_window import _cached_closure_is_fresh
    root = _make(tmp_path / "root", {"a.py": b"x = 1\n", "b": b"bc", "d": b"e", "cd": b"e"})
    source = (root / "a.py").read_text()
    old_revision = evidence.dependency_revision(root, ("a.py", "b", "d"))
    (root / "b").write_bytes(b"b")
    assert evidence.dependency_revision(root, ("a.py", "b", "cd")) == old_revision
    monkeypatch.setattr(evidence, "_dependency_candidates", lambda *_args, **_kwargs: ("b", "cd"))
    plan = {"revisions": {"a.py": evidence.source_revision(source)},
            "dependency_revisions": {"a.py": old_revision},
            "dependency_paths": {"a.py": ["a.py", "b", "d"]}}
    assert not _cached_closure_is_fresh(root, plan)


def test_import_lru_binds_connected_paths(tmp_path, monkeypatch):
    import kernel_repo_intel
    from kernel_request_evidence_closure import dependency_revision
    from kernel_request_source_window import _import_fact_cached
    calls = []
    monkeypatch.setattr(kernel_repo_intel, "import_construction_fact",
                        lambda *_args: calls.append(1) or {"status": "ok"})
    _import_fact_cached.cache_clear()
    root = _make(tmp_path / "root", {"a.py": b"x = 1\n", "b": b"bc", "d": b"e", "cd": b"e"})
    first_revision = dependency_revision(root, ("a.py", "b", "d"))
    common = (str(root), "a.py", "target_digest", ("x",), first_revision)
    _import_fact_cached(*common, ("a.py", "b", "d"))
    (root / "b").write_bytes(b"b")
    assert dependency_revision(root, ("a.py", "b", "cd")) == first_revision
    _import_fact_cached(*common, ("a.py", "b", "cd"))
    assert len(calls) == 2


def test_timeline_cache_component_binds_dependency_paths(tmp_path):
    from kernel_expert_gap_sidecars import _timeline_dependency_path_key
    from kernel_request_evidence_closure import dependency_revision
    root = _make(tmp_path / "root", {"a.py": b"x = 1\n", "b": b"bc", "d": b"e", "cd": b"e"})
    digest = dependency_revision(root, ("a.py", "b", "d"))
    (root / "b").write_bytes(b"b")
    assert dependency_revision(root, ("a.py", "b", "cd")) == digest
    first = {"dependency_revisions": {"a.py": digest},
             "dependency_paths": {"a.py": ["a.py", "b", "d"]}}
    second = {"dependency_revisions": dict(first["dependency_revisions"]),
              "dependency_paths": {"a.py": ["a.py", "b", "cd"]}}
    assert first["dependency_revisions"] == second["dependency_revisions"]
    assert _timeline_dependency_path_key(first) != _timeline_dependency_path_key(second)


def test_expert_gap_freshness_rejects_equal_digest_different_dependency_paths(tmp_path):
    from kernel_request_evidence_closure import dependency_revision, source_revision
    root = _make(tmp_path / "root", {"a.py": b"x = 1\n", "b": b"bc", "d": b"e", "cd": b"e"})
    source = (root / "a.py").read_text()
    revision = source_revision(source)
    old_digest = dependency_revision(root, ("a.py", "b", "d"))
    (root / "b").write_bytes(b"b")
    assert dependency_revision(root, ("a.py", "b", "cd")) == old_digest
    identity = {"a.py": {"current_revision": revision}}
    plan = {
        "schema": "stateframe-request-evidence-closure/1",
        "dependency_digest_format": "stateframe.dependency_revision/2",
        "target_path": "a.py", "ticket_sha256": "ticket", "paths": ["a.py"],
        "omissions": [], "skipped_files": [],
        "coverage": {"completeness": {"status": "COMPLETE", "selected_files": 1, "declared_files": 1},
                     **{key: {"status": "COMPLETE", "count": 0}
                        for key in ("skipped_file", "omission", "lexical_site")},
                     "current_parent_identity": {"status": "CURRENT", "values": identity},
                     "parser_capability": {"status": "KNOWN"},
                     "dependency_revision": {"status": "CURRENT", "values": {"a.py": old_digest}}},
        "revisions": {"a.py": revision}, "dependency_revisions": {"a.py": old_digest},
        "dependency_paths": {"a.py": ["a.py", "b", "d"]},
        "adapters": {"a.py": {"completeness": "KNOWN", "status": "CURRENT"}},
        "source_contents": {"a.py": source}, "identities": identity,
        "workspace_authorization": {"scope": str(root), "generation": "1"},
    }
    closure = SimpleNamespace(closure_plan_for_runtime=lambda *_: plan,
                              language_adapter_for=lambda *_: object(),
                              _dependency_candidates=lambda *_args, **_kwargs: ("b", "cd"))
    runtime = SimpleNamespace(workspace_scope=root, workspace_revision="1")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert _fresh_closure(runtime, root, root_fd, "a.py", closure,
                              SimpleNamespace(sha256="ticket")) is None
    finally:
        os.close(root_fd)


def test_source_closure_multifile_dependency_revision_018_bytes(tmp_path):
    import hashlib
    from kernel_request_evidence_closure import build_source_closure, render_closure_receipt
    root = _make(tmp_path / "root", {"a.py": b"import d\n", "d.py": b"value = 1\n"})
    plan = build_source_closure(root, ("a.py", "d.py"), "task", target_path="a.py").as_dict()
    expected = hashlib.sha256(b"a.py\0import d\nd.py\0value = 1\n").hexdigest()
    assert plan["dependency_revisions"]["a.py"] == "sha256:" + expected
    assert plan["dependency_paths"]["a.py"] == ["a.py", "d.py"]
    receipt = render_closure_receipt(plan)
    assert "dependency_revision=" in receipt
    assert "dependency_paths" not in receipt
    assert "dependency_digest_format" not in receipt


def test_old_closure_plan_fails_closed(tmp_path):
    from kernel_request_source_window import _cached_closure_is_fresh
    from kernel_expert_gap_sidecars import _fresh_closure
    root = _make(tmp_path / "root", {"a.py": b"x = 1\n"})
    assert not _cached_closure_is_fresh(root, {"revisions": {}, "dependency_revisions": {}})
    plan = {"schema": "stateframe-request-evidence-closure/1",
            "dependency_digest_format": "stateframe.dependency_revision/2",
            "target_path": "a.py", "ticket_sha256": "ticket", "paths": ["a.py"]}
    closure = SimpleNamespace(closure_plan_for_runtime=lambda *_: plan)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert _fresh_closure(SimpleNamespace(), root, root_fd, "a.py", closure,
                              SimpleNamespace(sha256="ticket")) is None
    finally:
        os.close(root_fd)
