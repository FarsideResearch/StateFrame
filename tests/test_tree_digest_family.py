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
    # These values are rendered into fact blocks / tool observations, so for every input
    # that never collided (all regular files) the value must equal 0.3.18 byte for byte.
    root = _make(tmp_path / "root", {"a.py": b"x = 1\n", "pkg/b.py": b"", "c.txt": b"\xff\x00"})
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


def test_expert_gap_sidecar_accepts_compatible_dependency_revision(tmp_path):
    (tmp_path / "a.py").write_text("VALUE = 1\n")
    revision = source_revision("VALUE = 1\n")
    dependency = dependency_revision(tmp_path, ("a.py",))
    identity = {"a.py": {"current_revision": revision}}
    plan = {
        "schema": "stateframe-request-evidence-closure/1", "target_path": "a.py",
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
        "adapters": {"a.py": {"completeness": "KNOWN", "status": "CURRENT"}},
        "source_contents": {"a.py": "VALUE = 1\n"}, "identities": identity,
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
        plan["dependency_revisions"] = {"a.py": "sha256:" + "0" * 64}
        plan["coverage"]["dependency_revision"]["values"] = plan["dependency_revisions"]
        assert _fresh_closure(runtime, tmp_path, root_fd, "a.py", closure, ticket) is None
    finally:
        os.close(root_fd)
