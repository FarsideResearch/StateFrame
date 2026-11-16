"""Semantic path identities: only a declared root makes an absolute path repo-relative.

A generic directory name is not a root marker. Run workspaces live in a directory
named ``workspace``, so treating that segment as a known root turned every absolute
path of a run into a repo-relative identity and cut repository paths that merely
contain such a directory.
"""

import kernel_semantics


def test_absolute_path_outside_a_declared_root_has_no_identity(tmp_path):
    inside_run = tmp_path / "run" / "workspace" / "shop" / "service.py"
    assert kernel_semantics._safe_path(str(inside_run)) == ""
    assert kernel_semantics._safe_path("/srv/job/workspace/shop/service.py") == ""


def test_absolute_path_under_the_declared_root_is_repo_relative():
    root = "/srv/job/workspace"
    assert kernel_semantics._safe_path(root + "/shop/service.py", root) == "shop/service.py"


def test_relative_path_keeps_a_directory_named_workspace():
    assert kernel_semantics._safe_path("pkg/workspace/models.py") == "pkg/workspace/models.py"
    assert (kernel_semantics._safe_path("/srv/repo/pkg/workspace/models.py", "/srv/repo")
            == "pkg/workspace/models.py")


def test_anchor_from_an_absolute_run_path_carries_no_path():
    anchor = kernel_semantics._anchor_from_value("/srv/job/workspace/shop/service.py:12:total")
    assert anchor is not None
    assert anchor.path == ""
    assert anchor.symbol == "total"
