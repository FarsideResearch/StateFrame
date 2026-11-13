"""Existence diff-check: planted breaks flagged, consistent edits quiet, switch, determinism.

Fixtures are a miniature of the T22 shop package used in the step-A proof (xf2_proof/RESULT.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"))
import kernel_crossfile_existence_check as ec  # noqa: E402
from kernel_crossfile_index import CrossFileIndex  # noqa: E402

SHOP = {
    "shop/__init__.py": "",
    "shop/pricing/__init__.py": "from .engine import price_order\n",
    "shop/pricing/rules.py": "def money(x):\n    return x\n\ndef best_coupon(c):\n    return c\n",
    "shop/pricing/engine.py": ("from .rules import best_coupon, money\n\n"
                               "def price_order(lines):\n    return money(best_coupon(lines))\n"),
    "shop/service.py": ("from .pricing.engine import price_order\nfrom .pricing import rules\n\n"
                        "def checkout(lines):\n    return price_order(lines), rules.money(1)\n"),
}


def write(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def tree(tmp_path):
    root = write(tmp_path / "repo", SHOP)
    return root, CrossFileIndex.build(root)


def names(findings):
    return sorted((f.path, f.name) for f in findings)


def test_switch_defaults_off_and_rejects_junk(monkeypatch):
    monkeypatch.delenv(ec.CROSSFILE_EXISTENCE_CHECK_ENV, raising=False)
    assert ec.enabled() is False
    monkeypatch.setenv(ec.CROSSFILE_EXISTENCE_CHECK_ENV, " ON ")
    assert ec.enabled() is True
    monkeypatch.setenv(ec.CROSSFILE_EXISTENCE_CHECK_ENV, "off")
    assert ec.enabled() is False
    monkeypatch.setenv(ec.CROSSFILE_EXISTENCE_CHECK_ENV, "maybe")
    with pytest.raises(ValueError):
        ec.enabled()


def test_clean_tree_and_consistent_edit_are_quiet(tree):
    root, before = tree
    assert ec.check_edit(before, root, ["shop/service.py"]) == []
    # add a helper and use it in the same edit: nothing dangling
    write(root, {"shop/pricing/rules.py": SHOP["shop/pricing/rules.py"] + "\ndef extra(x):\n    return x\n",
                 "shop/pricing/engine.py": SHOP["shop/pricing/engine.py"].replace("import best_coupon", "import extra, best_coupon")})
    assert ec.check_edit(before, root, ["shop/pricing/rules.py", "shop/pricing/engine.py"]) == []


def test_consistent_rename_across_files_is_quiet(tree):
    root, before = tree
    write(root, {p: t.replace("price_order", "quote_order") for p, t in SHOP.items() if "price_order" in t})
    assert ec.check_edit(before, root, ["shop/pricing/__init__.py", "shop/pricing/engine.py", "shop/service.py"]) == []


def test_rename_breaks_importers_not_the_edited_file(tree):
    root, before = tree
    write(root, {"shop/pricing/engine.py": SHOP["shop/pricing/engine.py"].replace("def price_order", "def price_order_v2")})
    got = ec.check_edit(before, root, ["shop/pricing/engine.py"])
    assert ("shop/service.py", ".pricing.engine.price_order") in names(got)
    assert ("shop/pricing/__init__.py", ".engine.price_order") in names(got)
    assert all(f.reason and f.line > 0 for f in got)


def test_deleted_def_is_flagged_in_importer(tree):
    root, before = tree
    write(root, {"shop/pricing/rules.py": "def money(x):\n    return x\n"})
    got = ec.check_edit(before, root, ["shop/pricing/rules.py"])
    assert ("shop/pricing/engine.py", ".rules.best_coupon") in names(got)


def test_import_of_nonexistent_name_flagged_in_edited_file(tree):
    root, before = tree
    write(root, {"shop/service.py": SHOP["shop/service.py"].replace("import price_order", "import price_order_missing")})
    got = ec.check_edit(before, root, ["shop/service.py"])
    assert names(got) == [("shop/service.py", ".pricing.engine.price_order_missing")]


def test_deleted_file_breaks_its_importers(tree):
    root, before = tree
    (root / "shop/pricing/rules.py").unlink()
    got = ec.check_edit(before, root, ["shop/pricing/rules.py"])
    assert any(f.path == "shop/pricing/engine.py" for f in got)


def test_preexisting_break_not_reported_but_new_one_is(tmp_path):
    files = {**SHOP, "shop/service.py": SHOP["shop/service.py"] + "\nfrom .pricing.rules import ghost\n"}
    root = write(tmp_path / "repo", files)
    before = CrossFileIndex.build(root)
    assert before.dangling_refs()  # the break exists before the edit
    # an unrelated edit above the old break shifts its line number only
    write(root, {"shop/service.py": "# note\n" + files["shop/service.py"]})
    assert ec.check_edit(before, root, ["shop/service.py"]) == []
    write(root, {"shop/service.py": files["shop/service.py"] + "from .pricing.rules import ghost2\n"})
    assert names(ec.check_edit(before, root, ["shop/service.py"])) == [("shop/service.py", ".pricing.rules.ghost2")]


def test_without_before_index_builds_fresh_and_finds_break(tree):
    root, _ = tree
    write(root, {"shop/pricing/rules.py": "def money(x):\n    return x\n"})
    got = ec.check_edit(None, root, ["shop/pricing/rules.py"])
    assert ("shop/pricing/engine.py", ".rules.best_coupon") in names(got)


def test_non_python_and_empty_changes_are_ignored(tree):
    root, before = tree
    assert ec.check_edit(before, root, []) == []
    assert ec.check_edit(before, root, ["README.md"]) == []


def test_deterministic_and_does_not_mutate_before_index(tree):
    root, before = tree
    write(root, {"shop/pricing/engine.py": SHOP["shop/pricing/engine.py"].replace("def price_order", "def other")})
    first = ec.check_edit(before, root, ["shop/pricing/engine.py"])
    assert first == ec.check_edit(before, root, ["shop/pricing/engine.py"])
    assert first == ec.check_edit(before, root, ["./shop/pricing/engine.py", "shop/pricing/engine.py"])
    assert before.dangling_refs() == []  # incremental update returned a new index
