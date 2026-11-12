"""Public boundary material keeps working when a task ships non-Python public files.

A T22 run's public row lists TASK.md beside the product sources. Every non-.py path
used to raise "unsafe public boundary path", which aborted the whole boundary bundle.
"""
from __future__ import annotations

import pytest

from kernel_boundary_checks import public_material_from_row

PRODUCT = "def total(values):\n    return sum(values)\n"
TEST = "from shop.cart import total\n\n\ndef test_total():\n    assert total([1, 2]) == 3\n"


def _row(files):
    return {"brief": "Totals must add up.", "files": files,
            "public_tests": {"tests/test_cart.py": TEST}}


def test_non_python_public_files_are_skipped_not_fatal():
    _brief, sources, tests = public_material_from_row(_row({
        "TASK.md": {"parent_src": "# Checkout discrepancies\n"},
        "pyproject.toml": {"parent_src": "[project]\nname = 'shop'\n"},
        "shop/cart.py": {"parent_src": PRODUCT},
    }))
    assert sources == {"shop/cart.py": PRODUCT}
    assert "tests/test_cart.py" in tests


@pytest.mark.parametrize("path", [
    "../escape.md", "/etc/passwd.md", "C:/task/TASK.md", "./TASK.md", "a/../b.md",
    "hidden/notes.py", "oracle/fix.py", ".private/x.py", "grader.py/../x.md",
])
def test_unsafe_paths_still_refuse(path):
    with pytest.raises(ValueError, match="unsafe public boundary path"):
        public_material_from_row(_row({path: {"parent_src": "x = 1\n"},
                                       "shop/cart.py": {"parent_src": PRODUCT}}))
