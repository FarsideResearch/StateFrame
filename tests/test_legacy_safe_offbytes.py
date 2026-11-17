"""The BOUND_EDIT=off inline str_replace block is byte-pinned (dedented) so routing work cannot alter it."""
from __future__ import annotations

import hashlib
import pathlib
import textwrap

REACT = pathlib.Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine" / "probes" / "react_repair.py"
PINNED = "8abf4def129fcc79196aca765cfa9f792245e92d54a3c4cd77a7b6528c0470aa"  # dedented 42-line block at a69f5b52 (verified)


def _block() -> str:
    src = REACT.read_text(encoding="utf-8")
    fn = src.index("def str_replace_edit(")
    start = src.index("n = src.count(old_str)", fn)
    end = src.index("Nothing was changed.\")", src.index("the file would not parse afterwards", src.index("if n > 1:", start)))
    end = src.index("\n", end) + 1
    return textwrap.dedent(src[src.rfind("\n", 0, start) + 1:end])


def test_inline_legacy_block_is_byte_pinned():
    assert hashlib.sha256(_block().encode("utf-8")).hexdigest() == PINNED
