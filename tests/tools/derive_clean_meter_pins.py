"""Derive the clean-meter request pins from a StateFrame source tree.

Run explicitly with --tree /path/to/untouched/tree; normal pytest does not invoke this script.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT, ROOT / "tests", ROOT / "src", ROOT / "src/stateframe/engine"):
    sys.path.insert(0, str(source))
# Another installed package can own "tests"; bind this script's test helpers to this tree.
local_tests = ModuleType("tests")
local_tests.__path__ = [str(ROOT / "tests")]
sys.modules["tests"] = local_tests

from tests.test_request_tokenizer_reprobe import (  # noqa: E402
    _capture_clean_meter_request_bodies,
    _clean_meter_fingerprints,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, type=Path, help="untouched StateFrame tree")
    args = parser.parse_args()
    tree = args.tree.resolve(strict=True)
    source = tree / "src"
    if not (source / "stateframe" / "_child.py").is_file():
        parser.error(f"StateFrame source tree is incomplete: {tree}")

    with tempfile.TemporaryDirectory(prefix="sf-clean-meter-pins-") as temporary:
        with patch.dict(os.environ, {"STATEFRAME_REPLAY_SOURCE": str(source)}):
            bodies = _capture_clean_meter_request_bodies(Path(temporary))
        hashes, counts = _clean_meter_fingerprints(bodies)

    print(f"masked SHA-256 requests 1-2: {hashes[:2]!r}")
    print(f"countdown-copy counts requests 1-6: {counts!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
