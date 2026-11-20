"""Cross-process cost and integrity gates for the opt-in symbol cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from time import sleep


PROBE = r"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

from symbol_registry import SymbolRegistry

root = Path(sys.argv[1]).resolve()
original_parse = ast.parse
original_read = Path.read_text
counts = {"parses": 0, "source_reads": 0, "row_comparisons": 0}

if importlib.util.find_spec("kernel_symbol_cache") is not None:
    import kernel_symbol_cache
    original_reusable = kernel_symbol_cache.reusable

    def reusable(*args, **kwargs):
        counts["row_comparisons"] += 1
        return original_reusable(*args, **kwargs)

    kernel_symbol_cache.reusable = reusable

def parse(*args, **kwargs):
    counts["parses"] += 1
    return original_parse(*args, **kwargs)

def read(path, *args, **kwargs):
    if path.suffix == ".py" and path.is_relative_to(root):
        counts["source_reads"] += 1
    return original_read(path, *args, **kwargs)

ast.parse = parse
Path.read_text = read
started = perf_counter()
registry = SymbolRegistry().build(root)
counts.update(seconds=perf_counter() - started, files=registry.files_scanned,
              symbols=len(registry.defs), answer_count=len(registry.where("answer")))
print(json.dumps(counts))
"""


def _probe(root: Path, cache: Path) -> dict[str, object]:
    env = dict(os.environ)
    source = Path(__file__).resolve().parents[1] / "src"
    env.update(
        PYTHONPATH=os.pathsep.join((str(source), str(source / "stateframe" / "engine"))),
        STATEFRAME_TURN_REPO_INTEL="on",
        STATEFRAME_REPO_CACHE_DIR=str(cache),
    )
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(root)], env=env, capture_output=True,
        text=True, timeout=40, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_cold_restart_reuses_index_and_one_edit_rebuilds_one_file(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for number in range(5000):
        (root / f"module_{number:04}.py").write_text(
            f"def answer():\n    return {number}\n", encoding="utf-8",
        )
    # A newly-created inode is racy under the cache's timestamp guard.
    sleep(2.1)
    cache = tmp_path / "cache"
    first = _probe(root, cache)
    second = _probe(root, cache)
    print(f"first={first} second={second}")
    assert first["parses"] == first["source_reads"] == 5000
    assert second["files"] == 5000 and second["symbols"] == 1
    assert (second["parses"], second["source_reads"]) == (0, 0), second
    assert second["row_comparisons"] <= 5000, second
    assert second["seconds"] < 5, second

    (root / "module_2500.py").write_text("def changed():\n    return 2500\n", encoding="utf-8")
    edited = _probe(root, cache)
    print(f"edited={edited}")
    assert (edited["parses"], edited["source_reads"]) == (1, 1), edited
    assert edited["row_comparisons"] <= 5000, edited
    assert edited["files"] == 5000 and edited["symbols"] == 2
    assert edited["answer_count"] == 4999


def test_corrupt_or_incomplete_cache_rebuilds(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for number in range(8):
        (root / f"module_{number}.py").write_text(f"def name_{number}(): pass\n")
    sleep(2.1)
    cache = tmp_path / "cache"
    assert _probe(root, cache)["parses"] == 8
    assert _probe(root, cache)["parses"] == 0
    files = list(cache.rglob("symbols-v1.json"))
    assert len(files) == 1
    files[0].write_text("{", encoding="utf-8")
    rebuilt = _probe(root, cache)
    assert (rebuilt["parses"], rebuilt["source_reads"]) == (8, 8), rebuilt
