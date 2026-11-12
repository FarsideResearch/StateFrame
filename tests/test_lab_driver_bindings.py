"""Bench drivers reuse shipped helpers and preserve their original path bindings."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


def test_extracted_lab_drivers_keep_shipped_bindings() -> None:
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent("""
        import importlib
        import pathlib
        import sys

        root = pathlib.Path(sys.argv[1])
        engine = root / 'src/stateframe/engine'
        sys.path.insert(0, str(engine))
        sys.path.insert(0, str(root / 'bench/lab'))

        from probes import real_repo_baseline as core1
        from probes import real_repo_baseline_v2 as core2
        from probes import real_repo_baseline_v3 as core3
        from probes import stateframe_shadow as shadow
        import func_index as index

        lab1 = importlib.import_module('real_repo_baseline')
        lab2 = importlib.import_module('real_repo_baseline_v2')
        lab3 = importlib.import_module('real_repo_baseline_v3')
        lab_shadow = importlib.import_module('stateframe_shadow')
        lab_index = importlib.import_module('func_index_shadow')

        assert core1.POC == core2.POC == core3.POC == engine
        assert core1.OUT == engine / 'results/real_repo_baseline.jsonl'
        assert core2.OUT == engine / 'results/real_repo_baseline_v2.jsonl'
        assert core3.OUT == engine / 'results/real_repo_baseline_v3.jsonl'
        assert lab1.INJECTORS is core1.INJECTORS
        assert lab1.REPO is core1.REPO and lab1.OUT is core1.OUT
        assert lab2.build_index is core2.build_index and lab2.OUT is core2.OUT
        assert lab3.enumerate_mutations is core3.enumerate_mutations
        assert lab3.OUT is core3.OUT
        assert lab_shadow.__file__ == shadow.__file__
        assert lab_shadow.Turn is shadow.Turn
        assert lab_shadow.fold_turn is shadow.fold_turn
        assert lab_index.REPO_DEFAULT is index.REPO_DEFAULT
        assert lab_index.build_func_index is index.build_func_index
        assert core1.harvest_tasks is lab1.harvest_tasks
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
