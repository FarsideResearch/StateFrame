"""Measure hot SourceEvidenceStore.select calls against a synthetic 3,000-file repo.

Run from the repository root with:
    python3 tests/benchmark_source_evidence_select.py

The benchmark reports five batches of 100 selections and their median per-call
time. Store construction and the initial uncached selection are outside timing.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/stateframe/engine"))
os.environ["STATEFRAME_EVIDENCE_DELIVERY"] = "off"

from kernel_source_evidence_store import SourceEvidenceStore  # noqa: E402


FILE_COUNT = 3_000
STORE_PAGE_COUNT = 256
REPEATS = 5
SELECTS_PER_REPEAT = 100


def _revision(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_pair(path: str, content: str, call_id: str):
    revision = _revision(content)
    end_line = len(content.splitlines())
    header = (f"[read page] path={path} revision={revision} lines=1-{end_line}/{end_line} "
              f"chars={len(content)} truncated=false decoded_with_replacement=false")
    assistant = {"role": "assistant", "tool_calls": [{
        "id": call_id,
        "type": "function",
        "function": {"name": "read", "arguments": json.dumps({"path": path})},
    }]}
    result = {"role": "tool", "tool_call_id": call_id,
              "content": f"{header}\nnotice\n{content}"}
    return [assistant, result]


def _build_store(root: Path) -> tuple[SourceEvidenceStore, dict[str, str]]:
    store = SourceEvidenceStore(lambda *_args, **_kwargs: None)
    current: dict[str, str] = {}
    for index in range(FILE_COUNT):
        path = f"pkg_{index // 100:02}/module_{index:04}.py"
        content = (f"VALUE = {index}\n\n"
                   f"def get_value_{index}():\n"
                   f"    return VALUE\n")
        source = root / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content)
        current[path] = _revision(content)
        if index < STORE_PAGE_COUNT:
            store.remember_exchange(_read_pair(path, content, f"read-{index}"))
    return store, current


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="source-evidence-select-") as temp:
        store, current = _build_store(Path(temp))
        if len(store.pages) != STORE_PAGE_COUNT:
            raise RuntimeError(f"expected {STORE_PAGE_COUNT} stored pages, got {len(store.pages)}")
        store.select(current, "", None)

        samples_ns: list[float] = []
        for _ in range(REPEATS):
            started = time.perf_counter_ns()
            for _ in range(SELECTS_PER_REPEAT):
                store.select(current, "", None)
            elapsed_ns = time.perf_counter_ns() - started
            samples_ns.append(elapsed_ns / SELECTS_PER_REPEAT)

        median_ns = statistics.median(samples_ns)
        print(json.dumps({
            "python_files": FILE_COUNT,
            "stored_pages": len(store.pages),
            "working_set_capacity": store.working_set.limit,
            "resident_spans": len(store.working_set.spans),
            "repeats": REPEATS,
            "selects_per_repeat": SELECTS_PER_REPEAT,
            "median_ns_per_select": round(median_ns),
            "median_us_per_select": round(median_ns / 1_000, 3),
            "samples_us_per_select": [round(sample / 1_000, 3) for sample in samples_ns],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
