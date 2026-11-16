"""Offline allocator benchmark; no engine, model server or workspace replay.

Reconstruct with --run RUN --write-fixture, then profile the portable fixture
with --profile /tmp/working-set.prof. The prewire envelope and closure come
from the recorded run; this does not claim the original live snapshot digest.
"""
from __future__ import annotations

import argparse
import cProfile
import gzip
import json
import pstats
import sys
import zlib
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/stateframe/engine"), str(ROOT / "src")]

from _journal import journal_bytes, measurement_journal_paths
from kernel_read_receipts import DecodedSpan, ReceiptIdentity
from kernel_working_set_snapshot import (
    PreparationSnapshot,
    SourceSlot,
    freeze_preparation,
)

FIXTURE = ROOT / "tests/fixtures/working_set_98.json.gz"


def reconstruct(run: Path) -> PreparationSnapshot:
    events = measurement_journal_paths(run)[0]
    rows = [json.loads(line) for line in journal_bytes(events, decompress=True).decode("utf-8").splitlines()]

    def hydrate(value):
        if isinstance(value, dict):
            if "$blob" in value:
                raw = zlib.decompress((events.parent / "blobs" / (value["$blob"] + ".zlib")).read_bytes())
                decoded = json.loads(raw)
                if isinstance(decoded, str):
                    try:
                        decoded = json.loads(decoded)
                    except json.JSONDecodeError:
                        pass  # Source-content blobs are ordinary strings.
                return hydrate(decoded)
            return {key: hydrate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [hydrate(item) for item in value]
        return value

    prepared = next(row for row in rows if row["kind"] == "request_prepared")
    closure = next(row for row in reversed(rows[:rows.index(prepared)])
                   if row["kind"] == "request_source_closure")
    plan = hydrate(closure["closure"])
    body = json.loads((run / "wire/0001.request.json").read_bytes())
    body["messages"] = hydrate(prepared["messages"])
    body["tools"] = hydrate(prepared["tools"])
    runtime = SimpleNamespace(_request_source_closure=plan, _empirical_frame_sources={})
    snap = freeze_preparation(body, runtime, request_id="recorded-98", byte_cap=82_026)
    return replace(snap, context_limit=65_536, output_reserve=4096, safety_reserve=4096)


def write_fixture(snapshot: PreparationSnapshot, run: Path) -> None:
    body = json.loads((run / "wire/0002.request.json").read_bytes())
    call = next(call for row in body["messages"] for call in row.get("tool_calls", ())
                if call["function"]["name"] == "read")
    arguments = json.loads(call["function"]["arguments"])
    data = asdict(snapshot)
    data["benchmark_read"] = {"arguments": arguments,
                              "raw": (run / "workspace" / arguments["path"]).read_bytes()}
    raw = json.dumps(data, sort_keys=True,
                     default=lambda value: {"$bytes": value.hex()}).encode()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(gzip.compress(raw, mtime=0))


def recorded_snapshot() -> PreparationSnapshot:
    data = json.loads(gzip.decompress(FIXTURE.read_bytes()), object_hook=lambda row:
                      bytes.fromhex(row["$bytes"]) if set(row) == {"$bytes"} else row)
    data.pop("benchmark_read")
    sources = []
    for row in data.pop("sources"):
        identity = row.pop("identity")
        if identity["lossy_span"] is not None:
            identity["lossy_span"] = DecodedSpan(**identity["lossy_span"])
        sources.append(SourceSlot(**row, identity=ReceiptIdentity(**identity)))
    return PreparationSnapshot(**data, sources=tuple(sources))


def recorded_read():
    from kernel_source_delivery import deliver_source, normalize_read_request
    data = json.loads(gzip.decompress(FIXTURE.read_bytes()))["benchmark_read"]
    raw = bytes.fromhex(data["raw"]["$bytes"])
    request = normalize_read_request(data["arguments"])
    display = data["arguments"]["path"]
    return raw, display, request, deliver_source(raw, display, 1, args=request)


def admission(snapshot: PreparationSnapshot):
    """Cold request floor plus exact admission of a fresh 120-line page."""
    from kernel_working_set_live import read_admission_floor, read_admission_slot

    _raw, display, request, page = recorded_read()
    slot = read_admission_slot(page, path=display, arguments=request.arguments,
                               ordinal=max(s.admission_ordinal for s in snapshot.sources) + 1)
    base = read_admission_floor(snapshot)
    floor = read_admission_floor(snapshot, ((slot, None),), require_feasible=True)
    assert floor is not None
    return base, floor


def admit_request(snapshot: PreparationSnapshot):
    """Exercise real admission, supplying only its already frozen input."""
    from unittest.mock import patch

    from kernel_read_admission import admit

    raw, display, request, page = recorded_read()
    events = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields))))
    with patch("kernel_working_set_live.read_admission_snapshot", return_value=snapshot):
        result = admit(runtime, raw=raw, display=display, start=1,
                       request=request, source=page, visible_messages=None)
    return result, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--write-fixture", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-engine", type=Path,
                        help="directory of pinned old engine modules for differential timing")
    args = parser.parse_args()
    snap = reconstruct(args.run) if args.run else recorded_snapshot()
    if args.write_fixture:
        if args.run is None:
            parser.error("--write-fixture requires --run")
        write_fixture(snap, args.run)
    if args.reference_engine:
        sys.path.insert(0, str(args.reference_engine))
    from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    started = perf_counter()
    result = prepare_candidate(snap)
    prepare_seconds = perf_counter() - started
    assert isinstance(result, PreparedCandidate), result
    started = perf_counter()
    delivered, events = admit_request(snap)
    admission_seconds = perf_counter() - started
    assert delivered.result_class == "OK" and events[-1][1]["decision"] == "FULL"
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(args.profile))
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(10)
    receipt = events[-1][1]
    floor_bytes = snap.byte_cap - receipt["allowance"]
    print(json.dumps({"slots": len(snap.sources), "prepare_seconds": prepare_seconds,
                      "admission_seconds": admission_seconds, "bytes": result.total_bytes,
                      "floor_bytes": floor_bytes,
                      "admitted_floor_bytes": floor_bytes + receipt["admitted_bytes"]}))
    if args.output:
        import pickle
        args.output.write_bytes(pickle.dumps((result, delivered, events)))


if __name__ == "__main__":
    main()
