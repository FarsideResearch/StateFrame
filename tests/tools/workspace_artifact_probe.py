"""Independent filesystem and wire observations for the workspace release gates."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import stat
import zlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO


RECORD_LIMIT = 4 << 10
METADATA_PER_TURN = 64 << 10
_CHUNK = 128 << 10
_DECODE_LIMIT = 256 << 20
_OID = re.compile(r"[0-9a-f]{64}\Z")


@dataclass
class ArtifactSample:
    files: int = 0
    bytes_scanned: int = 0
    decoded_bytes_scanned: int = 0
    allocated_bytes: int = 0
    metadata_bytes: int = 0
    blob_copies: Counter[str] = field(default_factory=Counter)
    blob_sizes: dict[str, int] = field(default_factory=dict)
    copies: list[str] = field(default_factory=list)
    checkpoints: dict[Path, int] = field(default_factory=dict)
    candidates: dict[Path, int] = field(default_factory=dict)
    invalid_records: list[str] = field(default_factory=list)

    def counts(self) -> str:
        return (f"files={self.files} scanned_bytes={self.bytes_scanned} "
                f"decoded_bytes={self.decoded_bytes_scanned} allocated_bytes={self.allocated_bytes} "
                f"metadata_bytes={self.metadata_bytes} blob_files={sum(self.blob_copies.values())} "
                f"copies={len(self.copies)} checkpoints={len(self.checkpoints)} "
                f"candidates={len(self.candidates)}")


def _needles(payloads: Iterable[bytes]) -> tuple[bytes, ...]:
    # A match is evidence of more than 4 KiB of source bytes, rather than a
    # visible filename, digest, short excerpt, or coincidence in a journal.
    size = RECORD_LIMIT + 8
    needles = set()
    for data in payloads:
        if len(data) < size + 6:
            raise ValueError("workspace witness must contain more than 4 KiB of content")
        for start in (0, (len(data) - size - 6) // 2, len(data) - size - 6):
            raw = data[start:start + size]
            needles.update((raw, raw.hex().encode("ascii")))
            # The original bytes may start at any offset inside a base64 field.
            for shift in range(3):
                offset = start + shift
                piece = data[offset:offset + size - size % 3]
                needles.update((base64.b64encode(piece), base64.urlsafe_b64encode(piece)))
    return tuple(needles)


def _stream_match(chunks: Iterable[bytes], needles: tuple[bytes, ...], *, digest=None) -> tuple[int, bool]:
    overlap = max(map(len, needles), default=1) - 1
    tail = b""
    count = 0
    matched = False
    for chunk in chunks:
        count += len(chunk)
        if digest is not None:
            digest.update(chunk)
        if needles and not matched:
            window = tail + chunk
            matched = any(needle in window for needle in needles)
            tail = window[-overlap:]
    return count, matched


def _chunks(stream: BinaryIO) -> Iterable[bytes]:
    return iter(lambda: stream.read(_CHUNK), b"")


def _decoded_chunks(stream: BinaryIO, header: bytes) -> Iterable[bytes]:
    if header == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=stream) as decompressed:
            yield from _bounded_chunks(_chunks(decompressed))
    # The saved pickle and journal writers use the RFC 1950 zlib wrapper.
    elif len(header) == 2 and header[0] & 0x0f == 8 and int.from_bytes(header) % 31 == 0:
        yield from _bounded_chunks(_inflate(stream))


def _inflate(stream: BinaryIO) -> Iterable[bytes]:
    decoder = zlib.decompressobj()
    for chunk in _chunks(stream):
        while chunk and not decoder.eof:
            yield decoder.decompress(chunk, _CHUNK)
            chunk = decoder.unconsumed_tail
        if decoder.eof:
            return


def _bounded_chunks(chunks: Iterable[bytes]) -> Iterable[bytes]:
    total = 0
    for chunk in chunks:
        total += len(chunk)
        if total > _DECODE_LIMIT:
            raise AssertionError("artifact exceeded the independent decompression budget")
        yield chunk


def _files(root: Path) -> Iterable[tuple[Path, os.stat_result]]:
    def failed(error: OSError) -> None:
        raise error

    if not root.exists():
        return
    for directory, _, names in os.walk(root, followlinks=False, onerror=failed):
        for name in names:
            path = Path(directory, name)
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                yield path, info


def _object_oid(relative: Path) -> str | None:
    parts = relative.parts
    if (len(parts) == 5 and parts[:2] == ("workspace-state", "1")
            and parts[3] == "objects" and _OID.fullmatch(parts[2]) and _OID.fullmatch(parts[4])):
        return parts[4]
    return None


def _record(sample: ArtifactSample, path: Path, relative: Path, size: int, schema: str) -> None:
    if size > RECORD_LIMIT:
        sample.invalid_records.append(f"{relative}: {size} bytes exceeds the {RECORD_LIMIT}-byte record limit")
        return
    try:
        record = json.loads(path.read_bytes())
    except (ValueError, UnicodeError):
        sample.invalid_records.append(f"{relative}: invalid JSON record")
        return
    if not isinstance(record, dict) or record.get("schema") != schema or "files_base64" in record:
        sample.invalid_records.append(f"{relative}: expected {schema} without inline file data")


def observe_artifacts(run: Path, home: Path, *, payloads: Iterable[bytes] = (),
                      fixture_inputs: Iterable[Path] = ()) -> ArtifactSample:
    """Walk both roots, including unexpected filenames and workspace-state objects."""
    payloads = tuple(payloads)
    known_blobs = {hashlib.sha256(data).hexdigest(): len(data) for data in payloads}
    needles = _needles(payloads)
    fixture_inputs = frozenset(fixture_inputs)
    sample = ArtifactSample()
    allocated = set()
    checkpoint_sizes: Counter[Path] = Counter()
    checkpoint_files: Counter[Path] = Counter()
    for label, root in (("run", run), ("home", home)):
        for path, info in _files(root):
            relative = path.relative_to(root)
            parts = relative.parts
            inode = (info.st_dev, info.st_ino)
            sample.files += 1
            if inode not in allocated:
                allocated.add(inode)
                sample.allocated_bytes += info.st_blocks * 512

            object_oid = _object_oid(relative) if label == "home" else None
            live_file = label == "run" and (parts[0] == "workspace" or path in fixture_inputs)
            digest = hashlib.sha256() if object_oid or live_file else None
            with path.open("rb") as stream:
                size, matched = _stream_match(_chunks(stream), needles, digest=digest)
                assert size == info.st_size, f"artifact changed while reading {label}/{relative}"
                if needles and not matched:
                    stream.seek(0)
                    header = stream.read(2)
                    stream.seek(0)
                    try:
                        decoded, matched = _stream_match(_decoded_chunks(stream, header), needles)
                    except (OSError, EOFError, zlib.error):
                        # A random binary file may start with compression magic.
                        decoded = 0
                    sample.decoded_bytes_scanned += decoded
            sample.bytes_scanned += size

            content_oid = digest.hexdigest() if digest is not None else None
            if object_oid:
                assert object_oid == content_oid, f"misnamed or corrupt object {label}/{relative}"
            canonical_blob = object_oid is not None and known_blobs.get(object_oid) == size
            if canonical_blob:
                sample.blob_copies[object_oid] += 1
                sample.blob_sizes[object_oid] = size
            # The normal, writable checkout is allowed. A copy inside it must
            # be the original raw payload, not another serialized snapshot.
            if matched and not (canonical_blob or (live_file and content_oid in known_blobs)):
                sample.copies.append(f"{label}/{relative} ({size} bytes)")

            if label == "home" and parts[:2] == ("workspace-state", "1") and not canonical_blob:
                sample.metadata_bytes += size
            if label != "run":
                continue
            if len(parts) >= 3 and parts[0] == "checkpoints":
                directory = Path(*parts[:2])
                checkpoint_sizes[directory] += size
                checkpoint_files[directory] += 1
                if len(parts) == 3 and path.name == "checkpoint.json":
                    sample.checkpoints[relative.parent] = size
                    _record(sample, path, relative, size, "stateframe-engine-checkpoint/2")
            if len(parts) == 3 and parts[:2] == ("worker", "candidates") and path.stem.isdecimal():
                sample.candidates[relative] = size
                _record(sample, path, relative, size, "stateframe-candidate-trees/1")
                sample.metadata_bytes += size

    for directory in sample.checkpoints:
        sample.checkpoints[directory] = checkpoint_sizes[directory]
        sample.metadata_bytes += checkpoint_sizes[directory]
        if checkpoint_files[directory] != 1:
            sample.invalid_records.append(f"{directory}: {checkpoint_files[directory]} files in one checkpoint")
    sample.copies.sort()
    sample.invalid_records.sort()
    return sample


def assert_size_class(before: ArtifactSample, after: ArtifactSample, *,
                      unique_payload_bytes: int, changed_paths: int, turns: int) -> None:
    problems = []
    if after.copies:
        problems.append(f"workspace content outside one raw object or the live checkout: {after.copies[:5]}")
    if after.invalid_records:
        problems.append(f"invalid workspace records: {after.invalid_records[:5]}")
    if any(count != 1 for count in after.blob_copies.values()):
        problems.append(f"duplicate raw objects: {after.blob_copies}")
    if not before.blob_sizes.keys() <= after.blob_sizes.keys():
        problems.append("a witness blob disappeared while its prior boundary is retained")
    actual_growth = sum(after.blob_sizes[oid] for oid in after.blob_sizes.keys() - before.blob_sizes.keys())
    if actual_growth != unique_payload_bytes:
        problems.append(f"new witness blob bytes={actual_growth}, expected {unique_payload_bytes}")
    captured = len(after.checkpoints.keys() - before.checkpoints.keys())
    if captured != turns:
        problems.append(f"new checkpoints={captured}, expected {turns}")
    captured_candidates = len(after.candidates.keys() - before.candidates.keys())
    if captured_candidates < turns:
        problems.append(f"new candidates={captured_candidates}, expected at least {turns}")
    sizes = list(after.checkpoints.values())
    if sizes and (max(sizes) > RECORD_LIMIT or max(sizes) - min(sizes) > 128):
        problems.append(f"checkpoint sizes must stay flat and within {RECORD_LIMIT} bytes: {sizes}")
    metadata_growth = after.metadata_bytes - before.metadata_bytes
    if metadata_growth > (turns + changed_paths) * METADATA_PER_TURN:
        problems.append(f"new metadata bytes={metadata_growth}, over {METADATA_PER_TURN} per turn or edit")
    assert not problems, "\n".join([after.counts(), *problems])


def compare_request_bodies(left: Sequence[bytes], right: Sequence[bytes], *, first_send: int = 1) -> None:
    assert len(left) == len(right) and left, f"provider request counts differ or are empty: {len(left)}, {len(right)}"
    for index, (expected, actual) in enumerate(zip(left, right), first_send):
        assert expected == actual, (f"provider request {index} differs: "
                                    f"{hashlib.sha256(expected).hexdigest()} != "
                                    f"{hashlib.sha256(actual).hexdigest()}")


def delivery_outcome(run: Path) -> dict[str, object]:
    summary = json.loads((run / "summary.json").read_bytes())
    terminal = summary["engine_terminal"]
    return {
        **{name: summary.get(name) for name in (
            "engine_result", "succeeded", "delivery_refusal", "candidate_captured",
            "candidate_error", "candidate_paths_dropped", "candidate_files_written",
            "candidate_artifact_digest", "files_changed",
        )},
        "terminal_result": terminal.get("result"),
        "terminal_reason": terminal.get("reason_code"),
        "submission": terminal.get("submission"),
        "test_verdicts": tuple(summary[name]["passed"] for name in ("tests_before", "tests_after")),
        "final_diff": (run / "final.diff").read_bytes() if (run / "final.diff").is_file() else None,
    }


def compare_runs(base: Path, trial: Path, *, first_send: int = 1) -> None:
    def bodies(run: Path) -> list[bytes]:
        requests = {int(path.name.removesuffix(".request.json")): path
                    for path in (run / "wire").glob("*.request.json")}
        assert requests and set(requests) == set(range(1, max(requests) + 1)), sorted(requests)
        return [requests[send].read_bytes() for send in range(first_send, max(requests) + 1)]

    compare_request_bodies(bodies(base), bodies(trial), first_send=first_send)
    assert delivery_outcome(base) == delivery_outcome(trial), "delivery verdict or seal differs"
