"""Bounded, read-only discovery of the fixed run evidence sources."""
from __future__ import annotations

import errno
import hashlib
import json
import lzma
import math
import os
import re
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

from . import accounting
from .identity import from_seed
from .schema import _load_bounded_json, add_reason
from .types import Coverage, Cursor, Family, Namespace, PolicyView, Reason, RunRef, SourceOccurrence


_OWNER = os.geteuid()
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_DIR_FLAGS = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC
# A 0700 run root prevents group traversal. Native records beneath it can
# retain group write from umask 0002 without opening a group authority path.
_UNSAFE_RECORD_MODE = stat.S_IWOTH
_CHUNK = 64 * 1024
_MAX_FILE = 1 << 20
_MAX_RECORD = 1 << 20
_MAX_JOURNAL_BYTES = 128 << 20
_MAX_RECORDS = 65_536
_MAX_ENTRIES = 4096
_MAX_ATTEMPTS = 128
_MAX_UNITS = 256
_MAX_FDS = 128
_MAX_ORDINAL = (1 << 31) - 1
_FAMILY_ORDER = {
    Family.SPEC: 0,
    Family.PARALLEL: 1,
    Family.JOURNAL: 2,
    Family.PLAIN: 3,
    Family.SEND: 4,
    Family.TOOL: 8,
    Family.STOPPED: 90,
    Family.ENGINE_RESULT: 98,
    Family.SUMMARY: 99,
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_DECIMAL_NAME = re.compile(r"^[1-9][0-9]{0,8}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
class _SourceProblem(Exception):
    def __init__(self, reason: Reason):
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _File:
    data: bytes
    generation: bytes
    identity: tuple[int, int, int, int, int, int]
    offset: int = 0
    size: int = 0
    complete: bool = True


class Reader:
    """Read one capped cycle of fixed StateFrame run records without importing the engine."""

    def __init__(self, root: Path, key: bytes):
        self.root = Path(root)
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("invalid_field")
        self.key = key
        self._read = False
        self._root_fd: int | None = None
        self._root_error: Reason | None = None
        self._closed = False
        self._root_spec: dict[str, object] | None = None
        self._root_seed: bytes | None = None
        self._run_ref: RunRef | None = None
        self._entry_count = 0
        self._fd_count = 0
        try:
            if not self.root.is_absolute():
                raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
            self._root_fd = _open_absolute_dir(self.root)
        except _SourceProblem as error:
            self._root_error = error.reason
        except OSError as error:
            self._root_error = _reason_for_oserror(error)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        descriptor = self._root_fd
        self._root_fd = None
        if descriptor is not None:
            os.close(descriptor)

    def poll(self) -> tuple[tuple[SourceOccurrence, ...], Coverage]:
        """Read the fixed historical snapshot once."""
        if self._closed or self._root_error is not None or self._root_fd is None:
            reason = self._root_error or Reason.SOURCE_UNTRUSTED
            return (), Coverage(False, ((reason, None),))
        if self._read:
            return (), Coverage(True, ())
        self._read = True
        reasons: dict[Reason, int | None] = {}
        occurrences: list[SourceOccurrence] = []
        try:
            self._load_root_spec()
            if self._root_spec is None or self._root_seed is None or self._run_ref is None:
                return (), Coverage(False, ((Reason.SOURCE_MISSING, None),))
            for attempt, base_parts, spec in self._attempt_specs(reasons):
                namespace = Namespace(self._run_ref, attempt, None, None)
                if spec is not None:
                    self._append_document(occurrences, reasons, base_parts, "spec.json", Family.SPEC,
                                          namespace, spec_override=spec)
                self._append_document(occurrences, reasons, base_parts, "summary.json", Family.SUMMARY, namespace)
                self._append_document(occurrences, reasons, base_parts, "engine-result.json", Family.ENGINE_RESULT, namespace)
                self._scan_measurements(occurrences, reasons, base_parts, namespace, spec=spec)
                self._scan_plain(occurrences, reasons, base_parts, namespace)
                self._scan_sends(occurrences, reasons, base_parts, namespace, spec=spec)
                self._scan_parallel(occurrences, reasons, base_parts, namespace)
                self._scan_tool_tape(occurrences, reasons, base_parts, namespace)
            occurrences.sort(key=lambda item: (
                item.space.attempt, item.space.unit or b"", item.space.incarnation or 0,
                _FAMILY_ORDER[item.family], item.index, item.cursor.offset,
            ))
            _reconcile_sends(occurrences, reasons)
        except _SourceProblem as error:
            add_reason(reasons, error.reason, None)
        except OSError as error:
            add_reason(reasons, _reason_for_oserror(error), None)
        except (UnicodeError, json.JSONDecodeError, ValueError, lzma.LZMAError, RecursionError):
            add_reason(reasons, Reason.SOURCE_CORRUPT, None)
        complete = not reasons
        return tuple(occurrences), Coverage(complete, tuple(sorted(reasons.items(), key=lambda item: item[0].value)))

    def _load_root_spec(self) -> None:
        if self._root_spec is not None:
            return
        document = self._read_document(("spec.json",), required=True)
        if document is None or document.get("schema") != "stateframe-run/1":
            raise _SourceProblem(Reason.UNSUPPORTED_SCHEMA)
        if _epoch_ns(document, Family.SPEC) is None:
            raise _SourceProblem(Reason.SOURCE_CORRUPT)
        _attempt_from_spec(document)
        seed_text = document.get("run_identity_seed")
        if type(seed_text) is not str or not re.fullmatch(r"[0-9a-f]{32}", seed_text):
            raise _SourceProblem(Reason.SOURCE_CORRUPT)
        try:
            seed = bytes.fromhex(seed_text)
        except ValueError as error:
            raise _SourceProblem(Reason.SOURCE_CORRUPT) from error
        self._root_spec = document
        self._root_seed = seed
        self._run_ref, _trace = from_seed(self.key, seed)

    def _attempt_specs(self, reasons: dict[Reason, int | None]):
        assert self._root_spec is not None
        current = _attempt_from_spec(self._root_spec)
        result: list[tuple[int, tuple[str, ...], dict[str, object] | None]] = []
        root_spec = self._root_spec
        if current == 1:
            result.append((1, (), root_spec))
        else:
            result.append((current, (), root_spec))
        names = self._list_dir(("attempts",), reasons)
        seen_attempts: set[int] = set()
        for name in names:
            if not _DECIMAL_NAME.fullmatch(name):
                continue
            attempt = int(name)
            if attempt >= current or attempt > _MAX_ATTEMPTS:
                continue
            if len(seen_attempts) >= _MAX_ATTEMPTS:
                add_reason(reasons, Reason.DISCOVERY_LIMIT, None)
                break
            base = ("attempts", name)
            archived = self._read_document((*base, "spec.json"), required=False)
            if archived is None:
                add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)
                continue
            if (archived.get("schema") != "stateframe-run/1"
                    or archived.get("run_identity_seed") != self._root_spec.get("run_identity_seed")
                    or _epoch_ns(archived, Family.SPEC) is None):
                add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)
                continue
            try:
                recorded_attempt = _attempt_from_spec(archived)
            except _SourceProblem:
                add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)
                continue
            if recorded_attempt != attempt:
                add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)
                continue
            seen_attempts.add(attempt)
            result.append((attempt, base, archived))
        for missing in range(1, current):
            if missing not in seen_attempts:
                add_reason(reasons, Reason.GENERATION_AMBIGUOUS, 1)
        result.sort(key=lambda item: item[0])
        return result

    def _scan_measurements(self, out, reasons, base, space, *, spec=None) -> None:
        root = (*base, "worker", "measurement-runs")
        checkpoint = _resume_checkpoint(spec)
        archive = self._prior_archive(spec, space) if checkpoint is not None else None
        for name in self._list_dir(root, reasons):
            if not _SAFE_NAME.fullmatch(name):
                continue
            directory = (*root, name)
            first = len(out)
            if self._exists_file((*directory, "events.jsonl"), reasons):
                self._append_jsonl(out, reasons, (*directory, "events.jsonl"), Family.JOURNAL, space)
            elif self._exists_file((*directory, "events.jsonl.xz"), reasons):
                self._append_jsonl(out, reasons, (*directory, "events.jsonl.xz"), Family.JOURNAL, space,
                                   compressed=True)
            if archive is not None and first < len(out):
                self._suppress_restored_journal(out, first, space, (*directory[len(base):], "events.jsonl"))

    def _prior_archive(self, spec, space) -> tuple[str, str] | None:
        if space.attempt <= 1 or space.unit is not None:
            return None
        archive = ("attempts", str(space.attempt - 1))
        previous = spec["resume"].get("previous_attempt")
        if (type(previous) is str and 0 < len(previous) <= 4096
                and tuple(previous.split("/")[-2:]) == archive
                and all(part not in {"", ".", ".."} for part in previous.removeprefix("/").split("/"))):
            return archive
        return None

    def _suppress_restored_journal(self, out, first, space, relative_path) -> None:
        parent = Namespace(space.run, space.attempt - 1, None, None)
        prior_heads = {(item.cursor.stream, item.index): item.cursor.head for item in out[:first]
                       if item.family is Family.JOURNAL and item.space == parent}
        for index in range(first, len(out)):
            item = out[index]
            stream = self._stream_id(Family.JOURNAL, parent, relative_path, item.row.get("run_id"))
            head = prior_heads.get((stream, item.index))
            if head is None or head != item.cursor.head:
                continue
            out[index] = replace(item, epoch_ns=None, row={})

    def _scan_plain(self, out, reasons, base, space) -> None:
        path = (*base, "worker", "plain", "events.jsonl")
        if self._exists_file(path, reasons):
            self._append_jsonl(out, reasons, path, Family.PLAIN, space)

    def _scan_sends(self, out, reasons, base, space, *, spec=None) -> None:
        path = (*base, "sends.jsonl")
        if self._exists_file(path, reasons):
            first = len(out)
            self._append_jsonl(out, reasons, path, Family.SEND, space)
            checkpoint = _resume_checkpoint(spec)
            restored = checkpoint.get("sends") if type(checkpoint) is dict else None
            replayed = Counter(
                item.row["request_sha256"] for item in out[first:]
                if item.row.get("replayed") is True and type(item.row.get("request_sha256")) is str
                and (type(restored) is not int or item.index > restored)
            )
            if replayed:
                self._suppress_replayed_requests(out[:first], reasons, space, replayed)
            if type(restored) is not int or not 1 <= restored <= _MAX_ORDINAL:
                return
            prior_heads = {
                item.row["send"]: item.cursor.head
                for item in out[:first]
                if item.family is Family.SEND and item.space.attempt == space.attempt - 1
                and item.space.unit is None and type(item.row.get("send")) is int
            } if self._prior_archive(spec, space) is not None else {}
            for item in out[first:]:
                ordinal = item.row.get("send")
                if type(ordinal) is int and 1 <= ordinal <= restored:
                    # A restored send belongs to the previous attempt.
                    item.row["replayed"] = True
                    if prior_heads.get(ordinal) != item.cursor.head:
                        add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)

    def _suppress_replayed_requests(self, rows, reasons, space, replayed) -> None:
        """Join replayed sends to their completed journal exchanges by wire digest."""
        journal = [item for item in rows if item.space == space and item.family is Family.JOURNAL]
        if not journal:
            return
        pending = {}
        # Replay consumes sends before the first fresh submission. Match completed
        # exchanges in recorded order so a later, identical fresh send survives.
        for item in sorted(journal, key=lambda row: (row.epoch_ns or 0, row.index)):
            row = item.row
            stream = item.cursor.stream
            if row.get("kind") == "request_submitted" and type(row.get("event_id")) is str:
                pending[(stream, row["event_id"])] = row
            if row.get("kind") != "response_received":
                continue
            submitted = pending.pop((stream, row.get("transport_attempt_id")), None)
            if submitted is None:
                continue
            digest = submitted.get("request_body_digest", submitted.get("payload_hash"))
            if type(digest) is not str:
                continue
            digest = digest.removeprefix("sha256:")
            if replayed[digest] > 0:
                submitted["replayed"] = row["replayed"] = True
                replayed[digest] -= 1
        if any(replayed.values()):
            add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)

    def _scan_parallel(self, out, reasons, base, space) -> None:
        parallel_root = (*base, "parallel")
        calls = self._list_dir(parallel_root, reasons)
        units_seen = 0
        for call in calls:
            if not _SAFE_NAME.fullmatch(call):
                continue
            call_root = (*parallel_root, call)
            result = self._read_document((*call_root, "result.json"), required=False)
            if result is not None:
                if (type(result.get("schema")) is not str
                        or result["schema"] not in {"stateframe-parallel-result/1", "stateframe-parallel-result/2"}
                        or result.get("call_id") != call):
                    add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)
                    result = None
                else:
                    self._append_document(out, reasons, call_root, "result.json", Family.PARALLEL, space,
                                          spec_override=result)
            parent_outcomes = _parent_dispositions(result, reasons)
            unit_root = (*call_root, "units")
            for unit in self._list_dir(unit_root, reasons):
                if not _SAFE_NAME.fullmatch(unit):
                    continue
                units_seen += 1
                if units_seen > _MAX_UNITS:
                    add_reason(reasons, Reason.DISCOVERY_LIMIT, None)
                    return
                unit_spec_name = "unit-spec.json"
                unit_spec = self._read_document((*unit_root, unit, unit_spec_name), required=False)
                if unit_spec is None:
                    unit_spec_name = "spec.json"
                    unit_spec = self._read_document((*unit_root, unit, "spec.json"), required=False)
                if unit_spec is not None and (
                    unit_spec.get("schema") != "stateframe-unit-spec/3"
                    or unit_spec.get("call_id") != call or unit_spec.get("unit_id") != unit
                    or type(unit_spec.get("execution_id")) is not str or not unit_spec["execution_id"]
                    or type(unit_spec.get("start_manifest_digest")) is not str
                    or not re.fullmatch(r"[0-9a-f]{64}", unit_spec["start_manifest_digest"])
                ):
                    add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)
                    unit_spec = None
                binding = _unit_binding(self.key, space, call, unit, unit_spec)
                host_units = (*call_root, "host", "units", unit)
                host_rows: list[tuple[int, dict[str, object]]] = []
                for filename in self._list_dir(host_units, reasons):
                    if not filename.endswith(".json"):
                        continue
                    number = filename[:-5]
                    if not _DECIMAL_NAME.fullmatch(number):
                        continue
                    incarnation = int(number)
                    host = self._read_document((*host_units, filename), required=False)
                    if host is None:
                        continue
                    if (host.get("schema") != "stateframe-parallel-host-worker/1"
                            or host.get("call_id") != call or host.get("unit_id") != unit
                            or host.get("incarnation") != incarnation
                            or (unit_spec is not None and (
                                host.get("execution_id") != unit_spec["execution_id"]
                                or host.get("start_manifest_digest") != unit_spec["start_manifest_digest"]))
                            or (host.get("attempt") is not None and host.get("attempt") != space.attempt)):
                        add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)
                        continue
                    normalized = _host_occurrence(host, call, unit, incarnation, parent_outcomes.get(unit))
                    if normalized is None:
                        add_reason(reasons, Reason.INVALID_FIELD, 1)
                        continue
                    host_rows.append((incarnation, normalized))
                    unit_space = Namespace(space.run, space.attempt, binding[0], incarnation)
                    self._append_document(out, reasons, (*call_root, "host", "units", unit), filename,
                                          Family.PARALLEL, unit_space, spec_override=normalized)
                if unit_spec is not None and not host_rows:
                    unit_space = Namespace(space.run, space.attempt, binding[0], binding[1])
                    self._append_document(out, reasons, (*unit_root, unit), unit_spec_name, Family.PARALLEL,
                                          unit_space, spec_override={
                                              "kind": "unit_context", "call_id": call, "unit_id": unit,
                                              "incarnation": binding[1],
                                          })
                if len(host_rows) != 1:
                    if len(host_rows) > 1:
                        add_reason(reasons, Reason.GENERATION_AMBIGUOUS, None)
                    continue
                incarnation, host = host_rows[0]
                unit_space = Namespace(space.run, space.attempt, binding[0], incarnation)
                if unit_spec is not None:
                    self._scan_parallel_unit(out, reasons, (*unit_root, unit), unit_space)

    def _scan_parallel_unit(self, out, reasons, root, space) -> None:
        for path in (("journal", "events.jsonl"), ("journal", "events.jsonl.xz")):
            full = (*root, *path)
            if self._exists_file(full, reasons):
                self._append_jsonl(out, reasons, full, Family.JOURNAL, space, compressed=path[-1].endswith(".xz"))
                break
        if self._exists_file((*root, "tool-tape.jsonl"), reasons):
            self._append_jsonl(out, reasons, (*root, "tool-tape.jsonl"), Family.TOOL, space)
        self._scan_measurements(out, reasons, root, space)
        self._scan_plain(out, reasons, root, space)
        self._scan_sends(out, reasons, root, space)

    def _scan_tool_tape(self, out, reasons, base, space) -> None:
        path = (*base, "tool-tape.jsonl")
        if self._exists_file(path, reasons):
            self._append_jsonl(out, reasons, path, Family.TOOL, space)

    def _append_document(self, out, reasons, base, name, family, space, *, spec_override=None) -> None:
        file = self._read_file((*base, name), _MAX_FILE, required=False)
        if file is None or not self._room_for_record(out, reasons):
            return
        document = spec_override if spec_override is not None else _decode_document(file.data)
        sequence = _sequence(document, 1)
        stream = self._stream_id(family, space, (*base, name), document.get("run_id"))
        cursor = Cursor(stream, file.generation, len(file.data), sequence, _head(file.data))
        out.append(SourceOccurrence(space, family, sequence, _epoch_ns(document, family), document, cursor))

    def _append_jsonl(self, out, reasons, path, family, space, *, compressed=False) -> None:
        source = self._read_file(path, _MAX_JOURNAL_BYTES, required=False, allow_large=True)
        if source is None or not source.data:
            return
        try:
            data, complete = (_decompress_bounded(source.data, _MAX_JOURNAL_BYTES)
                              if compressed else (source.data, source.complete))
        except lzma.LZMAError:
            add_reason(reasons, Reason.SOURCE_CORRUPT, 1)
            return
        if not complete or not source.complete:
            add_reason(reasons, Reason.SOURCE_OVERSIZE, None)
        stream = None
        offset = sequence = 0
        while offset < len(data):
            line_end = data.find(b"\n", offset)
            if line_end < 0:
                if complete:
                    add_reason(reasons, Reason.TORN_TAIL, 1)
                break
            start, offset = offset, line_end + 1
            if line_end - start > _MAX_RECORD:
                add_reason(reasons, Reason.SOURCE_OVERSIZE, 1)
                continue
            body = data[start:line_end].removesuffix(b"\r")
            try:
                row = _decode_document(body)
                index = _sequence(row, sequence + 1, plain=family is Family.PLAIN)
            except (ValueError, UnicodeError, RecursionError, _SourceProblem):
                add_reason(reasons, Reason.SOURCE_CORRUPT, 1)
                continue
            if index <= sequence:
                add_reason(reasons, Reason.SOURCE_CORRUPT, 1)
                continue
            sequence = index
            if family in {Family.JOURNAL, Family.PLAIN} and _invalid_text_fields(row):
                add_reason(reasons, Reason.INVALID_FIELD, 1)
                continue
            if stream is None:
                stream = self._stream_id(family, space, path, row.get("run_id"))
            if (family is Family.JOURNAL and row.get("kind") in {"check_receipt", "check_receipt_unchanged"}
                    and "record_hash" in row and not _valid_check_record_hash(row)):
                add_reason(reasons, Reason.SOURCE_CORRUPT, 1)
                continue
            row = _projection_fields(family, row)
            if row is None:
                continue
            if not self._room_for_record(out, reasons):
                break
            cursor = Cursor(stream, source.generation, offset, index, _head(body))
            if family is Family.SEND:
                ordinal = row.get("send")
                if type(ordinal) is not int or not 1 <= ordinal <= _MAX_ORDINAL:
                    add_reason(reasons, Reason.INVALID_FIELD, 1)
                    continue
                self._add_send_route(path[:-1], ordinal, row, reasons)
            out.append(SourceOccurrence(space, family, index, _epoch_ns(row, family), row, cursor))

    def _read_document(self, path, *, required: bool) -> dict[str, object] | None:
        file = self._read_file(path, _MAX_FILE, required=required)
        if file is None:
            return None
        document = _decode_document(file.data)
        if type(document) is not dict:
            raise _SourceProblem(Reason.SOURCE_CORRUPT)
        return document

    def _read_file(
        self,
        path: tuple[str, ...],
        limit: int,
        *,
        required: bool,
        offset: int = 0,
        allow_large: bool = False,
        appendable: bool = False,
    ) -> _File | None:
        if self._root_fd is None:
            raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
        if not path or any(not _valid_component(part) for part in path):
            raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
        parent = self._open_dir_path(path[:-1])
        if parent is None:
            if required:
                raise _SourceProblem(Reason.SOURCE_MISSING)
            return None
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(path[-1], _FILE_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                if required:
                    raise _SourceProblem(Reason.SOURCE_MISSING)
                return None
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.EISDIR, errno.ENXIO}:
                    raise _SourceProblem(Reason.SOURCE_UNTRUSTED) from error
                raise
            self._fd_count += 1
            if self._fd_count > _MAX_FDS:
                raise _SourceProblem(Reason.DISCOVERY_LIMIT)
            before = os.fstat(descriptor)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != _OWNER or before.st_nlink != 1
                    or before.st_mode & _UNSAFE_RECORD_MODE or not before.st_mode & stat.S_IRUSR):
                raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
            if (before.st_size < 0 or (not allow_large and before.st_size > limit)
                    or type(offset) is not int or offset < 0 or offset > before.st_size):
                raise _SourceProblem(Reason.SOURCE_OVERSIZE)
            if offset:
                os.lseek(descriptor, offset, os.SEEK_SET)
            chunks: list[bytes] = []
            total = 0
            read_limit = min(limit, before.st_size - offset)
            while total < read_limit:
                chunk = os.read(descriptor, min(_CHUNK, read_limit - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            after = os.fstat(descriptor)
            identity_before = _file_identity(before)
            if identity_before != _file_identity(after) or total != read_limit:
                raise _SourceProblem(Reason.SOURCE_BUSY)
            generation = _generation(self.key, identity_before, appendable=appendable)
            return _File(b"".join(chunks), generation, identity_before, offset, before.st_size,
                         offset + total == before.st_size)
        finally:
            if descriptor is not None:
                self._fd_count -= 1
                os.close(descriptor)
            os.close(parent)

    def _file_generation(self, path: tuple[str, ...]) -> _File | None:
        return self._read_file(path, _MAX_FILE, required=False)

    def _open_dir_path(self, parts: tuple[str, ...]) -> int | None:
        if self._root_fd is None:
            return None
        descriptor = os.dup(self._root_fd)
        try:
            for component in parts:
                if not _valid_component(component):
                    raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
                try:
                    child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
                except FileNotFoundError:
                    os.close(descriptor)
                    return None
                except OSError as error:
                    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise _SourceProblem(Reason.SOURCE_UNTRUSTED) from error
                    raise
                metadata = os.fstat(child)
                os.close(descriptor)
                if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != _OWNER
                        or metadata.st_mode & _UNSAFE_RECORD_MODE):
                    os.close(child)
                    raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
                descriptor = child
            return descriptor
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    def _list_dir(self, parts: tuple[str, ...], reasons: dict[Reason, int | None]) -> tuple[str, ...]:
        descriptor = self._open_dir_path(parts)
        if descriptor is None:
            return ()
        try:
            remaining = max(0, _MAX_ENTRIES - self._entry_count)
            entries: list[str] = []
            with os.scandir(descriptor) as iterator:
                for entry in iterator:
                    if len(entries) >= remaining:
                        add_reason(reasons, Reason.DISCOVERY_LIMIT, None)
                        break
                    entries.append(entry.name)
            self._entry_count += len(entries)
            entries.sort()
            return tuple(entries)
        except OSError as error:
            add_reason(reasons, _reason_for_oserror(error), None)
            return ()
        finally:
            os.close(descriptor)

    def _inspect_file(self, path: tuple[str, ...]) -> bool:
        if self._root_fd is None or not path or any(not _valid_component(part) for part in path):
            raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
        parent = self._open_dir_path(path[:-1])
        if parent is None:
            return False
        descriptor: int | None = None
        try:
            try:
                descriptor = os.open(path[-1], _FILE_FLAGS, dir_fd=parent)
            except FileNotFoundError:
                return False
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.EISDIR, errno.ENXIO}:
                    raise _SourceProblem(Reason.SOURCE_UNTRUSTED) from error
                raise
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != _OWNER or info.st_nlink != 1
                    or info.st_mode & _UNSAFE_RECORD_MODE or not info.st_mode & stat.S_IRUSR):
                raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
            return True
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def _exists_file(self, path: tuple[str, ...], reasons: dict[Reason, int | None]) -> bool:
        try:
            return self._inspect_file(path)
        except _SourceProblem as error:
            add_reason(reasons, error.reason, None)
            return False
        except OSError as error:
            add_reason(reasons, _reason_for_oserror(error), None)
            return False

    def _stream_id(self, family: Family, space: Namespace, path: tuple[str, ...], run_id: object) -> bytes:
        return accounting.private_cursor_stream(self.key, family.value, space, path, run_id)

    def _room_for_record(self, out, reasons: dict[Reason, int | None]) -> bool:
        if len(out) < _MAX_RECORDS:
            return True
        add_reason(reasons, Reason.DISCOVERY_LIMIT, None)
        return False

    def _add_send_route(self, send_root, ordinal, row, reasons) -> None:
        route = self._read_document((*send_root, "wire", f"{ordinal:04d}.route.json"), required=False)
        if route is None:
            return
        role = route.get("role")
        if type(role) is str and role in {"read", "plan", "edit", "finish", "decide2", "unit"}:
            row["role"] = role
        else:
            add_reason(reasons, Reason.INVALID_FIELD, 1)
        model = route.get("model")
        if type(model) is str and 0 < len(model) <= 256:
            row["model"] = model

    def _policy_view(self) -> PolicyView:
        self._load_root_spec()
        assert self._root_spec is not None
        environment: dict[str, object] = {}
        model_alias = self._root_spec.get("api_key_env")
        if type(model_alias) is str and _ENV_NAME.fullmatch(model_alias) and model_alias.startswith("OTEL_"):
            environment[model_alias] = True
        profile = self._root_spec.get("profile")
        if type(profile) is not str or len(profile) > 64:
            profile = None
        resume = self._root_spec.get("resume")
        attempt = resume.get("attempt", 1) if type(resume) is dict else 1
        if type(attempt) is not int or not 1 <= attempt <= _MAX_ATTEMPTS:
            attempt = None
        grants: list[Path] = []
        settings = self._root_spec.get("tool_settings")
        if type(settings) is dict:
            for key in ("read_roots", "write_roots", "extra_read_roots", "extra_write_roots"):
                values = settings.get(key)
                if type(values) is list:
                    for value in values[:64]:
                        if type(value) is str and value and len(value) <= 4096:
                            grants.append(Path(value))
        authority_records = [self.root / "spec.json"]
        if self._exists_file(("tool-settings.json",), {}):
            authority_records.append(self.root / "tool-settings.json")
        source_generation = self._file_generation(("spec.json",))
        return PolicyView(self._root_seed, attempt, profile, MappingProxyType(environment),
                          tuple(dict.fromkeys(grants)), tuple(authority_records),
                          source_generation.generation if source_generation is not None else None)


def policy_view(root: Path) -> PolicyView:
    """Return bounded local policy metadata; this view is evidence, never an admission decision."""
    reader = Reader(root, bytes(32))
    try:
        if reader._root_error is not None:
            return PolicyView(None, None, None, MappingProxyType({}), (), (), None)
        try:
            return reader._policy_view()
        except (_SourceProblem, OSError, UnicodeError, ValueError, RecursionError):
            return PolicyView(None, None, None, MappingProxyType({}), (), (), None)
    finally:
        reader.close()


_PROJECTION_FIELDS = frozenset({
    "kind", "run_id", "timestamp_ns", "request_id", "step", "event_id", "request_ordinal",
    "role", "dispatch", "replayed", "transport_attempt_id", "transport_event_id", "status_code",
    "http_status", "elapsed_ns", "elapsed_seconds", "seconds", "send", "prompt_tokens",
    "completion_tokens", "check_kind", "kind_name", "status", "outcome", "execution_id",
    "host_execution_id", "execution", "reexecuted", "duration_seconds", "duration_s", "selector",
    "tool", "call_id", "exit_code", "process_id", "process", "pid", "clock_id", "clock", "clock_source",
    "request_sha256", "request_body_digest", "payload_hash", "body_sha256", "model", "cost",
})
_JOURNAL_TEXT_FIELDS = frozenset({
    "kind", "run_id", "request_id", "event_id", "transport_attempt_id", "transport_event_id",
    "call_id", "execution_id", "host_execution_id",
})
_JOURNAL_KINDS = frozenset({
    "request_submitted", "response_received", "response", "transport_error", "action_receipt",
    "check_receipt", "check_receipt_unchanged", "operation_refused", "operation_blocked", "tool_reply_receipt",
})


def _invalid_text_fields(row: dict[str, object]) -> bool:
    receipt = row.get("receipt")
    return (type(row.get("kind")) is not str
            or any(field in row and row[field] is not None and type(row[field]) is not str
                   for field in _JOURNAL_TEXT_FIELDS if field != "kind")
            or type(receipt) is dict and any(field in receipt and receipt[field] is not None
                                             and type(receipt[field]) is not str
                                             for field in ("kind", "result_class")))


def _projection_fields(family: Family, row: dict[str, object]) -> dict[str, object] | None:
    kind = row.get("kind")
    if family in {Family.JOURNAL, Family.PLAIN} and (type(kind) is not str or kind not in _JOURNAL_KINDS):
        return None
    result = {key: value for key, value in row.items() if key in _PROJECTION_FIELDS
              and (type(value) in {bool, int, float} or type(value) is str and len(value) <= (4096 if key == "selector" else 256))}
    for field, keys in (
        ("receipt", ("kind", "result_class", "elapsed_ns", "duration_seconds", "elapsed_seconds", "duration_s")),
        ("response", ("model", "cost")),
        ("accounting", ("prompt_tokens", "completion_tokens")),
        ("usage", ("prompt_tokens", "completion_tokens")),
    ):
        nested = row.get(field)
        if type(nested) is dict:
            result[field] = {key: nested[key] for key in keys if key in nested
                             and (type(nested[key]) in {bool, int, float}
                                  or type(nested[key]) is str and len(nested[key]) <= (4096 if key == "model" else 256))}
    if family is Family.SEND:
        identity = row.get("endpoint_identity")
        model = identity.get("model") if type(identity) is dict else None
        if type(model) is str and 0 < len(model) <= 256:
            result["model"] = model
        result["send_failed"] = type(row.get("error")) is str
    return result


def _reconcile_sends(occurrences, reasons) -> None:
    sends, submissions = defaultdict(list), defaultdict(list)
    totals, timed = Counter(), Counter()
    for item in occurrences:
        row = item.row
        if item.family is Family.SEND:
            totals[item.space] += 1
            if row.get("replayed") is not True:
                sends[item.space].append(item)
        elif (item.family in {Family.JOURNAL, Family.PLAIN} and item.epoch_ns is not None
              and row.get("kind") == "request_submitted" and row.get("dispatch") is not False
              and row.get("replayed") is not True and (item.family is Family.PLAIN or
                  type(row.get("request_id")) is str and row["request_id"]
                  and type(row.get("step")) is int and 0 <= row["step"] <= _MAX_ORDINAL)):
            submissions[item.space].append(item)
            timed[item.space] += 1
    for space, records in sends.items():
        by_digest, submitted = defaultdict(list), defaultdict(list)
        for send in records:
            digest = send.row.get("request_sha256")
            if _is_wire_digest(digest):
                by_digest[digest].append(send)
        for item in submissions[space]:
            row = item.row
            digest = row.get("request_body_digest", row.get("payload_hash", row.get("body_sha256")))
            if type(digest) is str and _is_wire_digest(digest := digest.removeprefix("sha256:")):
                submitted[digest].append(item)
        for digest, matches in submitted.items():
            matching_sends = by_digest[digest]
            # Repeated identical wires can only be ordered within one journal stream.
            if len(matches) == len(matching_sends) and len({item.cursor.stream for item in matches}) == 1:
                for item, send in zip(matches, matching_sends):
                    item.row["send_evidence"] = send.row
                    send.row["joined"] = True
        for send in records:
            if send.row.get("joined") is True:
                continue
            if send.epoch_ns is None or submissions[space]:
                send.row["unprojectable"] = True
                add_reason(reasons, Reason.SOURCE_MISSING, 1)
            else:
                timed[space] += 1
    for item in occurrences:
        if item.family is Family.SUMMARY:
            expected = item.row.get("provider_sends")
            observed = totals[item.space]
            if type(expected) is not int:
                expected, observed = item.row.get("requests"), timed[item.space]
            if type(expected) is int and expected != observed:
                add_reason(reasons, Reason.SOURCE_MISSING, abs(expected - observed))


def _is_wire_digest(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _open_absolute_dir(path: Path) -> int:
    if not path.is_absolute() or path == Path(path.anchor or "/") or ".." in path.parts:
        raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
    flags = _DIR_FLAGS
    descriptor = os.open(path.anchor or "/", flags)
    from .authority import unsafe_parent_mode

    current = Path(path.anchor or "/")
    try:
        for component in path.parts[1:]:
            if component in ("", "."):
                continue
            if component == "..":
                raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
            child = os.open(component, flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            current = current / component
            last = current == path
            if (not stat.S_ISDIR(metadata.st_mode)
                    or (last and (metadata.st_uid != _OWNER or metadata.st_mode & 0o077))
                    or unsafe_parent_mode(current, metadata)):
                os.close(child)
                raise _SourceProblem(Reason.SOURCE_UNTRUSTED)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _valid_component(value: str) -> bool:
    return (type(value) is str and value not in ("", ".", "..") and "/" not in value
            and "\\" not in value and "\x00" not in value and len(value.encode("utf-8", "strict")) <= 255)


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode


def _generation(
    key: bytes,
    identity: tuple[int, int, int, int, int, int],
    *,
    appendable: bool = False,
) -> bytes:
    values = (identity[0], identity[1], identity[5]) if appendable else identity
    material = b"".join(value.to_bytes(8, "big", signed=False) for value in values)
    return accounting.keyed_identity(key, "generation", (material,), 32)


def _decode_document(data: bytes) -> dict[str, object]:
    if not data or len(data) > _MAX_RECORD:
        raise _SourceProblem(Reason.SOURCE_CORRUPT)
    def oversize() -> None:
        raise _SourceProblem(Reason.SOURCE_OVERSIZE)

    try:
        value = _load_bounded_json(data, oversize=oversize)
    except (ValueError, RecursionError) as error:
        raise _SourceProblem(Reason.SOURCE_CORRUPT) from error
    if type(value) is not dict:
        raise _SourceProblem(Reason.SOURCE_CORRUPT)
    return value


def _decompress_bounded(data: bytes, limit: int) -> tuple[bytes, bool]:
    if type(limit) is not int or limit < 0:
        raise _SourceProblem(Reason.INVALID_FIELD)
    decoder = lzma.LZMADecompressor(memlimit=16 << 20)
    chunks: list[bytes] = []
    total = 0
    for offset in range(0, len(data), _CHUNK):
        chunk = data[offset:offset + _CHUNK]
        pending = chunk
        while pending or not decoder.needs_input:
            if total >= limit:
                return b"".join(chunks), False
            output = decoder.decompress(pending, max_length=min(_CHUNK, limit - total))
            pending = b""
            if output:
                total += len(output)
                chunks.append(output)
            if decoder.eof:
                if decoder.unused_data or offset + len(chunk) < len(data):
                    raise _SourceProblem(Reason.SOURCE_CORRUPT)
                return b"".join(chunks), True
            if total >= limit:
                return b"".join(chunks), False
            if decoder.needs_input:
                break
    if not decoder.eof:
        raise _SourceProblem(Reason.SOURCE_CORRUPT)
    return b"".join(chunks), True


def _sequence(row: dict[str, object], fallback: int, *, plain: bool = False) -> int:
    keys = ("sequence",) if plain else ("sequence", "send", "ordinal", "seq")
    for key in keys:
        value = row.get(key)
        if type(value) is int and 1 <= value <= (1 << 31) - 1:
            return value
    return fallback


def _attempt_from_spec(document: dict[str, object]) -> int:
    resume = document.get("resume")
    if resume is None:
        return 1
    if type(resume) is not dict:
        raise _SourceProblem(Reason.SOURCE_CORRUPT)
    attempt = resume.get("attempt")
    if type(attempt) is not int or not 1 <= attempt <= _MAX_ATTEMPTS:
        raise _SourceProblem(Reason.SOURCE_CORRUPT)
    return attempt


def _epoch_ns(row: dict[str, object], family: Family) -> int | None:
    if family is Family.SPEC:
        value = row.get("created")
        if (type(value) is not str or len(value) > 64
                or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", value)):
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        try:
            delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
            nanos = (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
        except (OverflowError, OSError, ValueError):
            return None
        return nanos if 1 <= nanos <= (1 << 63) - 1 else None
    value = row.get("timestamp_ns")
    if type(value) is not int or not 1 <= value <= (1 << 63) - 1:
        return None
    return value


def _head(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _valid_check_record_hash(row: dict[str, object]) -> bool:
    supplied = row.get("record_hash")
    if type(supplied) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", supplied):
        return False
    try:
        body = json.dumps(
            {key: value for key, value in row.items() if key != "record_hash"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        return False
    return supplied == "sha256:" + hashlib.sha256(body).hexdigest()


def _resume_checkpoint(spec: dict[str, object] | None) -> dict[str, object] | None:
    resume = spec.get("resume") if spec is not None else None
    checkpoint = resume.get("checkpoint") if type(resume) is dict else None
    return checkpoint if type(checkpoint) is dict and checkpoint.get("status") != "skipped" else None


def _unit_binding(key: bytes, space: Namespace, call: str, unit: str,
                  spec: dict[str, object] | None) -> tuple[bytes, int]:
    call_id = spec.get("call_id") if spec is not None else None
    unit_id = spec.get("unit_id") if spec is not None else None
    incarnation = spec.get("incarnation") if spec is not None else None
    call_part = call_id.encode("utf-8") if type(call_id) is str and len(call_id) <= 256 else call.encode("utf-8")
    unit_part = unit_id.encode("utf-8") if type(unit_id) is str and len(unit_id) <= 256 else unit.encode("utf-8")
    opaque = accounting.keyed_identity(
        key, "unit", (accounting.namespace_material(space), call_part, unit_part), 16,
    )
    if type(incarnation) is not int or not 1 <= incarnation <= (1 << 31) - 1:
        incarnation = 1
    return opaque, incarnation


def _parent_dispositions(result, reasons) -> dict[str, str]:
    records = result.get("outcomes") if result is not None else []
    if type(records) is not list:
        add_reason(reasons, Reason.INVALID_FIELD, 1)
        return {}
    statuses = {}
    for item in records:
        if type(item) is not dict or type(item.get("unit_id")) is not str or type(item.get("status")) is not str:
            add_reason(reasons, Reason.INVALID_FIELD, 1)
            continue
        statuses[item["unit_id"]] = item["status"]
    return statuses


def _host_occurrence(row: dict[str, object], call: str, unit: str,
                     incarnation: int, parent_terminal: str | None) -> dict[str, object] | None:
    created = _finite_epoch_seconds(row.get("created_epoch"))
    started = _finite_epoch_seconds(row.get("started_epoch"))
    finished = _finite_epoch_seconds(row.get("finished_epoch"))
    if created is None:
        return None
    terminal = row.get("terminal")
    kind = "unit_finished" if finished is not None else "unit_started" if started is not None else "unit_registered"
    slot_wait = started - created if started is not None and started >= created else None
    return {
        "kind": kind,
        "sequence": 2 if finished is not None else 1,
        "call_id": call,
        "unit_id": unit,
        "attempt": row.get("attempt"),
        "incarnation": incarnation,
        "created_epoch": created,
        "started_epoch": started,
        "finished_epoch": finished,
        "queued": row.get("queued") if type(row.get("queued")) is bool else None,
        "terminal": terminal,
        "parent_disposition": parent_terminal,
        "exited_seq": row.get("exited_seq") if type(row.get("exited_seq")) is int else None,
        "slot_wait_seconds": slot_wait,
    }


def _finite_epoch_seconds(value: object) -> float | None:
    if type(value) is int:
        if not 0 < value <= ((1 << 63) - 1) // 1_000_000_000:
            return None
        return float(value)
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        return None
    if value > ((1 << 63) - 1) / 1_000_000_000:
        return None
    return float(value)



def _reason_for_oserror(error: OSError) -> Reason:
    if error.errno in {errno.EAGAIN, errno.EWOULDBLOCK, errno.EBUSY}:
        return Reason.SOURCE_BUSY
    if error.errno in {errno.ENOENT, errno.ENOTDIR}:
        return Reason.SOURCE_MISSING
    if error.errno in {errno.ELOOP, errno.EACCES, errno.EPERM, errno.ENXIO}:
        return Reason.SOURCE_UNTRUSTED
    return Reason.SOURCE_CORRUPT
