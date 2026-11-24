from __future__ import annotations

import json
import lzma
import os
from pathlib import Path

import pytest

from stateframe.otel.identity import from_seed
from stateframe.otel.levels import Levels, model_id
from stateframe.otel.projection import Projector
from stateframe.otel.sources import Reader
from stateframe.otel.types import Coverage, Family


KEY = bytes(range(32))
SEED = "00112233445566778899aabbccddeeff"
EPOCH = 1_730_000_000_000_000_000


def _write_json(path: Path, value: object, *, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for directory in path.parents:
        directory.chmod(0o700)
        if directory == root:
            break
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    path.chmod(0o600)


def _write_run(root: Path, *, attempt: int = 1) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    spec: dict[str, object] = {
        "schema": "stateframe-run/1",
        "created": "2026-11-22T00:00:00+00:00",
        "run_identity_seed": SEED,
        "profile": "restricted",
    }
    if attempt != 1:
        spec["resume"] = {"attempt": attempt}
    _write_json(root / "spec.json", spec, root=root)


def _journal(root: Path, rows: list[dict[str, object]], *, attempt_path: Path | None = None) -> Path:
    source_root = attempt_path or root
    directory = source_root / "worker" / "measurement-runs" / "one"
    path = directory / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in path.parent.parents:
        candidate.chmod(0o700)
        if candidate == source_root:
            break
    path.parent.chmod(0o700)
    path.write_bytes(b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    ))
    path.chmod(0o600)
    return path


def _event(sequence: int, kind: str = "request_submitted", **fields: object) -> dict[str, object]:
    return {
        "schema": "stateframe-audit/1",
        "kind": kind,
        "run_id": "journal-opaque",
        "event_id": f"journal-opaque:{sequence}",
        "sequence": sequence,
        "timestamp_ns": EPOCH + sequence,
        "previous_hash": None,
        **fields,
    }


def _value(value: object) -> str:
    return str(getattr(value, "value", value))


def test_reader_repeated_poll_does_not_repeat_returned_occurrences(private_root):
    root = private_root / "run"
    _write_run(root)
    _journal(root, [_event(1)])
    reader = Reader(root, KEY)
    try:
        first, _first_coverage = reader.poll()
        second, _second_coverage = reader.poll()
    finally:
        reader.close()
    assert first
    assert second == ()


def test_reader_orders_spec_before_evidence_and_terminal_last(private_root):
    root = private_root / "run"
    _write_run(root)
    _journal(root, [_event(1)])
    _write_json(root / "summary.json", {
        "runtime_outcome": "completed", "check_outcome": "pass", "artifact": "saved",
    }, root=root)

    reader = Reader(root, KEY)
    try:
        rows, _coverage = reader.poll()
    finally:
        reader.close()
    families = [_value(row.family) for row in rows]
    assert families.index("spec") < families.index("journal")
    assert families.index("journal") < families.index("summary")


def test_distinct_sources_with_shared_run_id_keep_distinct_streams(private_root):
    root = private_root / "run"
    _write_run(root)
    _journal(root, [_event(1)])
    second = root / "worker" / "measurement-runs" / "two" / "events.jsonl"
    second.parent.mkdir(parents=True)
    second.parent.chmod(0o700)
    second.write_text(json.dumps(_event(1), separators=(",", ":")) + "\n", encoding="utf-8")
    second.chmod(0o600)

    reader = Reader(root, KEY)
    try:
        rows, _coverage = reader.poll()
    finally:
        reader.close()
    journal_rows = [row for row in rows if _value(row.family) == "journal"]
    assert len(journal_rows) == 2
    assert len({row.cursor.stream for row in journal_rows}) == 2


def test_source_symlink_fifo_and_hardlink_rejected(private_root):
    outside = private_root / "outside.jsonl"
    outside.write_bytes(json.dumps(_event(1)).encode() + b"\n")
    outside.chmod(0o600)

    roots = [private_root / "symlink-run", private_root / "fifo-run", private_root / "hardlink-run"]
    for root in roots:
        _write_run(root)
    symlink = _journal(roots[0], [])
    symlink.unlink()
    symlink.symlink_to(outside)
    fifo = _journal(roots[1], [])
    fifo.unlink()
    os.mkfifo(fifo, 0o600)
    hardlink = _journal(roots[2], [_event(1)])
    alias = hardlink.with_name("alias.jsonl")
    os.link(hardlink, alias)

    for root in roots:
        reader = Reader(root, KEY)
        try:
            rows, coverage = reader.poll()
        finally:
            reader.close()
        assert not [row for row in rows if _value(row.family) == "journal"]
        assert not coverage.complete


def test_group_writable_records_are_confined_by_the_private_run_root(private_root):
    from stateframe.otel.sources import policy_view

    root = private_root / "run"
    _write_run(root)
    journal = _journal(root, [_event(1)])
    (root / "spec.json").chmod(0o664)
    journal.chmod(0o664)
    journal.parent.chmod(0o775)

    assert policy_view(root).seed == bytes.fromhex(SEED)
    reader = Reader(root, KEY)
    try:
        rows, _ = reader.poll()
    finally:
        reader.close()
    assert [row.index for row in rows if _value(row.family) == "journal"] == [1]

    root.chmod(0o770)
    assert policy_view(root).seed is None


def test_replaced_and_sealed_journal_same_occurrences(private_root):
    root = private_root / "run"
    _write_run(root)
    path = _journal(root, [_event(1), _event(2, "response_received")])

    reader = Reader(root, KEY)
    try:
        loose, _ = reader.poll()
    finally:
        reader.close()
    loose_rows = [row for row in loose if _value(row.family) == "journal"]
    plain = path.read_bytes()
    sealed = path.with_name(path.name + ".xz")
    sealed.write_bytes(lzma.compress(plain))
    sealed.chmod(0o600)
    path.unlink()

    reader = Reader(root, KEY)
    try:
        archived, _ = reader.poll()
    finally:
        reader.close()
    archived_rows = [row for row in archived if _value(row.family) == "journal"]

    assert [(row.cursor.stream, row.index, row.row["kind"]) for row in loose_rows] == [
        (row.cursor.stream, row.index, row.row["kind"]) for row in archived_rows
    ]


def test_interrupted_archive_preserves_occurrences(private_root):
    root = private_root / "run"
    _write_run(root, attempt=2)
    archived = root / "attempts" / "1"
    _write_json(archived / "spec.json", {
        "schema": "stateframe-run/1",
        "created": "2026-11-22T00:00:00+00:00",
        "run_identity_seed": SEED,
    }, root=root)
    old_path = _journal(root, [_event(1)], attempt_path=archived)
    old_path_xz = old_path.with_name(old_path.name + ".xz")
    old_path_xz.write_bytes(lzma.compress(old_path.read_bytes()))
    old_path_xz.chmod(0o600)
    old_path.unlink()
    _journal(root, [_event(1, "response_received"), _event(2, "request_submitted")])

    reader = Reader(root, KEY)
    try:
        rows, coverage = reader.poll()
    finally:
        reader.close()
    journal_rows = [row for row in rows if _value(row.family) == "journal"]
    assert {row.space.attempt for row in journal_rows} == {1, 2}
    assert {row.row["kind"] for row in journal_rows} >= {"request_submitted", "response_received"}
    assert coverage.complete


def test_restored_send_prefix_does_not_repeat_physical_work(private_root):
    root = private_root / "run"
    _write_run(root, attempt=2)
    archived = root / "attempts" / "1"
    _write_run(archived)
    archived.parent.chmod(0o700)
    spec_path = root / "spec.json"
    spec = json.loads(spec_path.read_bytes())
    spec["resume"].update(previous_attempt=str(archived), checkpoint={"sends": 1})
    _write_json(spec_path, spec, root=root)
    physical = b'{"send":1,"success":true,"prompt_tokens":21}\n'
    for path, data in (
        (archived / "sends.jsonl", physical),
        (root / "sends.jsonl", physical + b'{"send":2,"success":true,"prompt_tokens":13}\n'),
    ):
        path.write_bytes(data)
        path.chmod(0o600)

    reader = Reader(root, KEY)
    try:
        rows, _ = reader.poll()
    finally:
        reader.close()
    assert [(row.space.attempt, row.row["send"]) for row in rows
            if _value(row.family) == "send" and row.row.get("replayed") is not True] == [(1, 1), (2, 2)]


@pytest.mark.parametrize("matches", [True, False])
def test_worker_journal_requires_matching_execution_identity(private_root, matches):
    root = private_root / "run"
    _write_run(root)
    call = root / "parallel" / "call-a"
    unit = call / "units" / "unit-a"
    _write_json(unit / "unit-spec.json", {
        "schema": "stateframe-unit-spec/3", "call_id": "call-a", "unit_id": "unit-a",
        "execution_id": "execution-a", "start_manifest_digest": "a" * 64,
    }, root=root)
    _write_json(call / "host" / "units" / "unit-a" / "1.json", {
        "schema": "stateframe-parallel-host-worker/1", "call_id": "call-a", "unit_id": "unit-a",
        "execution_id": "execution-a" if matches else "execution-b",
        "start_manifest_digest": "a" * 64, "incarnation": 1,
        "created_epoch": EPOCH / 1e9, "started_epoch": EPOCH / 1e9 + 1,
    }, root=root)
    _journal(root, [_event(1)], attempt_path=unit)

    reader = Reader(root, KEY)
    try:
        rows, coverage = reader.poll()
    finally:
        reader.close()
    assert bool([row for row in rows if _value(row.family) == "journal" and row.space.unit]) is matches
    if not matches:
        assert not coverage.complete


@pytest.mark.parametrize(("terminal", "state", "exit_code", "outcome", "disposition"), [
    ("SUBMITTED", "EXITED", 0, "ok", "unknown"),
    (None, "EXITED", 0, "unknown", "unknown"),
    ("ENGINE_ERROR", "EXITED", 1, "error", "unknown"),
    ("REFUSED", "EXITED", 1, "refused", "unknown"),
    ("BUDGET_EXHAUSTED", "EXITED", 1, "exhausted", "unknown"),
    (None, "RUNNING", None, "unknown", "unknown"),
])
def test_worker_terminal_projects_its_outcome_and_disposition(
    private_root, terminal, state, exit_code, outcome, disposition,
):
    root = private_root / "run"
    _write_run(root)
    call = root / "parallel" / "call-a"
    unit = call / "units" / "unit-a"
    _write_json(unit / "unit-spec.json", {
        "schema": "stateframe-unit-spec/3", "call_id": "call-a", "unit_id": "unit-a",
        "execution_id": "execution-a", "start_manifest_digest": "a" * 64,
    }, root=root)
    _write_json(call / "host" / "units" / "unit-a" / "1.json", {
        "schema": "stateframe-parallel-host-worker/1", "call_id": "call-a", "unit_id": "unit-a",
        "execution_id": "execution-a", "start_manifest_digest": "a" * 64, "incarnation": 1,
        "created_epoch": EPOCH / 1e9, "started_epoch": EPOCH / 1e9 + 1,
        "finished_epoch": EPOCH / 1e9 + 2, "state": state, "terminal": terminal, "exit_code": exit_code,
    }, root=root)

    reader = Reader(root, KEY)
    try:
        occurrences, coverage = reader.poll()
    finally:
        reader.close()
    assert coverage.complete
    worker = next(row for row in occurrences if row.family is Family.PARALLEL)
    run_ref, trace = from_seed(KEY, bytes.fromhex(SEED))
    projector = Projector(KEY, run_ref, trace)
    spans = [span for row in occurrences for batch in projector.consume(row) for span in batch.spans]
    spans.extend(span for batch in projector.seal(worker.space, Coverage(True, ())) for span in batch.spans)
    subframe = next(span for span in spans if span.name == "stateframe.subframe")
    attrs = {attr.key: attr.value for attr in subframe.attrs}
    assert attrs["stateframe.outcome"] == outcome
    assert attrs["stateframe.parallel.disposition"] == disposition


@pytest.mark.parametrize(("field", "value", "kind"), [
    ("kind", [], "request_submitted"),
    ("kind", 7, "request_submitted"),
    ("event_id", [], "request_submitted"),
    ("transport_attempt_id", {}, "response_received"),
    ("request_id", {}, "request_submitted"),
    ("receipt.kind", [], "action_receipt"),
    ("parallel.schema", [], "request_submitted"),
])
def test_malformed_record_text_fields_damage_coverage_and_keep_other_records(private_root, field, value, kind):
    root = private_root / "run"
    _write_run(root)
    if field == "parallel.schema":
        _journal(root, [_event(1)])
        _write_json(root / "parallel/call-a/result.json", {"schema": value, "call_id": "call-a"}, root=root)
    else:
        damaged = _event(2, kind)
        if "." in field:
            damaged["receipt"] = {field.split(".", 1)[1]: value}
        else:
            damaged[field] = value
        _journal(root, [_event(1), damaged])

    reader = Reader(root, KEY)
    try:
        rows, coverage = reader.poll()
    finally:
        reader.close()

    assert [row.index for row in rows if row.family is Family.JOURNAL] == [1]
    assert not coverage.complete


def test_fake_sf_stdout_is_not_evidence(private_root):
    root = private_root / "run"
    _write_run(root)
    (root / "stdout.log").write_text(
        '{"schema":"stateframe-audit/1","kind":"response_received","sequence":1}\n',
        encoding="utf-8",
    )
    (root / "stdout.log").chmod(0o600)

    reader = Reader(root, KEY)
    try:
        rows, _coverage = reader.poll()
    finally:
        reader.close()
    assert not [row for row in rows if row.row.get("kind") == "response_received"]


def test_compaction_and_gc_do_not_fabricate_coverage(private_root):
    root = private_root / "run"
    _write_run(root)
    _write_json(root / "summary.json", {"requests": 99, "runtime_outcome": "completed"}, root=root)
    _write_json(root / "checkpoints" / "checkpoint-refused.json", {"reason": "compacted"}, root=root)

    reader = Reader(root, KEY)
    try:
        rows, coverage = reader.poll()
    finally:
        reader.close()
    assert not [row for row in rows if _value(row.family) == "journal"]
    assert not coverage.complete


@pytest.mark.parametrize("retained", [0, 1, 4])
@pytest.mark.parametrize("summary_count", [3, 4, 5])
def test_send_evidence_reconciles_coverage_and_supplies_details(private_root, retained, summary_count):
    root = private_root / "run"
    _write_run(root)
    _write_json(root / "summary.json", {"provider_sends": summary_count}, root=root)
    rows, sends = [], []
    for ordinal in range(1, 5):
        digest = f"{ordinal:064x}"
        sends.append({"send": ordinal, "request_sha256": digest, "prompt_tokens": 10 * ordinal,
                      "completion_tokens": ordinal})
        _write_json(root / "wire" / f"{ordinal:04d}.route.json",
                    {"role": "read", "model": f"local-model-{ordinal}"}, root=root)
        if ordinal <= retained:
            fields = {"request_id": f"request-{ordinal}", "step": ordinal}
            rows.extend([
                _event(ordinal * 2 - 1, request_body_digest="sha256:" + digest, **fields),
                _event(ordinal * 2, "response_received", elapsed_ns=1, **fields),
            ])
    _journal(root, rows)
    (root / "sends.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sends))
    reader = Reader(root, KEY)
    try:
        occurrences, coverage = reader.poll()
    finally:
        reader.close()
    run, trace = from_seed(KEY, bytes.fromhex(SEED))
    projector = Projector(KEY, run, trace, levels=Levels(include_cost=True), price_rates=(2, 4))
    spans = [span for occurrence in occurrences for item in projector.consume(occurrence) for span in item.spans]
    spans.extend(span for item in projector.seal(occurrences[0].space, Coverage(True, ())) for span in item.spans)
    assert coverage.complete is (retained == 4 and summary_count == 4)
    chats = [span for span in spans if span.name == "chat"]
    assert len(chats) == retained
    for ordinal, chat in enumerate(chats, 1):
        attrs = {attr.key: attr.value for attr in chat.attrs}
        assert attrs["gen_ai.request.model"] == model_id(KEY, f"local-model-{ordinal}")
        assert attrs["gen_ai.usage.input_tokens"] == 10 * ordinal
        assert attrs["gen_ai.usage.output_tokens"] == ordinal
        assert attrs["stateframe.cost.microunits"] == 24 * ordinal


def test_torn_tail_reports_incomplete_coverage_and_preserves_complete_rows(private_root):
    root = private_root / "run"
    _write_run(root)
    path = _journal(root, [_event(1)])
    with path.open("ab") as stream:
        stream.write(b'{"kind":"response_received","sequence":2')
    reader = Reader(root, KEY)
    try:
        rows, coverage = reader.poll()
    finally:
        reader.close()
    assert [row.index for row in rows if _value(row.family) == "journal"] == [1]
    assert not coverage.complete
    assert [reason.value for reason, _ in coverage.reasons] == ["torn_tail"]


def test_compressed_journal_reads_past_the_former_decoded_window(private_root):
    root = private_root / "run"
    _write_run(root)
    path = _journal(root, [_event(index, payload="x" * 16_000) for index in range(1, 301)])
    raw = path.read_bytes()
    assert len(raw) > 4 * 1024 * 1024
    sealed = path.with_suffix(".jsonl.xz")
    sealed.write_bytes(lzma.compress(raw))
    path.unlink()
    reader = Reader(root, KEY)
    try:
        rows, coverage = reader.poll()
    finally:
        reader.close()
    journal = [row for row in rows if _value(row.family) == "journal"]
    assert coverage.complete
    assert [row.index for row in journal] == list(range(1, 301))
    assert all("payload" not in row.row for row in journal)


def test_replayed_model_exchange_does_not_hide_an_identical_fresh_send(private_root):
    root = private_root / "run"
    _write_run(root)
    digest = "a" * 64
    _journal(root, [
        _event(1, request_body_digest="sha256:" + digest, request_id="replay", step=1),
        _event(2, "response_received", transport_attempt_id="journal-opaque:1", request_id="replay", step=1),
        _event(3, request_body_digest="sha256:" + digest, request_id="fresh", step=2),
        _event(4, "response_received", transport_attempt_id="journal-opaque:3", request_id="fresh", step=2),
    ])
    (root / "sends.jsonl").write_text(
        json.dumps({"send": 1, "request_sha256": digest, "replayed": True}) + "\n"
        + json.dumps({"send": 2, "request_sha256": digest}) + "\n",
        encoding="utf-8",
    )
    reader = Reader(root, KEY)
    try:
        rows, coverage = reader.poll()
    finally:
        reader.close()
    assert coverage.complete
    assert [item.index for item in rows if item.family == "journal" and item.row.get("replayed")] == [1, 2]
    assert [item.index for item in rows if item.family == "journal" and not item.row.get("replayed")] == [3, 4]
    fresh = next(item for item in rows if item.family == "journal" and item.index == 3)
    assert fresh.row["send_evidence"]["send"] == 2
