"""Deterministic metadata projection from trusted source occurrences."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone

from .. import __version__
from .levels import Levels, extra_attributes

from . import accounting
from .identity import namespace_bytes
from .operations import (
    ROLES,
    epoch_seconds_ns,
    fixed_check_kind,
    fixed_outcome,
    fixed_tool,
    finite_duration,
    outcome_for,
    terminal_fields,
)
from .schema import coverage_merge
from .types import (
    AttrKey,
    Coverage,
    Cursor,
    Family,
    Namespace,
    Projection,
    Reason,
    RunRef,
    SafeAttr,
    SafeLink,
    SafeSpan,
    SourceOccurrence,
)


_MAX_TOKEN = (1 << 31) - 1
_MAX_ORDINAL = (1 << 31) - 1
_MAX_EPOCH = (1 << 63) - 1
@dataclass(frozen=True, slots=True)
class _DetailRecord:
    request_model: object = None
    response_model: object = None
    provider: object = None
    engine_version: str | None = None
    reported_cost: object = None
    price_table: tuple[int | float, int | float] | None = None
    usage: tuple[int, int] | None = None


class Projector:
    """Build stable spans from one run's source records."""

    def __init__(self, key: bytes, run: RunRef, trace_id: bytes, *, levels: Levels = Levels(),
                 detail_attempt: int = 1, price_rates: tuple[int | float, int | float] | None = None):
        if type(key) is not bytes or len(key) != 32 or type(run) is not RunRef or len(run.value) != 16:
            raise ValueError("invalid_field")
        if type(trace_id) is not bytes or len(trace_id) != 16 or not any(trace_id):
            raise ValueError("invalid_field")
        self.key = key
        self.run = run
        self.trace_id = trace_id
        self._levels = levels
        self._detail_attempt = detail_attempt
        self._price_table = price_rates
        self._detail_records: dict[int, _DetailRecord] = {}
        self._coverage = Coverage(True, ())
        self._run_context: bytes | None = None
        self._attempt_contexts: dict[int, bytes] = {}
        self._parallel_contexts: dict[tuple[int, bytes], bytes] = {}
        self._subframe_contexts: dict[tuple[int, bytes | None, int | None, bytes | None], bytes] = {}
        self._attempt_anchor_occurrences: dict[int, SourceOccurrence] = {}
        self._pending_requests: dict[tuple, SourceOccurrence] = {}
        self._pending_transports: dict[tuple[Namespace, bytes, str], tuple] = {}
        self._pending_tools: dict[tuple, SourceOccurrence] = {}
        self._seen_checks: set[tuple] = set()
        self._seen_occurrences: set[bytes] = set()
        self._engine_results: dict[Namespace, tuple[bytes, SourceOccurrence]] = {}
        self._terminal_spaces: set[Namespace] = set()
        self._unit_starts: dict[tuple, tuple[SourceOccurrence, int]] = {}
        self._completed_units: set[tuple] = set()
        self._boundary_occurrences: dict[tuple[int, bytes | None, int | None], SourceOccurrence] = {}

    def consume(self, row: SourceOccurrence) -> tuple[Projection, ...]:
        if type(row) is not SourceOccurrence or type(row.row) is not dict:
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.INVALID_FIELD, 1),)))
            return ()
        if (type(row.space) is not Namespace or row.space.run != self.run or type(row.index) is not int
                or row.index < 1 or row.index > _MAX_ORDINAL or not _valid_namespace(row.space)
                or not _valid_cursor(row.cursor)):
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.INVALID_FIELD, 1),)))
            return ()
        family = _family(row.family)
        if family is None:
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.INVALID_FIELD, 1),)))
            return ()
        space_key = _space_key(row.space)
        anchor_record = (self._boundary_occurrences.get(space_key)
                         or self._attempt_anchor_occurrences.get(row.space.attempt))
        if family is Family.STOPPED and anchor_record is None:
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.TIMING_UNAVAILABLE, 1),)))
            return ()
        try:
            identity_record = anchor_record if family is Family.STOPPED else row
            identity_name = "observation_end" if family is Family.STOPPED else family.value
            occurrence_key = accounting.record_identity(
                self.key, "occurrence", identity_name, identity_record, size=16,
            )
        except (TypeError, ValueError, UnicodeError):
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.INVALID_FIELD, 1),)))
            return ()
        if occurrence_key in self._seen_occurrences:
            return ()
        if len(self._seen_occurrences) >= 65536:
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.DISCOVERY_LIMIT, None),)))
            return ()
        self._seen_occurrences.add(occurrence_key)
        if (_valid_epoch(row.epoch_ns) and family in {Family.JOURNAL, Family.PLAIN}
                and (space_key not in self._boundary_occurrences
                     or row.epoch_ns > self._boundary_occurrences[space_key].epoch_ns
                     or row.epoch_ns == self._boundary_occurrences[space_key].epoch_ns
                     and occurrence_key > accounting.record_identity(
                         self.key, "occurrence", str(self._boundary_occurrences[space_key].family),
                         self._boundary_occurrences[space_key], size=16,
                     ))):
            self._boundary_occurrences[space_key] = row
        spans = []
        if family is Family.SPEC:
            spans = self._contexts_from_spec(row)
        elif family is Family.SUMMARY:
            self._engine_results.pop(row.space, None)
            spans = self._terminal(row)
            for check_kind, field in (("baseline", "tests_before"), ("final", "tests_after")):
                check = row.row.get(field)
                if type(check) is dict and type(check.get("exit_code")) is int:
                    spans.extend(self._check(row, check, check_kind=check_kind))
        elif family is Family.ENGINE_RESULT:
            self._engine_results[row.space] = (occurrence_key, row)
        elif family is Family.STOPPED:
            spans = self._stopped(row)
        elif family in {Family.JOURNAL, Family.PLAIN}:
            spans = self._request_events(row, family)
            spans.extend(self._journal_operation(row))
        elif family is Family.SEND:
            spans = self._send(row)
        elif family is Family.PARALLEL:
            spans = self._parallel(row)
        elif family is Family.TOOL:
            spans = self._tool(row)
        return (Projection(occurrence_key, row.space, row.cursor, tuple(spans), self._coverage),)

    @property
    def coverage(self) -> Coverage:
        return self._coverage

    def seal(self, space: Namespace, coverage: Coverage) -> tuple[Projection, ...]:
        self._coverage = coverage_merge(self._coverage, coverage)
        pending = [key for key, row in self._pending_requests.items() if row.space == space]
        output = []
        for key in pending:
            row = self._pending_requests.pop(key)
            self._forget_transport(row)
            spans = self._model_span(row, None)
            identity = accounting.record_identity(self.key, "occurrence", "pending_request", row, size=16)
            output.append(Projection(identity, space, row.cursor, tuple(spans), self._coverage))
        engine_result = self._engine_results.pop(space, None)
        if engine_result is not None:
            identity, row = engine_result
            spans = self._terminal(row)
            output.append(Projection(identity, space, row.cursor, tuple(spans), self._coverage))
        return tuple(output)

    def _contexts_from_spec(self, occurrence: SourceOccurrence) -> list[SafeSpan]:
        run_epoch = occurrence.epoch_ns
        if not _valid_epoch(run_epoch):
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.TIMING_UNAVAILABLE, 1),)))
            return []
        attempt = occurrence.space.attempt
        record = occurrence.row
        if attempt not in self._detail_records:
            self._detail_records[attempt] = self._spec_detail(record, attempt)
        detail = self._detail_records[attempt]
        epoch = _spec_attempt_epoch(occurrence.row, attempt, run_epoch)
        output: list[SafeSpan] = []
        if self._run_context is None:
            run_space = Namespace(self.run, 0, None, None)
            run_span = accounting.point_span(
                self.key, "stateframe.run.context", self.trace_id, occurrence, None,
                self._base_attrs(occurrence.space, "record", "registration", True, include_attempt=False),
                at_ns=run_epoch, context_tag="run", identity_space=run_space,
            )
            if run_span is not None:
                self._run_context = run_span.span_id
                output.append(run_span)
        attempt_id = self._attempt_contexts.get(attempt)
        if attempt_id is None:
            if not _valid_epoch(epoch):
                self._coverage = coverage_merge(
                    self._coverage, Coverage(False, ((Reason.TIMING_UNAVAILABLE, 1),)),
                )
                return output
            attempt_space = Namespace(self.run, attempt, None, None)
            previous = self._attempt_contexts.get(attempt - 1)
            links = (SafeLink(self.trace_id, previous),) if previous is not None else ()
            attempt_span = accounting.point_span(
                self.key, "stateframe.attempt.context", self.trace_id, occurrence, self._run_context,
                [*self._base_attrs(attempt_space, "record", "registration", True),
                 *self._detail_attrs(attempt, detail)], at_ns=epoch, links=links,
                context_tag="attempt", identity_space=attempt_space,
            )
            if attempt_span is not None:
                attempt_id = attempt_span.span_id
                self._attempt_contexts[attempt] = attempt_id
                self._attempt_anchor_occurrences[attempt] = occurrence
                output.append(attempt_span)
        return output

    def _parent(self, space: Namespace) -> bytes | None:
        if space.unit is None:
            return self._attempt_contexts.get(space.attempt)
        candidates = {identifier for (attempt, unit, incarnation, _), identifier in self._subframe_contexts.items()
                      if (attempt, unit, incarnation) == _space_key(space)}
        return next(iter(candidates)) if len(candidates) == 1 else None

    def _spec_detail(self, row: dict[str, object], attempt: int) -> _DetailRecord:
        provider = row.get("provider")
        resume = row.get("resume")
        model = row.get("model")
        if type(resume) is dict and type(resume.get("model_swap")) is dict:
            model = resume["model_swap"].get("served", model)
        return _DetailRecord(request_model=model, provider=provider, engine_version=__version__)

    def _attempt_levels(self, attempt: int) -> Levels:
        return self._levels if attempt == self._detail_attempt else Levels()

    def _detail_attrs(self, attempt: int, record: _DetailRecord) -> list[SafeAttr]:
        return [_attr(name, value) for name, value in extra_attributes(
            self._attempt_levels(attempt), record, self.key,
        )]

    def _ensure_parallel_context(self, occurrence: SourceOccurrence, call_id: object) -> tuple[bytes | None, list[SafeSpan]]:
        call_token = self._parallel_call_token(occurrence, call_id)
        if call_token is None:
            return self._attempt_contexts.get(occurrence.space.attempt), []
        key = occurrence.space.attempt, call_token
        parent = self._attempt_contexts.get(occurrence.space.attempt)
        if key not in self._parallel_contexts:
            anchor_record = self._attempt_anchor_occurrences.get(occurrence.space.attempt)
            anchor = anchor_record.epoch_ns if anchor_record is not None else None
            if not _valid_epoch(anchor):
                self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.TIMING_UNAVAILABLE, 1),)))
                return parent, []
            attrs = self._base_attrs(occurrence.space, "record", "registration", True)
            context_space = Namespace(self.run, occurrence.space.attempt, None, None)
            span = accounting.point_span(
                self.key, "stateframe.parallel.context", self.trace_id, occurrence, parent, attrs,
                at_ns=anchor, context_tag="parallel", identity_space=context_space,
                identity_extra=call_token,
            )
            if span is None:
                return parent, []
            self._parallel_contexts[key] = span.span_id
            return span.span_id, [span]
        return self._parallel_contexts[key], []

    def _parallel_call_token(self, occurrence: SourceOccurrence, call_id: object) -> bytes | None:
        if not _bounded_text(call_id, 256):
            return None
        assert type(call_id) is str
        return accounting.keyed_identity(self.key, "parallel-call", (
            namespace_bytes(Namespace(self.run, occurrence.space.attempt, None, None)),
            call_id.encode("utf-8", "strict"),
        ), 16)

    def _ensure_subframe_context(self, occurrence: SourceOccurrence, call_id: object) -> tuple[bytes | None, list[SafeSpan]]:
        space = occurrence.space
        if space.unit is None:
            return self._ensure_parallel_context(occurrence, call_id)
        call_token = self._parallel_call_token(occurrence, call_id)
        key = (space.attempt, space.unit, space.incarnation, call_token)
        if key in self._subframe_contexts:
            return self._subframe_contexts[key], []
        parallel_parent, parallel_spans = self._ensure_parallel_context(occurrence, call_id)
        parent = parallel_parent or self._attempt_contexts.get(space.attempt)
        anchor_record = self._attempt_anchor_occurrences.get(space.attempt)
        anchor = anchor_record.epoch_ns if anchor_record is not None else None
        if not _valid_epoch(anchor):
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.TIMING_UNAVAILABLE, 1),)))
            return parent, parallel_spans
        attrs = self._base_attrs(space, "record", "registration", True)
        span = accounting.point_span(
            self.key, "stateframe.subframe.context", self.trace_id, occurrence, parent, attrs,
            at_ns=anchor, context_tag="subframe", identity_space=space,
            identity_extra=call_token or b"",
        )
        if span is None:
            return parent, parallel_spans
        self._subframe_contexts[key] = span.span_id
        return span.span_id, [*parallel_spans, span]

    def _terminal(self, occurrence: SourceOccurrence) -> list[SafeSpan]:
        if occurrence.space in self._terminal_spaces:
            return []
        row = occurrence.row
        terminal_row = row
        if _family(occurrence.family) is Family.ENGINE_RESULT:
            recorded = row.get("result")
            nested = recorded if type(recorded) is dict else {}
            result_value = nested.get("result") if nested else recorded
            terminal_row = {
                **nested,
                "engine_result": result_value,
                "engine_terminal": nested if nested else {"result": result_value},
                "exit_code": row.get("exit_status"),
            }
        runtime, check, artifact, delivery, resume, exit_code, error = terminal_fields(terminal_row)
        if occurrence.space.attempt > 1 and resume in {"initial", "unknown"}:
            resume = "resumed"
        result = terminal_row.get("engine_result")
        if type(result) is not str:
            terminal = terminal_row.get("engine_terminal")
            result = terminal.get("result") if type(terminal) is dict and type(terminal.get("result")) is str else None
        outcome = outcome_for(runtime, check, delivery, result)
        if result == "AUTHORITY_FAILURE":
            outcome = "error"
        record_time = occurrence.epoch_ns
        anchor_record = (self._boundary_occurrences.get(_space_key(occurrence.space))
                         or self._attempt_anchor_occurrences.get(occurrence.space.attempt))
        anchor_time = anchor_record.epoch_ns if anchor_record is not None else None
        if not _valid_epoch(record_time) and not _valid_epoch(anchor_time):
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.TIMING_UNAVAILABLE, 1),)))
            return []
        parent, context_spans = self._ensure_subframe_context(occurrence, None)
        attrs = self._base_attrs(occurrence.space, "record", "observed_boundary", self._coverage.complete)
        attrs.extend([
            _attr("stateframe.runtime", runtime), _attr("stateframe.check", check),
            _attr("stateframe.artifact", artifact), _attr("stateframe.delivery", delivery),
            _attr("stateframe.resume", resume), _attr("stateframe.outcome", outcome),
        ])
        if exit_code is not None:
            attrs.append(_attr("stateframe.exit_code", exit_code))
        name = "stateframe.attempt.outcome"
        if _valid_epoch(record_time):
            span = accounting.point_span(
                self.key, name, self.trace_id, occurrence, parent, attrs,
                error=error,
            )
        else:
            span = accounting.anchored_point_span(
                self.key, name, self.trace_id, occurrence, anchor_record, parent, attrs, error=error,
            )
        if span is not None:
            self._terminal_spaces.add(occurrence.space)
        return [*context_spans, *((span,) if span is not None else ())]

    def _request_events(self, occurrence: SourceOccurrence, family: Family) -> list[SafeSpan]:
        row = occurrence.row
        kind = row.get("kind")
        stream = occurrence.space, occurrence.cursor.stream
        if family is Family.PLAIN:
            key = stream
        else:
            request_id, step = row.get("request_id"), row.get("step")
            if not _bounded_text(request_id, 256) or type(step) is not int or not 0 <= step <= _MAX_ORDINAL:
                return []
            key = (*stream, request_id, step)
        if kind == "request_submitted" and row.get("dispatch") is not False and row.get("replayed") is not True:
            previous = self._pending_requests.get(key)
            transport_id = row.get("event_id")
            if previous is not None and previous.row.get("event_id") == transport_id and family is not Family.PLAIN:
                return []
            self._pending_requests[key] = occurrence
            if _bounded_text(transport_id, 256):
                self._pending_transports[(*stream, transport_id)] = key
            if previous is not None:
                self._forget_transport(previous)
                return self._model_span(previous, None)
            return []
        if (kind in ("response_received", "transport_error")
                or family is Family.PLAIN and kind == "response") and row.get("replayed") is not True:
            transport_id = row.get("transport_attempt_id", row.get("transport_event_id"))
            pending = self._pending_requests.get(key)
            if pending is None:
                return []
            if _bounded_text(transport_id, 256):
                bound_key = self._pending_transports.get((*stream, transport_id))
                if (bound_key is not None and bound_key != key
                        or _bounded_text(pending.row.get("event_id"), 256)
                        and pending.row.get("event_id") != transport_id):
                    self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.INVALID_FIELD, 1),)))
                    return []
            self._pending_requests.pop(key)
            self._forget_transport(pending)
            return self._model_span(pending, occurrence)
        return []

    def _forget_transport(self, row: SourceOccurrence) -> None:
        transport_id = row.row.get("event_id")
        if _bounded_text(transport_id, 256):
            self._pending_transports.pop((row.space, row.cursor.stream, transport_id), None)

    def _send(self, occurrence: SourceOccurrence) -> list[SafeSpan]:
        if occurrence.epoch_ns is None or any(occurrence.row.get(flag) is True
                                              for flag in ("joined", "unprojectable", "replayed")):
            return []
        return self._model_span(occurrence, occurrence)

    def _model_span(self, submitted: SourceOccurrence, response: SourceOccurrence | None) -> list[SafeSpan]:
        occurrence = response or submitted
        row = occurrence.row
        space = occurrence.space
        send = submitted.row if submitted.family == Family.SEND else submitted.row.get("send_evidence")
        send = send if type(send) is dict else {}
        role = _role(send.get("role", submitted.row.get("role", "unit" if space.unit is not None else None)))
        elapsed_ns = row.get("elapsed_ns")
        if type(elapsed_ns) is not int or not 0 <= elapsed_ns <= 604_800_000_000_000:
            seconds = finite_duration(row.get("elapsed_seconds", row.get("seconds")))
            elapsed_ns = round(seconds * 1_000_000_000) if seconds is not None else None
        attrs = [_attr("gen_ai.operation.name", "chat"), _attr("stateframe.role", role)]
        ordinal = send.get("send", submitted.row.get("request_ordinal"))
        if type(ordinal) is int and 1 <= ordinal <= _MAX_ORDINAL:
            attrs.append(_attr("stateframe.request.ordinal", ordinal))
        status = row.get("status_code", row.get("http_status"))
        if type(status) is int and 100 <= status <= 599:
            attrs.append(_attr("stateframe.http.status_code", status))
        body = row.get("response")
        body = body if type(body) is dict else {}
        usage = row.get("accounting")
        if type(usage) is not dict:
            usage = row if occurrence.family is Family.SEND else row.get("usage")
        usage = usage if type(usage) is dict else {}
        prompt, completion = (
            _token(send[key]) if key in send else _token(usage.get(key))
            for key in ("prompt_tokens", "completion_tokens")
        )
        for key, value in (("gen_ai.usage.input_tokens", prompt), ("gen_ai.usage.output_tokens", completion)):
            if value is not None:
                attrs.append(_attr(key, value))
        detail = self._detail_records.get(space.attempt)
        if detail is not None:
            levels = self._attempt_levels(space.attempt)
            spec = self._attempt_anchor_occurrences.get(space.attempt)
            request_model = send.get("model", None if spec and spec.row.get("routes") else detail.request_model)
            attrs.extend(self._detail_attrs(space.attempt, replace(
                detail, request_model=request_model, response_model=body.get("model"),
                reported_cost=send.get("cost", body.get("cost")),
                price_table=self._price_table if levels.include_cost else None,
                usage=(prompt, completion) if prompt is not None and completion is not None else None,
            )))
        if elapsed_ns is not None:
            attrs.append(_attr("stateframe.duration_ns", elapsed_ns))
        error = row.get("kind") == "transport_error" or send.get("send_failed") is True
        parent = self._parent(space)
        bounds = accounting.paired_bounds(submitted, occurrence)
        if bounds is not None and (elapsed_ns is None or bounds[1] - bounds[0] == elapsed_ns):
            span = accounting.paired_span(
                self.key, "chat", self.trace_id, submitted, occurrence, parent,
                [*self._base_attrs(space, "record", "exact", True), *attrs], error=error,
            )
            return [span] if span is not None else []
        return self._point("chat", occurrence, attrs, error=error, at_ns=submitted.epoch_ns)

    def _journal_operation(self, occurrence: SourceOccurrence) -> list[SafeSpan]:
        row = occurrence.row
        kind = row.get("kind")
        if kind in ("check_receipt", "check_receipt_unchanged"):
            return self._check(occurrence, row)
        if kind == "action_receipt":
            receipt = row.get("receipt")
            if type(receipt) is dict and type(receipt.get("result_class")) is str:
                return self._action(occurrence, receipt, receipt.get("kind"), receipt["result_class"])
        if occurrence.family is Family.PLAIN and kind == "tool_reply_receipt":
            return self._action(occurrence, row, row.get("tool"), row.get("outcome"))
        if kind in ("operation_refused", "operation_blocked"):
            return self._action(occurrence, row, row.get("tool"), kind.removeprefix("operation_"),
                                execution="not_dispatched")
        return []

    def _parallel(self, occurrence: SourceOccurrence):
        row = occurrence.row
        kind = row.get("kind")
        call_id = row.get("call_id")
        spans: list[SafeSpan] = []
        if type(kind) is str and kind in {"unit_started", "started"}:
            started = epoch_seconds_ns(row.get("started_epoch", row.get("created_epoch")))
            started = started if _valid_epoch(started) else occurrence.epoch_ns
            if _valid_epoch(started):
                _parent, contexts = self._ensure_subframe_context(occurrence, call_id)
                spans.extend(contexts)
                self._unit_starts[_space_key(occurrence.space)] = (occurrence, started)
            return spans
        if type(kind) is str and kind in {"unit_finished", "finished", "exited"}:
            unit_key = _space_key(occurrence.space)
            if unit_key in self._completed_units:
                return spans
            prior = self._unit_starts.pop(unit_key, None)
            if prior is not None and not all(
                prior[0].row.get(field) == row.get(field)
                for field in ("call_id", "unit_id", "incarnation")
            ):
                prior = None
            start_value = row.get("started_epoch")
            end_value = row.get("finished_epoch")
            explicit_bounds = accounting.seconds_bounds(start_value, end_value)
            paired_end = epoch_seconds_ns(end_value)
            outcome = fixed_outcome(row.get("terminal"))
            name = "stateframe.subframe"
            operation_attrs = [
                _attr("stateframe.outcome", outcome),
                _attr("stateframe.parallel.disposition", _parallel_disposition(row.get("parent_disposition"))),
                _attr("stateframe.execution", "executed"),
            ]
            attrs = [*self._base_attrs(occurrence.space, "record", "exact", True), *operation_attrs]
            span = None
            parent, contexts = self._ensure_subframe_context(occurrence, call_id)
            spans.extend(contexts)
            if explicit_bounds is not None:
                span = accounting.recorded_span(
                    self.key, name, self.trace_id, occurrence, parent, *explicit_bounds, attrs,
                    error=outcome in {"error", "timed_out", "recovery_required"},
                )
            elif (prior is not None and paired_end is not None
                  and accounting.paired_bounds(prior[0], occurrence,
                                              start_ns=prior[1], end_ns=paired_end) is not None):
                prior_record, prior_start = prior
                span = accounting.paired_span(
                    self.key, name, self.trace_id, prior_record, occurrence, parent, attrs,
                    error=outcome in {"error", "timed_out", "recovery_required"},
                    start_ns=prior_start, end_ns=paired_end,
                )
            elif paired_end is not None:
                span = accounting.point_span(
                    self.key, name, self.trace_id, occurrence, parent,
                    [*self._base_attrs(occurrence.space, "record", "observed_boundary", True),
                     *operation_attrs], at_ns=paired_end,
                    error=outcome in {"error", "timed_out", "recovery_required"},
                )
            if span is not None:
                spans.append(span)
                self._completed_units.add(unit_key)
            return spans
        if type(kind) is str and kind in {"unit_context", "unit_registered", "launch_refused", "refused"}:
            _parent, contexts = self._ensure_subframe_context(occurrence, call_id)
            spans.extend(contexts)
        return spans

    def _tool(self, occurrence: SourceOccurrence) -> list[SafeSpan]:
        row = occurrence.row
        kind, call_id = row.get("kind"), row.get("call_id")
        if kind in ("refused", "blocked"):
            return self._action(occurrence, row, row.get("tool"), kind, execution="not_dispatched")
        if not _bounded_text(call_id, 256):
            return []
        key = occurrence.space, occurrence.cursor.stream, call_id
        if kind == "intent":
            self._pending_tools[key] = occurrence
        if kind == "complete":
            intent = self._pending_tools.pop(key, None)
            if intent is not None and row.get("replayed") is not True and intent.row.get("replayed") is not True:
                outcome = row.get("outcome", row.get("status"))
                if outcome is None and type(row.get("exit_code")) is int:
                    outcome = "ok" if row["exit_code"] == 0 else "error"
                return self._action(occurrence, row, intent.row.get("tool"), outcome)
        return []

    def _point(self, name, occurrence, attrs, *, error=False, at_ns=None) -> list[SafeSpan]:
        space = occurrence.space
        anchor = self._boundary_occurrences.get(_space_key(space)) or self._attempt_anchor_occurrences.get(space.attempt)
        epoch = occurrence.epoch_ns or at_ns or (anchor.epoch_ns if anchor is not None else None)
        span = accounting.point_span(
            self.key, name, self.trace_id, occurrence, self._parent(space),
            [*self._base_attrs(space, "record", "observed_boundary", True), *attrs],
            at_ns=epoch, error=error,
        )
        if span is None:
            self._coverage = coverage_merge(self._coverage, Coverage(False, ((Reason.TIMING_UNAVAILABLE, 1),)))
        return [span] if span is not None else []

    def _action(self, occurrence, row, tool, result, *, execution="executed") -> list[SafeSpan]:
        if tool == "test":
            tool = "check"
        if type(result) is str and result in {"OK", "APPLIED", "GREEN", "SUBMIT_PENDING", "accepted"}:
            outcome = "ok"
        elif type(result) is str and result in {"RED", "FAILED", "NEEDS_WORK"}:
            outcome = "error"
        else:
            outcome = fixed_outcome(result)
        attrs = [_attr("stateframe.tool", fixed_tool(tool)), _attr("stateframe.outcome", outcome),
                 _attr("stateframe.execution", execution)]
        return self._point("stateframe.action", occurrence, [*attrs, *_duration_attrs(row)], error=outcome == "error")

    def _check(self, occurrence, row, *, check_kind=None) -> list[SafeSpan]:
        status = row.get("status", row.get("passed"))
        if status is True or status in ("PASS", "pass"):
            outcome = "pass"
        elif status is False or status in ("FAIL", "fail"):
            outcome = "fail"
        elif type(row.get("exit_code")) is int:
            outcome = "pass" if row["exit_code"] == 0 else "fail"
        else:
            return []
        identity = row.get("host_execution_id", row.get("execution_id"))
        if _bounded_text(identity, 256):
            key = occurrence.space, identity
            if key in self._seen_checks:
                return []
            self._seen_checks.add(key)
        kind = check_kind or fixed_check_kind(row.get("check_kind", row.get("kind_name")))
        execution = "reexecuted" if row.get("execution") == "reexecuted" or row.get("reexecuted") is True else "executed"
        attrs = [_attr("stateframe.outcome", outcome), _attr("stateframe.check.kind", kind),
                 _attr("stateframe.execution", execution), *_duration_attrs(row)]
        if check_kind is not None:
            # Multiple host check outcomes share the same summary occurrence.
            occurrence = replace(occurrence, cursor=replace(
                occurrence.cursor, head=accounting.keyed_identity(self.key, "summary-check", (
                    occurrence.cursor.head or b"", check_kind.encode("ascii"),
                ), 32),
            ))
        return self._point("stateframe.check", occurrence, attrs, error=outcome == "fail")

    def _stopped(self, occurrence: SourceOccurrence) -> list[SafeSpan]:
        anchor = (self._boundary_occurrences.get(_space_key(occurrence.space))
                  or self._attempt_anchor_occurrences.get(occurrence.space.attempt))
        if anchor is None:
            return []
        runtime = "interrupted" if occurrence.row.get("interrupted") is True else "unknown"
        attrs = self._base_attrs(occurrence.space, "historical_boundary", "observed_boundary", False)
        attrs.extend([_attr("stateframe.runtime", runtime), _attr("stateframe.outcome", runtime)])
        span = accounting.anchored_point_span(
            self.key, "stateframe.attempt.observation_end", self.trace_id,
            None, anchor, self._parent(occurrence.space), attrs, error=runtime == "interrupted",
        )
        return [span] if span is not None else []

    def _base_attrs(self, space: Namespace, source: str, timing: str, complete: bool,
                    *, include_attempt: bool = True) -> list[SafeAttr]:
        attrs = [
            _attr("stateframe.run.id", self.run.value.hex()),
            _attr("stateframe.source", source), _attr("stateframe.timing", timing),
            _attr("stateframe.complete", bool(complete)),
        ]
        if include_attempt and space.attempt > 0:
            attrs.append(_attr("stateframe.attempt", space.attempt))
        if space.incarnation is not None and 1 <= space.incarnation <= _MAX_ORDINAL:
            attrs.append(_attr("stateframe.incarnation", space.incarnation))
        return attrs

def _valid_namespace(space: Namespace) -> bool:
    return (type(space.run) is RunRef and type(space.run.value) is bytes and len(space.run.value) == 16
            and any(space.run.value) and type(space.attempt) is int and 1 <= space.attempt <= _MAX_ORDINAL
            and (space.unit is None or type(space.unit) is bytes and len(space.unit) == 16)
            and (space.incarnation is None or type(space.incarnation) is int
                 and 1 <= space.incarnation <= _MAX_ORDINAL))


def _valid_cursor(cursor: object) -> bool:
    return (type(cursor) is Cursor and type(cursor.stream) is bytes and len(cursor.stream) == 16
            and type(cursor.generation) is bytes and len(cursor.generation) == 32
            and type(cursor.offset) is int and 0 <= cursor.offset <= _MAX_EPOCH
            and type(cursor.sequence) is int and 0 <= cursor.sequence <= _MAX_ORDINAL
            and (cursor.head is None or type(cursor.head) is bytes and len(cursor.head) == 32))


def _spec_attempt_epoch(row: dict[str, object], attempt: int, run_epoch: int) -> int | None:
    if attempt == 1:
        return run_epoch
    resume = row.get("resume")
    resumed = resume.get("resumed") if type(resume) is dict else None
    if type(resumed) is not str or not resumed.endswith("+00:00"):
        return None
    try:
        parsed = datetime.fromisoformat(resumed)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    value = (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
    return value if _valid_epoch(value) else None


def _family(value: object) -> Family | None:
    if isinstance(value, Family):
        return value
    if type(value) is str:
        try:
            return Family(value)
        except ValueError:
            return None
    return None


def _valid_epoch(value: object) -> bool:
    return type(value) is int and 1 <= value <= _MAX_EPOCH


def _space_key(space: Namespace) -> tuple[int, bytes | None, int | None]:
    return space.attempt, space.unit, space.incarnation


def _attr(key: str, value: bool | int | float | str) -> SafeAttr:
    return SafeAttr(AttrKey(key), value)


def _role(value: object) -> str:
    return value if type(value) is str and value in ROLES else "unknown"


def _bounded_text(value: object, maximum: int) -> bool:
    if type(value) is not str or not value:
        return False
    try:
        return len(value.encode("utf-8", "strict")) <= maximum
    except UnicodeError:
        return False


def _duration_attrs(row: dict[str, object]) -> list[SafeAttr]:
    elapsed_ns = row.get("elapsed_ns")
    if type(elapsed_ns) is int and 0 <= elapsed_ns <= 604_800_000_000_000:
        return [_attr("stateframe.duration_ns", elapsed_ns)]
    for field in ("duration_seconds", "elapsed_seconds", "duration_s"):
        value = finite_duration(row.get(field))
        if value is not None:
            return [_attr("stateframe.duration_s", value)]
    return []


def _token(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= _MAX_TOKEN else None


def _parallel_disposition(value: object) -> str:
    if type(value) is not str:
        return "unknown"
    normalized = value.lower()
    normalized = {
        "budget_exhausted": "exhausted", "engine_error": "error", "canceled": "cancelled", "conflict": "refused",
    }.get(normalized, normalized)
    allowed = {
        "committed", "committed_unverified", "refused", "blocked", "exhausted", "error", "cancelled",
        "timed_out", "recovery_required", "unknown",
    }
    if normalized in allowed:
        return normalized
    return "unknown"
