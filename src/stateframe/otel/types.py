"""Closed, transport-independent records used by the optional exporter."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .levels import _ATTRIBUTE_NAMES


class Family(StrEnum):
    SPEC = "spec"
    JOURNAL = "journal"
    PLAIN = "plain"
    SEND = "send"
    PARALLEL = "parallel"
    TOOL = "tool"
    SUMMARY = "summary"
    ENGINE_RESULT = "engine_result"
    STOPPED = "stopped"


class Signal(StrEnum):
    TRACES = "traces"


class AttrKey(StrEnum):
    REQUEST_MODEL = _ATTRIBUTE_NAMES["request_model"]
    RESPONSE_MODEL = _ATTRIBUTE_NAMES["response_model"]
    PROVIDER_NAME = _ATTRIBUTE_NAMES["provider_name"]
    ENGINE_VERSION = _ATTRIBUTE_NAMES["engine_version"]
    COST_VALUE = _ATTRIBUTE_NAMES["cost_value"]
    COST_KIND = _ATTRIBUTE_NAMES["cost_kind"]
    RUN_ID = "stateframe.run.id"
    ATTEMPT = "stateframe.attempt"
    SOURCE = "stateframe.source"
    TIMING = "stateframe.timing"
    COMPLETE = "stateframe.complete"
    OUTCOME = "stateframe.outcome"
    EXECUTION = "stateframe.execution"
    RUNTIME = "stateframe.runtime"
    CHECK = "stateframe.check"
    ARTIFACT = "stateframe.artifact"
    DELIVERY = "stateframe.delivery"
    RESUME = "stateframe.resume"
    EXIT_CODE = "stateframe.exit_code"
    ROLE = "stateframe.role"
    REQUEST_ORDINAL = "stateframe.request.ordinal"
    RETRY = "stateframe.retry"
    HTTP_STATUS_CODE = "stateframe.http.status_code"
    CACHE = "stateframe.cache"
    CACHE_TOKENS = "stateframe.cache.tokens"
    USAGE = "stateframe.usage"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    TOOL = "stateframe.tool"
    CLEANUP = "stateframe.cleanup"
    CHECK_KIND = "stateframe.check.kind"
    HOOK_EVENT = "stateframe.hook.event"
    MUTATION_STAGE = "stateframe.mutation.stage"
    PARALLEL_DISPOSITION = "stateframe.parallel.disposition"
    INCARNATION = "stateframe.incarnation"
    REASON = "stateframe.reason"
    COUNT = "stateframe.count"
    DURATION_NS = "stateframe.duration_ns"
    DURATION_S = "stateframe.duration_s"


class Reason(StrEnum):
    DISABLED = "disabled"
    UNSUPPORTED_SETTING = "unsupported_setting"
    INVALID_CONFIG = "invalid_config"
    ISOLATION_UNVERIFIED = "isolation_unverified"
    UNSAFE_ALIAS = "unsafe_alias"
    SOURCE_UNTRUSTED = "source_untrusted"
    SOURCE_BUSY = "source_busy"
    SOURCE_OVERSIZE = "source_oversize"
    SOURCE_CORRUPT = "source_corrupt"
    SOURCE_MISSING = "source_missing"
    TORN_TAIL = "torn_tail"
    GENERATION_AMBIGUOUS = "generation_ambiguous"
    TIMING_UNAVAILABLE = "timing_unavailable"
    INVALID_FIELD = "invalid_field"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    QUEUE_FULL = "queue_full"
    ASSEMBLY_LIMIT = "assembly_limit"
    DISCOVERY_LIMIT = "discovery_limit"
    DISK_FULL = "disk_full"
    STATE_UNAVAILABLE = "state_unavailable"
    HISTORY_EXPIRED = "history_expired"
    HISTORY_UNKNOWN = "history_unknown"
    PARENT_UNAVAILABLE = "parent_unavailable"
    EXPIRED = "expired"
    PEER_FAILURE = "peer_failure"
    DNS_FAILED = "dns_failed"
    CONNECTION_FAILED = "connection_failed"
    TIMEOUT = "timeout"
    TLS_FAILED = "tls_failed"
    REDIRECT = "redirect"
    AUTH_FAILED = "auth_failed"
    PAYLOAD_REJECTED = "payload_rejected"
    THROTTLED = "throttled"
    PARTIAL_REJECTION = "partial_rejection"
    RECEIVER_WARNING = "receiver_warning"
    BAD_RESPONSE = "bad_response"
    DELIVERY_UNKNOWN = "delivery_unknown"
    CLOCK_DISCONTINUITY = "clock_discontinuity"
    SAMPLED_OUT = "sampled_out"


class ResultKind(StrEnum):
    ACCEPTED = "accepted"
    WARNING = "warning"
    PARTIAL = "partial"
    PERMANENT = "permanent"
    RETRY = "retry"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RunRef:
    value: bytes


@dataclass(frozen=True, slots=True)
class Namespace:
    run: RunRef
    attempt: int
    unit: bytes | None
    incarnation: int | None


@dataclass(frozen=True, slots=True)
class Cursor:
    stream: bytes
    generation: bytes
    offset: int
    sequence: int
    head: bytes | None


@dataclass(frozen=True, slots=True)
class SourceOccurrence:
    space: Namespace
    family: Family
    index: int
    epoch_ns: int | None
    row: dict[str, object]
    cursor: Cursor


@dataclass(frozen=True, slots=True)
class SafeAttr:
    key: AttrKey
    value: bool | int | float | str


@dataclass(frozen=True, slots=True)
class SafeEvent:
    name: str
    at_ns: int
    attrs: tuple[SafeAttr, ...]
    dropped_attrs: int


@dataclass(frozen=True, slots=True)
class SafeLink:
    trace_id: bytes
    span_id: bytes


@dataclass(frozen=True, slots=True)
class SafeSpan:
    name: str
    trace_id: bytes
    span_id: bytes
    parent_id: bytes | None
    start_ns: int
    end_ns: int
    attrs: tuple[SafeAttr, ...]
    events: tuple[SafeEvent, ...]
    links: tuple[SafeLink, ...]
    dropped_attrs: int
    dropped_events: int
    dropped_links: int
    error: bool


@dataclass(frozen=True, slots=True)
class SignalBatch:
    signal: Signal
    resource_id: bytes
    spans: tuple[SafeSpan, ...]


@dataclass(frozen=True, slots=True)
class Coverage:
    complete: bool
    reasons: tuple[tuple[Reason, int | None], ...]


@dataclass(frozen=True, slots=True)
class Projection:
    key: bytes
    space: Namespace
    cursor: Cursor
    spans: tuple[SafeSpan, ...]
    coverage: Coverage


@dataclass(frozen=True, slots=True)
class ExportResult:
    kind: ResultKind
    accepted: int | None
    rejected: int | None
    reason: Reason | None
    retry_delay_s: float | None
    may_have_delivered: bool


@dataclass(frozen=True, slots=True)
class PolicyView:
    seed: bytes | None
    attempt: int | None
    profile: str | None
    environment: Mapping[str, object]
    grants: tuple[Path, ...]
    authority_records: tuple[Path, ...]
    source_generation: bytes | None


__all__ = [
    "AttrKey", "Coverage", "Cursor", "ExportResult", "Family",
    "Namespace", "PolicyView", "Projection", "Reason", "ResultKind", "RunRef",
    "SafeAttr", "SafeEvent", "SafeLink", "SafeSpan", "Signal", "SignalBatch",
    "SourceOccurrence",
]
