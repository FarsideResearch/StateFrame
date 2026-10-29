"""Host-owned typed binding and lowering envelope.

The envelope is deliberately data-only.  A host registry supplies the
capability and executor; a model cannot smuggle executable source, shell
commands, result handles, ordering, or completion state through this API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping


SCHEMA = "iq-procedural-binding/1"


class BindingContractError(ValueError):
    """Raised for malformed envelopes or invalid host registrations."""


class BindingPhase(str, Enum):
    SCHEMA = "SCHEMA"
    COMPILE = "COMPILE"
    SETUP = "SETUP"
    REACHABILITY = "REACHABILITY"
    BEHAVIOR = "BEHAVIOR"
    SETTLEMENT = "SETTLEMENT"


class BindingStatus(str, Enum):
    COMPILED = "COMPILED"
    COMPILE_FAILED = "COMPILE_FAILED"
    SETUP_FAILED = "SETUP_FAILED"
    UNREACHABLE = "UNREACHABLE"
    BEHAVIOR_PASS = "BEHAVIOR_PASS"
    BEHAVIOR_FAIL = "BEHAVIOR_FAIL"
    UNSUPPORTED = "UNSUPPORTED"
    ASYNC_PENDING = "ASYNC_PENDING"
    SERIALIZATION_MISMATCH = "SERIALIZATION_MISMATCH"
    INVALID = "INVALID"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass(frozen=True, slots=True)
class Fixture:
    fixture_id: str
    revision: str
    operation_kind: str
    payload: object
    available: bool = True

    def __post_init__(self) -> None:
        for field_name in ("fixture_id", "revision", "operation_kind"):
            _text(getattr(self, field_name), field_name)
        _json_value(self.payload, "fixture payload")


@dataclass(frozen=True, slots=True)
class ObservableSpec:
    observable_id: str
    path: tuple[str, ...]
    expected: object
    relation: str = "equals"

    def __post_init__(self) -> None:
        _text(self.observable_id, "observable_id")
        if isinstance(self.path, str):
            raise BindingContractError("observable path must be a tuple")
        object.__setattr__(
            self,
            "path",
            tuple(_text(part, "observable path part") for part in self.path),
        )
        if self.relation not in {"equals", "contains"}:
            raise BindingContractError("unsupported observable relation")
        _json_value(self.expected, "observable expectation")


@dataclass(frozen=True, slots=True)
class BindingEnvelope:
    request_id: str
    target: str
    language: str
    operation_kind: str
    fixture_id: str
    fixture_revision: str
    observable: ObservableSpec
    expectation_source: str
    preservation_refs: tuple[str, ...] = ()
    serialization: str = "json"
    requires_async: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "target",
            "language",
            "operation_kind",
            "fixture_id",
            "fixture_revision",
            "expectation_source",
        ):
            _text(getattr(self, field_name), field_name)
        if not isinstance(self.observable, ObservableSpec):
            raise BindingContractError("observable must be ObservableSpec")
        object.__setattr__(
            self,
            "preservation_refs",
            _text_tuple(self.preservation_refs, "preservation ref"),
        )
        if self.serialization not in {"json"}:
            raise BindingContractError("only the host JSON serialization is supported")
        if not isinstance(self.requires_async, bool):
            raise BindingContractError("requires_async must be boolean")


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    capability_id: str
    operation_kind: str
    languages: tuple[str, ...]
    serialization_formats: tuple[str, ...] = ("json",)
    supports_async: bool = False
    compile_supported: bool = True
    reachable: bool = True

    def __post_init__(self) -> None:
        _text(self.capability_id, "capability_id")
        _text(self.operation_kind, "capability operation_kind")
        object.__setattr__(
            self, "languages", _text_tuple(self.languages, "capability language")
        )
        if not self.languages:
            raise BindingContractError("capability must declare at least one language")
        object.__setattr__(
            self,
            "serialization_formats",
            _text_tuple(self.serialization_formats, "serialization format"),
        )
        if not self.serialization_formats:
            raise BindingContractError("capability must declare a serialization format")


@dataclass(frozen=True, slots=True)
class BindingContext:
    execution_handle: str
    order_index: int
    fixture: Fixture
    envelope: BindingEnvelope
    async_token: str | None


@dataclass(frozen=True, slots=True)
class PendingResult:
    """Marker returned by a registered host executor for async work."""

    token: str


@dataclass(frozen=True, slots=True)
class BindingReceipt:
    binding_id: str
    request_id: str
    phase: BindingPhase
    status: BindingStatus
    capability_id: str | None
    execution_handle: str | None
    order_index: int | None
    fixture_revision: str | None
    output_handle: str | None = None
    output_digest: str | None = None
    observed: object = None
    fallback: Mapping[str, object] | None = None
    error_code: str | None = None
    async_token: str | None = None
    preservation_refs: tuple[str, ...] = ()

    @property
    def settled(self) -> bool:
        return self.status not in {BindingStatus.ASYNC_PENDING}

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "iq-procedural-binding-receipt/1",
            "binding_id": self.binding_id,
            "request_id": self.request_id,
            "phase": self.phase.value,
            "status": self.status.value,
            "capability_id": self.capability_id,
            "execution_handle": self.execution_handle,
            "order_index": self.order_index,
            "fixture_revision": self.fixture_revision,
            "output_handle": self.output_handle,
            "output_digest": self.output_digest,
            "observed": self.observed,
            "fallback": dict(self.fallback) if self.fallback is not None else None,
            "error_code": self.error_code,
            "async_token": self.async_token,
            "preservation_refs": list(self.preservation_refs),
        }


Executor = Callable[[BindingContext], object]


class BindingHost:
    """A small host registry that lowers only registered operations."""

    def __init__(self) -> None:
        self._capabilities: dict[str, tuple[CapabilityDescriptor, Executor]] = {}
        self._fixtures: dict[str, Fixture] = {}
        self._counter = 0
        self._pending: dict[
            str, tuple[BindingReceipt, BindingEnvelope, CapabilityDescriptor]
        ] = {}

    def register_capability(
        self, descriptor: CapabilityDescriptor, executor: Executor
    ) -> None:
        if not callable(executor):
            raise BindingContractError(
                "capability executor must be host-registered and callable"
            )
        if descriptor.capability_id in self._capabilities:
            raise BindingContractError(
                f"duplicate capability: {descriptor.capability_id}"
            )
        self._capabilities[descriptor.capability_id] = (descriptor, executor)

    def register_fixture(self, fixture: Fixture) -> None:
        if fixture.fixture_id in self._fixtures:
            raise BindingContractError(f"duplicate fixture: {fixture.fixture_id}")
        self._fixtures[fixture.fixture_id] = fixture

    def dispatch(self, envelope: BindingEnvelope) -> BindingReceipt:
        try:
            self._validate_schema(envelope)
        except BindingContractError as exc:
            request_id = (
                envelope.request_id
                if isinstance(envelope, BindingEnvelope)
                else "invalid-request"
            )
            return BindingReceipt(
                f"invalid-{self._counter}",
                request_id,
                BindingPhase.SCHEMA,
                BindingStatus.INVALID,
                None,
                None,
                None,
                None,
                error_code=str(exc),
            )
        self._counter += 1
        binding_id = f"binding-{self._counter}"
        registered = self._capabilities.get(envelope.target)
        if registered is None:
            return self._unsupported(envelope, binding_id, "capability_not_registered")
        descriptor, executor = registered
        if (
            descriptor.operation_kind != envelope.operation_kind
            or envelope.language not in descriptor.languages
            or envelope.serialization not in descriptor.serialization_formats
        ):
            return self._unsupported(
                envelope,
                binding_id,
                "capability_shape_not_supported",
                descriptor.capability_id,
            )
        if envelope.requires_async and not descriptor.supports_async:
            return self._unsupported(
                envelope, binding_id, "async_not_supported", descriptor.capability_id
            )
        if not descriptor.compile_supported:
            return self._receipt(
                envelope,
                BindingPhase.COMPILE,
                BindingStatus.COMPILE_FAILED,
                capability_id=descriptor.capability_id,
                binding_id=binding_id,
                error_code="compile_rejected",
            )
        fixture = self._fixtures.get(envelope.fixture_id)
        if (
            fixture is None
            or fixture.revision != envelope.fixture_revision
            or not fixture.available
        ):
            return self._receipt(
                envelope,
                BindingPhase.SETUP,
                BindingStatus.SETUP_FAILED,
                capability_id=descriptor.capability_id,
                binding_id=binding_id,
                error_code="fixture_unavailable",
            )
        if not descriptor.reachable:
            return self._receipt(
                envelope,
                BindingPhase.REACHABILITY,
                BindingStatus.UNREACHABLE,
                capability_id=descriptor.capability_id,
                binding_id=binding_id,
                error_code="target_unreachable",
            )
        self._counter += 1
        handle = f"exec-{self._counter}"
        async_token = f"async-{self._counter}" if envelope.requires_async else None
        context = BindingContext(handle, self._counter, fixture, envelope, async_token)
        try:
            output = executor(context)
        except Exception as exc:  # host receipt contains only a stable class name
            return self._receipt(
                envelope,
                BindingPhase.BEHAVIOR,
                BindingStatus.EXECUTION_ERROR,
                capability_id=descriptor.capability_id,
                binding_id=binding_id,
                execution_handle=handle,
                order_index=context.order_index,
                error_code=f"executor_{type(exc).__name__}",
            )
        if isinstance(output, PendingResult):
            if not envelope.requires_async or output.token != async_token:
                return self._receipt(
                    envelope,
                    BindingPhase.SETTLEMENT,
                    BindingStatus.SERIALIZATION_MISMATCH,
                    capability_id=descriptor.capability_id,
                    binding_id=binding_id,
                    execution_handle=handle,
                    order_index=context.order_index,
                    error_code="async_token_mismatch",
                )
            receipt = self._receipt(
                envelope,
                BindingPhase.BEHAVIOR,
                BindingStatus.ASYNC_PENDING,
                capability_id=descriptor.capability_id,
                binding_id=binding_id,
                execution_handle=handle,
                order_index=context.order_index,
                async_token=async_token,
            )
            self._pending[async_token] = (receipt, envelope, descriptor)
            return receipt
        return self._settle_output(
            envelope, descriptor, binding_id, handle, context.order_index, output
        )

    def complete_async(self, token: str, output: object) -> BindingReceipt:
        record = self._pending.pop(token, None)
        if record is None:
            raise BindingContractError("unknown async token")
        receipt, envelope, descriptor = record
        return self._settle_output(
            envelope,
            descriptor,
            receipt.binding_id,
            receipt.execution_handle or "",
            receipt.order_index or 0,
            output,
            token,
        )

    def _settle_output(
        self,
        envelope: BindingEnvelope,
        descriptor: CapabilityDescriptor,
        binding_id: str,
        handle: str,
        order_index: int,
        output: object,
        async_token: str | None = None,
    ) -> BindingReceipt:
        try:
            serialized = json.dumps(
                output, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            decoded = json.loads(serialized)
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._receipt(
                envelope,
                BindingPhase.SETTLEMENT,
                BindingStatus.SERIALIZATION_MISMATCH,
                capability_id=descriptor.capability_id,
                binding_id=binding_id,
                execution_handle=handle,
                order_index=order_index,
                error_code="output_not_json",
                async_token=async_token,
            )
        try:
            observed = _lookup(decoded, envelope.observable.path)
        except BindingContractError:
            return self._receipt(
                envelope,
                BindingPhase.BEHAVIOR,
                BindingStatus.BEHAVIOR_FAIL,
                capability_id=descriptor.capability_id,
                binding_id=binding_id,
                execution_handle=handle,
                order_index=order_index,
                error_code="observable_missing",
                async_token=async_token,
            )
        passed = (
            envelope.observable.expected in observed
            if envelope.observable.relation == "contains"
            and isinstance(observed, (list, str, dict))
            else observed == envelope.observable.expected
        )
        return self._receipt(
            envelope,
            BindingPhase.BEHAVIOR,
            BindingStatus.BEHAVIOR_PASS if passed else BindingStatus.BEHAVIOR_FAIL,
            capability_id=descriptor.capability_id,
            binding_id=binding_id,
            execution_handle=handle,
            order_index=order_index,
            observed=observed,
            output_handle=f"output-{binding_id}",
            output_digest=_digest(serialized),
            async_token=async_token,
        )

    def _validate_schema(self, envelope: BindingEnvelope) -> None:
        if not isinstance(envelope, BindingEnvelope):
            raise BindingContractError("binding must be a BindingEnvelope")
        if envelope.target.startswith(("python:", "shell:", "eval:", "exec:")):
            raise BindingContractError("target cannot carry executable model code")

    def _unsupported(
        self,
        envelope: BindingEnvelope,
        binding_id: str,
        reason: str,
        capability_id: str | None = None,
    ) -> BindingReceipt:
        return self._receipt(
            envelope,
            BindingPhase.COMPILE,
            BindingStatus.UNSUPPORTED,
            capability_id=capability_id,
            binding_id=binding_id,
            error_code=reason,
            fallback={"route": "authorized_work", "reason": reason},
        )

    def _receipt(
        self,
        envelope: BindingEnvelope,
        phase: BindingPhase,
        status: BindingStatus,
        *,
        capability_id: str | None = None,
        binding_id: str = "",
        execution_handle: str | None = None,
        order_index: int | None = None,
        fixture_revision: str | None = None,
        output_handle: str | None = None,
        output_digest: str | None = None,
        observed: object = None,
        fallback: Mapping[str, object] | None = None,
        error_code: str | None = None,
        async_token: str | None = None,
    ) -> BindingReceipt:
        return BindingReceipt(
            binding_id or f"invalid-{self._counter}",
            envelope.request_id,
            phase,
            status,
            capability_id,
            execution_handle,
            order_index,
            fixture_revision or envelope.fixture_revision,
            output_handle,
            output_digest,
            observed,
            fallback,
            error_code,
            async_token,
            envelope.preservation_refs,
        )


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BindingContractError(f"{field_name} must be non-empty")
    return value.strip()


def _text_tuple(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise BindingContractError(f"{field_name} must be a tuple")
    return tuple(dict.fromkeys(_text(value, field_name) for value in values))


def _json_value(value: object, field_name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BindingContractError(f"{field_name} must be JSON-compatible") from exc


def _lookup(value: object, path: tuple[str, ...]) -> object:
    current = value
    for part in path:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise BindingContractError("observable path is not reachable")
    return current


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "BindingContext",
    "BindingContractError",
    "BindingEnvelope",
    "BindingHost",
    "BindingPhase",
    "BindingReceipt",
    "BindingStatus",
    "CapabilityDescriptor",
    "Fixture",
    "ObservableSpec",
    "PendingResult",
    "SCHEMA",
]
