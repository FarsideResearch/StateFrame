"""Immutable host-side authority for one FRAME_ONLY request allocation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from typing import cast


ALLOCATION_SCHEMA = "stateframe-frame-request-allocation/1"
ALLOCATION_MODE = "FRAME_ONLY"
_DIGEST_DOMAIN = b"stateframe/frame-request-allocation/1\0"
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "mode",
        "identity",
        "capacity",
        "source_obligations",
        "source_layout",
        "optional_heads",
        "site_inventory",
        "outcome",
    }
)


class FrameRequestAllocationError(ValueError):
    """A host allocation is malformed, stale, or fails its integrity binding."""


def _validate_json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FrameRequestAllocationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise FrameRequestAllocationError(f"{path} has a non-string key")
            _validate_json_value(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    raise FrameRequestAllocationError(f"{path} is not JSON data")


def canonical_allocation_bytes(value: object) -> bytes:
    """Serialize detached JSON data with the allocation's stable wire encoding."""
    _validate_json_value(value, "allocation")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise FrameRequestAllocationError("allocation cannot be serialized") from exc


def frame_request_allocation_digest(payload: bytes) -> str:
    """Return the domain-separated digest used by builders and validators."""
    if type(payload) is not bytes:
        raise FrameRequestAllocationError("allocation payload must be bytes")
    return "sha256:" + hashlib.sha256(_DIGEST_DOMAIN + payload).hexdigest()


def verify_frame_request_allocation_digest(payload: bytes, digest: object) -> bool:
    """Validate an allocation digest using the same helper as allocation creation."""
    if not isinstance(digest, str):
        return False
    try:
        expected = frame_request_allocation_digest(payload)
    except FrameRequestAllocationError:
        return False
    return hmac.compare_digest(expected, digest)


def _object(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise FrameRequestAllocationError(f"{field} must be a string-keyed object")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrameRequestAllocationError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FrameRequestAllocationError(f"{field} must be an integer >= {minimum}")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise FrameRequestAllocationError(f"{field} must be a list")
    result = [_text(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise FrameRequestAllocationError(f"{field} contains duplicate identities")
    return result


def _message(value: object, field: str, roles: set[str]) -> Mapping[str, object]:
    message = _object(value, field)
    if message.get("role") not in roles:
        raise FrameRequestAllocationError(f"{field} has an invalid role")
    return message


def _validate_identity(value: object) -> None:
    identity = _object(value, "identity")
    for field in (
        "request_lineage",
        "product_revision",
        "task_digest",
        "ledger_digest",
        "newest_exchange_digest",
    ):
        _text(identity.get(field), f"identity.{field}")
    _integer(identity.get("generation"), "identity.generation")
    authorization = identity.get("workspace_authorization")
    if isinstance(authorization, str):
        _text(authorization, "identity.workspace_authorization")
    else:
        _object(authorization, "identity.workspace_authorization")
    cutoff = identity.get("journal_semantic_cutoff")
    if isinstance(cutoff, str):
        _text(cutoff, "identity.journal_semantic_cutoff")
    else:
        _object(cutoff, "identity.journal_semantic_cutoff")


def _validate_capacity(value: object) -> None:
    capacity = _object(value, "capacity")
    for field in ("byte_ceiling", "token_ceiling", "frame_ceiling"):
        _integer(capacity.get(field), f"capacity.{field}", minimum=1)
    for field in ("measurement_identity", "template_identity", "tool_schema_digest"):
        _text(capacity.get(field), f"capacity.{field}")


def _validate_obligations(value: object) -> None:
    if not isinstance(value, list):
        raise FrameRequestAllocationError("source_obligations must be a list")
    handles: set[str] = set()
    for index, raw in enumerate(value):
        row = _object(raw, f"source_obligations[{index}]")
        handle = _text(row.get("handle"), f"source_obligations[{index}].handle")
        _text(row.get("path"), f"source_obligations[{index}].path")
        _text(row.get("revision"), f"source_obligations[{index}].revision")
        _text(row.get("content_digest"), f"source_obligations[{index}].content_digest")
        if handle in handles:
            raise FrameRequestAllocationError(
                "source_obligations contains duplicate handles"
            )
        handles.add(handle)
        if row.get("disposition") not in {"RETAINED", "DEFERRED"}:
            raise FrameRequestAllocationError(
                f"source_obligations[{index}].disposition is invalid"
            )
        ranges = row.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            raise FrameRequestAllocationError(
                f"source_obligations[{index}].ranges is empty"
            )
        for range_index, raw_range in enumerate(ranges):
            span = _object(
                raw_range, f"source_obligations[{index}].ranges[{range_index}]"
            )
            start = _integer(span.get("start_line"), "source range start", minimum=1)
            end = _integer(span.get("end_line"), "source range end", minimum=1)
            if end < start:
                raise FrameRequestAllocationError("source range ends before it starts")


def _validate_tool_exchange_layout(
    layout: Mapping[str, object],
) -> tuple[list[object], int]:
    exchange = layout.get("newest_tool_exchange")
    if not isinstance(exchange, list):
        raise FrameRequestAllocationError("source_layout.newest_tool_exchange must be a list")
    if not exchange:
        if layout.get("newest_exchange_state") != "NO_TOOL_CALLS":
            raise FrameRequestAllocationError("empty exchange lacks an explicit no-tool-calls binding")
        return exchange, _integer(
            layout.get("newest_exchange_start"), "source_layout.newest_exchange_start",
        )
    for index, row in enumerate(exchange):
        _message(
            row, f"source_layout.newest_tool_exchange[{index}]", {"assistant", "tool"}
        )
    first = _object(exchange[0], "source_layout.newest_tool_exchange[0]")
    if first.get("role") != "assistant":
        raise FrameRequestAllocationError(
            "newest tool exchange must start with assistant tool calls"
        )
    raw_calls = first.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        raise FrameRequestAllocationError(
            "newest tool exchange has no tool-call inventory"
        )
    call_ids = [
        _text(
            _object(call, f"newest tool call {index}").get("id"), "newest tool-call id"
        )
        for index, call in enumerate(raw_calls)
    ]
    result_ids = [
        _text(row.get("tool_call_id"), f"newest_tool_exchange[{index}].tool_call_id")
        for index, row in enumerate(exchange[1:], start=1)
        if isinstance(row, Mapping) and row.get("role") == "tool"
    ]
    if (
        len(call_ids) != len(set(call_ids))
        or len(result_ids) != len(exchange) - 1
        or len(result_ids) != len(set(result_ids))
        or set(result_ids) != set(call_ids)
    ):
        raise FrameRequestAllocationError(
            "newest tool exchange results do not match tool calls"
        )
    exchange_start = _integer(
        layout.get("newest_exchange_start"),
        "source_layout.newest_exchange_start",
    )
    return exchange, exchange_start


def _validate_current_user_facts(
    value: object,
    suffix_start: int,
) -> tuple[set[str], set[int]]:
    if not isinstance(value, list):
        raise FrameRequestAllocationError(
            "source_layout.current_user_facts must be a list"
        )
    fact_ids: set[str] = set()
    fact_locations: list[int] = []
    insert_locations: set[int] = set()
    for index, raw in enumerate(value):
        fact = _object(raw, f"source_layout.current_user_facts[{index}]")
        fact_id = _text(fact.get("fact_id"), "current_user_facts.fact_id")
        location = _integer(fact.get("location"), "current_user_facts.location")
        _message(fact.get("message"), "current_user_facts.message", {"user"})
        if fact_id in fact_ids or location in insert_locations:
            raise FrameRequestAllocationError(
                "source_layout facts contain duplicate identity/location"
            )
        fact_ids.add(fact_id)
        fact_locations.append(location)
        insert_locations.add(location)
        if location >= suffix_start:
            raise FrameRequestAllocationError(
                "current-user facts must precede newest user suffix"
            )
    if fact_locations != sorted(fact_locations):
        raise FrameRequestAllocationError(
            "current-user facts are not in allocated order"
        )
    return fact_ids, insert_locations


def _validate_source_carriers(
    value: object,
    suffix: list[object],
    suffix_start: int,
    exchange: list[object],
    exchange_start: int,
    insert_locations: set[int],
) -> tuple[set[str], set[str]]:
    if not isinstance(value, list):
        raise FrameRequestAllocationError("source_layout.carriers must be a list")
    carrier_ids: set[str] = set()
    carrier_locations: set[int] = set()
    ordered_locations: list[int] = []
    represented_handles: set[str] = set()
    for index, raw in enumerate(value):
        carrier = _object(raw, f"source_layout.carriers[{index}]")
        carrier_id = _text(carrier.get("carrier_id"), "source_layout.carrier_id")
        kind = carrier.get("kind")
        if kind not in {"prepared_source", "newest_user_source", "newest_tool_result"}:
            raise FrameRequestAllocationError("source_layout carrier kind is invalid")
        location = _integer(carrier.get("location"), "source_layout.carrier.location")
        if carrier_id in carrier_ids or location in carrier_locations:
            raise FrameRequestAllocationError(
                "source_layout contains duplicate carrier identity/location"
            )
        carrier_ids.add(carrier_id)
        carrier_locations.add(location)
        ordered_locations.append(location)
        role = "tool" if kind == "newest_tool_result" else "user"
        message = _message(
            carrier.get("message"), f"source_layout.carriers[{index}].message", {role}
        )
        handles = carrier.get("unit_handles")
        unit_handles = (
            _string_list(handles, f"source_layout.carriers[{index}].unit_handles")
            if handles is not None
            else []
        )
        if kind in {"prepared_source", "newest_user_source"} and not unit_handles:
            raise FrameRequestAllocationError("source carrier has no unit handles")
        if represented_handles.intersection(unit_handles):
            raise FrameRequestAllocationError(
                "source unit is delivered by multiple carriers"
            )
        represented_handles.update(unit_handles)
        if kind == "prepared_source":
            if location >= suffix_start or location in insert_locations:
                raise FrameRequestAllocationError(
                    "prepared source carrier location is invalid"
                )
            insert_locations.add(location)
        elif kind == "newest_user_source":
            offset = location - suffix_start
            if offset < 0 or offset >= len(suffix) or suffix[offset] != message:
                raise FrameRequestAllocationError(
                    "newest-user source carrier moved from its suffix"
                )
        else:
            offset = location - exchange_start
            if offset < 0 or offset >= len(exchange) or exchange[offset] != message:
                raise FrameRequestAllocationError(
                    "newest tool result carrier moved from its exchange"
                )
    if ordered_locations != sorted(ordered_locations):
        raise FrameRequestAllocationError("source carriers are not in allocated order")
    return carrier_ids, represented_handles


def _validate_source_layout(value: object) -> set[str]:
    layout = _object(value, "source_layout")
    task = _message(layout.get("task_message"), "source_layout.task_message", {"user"})
    task_content = task.get("content")
    if not isinstance(task_content, (str, list, type(None))) or (
        isinstance(task_content, str) and task_content.startswith("[StateFrame]\n")
    ):
        raise FrameRequestAllocationError(
            "source_layout.task_message content is invalid"
        )

    suffix = layout.get("newest_user_suffix")
    if not isinstance(suffix, list):
        raise FrameRequestAllocationError(
            "source_layout.newest_user_suffix must be a list"
        )
    for index, row in enumerate(suffix):
        _message(row, f"source_layout.newest_user_suffix[{index}]", {"user"})
    suffix_start = _integer(
        layout.get("newest_user_suffix_start"),
        "source_layout.newest_user_suffix_start",
    )
    exchange, exchange_start = _validate_tool_exchange_layout(layout)
    if suffix_start + len(suffix) != exchange_start:
        raise FrameRequestAllocationError(
            "newest user suffix must directly precede newest exchange"
        )

    fact_ids, insert_locations = _validate_current_user_facts(
        layout.get("current_user_facts"),
        suffix_start,
    )
    carrier_ids, represented_handles = _validate_source_carriers(
        layout.get("carriers"),
        suffix,
        suffix_start,
        exchange,
        exchange_start,
        insert_locations,
    )
    if carrier_ids & fact_ids:
        raise FrameRequestAllocationError(
            "source_layout fact and carrier identities overlap"
        )
    return represented_handles


def _validate_optional_heads(value: object) -> None:
    heads = _object(value, "optional_heads")
    candidates = heads.get("candidates")
    if not isinstance(candidates, list):
        raise FrameRequestAllocationError("optional_heads.candidates must be a list")
    candidate_ids: set[str] = set()
    candidate_order: list[str] = []
    for index, raw in enumerate(candidates):
        candidate = _object(raw, f"optional_heads.candidates[{index}]")
        candidate_id = _text(
            candidate.get("candidate_id"), "optional head candidate_id"
        )
        _text(candidate.get("source_revision"), "optional head source_revision")
        candidate_order.append(candidate_id)
        dependencies = _object(
            candidate.get("dependency_revisions"), "dependency_revisions"
        )
        for path, revision in dependencies.items():
            _text(path, "dependency path")
            _text(revision, "dependency revision")
        if candidate_id in candidate_ids:
            raise FrameRequestAllocationError(
                "optional head candidates contain duplicates"
            )
        candidate_ids.add(candidate_id)
    retained = _string_list(
        heads.get("retained_identities"), "optional_heads.retained_identities"
    )
    if not set(retained).issubset(candidate_ids) or retained != [
        identity for identity in candidate_order if identity in set(retained)
    ]:
        raise FrameRequestAllocationError("retained head order or inventory is invalid")
    for field in ("allowance", "omitted_count", "deduplicated_count"):
        _integer(heads.get(field), f"optional_heads.{field}")
    _text(heads.get("render_policy_identity"), "optional_heads.render_policy_identity")


def _validate_site_inventory(value: object) -> None:
    inventory = _object(value, "site_inventory")
    all_ids = _string_list(
        inventory.get("inventory_identity"), "site_inventory.inventory_identity"
    )
    selected = _string_list(
        inventory.get("selected_identities"),
        "site_inventory.selected_identities",
    )
    open_sites = _string_list(inventory.get("open_sites"), "site_inventory.open_sites")
    selected_set = set(selected)
    if selected != all_ids:
        raise FrameRequestAllocationError("semantic site inventory cannot be shed")
    if not selected_set.issubset(all_ids) or selected != [
        identity for identity in all_ids if identity in selected_set
    ]:
        raise FrameRequestAllocationError("selected site order or inventory is invalid")
    mode = inventory.get("encoding_mode")
    next_site = inventory.get("next_site")
    if next_site is not None:
        _text(next_site, "site_inventory.next_site")
        if next_site not in all_ids:
            raise FrameRequestAllocationError(
                "next site is absent from semantic inventory"
            )
    if mode == "singleton_next_site":
        if (
            not isinstance(next_site, str)
            or next_site not in selected_set
            or open_sites != [next_site]
            or all_ids != [next_site]
        ):
            raise FrameRequestAllocationError(
                "singleton site encoding is not an exact next-site alias"
            )
    elif mode in {"full_inventory", "location_inventory"}:
        if open_sites != all_ids:
            raise FrameRequestAllocationError(
                "full site encoding differs from semantic inventory"
            )
        if mode == "location_inventory" and (len(all_ids) < 2 or next_site != all_ids[0]):
            raise FrameRequestAllocationError("location inventory lacks its resident frontier")
    else:
        raise FrameRequestAllocationError("site_inventory.encoding_mode is invalid")


def _validate_outcome(value: object) -> None:
    outcome = _object(value, "outcome")
    status = outcome.get("status")
    if status not in {"COMPLETE", "DEFERRED"}:
        raise FrameRequestAllocationError(
            "allocation outcome must be COMPLETE or DEFERRED"
        )
    route = outcome.get("read_route")
    if status == "DEFERRED":
        route_map = _object(route, "outcome.read_route")
        if route_map.get("executable") is not True:
            raise FrameRequestAllocationError(
                "deferred allocation lacks executable read route"
            )
        _text(route_map.get("runner_identity"), "outcome.read_route.runner_identity")
        _text(
            route_map.get("tool_schema_digest"), "outcome.read_route.tool_schema_digest"
        )
    elif route is not None:
        _object(route, "outcome.read_route")
    _text(outcome.get("final_verification_digest"), "outcome.final_verification_digest")
    _text(outcome.get("body_digest"), "outcome.body_digest")


def _validate_document(document: Mapping[str, object]) -> None:
    if set(document) != _TOP_LEVEL_KEYS:
        raise FrameRequestAllocationError(
            "allocation has an invalid top-level field set"
        )
    if document.get("schema") != ALLOCATION_SCHEMA:
        raise FrameRequestAllocationError("allocation schema is unsupported")
    if document.get("mode") != ALLOCATION_MODE:
        raise FrameRequestAllocationError("allocation mode must be FRAME_ONLY")
    _validate_identity(document.get("identity"))
    _validate_capacity(document.get("capacity"))
    _validate_obligations(document.get("source_obligations"))
    represented_handles = _validate_source_layout(document.get("source_layout"))
    _validate_optional_heads(document.get("optional_heads"))
    _validate_site_inventory(document.get("site_inventory"))
    _validate_outcome(document.get("outcome"))
    obligations = document.get("source_obligations")
    obligation_rows = obligations if isinstance(obligations, list) else []
    obligation_handles = {
        str(row["handle"]) for row in obligation_rows if isinstance(row, Mapping)
    }
    retained_handles = {
        str(row["handle"])
        for row in obligation_rows
        if isinstance(row, Mapping) and row.get("disposition") == "RETAINED"
    }
    deferred_handles = {
        str(row["handle"])
        for row in obligation_rows
        if isinstance(row, Mapping) and row.get("disposition") == "DEFERRED"
    }
    if (
        not retained_handles.issubset(represented_handles)
        or deferred_handles & represented_handles
        or represented_handles - obligation_handles
    ):
        raise FrameRequestAllocationError(
            "source carriers disagree with source obligations"
        )
    outcome = document.get("outcome")
    has_deferred_sources = isinstance(obligations, list) and any(
        isinstance(row, Mapping) and row.get("disposition") == "DEFERRED"
        for row in obligations
    )
    if (
        isinstance(outcome, Mapping)
        and (outcome.get("status") == "DEFERRED") != has_deferred_sources
    ):
        raise FrameRequestAllocationError(
            "allocation outcome disagrees with source obligations"
        )


@dataclass(frozen=True, slots=True)
class FrameRequestAllocation:
    """A digest-checked, JSON-detached FRAME_ONLY allocation snapshot."""

    _payload: bytes
    digest: str

    def __post_init__(self) -> None:
        if type(self._payload) is not bytes:
            raise FrameRequestAllocationError(
                "allocation payload must be immutable bytes"
            )
        document = _decode_document(self._payload)
        _validate_document(document)
        if canonical_allocation_bytes(document) != self._payload:
            raise FrameRequestAllocationError("allocation payload is not canonical")
        if not verify_frame_request_allocation_digest(self._payload, self.digest):
            raise FrameRequestAllocationError(
                "allocation digest does not match its contents"
            )

    @classmethod
    def create(
        cls,
        *,
        identity: Mapping[str, object],
        capacity: Mapping[str, object],
        source_obligations: Sequence[Mapping[str, object]],
        source_layout: Mapping[str, object],
        optional_heads: Mapping[str, object],
        site_inventory: Mapping[str, object],
        outcome: Mapping[str, object],
    ) -> FrameRequestAllocation:
        document: dict[str, object] = {
            "schema": ALLOCATION_SCHEMA,
            "mode": ALLOCATION_MODE,
            "identity": dict(identity),
            "capacity": dict(capacity),
            "source_obligations": [dict(row) for row in source_obligations],
            "source_layout": dict(source_layout),
            "optional_heads": dict(optional_heads),
            "site_inventory": dict(site_inventory),
            "outcome": dict(outcome),
        }
        _validate_document(document)
        payload = canonical_allocation_bytes(document)
        return cls(payload, frame_request_allocation_digest(payload))

    @classmethod
    def from_bytes(cls, encoded: bytes) -> FrameRequestAllocation:
        if type(encoded) is not bytes:
            raise FrameRequestAllocationError("serialized allocation must be bytes")
        document = _decode_document(encoded)
        digest = document.pop("digest", None)
        if not isinstance(digest, str):
            raise FrameRequestAllocationError("serialized allocation has no digest")
        payload = canonical_allocation_bytes(document)
        if canonical_allocation_bytes({**document, "digest": digest}) != encoded:
            raise FrameRequestAllocationError("serialized allocation is not canonical")
        return cls(payload, digest)

    def validate(self) -> None:
        """Recheck the domain-separated digest before using captured allocation data."""
        if not verify_frame_request_allocation_digest(self._payload, self.digest):
            raise FrameRequestAllocationError(
                "allocation digest does not match its contents"
            )

    def to_bytes(self) -> bytes:
        self.validate()
        document = _decode_document(self._payload)
        document["digest"] = self.digest
        return canonical_allocation_bytes(document)

    def to_document(self) -> dict[str, object]:
        document = _decode_document(self._payload)
        document["digest"] = self.digest
        return document

    @property
    def mode(self) -> str:
        return ALLOCATION_MODE

    @property
    def identity(self) -> dict[str, object]:
        return cast(dict[str, object], self._section("identity"))

    @property
    def capacity(self) -> dict[str, object]:
        return cast(dict[str, object], self._section("capacity"))

    @property
    def source_obligations(self) -> list[dict[str, object]]:
        return cast(list[dict[str, object]], self._section("source_obligations"))

    @property
    def source_layout(self) -> dict[str, object]:
        return cast(dict[str, object], self._section("source_layout"))

    @property
    def optional_heads(self) -> dict[str, object]:
        return cast(dict[str, object], self._section("optional_heads"))

    @property
    def site_inventory(self) -> dict[str, object]:
        return cast(dict[str, object], self._section("site_inventory"))

    @property
    def outcome(self) -> dict[str, object]:
        return cast(dict[str, object], self._section("outcome"))

    def _section(self, name: str) -> object:
        document = _decode_document(self._payload)
        return document[name]


def _decode_document(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise FrameRequestAllocationError(
            "allocation payload is not valid JSON"
        ) from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise FrameRequestAllocationError("allocation payload must be an object")
    return cast(dict[str, object], value)
