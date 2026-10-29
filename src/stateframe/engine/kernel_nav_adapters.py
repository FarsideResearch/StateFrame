"""Generic typed navigation adapters for code, documents, data, and corpora.

Adapters project domain-owned records into the shared inspect vocabulary. They
do not own storage, parsing, ACL policy, embeddings, or a graph database. A
caller supplies an immutable record set and receives snapshot/version-bound
receipts with honest coverage and continuation state.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache


def _fallback_normalize_inspect_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    return dict(arguments)


def _fallback_validate_inspect_semantics(arguments: Mapping[str, object]) -> None:
    del arguments


@lru_cache(maxsize=1)
def _inspect_contract() -> tuple[object, object, object]:
    from kernel_optional_engine_module import optional_module
    contract = optional_module("STATEFRAME_NAV_CONTRACT_MODULE")
    if contract is None:
        return ValueError, _fallback_normalize_inspect_arguments, _fallback_validate_inspect_semantics
    return (
        contract.InspectContractError,
        contract.normalize_inspect_arguments,
        contract.validate_inspect_semantics,
    )


def __getattr__(name: str) -> object:
    contract_names = {
        "InspectContractError": 0,
        "normalize_inspect_arguments": 1,
        "validate_inspect_semantics": 2,
    }
    if name not in contract_names:
        raise AttributeError(name)
    return _inspect_contract()[contract_names[name]]


SCHEMA = "stateframe/nav-domain-adapter/1"
CURSOR_SCHEMA = "stateframe/nav-domain-cursor/1"
DOMAINS = frozenset({"code", "document", "section", "dataset", "schema", "partition", "corpus"})
MAX_PAGE_SIZE = 24
MAX_BODY_BYTES = 256 * 1024


class AdapterContractError(ValueError):
    """Raised for invalid immutable adapter input or cursor use."""


@dataclass(frozen=True, slots=True)
class DomainItem:
    item_id: str
    locator: str
    body: bytes
    kind: str
    version: str
    acl: str = "public"
    parent_id: str | None = None
    relations: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    parse_status: str = "OK"
    semantic_available: bool = False

    def __post_init__(self) -> None:
        for name, value in (("item_id", self.item_id), ("locator", self.locator), ("kind", self.kind), ("version", self.version), ("acl", self.acl)):
            if not isinstance(value, str) or not value.strip() or len(value) > 1024:
                raise AdapterContractError(f"domain item {name} is invalid")
        if not isinstance(self.body, bytes):
            raise AdapterContractError("domain item body must be immutable bytes")
        if len(self.body) > MAX_BODY_BYTES:
            raise AdapterContractError("domain item body exceeds the adapter byte bound")
        if self.parse_status not in {"OK", "FAILED_OCR", "FAILED_PARSE", "UNSUPPORTED"}:
            raise AdapterContractError("domain item parse_status is invalid")
        if self.parent_id is not None and not isinstance(self.parent_id, str):
            raise AdapterContractError("domain item parent_id is invalid")
        object.__setattr__(self, "relations", tuple(dict.fromkeys(str(item) for item in self.relations if str(item))))
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))

    @property
    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.body).hexdigest()


@dataclass(frozen=True, slots=True)
class DomainCursor:
    handle: str
    snapshot_id: str
    authorization_epoch: str
    query_digest: str
    version_digest: str
    offset: int
    expires_at: int

    def payload(self) -> dict[str, object]:
        return {
            "schema": CURSOR_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "authorization_epoch": self.authorization_epoch,
            "query_digest": self.query_digest,
            "version_digest": self.version_digest,
            "offset": self.offset,
            "expires_at": self.expires_at,
        }


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _make_cursor(
    *, snapshot_id: str, authorization_epoch: str, query_digest: str,
    version_digest: str, offset: int, now: int, ttl: int,
) -> DomainCursor:
    payload = {
        "schema": CURSOR_SCHEMA,
        "snapshot_id": snapshot_id,
        "authorization_epoch": authorization_epoch,
        "query_digest": query_digest,
        "version_digest": version_digest,
        "offset": offset,
        "expires_at": now + ttl,
    }
    encoded = base64.urlsafe_b64encode(_canonical(payload)).decode("ascii").rstrip("=")
    handle = "navcursor:" + _digest(payload) + "." + encoded
    return DomainCursor(handle, snapshot_id, authorization_epoch, query_digest, version_digest, offset, now + ttl)


def _decode_cursor(value: str) -> DomainCursor:
    if not isinstance(value, str) or not value.startswith("navcursor:") or "." not in value:
        raise AdapterContractError("invalid domain continuation")
    digest, encoded = value[len("navcursor:"):].split(".", 1)
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterContractError("invalid domain continuation payload") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != CURSOR_SCHEMA or _digest(payload) != digest:
        raise AdapterContractError("domain continuation signature mismatch")
    try:
        return DomainCursor(
            value, str(payload["snapshot_id"]), str(payload["authorization_epoch"]),
            str(payload["query_digest"]), str(payload["version_digest"]),
            int(payload["offset"]), int(payload["expires_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterContractError("domain continuation fields are invalid") from exc


class GenericDomainAdapter:
    """Read-only adapter with one operation vocabulary across domain types."""

    def __init__(
        self,
        domain: str,
        items: Sequence[DomainItem],
        *,
        snapshot_id: str,
        authorization_epoch: str = "auth:initial",
        sampled: bool = False,
        semantic_channel_available: bool = False,
        baseline: Mapping[str, bytes] | None = None,
        cursor_ttl_seconds: int = 300,
    ) -> None:
        if domain not in DOMAINS:
            raise AdapterContractError("unsupported navigation domain")
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise AdapterContractError("snapshot_id is required")
        if not isinstance(authorization_epoch, str) or not authorization_epoch.strip():
            raise AdapterContractError("authorization_epoch is required")
        if type(cursor_ttl_seconds) is not int or cursor_ttl_seconds < 1:
            raise AdapterContractError("cursor_ttl_seconds must be positive")
        ordered = tuple(sorted(items, key=lambda item: (item.item_id, item.locator, item.version)))
        if len({item.item_id for item in ordered}) != len(ordered):
            raise AdapterContractError("domain item IDs must be unique within a snapshot")
        self.domain = domain
        self.items = ordered
        self.snapshot_id = snapshot_id
        self.authorization_epoch = authorization_epoch
        self.sampled = bool(sampled)
        self.semantic_channel_available = bool(semantic_channel_available)
        self.baseline = {str(key): bytes(value) for key, value in (baseline or {}).items()}
        self.cursor_ttl_seconds = cursor_ttl_seconds
        # ACL/parse state is part of the pagination identity.  A cursor must
        # not silently cross an authorization or availability change.
        self.version_digest = "version:" + _digest([
            (item.item_id, item.version, item.acl, item.parse_status, item.digest)
            for item in ordered
        ])

    def _visible(self, principal: str) -> tuple[DomainItem, ...]:
        return tuple(item for item in self.items if item.acl in {"public", principal})

    def _coverage(self, visible: Sequence[DomainItem], *, failed: int | None = None) -> dict[str, object]:
        failed_count = sum(item.parse_status != "OK" for item in visible) if failed is None else failed
        complete = not self.sampled and failed_count == 0
        return {
            "predicate": f"{self.domain}:authorized-snapshot",
            "status": "COMPLETE" if complete else "PARTIAL",
            "complete": complete,
            "sampled": self.sampled,
            "visible_items": len(visible),
            "failed_items": failed_count,
            "reason": "sampled_or_failed_parse" if not complete else None,
        }

    def _rows(self, items: Sequence[DomainItem]) -> list[dict[str, object]]:
        return [
            {
                "item_id": item.item_id,
                "locator": item.locator,
                "kind": item.kind,
                "version": item.version,
                "source_digest": item.digest,
                "parse_status": item.parse_status,
                "metadata": copy.deepcopy(dict(item.metadata)),
            }
            for item in items
        ]

    def _page(
        self,
        rows: Sequence[DomainItem],
        *,
        query: object,
        page_size: int,
        cursor: str | None,
        now: int,
    ) -> dict[str, object]:
        if type(page_size) is not int or not 1 <= page_size <= MAX_PAGE_SIZE:
            raise AdapterContractError("page_size is outside the bounded range")
        query_digest = "query:" + _digest(query)
        offset = 0
        if cursor:
            token = _decode_cursor(cursor)
            if token.snapshot_id != self.snapshot_id or token.authorization_epoch != self.authorization_epoch:
                raise AdapterContractError("domain continuation is stale or unauthorized")
            if token.query_digest != query_digest or token.version_digest != self.version_digest:
                raise AdapterContractError("domain continuation query/version changed")
            if token.expires_at < now:
                raise AdapterContractError("domain continuation expired")
            offset = token.offset
        page = tuple(rows[offset:offset + page_size])
        next_cursor = None
        if offset + page_size < len(rows):
            next_cursor = _make_cursor(
                snapshot_id=self.snapshot_id, authorization_epoch=self.authorization_epoch,
                query_digest=query_digest, version_digest=self.version_digest,
                offset=offset + page_size, now=now, ttl=self.cursor_ttl_seconds,
            )
        return {
            "rows": self._rows(page),
            "counts": {"shown": len(page), "total": len(rows) if not self.sampled else None, "total_relation": "exact" if not self.sampled else "unknown"},
            "next_handle": next_cursor.handle if next_cursor else None,
        }

    def _read_item(self, item: DomainItem, view: str, since: str | None) -> dict[str, object]:
        if item.parse_status != "OK":
            return {
                "status": "PARTIAL",
                "item_id": item.item_id,
                "locator": item.locator,
                "omission": {"reason": item.parse_status, "availability": "UNAVAILABLE"},
                "coverage": "parse_failed",
            }
        if view == "diff":
            baseline = self.baseline.get(str(since or ""))
            if baseline is None:
                return {
                    "status": "BASELINE_UNAVAILABLE",
                    "item_id": item.item_id,
                    "locator": item.locator,
                    "source_digest": item.digest,
                    "body": item.body.decode("utf-8", errors="replace"),
                    "expansion_handle": "expand:" + _digest((self.snapshot_id, item.item_id, item.digest)),
                }
            before = baseline.decode("utf-8", errors="replace")
            after = item.body.decode("utf-8", errors="replace")
            return {"status": "FOUND", "item_id": item.item_id, "locator": item.locator, "view": view, "since": since, "diff": "" if before == after else f"- {before}\n+ {after}"}
        return {
            "status": "FOUND",
            "item_id": item.item_id,
            "locator": item.locator,
            "view": view,
            "source_digest": item.digest,
            "version": item.version,
            "body": item.body.decode("utf-8", errors="replace"),
            "metadata": copy.deepcopy(dict(item.metadata)),
            "expansion_handle": "expand:" + _digest((self.snapshot_id, item.item_id, item.digest, view)),
        }

    def execute(
        self,
        arguments: Mapping[str, object],
        *,
        principal: str = "public",
        now: int | None = None,
    ) -> dict[str, object]:
        try:
            return self._execute(arguments, principal=principal, now=now)
        except AdapterContractError as exc:
            detail = str(exc)
            reason = "CONTINUATION_REJECTED" if "continuation" in detail or "cursor" in detail else "ADAPTER_REJECTED"
            return {
                "schema": SCHEMA,
                "domain": self.domain,
                "status": "STALE" if reason == "CONTINUATION_REJECTED" else "REJECTED",
                "reason_code": reason,
                "snapshot_id": self.snapshot_id,
                "detail": detail,
            }

    def _execute(
        self,
        arguments: Mapping[str, object],
        *,
        principal: str = "public",
        now: int | None = None,
    ) -> dict[str, object]:
        inspect_error, normalize_arguments, validate_arguments = _inspect_contract()
        try:
            request = normalize_arguments(arguments)
            validate_arguments(request)
        except (inspect_error, TypeError, ValueError) as exc:
            return {"schema": SCHEMA, "status": "REJECTED", "reason_code": "INVALID_ARGUMENTS", "detail": str(exc)}
        current = int(time.time()) if now is None else int(now)
        visible = self._visible(principal)
        op = request.get("op")
        common = {"schema": SCHEMA, "domain": self.domain, "operation": op, "snapshot_id": self.snapshot_id, "version_digest": self.version_digest}
        if op == "orient":
            rows = visible[:MAX_PAGE_SIZE]
            return common | {"status": "FOUND" if rows else "NOT_FOUND", "phase": request.get("phase"), "rows": self._rows(rows), "coverage": self._coverage(visible)}
        if op == "lookup":
            query = str(request.get("query", ""))
            matches = tuple(item for item in visible if item.item_id == query or item.locator == query or str(item.metadata.get("title", "")) == query)
            status = "FOUND" if len(matches) == 1 else ("AMBIGUOUS" if matches else "NOT_FOUND")
            return common | {"status": status, "query": query, "rows": self._rows(matches[:MAX_PAGE_SIZE]), "coverage": self._coverage(visible)}
        if op == "search":
            query = str(request.get("query", ""))
            mode = str(request.get("mode", "literal"))
            semantic_unavailable = mode == "semantic" and not self.semantic_channel_available
            search_mode = "literal" if semantic_unavailable else mode
            try:
                matcher = re.compile(query) if search_mode == "pattern" else None
            except re.error:
                return common | {
                    "status": "REJECTED",
                    "reason_code": "INVALID_QUERY",
                    "detail": "pattern query is not valid",
                    "coverage": self._coverage(visible),
                }
            filter_value = request.get("filter")
            candidates = visible
            if isinstance(filter_value, Mapping):
                filter_kind = filter_value.get("kind")
                filter_path = filter_value.get("path")
                candidates = tuple(
                    item for item in visible
                    if (not isinstance(filter_kind, str) or item.kind == filter_kind)
                    and (not isinstance(filter_path, str) or item.locator == filter_path or item.locator.startswith(filter_path.rstrip("/") + "/"))
                )
            matches = []
            for item in candidates:
                haystack = item.body.decode("utf-8", errors="replace")
                found = bool(matcher.search(haystack)) if matcher is not None else query.casefold() in haystack.casefold()
                if found:
                    matches.append(item)
            page = self._page(matches, query=(query, search_mode, request.get("filter")), page_size=int(request.get("page_size", MAX_PAGE_SIZE)), cursor=request.get("cursor") if isinstance(request.get("cursor"), str) else None, now=current)
            return common | {"status": "UNAVAILABLE" if semantic_unavailable else ("FOUND" if matches else "NOT_FOUND"), "query": query, "requested_mode": mode, "effective_mode": search_mode, "semantic_channel": "UNAVAILABLE" if semantic_unavailable else ("AVAILABLE" if mode == "semantic" else "NOT_REQUESTED"), "fallback": "lexical" if semantic_unavailable else None, "coverage": self._coverage(visible), **page}
        if op == "map":
            scope = str(request.get("scope", ""))
            matches = tuple(item for item in visible if item.locator == scope or item.locator.startswith(scope.rstrip("/") + "/")) if scope else visible
            page = self._page(matches, query=("map", scope, request.get("depth", 1)), page_size=int(request.get("page_size", MAX_PAGE_SIZE)), cursor=request.get("cursor") if isinstance(request.get("cursor"), str) else None, now=current)
            return common | {"status": "FOUND" if matches else "NOT_FOUND", "scope": scope, "coverage": self._coverage(visible), **page}
        if op == "related":
            entity = request.get("entity")
            entity_id = str(entity.get("entity_id")) if isinstance(entity, Mapping) and entity.get("entity_id") else str(entity.get("path")) if isinstance(entity, Mapping) else str(entity)
            matches = tuple(item for item in visible if entity_id in item.relations or item.parent_id == entity_id)
            return common | {"status": "FOUND" if matches else "EMPTY", "relation": request.get("relation"), "rows": self._rows(matches[:MAX_PAGE_SIZE]), "coverage": self._coverage(visible)}
        if op == "read":
            if isinstance(request.get("continuation"), str):
                return common | {"status": "CONTINUATION_REQUIRES_OWNER", "next_handle": request["continuation"]}
            target = request.get("target")
            target_value = str(target.get("entity_id")) if isinstance(target, Mapping) and target.get("entity_id") else str(target.get("path")) if isinstance(target, Mapping) else str(target)
            target_handle = str(target.get("handle")) if isinstance(target, Mapping) and target.get("handle") else ""
            item = next((candidate for candidate in visible if candidate.item_id == target_value or candidate.locator == target_value), None)
            if item is None and target_handle:
                item = next(
                    (
                        candidate for candidate in visible
                        if target_handle in {
                            "expand:" + _digest((self.snapshot_id, candidate.item_id, candidate.digest)),
                            *(
                                "expand:" + _digest((self.snapshot_id, candidate.item_id, candidate.digest, item_view))
                                for item_view in ("span", "outline", "signatures", "diff")
                            ),
                        }
                    ),
                    None,
                )
            if item is None:
                return common | {"status": "NOT_FOUND", "coverage": self._coverage(visible)}
            return common | self._read_item(item, str(request.get("view")), request.get("since") if isinstance(request.get("since"), str) else None) | {"coverage": self._coverage(visible, failed=sum(candidate.parse_status != "OK" for candidate in visible))}
        return common | {"status": "REJECTED", "reason_code": "UNKNOWN_OPERATION"}


__all__ = [
    "AdapterContractError",
    "CURSOR_SCHEMA",
    "DOMAINS",
    "DomainCursor",
    "DomainItem",
    "GenericDomainAdapter",
    "SCHEMA",
]
