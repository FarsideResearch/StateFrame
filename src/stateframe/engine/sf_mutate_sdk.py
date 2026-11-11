"""Small, authority-free interface available to mutation units."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys


class MutationError(Exception):
    """A unit request cannot be satisfied by its declared contract."""


class NeedMore(BaseException):
    """Stop this phase before unit code can catch an ordinary Exception."""


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict
    url: str
    body: bytes
    error: str | None


def request_key(url: str, accept: str | None = None) -> str:
    request = {"method": "GET", "url": url, "accept": accept or ""}
    data = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


class MutationContext:
    def __init__(self, doc: dict):
        self.call = doc["call"]
        self.inputs = {row["id"]: Path(row["path"]) for row in doc.get("inputs", [])}
        names: dict[str, list[str]] = {}
        for row in doc.get("inputs", []):
            names.setdefault(row.get("name", ""), []).append(row["id"])
        # Units often pass the file they were given instead of its id; accept an unambiguous file name.
        self._by_name = {name: ids[0] for name, ids in names.items() if name and len(ids) == 1}
        self._labels = ", ".join(f"{row['id']} ({row.get('name', '')})" for row in doc.get("inputs", []))
        self._responses = {row["key"]: row for row in doc.get("responses", [])}
        self._network = doc.get("grants", {}).get("network") is True
        self._writes = doc.get("grants", {}).get("writes") is True
        self._max_fetch = doc.get("max_fetch", 4)
        self.pending_fetches: list[dict] = []
        self.edits: list[dict] = []

    def input_bytes(self, id: str, limit: int | None = None) -> bytes:
        if id not in self.inputs and isinstance(id, str):
            id = self._by_name.get(id.replace("\\", "/").rsplit("/", 1)[-1], id)
        if id not in self.inputs:
            raise MutationError(f"unknown input id {str(id)[:80]!r}; inputs: {self._labels or 'none'}")
        if limit is not None and (type(limit) is not int or limit < 0):
            raise MutationError("input limit must be nonnegative")
        with self.inputs[id].open("rb") as stream:
            return stream.read(limit if limit is not None else -1)

    def input_text(self, id: str, encoding: str = "utf-8", errors: str = "replace",
                   limit: int | None = None) -> str:
        return self.input_bytes(id, limit).decode(encoding, errors)

    def _response(self, url: str, accept: str | None) -> Response | None:
        if not self._network:
            raise MutationError("network grant is disabled")
        if not isinstance(url, str) or not url or len(url) > 2048:
            raise MutationError("invalid fetch URL")
        if accept is not None and (not isinstance(accept, str) or len(accept) > 8192):
            raise MutationError("invalid accept header")
        key = request_key(url, accept)
        row = self._responses.get(key)
        if row is not None:
            body = Path(row["body"]).read_bytes() if row.get("body") else b""
            return Response(row["status"], dict(row["headers"]), row["final_url"], body,
                            row.get("error"))
        request = {"method": "GET", "url": url, "accept": accept or ""}
        if request not in self.pending_fetches:
            if len(self.pending_fetches) >= self._max_fetch:
                raise MutationError("fetch count exceeded")
            self.pending_fetches.append(request)
        return None

    def fetch(self, url: str, accept: str | None = None) -> Response:
        result = self._response(url, accept)
        if result is None:
            raise NeedMore()
        return result

    def fetch_many(self, requests: list[tuple[str, str | None]]) -> list[Response]:
        results = [self._response(url, accept) for url, accept in requests]
        if any(result is None for result in results):
            raise NeedMore()
        return [result for result in results if result is not None]

    def propose_edit(self, op: str, path: str, text: str | None = None,
                     base_sha256: str | None = None) -> None:
        if not self._writes:
            raise MutationError("write grant is disabled")
        if op not in {"replace", "create", "delete"} or not isinstance(path, str):
            raise MutationError("invalid edit proposal")
        self.edits.append({"op": op, "path": path, "text": text, "base_sha256": base_sha256})

    def log(self, msg: str) -> None:
        print(str(msg), file=sys.stderr)
