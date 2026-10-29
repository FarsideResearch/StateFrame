"""Small release qualification capsules derived from public settled receipts."""
from __future__ import annotations
import json
import re
from collections.abc import Mapping

READ_PREFIX = "[WIRE_READ_CONTRACT] "


def read_contract_capsule(runtime: object) -> tuple[str | None, dict[str, object] | None]:
    from kernel_source_delivery import READ_CONTRACT_FIELDS
    rows = getattr(runtime, "observations", ()) if runtime is not None else ()
    for row in reversed(tuple(rows)):
        if not isinstance(row, Mapping) or str(row.get("action", "")).split(" ", 1)[0] != "read":
            continue
        if not all(field in row for field in READ_CONTRACT_FIELDS):
            continue
        fields = {field: row[field] for field in READ_CONTRACT_FIELDS}
        if fields["effective_range"] is None:
            continue
        document = {"schema": "read-contract-wire/1", "receipt_id": row.get("receipt_id"),
                    "read_contract": fields}
        return READ_PREFIX + json.dumps(document, sort_keys=True, separators=(",", ":")), fields
    return None, None


def scope_turns_on_wire(messages: object) -> list[dict[str, object]]:
    result = []
    for message in messages if isinstance(messages, list) else ():
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        match = re.match(r"^Edit turn ([0-9]+)/([0-9]+): handle (.+) in (.+)\. Continue the edit\.",
                         str(message.get("content", "")))
        if match and 1 <= int(match[1]) <= int(match[2]) and int(match[2]) >= 2:
            result.append({"turn": int(match[1]), "total": int(match[2]),
                           "owner": match[3], "path": match[4]})
    return result


def latest_scope_continuation(messages):
    from kernel_step_contract import EDIT_NTURN_NO_CHANGE_LINE

    for message in reversed(list(messages)):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        if not content.endswith("\n" + EDIT_NTURN_NO_CHANGE_LINE):
            continue
        # A later guarded/bare turn supersedes an older owner turn.
        if not scope_turns_on_wire([message]):
            return None
        return {"role": "user", "content": content, "_request_budget_section": "contract"}
    return None
