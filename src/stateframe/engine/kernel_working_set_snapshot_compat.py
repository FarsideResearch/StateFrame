"""Read-only materialization of recorded source-reference cache overlays.

Some recorded wires inline authority values through ``source_refs``. The live
cache reader intentionally accepts only its current grammar. This snapshot-only
adapter reconstructs those old values before the allocator owns their bytes.
It never changes a recorded or submitted wire or grants exposure authority.
"""
from __future__ import annotations

import copy
import json


def snapshot_semantic_messages(messages):
    from kernel_cache_layout_view import semantic_messages, CacheLayoutError
    from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER

    try:
        return semantic_messages(messages)
    except CacheLayoutError:
        # Only this precise recorded extension is understood. Other malformed
        # overlays remain failures, never an excuse to drop unknown authority.
        rows = copy.deepcopy(messages)
        content = rows[-1].get("content")
        marker = AUTHORITY_OVERLAY_HEADER + "\n"
        if not isinstance(content, str) or content.count(marker) != 1:
            raise
        prefix, encoded = content.split(marker)
        overlay, consumed = json.JSONDecoder().raw_decode(encoded)
        references = []
        for field in overlay.get("fields", ()):
            if "source_refs" not in field:
                continue
            if set(field) != {"path", "source_refs", "value"}:
                raise ValueError("unsupported recorded source-reference field")
            references.append((field["path"], field.pop("source_refs")))
        if not references:
            raise
        rows[-1]["content"] = prefix + marker + json.dumps(overlay) + encoded[consumed:]
        restored = semantic_messages(rows)
        frames = [row for row in restored if isinstance(row.get("content"), str)
                  and row["content"].startswith("[StateFrame]\n")]
        if len(frames) != 1:
            raise ValueError("recorded overlay lacks a unique frame")
        frame = json.loads(frames[0]["content"].split("\n", 1)[1])
        for path, refs in references:
            if not isinstance(path, list) or not isinstance(refs, list):
                raise ValueError("invalid recorded source-reference coordinates")
            target = frame
            for part in path:
                target = target[part]
            for ref in refs:
                if not isinstance(ref, dict) or set(ref) != {"key", "message", "span"}:
                    raise ValueError("invalid recorded source-reference binding")
                index, span, key = ref["message"], ref["span"], ref["key"]
                if (type(index) is not int or not 0 <= index < len(messages)
                        or not isinstance(span, list) or len(span) != 2
                        or any(type(n) is not int for n in span)
                        or not isinstance(target, dict) or key not in target
                        or target[key] is not None):
                    raise ValueError("invalid recorded source-reference target")
                # source_refs indexes are physical submitted-wire positions;
                # moved authority rows use the separate original-order map.
                source = messages[index].get("content")
                start, end = span
                if not isinstance(source, str) or not 0 <= start <= end <= len(source):
                    raise ValueError("recorded source-reference span exceeds carrier")
                target[key] = source[start:end]
        frames[0]["content"] = "[StateFrame]\n" + json.dumps(frame, sort_keys=True, separators=(",", ":"))
        return restored
