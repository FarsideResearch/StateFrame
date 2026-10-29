"""Bounded, parser-backed acquisition of public owner spans.

This module only acquires a small context around public identifier hits.  It
does not choose a repair target, build a graph, or cache a source snapshot.
The existing public locator and catalog remain authoritative for normal-sized
sources; this seam exists so an oversized file can still yield a truthful
owner span without reading the whole file into the bundle.
"""

from __future__ import annotations

import hashlib
import re
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


SCAN_CHUNK_BYTES = 8192
MAX_SCAN_BYTES = 256 * 1024
MAX_CONTEXT_BEFORE_BYTES = 8192
MAX_CONTEXT_AFTER_BYTES = 8192
MAX_CONTEXT_BYTES = MAX_CONTEXT_BEFORE_BYTES + MAX_CONTEXT_AFTER_BYTES
MAX_HITS = 16
MAX_CANDIDATES = 3
MAX_ACQUISITION_BYTES = MAX_SCAN_BYTES + MAX_HITS * MAX_CONTEXT_BYTES


@dataclass(frozen=True)
class _Hit:
    offset: int
    line: int
    token: str


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _identifier_byte(value: int) -> bool:
    return (48 <= value <= 57 or 65 <= value <= 90 or 97 <= value <= 122
            or value in (36, 95))


def _token_hits(raw: bytes, tokens: Sequence[str], *, base_offset: int, base_line: int) -> list[_Hit]:
    lowered = raw.lower()
    hits: list[_Hit] = []
    seen: set[tuple[int, str]] = set()
    for token in tokens:
        encoded = token.encode("utf-8", errors="ignore")
        if not encoded:
            continue
        needle = encoded.lower()
        cursor = 0
        while True:
            found = lowered.find(needle, cursor)
            if found < 0:
                break
            before = raw[found - 1] if found else (0 if base_offset == 0 else 95)
            after_index = found + len(encoded)
            after = raw[after_index] if after_index < len(raw) else 0
            if not _identifier_byte(before) and not _identifier_byte(after):
                absolute = base_offset + found
                line = base_line + raw[:found].count(b"\n")
                key = (absolute, token.casefold())
                if key not in seen:
                    seen.add(key)
                    hits.append(_Hit(absolute, line, token))
            cursor = found + max(1, len(encoded))
    return hits


def _public_tokens(identifiers: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for identifier in identifiers:
        value = str(identifier or "").strip()
        if not value:
            continue
        for token in (value, value.rsplit(".", 1)[-1], value.rsplit("::", 1)[-1]):
            if token and token not in result:
                result.append(token)
    return tuple(result)


def _read_at(handle: int, offset: int, size: int) -> bytes:
    if size <= 0:
        return b""
    return os.pread(handle, size, offset)


def _safe_target(root: Path, relative: str) -> tuple[Path | None, str | None]:
    candidate = root / relative
    if candidate.is_symlink():
        return None, "symlink"
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None, "outside_workspace"
    if not resolved.is_file() or resolved.is_symlink():
        return None, "not_regular_file"
    cursor = candidate
    while cursor != root and root in cursor.parents:
        if cursor.is_symlink():
            return None, "symlink"
        cursor = cursor.parent
    return resolved, None


def _record_matches(record: dict[str, object], identifiers: Sequence[str]) -> tuple[str, list[str], bool] | None:
    name = str(record.get("name") or "").strip()
    owner = str(record.get("owner") or "").strip()
    qualified = ".".join(item for item in (owner, name) if item)
    values = {name.casefold(), qualified.casefold()}

    def _matches(identifier: str) -> bool:
        if identifier.casefold() in values:
            return True
        qualified_identifier = "." in identifier or "::" in identifier
        if qualified_identifier and owner:
            # W8: `Beta.run` names Beta's run; a same-name sibling under a
            # different owner (Alpha.run) is not a match.
            return False
        return (identifier.rsplit(".", 1)[-1].casefold() == name.casefold()
                or identifier.rsplit("::", 1)[-1].casefold() == name.casefold())

    matches = [identifier for identifier in identifiers if _matches(identifier)]
    if not matches:
        return None
    exact = any(identifier.casefold() in {name.casefold(), qualified.casefold()} for identifier in identifiers)
    return qualified or name, matches[:4], exact


def _top_level_window(context: bytes, context_start: int, hit_offset: int,
                      total_bytes: int) -> tuple[bytes, int]:
    """Trim a byte window to top-level boundaries around the hit.

    A window that starts or ends inside a block is not valid source for an
    indentation or brace parser.  Start at the last column-0 line at or before
    the hit and end before the first column-0 line after the hit's block (or at
    the last complete line).  Language-generic: top-level declarations start at
    column 0 in Python, Go, TypeScript, Java and Rust.  Never widens the read.
    """
    relative_hit = max(0, min(len(context), hit_offset - context_start))
    lines = context.splitlines(keepends=True)
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)

    def top_level(index: int) -> bool:
        line = lines[index]
        return bool(line.strip()) and line[:1] not in (b" ", b"\t", b"}", b")", b"]")

    hit_index = max(0, next((i for i, start in enumerate(offsets) if start > relative_hit), len(lines)) - 1)
    first_complete = 1 if context_start > 0 else 0
    start_index = next((i for i in range(hit_index, first_complete - 1, -1) if top_level(i)), first_complete)
    last_complete = len(lines) - 1 if (context_start + len(context) >= total_bytes
                                       or context.endswith(b"\n")) else len(lines) - 2
    end_index = next((i for i in range(hit_index + 1, last_complete + 1) if top_level(i)), last_complete + 1)
    if start_index >= len(lines) or end_index <= start_index:
        return context, context_start
    trimmed = b"".join(lines[start_index:end_index])
    return trimmed, context_start + offsets[start_index]


def _context_candidate(
        context: bytes,
        *,
        path: str,
        context_start: int,
        context_start_line: int,
        total_bytes: int,
        max_source_chars: int,
        identifiers: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        from mc_repo_intel_vendor.ts_dispatch import structural_spans
        parsed = structural_spans(context.decode("utf-8", errors="replace"), path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        return [], [{"kind": "owner_span_parser", "path": path, "reason": type(error).__name__}]
    if not isinstance(parsed, dict) or parsed.get("degraded") or not isinstance(parsed.get("spans"), list):
        reason = str(parsed.get("reason") or "parser_degraded") if isinstance(parsed, dict) else "parser_degraded"
        return [], [{"kind": "owner_span_parser", "path": path, "reason": reason}]

    text = context.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    candidates: list[dict[str, object]] = []
    for raw_record in parsed["spans"]:
        if not isinstance(raw_record, dict) or raw_record.get("kind") in {"sfc-block", "parser-status"}:
            continue
        match = _record_matches(raw_record, identifiers)
        if match is None:
            continue
        symbol, matched, exact_match = match
        relative_start = max(1, int(raw_record.get("line", 1) or 1))
        relative_end = max(relative_start, int(raw_record.get("end_line", relative_start) or relative_start))
        start_line = context_start_line + relative_start - 1
        requested_end_line = context_start_line + relative_end - 1
        clipped_by_context = bool(context_start + len(context) < total_bytes and relative_end >= len(lines))
        selected_lines = lines[relative_start - 1:min(relative_end, len(lines))]
        selected = "".join(selected_lines)
        clipped = clipped_by_context
        if len(selected) > max_source_chars:
            selected = selected[:max_source_chars]
            clipped = True
        shown_end_line = start_line + max(0, selected.count("\n") - (1 if selected.endswith("\n") else 0))
        if selected and not selected.endswith("\n"):
            shown_end_line = start_line + selected.count("\n")
        if not selected:
            shown_end_line = start_line - 1
        parent = str(raw_record.get("owner") or "").strip()
        owner = symbol
        record_kind = str(raw_record.get("kind") or "unknown")
        type_name = parent if record_kind in {"function", "method"} and parent else (
            symbol if record_kind == "type" else None
        )
        candidates.append({
            "path": path,
            "symbol": symbol,
            "line": start_line,
            "end_line": shown_end_line,
            "requested_end_line": requested_end_line,
            "kind": record_kind,
            "owner": owner,
            "module": path,
            "type": type_name or (parent if record_kind in {"function", "method"} else None),
            "matched_identifiers": matched,
            "selection_sources": ["public_ticket", "declared_source", "bounded_source_span", "structural_parser"],
            "_segment": {
                "path": path,
                "start_line": start_line,
                "end_line": shown_end_line,
                "requested_end_line": requested_end_line,
                "content": selected,
                "content_digest": _digest(selected.encode("utf-8")),
                "clipped": clipped,
                "coverage": "PARTIAL" if clipped else "COMPLETE",
                "parser": str(raw_record.get("parser") or parsed.get("parser") or "unknown"),
            },
            "_rank": (0 if exact_match else 1, start_line, symbol),
        })
    return candidates, []


_WORD_STOP = frozenset({
    "this", "that", "with", "from", "when", "into", "than", "then", "they", "them", "their",
    "should", "would", "could", "must", "have", "been", "being", "does", "every", "only",
    "silently", "updated", "update", "crash", "crashes", "fails", "failing", "error", "bug",
})


def _name_parts(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return {part.lower() for part in re.split(r"[^A-Za-z0-9]+", spaced) if len(part) >= 3}


def ticket_words(ticket: str) -> tuple[str, ...]:
    """Content words of a natural-language ticket (language-generic, bounded)."""
    words: list[str] = []
    for raw in re.findall(r"[A-Za-z]{4,}", ticket):
        word = raw.lower()
        if word in _WORD_STOP:
            continue
        for form in (word, word[:-1] if word.endswith("s") else word):
            if form not in words:
                words.append(form)
    return tuple(words[:24])


def rank_owner_spans_by_words(
        root: Path | None, path: str, words: Sequence[str], *,
        override: bytes | str | None = None, max_source_chars: int = 6000,
) -> dict[str, object]:
    """Rank whole-file owner records by ticket-word overlap when no identifier exists.

    Host evidence only: candidates carry ``ticket_word_overlap`` provenance and
    the status is AMBIGUOUS unless one record strictly leads.  Bounded by the
    same scan budget; a file beyond it is reported, never partially guessed.
    """
    empty = {"candidates": [], "coverage": {"status": "UNAVAILABLE"}}
    vocabulary = {str(word).lower() for word in words if str(word).strip()}
    if not vocabulary:
        return {**empty, "omissions": [{"kind": "owner_span_unavailable", "path": path,
                                        "reason": "no_public_identifier"}]}
    if override is not None:
        raw = override.encode("utf-8") if isinstance(override, str) else bytes(override)
    else:
        target, reason = _safe_target(root, path) if root is not None else (None, "workspace_unavailable")
        if target is None:
            return {**empty, "omissions": [{"kind": "access_denied", "path": path, "reason": reason}]}
        if target.stat().st_size > MAX_SCAN_BYTES:
            return {**empty, "omissions": [{"kind": "word_rank_scan_bound", "path": path}]}
        raw = target.read_bytes()
    if len(raw) > MAX_SCAN_BYTES:
        return {**empty, "omissions": [{"kind": "word_rank_scan_bound", "path": path}]}
    try:
        from mc_repo_intel_vendor.ts_dispatch import structural_spans
        parsed = structural_spans(raw.decode("utf-8", errors="replace"), path)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        return {**empty, "omissions": [{"kind": "owner_span_parser", "path": path,
                                        "reason": type(error).__name__}]}
    records = parsed.get("spans") if isinstance(parsed, dict) else None
    if not isinstance(records, list) or parsed.get("degraded"):
        return {**empty, "omissions": [{"kind": "owner_span_parser", "path": path,
                                        "reason": "parser_degraded"}]}
    owners = [record for record in records
              if isinstance(record, dict) and record.get("kind") in {"function", "method", "type"}]
    # Rarity weighting within this file: a ticket word shared by many owner
    # names ("memory" in a memory service) says little; a rare one says more.
    document_frequency: dict[str, int] = {}
    for record in owners:
        for part in _name_parts(str(record.get("name") or "")):
            document_frequency[part] = document_frequency.get(part, 0) + 1
    scored: list[tuple[float, int, str]] = []
    for record in owners:
        name = str(record.get("name") or "")
        overlap = _name_parts(name) & vocabulary
        if overlap:
            weight = round(sum(1.0 / document_frequency[part] for part in overlap), 6)
            scored.append((weight, int(record.get("line", 0) or 0), name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    top = scored[:MAX_CANDIDATES]
    if not top:
        return {"candidates": [], "coverage": {"status": "COMPLETE", "read_bytes": len(raw)},
                "omissions": [{"kind": "owner_span_not_found", "path": path, "mode": "ticket_words"}]}
    rows, omissions = _context_candidate(
        raw, path=path, context_start=0, context_start_line=1, total_bytes=len(raw),
        max_source_chars=max(1, int(max_source_chars)), identifiers=[name for _o, _l, name in top],
    )
    order = {name: index for index, (_o, _l, name) in enumerate(top)}
    for row in rows:
        row["selection_sources"] = ["public_ticket", "declared_source", "structural_parser",
                                    "ticket_word_overlap"]
        row["_rank"] = (order.get(str(row.get("symbol")).rsplit(".", 1)[-1], 99), int(row["line"]),
                        str(row.get("symbol")))
    rows.sort(key=lambda row: row["_rank"])
    leads = len(top) == 1 or top[0][0] > top[1][0]
    return {"candidates": rows[:MAX_CANDIDATES], "omissions": omissions,
            "status": "FOUND" if leads and rows else "AMBIGUOUS",
            "coverage": {"status": "COMPLETE", "read_bytes": len(raw), "mode": "ticket_words"}}


def acquire_owner_spans(
        root: Path | None,
        path: str,
        identifiers: Sequence[str],
        *,
        override: bytes | str | None = None,
        max_source_chars: int = 6000,
        max_scan_bytes: int = MAX_SCAN_BYTES,
        max_context_bytes: int = MAX_CONTEXT_BYTES,
) -> dict[str, object]:
    """Acquire parser-backed owner spans under fixed scan/context budgets."""
    tokens = _public_tokens(identifiers)
    if not tokens:
        return {
            "candidates": [], "omissions": [{"kind": "owner_span_unavailable", "path": path, "reason": "no_public_identifier"}],
            "coverage": {"status": "UNAVAILABLE", "read_bytes": 0, "total_bytes": 0},
            "identity": {"mode": "override" if override is not None else "root", "content_digest": None},
        }

    handle: int | None = None
    raw_override = override.encode("utf-8") if isinstance(override, str) else bytes(override) if override is not None else None
    target: Path | None = None
    omissions: list[dict[str, object]] = []
    read_bytes = 0
    total_bytes = len(raw_override) if raw_override is not None else 0
    full_digest = _digest(raw_override) if raw_override is not None else None
    try:
        if raw_override is None:
            if root is None:
                return {
                    "candidates": [], "omissions": [{"kind": "source_unavailable", "path": path, "reason": "workspace_unavailable"}],
                    "coverage": {"status": "UNAVAILABLE", "read_bytes": 0, "total_bytes": 0},
                    "identity": {"mode": "root", "content_digest": None},
                }
            target, reason = _safe_target(root, path)
            if target is None:
                return {
                    "candidates": [], "omissions": [{"kind": "access_denied", "path": path, "reason": reason}],
                    "coverage": {"status": "UNAVAILABLE", "read_bytes": 0, "total_bytes": 0},
                    "identity": {"mode": "root", "content_digest": None},
                }
            handle = os.open(target, os.O_RDONLY)
            total_bytes = os.fstat(handle).st_size

        scan_limit = min(total_bytes, max(1, min(int(max_scan_bytes), MAX_SCAN_BYTES)))
        chunk_size = max(1, min(SCAN_CHUNK_BYTES, scan_limit or SCAN_CHUNK_BYTES))
        hits: list[_Hit] = []
        seen_hits: set[tuple[int, str]] = set()
        tail = b""
        line_before = 1
        offset = 0
        while offset < scan_limit:
            size = min(chunk_size, scan_limit - offset)
            data = raw_override[offset:offset + size] if raw_override is not None else _read_at(handle, offset, size)
            read_bytes += len(data) if raw_override is None else 0
            if not data:
                break
            combined = tail + data
            combined_offset = offset - len(tail)
            combined_line = line_before - tail.count(b"\n")
            for hit in _token_hits(combined, tokens, base_offset=combined_offset, base_line=combined_line):
                key = (hit.offset, hit.token.casefold())
                if key not in seen_hits and len(hits) < MAX_HITS:
                    seen_hits.add(key)
                    hits.append(hit)
            tail = data[-max((len(token.encode("utf-8")) for token in tokens), default=1):]
            line_before += data.count(b"\n")
            offset += len(data)
        hits.sort(key=lambda item: (item.offset, item.token.casefold()))
        if not hits:
            omissions.append({
                "kind": "owner_span_not_found" if scan_limit >= total_bytes else "owner_span_scan_bound",
                "path": path, "scanned_bytes": scan_limit, "total_bytes": total_bytes,
            })
        if len(hits) >= MAX_HITS:
            omissions.append({"kind": "owner_span_hit_bound", "path": path, "max_hits": MAX_HITS})

        candidates: list[dict[str, object]] = []
        seen_candidates: set[tuple[str, int, int]] = set()
        context_limit = max(1, min(int(max_context_bytes), MAX_CONTEXT_BYTES))
        before = min(MAX_CONTEXT_BEFORE_BYTES, context_limit // 2)
        after = context_limit - before
        whole_file_mode = bool(hits) and total_bytes <= scan_limit
        if whole_file_mode:
            # W8: the scan already read every byte within budget, so parse the
            # whole file once.  A mid-file window that starts inside a block is
            # not valid source for an indentation/brace parser (the real AHA
            # 24.8 KB target degraded to zero candidates in window mode).
            whole = (raw_override if raw_override is not None
                     else b"".join(_read_at(handle, start, min(SCAN_CHUNK_BYTES, total_bytes - start))
                                   for start in range(0, total_bytes, SCAN_CHUNK_BYTES)))
            if raw_override is None:
                read_bytes += len(whole)
            rows, row_omissions = _context_candidate(
                whole, path=path, context_start=0, context_start_line=1,
                total_bytes=total_bytes, max_source_chars=max(1, int(max_source_chars)),
                identifiers=identifiers,
            )
            omissions.extend(row_omissions)
            for row in rows:
                key = (str(row["symbol"]), int(row["line"]), int(row["requested_end_line"]))
                if key not in seen_candidates:
                    seen_candidates.add(key)
                    candidates.append(row)
        for hit in ([] if whole_file_mode else hits):
            context_start = max(0, hit.offset - before)
            context_end = min(total_bytes, hit.offset + after)
            context_size = context_end - context_start
            context = (raw_override[context_start:context_end] if raw_override is not None
                       else _read_at(handle, context_start, context_size))
            if raw_override is None:
                read_bytes += len(context)
            context, context_start = _top_level_window(context, context_start, hit.offset, total_bytes)
            prefix_length = max(0, hit.offset - context_start)
            context_start_line = hit.line - context[:prefix_length].count(b"\n")
            rows, row_omissions = _context_candidate(
                context, path=path, context_start=context_start,
                context_start_line=context_start_line, total_bytes=total_bytes,
                max_source_chars=max(1, int(max_source_chars)), identifiers=identifiers,
            )
            omissions.extend(row_omissions)
            for row in rows:
                key = (str(row["symbol"]), int(row["line"]), int(row["requested_end_line"]))
                if key not in seen_candidates:
                    seen_candidates.add(key)
                    candidates.append(row)
        candidates.sort(key=lambda row: row["_rank"])
        candidate_overflow = len(candidates) > MAX_CANDIDATES
        candidates = candidates[:MAX_CANDIDATES]
        if candidate_overflow:
            omissions.append({"kind": "candidate_bound", "path": path})
        status = "FOUND" if len(candidates) == 1 and not any(item.get("kind") == "owner_span_parser" for item in omissions) else (
            "AMBIGUOUS" if len(candidates) > 1 else "PARTIAL" if omissions else "NOT_FOUND"
        )
        for row in candidates:
            row.pop("_rank", None)
        return {
            "candidates": candidates,
            "omissions": omissions,
            "coverage": {
                "status": status, "read_bytes": read_bytes, "total_bytes": total_bytes,
                "scan_bytes": scan_limit, "scan_bound": scan_limit < total_bytes,
                "context_bytes": context_limit, "max_hits": MAX_HITS,
                "max_read_bytes": MAX_ACQUISITION_BYTES,
            },
            "identity": {"mode": "override" if raw_override is not None else "root",
                         "content_digest": full_digest, "bytes": total_bytes},
        }
    except OSError as error:
        omissions.append({"kind": "source_unavailable", "path": path, "reason": type(error).__name__})
        return {
            "candidates": [], "omissions": omissions,
            "coverage": {"status": "UNAVAILABLE", "read_bytes": read_bytes, "total_bytes": total_bytes},
            "identity": {"mode": "override" if raw_override is not None else "root", "content_digest": full_digest},
        }
    finally:
        if handle is not None:
            try:
                os.close(handle)
            except OSError:
                pass


__all__ = [
    "MAX_ACQUISITION_BYTES", "MAX_CONTEXT_BYTES", "MAX_HITS", "MAX_SCAN_BYTES", "SCAN_CHUNK_BYTES",
    "acquire_owner_spans",
]
