"""Bounded revision-bound read spans, separate from immutable archived bytes."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from math import ceil

from kernel_measurement import digest
from kernel_evictfix_mode import enabled as _evictfix_enabled


SOURCE_STORE_ENV = 'STATEFRAME_SOURCE_STORE'
_COMPACT_LEDGER_ROWS = 12
_COMPACT_LEDGER_BYTES = 2048


def evictfix_enabled() -> bool:
    """Return whether the explicit EVICTFIX opt-in is active."""
    return _evictfix_enabled()


def _read_ranges(row: Mapping[str, object]) -> list[list[int]]:
    lines = row.get('ranges')
    if not isinstance(lines, (list, tuple)):
        lines = [row.get('lines')]
    ranges: list[list[int]] = []
    for item in lines:
        if isinstance(item, Mapping):
            start, end = item.get('start'), item.get('end')
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            continue
        if type(start) is int and type(end) is int and 0 < start <= end:
            ranges.append([start, end])
    return ranges


def compact_read_ledger(read_rows: Sequence[Mapping[str, object]],
                        revisions: Mapping[str, str]) -> list[dict[str, object]]:
    """Newest-first read identity only; at most 12 rows and 2 KiB of JSON.

    Changed revisions are excluded. Unknown paths keep their observed revision,
    so a partial current-revision map does not erase the read history.
    """
    if not evictfix_enabled():
        return []
    retained: list[dict[str, object]] = []
    for source in reversed(read_rows):
        if not isinstance(source, Mapping):
            continue
        path, revision = source.get('path'), source.get('revision')
        ids = source.get('call_ids')
        call_id = source.get('call_id')
        if not isinstance(call_id, str) and isinstance(ids, (list, tuple)) and ids:
            call_id = ids[0]
        ranges = _read_ranges(source)
        if (not isinstance(path, str) or not path
                or not isinstance(revision, str) or not revision
                or not isinstance(call_id, str) or not call_id or not ranges):
            continue
        if path in revisions and revisions[path] != revision:
            continue
        row: dict[str, object] = {
            'path': path, 'ranges': ranges, 'revision': revision, 'call_id': call_id,
        }
        candidate = [*retained, row]
        size = len(json.dumps(candidate, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        if size <= _COMPACT_LEDGER_BYTES:
            retained.append(row)
        if len(retained) >= _COMPACT_LEDGER_ROWS:
            break
    return retained


def source_store_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(SOURCE_STORE_ENV, 'off')).strip().lower()
    if value not in {'on', 'off'}:
        raise ValueError(f'{SOURCE_STORE_ENV} must be on or off')
    return value


def source_store_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return source_store_mode(environ) == 'on'


def _journal_safe_fields(payload):
    """Emit boundary (10-05 L109): rename journal-RESERVED keys; see kernel_continuation_journal."""
    from kernel_continuation_journal import journal_safe_fields

    return journal_safe_fields(payload)


def capacity() -> int:
    value = os.environ.get('KERNEL_RETAINED_WORKING_SET', '0')
    if value not in {'0', '2', '3', '4', '5', '6', '7', '8'}:
        raise ValueError('KERNEL_RETAINED_WORKING_SET must be 0 or an integer from 2 through 8')
    return int(value)


def _bound(env_name: str, default: int, minimum: int, maximum: int) -> int:
    value = os.environ.get(env_name, str(default))
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        raise ValueError(f'{env_name} must be an integer from {minimum} through {maximum}') from None


def _source_order_content(span: Mapping[str, object]) -> str:
    """Canonical source metadata, excluding recency and per-run receipt identity."""
    operational = {'_content', 'recency', 'span_id', 'reference',
                   'receipt_id', 'receipt_ids', 'receipt_digests'}
    return json.dumps({key: value for key, value in span.items() if key not in operational},
                      sort_keys=True, ensure_ascii=False)


class ReadWorkingSet:
    def __init__(self, limit: int, emit, *, max_bytes: int | None = None,
                 max_tokens: int | None = None):
        if type(limit) is not int or not 2 <= limit <= 8:
            raise ValueError('read working set requires integer capacity 2 through 8')
        self.limit, self.emit = limit, emit
        self.max_bytes = (max_bytes if max_bytes is not None else
                          _bound('KERNEL_RETAINED_WORKING_SET_BYTES', limit * 12000, 1024, 2_000_000))
        self.max_tokens = (max_tokens if max_tokens is not None else
                           _bound('KERNEL_RETAINED_WORKING_SET_TOKENS', limit * 3000, 256, 500_000))
        if type(self.max_bytes) is not int or self.max_bytes < 1:
            raise ValueError('read working set byte bound must be a positive integer')
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError('read working set token bound must be a positive integer')
        self.spans: dict[str, dict] = {}
        self.removed: list[dict] = []
        self.edit_focused: set[str] = set()
        self._recency: list[str] = []
        self._sequence = 0

    @staticmethod
    def _token_count(span: Mapping[str, object]) -> int:
        value = span.get('token_count')
        if isinstance(value, int) and value >= 0:
            return value
        return ceil(int(span.get('max_chars', 0) or 0) / 4)

    def _usage(self) -> tuple[int, int]:
        return (sum(int(span.get('byte_length', span.get('max_chars', 0)) or 0)
                    for span in self.spans.values()),
                sum(self._token_count(span) for span in self.spans.values()))

    def _touch(self, key: str) -> None:
        self._recency = [item for item in self._recency if item != key]
        self._recency.append(key)

    def _protected(self, span: Mapping[str, object]) -> bool:
        owners = {str(span.get('owner_role', '')).casefold()}
        owners.update(str(item).casefold() for item in span.get('owner_roles', ()))
        evidence = {str(span.get('evidence_kind', '')).casefold()}
        evidence.update(str(item).casefold() for item in span.get('evidence_kinds', ()))
        return (bool(owners & {'current', 'requested_owner', 'required_caller',
                               'recently_edited', 'declaration', 'import'})
                or bool(evidence & {'requested_owner', 'required_caller',
                                    'declaration', 'import', 'imports'})
                or bool(span.get('edit_focused')))

    @staticmethod
    def _observed(span: Mapping[str, object]) -> bool:
        roles = {str(span.get('owner_role', '')).casefold()}
        roles.update(str(role).casefold() for role in span.get('owner_roles', ()))
        return 'observed' in roles

    def recent_observed_references(self, rows: Sequence[Mapping[str, object]]) -> set[str]:
        if not evictfix_enabled():
            return set()
        observed = sorted(
            (row for row in rows if self._observed(row)),
            key=lambda row: (int(row.get('recency', 0)), str(row.get('reference', ''))),
        )
        return {str(row.get('reference', '')) for row in observed[-3:]}

    def eviction_priority(self, span: Mapping[str, object],
                          recent_observed: set[str]) -> int:
        if (evictfix_enabled() and self._observed(span)
                and str(span.get('reference', '')) in recent_observed):
            return 2
        return 1 if self._protected(span) else 0

    def _eviction_key(self, key: str) -> tuple[int, int, str]:
        span = self.spans[key]
        recent_observed = self.recent_observed_references(list(self.spans.values()))
        priority = self.eviction_priority(span, recent_observed)
        return (priority,
                 self._recency.index(key) if key in self._recency else -1,
                 key)

    def _remove(self, key: str, reason: str) -> None:
        span = self.spans.pop(key)
        self.edit_focused.discard(key)
        self._recency = [item for item in self._recency if item != key]
        notice = {key: span[key] for key in ('span_id', 'path', 'revision', 'start_line', 'start_column')}
        notice['reason'] = reason
        self.removed = [*self.removed[-7:], notice]
        self.emit('read_working_set_removed', **_journal_safe_fields(notice))

    def invalidate(self, current: dict) -> None:
        for key, span in list(self.spans.items()):
            if current.get(span['path']) != span['revision']:
                self._remove(key, 'revision_changed_or_missing')

    def focus_edit(self, path: str, revision: str) -> None:
        """Keep a current applied edit visible beside the working source."""
        focused = next((key for key, span in self.spans.items()
                        if span['path'] == path and span['revision'] == revision), None)
        if focused is not None:
            self.edit_focused.add(focused)
            self.spans[focused]['owner_role'] = 'recently_edited'
            self.spans[focused]['edit_focused'] = True
            self._touch(focused)

    def retain_rejected_owner(self, page: dict) -> None:
        """Keep attempted owner's current bytes after a rejected replacement."""
        if not isinstance(page, Mapping):
            return
        candidate = dict(page)
        attempted = candidate.get('attempted_owner')
        if isinstance(attempted, Mapping):
            # Rejection packets can carry both the attempted owner and a
            # threatened replacement.  Only the owner projection is admissible
            # for retention; the threatened definition is never a fallback.
            candidate.update(dict(attempted))
        for key in ('attempted_owner_content', 'current_content'):
            if isinstance(page.get(key), str):
                candidate['content'] = page[key]
                break
        if not isinstance(candidate.get('content'), str) or not candidate.get('content'):
            return
        if str(candidate.get('owner_role', '')).casefold() in {'threatened', 'definition', 'replacement'} \
                and not isinstance(attempted, Mapping) \
                and not any(isinstance(page.get(key), str) for key in ('attempted_owner_content', 'current_content')):
            return
        candidate['owner_role'] = 'current'
        candidate['rejected_replacement_owner'] = True
        candidate['freshness'] = 'current'
        self.remember(candidate)
        key = next((key for key, span in self.spans.items()
                    if span.get('path') == candidate.get('path')
                    and span.get('revision') == candidate.get('revision')
                    and span.get('rejected_replacement_owner')), None)
        if key is not None:
            self.edit_focused.add(key)
            self.spans[key]['edit_focused'] = True
            self._touch(key)

    remember_attempted_owner = retain_rejected_owner

    def remember(self, page: dict) -> None:
        if page['freshness'] != 'current' or not page['content']:
            return
        span = {key: page[key] for key in ('reference', 'path', 'revision', 'start_line')}
        end_line = page.get('selected_end_line', page['start_line'])
        span.update(start_column=page.get('start_column', 0),
                    max_lines=end_line - page['start_line'] + 1,
                    max_chars=page.get('byte_end', 0) - page.get('byte_start', 0)
                    if page.get('byte_end') is not None and page.get('byte_start') is not None
                    and page.get('byte_end', 0) >= page.get('byte_start', 0)
                    else len(page['content']),
                    token_count=page.get('token_count', ceil(len(page['content']) / 4)),
                    byte_length=page.get(
                        'byte_length', len(page['content'].encode(str(page.get('decoding', 'utf-8')), errors='replace'))),
                    content_hash=digest(page['content']),
                    byte_start=page.get('byte_start'), byte_end=page.get('byte_end'),
                    decoding=page.get('decoding', 'utf-8'),
                    presentation_version=page.get('presentation_version', 'read-page/1'),
                    owner_role=page.get('owner_role', 'observed'),
                    evidence_kind=page.get('evidence_kind', page.get('role', 'source')),
                    **{key: page[key] for key in (
                        'source_store', 'receipt_id', 'receipt_ids', 'receipt_digests',
                        'owner_symbol', 'required_caller', 'complete_lines', 'truncated',
                        'decoded_with_replacement',
                    ) if key in page},
                    owner_roles=tuple(dict.fromkeys(
                        str(item) for item in page.get('owner_roles', ())
                        if str(item).strip()
                    )) or (str(page.get('owner_role', 'observed')),),
                    evidence_kinds=tuple(dict.fromkeys(
                        str(item) for item in page.get('evidence_kinds', ())
                        if str(item).strip()
                    )) or (str(page.get('evidence_kind', page.get('role', 'source'))),),
                    edit_focused=bool(page.get('edit_focused', False)),
                    rejected_replacement_owner=bool(page.get('rejected_replacement_owner', False)),
                    _content=page['content'])
        identity = {key: value for key, value in span.items() if key != 'reference'}
        key = 'read-span:' + digest(identity)
        span['span_id'] = key
        # Exact byte ranges on one revision coalesce only when their overlap is
        # verifiable.  Unpositioned legacy pages remain distinct rather than
        # claiming a digest for bytes that were never observed.
        byte_start, byte_end = span.get('byte_start'), span.get('byte_end')
        if isinstance(byte_start, int) and isinstance(byte_end, int):
            for old_key, old in list(self.spans.items()):
                if (old.get('path') != span['path'] or old.get('revision') != span['revision']
                        or old.get('decoding') != span['decoding']
                        or not isinstance(old.get('byte_start'), int)
                        or not isinstance(old.get('byte_end'), int)
                        or byte_start >= old['byte_end'] or old['byte_start'] >= byte_end):
                    continue
                merged_start = min(byte_start, old['byte_start'])
                merged_end = max(byte_end, old['byte_end'])
                old_bytes = str(old.get('_content', '')).encode(str(old.get('decoding', 'utf-8')), errors='replace')
                new_bytes = str(span.get('_content', '')).encode(str(span.get('decoding', 'utf-8')), errors='replace')
                if (len(old_bytes) != old['byte_end'] - old['byte_start']
                        or len(new_bytes) != byte_end - byte_start):
                    # A clipped/newline-replaced page is not an exact span and
                    # cannot safely be used to synthesize a larger one.
                    continue
                old_offset = old['byte_start'] - merged_start
                new_offset = byte_start - merged_start
                merged = bytearray(merged_end - merged_start)
                occupied = bytearray(merged_end - merged_start)
                conflict = False
                for offset, body in ((old_offset, old_bytes), (new_offset, new_bytes)):
                    for index, value in enumerate(body[:merged_end - merged_start - offset]):
                        position = offset + index
                        if occupied[position] and merged[position] != value:
                            conflict = True
                            break
                        merged[position] = value
                        occupied[position] = 1
                    if conflict:
                        break
                if conflict:
                    continue
                merged_owner_roles = tuple(dict.fromkeys(
                    (*old.get('owner_roles', (old.get('owner_role', 'observed'),)),
                     *span.get('owner_roles', (span.get('owner_role', 'observed'),)))
                ))
                preferred_owner = next(
                    (role for role in ('recently_edited', 'current', 'declaration', 'import', 'observed')
                     if role in merged_owner_roles),
                    merged_owner_roles[0] if merged_owner_roles else 'observed',
                )
                merged_evidence_kinds = tuple(dict.fromkeys(
                    (*old.get('evidence_kinds', (old.get('evidence_kind', 'source'),)),
                     *span.get('evidence_kinds', (span.get('evidence_kind', 'source'),)))
                ))
                try:
                    merged_text = bytes(merged).decode(str(span['decoding']))
                except UnicodeDecodeError:
                    merged_text = bytes(merged).decode(str(span['decoding']), errors='replace')
                span.update(byte_start=merged_start, byte_end=merged_end,
                            max_chars=merged_end - merged_start,
                            max_lines=max(span['max_lines'], old.get('max_lines', 0)),
                            token_count=ceil(len(merged_text) / 4),
                            byte_length=merged_end - merged_start,
                            content_hash=digest(merged_text), _content=merged_text,
                            start_line=min(span['start_line'], old['start_line']),
                            owner_role=preferred_owner,
                            owner_roles=merged_owner_roles,
                            evidence_kinds=merged_evidence_kinds,
                            evidence_kind=(
                                'declaration' if 'declaration' in merged_evidence_kinds
                                else 'import' if 'import' in merged_evidence_kinds
                                else merged_evidence_kinds[0] if merged_evidence_kinds else 'source'
                            ))
                key = old_key
                self.spans.pop(old_key, None)
                self._recency = [item for item in self._recency if item != old_key]
                # Continue across the snapshot: one new span can bridge several
                # previously retained overlaps, and must not leave an overlap
                # behind merely because the first match was removed.
                byte_start, byte_end = merged_start, merged_end
        # Re-observing the identical range refreshes recency without consuming capacity.
        self.spans.pop(key, None)
        self.spans[key] = span
        self._sequence += 1
        span['recency'] = self._sequence
        self._touch(key)
        self.emit('read_working_set_remembered', span=dict(span))
        while (len(self.spans) > self.limit
               or self._usage()[0] > self.max_bytes
               or self._usage()[1] > self.max_tokens):
            self._remove(min(self.spans, key=self._eviction_key), 'capacity_eviction')

    def select(self, archive, current: dict, working_path: str, working_content: str | None):
        self.invalidate(current)
        selected, active = [], []
        # Restored spans may share recency; mapping insertion order cannot decide
        # which exact source page is delivered first.
        for span in sorted(self.spans.values(), key=lambda item: (
                -int(item.get('recency', 0)), str(item.get('path', '')),
                str(item.get('revision', '')), int(item.get('start_line', 0)),
                int(item.get('start_column', 0)), int(item.get('max_lines', 0)),
                int(item.get('max_chars', 0)), str(item.get('content_hash', '')),
                _source_order_content(item))):
            try:
                page = archive.recall(span['reference'], current,
                    start_line=span['start_line'], start_column=span['start_column'],
                    max_lines=span['max_lines'], max_chars=span['max_chars'], continuation=True)
                if not isinstance(page, Mapping) or not isinstance(page.get('content'), str):
                    raise KeyError('stale read restart handle returned no content')
                page_digest = digest(page['content'])
            except (KeyError, OSError, TypeError, ValueError):
                self._remove(span['span_id'], 'stale_restart_handle')
                continue
            if page_digest != span['content_hash']:
                retained_content = span.get('_content')
                if not isinstance(retained_content, str) or digest(retained_content) != span['content_hash']:
                    raise ValueError('read working set content no longer matches its bound span')
                page = {**page, 'content': retained_content}
            self._touch(span['span_id'])
            archived = getattr(archive, 'blobs', {}).get(span['revision'])
            covered = (span['span_id'] not in self.edit_focused
                       and span['path'] == working_path and working_content is not None
                       and isinstance(archived, (bytes, bytearray))
                       and bytes(archived).decode('utf-8', errors='replace') == working_content)
            active.append({**{key: span[key] for key in
                ('span_id', 'path', 'revision', 'start_line', 'start_column')},
                'owner_role': span.get('owner_role'),
                'owner_roles': tuple(span.get('owner_roles', ())),
                'evidence_kind': span.get('evidence_kind'),
                'evidence_kinds': tuple(span.get('evidence_kinds', ())),
                'delivery': 'working_source' if covered else 'retained_source'})
            if not covered:
                selected.append({**page, 'span_id': span['span_id'], 'requested_delivery': True})
        used_bytes, used_tokens = self._usage()
        return selected, {'capacity': self.limit, 'max_bytes': self.max_bytes,
                          'max_tokens': self.max_tokens, 'used_bytes': used_bytes,
                          'used_tokens': used_tokens, 'active': active,
                          'recent_removals': list(self.removed)}

    def forget_reference(self, reference: str, reason: str = 'source_superseded') -> None:
        """Remove spans backed by one source-store reference through normal eviction accounting."""
        for key, span in list(self.spans.items()):
            if span.get('reference') == reference:
                self._remove(key, reason)

    def set_owner(self, reference: str, owner_role: str, evidence_kind: str) -> None:
        """Update focus labels without changing the remembered bytes or their identity."""
        for span in self.spans.values():
            if span.get('reference') != reference:
                continue
            span['owner_role'] = owner_role
            span['owner_roles'] = (owner_role,)
            span['evidence_kind'] = evidence_kind
            span['evidence_kinds'] = (evidence_kind,)

    def set_receipts(self, reference: str, receipt_ids: Sequence[str],
                     receipt_digests: Sequence[Mapping[str, object]]) -> None:
        """Keep working-set metadata bounded when old receipt identities age out."""
        for span in self.spans.values():
            if span.get('reference') == reference:
                span['receipt_ids'] = tuple(receipt_ids)
                span['receipt_digests'] = tuple(dict(item) for item in receipt_digests)


def cover_observed_reads(frame: dict, retained: list[dict], *, supersede_historical: bool = False) -> dict:
    """Suppress only an older display copy covered by a required complete current file."""
    import copy
    complete = {(row['path'], row['revision']): row['span_id'] for row in retained
                if row.get('span_id') and row.get('requested_delivery') is True
                and row.get('freshness') == 'current' and row.get('start_line') == 1
                and row.get('start_column', 0) == 0
                and row.get('more_content_available') is False
                and row.get('omitted_chars_in_selection') == 0}
    projected = copy.deepcopy(frame)
    for row in projected.get('earlier_other_file_reads', []):
        key = (row.get('path'), row.get('file_revision'))
        if key in complete and row.get('revision_relevance') == 'same_file_revision':
            row.pop('content', None)
            row['covered_by_retained_span'] = complete[key]
        elif supersede_historical:
            current = next((span for (path, _), span in complete.items() if path == row.get("path")), None)
            if current:
                row.pop("content", None)
                row["superseded_by_retained_span"] = current

    return projected
