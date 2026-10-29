"""Revision-bound source pages retained independently of the F4 exchange ring."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from kernel_read_working_set import ReadWorkingSet, evictfix_enabled, source_store_mode
from kernel_measurement import digest


_PAGE_HEADER = re.compile(
    r'^\[read page\]\s+path=(.*?)\s+revision=(sha256:[0-9a-f]{64})\s+'
    r'lines=(\d+)-(\d+)/(\d+)\s+chars=(\d+)\s+'
    r'truncated=(true|false)\s+decoded_with_replacement=(true|false)$', re.IGNORECASE)
_MAX_SPANS = 256
_MAX_BYTES = 2_000_000
_MAX_RECEIPTS = 1024


def _sha(text: str) -> str:
    return 'sha256:' + hashlib.sha256(text.encode('utf-8')).hexdigest()


def _normalized_path(value: object) -> str:
    path = str(value or '').replace('\\', '/')
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or '..' in parsed.parts:
        return ''
    normalized = parsed.as_posix()
    return '' if normalized == '.' else normalized


def _tool_name(call: Mapping[str, object]) -> str:
    function = call.get('function')
    return str(function.get('name', '')) if isinstance(function, Mapping) else ''


def _function_name(value: object) -> str:
    return str(value or '').strip().rsplit('.', 1)[-1]


def _tool_arguments(call: Mapping[str, object]) -> dict[str, object]:
    function = call.get('function')
    raw = function.get('arguments') if isinstance(function, Mapping) else None
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _call_role(arguments: Mapping[str, object]) -> tuple[str, str, str]:
    role_value = str(arguments.get('role', arguments.get('owner_role', ''))).casefold()
    caller = arguments.get('required_caller', arguments.get('caller'))
    why = str(arguments.get('why', '')).casefold()
    if role_value in {'required_caller', 'caller'} or caller or ('required' in why and 'caller' in why):
        return 'required_caller', 'required_caller', str(caller or '')
    owner_symbol = str(arguments.get('symbol', arguments.get('owner_symbol', ''))).strip()
    if role_value in {'owner', 'requested_owner', 'current'} or owner_symbol:
        return 'requested_owner', 'requested_owner', owner_symbol
    return 'observed', 'source', owner_symbol


class SourceEvidenceStore:
    """Keep delivered read pages by source identity and select them through ReadWorkingSet."""

    def __init__(self, emit, *, owner_symbol: str = '',
                 required_caller_path: str = '', required_caller_symbol: str = '') -> None:
        self.emit = emit
        self.initial_owner_symbol = str(owner_symbol or '')
        self.initial_caller_path = _normalized_path(required_caller_path)
        self.initial_caller_symbol = str(required_caller_symbol or '')
        self.pages: dict[str, dict[str, object]] = {}
        self.evidence: dict[tuple[str, str, int, int, str, str], dict[str, object]] = {}
        self.working_set = ReadWorkingSet(
            8, lambda *_args, **_kwargs: None,
            max_bytes=_MAX_BYTES, max_tokens=500_000)
        self._sequence = 0
        self._focus_key: tuple[str, str, str, str] | None = None
        self._selection_key: tuple[object, ...] | None = None
        self._selection_cache: tuple[list[dict[str, object]], dict[str, object]] | None = None
        self.blobs: dict[str, bytes] = {}

    def _invalidate_selection(self) -> None:
        self._selection_key = None
        self._selection_cache = None

    @staticmethod
    def _reference(path: str, revision: str, start: int, end: int,
                   content_hash: str) -> str:
        return 'source-page:' + digest({
            'path': path, 'revision': revision, 'start_line': start,
            'end_line': end, 'content_hash': content_hash,
        })

    def _parse_pair(self, pair: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
        assistant, result = pair
        receipt_id = str(result.get('tool_call_id', '')).strip()
        calls = assistant.get('tool_calls')
        if not receipt_id or not isinstance(calls, Sequence) or isinstance(calls, (str, bytes, bytearray)):
            return None
        call = next((item for item in calls if isinstance(item, Mapping)
                     and str(item.get('id', '')) == receipt_id), None)
        if not isinstance(call, Mapping) or _tool_name(call) != 'read':
            return None
        raw = result.get('content')
        if not isinstance(raw, str):
            return None
        header, first_sep, tail = raw.partition('\n')
        _notice, second_sep, content = tail.partition('\n')
        match = _PAGE_HEADER.fullmatch(header)
        if not first_sep or not second_sep:
            return None
        if match is not None:
            path = _normalized_path(match.group(1))
            revision = match.group(2).lower()
            start, end, total, char_count = (int(match.group(index)) for index in range(3, 7))
            truncated = match.group(7).lower() == 'true'
            replaced = match.group(8).lower() == 'true'
            byte_count = None
        else:
            if not evictfix_enabled():
                return None
            from kernel_request_source_window import _READ_PAGE_HEADER

            production = _READ_PAGE_HEADER.fullmatch(header.removesuffix('\r'))
            if production is None:
                return None
            fields = production.groupdict()
            path = _normalized_path(fields['path'])
            revision = fields['revision'].lower()
            start, end, total, char_count = (int(fields[name]) for name in
                                             ('start', 'end', 'total', 'chars'))
            if (start != int(fields['effective_start'])
                    or end != int(fields['effective_end'])
                    or int(fields['start_column']) != 0):
                return None
            truncated = fields['truncated'].lower() == 'true'
            replaced = fields['replaced'].lower() == 'true'
            byte_count = int(fields['bytes'])
        page_content = content[:char_count]
        if (not path or start < 1 or end < start or total < end
                or len(page_content) != char_count
                or (byte_count is None and len(content) != char_count)
                or (byte_count is not None
                    and len(page_content.encode('utf-8')) != byte_count)):
            return None
        content = page_content
        arguments = _tool_arguments(call)
        owner_role, evidence_kind, owner_symbol = _call_role(arguments)
        column = arguments.get('column', 0)
        start_column = column if type(column) is int and column >= 0 else 0
        content_hash = _sha(content)
        receipt = {
            'receipt_id': receipt_id, 'path': path, 'revision': revision,
            'start_line': start, 'end_line': end, 'content_hash': content_hash,
        }
        self._sequence += 1
        return {
            'reference': self._reference(path, revision, start, end, content_hash),
            'path': path, 'revision': revision, 'freshness': 'current',
            'start_line': start, 'start_column': start_column,
            'selected_end_line': end, 'total_lines': total, 'max_lines': end - start + 1,
            'max_chars': len(content), 'byte_length': len(content.encode('utf-8')),
            'token_count': (len(content) + 3) // 4, 'content': content,
            'content_hash': content_hash, 'receipt_id': receipt_id,
            'receipt_ids': (receipt_id,), 'receipt_digests': (receipt,),
            'owner_role': owner_role, 'owner_roles': (owner_role,),
            'evidence_kind': evidence_kind, 'evidence_kinds': (evidence_kind,),
            'owner_symbol': owner_symbol,
            'required_caller': str(arguments.get('required_caller', arguments.get('caller', ''))),
            'source_store': True, 'complete_lines': start_column == 0,
            'truncated': truncated, 'decoded_with_replacement': replaced,
            'recency': self._sequence,
        }

    @staticmethod
    def _line_parts(page: Mapping[str, object]) -> list[str] | None:
        if page.get('start_column', 0) != 0 or not page.get('complete_lines'):
            return None
        start, end = int(page['start_line']), int(page['selected_end_line'])
        content = page.get('content')
        if not isinstance(content, str):
            return None
        lines = content.splitlines(keepends=True)
        if len(lines) != end - start + 1:
            return None
        if page.get('truncated') and end < int(page.get('total_lines', end)) \
                and content and not content.endswith(('\n', '\r')):
            return None
        return lines

    @classmethod
    def _merge(cls, left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object] | None:
        if (left.get('path') != right.get('path') or left.get('revision') != right.get('revision')
                or left.get('start_column', 0) != right.get('start_column', 0)):
            return None
        left_start, left_end = int(left['start_line']), int(left['selected_end_line'])
        right_start, right_end = int(right['start_line']), int(right['selected_end_line'])
        if right_start > left_end or left_start > right_end:
            return None
        left_lines, right_lines = cls._line_parts(left), cls._line_parts(right)
        if left_lines is None or right_lines is None:
            return None
        start, end = min(left_start, right_start), max(left_end, right_end)
        merged_lines: dict[int, str] = {}
        for page, lines in ((left, left_lines), (right, right_lines)):
            page_start = int(page['start_line'])
            for offset, line in enumerate(lines):
                number = page_start + offset
                if number in merged_lines and merged_lines[number] != line:
                    return None
                merged_lines[number] = line
        if len(merged_lines) != end - start + 1:
            return None
        content = ''.join(merged_lines[number] for number in range(start, end + 1))
        receipt_digests = tuple({
            str(item['receipt_id']): dict(item)
            for page in (left, right)
            for item in page.get('receipt_digests', ()) if isinstance(item, Mapping)
        }.values())
        receipt_ids = tuple(dict.fromkeys(
            str(item) for page in (left, right)
            for item in page.get('receipt_ids', ()) if str(item)
        ))
        owner_roles = tuple(dict.fromkeys(
            str(item) for page in (left, right)
            for item in page.get('owner_roles', (page.get('owner_role', 'observed'),)) if str(item)
        ))
        preferred = next((role for role in ('required_caller', 'requested_owner', 'recently_edited', 'observed')
                          if role in owner_roles), 'observed')
        row = {**dict(right), 'start_line': start, 'selected_end_line': end,
               'max_lines': end - start + 1, 'content': content,
               'max_chars': len(content), 'byte_length': len(content.encode('utf-8')),
               'token_count': (len(content) + 3) // 4,
               'content_hash': _sha(content), 'receipt_ids': receipt_ids,
               'receipt_id': receipt_ids[-1] if receipt_ids else '',
               'receipt_digests': receipt_digests, 'owner_role': preferred,
               'owner_roles': owner_roles,
               'evidence_kinds': tuple(dict.fromkeys(
                   str(item) for page in (left, right)
                   for item in page.get('evidence_kinds', (page.get('evidence_kind', 'source'),))
                   if str(item))),
               'evidence_kind': preferred if preferred in {'required_caller', 'requested_owner'} else 'source',
               'complete_lines': True,
               'decoded_with_replacement': (
                   bool(left.get('decoded_with_replacement'))
                   or bool(right.get('decoded_with_replacement'))),
               'reference': cls._reference(str(right['path']), str(right['revision']),
                                           start, end, _sha(content)),
               'recency': max(int(left.get('recency', 0)), int(right.get('recency', 0)))}
        return row

    def _forget_page(self, reference: str, reason: str) -> None:
        self.pages.pop(reference, None)
        self.working_set.forget_reference(reference, reason)

    def _forget_receipt(self, key: tuple[str, str, int, int, str, str]) -> None:
        evidence = self.evidence.pop(key, None)
        if evidence is None:
            return
        receipt_id = str(evidence['receipt_id'])
        for reference, page in self.pages.items():
            receipt_ids = tuple(item for item in page.get('receipt_ids', ())
                                if str(item) != receipt_id)
            if receipt_ids == tuple(page.get('receipt_ids', ())):
                continue
            receipt_digests = tuple(item for item in page.get('receipt_digests', ())
                                    if not isinstance(item, Mapping)
                                    or str(item.get('receipt_id', '')) != receipt_id)
            page['receipt_ids'] = receipt_ids
            page['receipt_id'] = receipt_ids[-1] if receipt_ids else ''
            page['receipt_digests'] = receipt_digests
            self.working_set.set_receipts(reference, receipt_ids, receipt_digests)
        self.emit('source_store_evicted', path=evidence['path'], revision=evidence['revision'],
                  start_line=evidence['start_line'], end_line=evidence['end_line'],
                  receipt_id=receipt_id, reason='receipt_index_capacity')

    def _enforce_capacity(self) -> None:
        while len(self.evidence) > _MAX_RECEIPTS:
            oldest_key = min(self.evidence, key=lambda key: int(self.evidence[key].get('recency', 0)))
            self._forget_receipt(oldest_key)
        while (len(self.pages) > _MAX_SPANS
               or sum(int(row.get('byte_length', 0)) for row in self.pages.values()) > _MAX_BYTES):
            recent_observed = self.working_set.recent_observed_references(
                list(self.pages.values()))
            candidates = sorted(self.pages.values(), key=lambda row: (
                self.working_set.eviction_priority(row, recent_observed),
                int(row.get('recency', 0)), str(row['reference'])))
            oldest = candidates[0]
            self._forget_page(str(oldest['reference']), 'source_store_capacity')
            receipt_ids = tuple(oldest.get('receipt_ids', ()))
            for key in list(self.evidence):
                if key[-1] in receipt_ids:
                    self.evidence.pop(key, None)
            self.emit('source_store_evicted', path=oldest['path'], revision=oldest['revision'],
                      start_line=oldest['start_line'], end_line=oldest['selected_end_line'],
                      reason='source_store_capacity')

    def remember_exchange(self, pair: Sequence[Mapping[str, object]]) -> bool:
        """Remember an exact successful read result admitted by the real F4 ledger."""
        if (not isinstance(pair, Sequence) or isinstance(pair, (str, bytes, bytearray))
                or len(pair) != 2 or not all(isinstance(item, Mapping) for item in pair)):
            return False
        page = self._parse_pair(pair)
        if page is None:
            return False
        path, revision = str(page['path']), str(page['revision'])
        focus = self._focus_key
        focus_path = focus[0] if focus is not None else ''
        focus_owner_symbol = focus[1] if focus is not None else self.initial_owner_symbol
        focus_caller_path = focus[2] if focus is not None else self.initial_caller_path
        focus_caller_symbol = focus[3] if focus is not None else self.initial_caller_symbol
        owner_name = _function_name(focus_owner_symbol)
        caller_name = _function_name(focus_caller_symbol)
        owner_match = bool(owner_name and (not focus_path or path == focus_path) and re.search(
            rf'(?m)^\s*(?:async\s+)?(?:def|class)\s+'
            rf'{re.escape(owner_name)}\b', str(page['content'])))
        caller_match = bool(focus_caller_path
                            and page['path'] == focus_caller_path
                            and (not caller_name or re.search(
                                rf'(?m)^\s*(?:async\s+)?(?:def|class)\s+'
                                rf'{re.escape(caller_name)}\b', str(page['content']))))
        if owner_match or caller_match:
            role = 'required_caller' if caller_match else 'requested_owner'
            page['owner_role'] = role
            page['owner_roles'] = (role,)
            page['evidence_kind'] = role
            page['evidence_kinds'] = (role,)
        for old in list(self.pages.values()):
            if old['path'] == path and old['revision'] != revision:
                self._forget_page(str(old['reference']), 'revision_superseded')
        for key, old in list(self.evidence.items()):
            if old['path'] == path and old['revision'] != revision:
                self.evidence.pop(key, None)
        key = (path, revision, int(page['start_line']), int(page['selected_end_line']),
               str(page['content_hash']), str(page['receipt_id']))
        self.evidence[key] = {field: page[field] for field in (
            'path', 'revision', 'start_line', 'selected_end_line', 'content_hash',
            'receipt_id', 'recency')}
        candidate = page
        merged_refs: list[str] = []
        changed = True
        while changed:
            changed = False
            for old in list(self.pages.values()):
                if str(old['reference']) in merged_refs:
                    continue
                merged = self._merge(old, candidate)
                if merged is None:
                    continue
                merged_refs.append(str(old['reference']))
                candidate = merged
                changed = True
                break
        for reference in merged_refs:
            self._forget_page(reference, 'overlapping_source_merge')
        self.pages[str(candidate['reference'])] = candidate
        self.working_set.remember(candidate)
        self._enforce_capacity()
        self._invalidate_selection()
        self.emit('source_store_page_remembered', path=path, revision=revision,
                  start_line=candidate['start_line'], end_line=candidate['selected_end_line'],
                  content_hash=page['content_hash'], receipt_id=page['receipt_id'],
                  merged_receipt_ids=list(candidate.get('receipt_ids', ())))
        return True

    def invalidate(self, current: Mapping[str, str]) -> None:
        for row in list(self.pages.values()):
            if current.get(str(row['path'])) != row['revision']:
                self._forget_page(str(row['reference']), 'revision_changed_or_missing')
        for key, row in list(self.evidence.items()):
            if current.get(str(row['path'])) != row['revision']:
                self.evidence.pop(key, None)
        self._invalidate_selection()

    def _set_focus(self, path: str, owner_symbol: str, caller_path: str,
                   caller_symbol: str) -> None:
        key = (path, owner_symbol, caller_path, caller_symbol)
        if self._focus_key is not None and self._focus_key != key:
            for row in self.pages.values():
                roles = tuple(role for role in row.get('owner_roles', ())
                              if role not in {'requested_owner', 'required_caller'})
                row['owner_roles'] = roles or ('observed',)
                row['owner_role'] = roles[0] if roles else 'observed'
                row['evidence_kinds'] = tuple(kind for kind in row.get('evidence_kinds', ())
                                              if kind not in {'requested_owner', 'required_caller'}) or ('source',)
                row['evidence_kind'] = row['evidence_kinds'][0]
                self.working_set.set_owner(str(row['reference']), str(row['owner_role']),
                                           str(row['evidence_kind']))
        self._focus_key = key
        owner_name = _function_name(owner_symbol)
        caller_name = _function_name(caller_symbol)
        for row in self.pages.values():
            matches_owner = bool(owner_name and row['path'] == path and
                                 re.search(rf'(?m)^\s*(?:async\s+)?(?:def|class)\s+'
                                           rf'{re.escape(owner_name)}\b', str(row['content'])))
            matches_caller = bool(caller_path and row['path'] == caller_path
                                  and (not caller_name or re.search(
                                      rf'(?m)^\s*(?:async\s+)?(?:def|class)\s+'
                                      rf'{re.escape(caller_name)}\b', str(row['content']))))
            if matches_owner or matches_caller:
                role = 'required_caller' if matches_caller else 'requested_owner'
                row['owner_role'] = role
                row['owner_roles'] = tuple(dict.fromkeys((*row.get('owner_roles', ()), role)))
                row['evidence_kind'] = role
                row['evidence_kinds'] = tuple(dict.fromkeys((*row.get('evidence_kinds', ()), role)))
                self.working_set.set_owner(str(row['reference']), role, role)

    def recall(self, reference: str, current: Mapping[str, str], *, start_line: int = 1,
               max_lines: int = 120, max_chars: int = 12000, **_kwargs) -> dict[str, object]:
        row = self.pages.get(reference)
        if row is None or current.get(str(row['path'])) != row['revision']:
            raise KeyError('source page is no longer current')
        content = str(row['content'])
        return {**copy.deepcopy(row), 'content': content,
                'start_line': int(row['start_line']),
                'selected_end_line': int(row['selected_end_line']),
                'freshness': 'current'}

    def select(self, current: Mapping[str, str], working_path: str,
               working_content: str | None, *, owner_symbol: str = '',
               required_caller_path: str = '', required_caller_symbol: str = '') -> tuple[list[dict[str, object]], dict[str, object]]:
        self.invalidate(current)
        self._set_focus(working_path, owner_symbol, required_caller_path, required_caller_symbol)
        key = (tuple(sorted(current.items())), working_path, digest(working_content or ''),
               owner_symbol, required_caller_path, required_caller_symbol,
               tuple(sorted(self.pages)))
        if key == self._selection_key and self._selection_cache is not None:
            return copy.deepcopy(self._selection_cache)
        selected, state = self.working_set.select(self, dict(current), working_path, working_content)
        self._selection_key = key
        self._selection_cache = (copy.deepcopy(selected), copy.deepcopy(state))
        return selected, state

    def state(self) -> dict[str, object]:
        return {
            'effective': True,
            'spans_retained': len(self.pages),
            'bytes': sum(int(row.get('byte_length', 0)) for row in self.pages.values()),
        }


__all__ = ['SourceEvidenceStore', 'source_store_mode']
