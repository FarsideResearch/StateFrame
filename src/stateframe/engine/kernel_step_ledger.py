"""Join each proposal to observed feedback and actual workspace changes."""
from __future__ import annotations

import ast
import copy
import re
from collections.abc import Iterator

from kernel_measurement import canonical_bytes, digest
from kernel_ledger_bindings import BindingHistory
from kernel_memfix_history import SpoolSequence, ledger_records, legacy


def _text_symbol_span(text: str, symbol: str) -> tuple[int, int] | None:
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    pattern = re.compile(r'^(\s*)(?:async\s+)?(?:def|class)\s+'
                        + re.escape(symbol) + r'\b')
    start = next((index for index, line in enumerate(lines)
                  if pattern.match(line)), None)
    if start is None:
        return None
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        stripped = lines[end].strip()
        current_indent = len(lines[end]) - len(lines[end].lstrip())
        if stripped == '[... middle omitted ...]' \
                or (stripped and current_indent <= indent):
            break
        end += 1
    return offsets[start], offsets[end]


def _source_symbol_span(text: str, symbol: str | None, gap_start: int,
                        gap_end: int) -> tuple[int, int] | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        if '[... middle omitted ...]' not in text:
            return _text_symbol_span(text, symbol) if symbol is not None else None
        try:
            tree = ast.parse(text.replace('[... middle omitted ...]', ''))
        except SyntaxError:
            return _text_symbol_span(text, symbol) if symbol is not None else None
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if symbol is not None and node.name != symbol:
            continue
        start_line = min([node.lineno] + [item.lineno for item in node.decorator_list])
        end_line = getattr(node, 'end_lineno', node.lineno)
        candidates.append((offsets[start_line - 1],
                           offsets[min(end_line, len(offsets) - 1)]))
    if symbol is not None and candidates:
        return max(candidates, key=lambda span: span[1] - span[0])
    midpoint = (gap_start + gap_end) // 2
    crossing = [span for span in candidates
                if span[0] < gap_end and span[1] > gap_start]
    if not crossing:
        return None
    return min(crossing, key=lambda span: (not span[0] <= midpoint < span[1],
                                           span[1] - span[0]))


def _bounded_source(text: str, limit: int, symbol: str | None) -> str | None:
    marker = '\n[... middle omitted ...]\n'
    available = limit - len(marker)
    if available <= 0:
        return None
    gap_start = available // 2
    gap_end = len(text) - (available - gap_start)
    span = _source_symbol_span(text, symbol, gap_start, gap_end)
    if span is None:
        return None
    target = text[span[0]:span[1]]
    marker_count = int(span[0] > 0) + int(span[1] < len(text))
    budget = limit - len(target) - marker_count * len(marker)
    if budget < 0:
        return None
    before_limit = min(span[0], budget // 2)
    after_limit = min(len(text) - span[1], budget - before_limit)
    prefix = text[:before_limit]
    suffix = text[len(text) - after_limit:] if after_limit else ''
    result = prefix + (marker if prefix else '') + target
    if suffix:
        result += marker + suffix
    elif span[1] < len(text):
        result += marker
    return result[:limit]


def bounded_feedback(text: str, limit: int, *, symbol: str | None = None) -> str:
    """Keep error summaries at the tail as well as context at the head."""
    if len(text) <= limit:
        return text
    source = _bounded_source(text, limit, symbol)
    if source is not None:
        return source
    marker = '\n[... middle omitted ...]\n'
    available = limit - len(marker)
    return text[:available // 2] + marker + text[-(available - available // 2):]


def _slim_record(record: dict, *, version: int = 3) -> dict:
    """Keep scan fields resident; full historical payloads stay in the spool."""
    scan_fields = {'step', 'tool', 'target', 'arguments_hash', 'action_parameters',
                   'intent', 'result', 'receipt_ids', 'execution_receipt_ids',
                   'behavioral_receipt_ids', 'execution_evidence', 'observation_hash',
                   'observation_omitted_chars', 'changes', 'tool_call_id',
                   'continuation_id', 'continuation_effect_id',
                   'continuation_dispatch_ref', 'active_work_id', 'step_record_id',
                   'legitimate_recovery'}
    slim = {key: copy.deepcopy(value) for key, value in record.items()
            if key in scan_fields}
    before = record.get('before_binding', {})
    after = record.get('after_binding', {})
    slim['full_record_digest'] = digest(record)
    slim['before_binding_digest'] = digest(before)
    slim['after_binding_digest'] = digest(after)
    slim['after_binding_revisions'] = {key: after.get(key) for key in (
        'product_revision', 'test_harness_revision', 'environment_revision',
        *(() if version == 2 else ('artifact', 'revision')))
        if isinstance(after, dict) and key in after}
    if isinstance(after, dict):
        for key in ('product_revision', 'artifact', 'revision'):
            if key in after:
                slim[key] = after[key]
    # Refusal and public-check projections consume the bounded feedback text
    # while scanning; ordinary action feedback is hydrated only when selected.
    if record.get('tool') == 'run_tests' or record.get('result') in {
            'RED', 'FAILED', 'REFUSED', 'REFUSED_PREDISPATCH', 'DENIED_PATH',
            'REJECTED_CONTRACT', 'REJECTED', 'NO_OP', 'ERROR', 'TIMEOUT',
            'REPEAT_SUPPRESSED', 'REFUSED_SERIES'}:
        if 'observation' in record:
            slim['observation'] = record['observation']
    if record.get('tool') == 'run_tests':
        slim['finding_after_binding_present'] = 'after_binding' in record
        if version >= 3:
            from kernel_findings import _revision_from
            slim['finding_revision'] = _revision_from(record.get('after_binding'))
    if 'read_evidence' in record:
        slim['read_evidence'] = {key: copy.deepcopy(value)
                                 for key, value in record['read_evidence'].items()
                                 if key != 'content'}
    return slim


def _product_files(binding: object) -> dict:
    files = binding.get('product_files', {}) if isinstance(binding, dict) else {}
    return files if isinstance(files, dict) else {}


def _boundary_delta(previous: dict, following: dict) -> dict:
    return {path: previous.get(path) for path in previous.keys() | following.keys()
            if previous.get(path) != following.get(path)}


class SlimLedgerRecords(SpoolSequence):
    """Sequence of small records with explicit, bounded full-record hydration."""

    def __init__(self, directory=None):
        super().__init__(directory=directory)
        self._slim: list[dict] = []
        self._last_after_product_files: dict = {}
        self._boundary_deltas: dict[int, dict] = {}
        self._bindings = BindingHistory()

    def _set_boundary(self, index: int, previous: dict, following: dict) -> None:
        delta = _boundary_delta(_product_files(previous), _product_files(following))
        if delta:
            self._boundary_deltas[index] = delta
        else:
            self._boundary_deltas.pop(index, None)

    def append(self, value: dict) -> None:
        slim = _slim_record(value)
        if self._slim:
            self._set_boundary(len(self._slim) - 1,
                               {'product_files': self._last_after_product_files},
                               value.get('before_binding', {}))
        super().append(value)
        self._slim.append(slim)
        self._last_after_product_files = copy.deepcopy(_product_files(value.get('after_binding')))
        self._bindings.append(value.get("after_binding"))
        self.clear_cache()

    def __getitem__(self, index):
        if isinstance(index, slice):
            return copy.deepcopy(self._slim[index])
        return copy.deepcopy(self._slim[self._rows._normalize(index)])

    def __iter__(self) -> Iterator[dict]:
        for row in self._slim:
            yield copy.deepcopy(row)

    def __setitem__(self, index, value: dict) -> None:
        position = self._rows._normalize(index)
        slim = _slim_record(value)
        super().__setitem__(position, value)
        self._slim[position] = slim
        if position == len(self) - 1:
            self._bindings.replace_last(value.get('after_binding'))
        else:
            # Arbitrary historical replacement is rare; rebuild derived metadata
            # once here, never during request rendering.
            self._bindings = BindingHistory()
            for index in range(len(self)):
                self._bindings.append(self.full_record(index).get('after_binding'))
        if position:
            previous = self.full_record(position - 1)
            self._set_boundary(position - 1, previous.get('after_binding', {}),
                               value.get('before_binding', {}))
        if position + 1 < len(self):
            following = self.full_record(position + 1)
            self._set_boundary(position, value.get('after_binding', {}),
                               following.get('before_binding', {}))
        else:
            self._last_after_product_files = copy.deepcopy(_product_files(value.get('after_binding')))
        self.clear_cache()

    def full_record(self, index: int) -> dict:
        return copy.deepcopy(super().__getitem__(index))

    def findings_outcomes(self) -> list[dict]:
        return [{'result': row.get('result'), 'observation': row.get('observation'),
                 'after_binding': ({'revision': row['finding_revision']}
                                   if row.get('finding_revision') is not None else None)}
                for row in self._slim if row.get('tool') == 'run_tests']

    def __getstate__(self):
        return self._rows, copy.deepcopy(self._slim)

    def __setstate__(self, state):
        from kernel_memfix_history import LEDGER_CACHE_BYTES, _DecodedCache
        _DecodedCache.__init__(self, LEDGER_CACHE_BYTES)
        self._rows, saved = state[:2]
        if len(saved) != len(self._rows):
            raise ValueError('slim ledger and authenticated spool lengths differ')
        self._slim = []
        self._last_after_product_files = {}
        self._boundary_deltas = {}
        self._bindings = BindingHistory()
        for index in range(len(self._rows)):
            full = self.full_record(index)
            slim = _slim_record(full)
            if canonical_bytes(saved[index]) not in (
                    canonical_bytes(slim), canonical_bytes(_slim_record(full, version=2))):
                raise ValueError('slim ledger differs from authenticated spool')
            if index:
                self._set_boundary(index - 1,
                                   {'product_files': self._last_after_product_files},
                                   full.get('before_binding', {}))
            self._slim.append(slim)
            self._last_after_product_files = copy.deepcopy(_product_files(full.get('after_binding')))
            self._bindings.append(full.get('after_binding'))
        self.clear_cache()


class StepLedger:
    def __init__(self, *, complete_reads: bool = False,
                 target_symbol: str | None = None, directory=None):
        self.complete_reads = complete_reads
        self.target_symbol = target_symbol
        self.pending: dict | None = None
        self.records = ledger_records(directory=directory) if legacy() else SlimLedgerRecords(directory=directory)

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if legacy() and isinstance(self.records, SpoolSequence):
            self.records = [self.full_record(index) for index in range(len(self.records))]
        elif not legacy() and not isinstance(self.records, SlimLedgerRecords):
            previous = self.records
            self.records = SlimLedgerRecords()
            self.records.extend(previous)

    def restore_records(self, records: list[dict], *, directory=None) -> None:
        if any('before_binding_digest' in row and 'before_binding' not in row
               for row in records):
            raise ValueError('slim ledger rows require their authenticated spool')
        self.records = ledger_records(directory=directory) if legacy() else SlimLedgerRecords(directory=directory)
        self.records.extend(records)

    def full_record(self, index: int) -> dict:
        if isinstance(self.records, SlimLedgerRecords):
            return self.records.full_record(index)
        if isinstance(self.records, SpoolSequence):
            return copy.deepcopy(self.records[index])
        return copy.deepcopy(self.records[index])

    def recent_full(self, count: int) -> list[dict]:
        start = max(0, len(self.records) - count)
        return [self.full_record(index) for index in range(start, len(self.records))]

    def findings_test_outcomes(self) -> list[dict]:
        if isinstance(self.records, SlimLedgerRecords):
            return self.records.findings_outcomes()
        return [{'result': record.get('result'), 'observation': record.get('observation'),
                 'after_binding': record.get('after_binding')}
                for record in self.records if record.get('tool') == 'run_tests']

    def before_product_revisions(self, path: str, current: dict,
                                 default: str) -> list[str]:
        """Reconstruct per-step path revisions from slim change deltas.

        Boundary deltas are captured on append and restore, so snapshots never
        rehydrate historical bindings.
        """
        if not isinstance(self.records, SlimLedgerRecords):
            return [str(value) if value is not None else default
                    for record in self.records
                    for value in (record.get('before_binding', {}).get(
                        'product_files', {}).get(path),)]
        rows = self.records
        files = current.get('product_files', {})
        revision = files.get(path) if isinstance(files, dict) else None
        expected_digest = digest(current)
        result = [default] * len(rows)
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            if row.get('after_binding_digest') != expected_digest:
                if index == len(rows) - 1:
                    revision = rows._last_after_product_files.get(path)
                else:
                    delta = rows._boundary_deltas.get(index, {})
                    if path in delta:
                        revision = delta[path]
            for change in row.get('changes', ()):
                if change.get('role') == 'product' and change.get('path') == path:
                    revision = change.get('before')
                    break
            result[index] = str(revision) if revision is not None else default
            expected_digest = row.get('before_binding_digest')
        return result

    def replace_last(self, record: dict) -> None:
        if not self.records:
            raise IndexError('cannot replace an empty step ledger')
        self.records[-1] = copy.deepcopy(record)

    def slim_state(self) -> dict:
        """JSON-safe rows bound to full records in the authenticated action journal."""
        return {'version': 3, 'source': 'action_outcome_closed',
                'records': [copy.deepcopy(row) for row in self.records]}

    def restore_slim(self, state: dict, *, full_records=None, directory=None) -> None:
        """Authenticate every supported representation before publishing a new store."""
        if type(state) is not dict or type(state.get('version')) is not int:
            raise ValueError('invalid slim ledger checkpoint')
        version = state['version']
        if version not in {1, 2, 3} or full_records is None:
            raise ValueError('slim ledger requires authenticated closed actions')
        records = state.get('records')
        if version == 1 and isinstance(records, SlimLedgerRecords):
            records = [records.full_record(index) for index in range(len(records))]
        if (type(records) is not list or len(records) != len(full_records)
                or (version >= 2 and state.get('source') != 'action_outcome_closed')):
            raise ValueError('slim ledger requires authenticated closed actions')
        expected = (full_records if version == 1 else
                    [_slim_record(full, version=version) for full in full_records])
        if canonical_bytes(records) != canonical_bytes(expected):
            raise ValueError('slim ledger differs from authenticated closed actions')
        self.restore_records(full_records, directory=directory)

    def after_binding_matches(self, index: int, current: dict) -> bool:
        """The router's existing overlapping-file semantics without disk reads."""
        from kernel_stateframe_model_router import _same_revision
        if isinstance(self.records, SlimLedgerRecords):
            index = self.records._rows._normalize(index)
            row = self.records._slim[index]
            return (_same_revision(row['after_binding_revisions'], current)
                    and self.records._bindings.matches(index, current))
        return _same_revision(self.full_record(index).get('after_binding'), current)

    def start(self, tool: str, arguments: dict, step: int, binding: dict,
              *, tool_call_id: str | None = None) -> None:
        if self.pending is not None:
            raise RuntimeError('previous action must be closed before starting another')
        self.pending = {'step': step, 'tool': tool,
                        'arguments': {k: v for k, v in arguments.items() if k not in {'why', 'reason'}},
                        'intent': str(arguments.get('why') or arguments.get('reason') or '')[:600],
                        'before': copy.deepcopy(binding), 'receipts': []}
        if isinstance(tool_call_id, str) and tool_call_id:
            self.pending['tool_call_id'] = tool_call_id

    def receipt(self, receipt: dict) -> None:
        if self.pending is not None:
            self.pending['receipts'].append(copy.deepcopy(receipt))

    def feedback(self, observation: str) -> None:
        if self.pending is not None:
            self.pending['feedback'] = observation

    def read_evidence(self, *, path: str, revision: str, content: str,
                      start_line: int, returned_lines: int, total_lines: int,
                      requested_path: str | None = None,
                      delivery_status: str | None = None,
                      delivered_bytes: int | None = None) -> None:
        if self.pending is not None and self.pending['tool'] in {'read', 'text', 'recall_evidence'}:
            # Enforced frames and continuation return proof need the observed range.
            # Model-facing selection remains separately bounded by public()/the renderer.
            complete = self.complete_reads or 'continuation_id' in self.pending
            evidence = {'path': path, 'file_revision': revision,
                'requested_path': requested_path if requested_path is not None else path,
                'content': content if complete else bounded_feedback(
                    content, 6000, symbol=self.target_symbol),
                'content_hash': digest(content),
                'omitted_chars': 0 if complete else max(0, len(content) - 6000), 'start_line': start_line,
                'end_line': start_line + returned_lines - 1, 'total_lines': total_lines,
                'producer': 'workspace_read', 'authority': 'HOST_OBSERVED_READ'}
            if delivery_status is not None:
                evidence['delivery_status'] = str(delivery_status)
            if delivered_bytes is not None:
                evidence['delivered_bytes'] = max(0, int(delivered_bytes))
            self.pending['read_evidence'] = evidence

    def recalled_evidence(self, result: dict, symbol: str = '', *, public_reference: str | None = None) -> None:
        if self.pending is None or self.pending['tool'] != 'recall_evidence':
            raise RuntimeError('recall evidence requires an active recall step')
        self.read_evidence(path=result['path'], revision=result['revision'], content=result['content'],
            start_line=result['start_line'], returned_lines=len(result['content'].splitlines()),
            total_lines=result['total_lines'], requested_path=(
                public_reference if public_reference is not None else result['reference']))
        self.pending['read_evidence'].update(producer='retained_source_recall',
            omitted_chars=result['omitted_chars_in_selection'])
        self.pending['retained_evidence'] = {key: result[key] for key in (
            'reference', 'path', 'revision', 'start_line', 'selected_end_line', 'freshness',
            'omitted_chars_in_selection', 'more_lines_available')}
        self.pending['retained_evidence'].update(symbol=symbol, content_hash=digest(result['content']))
        for key in ('start_column', 'shown_end_line', 'more_content_available', 'continuation'):
            if key in result:
                self.pending['read_evidence'][key] = copy.deepcopy(result[key])
                self.pending['retained_evidence'][key] = copy.deepcopy(result[key])


    def close(self, messages: list[dict], binding: dict) -> dict | None:
        pending = self.pending
        if pending is None:
            return None
        self.pending = None
        index = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].get('role') == 'assistant'), None)
        feedback = [] if index is None else [str(message.get('content') or '')
            for message in messages[index + 1:] if message.get('role') in {'tool', 'user'}]
        if not feedback and messages and messages[-1].get('role') in {'tool', 'user'}:
            feedback = [str(messages[-1].get('content') or '')]
        text = pending.get('feedback', '\n'.join(feedback))
        changes = []
        for field, role in [('product_files', 'product'), ('harness_files', 'test')]:
            before, after = pending['before'].get(field, {}), binding.get(field, {})
            for path in sorted(set(before) | set(after)):
                if before.get(path) != after.get(path):
                    changes.append({'path': path, 'role': role, 'before': before.get(path),
                                    'after': after.get(path)})
        receipts = pending['receipts']
        action_receipts = [receipt for receipt in receipts if receipt.get('kind') != 'verdict']
        # Keep bookkeeping separate from host execution evidence.  The state
        # frame may later add a closure receipt for an action with no runner
        # response; that row must remain visibly non-semantic to adapters.
        execution_receipts = [receipt for receipt in action_receipts
                              if receipt.get('execution_attempt_id')
                              and receipt.get('dispatch_id')]
        behavioral_receipts = [receipt for receipt in execution_receipts
                               if receipt.get('semantic_eligible') is True
                               and receipt.get('authenticated') is True]
        result = action_receipts[-1].get('result_class', 'UNKNOWN') if action_receipts else 'NO_EXECUTION_RECEIPT'
        if not receipts and '[step-contract]' in text:
            result = 'REJECTED_CONTRACT'
        elif not receipts and '[information-policy]' in text:
            result = 'DENIED_PATH'
        record = {'step': pending['step'], 'tool': pending['tool'],
                  'target': str(pending['arguments'].get('path') or
                                pending['arguments'].get('target') or '')[:300],
                  'arguments_hash': digest(pending['arguments']),
                  'action_parameters': {key: value if isinstance(value, (int, bool)) else str(value)[:300]
                                        for key, value in pending['arguments'].items()
                                        if key in {'path', 'target', 'pattern', 'line'}},
                  'intent': {'text': pending['intent'], 'status': 'UNVERIFIED_MODEL_INTENT'},
                   'result': result, 'receipt_ids': [item.get('receipt_id') for item in receipts],
                   'execution_receipt_ids': [item.get('receipt_id') for item in execution_receipts],
                   'behavioral_receipt_ids': [item.get('receipt_id') for item in behavioral_receipts],
                   'execution_evidence': ('QUALIFYING_BEHAVIORAL'
                                          if behavioral_receipts else
                                          'EXECUTION_RECEIPT' if execution_receipts else
                                          'NO_EXECUTION_RECEIPT'),
                   'observation': bounded_feedback(text, 6000, symbol=self.target_symbol),
                   'observation_hash': digest(text),
                   'observation_omitted_chars': max(0, len(text) - 6000),
                   'changes': changes, 'before_binding': pending['before'],
                   'after_binding': copy.deepcopy(binding)}
        if isinstance(pending.get('tool_call_id'), str) and pending['tool_call_id']:
            record['tool_call_id'] = pending['tool_call_id']
        for key in ('continuation_id', 'continuation_effect_id', 'continuation_dispatch_ref'):
            if key in pending:
                record[key] = pending[key]
        if 'active_work_id' in pending:
            record['active_work_id'] = pending['active_work_id']
        if 'read_evidence' in pending:
            record['read_evidence'] = copy.deepcopy(pending['read_evidence'])
        if 'retained_evidence' in pending:
            record['retained_evidence'] = copy.deepcopy(pending['retained_evidence'])
        record['step_record_id'] = digest(record)
        self.records.append(record)
        return copy.deepcopy(record)

    def public(self, *, current_binding: dict | None = None,
               working_source_path: str | None = None,
               complete_latest_read: bool = False) -> dict:
        edits = [record for record in self.records if record['changes']]
        current = current_binding if current_binding is not None else (
            self.full_record(-1)['after_binding'] if self.records else {})
        reads: dict[tuple, int] = {}
        for index, record in enumerate(self.records):
            if record['tool'] != 'read' and 'read_evidence' not in record:
                continue
            evidence = record.get('read_evidence')
            path = evidence['path'] if evidence else record['target']
            if path == working_source_path:
                continue
            start = evidence['start_line'] if evidence else record['action_parameters'].get('line', 1)
            key = (path, start)
            if evidence is None and key in reads and self.records[reads[key]].get('read_evidence'):
                continue  # Advisory/suppression feedback is not replacement source evidence.
            reads.pop(key, None)
            reads[key] = index
        def read_view(index: int) -> dict:
            record = self.full_record(index)
            evidence = record.get('read_evidence')
            if evidence is None:
                return {'path': record['target'], 'step': record['step'],
                    'content': bounded_feedback(record['observation'], 1200,
                                                symbol=self.target_symbol),
                    'status': 'historical_observation', 'revision_relevance': 'unbound',
                    'omitted_chars': max(0, len(record['observation']) - 1200)
                        + record['observation_omitted_chars']}
            revision = {**current.get('product_files', {}), **current.get('harness_files', {})}.get(evidence['path'])
            # A requested read is the product of the preceding decision. The
            # enforced renderer must deliver its current complete range or
            # report overflow, rather than invite endless rereads of an excerpt.
            complete = (complete_latest_read and index == next(reversed(reads.values()))
                        and revision == evidence['file_revision']
                        and evidence['omitted_chars'] == 0)
            limit = len(evidence['content']) if complete else 1200
            return {**{key: value for key, value in evidence.items() if key not in {'content', 'omitted_chars'}},
                'step': record['step'], 'status': 'observed_source_range' if complete else 'observed_source_excerpt',
                'content': bounded_feedback(evidence['content'], limit,
                                            symbol=self.target_symbol),
                'omitted_chars': max(0, len(evidence['content']) - limit) + evidence['omitted_chars'],
                'revision_relevance': 'same_file_revision' if revision == evidence['file_revision']
                                      else 'historical_file_revision'}
        def retained(index: int) -> dict:
            record = self.full_record(index)
            binding = record['after_binding']
            return {'step': record['step'], 'tool': record['tool'],
                    'parameters': record['action_parameters'], 'result': record['result'],
                    'feedback': bounded_feedback(record['observation'], 1600,
                                                 symbol=self.target_symbol),
                    'feedback_omitted_chars': max(0, len(record['observation']) - 1600)
                        + record['observation_omitted_chars'],
                    'binding': {key: binding.get(key) for key in
                                ('product_revision', 'test_harness_revision', 'environment_revision')},
                    'relevance': 'same_workspace_revision' if binding == current
                                 else 'historical_workspace_revision'}
        # A read must not erase the failing check that motivated it. A newer
        # invocation of the same requested check supersedes its old feedback.
        checks: dict[str, int] = {}
        negatives: list[int] = []
        for index, record in enumerate(self.records):
            if record['tool'] == 'run_tests':
                checks.pop(record['target'], None)
                checks[record['target']] = index
            elif record['result'] in {
                    'RED', 'FAILED', 'REFUSED', 'REFUSED_PREDISPATCH', 'DENIED_PATH',
                    'REJECTED_CONTRACT', 'REJECTED', 'NO_OP', 'ERROR', 'TIMEOUT',
                    'REPEAT_SUPPRESSED', 'REFUSED_SERIES'}:
                negatives.append(index)
        last = self.full_record(-1) if self.records else None
        return {'recent_test_feedback': [retained(index) for index in list(checks.values())[-3:]],
            'recent_test_feedback_omitted': max(0, len(checks) - 3),
            'recent_negative_actions': [retained(index) for index in negatives[-4:]],
            'recent_negative_actions_omitted': max(0, len(negatives) - 4),
            'recent_changes': [
            {'step': record['step'], 'tool': record['tool'], 'stated_intent': record['intent'],
             'changed_files': [{'path': change['path'], 'role': change['role'],
                                'change': 'created' if change['before'] is None else
                                          'deleted' if change['after'] is None else 'modified'}
                               for change in record['changes']],
             'observed_result': record['result']}
            for record in edits[-4:]], 'recent_changes_omitted': max(0, len(edits) - 4),
            'recent_reads': [read_view(index) for index in list(reads.values())[-3:]],
            'last_action': ({'tool': last['tool'], 'step': last['step'],
                             'parameters': last['action_parameters'],
                             'stated_intent': last['intent'],
                             'result': last['result'],
                             'feedback': bounded_feedback(last['observation'], 2000,
                                                           symbol=self.target_symbol),
                             'feedback_omitted_chars': max(0, len(last['observation']) - 2000)
                                  + last['observation_omitted_chars']}
                            if last is not None else None)}
