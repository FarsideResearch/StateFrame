"""Immutable, scoped source revisions behind the world-knowledge projection.

The micrograph is a bounded relevance index, never a freshness authority. Source
bytes survive graph eviction and visible-context selection. Host binding hashes
are required to label a recalled revision current.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from collections.abc import Callable, Mapping

from detreason.micrograph import GraphDelta, MicroGraph, make_edge, make_node
from detreason.world_knowledge_extractors import normalize_path


def _sha(content: bytes) -> str:
    return 'sha256:' + hashlib.sha256(content).hexdigest()


def _event_id(event: Mapping) -> str:
    return _sha(json.dumps({k: v for k, v in event.items() if k != 'event_id'},
                           sort_keys=True, separators=(',', ':')).encode())


class SourceArchiveCapacityError(ValueError):
    """The bounded archive is full; the cap is right, killing the run is not."""

    capacity_exhausted = True


class NonProductSourceError(ValueError):
    """Tool output (coverage/caches) is never retained as product source."""


class SourceArchive:
    def __init__(self, scope: str, *, sink: Callable[[dict], None] | None = None,
                 max_bytes: int = 64_000_000, max_file_bytes: int = 8_000_000,
                 max_revisions: int = 4096, max_steps: int = 10000, max_events: int = 50000):
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError('source archive needs a unique run scope')
        if type(max_bytes) is not int or type(max_file_bytes) is not int or not 0 < max_file_bytes <= max_bytes:
            raise ValueError('invalid source storage limits')
        self.scope, self.sink = scope, sink
        self.max_bytes, self.max_file_bytes = max_bytes, max_file_bytes
        if any(type(value) is not int or value <= 0 for value in (max_revisions, max_steps, max_events)):
            raise ValueError('invalid source index limits')
        self.max_revisions, self.max_steps = max_revisions, max_steps
        self.max_events = max_events
        self.blobs: dict[str, bytes] = {}
        self.revisions: dict[str, dict] = {}
        self.latest: dict[str, str] = {}
        self.steps: dict[str, dict] = {}
        self.graph = MicroGraph(workspace_id=scope, case_id=scope)
        self.used_bytes = 0
        self.seen_events: dict[int, str] = {}

    def retain(self, path: str, content: bytes, evidence_ref: str) -> str:
        path = normalize_path(path)
        if not isinstance(content, bytes) or not isinstance(evidence_ref, str) or not evidence_ref:
            raise ValueError('host source bytes and evidence reference required')
        from kernel_non_product_paths import is_non_product
        # Audit 2026-10-05 #2: refuse tool output before counting.  Only a
        # workspace-relative path is classified (an absolute path's ancestors,
        # e.g. a scratch root under .test-tmp, say nothing about the file).
        if not path.startswith('/') and is_non_product(path):
            raise NonProductSourceError(f'non-product path not retained: {path}')
        revision = _sha(content)
        reference = _sha((self.scope + '\0' + path + '\0' + revision).encode())
        if reference in self.revisions:
            event = {'schema': 'retained-source/1', 'scope': self.scope,
                     'kind': 'source_reobserved', 'reference': reference, 'evidence_ref': evidence_ref}
            self._publish(event)
            self.latest[path] = reference
            self.seen_events[event['sequence']] = _event_id(event)
            return reference
        additional = 0 if revision in self.blobs else len(content)
        if (len(content) > self.max_file_bytes or self.used_bytes + additional > self.max_bytes
                or len(self.revisions) >= self.max_revisions):
            raise SourceArchiveCapacityError('source archive capacity exceeded; content not retained')
        event = {'schema': 'retained-source/1', 'scope': self.scope, 'kind': 'source_retained',
                 'reference': reference, 'path': path, 'revision': revision,
                 'evidence_ref': evidence_ref, 'content_base64': base64.b64encode(content).decode()}
        self._publish(event)
        self._ingest_source(event, content)
        self.seen_events[event['sequence']] = _event_id(event)
        return reference

    def _publish(self, event: dict) -> None:
        if len(self.seen_events) >= self.max_events:
            raise SourceArchiveCapacityError('source event index capacity exceeded')
        event['sequence'] = len(self.seen_events) + 1
        if self.sink is not None:
            self.sink(event)  # A durable failure must not masquerade as retained evidence.

    def _ingest_source(self, event: dict, content: bytes) -> None:
        ref, revision, path = event['reference'], event['revision'], event['path']
        if ref in self.revisions:
            return
        if revision not in self.blobs:
            self.blobs[revision] = content
            self.used_bytes += len(content)
        self.revisions[ref] = {key: event[key] for key in
                               ('reference', 'path', 'revision', 'evidence_ref')}
        self.latest[path] = ref
        artifact = make_node('value', {'scope': self.scope, 'path': path},
                             {'role': 'source_path', 'path': path})
        source = make_node('evidence', ref, {**self.revisions[ref], 'authority': 'HOST_OBSERVED_SOURCE'})
        self.graph.apply_delta(GraphDelta((artifact, source),
            (make_edge('flows-into', source.id, artifact.id),)))

    def bind_step(self, record: Mapping, *, work_id: str = '', work_version: int = 0) -> None:
        step_id = record.get('step_record_id')
        if not isinstance(step_id, str) or not step_id:
            raise ValueError('only closed host steps may be linked')
        if step_id in self.steps:
            return
        if len(self.steps) >= self.max_steps:
            raise ValueError('source step index capacity exceeded')
        binding = record.get('after_binding', {})
        hashes = {**binding.get('product_files', {}), **binding.get('harness_files', {})}
        paths = {record.get('target', '')}
        paths.update(item['path'] for item in record.get('changes', []))
        references = [ref for ref, value in self.revisions.items()
                      if record.get('result') in {'OK', 'APPLIED', 'RECALLED', 'GREEN', 'RED'}
                      and value['path'] in paths and hashes.get(value['path']) == value['revision']]
        recalled = record.get('retained_evidence')
        if recalled is not None and record.get('result') == 'RECALLED':
            ref = recalled['reference']
            if ref not in self.revisions or self.revisions[ref]['revision'] != recalled['revision']:
                raise ValueError('recall step references unknown source revision')
            references = list(dict.fromkeys([*references, ref]))
        event = {'schema': 'retained-source/1', 'scope': self.scope, 'kind': 'source_step_linked',
                 'step_record_id': step_id, 'step': record.get('step'), 'tool': record.get('tool'),
                 'references': references, 'work_id': work_id, 'work_version': work_version,
                 'work_authority': 'MODEL_UNVERIFIED', 'result': record.get('result'),
                 'source_relation': 'HOST_ACTION_INPUT_OR_OUTPUT',
                 'recalled_excerpt': dict(recalled) if recalled is not None else None}
        self._publish(event)
        self._ingest_step(event)
        self.seen_events[event['sequence']] = _event_id(event)

    def _ingest_step(self, event: dict) -> None:
        self.steps[event['step_record_id']] = dict(event)
        step = make_node('evidence', event['step_record_id'],
                         {'step': event['step'], 'tool': event['tool'], 'result': event['result']})
        nodes, edges = [step], []
        for ref in event['references']:
            value = self.revisions[ref]
            source = make_node('evidence', ref, {**value, 'authority': 'HOST_OBSERVED_SOURCE'})
            nodes.append(source)
            edges.append(make_edge('flows-into', source.id, step.id))
        if event['work_id']:
            work = make_node('claim', {'scope': self.scope, 'work_id': event['work_id'],
                                      'version': event['work_version']},
                             {'authority': 'MODEL_UNVERIFIED'})
            nodes.append(work)
            edges.append(make_edge('explains', step.id, work.id))
        self.graph.apply_delta(GraphDelta(tuple(nodes), tuple(edges)))

    def recall(self, reference: str, current_hashes: Mapping[str, str], *,
               start_line: int = 1, max_lines: int = 120, max_chars: int = 6000,
               symbol: str = '', start_column: int = 0, continuation: bool = False) -> dict:
        if type(start_line) is not int or start_line < 1 or type(max_lines) is not int or not 1 <= max_lines <= 200:
            raise ValueError('invalid recall line range')
        if type(max_chars) is not int or not 1 <= max_chars <= 12000:
            raise ValueError('invalid recall character budget')
        if type(start_column) is not int or start_column < 0:
            raise ValueError('invalid recall column')
        if start_column and (not continuation or symbol):
            raise ValueError('column requires continuation without symbol')
        value = self.revisions[reference]
        raw = self.blobs[value['revision']]
        try:
            text, replaced = raw.decode('utf-8'), False
        except UnicodeDecodeError:
            text, replaced = raw.decode('utf-8', errors='replace'), True
        lines = text.splitlines(keepends=True)
        end = min(len(lines), start_line - 1 + max_lines)
        if symbol:
            matches = []
            def visit(node, prefix=''):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = prefix + child.name
                        if name == symbol:
                            matches.append(child)
                        visit(child, name + '.')
                    else:
                        visit(child, prefix)
            visit(ast.parse(text))
            if len(matches) != 1:
                raise ValueError('symbol must identify exactly one stored definition')
            start_line = min([matches[0].lineno] + [d.lineno for d in matches[0].decorator_list])
            end = min(matches[0].end_lineno, start_line - 1 + max_lines)
        if start_column and (start_line > len(lines) or start_column >= len(lines[start_line - 1])):
            raise ValueError('recall column outside line')
        selected = ''.join(lines[start_line - 1:end])[start_column:]
        shown = selected[:max_chars]
        pagination = {}
        if continuation:
            cursor_line, cursor_column, remaining = start_line, start_column, len(shown)
            for line in lines[start_line - 1:]:
                available = len(line) - cursor_column
                if remaining < available:
                    cursor_column += remaining
                    break
                remaining -= available
                cursor_line += 1
                cursor_column = 0
                if not remaining:
                    break
            more = start_line <= len(lines) and (len(shown) < len(selected) or end < len(lines))
            pagination = {
                'start_column': start_column,
                'shown_end_line': cursor_line if cursor_column else cursor_line - 1,
                'more_content_available': more,
                'more_lines_available': more,
                'continuation': {'line': cursor_line, 'column': cursor_column} if more else None,
            }
        return {**value, 'freshness': 'current' if current_hashes.get(value['path']) == value['revision']
                else 'historical', 'authority': 'HOST_OBSERVED_SOURCE', 'content': shown,
                'start_line': start_line, 'selected_end_line': end, 'total_lines': len(lines),
                'omitted_chars_in_selection': max(0, len(selected) - len(shown)),
                'more_lines_available': end < len(lines), 'decoded_with_replacement': replaced,
                'range_status': 'empty_file' if not lines else
                                'out_of_range' if start_line > len(lines) else 'available',
                'current_reference': next((ref for ref, item in self.revisions.items()
                    if item['path'] == value['path'] and item['revision'] == current_hashes.get(value['path'])), None), **pagination}

    def select(self, current_hashes: Mapping[str, str], working_path: str, *,
               work_id: str = '', work_version: int = 0, requested: str = '',
               requested_start_line: int = 1, requested_symbol: str = '',
               stable_public_order: bool = False, deliver_requested: bool = False,
               requested_column: int = 0) -> dict:
        """Prefer work-linked evidence, then recent observed source; show at most two excerpts."""
        preferred = [requested] if requested else []
        if work_id:
            work = make_node('claim', {'scope': self.scope, 'work_id': work_id, 'version': work_version})
            steps = {edge.source for edge in self.graph.edges if edge.target == work.id and edge.kind == 'explains'}
            linked = []
            for edge in self.graph.edges:
                node = self.graph.node(edge.source)
                if edge.kind == 'flows-into' and edge.target in steps and node is not None:
                    ref = node.attrs.get('reference')
                    if ref in self.revisions:
                        linked.append(ref)
            if stable_public_order:
                # Graph IDs include run scope. Equal-relevance sources need a
                # content/path ordering before the bounded selection is applied.
                linked.sort(key=lambda ref: (self.revisions[ref]['path'], self.revisions[ref]['revision']))
            preferred.extend(linked)
        for step in reversed(list(self.steps.values())):
            preferred.extend(step['references'])
        preferred.extend(reversed(list(self.latest.values())))
        unique, paths = [], set()
        for ref in preferred:
            if ref not in self.revisions:
                continue
            value = self.revisions[ref]
            latest = self.latest.get(value['path'])
            if ref != requested and latest is not None and (
                    self.revisions[latest]['revision'] == current_hashes.get(value['path'])):
                ref, value = latest, self.revisions[latest]
            if ref in unique or value['path'] in paths:
                continue
            # An explicit historical recall must not hide the current revision.
            if ref != requested or value['revision'] == current_hashes.get(value['path']):
                paths.add(value['path'])
            unique.append(ref)
        catalog = []
        selected = []
        for ref in unique:
            value = self.revisions[ref]
            current = current_hashes.get(value['path']) == value['revision']
            if len(catalog) < 12:
                catalog.append({**value, 'freshness': 'current' if current else 'historical',
                                'in_working_source': value['path'] == working_path})
            requested_item = deliver_requested and ref == requested
            if (current or requested_item) and (value['path'] != working_path or requested_item) and len(selected) < 2:
                item = self.recall(ref, current_hashes, max_chars=6000 if requested_item else 1800,
                    start_line=requested_start_line if ref == requested else 1,
                    symbol=requested_symbol if ref == requested else '',
                    start_column=requested_column if requested_item else 0,
                    continuation=deliver_requested)
                if requested_item:
                    item['requested_delivery'] = True
                selected.append(item)
        return {'catalog': catalog,
                'omitted_paths': len({self.revisions[ref]['path'] for ref in unique}
                                     - {item['path'] for item in catalog}),
                'omitted_references': max(0, len(unique) - len(catalog)),
                'selected': selected, 'stored_revisions': len(self.revisions),
                'stored_bytes': self.used_bytes, 'graph_digest': self.graph.digest()}

    def replay(self, events) -> None:
        for event in events:
            if event.get('schema') != 'retained-source/1' or event.get('scope') != self.scope:
                continue
            identity = _event_id(event)
            sequence = event.get('sequence')
            if type(sequence) is not int or sequence < 1:
                raise ValueError('retained source event sequence missing or invalid')
            if sequence in self.seen_events:
                if self.seen_events[sequence] != identity:
                    raise ValueError('retained source event integrity mismatch')
                continue
            if sequence != len(self.seen_events) + 1:
                raise ValueError('retained source replay sequence gap')
            if len(self.seen_events) >= self.max_events:
                raise ValueError('replay exceeds source event index capacity')
            if event.get('kind') == 'source_retained':
                if len(event['content_base64']) > 4 * ((self.max_file_bytes + 2) // 3):
                    raise ValueError('replay exceeds source archive capacity')
                content = base64.b64decode(event['content_base64'], validate=True)
                path = normalize_path(event['path'])
                revision = _sha(content)
                ref = _sha((self.scope + '\0' + path + '\0' + revision).encode())
                if event['reference'] != ref or event['revision'] != revision:
                    raise ValueError('retained source integrity mismatch')
                additional = 0 if revision in self.blobs else len(content)
                if (len(content) > self.max_file_bytes or self.used_bytes + additional > self.max_bytes
                        or ref not in self.revisions and len(self.revisions) >= self.max_revisions):
                    raise ValueError('replay exceeds source archive capacity')
                self._ingest_source({**event, 'path': path}, content)
            elif event.get('kind') == 'source_step_linked':
                if event['step_record_id'] not in self.steps and len(self.steps) >= self.max_steps:
                    raise ValueError('replay exceeds source step index capacity')
                if any(ref not in self.revisions for ref in event['references']):
                    raise ValueError('step references missing source evidence')
                self._ingest_step(event)
            elif event.get('kind') == 'source_reobserved':
                value = self.revisions[event['reference']]
                self.latest[value['path']] = event['reference']
            else:
                continue
            self.seen_events[sequence] = identity
