"""Payload-free joins for native delivery and historical source acquisition."""
from __future__ import annotations

from bisect import bisect_left, insort
import hashlib


class EvidenceIndex:
    def __init__(self):
        self.requests: dict[tuple, list[str]] = {}
        self.origins: dict[tuple, list[str]] = {}
        self.steps: dict[str, list[str]] = {}
        self.attempts: dict[tuple, list[str]] = {}
        self.sources: dict[tuple, list[tuple]] = {}
        self.order: dict[str, int] = {}

    def add(self, event: dict) -> None:
        key, kind = event['event_id'], event['kind']
        self.order[key] = len(self.order)
        for field, index in (('request_id', self.requests), ('origin_request_id', self.origins)):
            value = event.get(field)
            if isinstance(value, str):
                index.setdefault((kind, value), []).append(key)
        if kind == 'action_outcome_closed':
            record = event.get('record')
            step = record.get('step_record_id') if isinstance(record, dict) else None
            if isinstance(step, str):
                self.steps.setdefault(step, []).append(key)
        if kind == 'request_final_wire_prepared':
            attempt = event.get('transport_attempt')
            if isinstance(attempt, (int, str)) or attempt is None:
                self.attempts.setdefault((event.get('request_id'), attempt), []).append(key)
        if (kind != 'request_source_closure'
                or event.get('producer') != 'kernel_request_source_window'
                or event.get('dispatch') is not False):
            return
        plan = event.get('closure')
        if not isinstance(plan, dict):
            return
        revisions, sources = plan.get('revisions'), plan.get('source_contents')
        if not isinstance(revisions, dict) or not isinstance(sources, dict):
            return
        for path, source in sources.items():
            if not isinstance(source, str):
                continue
            revision = 'sha256:' + hashlib.sha256(source.encode()).hexdigest()
            if revisions.get(path) == revision and type(event.get('sequence')) is int:
                insort(self.sources.setdefault((path, revision), []),
                       (event['sequence'], self.order[key], key))

    def native_events(self, events, origin: str) -> list[dict]:
        from kernel_native_prefix import VALIDATION_KINDS
        ids = set()
        for kind in VALIDATION_KINDS - {'request_source_closure', 'request_final_wire_prepared'}:
            ids.update(self.requests.get((kind, origin), ()))
            ids.update(self.origins.get((kind, origin), ()))
        rows = [events[key] for key in sorted(ids, key=self.order.__getitem__)]
        for row in tuple(rows):
            if row['kind'] == 'native_batch_action_closed':
                for key in self.steps.get(row.get('step_record_id'), ()):
                    if key not in ids:
                        rows.append(events[key])
                        ids.add(key)
        rows.sort(key=lambda row: self.order[row['event_id']])
        return rows

    def final_events(self, events, request: str, attempt) -> list[dict]:
        ids = self.attempts.get((request, attempt), ())
        if not ids:
            # A different attempt must still disable the old-journal fallback.
            ids = self.requests.get(('request_final_wire_prepared', request), ())[:1]
        return [events[key] for key in ids]

    def acquired_events(self, events, path: str, revision: str, before: int) -> list[dict]:
        candidates = self.sources.get((path, revision), ())
        end = bisect_left(candidates, (before,))
        if not end:
            return []
        return [events[candidates[end - 1][2]]]
