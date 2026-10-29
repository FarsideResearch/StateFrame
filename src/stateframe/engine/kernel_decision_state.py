"""Bounded model-authored work state, separate from kernel-owned evidence."""
from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping

from kernel_measurement import digest
from kernel_working_notebook import (
    enabled as notebook_enabled, note_schema, tool_surface as notebook_tool_surface,
    validate_note,
)

STATUSES = ('planned', 'active', 'verifying', 'ready', 'blocked', 'completed', 'superseded')
MAX_ITEMS = 8
MAX_NOTES = 6


def enabled() -> bool:
    value = os.environ.get('KERNEL_DECISION_STATE', '0')
    if value not in {'0', '1'}:
        raise ValueError('KERNEL_DECISION_STATE must be 0 or 1')
    return value == '1' or notes_enabled()


def notes_enabled() -> bool:
    return notebook_enabled()


def notes_only() -> bool:
    return notes_enabled() and os.environ.get('KERNEL_DECISION_STATE', '0') == '0'


def tool_surface(base: list[dict]) -> list[dict]:
    if not enabled():
        return base
    result = list(base)
    existing = {tool['function']['name']: tool for tool in base}
    for tool in tools():
        if notes_only() and tool['function']['name'] != 'note_update':
            continue
        name = tool['function']['name']
        if name in existing and existing[name] != tool:
            raise ValueError('conflicting decision tool definition: ' + name)
        if name not in existing:
            result.append(tool)
    return notebook_tool_surface(result)



def instructions(base: str) -> str:
    if not enabled():
        return base
    if notes_only():
        return base + ('\nOptional note_update retains a concise finding, hypothesis or next question. '
                       'Notes are unverified model judgments, not facts or completion requirements. '
                       'Use expected_version=0 for a new id or the displayed note version to revise; '
                       'empty text retires an existing note. evidence_steps may be []. '
                       'Notes consume decisions; direct actions and done remain available without notes.')
    return base + (
        '\nPersistent work and notes are available when useful: work_update records a plan, '
        'next_action, acceptance criteria or blocker; note_update retains a bounded hypothesis or context. '
        'These are optional tools, not a requirement to plan before every edit. '
        'Use expected_version=0 for a new id, otherwise the current version shown in decision_state. '
        'State updates consume decisions; execute useful workspace actions between updates. '
        'Notes and plans remain unverified. Marking work completed requires current public test receipts '
        'and does not prove arbitrary acceptance claims.')


def _tool(name: str, description: str, properties: dict) -> dict:
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties,
                       'required': list(properties), 'additionalProperties': False}}}


def tools() -> list[dict]:
    def text(limit: int) -> dict:
        return {'type': 'string', 'maxLength': limit}
    result = [_tool('work_update',
        'Create or revise persistent work; all fields are required. Use "" for unused text and [] '
        'for unused reference lists. expected_version=0 creates; otherwise use the displayed current version. '
        'IDs use letters, digits, underscores or hyphens and start with a letter or digit. '
        'New work starts planned, active or blocked. objective and acceptance must be nonempty. '
        'planned, active and verifying require a nonempty next_action. blocked and superseded require '
        'a nonempty blocker. superseded requires a different existing replacement id in superseded_by; '
        'all other statuses require superseded_by="". depends_on lists existing work ids; '
        'proof_steps lists observed action step numbers, or [] when no proof exists. '
        'Only one item may be active. Dependencies must be ready or completed before dependent work starts. '
        'ready is an unverified handoff for dependent work, not final completion. completed requires '
        'current public checks and a matching observed run_tests proof step; it does not prove arbitrary '
        'acceptance text. Plans remain unverified. State updates consume a decision; execute useful work.', {
            'id': text(40), 'expected_version': {'type': 'integer', 'minimum': 0},
            'status': {'type': 'string', 'enum': list(STATUSES)},
            'objective': text(300), 'acceptance': text(400), 'next_action': text(300),
            'blocker': text(300), 'depends_on': {'type': 'array', 'maxItems': 8,
                'items': text(40)}, 'proof_steps': {'type': 'array', 'maxItems': 8,
                'items': {'type': 'integer', 'minimum': 0}}, 'superseded_by': text(40)}),
        _tool('note_update',
            'Retain bounded unverified context or a hypothesis. Notes cannot change facts, permissions, '
            'tests or completion. All fields are required. expected_version=0 creates; otherwise use '
            'the displayed current version. New notes require nonempty text; empty text retires an '
            'existing note. evidence_steps lists observed action step numbers, or [] when none. '
            'IDs use letters, digits, underscores or hyphens and start with a letter or digit. '
                'Unsupported notes remain unverified. New v44 notes may include the full '
                'scope/kind/refs/dependency_revision/status/supersedes contract.', {
                'id': text(40), 'expected_version': {'type': 'integer', 'minimum': 0},
                'text': text(600), 'evidence_steps': {'type': 'array', 'maxItems': 8,
                    'items': {'type': 'integer', 'minimum': 0}},
                'scope': note_schema()['properties']['scope'],
                'kind': note_schema()['properties']['kind'],
                'refs': note_schema()['properties']['refs'],
                'dependency_revision': note_schema()['properties']['dependency_revision'],
                'status': note_schema()['properties']['status'],
                'supersedes': note_schema()['properties']['supersedes']})]
    # Keep the historical note_update call contract.  The v44 model-owned
    # contract is carried as an optional sidecar on useful actions.
    result[1]['function']['parameters']['required'] = [
        'id', 'expected_version', 'text', 'evidence_steps',
    ]
    return result


class DecisionState:
    def __init__(self):
        self.items: dict[str, dict] = {}
        self.notes: dict[str, dict] = {}
        self.note_archive: dict[str, dict] = {}
        self.sequence = 0

    @staticmethod
    def _string(value: object, name: str, limit: int, *, required: bool = False) -> str:
        if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
            raise ValueError(f'{name} must be a string of at most {limit} characters'
                             + (' and nonempty' if required else ''))
        return value

    @staticmethod
    def _refs(value: object, known: set, name: str) -> list:
        if not isinstance(value, list) or len(value) > 8:
            raise ValueError(f'{name} must be a list of at most 8 references')
        if any(type(item) not in {str, int} or item not in known for item in value):
            raise ValueError(f'{name} contains an unknown reference')
        if len(set(value)) != len(value):
            raise ValueError(f'{name} contains duplicate references')
        return list(value)

    def apply(self, name: str, args: dict, *, records: list[dict], binding: dict,
              public_verified: bool, step: int) -> dict:
        """Validate completely before committing. Rejections leave all state unchanged."""
        if notes_only() and name != 'note_update':
            raise ValueError('working-notes mode does not enable work_update')
        schemas = {tool['function']['name']: tool['function']['parameters'] for tool in tools()}
        schema = schemas.get(name)
        if (schema is None
                or set(args) - set(schema['properties'])
                or set(schema.get('required', ())) - set(args)):
            raise ValueError('unknown state operation or fields')
        ident = self._string(args['id'], 'id', 40, required=True)
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', ident):
            raise ValueError('id must contain only letters, numbers, underscores or hyphens')
        collection = self.items if name == 'work_update' else self.notes
        previous = collection.get(ident)
        version = args['expected_version']
        if type(version) is not int or version != (previous['version'] if previous else 0):
            raise ValueError('stale expected_version; use the current version in the frame')
        limit = MAX_ITEMS if name == 'work_update' else MAX_NOTES
        if previous is None and len(collection) >= limit:
            raise ValueError(f'state capacity reached ({limit}); revise an existing entry')
        by_step = {record['step']: record for record in records}
        candidate = {'id': ident, 'version': version + 1, 'updated_step': step,
                     'author': 'model', 'binding': copy.deepcopy(binding)}
        if name == 'note_update':
            note_args = self._normalize_note_args(args, binding=binding, by_step=by_step)
            return self._commit_note(
                note_args, previous=previous, binding=binding, step=step,
                action_record=None, operation=name,
            )
        else:
            status = args['status']
            if not isinstance(status, str) or status not in STATUSES:
                raise ValueError('unknown work status')
            if previous is None and status not in {'planned', 'active', 'blocked'}:
                raise ValueError('new work starts planned, active or blocked')
            candidate.update(status=status,
                objective=self._string(args['objective'], 'objective', 300, required=True),
                acceptance=self._string(args['acceptance'], 'acceptance', 400, required=True),
                next_action=self._string(args['next_action'], 'next_action', 300,
                                         required=status in {'planned', 'active', 'verifying'}),
                blocker=self._string(args['blocker'], 'blocker', 300,
                                     required=status in {'blocked', 'superseded'}),
                depends_on=self._refs(args['depends_on'], set(self.items), 'depends_on'),
                proof_steps=self._refs(args['proof_steps'], set(by_step), 'proof_steps'),
                superseded_by=self._string(args['superseded_by'], 'superseded_by', 40),
                authority='MODEL_WORK_CLAIM', public_checks_current_at_update=False)
            if status == 'superseded':
                if candidate['superseded_by'] not in self.items or candidate['superseded_by'] == ident:
                    raise ValueError('superseded work requires a different existing replacement item')
            elif candidate['superseded_by']:
                raise ValueError('superseded_by is only valid for superseded work')
            graph = {key: item['depends_on'] + ([item['superseded_by']] if item['superseded_by'] else [])
                     for key, item in self.items.items()}
            graph[ident] = candidate['depends_on'] + ([candidate['superseded_by']] if candidate['superseded_by'] else [])
            def visit(key: str, stack: set[str]) -> None:
                if key in stack:
                    raise ValueError('cyclic work dependency')
                for dependency in graph[key]:
                    visit(dependency, stack | {key})
            visit(ident, set())
            if status in {'active', 'verifying', 'ready', 'completed'}:
                if any(not self._handoff_ready(self.items[key]) for key in candidate['depends_on']):
                    raise ValueError('dependencies must be ready or completed before this work starts; readiness is a model claim')
            if status == 'active' and any(item['status'] == 'active' and key != ident
                                          for key, item in self.items.items()):
                raise ValueError('another work item is active; update it first')
            if status == 'completed':
                proofs = [by_step[ref] for ref in candidate['proof_steps']]
                if not public_verified or not any(record['tool'] == 'run_tests'
                    and (record['after_binding_digest'] == digest(binding)
                         if 'after_binding_digest' in record else record['after_binding'] == binding)
                    and record['result'] in {'GREEN', 'REUSED_VERIFICATION'} for record in proofs):
                    raise ValueError('completion requires current public checks and a matching observed test step')
                candidate['public_checks_current_at_update'] = True
        if previous is not None:
            def semantic(item: dict) -> dict:
                return {key: value for key, value in item.items()
                        if key not in {'version', 'updated_step', 'binding'}}
            if semantic(previous) == semantic(candidate) and previous['binding'] == binding:
                raise ValueError('semantic no-op; execute useful work instead of repeating the state update')
        self.sequence += 1
        collection[ident] = candidate
        return {'sequence': self.sequence, 'operation': name, 'previous': copy.deepcopy(previous),
                'current': copy.deepcopy(candidate), 'state_hash': digest({'items': self.items, 'notes': self.notes})}

    @staticmethod
    def _normalize_note_args(args: dict, *, binding: dict, by_step: dict) -> dict:
        """Accept the legacy note_update shape while normalizing v44 sidecars."""
        if 'scope' not in args:
            normalized = validate_note(args, allow_legacy=True)
            normalized['evidence_steps'] = [
                step for step in normalized.get('evidence_steps', []) if step in by_step
            ]
            normalized['dependency_revision'] = digest(binding)
            return normalized
        normalized = validate_note(args)
        normalized['evidence_steps'] = []
        return normalized

    def _archive_note(self, note: dict, *, reason: str) -> None:
        archived = copy.deepcopy(note)
        archived['archive_reason'] = reason
        key = f"{note['id']}@v{note['version']}@{reason}"
        self.note_archive[key] = archived

    def _commit_note(self, note_args: dict, *, previous: dict | None, binding: dict,
                     step: int, action_record: dict | None, operation: str,
                     dependency_snapshot: dict | None = None) -> dict:
        identifier = note_args['id']
        if previous is not None and previous.get('status') == 'retired' and not note_args['text']:
            raise ValueError('note is already retired; revise it with nonempty text first')
        if note_args.get('supersedes'):
            superseded = self.notes.get(str(note_args['supersedes']))
            if superseded is None or superseded['id'] == identifier:
                raise ValueError('supersedes must name an existing different note')
            self._archive_note(superseded, reason='superseded')
            superseded['status'] = 'superseded'
        elif previous is not None:
            prior_semantic = {
                key: value for key, value in previous.items()
                if key not in {'version', 'updated_step', 'binding', 'dependency_snapshot'}
            }
            requested_semantic = {
                key: value for key, value in note_args.items()
                if key not in {'expected_version', 'evidence_steps'}
            }
            if prior_semantic == requested_semantic and previous.get('binding') == binding:
                raise ValueError('semantic no-op; revise the note or execute useful work')
            self._archive_note(previous, reason='revised')

        refs = copy.deepcopy(note_args['refs'])
        if action_record is not None:
            target = action_record.get('target')
            if isinstance(target, str) and target and target not in refs['files']:
                refs['files'].append(target)
            for receipt in action_record.get('receipt_ids', ()):
                if isinstance(receipt, str) and receipt and receipt not in refs['receipts']:
                    refs['receipts'].append(receipt)
        if len(refs['files']) > 8 or len(refs['receipts']) > 8:
            raise ValueError('note reference capacity reached (8 per reference kind)')
        if dependency_snapshot is None:
            dependency_snapshot = {'binding': copy.deepcopy(binding)}
        candidate = {
            'id': identifier,
            'version': int(note_args['expected_version']) + 1,
            'updated_step': step,
            'author': 'model',
            'scope': copy.deepcopy(note_args['scope']),
            'kind': note_args['kind'],
            'text': note_args['text'],
            'refs': refs,
            'dependency_revision': digest(dependency_snapshot),
            'declared_dependency_revision': note_args.get('dependency_revision', ''),
            'status': note_args['status'],
            'supersedes': note_args.get('supersedes'),
            'binding': copy.deepcopy(binding),
            'dependency_snapshot': copy.deepcopy(dependency_snapshot),
            'authority': 'UNVERIFIED_MODEL_NOTE',
        }
        if candidate['status'] == 'retired':
            self._archive_note(candidate, reason='retired')
        self.notes[identifier] = candidate
        self.sequence += 1
        event = {
            'sequence': self.sequence,
            'operation': operation,
            'previous': copy.deepcopy(previous),
            'current': copy.deepcopy(candidate),
            'state_hash': digest({'items': self.items, 'notes': self.notes,
                                  'archive': self.note_archive}),
        }
        if action_record is not None:
            event['action_record'] = copy.deepcopy(action_record)
            event['arguments'] = copy.deepcopy(note_args)
        return event

    def apply_attached_note(self, value: object, *, action_record: dict,
                            binding: dict, step: int,
                            dependency_snapshot: dict | None = None) -> dict:
        if not notebook_enabled():
            raise ValueError('working notebook is disabled')
        note_args = validate_note(value)
        previous = self.notes.get(note_args['id'])
        expected = note_args['expected_version']
        if expected != (previous['version'] if previous else 0):
            raise ValueError('stale expected_version; use the current note version in the frame')
        return self._commit_note(
            note_args, previous=previous, binding=binding, step=step,
            action_record=action_record, operation='note_attached',
            dependency_snapshot=dependency_snapshot,
        )

    def refresh_notes(self, dependency_snapshot: dict) -> None:
        """Demote changed dependencies without deleting or rewriting model text."""
        for note in self.notes.values():
            if note.get('status') != 'active':
                continue
            previous = note.get('dependency_snapshot')
            if previous is not None and self._dependency_changed(previous, dependency_snapshot):
                note['status'] = 'stale-check'
                note['stale_reason'] = 'referenced file, receipt or dependency revision changed'
                note['stale_at_revision'] = digest(dependency_snapshot)
            elif previous is None and note.get('binding') != dependency_snapshot.get('binding'):
                note['status'] = 'stale-check'
                note['stale_reason'] = 'dependency revision changed'
                note['stale_at_revision'] = digest(dependency_snapshot)

    @staticmethod
    def _dependency_changed(previous: dict, current: dict) -> bool:
        if previous.get('binding') != current.get('binding'):
            return True
        for kind in ('files', 'receipts'):
            old_values = previous.get(kind, {})
            new_values = current.get(kind, {})
            if any(new_values.get(key) != value for key, value in old_values.items()):
                return True
        return False

    def retrieve_notes(self, *, note_id: str | None = None,
                       question: str | None = None) -> dict:
        from kernel_working_notebook import RETRIEVAL_SCHEMA, retrieval_matches
        if note_id is not None and (not isinstance(note_id, str) or len(note_id) > 40):
            raise ValueError('note id is invalid')
        if question is not None and (not isinstance(question, str) or len(question) > 300):
            raise ValueError('note question is invalid')
        values = [*self.notes.values(), *self.note_archive.values()]
        selected = [copy.deepcopy(note) for note in values
                    if retrieval_matches(note, note_id=note_id, question=question)]
        selected.sort(key=lambda item: (-int(item.get('updated_step', 0)), item.get('id', '')))
        return {
            'schema': RETRIEVAL_SCHEMA,
            'query': {'id': note_id or '', 'question': question or ''},
            'notes': [self._public_note(note) for note in selected[:MAX_NOTES]],
            'omitted_count': max(0, len(selected) - MAX_NOTES),
            'authority': 'UNVERIFIED_MODEL_NOTE; host facts and receipts remain authoritative',
        }

    @staticmethod
    def _public_note(note: dict) -> dict:
        return {key: copy.deepcopy(value) for key, value in note.items()
                if key not in {'binding', 'dependency_snapshot', 'declared_dependency_revision'}}

    def render_notebook(self, *, question: str | None = None) -> tuple[str, dict]:
        """Render the volatile active tail and report explicit archive omissions."""
        from kernel_working_notebook import approximate_tokens, budget_tokens, render_header
        active = [note for note in self.notes.values()
                  if note.get('status') in {'active', 'stale-check'} and note.get('text')]
        terms = set((question or '').casefold().split())

        def rank(note: dict) -> tuple[int, int, str]:
            haystack = ' '.join((str(note.get('text', '')), str(note.get('kind', '')),
                                 str(note.get('scope', '')), str(note.get('refs', '')))).casefold()
            return (sum(term in haystack for term in terms), int(note.get('updated_step', 0)),
                    str(note.get('id', '')))

        active.sort(key=rank, reverse=True)
        budget = budget_tokens()
        lines = [render_header()]
        used = approximate_tokens(lines[0])
        shown = []
        omitted = 0
        for note in active:
            scope = note.get('scope', {})
            scope_text = (f"{scope.get('kind')}:{scope.get('id')}"
                          if isinstance(scope, Mapping) else str(scope))
            line = (f"- {note['id']} v{note['version']} [{scope_text}] "
                    f"{note['kind']} status={note['status']}: {note['text']}")
            cost = approximate_tokens(line)
            if used + cost > budget:
                omitted += 1
                self._archive_note(note, reason='active-budget-overflow')
                continue
            lines.append(line)
            shown.append(note['id'])
            used += cost
        omitted += len(self.note_archive)
        if omitted:
            lines.append(
                f"ARCHIVE omitted_count={omitted}; retrieve by id/question with note_retrieve."
            )
        return '\n'.join(lines), {
            'schema': 'working-notebook/1', 'shown_ids': shown,
            'omitted_count': omitted, 'estimated_tokens': approximate_tokens('\n'.join(lines)),
            'budget_tokens': budget,
        }

    def _completed(self, item: dict, binding: dict) -> bool:
        if item['status'] == 'superseded':
            return self._completed(self.items[item['superseded_by']], binding)
        return item['status'] == 'completed' and item['binding'] == binding

    def _handoff_ready(self, item: dict) -> bool:
        if item['status'] == 'superseded':
            return self._handoff_ready(self.items[item['superseded_by']])
        return item['status'] in {'ready', 'completed'} and all(
            self._handoff_ready(self.items[key]) for key in item['depends_on'])

    def completion_blockers(self, binding: dict, *, public_verified: bool) -> list[str]:
        return [key for key, item in self.items.items()
                if not public_verified or not self._completed(item, binding)]

    def public(self, binding: dict, records: list[dict], *, public_verified: bool) -> dict:
        def visible(item: dict) -> dict:
            value = {key: copy.deepcopy(val) for key, val in item.items() if key != 'binding'}
            value['revision_relevance'] = 'current' if item['binding'] == binding else 'historical'
            if 'depends_on' in item:
                value['prerequisites_ready'] = all(self._handoff_ready(self.items[key])
                                                   for key in item['depends_on'])
            return value
        counts: dict[tuple[str, str], int] = {}
        unchanged_counts: dict[tuple[str, str, str], int] = {}
        for record in records:
            key = (record['tool'], record['arguments_hash'])
            counts[key] = counts.get(key, 0) + 1
            unchanged_key = (*key, record.get('before_binding_digest')
                             or digest(record.get('before_binding', {})))
            unchanged_counts[unchanged_key] = unchanged_counts.get(unchanged_key, 0) + 1
        trajectory = [{'step': record['step'], 'tool': record['tool'], 'result': record['result'],
                       'repeat_count': counts[(record['tool'], record['arguments_hash'])],
                       'same_workspace_repeat_count': unchanged_counts[(record['tool'],
                            record['arguments_hash'], record.get('before_binding_digest')
                            or digest(record.get('before_binding', {})))],
                       'active_work_id': record.get('active_work_id'),
                       'intent': record['intent']['text']}
                      for record in records[-5:]]
        result = {'schema': 'decision-state/1', 'sequence': self.sequence,
            'authority': 'Model-authored plans and notes are unverified. Only host receipts establish execution. '
                         'Completed work remains a model claim; public checks do not prove arbitrary acceptance text.',
            'work_items': [visible(item) for item in self.items.values()],
            'notes': [visible(item) for item in self.notes.values() if item['text']],
            'note_versions': {key: item['version'] for key, item in self.notes.items()},
            'completion_blockers': self.completion_blockers(binding, public_verified=public_verified),
            'current_public_checks_passed': public_verified, 'recent_trajectory': trajectory,
            'capacity': {'work_items': MAX_ITEMS, 'notes': MAX_NOTES},
            'instructions': ('Notes are optional unverified working judgments. Revise or retire them with note_update; '
                             'direct actions and done remain available without notes.' if notes_only() else
                             'Continue the active item. Use next_action and acceptance to stay on track. '
                            'Use ready for an unverified handoff when later work is needed before tests can pass. '
                            'Revise a decision when evidence changes; do not recreate it each step. '
                            'State tools consume decisions; execute useful work between updates. '
                            'Blocked means missing information or capability, not successful completion.')}
        if notes_enabled():
            result['note_archive'] = [self._public_note(item) for item in self.note_archive.values()]
        return result
