"""Opt-in live adapter for world-owned retained source and graph-linked recall."""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path


def enabled() -> bool:
    value = os.environ.get('KERNEL_RETAINED_CONTEXT', '0')
    if value not in {'0', '1'}:
        raise ValueError('KERNEL_RETAINED_CONTEXT must be 0 or 1')
    return value == '1'


def delivery_enabled() -> bool:
    value = os.environ.get('KERNEL_RETAINED_DELIVERY', '0')
    if value not in {'0', '1'}:
        raise ValueError('KERNEL_RETAINED_DELIVERY must be 0 or 1')
    return value == '1'


def read_focus_enabled() -> bool:
    value = os.environ.get('KERNEL_RETAINED_READ_FOCUS', '0')
    if value not in {'0', '1'}:
        raise ValueError('KERNEL_RETAINED_READ_FOCUS must be 0 or 1')
    return value == '1'


def edit_focus_enabled() -> bool:
    value = os.environ.get("KERNEL_RETAINED_EDIT_FOCUS", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_RETAINED_EDIT_FOCUS must be 0 or 1")
    return value == "1"


def tool_surface(base: list[dict]) -> list[dict]:
    if not enabled() or any(t['function']['name'] == 'recall_evidence' for t in base):
        return base
    result = [*base, {'type': 'function', 'function': {
        'name': 'recall_evidence',
        'description': 'Retrieve already retained source by its reference or exact known path, without searching. '
                       'Response labels current versus historical content. Optional symbol is a qualified Python name.',
        'parameters': {'type': 'object', 'properties': {
            'reference': {'type': 'string'}, 'line': {'type': 'integer', 'minimum': 1},
            'symbol': {'type': 'string'}, 'why': {'type': 'string'}},
            'required': ['reference'], 'additionalProperties': False}}}]
    if delivery_enabled():
        function = result[-1]['function']
        function['parameters']['properties']['column'] = {
            'type': 'integer', 'minimum': 0,
            'description': 'Zero-based character offset on line; use returned continuation line/column without symbol.'}
        function['description'] += ' Follow continuation line/column to retrieve omitted text; explicit recall is delivered in the next frame.'
    return result



def hashes(binding: dict) -> dict:
    return {**binding.get('product_files', {}), **binding.get('harness_files', {})}


_JOURNAL_SINK_MARKER = "_stateframe_retained_journal_sink"


def _bind_world_journal_sink(world, journal) -> None:
    """Bind one journal adapter without stacking callbacks across runtime setup."""
    current = getattr(world, "event_sink", None)
    if (getattr(current, _JOURNAL_SINK_MARKER, False)
            and getattr(current, "_stateframe_journal", None) is journal):
        return

    previous = current
    seen: set[int] = set()
    while callable(previous) and getattr(previous, _JOURNAL_SINK_MARKER, False):
        if id(previous) in seen:
            raise ValueError("retained journal sink chain contains a cycle")
        seen.add(id(previous))
        previous = getattr(previous, "_stateframe_previous_sink", None)

    def persist(event):
        from kernel_measurement import profile_active_host_span

        with profile_active_host_span("retained_context_subscribers"):
            journal.emit('world_knowledge_event', event=event)
            if previous is not None:
                previous(event) if callable(previous) else previous.append(event)

    setattr(persist, _JOURNAL_SINK_MARKER, True)
    setattr(persist, "_stateframe_journal", journal)
    setattr(persist, "_stateframe_previous_sink", previous)
    world.event_sink = persist


class RetainedContext:
    def __init__(self, owner, journal, *, scope=None):
        self.world = getattr(owner, 'world', None)
        if self.world is None:
            raise ValueError('retained context requires the world knowledge store')
        self.owner, self.journal = owner, journal
        self.context_scope = scope
        self._context_scope_error = None
        if self.context_scope is not None:
            from kernel_context_scope import ContextScopeError
            self._context_scope_error = ContextScopeError
        previous = self.world.event_sink
        try:
            self.world.enable_source_archive(journal.run_id)
            self.archive = self.world.source_archive
            if self.context_scope is not None:
                self.context_scope.bind_archive(self.archive)
        except BaseException:
            self.world.event_sink = previous
            raise
        _bind_world_journal_sink(self.world, journal)
        from kernel_retained_public import PublicSourceReferences, public_identity_mode
        self.public_references = PublicSourceReferences(self.archive, public_identity_mode())
        self.delivery = delivery_enabled()
        from kernel_lossless_store_paging import enabled as lossless_paging_enabled
        self.lossless_paging = lossless_paging_enabled()
        if self.lossless_paging and not self.delivery:
            raise ValueError('lossless store paging requires retained delivery')
        self.read_focus = read_focus_enabled()
        if self.read_focus and not self.delivery:
            raise ValueError('retained read focus requires retained delivery')
        from kernel_read_working_set import ReadWorkingSet, capacity
        size = capacity()
        if size and not (self.delivery and self.read_focus):
            raise ValueError('read working set requires retained delivery and read focus')
        self.working_set = ReadWorkingSet(size, journal.emit) if size else None
        self.edit_focus = edit_focus_enabled()
        if self.edit_focus and self.working_set is None:
            raise ValueError("retained edit focus requires a bounded read working set")

        self.requested_column = 0
        self.requested = ''
        self.requested_start_line = 1
        self.requested_symbol = ''

    def _guard_archive(self) -> None:
        if self.context_scope is not None:
            self.context_scope.assert_archive(self.archive)

    def checkpoint_snapshot(self) -> dict:
        self._guard_archive()
        from kernel_retained_restart import snapshot
        return snapshot(self)

    def restore_snapshot(self, saved: dict, records: list[dict], boundary: int) -> None:
        self._guard_archive()
        from kernel_retained_restart import restore
        restore(self, saved, records, boundary)

    def observe_source(self, path: str, content: bytes, evidence_ref: str) -> str | None:
        from detreason.source_archive import NonProductSourceError, SourceArchiveCapacityError
        try:
            self._guard_archive()
            return self.archive.retain(path, content, evidence_ref)
        except SourceArchiveCapacityError as error:
            # Audit 2026-10-05 #2: a full archive is journaled, never fatal
            # Report retained-context failures without treating them as task success.
            self.journal.emit('retained_context_error', operation='retain',
                              error_type=type(error).__name__, capacity_exhausted=True,
                              path=path, bytes=len(content) if isinstance(content, bytes) else None)
            return None
        except NonProductSourceError as error:
            self.journal.emit('retained_context_error', operation='retain',
                              error_type=type(error).__name__, non_product=True, path=path)
            return None
        except Exception as error:
            self.journal.emit('retained_context_error', operation='retain', error_type=type(error).__name__)
            raise

    def invalidate_changes(self, before: dict, after: dict, evidence_ref: str) -> list[str]:
        self._guard_archive()
        old, new = hashes(before), hashes(after)
        changed = [path for path in set(old) | set(new) if old.get(path) != new.get(path)]
        if not changed:
            return []
        if self.working_set is not None:
            self.working_set.invalidate(new)
        previous = self.world.current_revision
        revision = max(previous + 1, getattr(self.owner, 'workspace_revision', 0) + 1)
        for path in sorted(changed):
            self.world.invalidate_edit(path=path, before_revision=previous, after_revision=revision,
                before_digest=old.get(path, 'MISSING'), after_digest=new.get(path, 'MISSING'),
                evidence_ref=evidence_ref)
        return sorted(changed)

    def refresh_changes(self, changed: list[str], binding: dict, evidence_ref: str) -> None:
        self._guard_archive()
        root = Path(self.owner.root).resolve()
        from kernel_non_product_paths import is_non_product
        for relative in changed:
            if is_non_product(relative):  # coverage/caches are not changed source (audit #2)
                continue
            path = root / relative
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            information = getattr(self.owner, 'information_policy', None)
            if information is not None and not information.check_path(root, str(path)):
                continue
            raw = path.read_bytes()
            if 'sha256:' + hashlib.sha256(raw).hexdigest() != hashes(binding).get(relative):
                self.journal.emit('retained_refresh_raced', path=relative)
                continue
            self.observe_source(relative, raw, evidence_ref)

    def close_step(self, record: dict, decision_state) -> None:
        self._guard_archive()
        work_id = record.get('active_work_id') or ''
        item = decision_state.items.get(work_id, {}) if decision_state is not None else {}
        self.archive.bind_step(record, work_id=work_id, work_version=item.get('version', 0))
        if (self.delivery and record.get('tool') != 'recall_evidence'
                and record.get('result') not in {'REJECTED_CONTRACT', 'DENIED_PATH'}):
            self.requested = ''
            self.requested_start_line, self.requested_column, self.requested_symbol = 1, 0, ''
        if self.read_focus and record.get('tool') == 'read' and record.get('result') == 'OK':
            self.focus_observed_read(record)
        if self.edit_focus:
            self.focus_applied_edit(record)



    def focus_applied_edit(self, record: dict) -> None:
        """Select current observed edit results; never refresh old test evidence."""
        self._guard_archive()
        if record.get("tool") not in {"edit", "str_replace", "file_create"} or record.get("result") != "APPLIED":
            return
        before = record.get("before_binding", {}).get("product_files", {})
        after = record.get("after_binding", {}).get("product_files", {})
        for path, revision in sorted(after.items()):
            if before.get(path) == revision:
                continue
            reference = next((ref for ref, item in self.archive.revisions.items()
                              if item["path"] == path and item["revision"] == revision), None)
            if reference is None:
                self.journal.emit("retained_edit_focus_unavailable", path=path,
                                  step_record_id=record["step_record_id"],
                                  reason="post_edit_source_not_archived")
                continue
            page = self.archive.recall(reference, hashes(record["after_binding"]), continuation=True)
            self.working_set.remember(page)
            self.working_set.focus_edit(path, revision)
            self.journal.emit("retained_edit_focused", path=path, revision=revision,
                              reference=reference, step_record_id=record["step_record_id"])

    def focus_observed_read(self, record: dict) -> None:
        """Deliver the last successful read from its exact observed archive revision."""
        self._guard_archive()
        evidence = record.get('read_evidence', {})
        path, revision = evidence.get('path'), evidence.get('file_revision')
        line = evidence.get('start_line')
        reference = next((ref for ref, item in self.archive.revisions.items()
                          if item['path'] == path and item['revision'] == revision), None)
        if (reference is None or type(line) is not int or line < 1
                or evidence.get('producer') != 'workspace_read'
                or evidence.get('authority') != 'HOST_OBSERVED_READ'
                or hashes(record.get('after_binding', {})).get(path) != revision):
            self.journal.emit('retained_read_focus_unavailable',
                              step_record_id=record['step_record_id'],
                              reason='missing_or_unbound_observed_source')
            return
        self.requested = reference
        self.requested_start_line, self.requested_column, self.requested_symbol = line, 0, ''
        if self.working_set is not None:
            self.working_set.remember(self.archive.recall(reference,
                hashes(record['after_binding']), start_line=line, continuation=True))
        self.journal.emit('retained_read_focused', reference=reference, path=path,
                          revision=revision, start_line=line,
                          step_record_id=record['step_record_id'])



    def view(self, binding: dict, working_path: str, decision_state, *,
             working_content: str | None = None) -> dict:
        self._guard_archive()
        active = next((item for item in decision_state.items.values() if item['status'] == 'active'), {}) \
            if decision_state is not None else {}
        view = self.archive.select(hashes(binding), working_path, work_id=active.get('id', ''),
                                   work_version=active.get('version', 0), requested=self.requested,
                                   requested_start_line=self.requested_start_line,
                                   requested_symbol=self.requested_symbol,
                                   stable_public_order=self.public_references.mode == 'semantic',
                                   deliver_requested=self.delivery, requested_column=self.requested_column)
        if self.working_set is not None:
            required, state = self.working_set.select(self.archive, hashes(binding),
                                                      working_path, working_content)
            covered = {(item['path'], item['revision']) for item in state['active']}
            # Required spans replace shorter automatic copies, not one another.
            # An explicit different historical recall remains independently deliverable.
            extras = [item for item in view['selected']
                      if (item['path'], item['revision']) not in covered
                      or (item.get('requested_delivery') and not item['content'])]
            view['selected'] = [*required, *extras]
            view['read_working_set'] = state
        if self.lossless_paging:
            from kernel_lossless_store_paging import attach_index
            view = attach_index(view, self.archive, hashes(binding))
        return view

    def recall(self, args: dict, binding: dict) -> str:
        requested_reference = args['reference']
        try:
            self._guard_archive()
            if self.lossless_paging:
                from kernel_lossless_store_paging import (
                    STABLE_REFERENCE_PREFIX,
                    cold_page,
                    resolve_stable_reference,
                )
                if (isinstance(requested_reference, str)
                        and requested_reference.startswith(STABLE_REFERENCE_PREFIX)):
                    reference = resolve_stable_reference(self.archive, requested_reference)
                else:
                    reference = self.public_references.resolve(requested_reference)
                result = cold_page(self.archive, reference, hashes(binding),
                                   start_line=args.get('line', 1),
                                   symbol=args.get('symbol', ''),
                                   start_column=args.get('column', 0))
            else:
                reference = self.public_references.resolve(requested_reference)
                result = self.archive.recall(reference, hashes(binding), start_line=args.get('line', 1),
                                             symbol=args.get('symbol', ''), start_column=args.get('column', 0),
                                             continuation=self.delivery)
        except (KeyError, ValueError, SyntaxError, TypeError) as error:
            self.journal.emit('retained_recall_rejected', error_type=type(error).__name__)
            return 'Recall unavailable: use an exact retained reference/path and a valid range or symbol.'
        except Exception as error:
            if (self._context_scope_error is None
                    or not isinstance(error, self._context_scope_error)):
                raise
            self.journal.emit('retained_recall_rejected', error_type=type(error).__name__)
            return 'Recall unavailable: the retained evidence belongs to another context scope.'
        if self.working_set is not None:
            self.working_set.remember(result)
        self.requested = reference
        self.requested_column = args.get('column', 0)
        self.requested_start_line = args.get('line', 1)
        self.requested_symbol = args.get('symbol', '')
        self.journal.emit('retained_recall', reference=reference, freshness=result['freshness'],
                          start_line=result['start_line'], selected_end_line=result['selected_end_line'],
                          selection={'reference': reference, 'line': self.requested_start_line,
                                     'column': self.requested_column, 'symbol': self.requested_symbol})
        return json.dumps(result, ensure_ascii=False)
