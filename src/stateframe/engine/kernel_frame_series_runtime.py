"""Live adapter for the opt-in persistent progress cursor.

Only host receipts and bindings enter the cursor. Model-visible restrictions
describe exact repeated actions; dispatch enforces them without claiming success.
"""
from __future__ import annotations

import os
from pathlib import Path

from kernel_frame_series import (
    FrameSeries, canonical_argument_digest, default_control_enabled, default_enabled,
)
from kernel_measurement import digest, profiled_function_span


def configuration() -> tuple[bool, bool]:
    envelope, control = default_enabled(), default_control_enabled()
    if control and not envelope:
        raise ValueError('frame-series control requires the frame-series envelope')
    return envelope, control


def public_identity_mode() -> str:
    mode = os.environ.get('KERNEL_FRAME_SERIES_PUBLIC_ID', 'run')
    if mode not in {'run', 'objective'}:
        raise ValueError('KERNEL_FRAME_SERIES_PUBLIC_ID must be run or objective')
    return mode


def _stagnation_cap(objective_revision: str):
    """Load the optional cap only after its switch is explicitly enabled."""
    raw = str(os.environ.get("KERNEL_STAGNATION_CAP", "off") or "off").strip().lower()
    if raw in {"0", "false", "off"}:
        return None
    from kernel_stagnation_cap import StagnationCap, enabled as stagnation_cap_enabled
    return (StagnationCap(objective_revision, enabled=True)
            if stagnation_cap_enabled() else None)


class LiveFrameSeries:
    def __init__(self, runtime):
        self.runtime = runtime
        self.public_identity = public_identity_mode()
        owner = runtime.owner
        self.path = ""
        if runtime.target is not None:
            self.path = runtime.target.resolve().relative_to(
                Path(owner.root).resolve()).as_posix()
        self.public_checks = tuple(selector for selector in getattr(owner, 'f2p_inworld', ())
                                   if (Path(owner.root) / selector.split('::')[0]).is_file())
        plan = getattr(owner, 'unit_plan', None)
        objective_revision = digest({'path': self.path, 'symbol': owner.fn,
            'public_requirements': getattr(owner, 'task_row', {}).get('msg', '')})
        self.cursor = FrameSeries(task_id=owner.task_id, path=self.path, symbol=owner.fn,
            scope={'id': runtime.journal.run_id},
            plan={'digest': plan.plan_digest} if plan is not None else None,
            objective_revision=objective_revision,
            parent_budget=runtime.parent, enabled=True, control_enabled=default_control_enabled())
        self.restrictions: list[dict] = []
        self.stagnation_cap = _stagnation_cap(objective_revision)

    def _missing(self) -> list[dict]:
        items = []
        if self.runtime.target is None:
            items.append({'code': 'target_not_bound', 'kind': 'localization',
                          'summary': 'No working source is selected; search, read, or localize a product path before editing.'})
        elif not self.runtime.target.is_file():
            items.append({'code': 'target_source_absent', 'kind': 'source',
                          'summary': 'The target file is absent; recreate or locate it before verification.'})
        for selector in self.public_checks:
            if not (Path(self.runtime.owner.root) / selector.split('::')[0]).is_file():
                items.append({'code': 'public_check_absent', 'kind': 'verification',
                              'summary': str(selector)})
        return items[:8]

    def bind_target(self, path: str, symbol: str = "") -> None:
        """Move the cursor from workspace orientation to an admitted source."""
        self.path = path.replace("\\", "/").lstrip("./")
        self.cursor.scope["path"] = self.path
        self.cursor.scope["symbol"] = str(symbol or "")
        self.runtime.journal.emit(
            "frame_series_target_bound", path=self.path,
            symbol=str(symbol or ""), dispatch=True,
        )

    def close_step(self, record: dict) -> None:
        runtime = self.runtime
        proof = None
        if record['tool'] == 'run_tests':
            # Select first, then validate: uncertainty must revoke older proof.
            receipt = next((item for item in reversed(runtime.test_receipts)
                            if item.get('step') == record['step']), None)
            if receipt is not None:
                valid = (receipt.get('binding_stable') is True
                         and type(receipt.get('exit_code')) is int)
                proof = {'host_observed': valid, 'receipt_id': receipt.get('receipt_id', ''),
                         'binding': receipt.get('binding', {}),
                         'selectors': receipt.get('selectors', []),
                         'result': ('GREEN' if receipt['exit_code'] == 0 else
                                    'RED' if receipt['exit_code'] == 1 else 'ERROR')
                                   if valid else 'ERROR'}
            elif record['result'] in {'GREEN', 'RED', 'ERROR', 'TIMEOUT', 'NO_VERDICT'}:
                # An execution outcome without its receipt cannot preserve authority.
                proof = {'host_observed': False, 'result': 'ERROR'}
        missing = self._missing()
        if record['tool'] == 'done' and record['result'] == 'ACCEPTED':
            phase, reason = 'stopped', 'Worker accepted a stop request; private correctness remains unknown.'
        elif missing:
            phase, reason = 'blocked', 'Required public workspace input is absent.'
        elif runtime.bound_public_verification():
            phase, reason = 'reviewing', 'Required public checks are current; review all stated requirements.'
        elif record['result'] in {'RED', 'ERROR', 'TIMEOUT', 'SELECTOR_ERROR', 'NO_TESTS', 'NO_VERDICT'}:
            phase, reason = 'repairing', 'Use the recorded failure to change the implementation or approach.'
        elif record['changes']:
            phase, reason = 'verifying', 'Workspace changed; prior verification no longer covers this revision.'
        else:
            phase = self.cursor.public()['phase']['id']
            if phase == 'orienting':
                phase = 'implementing'
            reason = 'Continue the same objective using the supplied current evidence.'
        state = runtime.decision_state
        version = digest(state.items) if state is not None else None
        # Preserve historical inputs: replay must not consult the final workspace.
        fold_inputs = {'record': record, 'binding': record['after_binding'], 'proof': proof,
                       'decision_state_version': version, 'missing_info': missing,
                       'phase': phase, 'phase_reason': reason,
                       'phase_evidence_ref': record['step_record_id']}
        event = self.cursor.fold(**fold_inputs)
        runtime.journal.emit('frame_series_folded', event=event, cursor=self.cursor.public(),
                             fold_inputs=fold_inputs)
        stagnation_cap = getattr(self, 'stagnation_cap', None)
        if stagnation_cap is not None:
            cap = stagnation_cap.observe(
                {'tool': record.get('tool'),
                 'meaningful_progress': event.get('meaningful_progress') is True,
                 'legitimate_recovery': record.get('legitimate_recovery') is True,
                 'sequence': event.get('sequence')},
                record=record, binding=record['after_binding'])
            runtime.journal.emit('stagnation_cap_observed', cap=cap,
                                 step=record.get('step'), dispatch=False)

    @profiled_function_span("frame_series_restrictions_projection")
    def prepare_restrictions(self) -> list[dict]:
        current = digest(self.runtime.binding())
        self.restrictions = [item for item in self.cursor.restricted_argument_identities()
                             if item['binding_digest'] == current]
        self.runtime.journal.emit('frame_series_restrictions', restrictions=self.restrictions,
                                 step=getattr(self.runtime.owner, 'current_step', 0))
        return self.restrictions

    def before_dispatch(self, tool: str, arguments: dict) -> str | None:
        stagnation_cap = getattr(self, 'stagnation_cap', None)
        if stagnation_cap is not None:
            feedback = stagnation_cap.before_dispatch(
                tool, binding=self.runtime.binding())
            if feedback is not None:
                self.runtime.journal.emit(
                    'stagnation_cap_blocked', tool=tool,
                    cap=stagnation_cap.public(), dispatch=False)
                return feedback
        identity = (tool, canonical_argument_digest(tool, arguments), digest(self.runtime.binding()))
        blocked = any(identity == (item['tool'], item['arguments_hash'], item['binding_digest'])
                      for item in self.restrictions)
        self.runtime.journal.emit('frame_series_dispatch', tool=tool, arguments_hash=identity[1],
                                 binding_digest=identity[2], blocked=blocked)
        if not blocked:
            return None
        return ('[frame-series] This exact action already produced evidence on the unchanged workspace '
                'and is paused for this decision. Use the retained result. Choose a different useful '
                'action, investigate a specific missing requirement, or request done if satisfied. '
                'Different arguments and changed workspace revisions remain available.')

    def model_view(self) -> dict:
        view = self.cursor.public()
        current = self.runtime.binding()
        proof_status = view['proof']['status']
        if proof_status != 'none' and view['binding']['digest'] != digest(current):
            proof_status = 'stale_binding'
        records = self.runtime.step_ledger.records[-6:]
        current_digest = digest(current)
        recent = [{'step': row['step'], 'tool': row['tool'], 'parameters': row['action_parameters'],
                   'result': row['result'], 'current_workspace': (
                       row['after_binding_digest'] == current_digest
                       if 'after_binding_digest' in row else row['after_binding'] == current)}
                  for row in records]
        restricted = []
        for item in self.restrictions:
            row = next((row for row in reversed(self.runtime.step_ledger.records)
                        if row['tool'] == item['tool']
                        and canonical_argument_digest(
                            row['tool'], row.get('action_parameters'),
                            fallback=row.get('arguments_hash'),
                        ) == item['arguments_hash']
                        and (row['after_binding_digest'] if 'after_binding_digest' in row
                             else digest(row['after_binding'])) == item['binding_digest']), None)
            restricted.append({'tool': item['tool'],
                'parameters': row['action_parameters'] if row is not None else {},
                'arguments_hash': item['arguments_hash'], 'reason': item['reason']})
        transitions = view['phase_transitions']
        phase = dict(view['phase'])
        if self.public_identity == 'objective' and transitions:
            # Host transition references bind run-specific receipts. Public references
            # identify only this objective's ordered phase history, never proof.
            last = transitions[-1]
            phase['latest_transition_ref'] = 'transition:' + digest({
                'objective_revision': view['objective_revision'],
                'sequence': last['sequence'], 'from_phase': last['from_phase'],
                'to_phase': last['to_phase'], 'reason': last['reason'],
            })
        public_id = view['series_id'] if self.public_identity == 'run' else 'objective:' + view['objective_revision']
        result = {'series_id': public_id, 'objective_revision': view['objective_revision'],
                'continuation': 'Continue this existing objective; a new frame is not a new task.',
                'scope': {'path': self.path, 'symbol': self.runtime.owner.fn},
                'phase': {**phase, 'reason': transitions[-1]['reason'] if transitions else
                          'Use the current source and requirements to begin a testable change.'},
                'progress': {key: view['progress'][key] for key in
                             ('actions', 'meaningful', 'no_progress_cycles', 'repeated_actions')},
                'recent_actions': recent, 'missing_information': self._missing(),
                'exact_actions_paused_this_decision': restricted,
                'proof_status': proof_status,
                'proof_result': view['proof'].get('result'),
                'completion_authority': 'Existing public completion gate; this cursor never declares success.'}
        stagnation_cap = getattr(self, 'stagnation_cap', None)
        if stagnation_cap is not None:
            result['stagnation_cap'] = stagnation_cap.public()
        return result
