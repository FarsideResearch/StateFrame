"""Once-per-attempt public review with isolated evidence and bounded reruns."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path

from kernel_effect_boundary import Mode
from kernel_measurement import RequestCapture, digest
from kernel_public_probe import DESIGN, REVIEW, SCHEMA, public_bundle
from kernel_public_contract import PublicContract, configured_profile
from kernel_public_probe_execution import execute_probe


def _journal_safe_fields(payload):
    """Emit boundary (10-05 L109): rename journal-RESERVED keys; see kernel_continuation_journal."""
    from kernel_continuation_journal import journal_safe_fields

    return journal_safe_fields(payload)


class PublicReview:
    def __init__(self, runtime):
        self.runtime = runtime
        from kernel_review_deadline import ReviewDeadline
        self.deadline_policy = ReviewDeadline.configured()
        self.contract_profile = configured_profile()
        self.contract = None
        self.citation_provenance = None
        self.attempted = False
        self.proposal = None
        self.reference = None
        self.executions = 0
        self.seen_bindings = set()
        self.last_receipt = None
        self.exposed = False
        # Fix (post-review r44, finding 1): callers need to distinguish a
        # hard PROBE_RED refusal from a PROBE_INCONCLUSIVE diagnostic-only
        # card -- the latter must never become completion evidence, but must
        # also never itself refuse an otherwise-admissible `done`.
        self.last_exposed_result = None
        # These fields are populated only by the explicit validity arm.  The
        # default path below keeps the historical proposal/receipt shape.
        self._probe_validity = None
        self._public_probe_row = None
        self.max_tokens = int(os.environ.get('KERNEL_PUBLIC_REVIEW_TOKENS', '24000'))
        if not 1 <= self.max_tokens <= 24000:
            raise ValueError('public review output allowance must be 1..24000')

    @staticmethod
    def _validity_enabled() -> bool:
        from kernel_probe_validity_gate import enabled
        return enabled()

    def _public_test_material(self) -> dict[str, str]:
        """Collect only declared public test text for the cross-check."""
        row = self._public_probe_row
        if not isinstance(row, Mapping):
            row = getattr(self.runtime.owner, 'task_row', {})
        if not isinstance(row, Mapping):
            return {}
        sources: dict[str, str] = {}
        try:
            from kernel_public_evidence import public_test_sources
            sources.update(public_test_sources(dict(row)))
        except (ImportError, TypeError, ValueError):
            # A legacy single-file row may have no declared test path.  The
            # explicit test_src is still public material, never an oracle.
            pass
        for key in ('public_test_source', 'public_test_src', 'test_source', 'test_src'):
            value = row.get(key)
            if isinstance(value, str) and value.strip() and value not in sources.values():
                sources[key] = value
        return sources

    def _assess_validity(self, *, validation_error: object = None):
        if not self._validity_enabled():
            return None
        from kernel_probe_validity_gate import assess_probe
        self._probe_validity = assess_probe(
            self.proposal, public_tests=self._public_test_material(),
            validation_error=validation_error)
        return self._probe_validity

    def _fallback_proposal(self, value: object) -> dict | None:
        """Keep a bounded invalid proposal inspectable without executing it."""
        if not isinstance(value, Mapping):
            return None
        source = value.get('test_source')
        if not isinstance(source, str) or len(source.encode()) > 8000:
            return None
        if self.contract_profile == 'referenced_suite' and self.contract is not None:
            try:
                candidate = self.contract._canonical(value)
            except (TypeError, ValueError, KeyError):
                candidate = None
            if isinstance(candidate, Mapping):
                value = candidate
        quotes = value.get('requirement_quotes', [])
        uncertainties = value.get('uncertainties', [])
        if not isinstance(quotes, list):
            quotes = []
        if not isinstance(uncertainties, list):
            uncertainties = []
        return {
            'test_source': source,
            'requirement_quotes': [item for item in quotes if isinstance(item, str)][:8],
            'uncertainties': [item for item in uncertainties if isinstance(item, str)][:8],
        }

    def _charge(self, kind: str) -> None:
        from kernel_pilot import ParentBudgetExhausted
        from kernel_step_limits import reached
        parent = self.runtime.parent
        # The sidecar itself is bounded even when the pilot is only measuring.
        if reached(parent.limits[kind], parent.used[kind]) or reached(parent.limits['total'], parent.used['total']):
            raise ParentBudgetExhausted(kind)
        parent.charge(kind, reason='public review')

    def _model(self, messages: list[dict], purpose: str, endpoint: str, key: str) -> dict:
        from kernel_step_contract import configure_reasoning_budget
        from kernel_prompt_cache import configure_pinned_slot
        runtime = self.runtime
        capture = RequestCapture(
            runtime.journal, runtime.owner.current_step, runtime.snapshot_id,
            call_purpose=purpose)
        from kernel_request_kind import bind_auxiliary
        bind_auxiliary(capture, "public_review")
        native = getattr(runtime, 'native_batch', None)
        dispatch_id = getattr(runtime, 'current_request_id', None)
        runtime.journal.emit('public_review_child', purpose=purpose, request_id=capture.request_id,
                             parent_request_id=native.origin if native is not None else dispatch_id,
                             dispatch_request_id=dispatch_id, stage='allocated')
        body = {'model': runtime.journal.manifest['model'], 'messages': messages,
                'temperature': runtime.journal.manifest['temperature'], 'max_tokens': self.max_tokens,
                'cache_prompt': False, 'response_format': {'type': 'json_schema', 'json_schema': {
                    'name': 'public_probe', 'strict': True,
                    'schema': self.contract.schema if self.contract is not None else SCHEMA}}}
        configure_reasoning_budget(body)
        for setting, field in [('REACT_SEED', 'seed'), ('REACT_SLOT_ID', 'id_slot')]:
            if setting in os.environ:
                value = int(os.environ[setting])
                if field == 'id_slot':
                    configure_pinned_slot(body, value)
                else:
                    body[field] = value
        self._charge('model_calls')
        # One transport attempt per stage; no silent retries or partial output dispatch.
        if self.deadline_policy is None:
            choice = capture.send(body, endpoint, key)
        else:
            choice = capture.send(body, endpoint, key,
                                  deadline=self.deadline_policy.call_deadline())
        if choice.get('finish_reason') != 'stop' or choice.get('message', {}).get('tool_calls'):
            raise ValueError('public review response incomplete or contains actions')
        return json.loads(choice['message']['content'])

    def _execute(self) -> dict:
        runtime = self.runtime
        binding = runtime.binding()
        identity = digest(binding)
        if self.executions >= 2 or identity in self.seen_bindings:
            raise ValueError('probe permits initial execution plus one changed-binding rerun')
        self._charge('tool_work')
        self.executions += 1
        self.seen_bindings.add(identity)
        options = {} if self.deadline_policy is None else {
            'deadline': min(self.deadline_policy.deadline,
                            time.monotonic() + self.deadline_policy.probe_seconds)}
        validity = self._probe_validity if self._validity_enabled() else None
        probe_source = self.proposal['test_source']
        if validity is not None and not validity.evidence_allowed:
            probe_source = validity.salvaged_source
        if validity is not None and not validity.evidence_allowed and not probe_source:
            result = {
                'result': 'PROBE_INCONCLUSIVE', 'exit_code': None, 'timeout': False,
                'output': 'probe quarantined before execution: no legal public operation',
                'omitted_output_bytes': 0, 'candidate_file_hashes': binding.get('product_files', {}),
                'authority': 'HOST_VALIDITY_GATE',
            }
            runtime.journal.emit('public_probe_quarantined',
                                 reason_codes=list(validity.reason_codes),
                                 validity=validity.to_dict(), probe_reference=self.reference)
        else:
            result = execute_probe(Path(runtime.owner.root), binding['product_files'],
                                   probe_source or self.proposal['test_source'],
                                   runtime.journal.path.parent, **options)
            if validity is not None and not validity.evidence_allowed:
                # A salvaged no-exception run is diagnostic only.  Its exit
                # code can never become a product-violation signal.
                result = {**result, 'result': 'PROBE_INCONCLUSIVE',
                          'authority': 'HOST_VALIDITY_GATE'}
        if runtime.binding() != binding:
            raise ValueError('public probe evidence became stale during execution')
        receipt = {**result, 'binding': binding, 'probe_reference': self.reference,
                   'probe_hash': digest(self.proposal), 'execution_index': self.executions}
        if self.citation_provenance is not None:
            receipt['citation_provenance'] = self.citation_provenance
        if validity is not None:
            from kernel_probe_validity_gate import validity_digest
            receipt['validity'] = validity.to_dict()
            receipt['validity_digest'] = validity_digest(validity)
        runtime.journal.emit('public_probe_executed', **_journal_safe_fields(receipt))
        self.last_receipt = receipt
        return receipt

    def _card(self, receipt: dict) -> str:
        current_binding = self.runtime.binding()
        stale = current_binding != receipt['binding']
        validity = receipt.get('validity') if self._validity_enabled() else None
        inconclusive = isinstance(validity, Mapping) and validity.get('status') == 'INCONCLUSIVE'
        from kernel_probe_validity_gate import public_probe_header
        header = public_probe_header(inconclusive)
        applicability = (
            'STALE after workspace change: this historical execution is not evidence about '
            'the current product. The unchanged proposal remains inspectable and unverified.'
            if stale else
            ('The host validity gate quarantined this model-written probe as inconclusive. '
             'It is read-only diagnostic material, never a product violation or completion evidence.'
             if inconclusive else
             'This execution applies to the current product; the proposal and expectation '
             'remain unverified and passing never certifies completion.')
        )
        payload = {
            'authority': 'HOST_VALIDITY_GATE' if inconclusive else 'UNVERIFIED_MODEL_PROPOSAL',
            'reference': self.reference,
            'execution_applicability': applicability,
            'current_binding_digest': digest(current_binding),
            'instruction': (
                'This probe is quarantined by the host validity gate. Do not treat it as evidence, '
                'do not edit to satisfy it, and do not use its red result as a product violation. '
                'The legal-prefix observation is diagnostic only.' if inconclusive else
                'This check may be wrong. Compare its expectation with the quoted public '
                'requirements. Repair a supported discrepancy or reject an unsupported '
                'expectation; no edit is mandatory. Passing never certifies completion. '
                'After an edit you may run_tests with target ' + self.reference +
                ' once to rerun this unchanged probe. Its temporary filename is not a workspace selector.'),
            **({'citation_provenance': self.citation_provenance} if self.citation_provenance is not None else {}),
            'requirement_quotes': self.proposal['requirement_quotes'],
            'uncertainties': self.proposal['uncertainties'],
            'test_source': self.proposal['test_source'], 'execution': {
                **{key: receipt[key] for key in ('result', 'exit_code', 'timeout', 'output',
                                                 'omitted_output_bytes', 'candidate_file_hashes')},
                'binding_digest': digest(receipt['binding']),
                'freshness': 'STALE' if stale else 'CURRENT',
                'applicable_to_current_product': not stale},
        }
        if validity is not None:
            payload['validity'] = validity
        return header + '\n' + json.dumps(payload, sort_keys=True)

    def context(self) -> str | None:
        if not self.exposed or self.last_receipt is None:
            return None
        return self._card(self.last_receipt)

    def completion(self, row: dict, remaining: int | None, endpoint: str, key: str) -> str | None:
        runtime = self.runtime
        if self.attempted:
            return None
        if not runtime.bound_public_verification():
            runtime.journal.emit('public_review_unexposed', reason='no current public verification')
            return None
        native = getattr(runtime, 'native_batch', None)
        if (remaining is not None and remaining <= 0) or (native is not None and native.pending):
            runtime.journal.emit('public_review_unexposed', reason='no fresh decision available or nonfinal done')
            return None
        if self.deadline_policy is not None:
            admission = self.deadline_policy.admit(2)
            runtime.journal.emit('public_review_admission', stage='initial', **_journal_safe_fields(admission))
            if not admission['admitted']:
                runtime.journal.emit('public_review_unexposed', reason='wall_budget_unavailable')
                return None
        self.attempted = True
        self._public_probe_row = row
        validation_error = None
        try:
            bundle = public_bundle(row)
            self.contract = PublicContract(bundle, self.contract_profile, DESIGN, REVIEW)
            messages = [{'role': 'system', 'content': self.contract.design},
                        {'role': 'user', 'content': json.dumps(self.contract.bundle, sort_keys=True)}]
            generated = self.contract.draft(self._model(messages, 'generate', endpoint, key))
            reviewed = self._model([*messages, {'role': 'assistant', 'content': json.dumps(generated)},
                                   {'role': 'user', 'content': self.contract.review}], 'blind_review', endpoint, key)
            try:
                self.proposal = self.contract.final(reviewed)
            except Exception as error:
                if not self._validity_enabled():
                    raise
                self.proposal = self._fallback_proposal(reviewed)
                if self.proposal is None:
                    raise
                validation_error = error
            self.citation_provenance = self.contract.provenance(reviewed)
            validity = self._assess_validity(validation_error=validation_error)
            identity = self.proposal if self.citation_provenance is None else {
                'proposal': self.proposal, 'citation_provenance': self.citation_provenance}
            self.reference = 'public_probe:' + digest(identity).removeprefix('sha256:')
            reviewed_fields = {
                'proposal': self.proposal, 'reference': self.reference,
                'authority': 'UNVERIFIED_MODEL_PROPOSAL',
                'public_input_hash': digest(self.contract.bundle),
                'contract_profile': self.contract_profile,
                'citation_provenance': self.citation_provenance,
            }
            if validity is not None:
                reviewed_fields['validity'] = validity.to_dict()
            runtime.journal.emit('public_probe_reviewed', **reviewed_fields)
            if validity is not None:
                runtime.journal.emit('public_probe_validity', reference=self.reference,
                                     validity=validity.to_dict(),
                                     evidence_allowed=validity.evidence_allowed,
                                     quarantined=validity.quarantined)
            if validation_error is not None:
                # A contract-level rejection is classified for telemetry but
                # remains non-executable.  The legal-prefix diagnostic arm is
                # reserved for proposals that passed the ordinary contract
                # and were rejected by this narrower validity gate.
                runtime.journal.emit('public_review_unexposed',
                                     reason='validity_validation_rejected',
                                     reference=self.reference,
                                     error_type=type(validation_error).__name__,
                                     detail=str(validation_error)[:300])
                return None
            receipt = self._execute()
        except Exception as error:
            # Review failure must not acquire completion authority; preserve its exact failure class.
            runtime.journal.emit('public_review_unexposed', reason='review_unavailable',
                                 error_type=type(error).__name__, detail=str(error)[:300])
            return None
        mode = runtime.modes['public_review']
        exposed_result = receipt['result'] == 'PROBE_RED' or (
            receipt['result'] == 'PROBE_INCONCLUSIVE' and self._validity_enabled())
        if mode == Mode.SHADOW or not exposed_result:
            runtime.journal.emit('public_review_unexposed', reason='shadow' if mode == Mode.SHADOW
                                 else receipt['result'], reference=self.reference)
            return None
        card = self._card(receipt)
        self.exposed = True
        self.last_exposed_result = receipt['result']
        runtime.journal.emit('public_review_advice_selected', reference=self.reference, text=card)
        return card

    def rerun(self, selector: str) -> str:
        if (not self.exposed or self.runtime.modes['public_review'] != Mode.BOTH
                or selector != self.reference):
            return 'PROBE_UNAVAILABLE: no exposed reviewed probe matches this reference.'
        if self.deadline_policy is not None:
            admission = self.deadline_policy.admit(0)
            self.runtime.journal.emit('public_review_admission', stage='rerun', **_journal_safe_fields(admission))
            if not admission['admitted']:
                self.runtime.journal.emit('public_probe_rerun_unavailable',
                                          error_type='TimeoutError', detail='wall_budget_unavailable')
                return 'PROBE_UNAVAILABLE: wall_budget_unavailable'
        try:
            receipt = self._execute()
        except Exception as error:
            self.runtime.journal.emit('public_probe_rerun_unavailable', error_type=type(error).__name__,
                                      detail=str(error)[:300])
            return 'PROBE_UNAVAILABLE: ' + str(error)[:300]
        return self._card(receipt)
