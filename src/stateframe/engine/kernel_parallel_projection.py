"""Deterministic subframe specifications and task text."""
from __future__ import annotations

import hashlib
from typing import Sequence

if __package__:
    from .kernel_parallel_contracts import project_interface
    from .kernel_parallel_records import (AdmissionResult, ParallelRequest, RequirementClause, UnitSpec)
else:
    from kernel_parallel_contracts import project_interface
    from kernel_parallel_records import (AdmissionResult, ParallelRequest, RequirementClause, UnitSpec)


def _identity(left: str, right: str) -> str:
    return hashlib.sha256((left + '\0' + right).encode('utf-8')).hexdigest()[:32]


def project_unit(admission: AdmissionResult, request: ParallelRequest, unit_id: str,
                 snapshot_at_start, *, parent_seed: str, policy_digest: str,
                 failure_projection: str, requirements: Sequence[RequirementClause]) -> UnitSpec:
    if admission.status != 'ADMITTED':
        raise ValueError('cannot project a refused parallel request')
    contract = next((item for item in request.units if item.id == unit_id), None)
    if contract is None:
        raise ValueError(f'unknown subframe {unit_id!r}')
    exports = {export.id: export for unit in request.units for export in unit.exports}
    exports_by_symbol = {(export.path, export.symbol): export
                         for unit in request.units for export in unit.exports}
    interfaces = []
    for consume in contract.consumes:
        if consume.export_id is not None:
            export = exports[consume.export_id]
            projection = project_interface(snapshot_at_start, export.path, export.symbol, export.id)
            interfaces.append(projection)
        else:
            export = exports_by_symbol.get((consume.path, consume.symbol))
            interfaces.append(project_interface(snapshot_at_start, consume.path, consume.symbol,
                                                export.id if export is not None else None))
    interfaces.sort(key=lambda item: (item.path, item.symbol, item.export_id or ''))
    visible = tuple(sorted(set((*contract.owns, *contract.creates, *contract.reads,
                                *contract.check.source_paths))))
    hidden = tuple(sorted(set(snapshot_at_start.list_files()) - set(visible)))
    relevant_paths = set((*contract.owns, *contract.creates, *contract.reads))
    consumed_symbols = {f'{item.path}::{item.symbol}' for item in interfaces}
    for consume in contract.consumes:
        if consume.export_id is not None:
            export = exports[consume.export_id]
            consumed_symbols.add(f'{export.path}::{export.symbol}')
        else:
            consumed_symbols.add(f'{consume.path}::{consume.symbol}')
    clauses = {}
    known = {clause.id for clause in requirements}
    unknown = set(contract.requirement_ids) - known
    if unknown:
        raise ValueError(f'UNKNOWN_REQUIREMENT: {", ".join(sorted(unknown))}')
    for clause in requirements:
        if clause.id in contract.requirement_ids:
            clauses[clause.id] = RequirementClause(clause.id, clause.text, clause.source, clause.refs)
        elif any(ref.partition('::')[0] in relevant_paths or ref in consumed_symbols for ref in clause.refs):
            clauses[clause.id] = RequirementClause(clause.id, clause.text, 'intersect', clause.refs)
    execution_id = _identity(request.call_id, unit_id)
    return UnitSpec(request.call_id, unit_id, execution_id, _identity(parent_seed, execution_id), contract,
                    tuple(interfaces), tuple(clauses[key] for key in sorted(clauses)), visible, hidden,
                    failure_projection, admission.generation, snapshot_at_start.generation,
                    snapshot_at_start.manifest_digest, policy_digest, admission.limits)


def render_unit_task(spec: UnitSpec) -> str:
    contract = spec.contract
    lines = [f'Subframe {spec.unit_id}: {contract.outcome}',
             'Owned: ' + (', '.join(contract.owns) or '(none)'),
             'Create: ' + (', '.join(contract.creates) or '(none)')]
    for export in contract.exports:
        signature = export.signature.to_json() if export.signature else 'UNKNOWN'
        lines.append(f'Export {export.id}: {export.path}:{export.symbol} {signature}')
    for interface in spec.interfaces:
        signature = interface.signature.to_json() if interface.signature else 'UNKNOWN'
        lines.append(f'Consume {interface.path}:{interface.line}:{interface.symbol} {signature}; '
                     f'{interface.doc_first_line}; sha256={interface.source_digest}')
    for clause in spec.requirement_clauses:
        lines.append(f'Requirement {clause.id}: {clause.text}')
    lines.append('Checks: ' + (', '.join(contract.check.selectors) or '(none)'))
    if spec.failure_projection:
        lines.append('Current failure: ' + spec.failure_projection)
    lines.append('Other files are read-only or withheld. Edit only owned paths and create only declared paths.')
    return '\n'.join(lines)
