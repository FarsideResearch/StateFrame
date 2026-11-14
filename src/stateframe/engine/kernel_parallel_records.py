"""Versioned, strict wire records for parallel subframes."""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field, fields
import hashlib
import json
import math
import os
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


class ParallelRecordError(ValueError):
    """A parallel record does not match its declared wire schema."""


_PUBLIC_LIMIT_NAMES = {
    "max_units": "max_subframes",
    "min_units": "min_subframes",
    "max_paths_per_unit": "max_paths_per_subframe",
    "unit_request_tokens": "subframe_request_tokens",
    "unit_max_requests": "subframe_max_requests",
}

SUBFRAME_TOKEN_CAP_FRACTION_NUMERATOR = 80
SUBFRAME_TOKEN_CAP_FRACTION_DENOMINATOR = 100
MAX_PARALLEL_UNITS = 1024


def default_subframe_token_cap(prompt_budget: int) -> int:
    """Scale the subframe first-request ceiling with the run's prompt budget."""
    if type(prompt_budget) is not int or prompt_budget <= 0:
        raise ValueError('prompt_budget must be a positive integer')
    return max(1, prompt_budget * SUBFRAME_TOKEN_CAP_FRACTION_NUMERATOR
               // SUBFRAME_TOKEN_CAP_FRACTION_DENOMINATOR)


def display_limit_name(setting: str) -> str:
    return _PUBLIC_LIMIT_NAMES.get(setting, setting)


class ParallelConfigError(ValueError):
    """A parallel limit setting is invalid for its source or current run."""

    def __init__(self, setting: str, value: object, source: str, reason: str):
        self.setting = setting
        self.value = value
        self.source = source
        self.reason = reason
        super().__init__(f"invalid parallel setting {display_limit_name(setting)}={value!r} "
                         f"from {source}: {reason}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _encode(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_json()
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _json_value(value: Any, path: str) -> Any:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_value(item, f'{path}[{index}]') for index, item in enumerate(value)]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item, f'{path}.{key}') for key, item in value.items()}
    raise ParallelRecordError(f'{path}: expected JSON value')


def _decode(value: Any, expected: Any, path: str) -> Any:
    origin = get_origin(expected)
    args = get_args(expected)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        variants = [variant for variant in args if variant is not type(None)]
        if len(variants) == 1:
            return _decode(value, variants[0], path)
        for variant in variants:
            try:
                return _decode(value, variant, path)
            except ParallelRecordError:
                continue
        raise ParallelRecordError(f"{path}: invalid union value")
    if origin is Literal:
        if value not in args or type(value) is not type(next(item for item in args if item == value)):
            raise ParallelRecordError(f"{path}: invalid literal {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, list):
            raise ParallelRecordError(f"{path}: expected array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(item, args[0], f"{path}[{index}]") for index, item in enumerate(value))
        if len(value) != len(args):
            raise ParallelRecordError(f"{path}: expected {len(args)} items")
        return tuple(_decode(item, typ, f"{path}[{index}]") for index, (item, typ) in enumerate(zip(value, args)))
    if origin is dict:
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ParallelRecordError(f"{path}: expected object with string keys")
        return {key: _decode(item, args[1], f"{path}.{key}") for key, item in value.items()}
    if expected is type(None):
        if value is not None:
            raise ParallelRecordError(f"{path}: expected null")
        return None
    if expected is Any:
        return _json_value(value, path)
    if isinstance(expected, type) and issubclass(expected, Record):
        try:
            return expected.from_json(value)
        except ParallelRecordError as exc:
            raise ParallelRecordError(f"{path}: {exc}") from exc
    if expected in (str, int, float, bool):
        if expected is float and type(value) in (int, float) and math.isfinite(value):
            return float(value)
        if type(value) is not expected:
            raise ParallelRecordError(f"{path}: expected {expected.__name__}")
        return value
    raise ParallelRecordError(f"{path}: unsupported field type {expected!r}")


class Record:
    def __post_init__(self) -> None:
        for item in fields(self):
            if item.name == 'schema' and getattr(self, item.name) != item.default:
                raise ParallelRecordError(f'{type(self).__name__}.schema: expected {item.default!r}')

    def to_json(self) -> dict[str, Any]:
        return {item.name: _encode(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def from_json(cls, value: dict[str, Any]):
        if not isinstance(value, dict):
            raise ParallelRecordError(f"{cls.__name__}: expected object")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            raise ParallelRecordError(f"{cls.__name__}: missing {sorted(expected - set(value), key=str)}, unknown {sorted(set(value) - expected, key=str)}")
        hints = get_type_hints(cls)
        values = {item.name: _decode(value[item.name], hints[item.name], f"{cls.__name__}.{item.name}")
                  for item in fields(cls) if item.init}
        for item in fields(cls):
            if item.name == 'schema' and value[item.name] != item.default:
                raise ParallelRecordError(f"{cls.__name__}.{item.name}: expected {item.default!r}")
        result = cls(**values)
        return result

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.to_json())).hexdigest()


@dataclass(frozen=True)
class Param(Record):
    name: str
    kind: Literal['posonly', 'pos', 'varpos', 'kwonly', 'varkw']
    has_default: bool
    annotation: str | None


@dataclass(frozen=True)
class CanonicalSignature(Record):
    kind: Literal['function', 'async_function', 'class', 'method', 'constant']
    params: tuple[Param, ...]
    returns: str | None
    decorators: tuple[str, ...]


@dataclass(frozen=True)
class ExportSpec(Record):
    id: str
    path: str
    symbol: str
    signature: CanonicalSignature | None


@dataclass(frozen=True)
class ConsumeSpec(Record):
    export_id: str | None
    path: str | None
    symbol: str | None

    def __post_init__(self) -> None:
        if self.export_id is None and (self.path is None or self.symbol is None):
            raise ParallelRecordError('ConsumeSpec: export_id or path and symbol are required')
        if (self.path is None) != (self.symbol is None):
            raise ParallelRecordError('ConsumeSpec: path and symbol must appear together')


@dataclass(frozen=True)
class CheckRecipe(Record):
    source_paths: tuple[str, ...]
    selectors: tuple[str, ...]
    command_kind: Literal['pytest', 'run_check_command', 'none']
    schema: str = field(default='stateframe-check-recipe/1', kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, 'source_paths', tuple(sorted(self.source_paths)))


@dataclass(frozen=True)
class UnitContract(Record):
    id: str
    outcome: str
    owns: tuple[str, ...]
    creates: tuple[str, ...]
    exports: tuple[ExportSpec, ...]
    consumes: tuple[ConsumeSpec, ...]
    reads: tuple[str, ...]
    requirement_ids: tuple[str, ...]
    check: CheckRecipe
    model_class: str = 'default'

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ('owns', 'creates', 'reads', 'requirement_ids'):
            object.__setattr__(self, name, tuple(sorted(getattr(self, name))))
        object.__setattr__(self, 'exports', tuple(sorted(self.exports, key=lambda item: item.id)))
        object.__setattr__(self, 'consumes', tuple(sorted(self.consumes,
                                                      key=lambda item: (item.export_id or '', item.path or '', item.symbol or ''))))


@dataclass(frozen=True)
class ParallelRequest(Record):
    call_id: str
    units: tuple[UnitContract, ...]
    mode: str
    schema: str = field(default='stateframe-parallel-request/1', kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, 'units', tuple(sorted(self.units, key=lambda item: item.id)))


# Names, CLI flags, and profile switches share one source so host CLI help and
# validation cannot drift apart. Each value contains its CLI flag and profile switch.
PARALLEL_LIMIT_OPTIONS: Mapping[str, Mapping[str, str]] = types.MappingProxyType({
    name: types.MappingProxyType({'flag': flag, 'switch': switch})
    for name, flag, switch in (
        ('max_units', '--parallel-max-subframes', 'STATEFRAME_PARALLEL_MAX_SUBFRAMES'),
        ('min_units', '--parallel-min-subframes', 'STATEFRAME_PARALLEL_MIN_SUBFRAMES'),
        ('max_paths_per_unit', '--parallel-max-paths-per-subframe',
         'STATEFRAME_PARALLEL_MAX_PATHS_PER_SUBFRAME'),
        ('max_paths_total', '--parallel-max-paths-total', 'STATEFRAME_PARALLEL_MAX_PATHS_TOTAL'),
        ('unit_request_tokens', '--parallel-subframe-tokens',
         'STATEFRAME_PARALLEL_SUBFRAME_REQUEST_TOKENS'),
        ('max_owned_source_bytes', '--parallel-max-owned-bytes', 'STATEFRAME_PARALLEL_MAX_OWNED_SOURCE_BYTES'),
        ('unit_max_requests', '--parallel-subframe-max-requests',
         'STATEFRAME_PARALLEL_SUBFRAME_MAX_REQUESTS'),
        ('parent_reserve_requests', '--parallel-parent-reserve', 'STATEFRAME_PARALLEL_PARENT_RESERVE_REQUESTS'),
        ('check_permits', '--parallel-check-permits', 'STATEFRAME_PARALLEL_CHECK_PERMITS'),
        ('max_concurrency', '--parallel-max', 'STATEFRAME_PARALLEL_MAX_CONCURRENCY'),
    )
})

PARALLEL_TUNING_SWITCHES = frozenset({
    'STATEFRAME_PARALLEL_BYTE_TOKEN_CALIBRATION',
    'STATEFRAME_PARALLEL_BYTE_TOKEN_HEADROOM_PCT',
    'STATEFRAME_PARALLEL_USAGE_FEEDBACK',
})


def format_parallel_limit(setting: str, value: int) -> str:
    option = PARALLEL_LIMIT_OPTIONS[setting]
    return (f"{display_limit_name(setting)}={value} "
            f"({option['flag']} / {option['switch']})")


def _default_check_permits() -> int:
    return max(1, min(4, (os.cpu_count() or 1) // 4))


@dataclass(frozen=True)
class ParallelLimits(Record):
    max_units: int = 6
    min_units: int = 2
    max_paths_per_unit: int = 6
    max_paths_total: int = 24
    max_owned_source_bytes: int = 24576
    # 0.80 x the shipped 33,344-token prompt budget = 26,675 tokens. This
    # admits realistic multi-file units with ~15,000-25,000-token first frames;
    # adapters4 measured 17,900-25,100 after child-frame scoping and site references.
    # The default remains below the 33,344-token engine max.
    unit_request_tokens: int = 26675
    unit_max_requests: int = 12
    parent_reserve_requests: int = 4
    check_permits: int = field(default_factory=_default_check_permits)
    max_concurrency: int = 4
    scan_max_files: int = 2000
    scan_max_bytes: int = 16 << 20
    scan_seconds: float = 10.0

    @classmethod
    def from_settings(cls, profile_env: Mapping[str, str], cli: Mapping[str, int | None],
                      *, prompt_budget: int) -> ParallelLimits:
        """Resolve profile limits with CLI overrides and validate them without clamping."""
        if not isinstance(profile_env, Mapping):
            raise ParallelConfigError('profile_env', profile_env, 'profile_env', 'must be a mapping')
        if not isinstance(cli, Mapping):
            raise ParallelConfigError('cli', cli, 'cli', 'must be a mapping')
        if type(prompt_budget) is not int or prompt_budget <= 0:
            raise ParallelConfigError('prompt_budget', prompt_budget, 'prompt_budget',
                                      'must be a positive integer')

        flag_to_setting = {option['flag']: setting
                           for setting, option in PARALLEL_LIMIT_OPTIONS.items()}
        cli_values: dict[str, int | None] = {}
        for cli_name, value in cli.items():
            if not isinstance(cli_name, str):
                raise ParallelConfigError(str(cli_name), value, 'cli', 'setting name must be a string')
            setting = cli_name if cli_name in PARALLEL_LIMIT_OPTIONS else flag_to_setting.get(cli_name)
            if setting is None:
                raise ParallelConfigError(cli_name, value, 'cli', 'unknown parallel limit setting')
            if setting in cli_values:
                raise ParallelConfigError(setting, value, 'cli', 'setting was supplied more than once')
            cli_values[setting] = value
        recognized_switches = ({option['switch'] for option in PARALLEL_LIMIT_OPTIONS.values()}
                               | PARALLEL_TUNING_SWITCHES)
        for switch, value in profile_env.items():
            if not isinstance(switch, str):
                raise ParallelConfigError(str(switch), value, 'profile_env', 'setting name must be a string')
            if (switch.startswith('STATEFRAME_PARALLEL_') and switch not in recognized_switches
                    and switch != 'STATEFRAME_PARALLEL_MODE'):
                setting = switch.removeprefix('STATEFRAME_PARALLEL_').lower()
                raise ParallelConfigError(setting, value, switch, 'unknown parallel limit setting')

        defaults = cls()
        values = {item.name: getattr(defaults, item.name) for item in fields(cls) if item.init}
        sources: dict[str, str] = {}
        for setting, option in PARALLEL_LIMIT_OPTIONS.items():
            flag = option['flag']
            switch = option['switch']
            if setting in cli_values and cli_values[setting] is not None:
                raw = cli_values[setting]
                source = flag
                if type(raw) is not int:
                    raise ParallelConfigError(setting, raw, source, 'must be a positive integer')
                value = raw
            elif switch in profile_env:
                raw = profile_env[switch]
                source = switch
                if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
                    raise ParallelConfigError(setting, raw, source, 'must be a positive integer')
                try:
                    value = int(raw, 10)
                except ValueError as error:
                    raise ParallelConfigError(setting, raw, source,
                                              'must be a positive integer') from error
            else:
                value = (default_subframe_token_cap(prompt_budget)
                         if setting == 'unit_request_tokens' else values[setting])
                source = 'default'
            if type(value) is not int or value <= 0:
                raise ParallelConfigError(setting, value, source, 'must be a positive integer')
            values[setting] = value
            sources[setting] = source

        if values['max_units'] > MAX_PARALLEL_UNITS:
            raise ParallelConfigError('max_units', values['max_units'], sources['max_units'],
                                      f'must be at most {MAX_PARALLEL_UNITS}')
        if values['min_units'] > values['max_units']:
            raise ParallelConfigError('min_units', values['min_units'], sources['min_units'],
                                      f"must not exceed max_subframes={values['max_units']}")
        if values['unit_request_tokens'] > prompt_budget:
            raise ParallelConfigError('unit_request_tokens', values['unit_request_tokens'],
                                      sources['unit_request_tokens'],
                                      f'must not exceed prompt_budget={prompt_budget}')
        if values['max_paths_per_unit'] > values['max_paths_total']:
            raise ParallelConfigError('max_paths_per_unit', values['max_paths_per_unit'],
                                      sources['max_paths_per_unit'],
                                      f"must not exceed max_paths_total={values['max_paths_total']}")
        return cls(**values)

@dataclass(frozen=True)
class InterfaceProjection(Record):
    export_id: str | None
    path: str
    symbol: str
    signature: CanonicalSignature | None
    doc_first_line: str
    line: int
    source_digest: str


@dataclass(frozen=True)
class Evidence(Record):
    path: str
    line: int
    text: str

    def __post_init__(self) -> None:
        if len(self.text) > 200:
            raise ParallelRecordError('Evidence.text exceeds 200 characters')


@dataclass(frozen=True)
class CouplingEdge(Record):
    src_unit: str
    dst_unit: str
    source: Literal['declared', 'static_import', 'static_name']
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class CouplingPair(Record):
    a: str
    b: str
    status: Literal['COUPLED', 'INDEPENDENT_BY_SCAN', 'UNKNOWN']
    reason: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Refusal(Record):
    reason: str
    detail: str
    paths: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    members: tuple[str, ...]
    hint: str | None = None

    def __post_init__(self) -> None:
        if self.hint is not None and not isinstance(self.hint, str):
            raise ParallelRecordError('Refusal.hint: expected string or null')

    def to_json(self) -> dict[str, Any]:
        result = super().to_json()
        if self.hint is None:
            result.pop('hint')
        return result

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> Refusal:
        if not isinstance(value, dict):
            raise ParallelRecordError('Refusal: expected object')
        return super().from_json({**value, 'hint': value.get('hint')})


@dataclass(frozen=True)
class AdmissionAdjustment(Record):
    unit_id: str
    action: Literal['CREATE_TO_EDIT', 'DROP_UNIT', 'DROP_DEPENDENT']
    path: str
    reason: str


@dataclass(frozen=True)
class AdmissionResult(Record):
    status: Literal['ADMITTED', 'REFUSED']
    refusal: Refusal | None
    edges: tuple[CouplingEdge, ...]
    pairs: tuple[CouplingPair, ...]
    order: tuple[str, ...]
    generation: int
    manifest_digest: str
    limits: ParallelLimits
    units: tuple[UnitContract, ...] = ()
    adjustments: tuple[AdmissionAdjustment, ...] = ()
    schema: str = field(default='stateframe-parallel-admission/2', kw_only=True)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> AdmissionResult:
        if isinstance(value, dict) and value.get('schema') == 'stateframe-parallel-admission/1':
            if 'units' in value or 'adjustments' in value:
                raise ParallelRecordError('AdmissionResult: legacy schema cannot carry adjustments')
            value = {**value, 'schema': 'stateframe-parallel-admission/2',
                     'units': [], 'adjustments': []}
        return super().from_json(value)


@dataclass(frozen=True)
class RequirementClause(Record):
    id: str
    text: str
    source: Literal['planner', 'intersect']
    refs: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        for ref in self.refs:
            if not isinstance(ref, str) or ref.count('::') > 1:
                raise ParallelRecordError('RequirementClause.refs: invalid structured reference')
            path, separator, symbol = ref.partition('::')
            if (not path or path.startswith('/') or '\\' in path or
                    any(part in ('', '.', '..') for part in path.split('/')) or
                    separator and (not symbol or any(char.isspace() for char in symbol))):
                raise ParallelRecordError('RequirementClause.refs: invalid structured reference')
        object.__setattr__(self, 'refs', tuple(sorted(set(self.refs))))


@dataclass(frozen=True)
class UnitSpec(Record):
    call_id: str
    unit_id: str
    execution_id: str
    run_seed: str
    contract: UnitContract
    interfaces: tuple[InterfaceProjection, ...]
    requirement_clauses: tuple[RequirementClause, ...]
    visible_paths: tuple[str, ...]
    hidden_paths: tuple[str, ...]
    failure_projection: str
    admission_generation: int
    start_generation: int
    start_manifest_digest: str
    policy_digest: str
    limits: ParallelLimits
    meter_calibration: dict[str, Any] | None = None
    subframe_cap_calibration: dict[str, Any] | None = None
    schema: str = field(default='stateframe-unit-spec/3', kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.meter_calibration is not None:
            _json_value(self.meter_calibration, 'UnitSpec.meter_calibration')
            if not isinstance(self.meter_calibration, dict):
                raise ParallelRecordError('UnitSpec.meter_calibration: expected object or null')
        if self.subframe_cap_calibration is not None:
            _json_value(self.subframe_cap_calibration, 'UnitSpec.subframe_cap_calibration')
            if not isinstance(self.subframe_cap_calibration, dict):
                raise ParallelRecordError('UnitSpec.subframe_cap_calibration: expected object or null')

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> UnitSpec:
        if isinstance(value, dict) and value.get('schema') == 'stateframe-unit-spec/1':
            if 'meter_calibration' in value or 'subframe_cap_calibration' in value:
                raise ParallelRecordError('UnitSpec: legacy schema cannot carry calibration fields')
            value = {**value, 'schema': 'stateframe-unit-spec/2',
                     'meter_calibration': None}
        if isinstance(value, dict) and value.get('schema') == 'stateframe-unit-spec/2':
            if 'subframe_cap_calibration' in value:
                raise ParallelRecordError('UnitSpec: legacy schema cannot carry subframe cap calibration')
            value = {**value, 'schema': 'stateframe-unit-spec/3',
                     'subframe_cap_calibration': None}
        return super().from_json(value)


@dataclass(frozen=True)
class ProposalEntry(Record):
    path: str
    kind: Literal['modify', 'create']
    before_sha256: str | None
    after_sha256: str
    mode: int


@dataclass(frozen=True)
class CheckReceipt(Record):
    recipe_digest: str
    product_manifest_digest: str
    harness_manifest_digest: str
    environment_digest: str
    policy_digest: str
    status: Literal['PASS', 'FAIL', 'INVALID', 'NOT_RUN']
    invalid_reason: str | None
    wrote_paths: tuple[str, ...]
    raw_evidence_ref: str
    projected_observation: str
    product_manifest_modes: dict[str, int] = field(default_factory=dict)
    schema: str = field(default='stateframe-parallel-check/2', kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (not isinstance(self.product_manifest_modes, dict)
                or any(not isinstance(path, str) or type(mode) is not int
                       or mode < 0 or mode > 0o7777
                       for path, mode in self.product_manifest_modes.items())):
            raise ParallelRecordError('CheckReceipt.product_manifest_modes: invalid modes mapping')

    @classmethod
    def from_json(cls, value: dict[str, Any]):
        if (isinstance(value, dict)
                and value.get('schema') == 'stateframe-parallel-check/1'):
            legacy_fields = {item.name for item in fields(cls)
                             if item.name != 'product_manifest_modes'}
            if set(value) == legacy_fields:
                value = {**value, 'product_manifest_modes': {},
                         'schema': 'stateframe-parallel-check/2'}
        return super().from_json(value)


@dataclass(frozen=True)
class Proposal(Record):
    call_id: str
    unit_id: str
    execution_id: str
    incarnation: int
    entries: tuple[ProposalEntry, ...]
    contract_digest: str
    check_receipt: CheckReceipt | None
    schema: str = field(default='stateframe-parallel-proposal/1', kw_only=True)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> Proposal:
        proposal = super().from_json(value)
        receipt = value.get('check_receipt') if isinstance(value, dict) else None
        if isinstance(receipt, dict) and receipt.get('schema') == 'stateframe-parallel-check/1':
            # CheckReceipt upgrades v1 to v2 on decode. A previously sealed
            # proposal must still hash to its original, validated wire bytes.
            object.__setattr__(proposal, '_legacy_wire', copy.deepcopy(value))
            object.__setattr__(proposal, '_migrated_wire', Record.to_json(proposal))
        return proposal

    def to_json(self) -> dict[str, Any]:
        current = super().to_json()
        if current == getattr(self, '_migrated_wire', None):
            return copy.deepcopy(self._legacy_wire)
        return current


@dataclass(frozen=True)
class ContractVerdict(Record):
    export_id: str
    status: Literal['PASS', 'FAIL', 'UNKNOWN']
    expected: CanonicalSignature | None
    actual: CanonicalSignature | None
    reason: str


@dataclass(frozen=True)
class ImportVerdict(Record):
    path: str
    line: int
    module: str
    status: Literal['RESOLVED', 'EXTERNAL', 'UNRESOLVED', 'UNKNOWN']


@dataclass(frozen=True)
class ContractReport(Record):
    verdicts: tuple[ContractVerdict, ...]
    imports: tuple[ImportVerdict, ...]
    ok: bool


@dataclass(frozen=True)
class EngineBinding(Record):
    product_revision: str
    harness_revision: str


@dataclass(frozen=True)
class UnitUsage(Record):
    requests: int
    physical_sends: int
    prompt_tokens: int
    completion_tokens: int
    initial_frame_tokens: int | None


@dataclass(frozen=True)
class TxnRecord(Record):
    txn_id: int
    call_id: str
    unit_id: str
    execution_id: str
    pre_generation: int
    post_generation: int
    entries: tuple[ProposalEntry, ...]
    pre_manifest_digest: str
    post_manifest_digest: str
    pre_binding: EngineBinding
    post_binding: EngineBinding
    contract_report_digest: str
    check_receipt_digest: str | None
    usage: UnitUsage
    schema: str = field(default='stateframe-parallel-txn/1', kw_only=True)


TxnDisposition = Literal['DECIDED', 'MATERIALIZED', 'PUBLISHED']


@dataclass(frozen=True)
class FileDelta(Record):
    path: str
    added_lines: int
    removed_lines: int
    kind: Literal['modify', 'create']


@dataclass(frozen=True)
class ParallelMeasured(Record):
    parallel_wall_seconds: float
    estimated_serial_seconds: float
    speedup: float
    peak_active_workers: int
    queue_wait_seconds: float


@dataclass(frozen=True)
class UnitOutcome(Record):
    unit_id: str
    status: Literal['COMMITTED', 'COMMITTED_UNVERIFIED', 'REFUSED', 'BLOCKED', 'BUDGET_EXHAUSTED', 'ENGINE_ERROR', 'CANCELLED', 'TIMED_OUT', 'RECOVERY_REQUIRED']
    reason: str | None
    detail: str | None
    txn_id: int | None
    files: tuple[FileDelta, ...]
    contract_ok: bool | None
    check: str
    usage: UnitUsage
    wall_seconds: float
    queue_wait_seconds: float


@dataclass(frozen=True)
class ParallelResult(Record):
    call_id: str
    admission: AdmissionResult
    outcomes: tuple[UnitOutcome, ...]
    txn_order: tuple[int, ...]
    final_check: str
    unknown_coupling: tuple[CouplingPair, ...]
    integration: str
    observation: str
    measured: ParallelMeasured
    schema: str = field(default='stateframe-parallel-result/1', kw_only=True)


@dataclass(frozen=True)
class SendReservation(Record):
    send_id: str
    execution_id: str
    state: Literal['AUTHORIZED', 'SETTLED']
    usage: dict[str, Any] | None


@dataclass(frozen=True)
class ExecutionState(Record):
    unit_id: str
    execution_id: str
    run_dir: str
    spec_digest: str
    last_incarnation: int
    terminal: str | None


@dataclass(frozen=True)
class RecoveryDescriptor(Record):
    call_id: str
    intent_digest: str
    txns: tuple[tuple[int, TxnDisposition], ...]
    proposals: tuple[str, ...]
    executions: tuple[ExecutionState, ...]
    schema: str = field(default='stateframe-parallel-recovery/1', kw_only=True)


@dataclass(frozen=True)
class HostRequest(Record):
    attempt: int
    call_id: str
    seq: int
    op: str
    scope: str
    payload: dict[str, Any]
    payload_sha256: str
    schema: str = field(default='stateframe-parallel-rq/1', kw_only=True)


@dataclass(frozen=True)
class HostAnswer(Record):
    attempt: int
    call_id: str
    seq: int
    op: str
    status: Literal['OK', 'ERROR', 'PENDING']
    result_id: str | None
    payload: dict[str, Any]
    schema: str = field(default='stateframe-parallel-an/1', kw_only=True)
