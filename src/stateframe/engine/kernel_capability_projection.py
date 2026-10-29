"""Small capability folds with a verified cold reader over the durable journal."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from kernel_history_spool import legacy
from kernel_measurement import profile_active_host_span

_SEED = '0' * 64
_MENU = 'green_lock_menu_authority'


def _advance(chain: str, line: str) -> str:
    return hashlib.sha256(bytes.fromhex(chain) + line.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class JournalPrefix(Sequence):
    """Immutable prefix identity, not another retained copy of parsed records.

    Parsing/validation is completed before returning any cold-reader records.
    The runtime's writer lease makes the journal append-only; a changed prefix
    is refused rather than exposing different authority under an old cursor.
    """

    path: str
    size: int = 0
    chain: str = _SEED
    count: int = 0

    def advance(self, line: str, *, record: bool):
        return replace(self, chain=_advance(self.chain, line), count=self.count + int(record))

    def finish(self, size: int):
        return replace(self, size=size)

    def records(self):
        if not self.size:
            if self.count or self.chain != _SEED:
                raise ValueError('capability prefix is incomplete')
            return ()
        with profile_active_host_span('capability_prefix_read'):
            with Path(self.path).open('rb') as source:
                data = source.read(self.size)
        if len(data) != self.size or not data.endswith(b'\n'):
            raise ValueError('capability journal prefix is truncated')
        with profile_active_host_span('capability_prefix_hash_verify'):
            result, chain = [], _SEED
            for line in data.decode('utf-8').splitlines():
                chain = _advance(chain, line)
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, Mapping):
                    result.append(value)
            if chain != self.chain or len(result) != self.count:
                raise ValueError('capability journal prefix changed')
        return tuple(result)

    def __len__(self):
        return self.count

    def __iter__(self):
        return iter(self.records())

    def __getitem__(self, index):
        return self.records()[index]


def _restore_projection(values):
    if not legacy():
        return CapabilityProjection(values)
    # Restore the original structures AND shared menu/record objects. The
    # durable journal is already a dependency of runtime checkpoint restore.
    records = list(values['records'])
    values['records'] = records
    values['menu'] = [record for record in records if record.get('kind') == _MENU]
    return values


class CapabilityProjection(dict):
    def __reduce_ex__(self, protocol):
        return _restore_projection, (dict(self),)


def fold_menu(menu, record):
    """Only successful-call existence and any successful change are observed."""
    if record.get('error_type'):
        return menu
    changed = bool((menu and menu[0].get('added')) or record.get('added') or record.get('removed'))
    return [{'added': changed, 'removed': False}]
