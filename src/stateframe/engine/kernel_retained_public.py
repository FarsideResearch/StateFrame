"""Public source handles without run-scoped provenance; host archive stays authoritative."""
from __future__ import annotations

import os
from kernel_measurement import digest
from kernel_lossless_store_paging import (
    OMITTED_REFERENCE_INDEX,
    REVISION_REF_FIELD,
    STABLE_REFERENCE_PREFIX,
)


def public_identity_mode() -> str:
    mode = os.environ.get('KERNEL_RETAINED_PUBLIC_ID', 'run')
    if mode not in {'run', 'semantic'}:
        raise ValueError('KERNEL_RETAINED_PUBLIC_ID must be run or semantic')
    return mode


class PublicSourceReferences:
    def __init__(self, archive, mode: str):
        if mode not in {'run', 'semantic'}:
            raise ValueError('source reference mode must be run or semantic')
        self.archive, self.mode = archive, mode
        self._public: dict[str, str] = {}
        self._internal: dict[str, str] = {}

    def _sync(self) -> None:
        for ref, row in self.archive.revisions.items():
            if ref in self._public:
                continue
            handle = STABLE_REFERENCE_PREFIX + digest({
                'path': row['path'], 'revision': row['revision'],
            })
            previous = self._internal.get(handle)
            if previous is not None and previous != ref:
                raise ValueError('ambiguous retained source handle')
            self._public[ref], self._internal[handle] = handle, ref

    def reference(self, internal: str | None) -> str | None:
        if self.mode == 'run' or internal is None:
            return internal
        self._sync()
        return self._public[internal]

    def resolve(self, public: str) -> str:
        # A handle resolves only inside this archive, never through another run.
        if public in self.archive.latest:
            return self.archive.latest[public]
        if self.mode == 'run':
            return public
        self._sync()
        return self._internal[public]

    def source(self, raw: dict) -> dict:
        if self.mode == 'run':
            return dict(raw)
        result = {key: value for key, value in raw.items() if key != 'evidence_ref'}
        for key in ('reference', 'current_reference', REVISION_REF_FIELD):
            if key in result:
                if result[key] in self.archive.revisions:
                    result[key] = self.reference(result[key])
        return result

    def view(self, raw: dict) -> dict:
        if self.mode == 'run':
            return dict(raw)
        result = {key: value for key, value in raw.items() if key != 'graph_digest'}
        for key in ('catalog', 'selected'):
            result[key] = [self.source(row) for row in raw[key]]
        if OMITTED_REFERENCE_INDEX in raw:
            result[OMITTED_REFERENCE_INDEX] = [self.source(row) for row in raw[OMITTED_REFERENCE_INDEX]]
        return result
