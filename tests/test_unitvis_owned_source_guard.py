"""UNITVIS-073: allocator-ON unallocated assembly must never silently drop owned source."""
import pytest

from kernel_request_source_window import verify_current_source_message
from kernel_stateframe_framefit5 import FrameFit5Error, _require_owned_source_rendered

SOURCE = 'def price(x):\n    return x\n'


def _plan(retained=('unit:a',), deferred=(), bounded=True):
    plan = {
        'units': [{'unit_handle': 'unit:a', 'path': 'shop/pricing.py'}],
        'deferred_paths': list(deferred),
    }
    if bounded:
        plan['bounded_delivery'] = {'retained_handles': list(retained), 'omitted_handles': []}
    return plan


def _rendered():
    return [{'role': 'user', 'content': '[Current source: shop/pricing.py]\n' + SOURCE}]


def test_guard_accepts_verified_owned_source():
    content = _rendered()[0]['content']
    assert verify_current_source_message(content, path='shop/pricing.py', source=SOURCE)
    _require_owned_source_rendered(_rendered(), _plan(), {'shop/pricing.py': SOURCE})


def test_guard_raises_when_retained_owned_source_is_missing_from_request():
    with pytest.raises(FrameFit5Error, match='was not rendered'):
        _require_owned_source_rendered([], _plan(), {'shop/pricing.py': SOURCE})


def test_guard_raises_when_retained_path_has_no_snapshot():
    with pytest.raises(FrameFit5Error, match='no snapshot'):
        _require_owned_source_rendered(_rendered(), _plan(), {})
    with pytest.raises(FrameFit5Error, match='no snapshot'):
        _require_owned_source_rendered(_rendered(), _plan(), None)


def test_guard_exempts_deferred_paths_and_absent_bounded_delivery():
    _require_owned_source_rendered([], _plan(deferred=('shop/pricing.py',)),
                                   {'shop/pricing.py': SOURCE})
    _require_owned_source_rendered([], _plan(bounded=False), {})


def test_guard_ignores_units_outside_retained_handles():
    _require_owned_source_rendered([], _plan(retained=('unit:other',)), {})


def test_guard_rejects_malformed_retained_handles():
    plan = _plan()
    plan['bounded_delivery']['retained_handles'] = 'unit:a'
    with pytest.raises(FrameFit5Error, match='malformed retained handles'):
        _require_owned_source_rendered([], plan, {'shop/pricing.py': SOURCE})
