"""A no-op fan-out and an INVALID unit check must be explicit to the parent and model."""
from types import SimpleNamespace as NS

from stateframe.engine import kernel_parallel as parallel


def _outcome(unit_id, *, files=(), check="NOT_RUN", status="COMMITTED_UNVERIFIED"):
    return NS(unit_id=unit_id, status=status, reason=None, detail=None, files=tuple(files),
              contract_ok=True, check=check, usage=NS(requests=1))


def _delta(path):
    return NS(path=path, added_lines=1, removed_lines=0, kind="modified")


def _observation(outcomes):
    admission = NS(refusal=None, adjustments=(), limits=NS(unit_max_requests=4))
    return parallel._observation(admission, tuple(outcomes), "NOT_RUN", "OK", None, None)


def _summary(outcomes):
    units = [NS(id=item.unit_id, owns=("pkg/a.py",), consumes=(), check=NS(selectors=()))
             for item in outcomes]
    result = NS(call_id="c", admission=NS(status="ADMITTED", refusal=None),
                outcomes=tuple(outcomes), final_check="NOT_RUN", integration="OK")
    return parallel.action_receipt_summary(result, NS(call_id="c", units=units))[0]


def test_noop_fanout_says_no_files_changed_in_observation_and_receipt():
    outcomes = [_outcome("a"), _outcome("b")]
    assert "No files changed by any subframe." in _observation(outcomes)
    assert "No files changed by any subframe." in _summary(outcomes)


def test_changed_files_suppress_the_noop_line():
    outcomes = [_outcome("a", files=[_delta("pkg/a.py")]), _outcome("b")]
    assert "No files changed" not in _observation(outcomes)
    assert "No files changed" not in _summary(outcomes)


def test_invalid_check_is_named_and_never_reported_as_all_committed():
    outcomes = [_outcome("a", files=[_delta("pkg/a.py")], check="INVALID")]
    observation = _observation(outcomes)
    summary = _summary(outcomes)
    assert "Check INVALID (not a pass) for: a" in observation
    assert "1 check INVALID (not a pass)" in summary
    assert "check_status=INVALID" in summary
