"""Data workspace evidence packet contract: default-off switch, typed counts, invariants, receipts, budget fit."""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from stateframe import runner
from stateframe.engine import kernel_data_packet as dp


def _qualifiers(**coverage: object) -> dp.Qualifiers:
    return dp.Qualifiers(coverage=dict(coverage))


def _count(value: int | None = 3, certainty: str = "exact", basis: str = "full_scan") -> dp.TypedCount:
    return dp.TypedCount(value=value, certainty=certainty, basis=basis)  # type: ignore[arg-type]


def _packet(**overrides: object) -> dp.EvidencePacket:
    base = dp.EvidencePacket(
        status="ok",
        revision="r7",
        qualifiers=_qualifiers(complete=True),
        counts={"rows_scanned": _count(3)},
        rows=({"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": None}),
        aggregates={"total": _count(3)},
        error=None,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def _code(exc_info: pytest.ExceptionInfo[dp.PacketInvariantError]) -> str:
    return exc_info.value.code


# --- switch -----------------------------------------------------------------

def test_switch_default_is_off():
    assert dp.data_workspace_enabled({}) is False


def test_switch_on_enables_and_is_case_and_space_tolerant():
    assert dp.data_workspace_enabled({dp.SWITCH_ENV: "on"}) is True
    assert dp.data_workspace_enabled({dp.SWITCH_ENV: " ON "}) is True


@pytest.mark.parametrize("value", ["maybe", "", "1", "true"])
def test_switch_unknown_or_empty_fails_closed(value):
    with pytest.raises(ValueError, match=dp.SWITCH_ENV):
        dp.data_workspace_enabled({dp.SWITCH_ENV: value})


def test_runner_opt_in_switches_has_data_workspace_entry():
    assert runner.OPT_IN_SWITCHES[dp.SWITCH_ENV] == ("off", "on")


# --- TypedCount -------------------------------------------------------------

def test_typed_count_rejects_bool_as_int():
    with pytest.raises(dp.PacketInvariantError) as exc:
        dp.TypedCount(value=True, certainty="exact", basis="full_scan")  # type: ignore[arg-type]
    assert _code(exc) == "typed_count_invalid"


def test_typed_count_from_dict_rejects_bool_as_int():
    with pytest.raises(dp.PacketInvariantError) as exc:
        dp.TypedCount.from_dict({"value": False, "certainty": "exact", "basis": "full_scan"})
    assert _code(exc) == "typed_count_invalid"


@pytest.mark.parametrize(
    ("value", "certainty"),
    [(None, "exact"), (4, "unavailable"), (-1, "exact"), (1, "bogus")],
)
def test_typed_count_invariants(value, certainty):
    with pytest.raises(dp.PacketInvariantError) as exc:
        dp.TypedCount(value=value, certainty=certainty, basis="full_scan")  # type: ignore[arg-type]
    assert _code(exc) == "typed_count_invalid"


def test_typed_count_requires_basis():
    with pytest.raises(dp.PacketInvariantError):
        dp.TypedCount(value=1, certainty="exact", basis="")


def test_typed_count_unavailable_has_none_value_and_round_trips():
    count = dp.TypedCount(value=None, certainty="unavailable", basis="catalog_stats")
    assert dp.TypedCount.from_dict(count.to_dict()) == count


# --- invariants a-e ---------------------------------------------------------

def test_valid_ok_packet_passes():
    _packet().validate()


def test_a_failed_requires_error():
    with pytest.raises(dp.PacketInvariantError) as exc:
        _packet(status="failed", rows=(), aggregates={}, error=None).validate()
    assert _code(exc) == "failed_without_error"


def test_a_failed_has_no_results():
    with pytest.raises(dp.PacketInvariantError) as exc:
        _packet(status="failed", aggregates={}, error="boom").validate()
    assert _code(exc) == "failed_has_result"
    with pytest.raises(dp.PacketInvariantError) as exc:
        _packet(status="failed", rows=(), aggregates={"total": _count(1)}, error="boom").validate()
    assert _code(exc) == "failed_has_result"


def test_failed_with_error_and_no_results_is_valid():
    _packet(status="failed", rows=(), aggregates={}, error="warehouse timeout").validate()


def test_b_ok_or_empty_rejects_error():
    with pytest.raises(dp.PacketInvariantError) as exc:
        _packet(status="ok", error="stale").validate()
    assert _code(exc) == "success_with_error"
    with pytest.raises(dp.PacketInvariantError) as exc:
        _packet(status="empty", rows=(), error="stale").validate()
    assert _code(exc) == "success_with_error"


def test_c_empty_with_rows_is_invalid():
    with pytest.raises(dp.PacketInvariantError) as exc:
        _packet(status="empty", rows=({"id": 1},)).validate()
    assert _code(exc) == "empty_with_rows"


def test_ok_with_zero_rows_is_valid_and_empty_with_no_rows_is_valid():
    _packet(status="ok", rows=()).validate()
    _packet(status="empty", rows=()).validate()


def test_d_exact_aggregate_over_incomplete_coverage_is_invalid():
    incomplete = _packet(qualifiers=_qualifiers(complete=False), aggregates={"total": _count(3)})
    with pytest.raises(dp.PacketInvariantError) as exc:
        incomplete.validate()
    assert _code(exc) == "exact_over_incomplete_coverage"
    missing = _packet(qualifiers=_qualifiers(), aggregates={"total": _count(3)})
    with pytest.raises(dp.PacketInvariantError) as exc:
        missing.validate()
    assert _code(exc) == "exact_over_incomplete_coverage"


def test_d_non_exact_aggregate_over_incomplete_coverage_is_valid():
    estimate = dp.TypedCount(value=10, certainty="lower_bound", basis="preview_rows")
    _packet(qualifiers=_qualifiers(complete=False), aggregates={"total": estimate}).validate()


def test_e_typed_count_invariants_checked_at_validate():
    # Bypass the constructor check on purpose to prove validate() re-checks counts.
    bad = dp.TypedCount(value=1, certainty="exact", basis="full_scan")
    object.__setattr__(bad, "value", None)
    with pytest.raises(dp.PacketInvariantError) as exc:
        _packet(counts={"rows_scanned": bad}).validate()
    assert _code(exc) == "typed_count_invalid"


def test_unknown_status_and_blank_revision_are_rejected():
    with pytest.raises(dp.PacketInvariantError):
        _packet(status="maybe").validate()
    with pytest.raises(dp.PacketInvariantError):
        _packet(revision="").validate()


# --- binding ----------------------------------------------------------------

def test_binding_is_exact_revision_equality():
    packet = _packet(revision="r7")
    assert dp.satisfies_binding(packet, "r7") is True
    assert dp.satisfies_binding(packet, "r8") is False


# --- canonical bytes and receipt -------------------------------------------

def test_canonical_bytes_ignore_dict_insertion_order():
    first = _packet(rows=({"id": 1, "name": "a"},))
    reordered_row = {"name": "a", "id": 1}
    second = _packet(rows=(reordered_row,), qualifiers=dp.Qualifiers(coverage={"complete": True}))
    assert dp.packet_bytes(first) == dp.packet_bytes(second)


def test_receipt_round_trip_is_exact():
    packet = _packet()
    receipt = dp.receipt(packet)
    assert receipt["schema"] == "stateframe-data-packet/1"
    assert receipt["sha256"] == hashlib.sha256(dp.packet_bytes(packet)).hexdigest()
    assert dp.from_receipt(receipt) == packet


def test_tampered_receipt_is_rejected_by_digest():
    receipt = dp.receipt(_packet(revision="r7"))
    receipt["packet"]["revision"] = "r8"
    with pytest.raises(dp.PacketInvariantError) as exc:
        dp.from_receipt(receipt)
    assert _code(exc) == "receipt_digest_mismatch"


def test_receipt_with_wrong_schema_is_rejected():
    receipt = dp.receipt(_packet())
    receipt["schema"] = "stateframe-data-packet/0"
    with pytest.raises(dp.PacketInvariantError) as exc:
        dp.from_receipt(receipt)
    assert _code(exc) == "receipt_schema_invalid"


# --- budget fit -------------------------------------------------------------

def _many_rows(n: int) -> tuple[dict[str, int], ...]:
    return tuple({"id": i, "label": f"row-{i:04d}"} for i in range(n))


def test_fit_rows_keeps_qualifiers_and_drops_tail_rows():
    qualifiers = dp.Qualifiers(coverage={"complete": True}, display={"unit": "rows"})
    packet = _packet(qualifiers=qualifiers, rows=_many_rows(200), aggregates={"total": _count(200)})
    budget = len(dp.packet_bytes(packet)) // 2
    fitted = dp.fit_rows(packet, budget)
    assert len(dp.packet_bytes(fitted)) <= budget
    shown = fitted.qualifiers.display["rows_shown"]
    assert isinstance(shown, int) and 0 < shown < 200
    assert fitted.qualifiers.display["rows_truncated"] is True
    assert fitted.qualifiers.display["unit"] == "rows"
    assert fitted.rows == packet.rows[:shown]
    assert fitted.qualifiers.coverage == packet.qualifiers.coverage
    assert fitted.counts == packet.counts
    assert fitted.aggregates == packet.aggregates
    assert fitted.error == packet.error
    fitted.validate()


def test_fit_rows_is_identity_when_everything_fits():
    packet = _packet()
    assert dp.fit_rows(packet, len(dp.packet_bytes(packet))) == packet


def test_fit_rows_is_maximal():
    packet = _packet(rows=_many_rows(120), aggregates={"total": _count(120)})
    budget = len(dp.packet_bytes(packet)) // 3
    fitted = dp.fit_rows(packet, budget)
    shown = fitted.qualifiers.display["rows_shown"]
    assert isinstance(shown, int)
    bigger = dataclasses.replace(
        packet,
        rows=packet.rows[: shown + 1],
        qualifiers=dp.Qualifiers(
            input=packet.qualifiers.input,
            coverage=packet.qualifiers.coverage,
            numeric=packet.qualifiers.numeric,
            selection=packet.qualifiers.selection,
            storage=packet.qualifiers.storage,
            display={**packet.qualifiers.display, "rows_truncated": True, "rows_shown": shown + 1},
        ),
    )
    assert len(dp.packet_bytes(bigger)) > budget


def test_qualifiers_exceed_budget_when_header_alone_does_not_fit():
    qualifiers = dp.Qualifiers(coverage={"complete": True}, display={"note": "x" * 2000})
    packet = _packet(qualifiers=qualifiers, rows=_many_rows(5))
    with pytest.raises(dp.PacketInvariantError) as exc:
        dp.fit_rows(packet, 500)
    assert _code(exc) == "qualifiers_exceed_budget"


def test_fit_rows_validates_its_input_first():
    with pytest.raises(dp.PacketInvariantError) as exc:
        dp.fit_rows(_packet(status="empty", rows=({"id": 1},)), 10_000)
    assert _code(exc) == "empty_with_rows"
