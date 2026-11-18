"""Location inventory must survive re-encoding after the delivery-truth pass.

Evidence (feat-1731, every arm): a location-encoded open_sites inventory is
stored in the StateFrame message. On the next provider call,
kernel_working_set_facts._update_frame runs over that stored frame and stamps
every open_sites row with current_bytes=None and read_this_request=False. Those
keys are not in location_inventory_candidate's output, so expand_location_inventory
rejects its own encoding with FrameFit5Error at the final receipt comparison.
"""
from __future__ import annotations

import copy
import types
from pathlib import Path

import kernel_open_site_capacity as capacity
import kernel_stateframe_framefit5 as framefit5
import kernel_working_set_facts as facts


def _workspace(root: Path) -> None:
    (root / "app").mkdir(parents=True)
    (root / "app" / "api.py").write_text("".join(f"api line {n}\n" for n in range(1, 27)))
    (root / "app" / "ports.py").write_text("".join(f"port line {n}\n" for n in range(1, 10)))
    (root / "app" / "service.py").write_text("".join(f"svc line {n}\n" for n in range(1, 17)))


def _encoded_frame(root: Path) -> dict[str, object]:
    """Four sites shaped like the evidence: two overlapping windows in one file plus two others."""
    full = [
        framefit5._render_next_site(root, "app/api.py", 13, source="remaining_sites_inventory", literal="create"),
        framefit5._render_next_site(root, "app/api.py", 14, source="remaining_sites_inventory", literal="create"),
        framefit5._render_next_site(root, "app/ports.py", 9, source="remaining_sites_inventory", literal="delete"),
        framefit5._render_next_site(root, "app/service.py", 9, source="remaining_sites_inventory", literal="create"),
    ]
    encoded = capacity.location_inventory_candidate({"open_sites": full, "next_site": full[0]})
    assert encoded is not None
    assert encoded["site_inventory_encoding"] == capacity.LOCATION_MODE
    return encoded


def _apply_delivery_truth_pass(frame: dict[str, object]) -> dict[str, object]:
    """Run the real delivery-truth pass over open_sites only; next_site stays intact.

    Nothing from the current request is shown here, so every row is treated as a
    pointer. next_site is detached first so this isolates the open_sites rows.
    """
    result = copy.deepcopy(frame)
    next_site = result.pop("next_site")
    snapshot = types.SimpleNamespace(current_revisions=(), sources=(), truth_message_indexes=())
    facts._update_frame(result, snapshot, [])
    result["next_site"] = next_site
    return result


def test_location_inventory_round_trips_before_truth_pass(tmp_path: Path) -> None:
    _workspace(tmp_path)
    encoded = _encoded_frame(tmp_path)
    assert len(capacity.expand_location_inventory(encoded, tmp_path)) == 4


def test_expand_accepts_inventory_after_delivery_truth_pass(tmp_path: Path) -> None:
    """The truth pass must leave encoded rows byte-identical, so the stored receipt still expands."""
    _workspace(tmp_path)
    encoded = _encoded_frame(tmp_path)
    stored = _apply_delivery_truth_pass(encoded)
    assert stored["open_sites"] == encoded["open_sites"]
    assert len(capacity.expand_location_inventory(stored, tmp_path)) == 4


def test_truth_pass_still_checks_unencoded_sites(tmp_path: Path) -> None:
    """Full (legacy) open_sites rows are still demoted to pointers when their bytes are not shown."""
    _workspace(tmp_path)
    full = [framefit5._render_next_site(tmp_path, "app/api.py", 13, source="remaining_sites_inventory",
                                        literal="create"),
            framefit5._render_next_site(tmp_path, "app/ports.py", 9, source="remaining_sites_inventory",
                                        literal="delete")]
    frame = {"open_sites": copy.deepcopy(full), "next_site": copy.deepcopy(full[0])}
    stored = _apply_delivery_truth_pass(frame)
    assert all(row.get("read_this_request") is False and row.get("current_bytes") is None
               for row in stored["open_sites"])
