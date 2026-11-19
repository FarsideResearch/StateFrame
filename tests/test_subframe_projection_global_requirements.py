"""Parent obligations survive the full child task and typed-scope projection path."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from kernel_parallel import _requirements
from kernel_parallel_admission import DirectorySnapshot, admit
from kernel_parallel_projection import project_unit
from kernel_parallel_records import (
    CheckRecipe,
    ParallelLimits,
    ParallelRequest,
    RequirementClause,
    UnitContract,
)
from kernel_stateframe_framefit5 import scope_typed_state_for_unit
from kernel_unit_task import build_public_row


@pytest.mark.parametrize("unit_id", ("a", "b"))
@pytest.mark.parametrize("explicit_id", (False, True), ids=("no-planner-ids", "explicit-id"))
@pytest.mark.parametrize("global_contract", (
    pytest.param({}, id="empty-contract"),
    pytest.param({"subject_binding_ids": []}, id="unbound"),
    pytest.param({"subject_binding_ids": ["B-UNRESOLVED"]}, id="unresolved-binding"),
))
def test_parent_global_requirement_reaches_each_child(
    tmp_path: Path, unit_id: str, explicit_id: bool, global_contract: dict[str, object],
) -> None:
    requirement_id = "R-global"
    requirement_text = "Preserve backward compatibility across all public interfaces."
    unrelated_text = "Keep the unrelated module unchanged."
    runtime = SimpleNamespace(requirements_ledger={
        "requirements": [
            {"id": requirement_id, "exact_clause": requirement_text, "contract": global_contract},
            {"id": "R-unrelated", "exact_clause": unrelated_text,
             "contract": {"subject_binding_ids": ["B-other"]}},
        ],
        "bindings": [{"id": "B-other", "locator": {"path": "other.py"}}],
    })
    requirements = _requirements(runtime)
    assert requirements == (
        RequirementClause(requirement_id, requirement_text, "intersect", ()),
        RequirementClause("R-unrelated", unrelated_text, "intersect", ("other.py",)),
    )

    start_files = {"a.py": b"LEFT = 1\n", "b.py": b"RIGHT = 2\n", "other.py": b"OTHER = 3\n"}
    for path, content in start_files.items():
        (tmp_path / path).write_bytes(content)
    snapshot = DirectorySnapshot(tmp_path, 0)
    requirement_ids = (requirement_id,) if explicit_id else ()
    request = ParallelRequest("global-requirements", tuple(
        UnitContract(
            id=child_id, outcome=f"Update {child_id}.py", owns=(f"{child_id}.py",),
            creates=(), exports=(), consumes=(), reads=(), requirement_ids=requirement_ids,
            check=CheckRecipe((), (), "none"),
        )
        for child_id in ("a", "b")
    ), "2")
    assert all(child.requirement_ids == requirement_ids for child in request.units)
    admission = admit(request, snapshot, requirements, ParallelLimits(),
                      protected=lambda _: False, case_insensitive=False)
    assert admission.status == "ADMITTED", admission.refusal

    spec = project_unit(
        admission, request, unit_id, snapshot, parent_seed="seed", policy_digest="policy",
        failure_projection="", requirements=requirements,
    )
    # Explicit selection preserves the extracted source; implicit selection is global.
    expected = RequirementClause(
        requirement_id, requirement_text, "intersect" if explicit_id else "global", (),
    )
    assert spec.requirement_clauses == (expected,)
    assert spec.contract.requirement_ids == requirement_ids

    row = build_public_row(spec, start_files)
    assert f"Requirement {requirement_id}: {requirement_text}" in row["msg"].splitlines()
    assert "R-unrelated" not in row["msg"]
    assert unrelated_text not in row["msg"]

    task_identity = {
        "source": "original_task_message",
        "sha256": hashlib.sha256(row["msg"].encode("utf-8")).hexdigest(),
    }
    scoped = scope_typed_state_for_unit({"task": task_identity}, spec)
    assert scoped["task"] == task_identity
    scope = scoped["subframe_scope"]
    assert scope["unit_id"] == unit_id
    assert scope["contract"]["requirement_ids"] == list(requirement_ids)
    assert scope["requirements"] == [expected.to_json()]
    assert scoped["open_work"]["ticket_clause_completion"] == "OPEN"
