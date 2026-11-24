import json
import random

from kernel_work_record import MAX_RENDER_CHARS, WorkRecordStore


def _payload(rendered: str) -> dict:
    assert rendered.startswith("[WORK RECORD]\n")
    assert rendered.endswith("\n[/WORK RECORD]")
    return json.loads(rendered[len("[WORK RECORD]\n"):-len("\n[/WORK RECORD]")])


def _state(goals: list[dict], steps: list[dict], decisions: list[dict] | None = None) -> dict:
    return {"schema": "stateframe-work-record/1", "goals": goals, "steps": steps,
            "decisions": decisions or []}


def _open_counts(state: dict) -> tuple[int, int]:
    return (sum(row["status"] in {"open", "blocked"} for row in state["goals"]),
            sum(row["status"] in {"todo", "doing", "blocked"} for row in state["steps"]))


def test_31_open_steps_keep_a_visible_prefix_at_the_default_cap():
    state = _state(
        [{"id": "goal-0", "title": "Implement the task", "status": "open",
          "fields": {"private": "removed in fallback"}}],
        [{"id": f"step-{index}", "goal_id": "goal-0",
          "text": f"Step {index}: " + "x" * 380, "status": "todo",
          "fields": {"private": "removed in fallback"}} for index in range(31)]
        + [{"id": "step-done", "goal_id": "goal-0", "text": "Finished",
            "status": "done", "fields": {}}],
        [{"id": "decision-0", "choice": "Use the current plan", "basis_refs": [],
          "fields": {}}],
    )

    rendered = WorkRecordStore.from_state(state).render()
    payload = _payload(rendered)
    shown_goals = payload["goals"]["open"]
    shown_steps = payload["steps"]["open"]
    omitted = payload["omitted"]

    assert len(rendered) <= MAX_RENDER_CHARS
    assert shown_goals == [{"id": "goal-0", "status": "open", "title": "Implement the task"}]
    assert shown_steps
    assert [row["id"] for row in shown_steps] == [f"step-{i}" for i in range(len(shown_steps))]
    assert all(set(row) == {"id", "status", "text"} for row in shown_steps)
    assert all(set(row) == {"id", "status", "title"} for row in shown_goals)
    assert omitted["goals_open"] == 0
    assert len(shown_goals) + omitted["goals_open"] == 1
    assert len(shown_steps) + omitted["steps_open"] == 31
    assert payload["goals"]["done"] == payload["goals"]["dropped"] == []
    assert payload["steps"]["done"] == payload["steps"]["dropped"] == []
    assert payload["decisions"] == []


def test_long_unicode_goal_titles_keep_their_text_and_order():
    state = _state(
        [{"id": f"goal-{index}", "title": "界🧭" * 120, "status": "blocked",
          "fields": {"detail": "discard this field"}} for index in range(31)],
        [],
    )

    payload = _payload(WorkRecordStore.from_state(state).render())
    shown = payload["goals"]["open"]

    assert shown
    assert [row["id"] for row in shown] == [f"goal-{index}" for index in range(len(shown))]
    assert all(row["title"] == "界🧭" * 120 for row in shown)
    assert all(set(row) == {"id", "status", "title"} for row in shown)
    assert len(shown) + payload["omitted"]["goals_open"] == 31


def test_tiny_cap_returns_empty_summary_with_open_counts():
    state = _state(
        [{"id": "goal-0", "title": "界🧭" * 80, "status": "open", "fields": {}}],
        [{"id": "step-0", "goal_id": "goal-0", "text": "x" * 400,
          "status": "doing", "fields": {}}],
    )

    rendered = WorkRecordStore.from_state(state).render(max_chars=256)
    payload = _payload(rendered)

    assert len(rendered) <= 256
    assert payload["goals"]["open"] == []
    assert payload["steps"]["open"] == []
    assert payload["omitted"]["goals_open"] == 1
    assert payload["omitted"]["steps_open"] == 1


def test_final_fallback_counts_hidden_obligation_steps():
    state = _state(
        [{"id": "goal-0", "title": "Implement the task", "status": "open",
          "fields": {}}],
        [{"id": f"step-{index}", "goal_id": "goal-0", "text": "x" * 400,
          "status": "todo", "fields": {"obligation_id": f"obl-{index}"}}
         for index in range(15)],
    )

    payload = _payload(WorkRecordStore.from_state(state).render(max_chars=1024))

    assert payload["omitted"]["render_compacted"] is True
    assert payload["omitted"]["steps_open"] == 14
    assert payload["omitted"]["obligations_not_shown"] == 14
    assert payload["notice"] == "(14 more obligations not shown; all are tracked)"

    retrimmed = _payload(WorkRecordStore.from_state(state).render(max_chars=800))
    assert retrimmed["steps"]["open"] == []
    assert retrimmed["omitted"]["steps_open"] == 15
    assert retrimmed["omitted"]["obligations_not_shown"] == 15

    tiny = _payload(WorkRecordStore.from_state(state).render(max_chars=256))
    assert tiny["steps"]["open"] == []
    assert tiny["omitted"]["steps_open"] == 15
    assert "obligations_not_shown" not in tiny["omitted"]


def test_records_that_fit_match_the_parent_copy_golden_bytes():
    state = _state(
        [{"id": "goal-0", "title": "Keep original bytes", "status": "open",
          "fields": {"owner": "host"}}],
        [{"id": "step-0", "goal_id": "goal-0", "text": "A compact step 🧭",
          "status": "todo", "fields": {"priority": 1}}],
        [{"id": "decision-0", "choice": "Keep the first path", "basis_refs": ["task"],
          "fields": {}}],
    )
    # Captured from .tmp/work-record-attempt2/parent before editing the renderer.
    expected = (
        "[WORK RECORD]\n"
        '{"decisions":[{"basis_refs":["task"],"choice":"Keep the first path",'
        '"fields":{},"id":"decision-0"}],"goals":{"done":[],"dropped":[],"open":'
        '[{"fields":{"owner":"host"},"id":"goal-0","status":"open",'
        '"title":"Keep original bytes"}]},"omitted":{"goals_done":0,"goals_dropped":0,'
        '"goals_open":0,"steps_done":0,"steps_dropped":0,"steps_open":0},'
        '"schema":"stateframe-work-record/1","steps":{"done":[],"dropped":[],"open":'
        '[{"fields":{"priority":1},"goal_id":"goal-0","id":"step-0","status":"todo",'
        '"text":"A compact step 🧭"}]}}\n[/WORK RECORD]'
    )

    assert WorkRecordStore.from_state(state).render() == expected


def test_visible_and_omitted_open_counts_match_for_200_fixed_random_records():
    rng = random.Random(418_2026)
    goal_statuses = ("open", "blocked", "done", "dropped")
    step_statuses = ("todo", "doing", "blocked", "done", "dropped")

    for case in range(200):
        goals = []
        for index in range(rng.randint(1, 45)):
            status = rng.choice(goal_statuses)
            title = rng.choice(("Short", "界🧭" * 40, "g" * rng.randint(1, 240)))
            row = {"id": f"goal-{case}-{index}", "title": title, "status": status,
                   "fields": {"extra": "x" * rng.randint(0, 160)}}
            if status == "dropped":
                row["drop_reason"] = "No longer needed"
            goals.append(row)

        steps = []
        for index in range(rng.randint(0, 60)):
            status = rng.choice(step_statuses)
            text = rng.choice(("Short step", "界🧭" * 100, "s" * rng.randint(1, 400)))
            row = {"id": f"step-{case}-{index}", "goal_id": rng.choice(goals)["id"],
                   "text": text, "status": status,
                   "fields": {"extra": "y" * rng.randint(0, 160)}}
            if status == "dropped":
                row["drop_reason"] = "No longer needed"
            steps.append(row)

        decisions = [{"id": f"decision-{case}-{index}", "choice": "Keep this choice",
                      "basis_refs": [], "fields": {}}
                     for index in range(rng.randint(0, 12))]
        state = _state(goals, steps, decisions)
        cap = rng.choice((256, 512, 1024, MAX_RENDER_CHARS))
        payload = _payload(WorkRecordStore.from_state(state).render(max_chars=cap))
        goals_total, steps_total = _open_counts(state)

        assert len(payload["goals"]["open"]) + payload["omitted"].get("goals_open", 0) == goals_total
        assert len(payload["steps"]["open"]) + payload["omitted"].get("steps_open", 0) == steps_total
