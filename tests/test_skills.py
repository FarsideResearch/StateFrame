"""Skill validation, bounded loading, and registry contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from stateframe import skills as host
from kernel_skills import SkillRuntime, note_budget_omission
from kernel_agent_schemas import OBSERVATION_NAMES
from kernel_bounded_locate_code import OBSERVATION_TOOLS, READ_TOOLS
from kernel_completion_owner import TOOL_SPECS
from kernel_continuation_evidence import READ_ONLY, READ_OUTCOMES, READ_RESULT_CLASSES, READ_RECEIPT_KINDS
from kernel_continuation_menu_authority import EFFECT_TOOLS
from kernel_lean import _KNOWN_TOOL_NAMES
from kernel_request_budget import _CORE_TOOLS, _OPTIONAL_SECTIONS, _DIET_PROTECTED_SECTIONS
from kernel_request_diet import SECTION_NAMES, STAGE_SECTION_ALLOWLIST, _DROP_ORDER, _PROTECTED_SECTIONS, apply_diet, RequestTokenMeter
from kernel_request_diet import section_name
from kernel_request_tool_surface import _REQUEST_TOOL_FLOOR
from probes.react_repair import TOOLS


def make_skill(root: Path, name: str = "example-skill", *, body: str = "Follow this instruction.\n",
               description: str = "A useful instruction bundle.", tools: dict | None = None) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}", encoding="utf-8")
    if tools is not None:
        (directory / "tools.json").write_text(json.dumps(tools), encoding="utf-8")
    return directory


@pytest.mark.parametrize("replacement,fragment", [
    ("name: other-name", "name must match"),
    ("name: example-skill\nname: example-skill", "duplicate"),
    ("description: ", "description"),
    ("bad frontmatter", "malformed"),
    ("name:\texample-skill", "malformed"),
    ("name: example-skill\nunknown: a\nunknown: b", "duplicate"),
])
def test_frontmatter_refusals(tmp_path, replacement, fragment):
    directory = make_skill(tmp_path)
    file = directory / "SKILL.md"
    text = file.read_text()
    text = text.replace("name: example-skill", replacement, 1) if "name" in replacement or "bad" in replacement else text.replace("description: A useful instruction bundle.", replacement)
    if replacement == "bad frontmatter":
        text = text.replace("name: example-skill", replacement, 1)
    file.write_text(text)
    with pytest.raises(host.SkillError, match=fragment):
        host.parse_skill(directory)


def test_validation_limits_and_symlinks(tmp_path):
    directory = make_skill(tmp_path, body="x" * (32 * 1024 + 1))
    with pytest.raises(host.SkillError, match="32 KiB"):
        host.parse_skill(directory)
    (directory / "SKILL.md").write_text("---\nname: example-skill\ndescription: Good\n---\n")
    with pytest.raises(host.SkillError, match="nonempty"):
        host.parse_skill(directory)
    (directory / "SKILL.md").unlink()
    source = tmp_path / "outside.md"
    source.write_text("---\nname: example-skill\ndescription: Good\n---\nbody")
    (directory / "SKILL.md").symlink_to(source)
    with pytest.raises(host.SkillError, match="symlinks"):
        host.parse_skill(directory)


def test_frontmatter_requires_exact_delimiters_and_utf8(tmp_path):
    directory = make_skill(tmp_path)
    file = directory / "SKILL.md"
    file.write_text(file.read_text().replace("---\n", " ---\n", 1))
    with pytest.raises(host.SkillError, match="frontmatter must start"):
        host.parse_skill(directory)
    file.write_bytes(b"---\nname: example-skill\ndescription: good\n---\n\xff")
    with pytest.raises(host.SkillError, match="UTF-8"):
        host.parse_skill(directory)


def test_install_list_remove_and_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = make_skill(tmp_path)
    (source / "extra.txt").write_text("copied but never shown")
    installed = host.install(source)
    assert (installed.path / "extra.txt").read_text() == "copied but never shown"
    assert host.list_installed() == (installed,)
    assert host.resolve_skills(["example-skill"], trusted=False) == (installed,)
    with pytest.raises(host.SkillError, match="already installed"):
        host.install(source)
    with pytest.raises(host.SkillError, match="duplicate selected"):
        host.resolve_skills(["example-skill", str(source)], trusted=False)
    host.remove("example-skill")
    assert host.list_installed() == ()


def test_install_rejects_unsafe_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = make_skill(tmp_path)
    (source / "linked").symlink_to(tmp_path / "outside")
    with pytest.raises(host.SkillError, match="symlinks"):
        host.install(source)
    (source / "linked").unlink()
    (source / "large.bin").write_bytes(b"x" * (1 << 20))
    with pytest.raises(host.SkillError, match="1 MiB"):
        host.install(source)


def test_record_identity_and_missing(tmp_path):
    directory = make_skill(tmp_path)
    skill = host.parse_skill(directory)
    records = [skill.record(trusted=False)]
    digest = host.skills_digest(records)
    assert host.verify_records(records, digest) == (skill,)
    (directory / "SKILL.md").write_text((directory / "SKILL.md").read_text() + "changed")
    with pytest.raises(host.SkillError, match="SKILL.md digest"):
        host.verify_records(records, digest)
    (directory / "SKILL.md").unlink()
    with pytest.raises(host.SkillError, match="SKILL.md digest"):
        host.verify_records(records, digest)


def test_catalog_body_caps_and_eviction(tmp_path):
    names = ["alpha-skill", "beta-skill", "gamma-skill"]
    records = [host.parse_skill(make_skill(tmp_path, name)).record(trusted=False) for name in names]
    runtime = SkillRuntime(records, host.skills_digest(records), None)
    assert [line.split(":")[0] for line in runtime.catalog()["catalog"]] == names
    assert "SKILL_UNKNOWN" in runtime.load("unknown")
    assert runtime.load(names[0]) == "skill alpha-skill loaded"
    assert runtime.load(names[0]) == "skill alpha-skill already loaded"
    runtime.load(names[1])
    assert runtime.load(names[2]) == "skill gamma-skill loaded; evicted: alpha-skill"
    assert runtime.resident == names[1:]
    assert all(message["role"] == "user" for message in runtime.messages())
    assert "evicted: alpha-skill" in runtime.messages()[0]["content"]
    for _ in range(20):
        runtime.load(names[0])
        runtime.load(names[1])
        runtime.load(names[2])
    assert len(runtime.evicted) <= len(names)
    assert not set(runtime.evicted).intersection(runtime.resident)
    huge = make_skill(tmp_path, "oversized-skill", body="x" * 32768)
    record = host.parse_skill(huge).record(trusted=False)
    with pytest.raises(host.SkillError, match="SKILL_TOO_LARGE"):
        SkillRuntime([record], host.skills_digest([record]), None)


def test_catalog_cap_names_selected_skills(tmp_path):
    records = [host.parse_skill(make_skill(tmp_path, f"skill-{i:02d}",
                  description="d" * 300)).record(trusted=False) for i in range(16)]
    with pytest.raises(host.SkillError, match="skills catalog exceeds 1200 tokens.*skill-00"):
        SkillRuntime(records, host.skills_digest(records), None)


def test_budget_omission_notice_and_protected_state(tmp_path):
    directory = make_skill(tmp_path)
    skill = host.parse_skill(directory)
    runtime = SkillRuntime([skill.record(trusted=False)], host.skills_digest([skill.record(trusted=False)]), None)
    runtime.load(skill.name)
    body = runtime.messages()[0]
    messages = [{"role": "system", "content": "guard"},
                {"role": "user", "content": "Task: edit product"},
                {"role": "user", "content": "[StateFrame]\n{}"}, body]
    note_budget_omission(messages, messages.pop())
    assert "skill body example-skill omitted for budget; call skill_load again" in messages[3]["content"]
    assert messages[0]["content"] == "guard"
    assert messages[1]["content"] == "Task: edit product"
    runtime.commit_delivery(messages)
    assert runtime.resident == []
    assert runtime.load(skill.name) == "skill example-skill loaded"


def test_diet_sheds_body_at_33344_edge_and_preserves_authority(tmp_path):
    from kernel_stateframe_framefit5 import PRODUCT_MODE
    skill = host.parse_skill(make_skill(tmp_path, body="instructions\n" * 500))
    runtime = SkillRuntime([skill.record(trusted=False)], host.skills_digest([skill.record(trusted=False)]), None)
    runtime.load(skill.name)
    meter = RequestTokenMeter()
    fixed = [{"role": "system", "content": "guards and protected tests stay in force"},
             {"role": "user", "content": "Task: "},
             {"role": "user", "content": "[StateFrame]\n" +
              json.dumps({"state": {"framefit5": {"mode": PRODUCT_MODE}}})}]
    lo, hi = 0, 81000
    while lo < hi:
        middle = (lo + hi + 1) // 2
        fixed[1]["content"] = "Task: " + "x" * middle
        count, _ = meter.measure({"messages": fixed, "tools": []})
        if count <= 33344 - 100:
            lo = middle
        else:
            hi = middle - 1
    fixed[1]["content"] = "Task: " + "x" * lo
    body = {"messages": [*fixed, *runtime.messages()], "tools": []}
    assert meter.measure(body)[0] > 33344
    apply_diet(body, stage="FREE", max_request_bytes=1 << 20,
               token_budget=33344, token_meter=meter)
    assert body["messages"][0] == fixed[0]
    assert body["messages"][1] == fixed[1]
    assert body["messages"][2] == fixed[2]
    assert all(message.get("_request_budget_section") != "skill_bodies" for message in body["messages"])
    assert any("skill body example-skill omitted for budget; call skill_load again" in str(message.get("content"))
               for message in body["messages"])
    assert meter.measure(body)[0] <= 33344


def test_body_does_not_modify_system_or_task_messages(tmp_path):
    from kernel_stateframe_framefit5 import compose_frame_only, STATE_HEADER, PRODUCT_MODE
    import kernel_skills

    source = [{"role": "system", "content": "protected test policy"},
              {"role": "user", "content": "Task: fix app.py"}]
    prepared = [source[0], {"role": "user", "content": STATE_HEADER + "\n" +
                json.dumps({"state": {"framefit5": {"mode": PRODUCT_MODE}}})}]
    previous = kernel_skills.service()
    try:
        kernel_skills._active = None
        without = compose_frame_only(prepared, source)
        skill = host.parse_skill(make_skill(tmp_path))
        record = skill.record(trusted=False)
        kernel_skills.configure([record], host.skills_digest([record]), None)
        kernel_skills.service().load(skill.name)
        with_skill = compose_frame_only(prepared, source)
    finally:
        kernel_skills._active = previous
    assert without[0:3] == with_skill[0:3]
    assert with_skill[3]["role"] == "user"
    assert "USER-PROVIDED SKILL INSTRUCTIONS" in with_skill[3]["content"]


def test_registry_completeness():
    assert "skill_load" in {tool["function"]["name"] for tool in TOOLS}
    for registry in (TOOL_SPECS, OBSERVATION_NAMES, OBSERVATION_TOOLS, READ_ONLY,
                     READ_OUTCOMES, READ_RESULT_CLASSES, READ_RECEIPT_KINDS,
                     EFFECT_TOOLS, _KNOWN_TOOL_NAMES, _CORE_TOOLS, _REQUEST_TOOL_FLOOR):
        assert "skill_load" in registry, registry
    assert "skill_load" not in READ_TOOLS
    assert {"skills", "skill_bodies"} <= set(SECTION_NAMES)
    assert all("skill_bodies" in values for values in STAGE_SECTION_ALLOWLIST.values())
    assert "skill_bodies" in _DROP_ORDER and "skill_bodies" in _OPTIONAL_SECTIONS
    assert "skill_bodies" not in _PROTECTED_SECTIONS and "skill_bodies" not in _DIET_PROTECTED_SECTIONS
    assert section_name({"role": "user", "content": "body", "_request_budget_section": "skill_bodies"}) == "skill_bodies"
    root = Path(__file__).resolve().parents[1]
    assert '"skill_load"' in (root / "src/stateframe/runner.py").read_text()
    assert '"skill_load"' in (root / "src/stateframe/_child.py").read_text()
    engine = root / "src/stateframe/engine"
    manifest = json.loads((engine / "release-manifest.json").read_text())
    assert manifest["files"]["kernel_skills.py"] == hashlib.sha256(
        (engine / "kernel_skills.py").read_bytes()).hexdigest()
