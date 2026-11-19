"""Work budgets and output parity at the task-intake boundary."""

import ast
from collections import Counter
from time import perf_counter
from types import SimpleNamespace

import pytest

from detreason.front_door import build_state, interpret
import kernel_obligation_seed as seed
import kernel_v46_mechanism_wiring as wiring
from kernel_work_record import WorkRecordStore, brief_for_owner


FILE_COUNT = 4_500
CLAUSE_COUNT = 40
SWITCHES = (
    "STATEFRAME_WORKSPACE_OBJECTS",
    "STATEFRAME_TURN_ATOMIC_PUBLICATION",
    "STATEFRAME_TURN_BOUNDED_QUALIFICATION",
    "STATEFRAME_TURN_DELIVERY_PROOFS",
    "STATEFRAME_TURN_GENERATION_OVERLAP",
    "STATEFRAME_TURN_INDEXED_READS",
    "STATEFRAME_TURN_MINIMAL_LAUNCH",
    "STATEFRAME_TURN_PANELS",
    "STATEFRAME_TURN_REPO_INTEL",
    "STATEFRAME_TURN_TIMING",
)


def test_intake_without_cached_vocabulary_uses_the_two_argument_contract(monkeypatch, tmp_path):
    row = {"msg": "Preserve pricing behavior."}
    runtime = SimpleNamespace(
        owner=SimpleNamespace(task_row=row),
        journal=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(wiring, "enabled", lambda name: name == "front_door")
    compile_contract = wiring._working_contract

    def compile_without_cache(task_row, root):
        return compile_contract(task_row, root)

    monkeypatch.setattr(wiring, "_working_contract", compile_without_cache)
    wiring.task_start(runtime, row, tmp_path)
    assert runtime._v46_working_contract["verdict"]


def test_large_declared_intake_work_budget(monkeypatch, tmp_path):
    for switch in SWITCHES:
        monkeypatch.setenv(switch, "on")
    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", "on")
    monkeypatch.setenv("STATEFRAME_DETREASON_FRONT_DOOR", "on")
    monkeypatch.setenv("STATEFRAME_DETREASON_SUFFICIENCY", "off")
    monkeypatch.setenv("STATEFRAME_QGATES", "off")
    files = {
        f"pkg/module{index}.py": f"def operation{index}():\n    return 'source marker'\n"
        for index in range(FILE_COUNT)
    }
    row = {
        "files": files,
        "msg": "\n".join(f"Preserve module{index}." for index in range(CLAUSE_COUNT)),
    }
    owner = SimpleNamespace(task_row=row)
    runtime = SimpleNamespace(owner=owner, journal=SimpleNamespace(emit=lambda *_a, **_k: None))
    path_calls = Counter()
    word_calls = Counter()
    parse_calls = Counter()
    comparisons = Counter()
    path_factory, regex, parse = seed.PurePosixPath, seed.re, ast.parse

    def count_path(path):
        path_calls[path] += 1
        return path_factory(path)

    class CountedWords:
        def __init__(self, pattern):
            self.pattern = pattern

        def findall(self, source):
            if "source marker" in source:
                word_calls[source] += 1
            return self.pattern.findall(source)

    class CountedRegex:
        def __getattr__(self, name):
            return getattr(regex, name)

        def search(self, pattern, text, *args, **kwargs):
            if pattern.startswith(r"\bmodule"):
                comparisons["alias_candidates"] += 1
            return regex.search(pattern, text, *args, **kwargs)

    def count_parse(source, *args, **kwargs):
        if isinstance(source, str) and "source marker" in source:
            parse_calls[source] += 1
        return parse(source, *args, **kwargs)

    monkeypatch.setattr(seed, "PurePosixPath", count_path)
    monkeypatch.setattr(seed, "_WORD", CountedWords(seed._WORD))
    if hasattr(seed, "_SOURCE_WORD"):
        monkeypatch.setattr(seed, "_SOURCE_WORD", CountedWords(seed._SOURCE_WORD))
    monkeypatch.setattr(seed, "re", CountedRegex())
    monkeypatch.setattr(ast, "parse", count_parse)

    start = perf_counter()
    record = WorkRecordStore(brief_for_owner(owner))
    wiring.task_start(runtime, row, tmp_path)
    elapsed = perf_counter() - start

    assert len(record.steps) == CLAUSE_COUNT
    assert runtime._v46_working_contract["verdict"] in {"PROCEED", "ASK_ONE", "AMBIGUOUS"}
    observed = {
        "file name extractions": sum(path_calls.values()),
        "source word extractions": sum(word_calls.values()),
        "per-file name maximum": max(path_calls.values(), default=0),
        "per-file word maximum": max(word_calls.values(), default=0),
        "clause-by-file alias comparisons": comparisons["alias_candidates"],
        "AST parses": sum(parse_calls.values()),
        "duplicate AST parses": sum(count - 1 for count in parse_calls.values()),
    }
    print(f"intake counts={observed} elapsed={elapsed:.6f}s")
    assert len(path_calls) == FILE_COUNT
    assert observed["per-file name maximum"] <= 1
    assert len(word_calls) == FILE_COUNT
    assert observed["per-file word maximum"] <= 1
    assert observed["clause-by-file alias comparisons"] == 0
    assert observed["AST parses"] <= 16
    assert observed["duplicate AST parses"] == 0
    assert elapsed < 10.0

    # The same filename/content basis can already be present in the parse cache.
    # A changed declaration must still be parsed and produce fresh names.
    subset = dict(list(files.items())[:16])
    assert "operation0" in build_state(files=subset).symbols
    assert sum(parse_calls.values()) == observed["AST parses"]
    subset["pkg/module0.py"] = "def replacement():\n    return 'source marker'\n"
    updated = build_state(files=subset)
    assert "replacement" in updated.symbols and "operation0" not in updated.symbols
    assert sum(parse_calls.values()) == observed["AST parses"] + 1


@pytest.fixture(params=["on", "off"])
def switches(request, monkeypatch):
    for switch in SWITCHES:
        monkeypatch.setenv(switch, request.param)
    monkeypatch.setenv("STATEFRAME_OBLIGATION_SEED", request.param)
    return request.param


def test_clause_grounding_order_and_boundaries(switches):
    files = {
        "z/usage.py": "other = FOO + foo # comment\n",
        "a/Report-Detail.PY": "# module\n",
        "b/report-detail.py": "",
        "pkg/api.py": "",
        "z/delivery.py": "",
        "pkg/models.py": "",
        "pkg/cli.py": "",
        "same/target.py": "",
        "other/target.py": "",
        "pkg/not_report_detail.py": "foo_more = 1\n",
        "pkg/unicode.py": "éFooé = 1\n",
        "pkg/spell.py": "Kest = 0\n",
        "pkg/digits.py": "123foo = 0\n",
        "pkg/punct.other": "FOO = 0\n",
        "a\\archive.py": "",
        "pkg/。start!.py": "",
        "edge/.dash": "",
        ".": "",
    }
    brief = (
        "Focus: Keep report-detail, caller, and request with `Foo` and `Kest`.\n"
        "Focus: Preserve same/target.py while loading pkg/archive.py.\n"
        "Focus: Preserve x。start!.py and x.dash while ending.\n"
        "Keep xreport-detailx intact without breaking foobar.\n"
        "Preserve end-to-end."
    )
    obligations = seed.extract_obligations(brief, {"files": files})
    assert [item.id for item in obligations] == [f"obl-{index:03d}" for index in range(6)]
    assert [item.kind for item in obligations] == ["constraint"] * 4 + ["compat", "rename"]
    period = "" if switches == "on" else "."
    assert [item.text for item in obligations] == [
        *brief.splitlines()[:3], "Keep xreport-detailx intact", "without breaking foobar" + period,
        "Preserve end-to-end" + period,
    ]
    assert [item.unmappable for item in obligations] == [True] * 3 + [False] * 3
    assert [item.mentioned_symbols for item in obligations] == [
        ("Foo", "Kest", "FOO", "foo"), (), (), (), (), (),
    ]
    assert [item.mentioned_files for item in obligations] == [
        ("z/usage.py", "a/Report-Detail.PY", "b/report-detail.py", "pkg/cli.py",
         "pkg/spell.py", "pkg/punct.other", ".", "pkg/api.py", "z/delivery.py", "pkg/models.py"),
        ("same/target.py", "other/target.py", "a/archive.py", "."),
        ("pkg/。start!.py", "edge/.dash", "."),
        (".",),
        (".",),
        (".", *(path.replace("\\", "/") for path in list(files)[:-1])),
    ]
    explicit, contextual = seed.extract_obligations(
        "Focus: Preserve pkg/.hidden.py.\nFocus: Keep upperσ intact.",
        {"files": {"pkg/.hidden.py": "", "other/.hidden.py": "", "pkg/upperΣ.py": ""}},
    )
    assert explicit.mentioned_files == ("pkg/.hidden.py",)
    assert contextual.mentioned_files == ("pkg/upperΣ.py",)


def test_grounding_truncates_after_deduplication_and_in_input_order(switches):
    paths = [f"pkg{index:02d}/models.py" for index in reversed(range(32))]
    files = [
        {"path": path, "source": "prior = 1"}
        for path in [paths[0], *paths]
    ]
    quoted = " ".join(f"symbol{index:02d}" for index in reversed(range(30)))
    match, = seed.extract_obligations(f"Focus: Keep model end-to-end `{quoted} prior`.",
                                       {"files": files})
    assert match.mentioned_files == tuple(paths[:24])
    assert match.mentioned_symbols == tuple(quoted.split()[:24])


def test_front_door_definition_and_confidence_parity(switches):
    files = {
        "pkg/the_service.py": (
            "API_LEVEL = 3\nclass Service:\n    async def handle(self):\n"
            "        def reconcile():\n            pass\n"
            "        return reconcile()\n\n"
            "def rate(total):\n    return total\n"
        ),
        "pkg/client.py": "from the_service import rate\nrate(1, 2)\n",
        "pkg/partial.py": "def incomplete(",
        "pkg/text.txt": "def custom():\n    pass\n",
    }
    state = build_state(files=files)
    assert state.symbols == {
        "custom": ["pkg/text.txt"], "handle": ["pkg/the_service.py"],
        "rate": ["pkg/the_service.py"], "reconcile": ["pkg/the_service.py"],
    }
    assert [(row["kind"], row["name"]) for row in state.violations] == [
        ("arity_mismatch", "rate"), ("parse_error", "pkg/partial.py"),
    ]
    for brief, verdict, digest in [
        ("Fix rate which crashes.", "ASK_ONE", "a06d17b17dc966f51a78cb3e76276637e1e72419cb827409507c15dd64f1beea"),
        ("Fix reconcile.", "PROCEED", "fc1a61d7745d16418e44ef5c441a855a04457cdb7aeb27d7b083203e700eccb8"),
        ("Fix Service.", "PROCEED", "de5aa4b06e876ea5b8f60b778eaf0e1c76b2fb450117a6538e30bbee66f37a77"),
        ("Investigate nonexistent.", "PROCEED", "84462d1f1a339300e257a4975a72cbe55826939b43d83c20513b9ca9fa66d503"),
    ]:
        result = interpret(brief, extra_context={"state": state})
        assert result.verdict == verdict
        assert wiring._digest(result.to_dict()) == "sha256:" + digest

    classes = build_state(files={
        "only.py": "class Example:\n    class Inner:\n        pass\n",
        "plain.txt": "class Outer: pass\n",
    })
    assert classes.symbols == {"Example": ["only.py"], "Inner": ["only.py"]}
    assert interpret("Fix Inner.", extra_context={"state": classes}).verdict == "PROCEED"
    assert interpret("Fix Outer.", extra_context={"state": classes}).verdict == "ASK_ONE"


def test_front_door_preserves_the_declared_intake_window(switches, tmp_path):
    (tmp_path / "live.py").write_text("def live_target(): pass\n", encoding="utf-8")
    row = {
        "files": {
            "clip.py": {
                "parent_src": "# padding\n" * 12_000 + "def beyond_limit(): pass\n",
                "source": "def wrong_source(): pass\n",
            },
            **{f"file{index}.py": f"def operation{index}(): pass\n" for index in range(1, 17)},
        },
        "rel": "live.py",
    }
    projected = wiring._row_files(row, tmp_path)
    assert list(projected) == ["clip.py", *(f"file{index}.py" for index in range(1, 16)), "live.py"]
    assert len(projected["clip.py"]) == 120_000
    assert projected["clip.py"].startswith("# padding")
    assert projected["live.py"] == "def live_target(): pass\n"
    for name, verdict in (
        ("file15.py", "PROCEED"), ("operation15", "PROCEED"), ("operation16", "ASK_ONE"),
        ("beyond_limit", "ASK_ONE"), ("wrong_source", "ASK_ONE"), ("live.py", "ASK_ONE"),
    ):
        contract = wiring._working_contract({**row, "msg": f"Fix {name}."}, tmp_path)
        assert contract["verdict"] == verdict, name
