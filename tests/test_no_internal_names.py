"""Keep private project and measurement labels out of shipped prose."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import io
from pathlib import Path
import re
import tokenize


_PATTERNS = {
    "full_project_name": re.compile(r"\b" + "Mission" + r"\s+" + "Control" + r"\b", re.IGNORECASE),
    "project_alias": re.compile(
        r"\b(?:" + "A" + "HA|" + "M" + "C" + r")\b", re.IGNORECASE
    ),
    "numbered_rule": re.compile(
        r"\b" + r"rule\s*" + r"#\s*\d+[a-z]?" + r"(?:\s*/\s*#?\d+[a-z]?)*",
        re.IGNORECASE,
    ),
    "personal_host": re.compile(
        r"\b(?:crew-\d+|" + "su" + "san|" + "mi" + "ke|" + "found" + "ry|"
        + "mari" + "da|" + "farsi" + "de|" + r"team\s+op" + "us|sf-" + "work)"
        + "|" + "sf-test" + "-runners" + r"\b|"
        + r"/home/" + "mi" + "ke" + r"/",
        re.IGNORECASE,
    ),
    "machine_label": re.compile(r"\bhost_tok" + r"28\b", re.IGNORECASE),
    "private_pattern": re.compile("item28" + r"\[bc\]", re.IGNORECASE),
    "run_label": re.compile(
        r"\b(?:r2[a-z]{1,2}|r\d{2}[a-z]{0,2}|b\d{4,})\b", re.IGNORECASE
    ),
    "run_detail": re.compile(r"\b(?:batch|seq)\s*\d+\b", re.IGNORECASE),
    "contributor_label": re.compile(r"\bldev_\d+\b", re.IGNORECASE),
    "operator_label": re.compile(
        "FRAME" + "LAB|" + "OPERATOR" + "2|" + "RUN" + "REAL|" + "BUG" + "HUNT",
        re.IGNORECASE,
    ),
    "lab_version": re.compile(r"\btoken[ -]?" + "lab" + r"\s+v\d+", re.IGNORECASE),
    "lab_driver_path": re.compile(r"\bbench/" + "lab" + r"(?:/[^\s]+)?", re.IGNORECASE),
    "lab_repository": re.compile(r"\blab" + r"-repo\b", re.IGNORECASE),
    "lab_only": re.compile(r"\blab" + r"[- ]only\b", re.IGNORECASE),
    "synthetic_label": re.compile("permission" + r"-control", re.IGNORECASE),
}


@dataclass(frozen=True)
class _Allow:
    expected: int
    reason: str


_ALLOWLIST = {
    ("tests/test_allocation_replay_corpus.py", "personal_host"): _Allow(
        4, "the corpus scrub test names local identities it must reject"
    ),
    ("tests/test_allocation_replay_corpus.py", "machine_label"): _Allow(
        1, "the corpus scrub test rejects its known host marker"
    ),
    ("tests/test_allocation_replay_corpus.py", "private_pattern"): _Allow(
        1, "the corpus scrub test rejects both variants of a private marker"
    ),
    ("tests/test_engine_run.py", "synthetic_label"): _Allow(
        3, "stable subprocess label used by the permission-error regression"
    ),
    ("tests/test_check_selection.py", "project_alias"): _Allow(
        3, "repository selector key and temporary path in an oracle fixture"
    ),
    ("tests/test_final_cache_layout.py", "run_label"): _Allow(
        1, "synthetic request identifier in a test fixture, unrelated to a measurement lane"
    ),
    ("tests/test_shell_policy.py", "run_label"): _Allow(
        1, "carriage-return progress bytes in a shell normalization fixture"
    ),
    ("tests/test_lab_batch_drivers.py", "lab_driver_path"): _Allow(
        1, "the test loads extracted maintenance drivers from their shipped path"
    ),
    ("tests/test_lab_driver_bindings.py", "lab_driver_path"): _Allow(
        1, "the test verifies extracted driver imports keep their shipped bindings"
    ),
    ("tests/test_lab_move_integrity.py", "lab_driver_path"): _Allow(
        3, "the moved-driver regression checks the exact maintenance script paths"
    ),
}

_CANDIDATE_HINTS = (
    "mission" + " control",
    "a" + "ha", "m" + "c", "rule #", "crew-", "su" + "san", "mi" + "ke", "found" + "ry",
    "mari" + "da", "farsi" + "de", "team" + " opus", "sf-" + "work",
    "sf-test" + "-runners", "host_tok" + "28", "item28" + "[bc]",
    "ldev_", "frame" + "lab", "operator" + "2",
    "run" + "real", "bug" + "hunt", "token lab", "token-lab", "tokenlab v",
    "bench/" + "lab", "lab" + "-repo", "lab" + "-only", "permission" + "-control",
)
_RUN_CANDIDATE = re.compile(
    r"\b(?:r2[a-z]{1,2}|r\d{2}[a-z]{0,2}|b\d{4,})\b", re.IGNORECASE
)
_RUN_DETAIL_CANDIDATE = re.compile(r"\b(?:batch|seq)\s*\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class _Context:
    path: str
    line: int
    kind: str
    text: str


def _contexts(path: Path, source: str) -> list[_Context]:
    relative = path.as_posix()
    if relative.endswith("/kernel_semantics.py"):
        return []

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    doc_positions: set[tuple[int, int]] = set()
    names: list[_Context] = []
    if relative.startswith("tests/"):
        for match in re.finditer(
            r"^\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]*)\b|"
            r"^\s*class\s+(Test[A-Za-z0-9_]*)\b",
            source,
            re.MULTILINE,
        ):
            name = match.group(1) or match.group(2)
            names.append(
                _Context(relative, source.count("\n", 0, match.start()) + 1,
                         "test-name", name.replace("_", " "))
            )
    else:
        string_candidates = [
            token for token in tokens
            if token.type == tokenize.STRING
            and any(pattern.search(token.string) for pattern in _PATTERNS.values())
        ]
        if string_candidates:
            tree = ast.parse(source, filename=relative)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                    first = node.body[0]
                    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                            and isinstance(first.value.value, str)):
                        doc_positions.add((first.value.lineno, first.value.col_offset))

    contexts = names
    for token in tokens:
        if token.type == tokenize.COMMENT:
            if relative.endswith("/repo_map.py") and token.start[0] in {336, 337}:
                continue
            contexts.append(_Context(relative, token.start[0], "comment", token.string))
        elif token.type == tokenize.STRING and token.start in doc_positions:
            if relative.endswith("/repo_map.py") and token.start[0] in {336, 337}:
                continue
            contexts.append(_Context(relative, token.start[0], "docstring", token.string))
        elif relative.startswith("tests/") and token.type == tokenize.STRING:
            contexts.append(_Context(relative, token.start[0], "test-literal", token.string))
    return contexts


def _find_private_labels(contexts: list[_Context], *, enforce_allowlist: bool = True):
    violations = []
    allowed_counts: dict[tuple[str, str], int] = {}
    for context in contexts:
        for name, pattern in _PATTERNS.items():
            matches = list(pattern.finditer(context.text))
            if not matches:
                continue
            key = (context.path, name)
            allow = _ALLOWLIST.get(key)
            if context.kind == "test-literal" and allow is not None:
                allowed_counts[key] = allowed_counts.get(key, 0) + len(matches)
                continue
            for match in matches:
                violations.append(
                    f"{context.path}:{context.line} [{context.kind}] {name}: {match.group()}"
                )

    if enforce_allowlist:
        for key, allow in _ALLOWLIST.items():
            actual = allowed_counts.get(key, 0)
            if actual != allow.expected:
                violations.append(
                    f"{key[0]} allowlist {key[1]} expected {allow.expected}, found {actual}: {allow.reason}"
                )
        unexpected = set(allowed_counts) - set(_ALLOWLIST)
        violations.extend(f"unexpected allowlist hit: {path} [{name}]" for path, name in sorted(unexpected))
    return violations


def _scan_tree(root: Path) -> list[_Context]:
    contexts: list[_Context] = []
    for source_root in (root / "src", root / "tests"):
        for path in sorted(source_root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            lowered = source.lower()
            if (not any(hint in lowered for hint in _CANDIDATE_HINTS)
                    and not _RUN_CANDIDATE.search(source)
                    and not _RUN_DETAIL_CANDIDATE.search(source)):
                continue
            contexts.extend(_contexts(path.relative_to(root), source))
    return contexts


def test_source_and_test_prose_has_no_private_names():
    root = Path(__file__).resolve().parents[1]
    assert _find_private_labels(_scan_tree(root)) == []


def test_guard_detects_a_planted_comment_label():
    planted = (
        "# " + "A" + "HA " + "M" + "C " + "crew-" + "08 " + "rule " + "#" + "7"
    )
    contexts = [_Context("planted.py", 1, "comment", planted)]
    found = _find_private_labels(contexts, enforce_allowlist=False)
    assert len(found) == 4
    assert sum("project_alias" in item for item in found) == 2
    assert sum("numbered_rule" in item for item in found) == 1
    assert sum("personal_host" in item for item in found) == 1


def test_guard_detects_full_project_name_in_prose():
    context = _Context("planted.py", 1, "comment", "# " + "Mission" + " Control")
    found = _find_private_labels([context], enforce_allowlist=False)
    assert len(found) == 1
    assert "full_project_name" in found[0]


def test_guard_does_not_confuse_permission_control_with_project_name():
    context = _Context("planted.py", 1, "comment", "# permission control")
    assert _find_private_labels([context], enforce_allowlist=False) == []
