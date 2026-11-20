"""Guard repository intelligence source and documentation against old identifiers."""

from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[1]
OLD_MODULE = "_".join(("mc", "repo", "intel", "vendor"))
PRIVATE_PHRASES = (
    r"\s+".join(("project", "rules")),
    r"rule\s*\#\s*\d+",
    "-".join(("agent", "runner")),
    r"\s+".join(("agent", "runner")),
    r"\s+".join(("mission", "control")),
    "-".join(("mission", "control")),
    "".join(("a", "h", "a")),
    "".join(("far", "side")),
    r"\s+".join(("Code", "Studio")),
    "_".join(("blast", "radius")) + r"\.py",
    "_".join(("repo", "intel")) + r"[./]signatures",
)
PRIVATE_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(PRIVATE_PHRASES) + r")(?!\w)", re.IGNORECASE
)
OLD_MODULE_PATTERN = re.compile(re.escape(OLD_MODULE))


def _source_files(directory: str):
    for path in (ROOT / directory).rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            yield path


# ENGINE_SOURCE.json keeps a patches[] provenance history keyed by the file path each patch was
# recorded against. Those historical entries must stay byte-stable for release carries, so the
# guard checks the rest of the record but not that history list.
_PROVENANCE = Path("src/stateframe/engine/ENGINE_SOURCE.json")


def _text(path: Path) -> str:
    text = path.read_bytes().decode("utf-8", errors="replace")
    if path.relative_to(ROOT) == _PROVENANCE:
        record = json.loads(text)
        record.pop("patches", None)
        text = json.dumps(record, indent=1, sort_keys=True)
    return text


def _hits(path: Path, pattern: re.Pattern[str]) -> list[str]:
    text = _text(path)
    return [
        f"{path.relative_to(ROOT)}:{text.count(chr(10), 0, match.start()) + 1}: {match.group()}"
        for match in pattern.finditer(text)
    ]


def test_old_repository_intelligence_module_is_absent():
    paths = [ROOT / "pyproject.toml"]
    paths.extend(path for directory in ("src", "tests", "docs") for path in _source_files(directory))
    hits = [hit for path in paths for hit in _hits(path, OLD_MODULE_PATTERN)]
    hits.extend(
        str(path.relative_to(ROOT))
        for path in paths
        if OLD_MODULE_PATTERN.search(path.relative_to(ROOT).as_posix())
    )
    assert not hits, "Old repository intelligence module references:\n" + "\n".join(hits)


def test_private_product_and_rule_references_are_absent():
    hits = [
        hit
        for directory in ("src", "docs")
        for path in _source_files(directory)
        for hit in _hits(path, PRIVATE_PATTERN)
    ]
    assert not hits, "Private references:\n" + "\n".join(hits)
