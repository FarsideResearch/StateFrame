"""Guard repository intelligence source and documentation against old identifiers."""

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_WORD_DIGESTS = frozenset(
    (
        "0ca758e42f6979815a567dfb58e968579596915fe9f7c106f3300be64568bd4f",
        "74273b01236c1ddf9e57c145950d53c27fd03f301ae79402629c2bc94f7daa26",
    )
)
# Keep the removed package identifier as component digests, with its underscore boundaries.
_OLD_MODULE_COMPONENT_DIGESTS = (
    "6bc0f9c4ecb0acbebf1b9f6c816bdd66f4885f67c917d9e141cf893014fba936",
    "071ca2227754705837aa3ef9748ed59e9f8a015fd765c42f391a4cbc271c6d5e",
    "96eebba49dbbf422d245f02290f9d4ed0eb02da9daa6bbceefb162800ff42481",
    "630ba09448af522154f38ef7685ef1f44b0f3e9430f80829a03ce24f400f3754",
)
_OLD_MODULE_COMPONENT_WIDTHS = (2, 4, 5, 6)
# Each pair digest maps to the separators its original expression accepted.
_PAIR_RULES = {
    "7ee1c8c1e9004ef9dfdf5c7d98eccf0221569cb950d262efb440c29f01197b7c": frozenset(
        ("space",)
    ),
    "5aa4ec111e5e49f74fcd03038673221f1fe0da005a7717b0cac2e852d4d79a38": frozenset(
        ("space", "-")
    ),
    "2cb9209d05b964e7291fb1ff5ddefd40f1ac9becffda4ba7d2cb29f158754dc9": frozenset(
        ("space", "-")
    ),
    "fbeb2786c8cd80042757514048f0d40dcd68cee1635c031a89f40d868fb14910": frozenset(
        ("space",)
    ),
}
_SHAPE_A = (
    "425c878178093ce652a4e845cba910c575b6b4dd6ab5f7202a3c3056ccd007ff",
    "6e491c02882ac36e3bcaaac4d151d4b2793135d4482adf227b1b730cdd90b46d",
)
_SHAPE_B = (
    "8ee6c3d7878b9fdecee520b8311a8b797595b383ec87213cf394c4b75dc4808f",
    "0316e180931892ae5b478d9408ec7271cf86a01016a15701bd41db3394c6ef31",
)
_SHAPED_RULES = {
    ".": frozenset((_SHAPE_A, _SHAPE_B)),
    "/": frozenset((_SHAPE_B,)),
}
RULE_PATTERN = re.compile(r"(?<!\w)rule\s*\#\s*\d+(?!\w)", re.IGNORECASE)


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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identifier_sequence_matches(text: str, component_digests: tuple[str, ...],
                                 widths: tuple[int, ...] = _OLD_MODULE_COMPONENT_WIDTHS):
    """Yield every substring match, as the plain pattern did: attached letters or digits do not hide it."""
    shape = "_".join(f"[A-Za-z0-9]{{{width}}}" for width in widths)
    for match in re.finditer(f"(?=({shape}))", text):
        found = match.group(1).lower()
        if tuple(map(_digest, found.split("_"))) == component_digests:
            yield match.start(), _digest(found)


def _digest_matches(
    text: str,
    pair_rules: dict[str, frozenset[str]] = _PAIR_RULES,
    word_digests: frozenset[str] = _WORD_DIGESTS,
    shaped_rules: dict[str, frozenset[tuple[str, str]]] = _SHAPED_RULES,
):
    tokens = tuple(re.finditer(r"\w+", text))
    for index, token in enumerate(tokens):
        word = token.group().lower()
        word_digest = _digest(word)
        if word_digest in word_digests:
            yield token.start(), word_digest
        if index + 1 == len(tokens):
            continue

        next_token = tokens[index + 1]
        next_word = next_token.group().lower()
        separator = text[token.end() : next_token.start()]
        if separator == "-" or re.fullmatch(r"\s+", separator):
            pair_digest = _digest(f"{word} {next_word}")
            allowed = pair_rules.get(pair_digest, frozenset())
            if (separator == "-" and "-" in allowed) or (
                separator != "-" and "space" in allowed
            ):
                yield token.start(), pair_digest
        elif (word_digest, _digest(next_word)) in shaped_rules.get(separator, frozenset()):
            yield token.start(), _digest(text[token.start() : next_token.end()].lower())
    yield from _identifier_sequence_matches(text, _OLD_MODULE_COMPONENT_DIGESTS)


def _hit(path: Path, text: str, offset: int, digest: str) -> str:
    line = text.count("\n", 0, offset) + 1
    return f"{path.relative_to(ROOT)}:{line}: sha256:{digest[:12]}"


def _private_hits(path: Path) -> list[str]:
    text = _text(path)
    hits = [
        _hit(path, text, offset, digest)
        for offset, digest in _digest_matches(text)
    ]
    hits.extend(
        _hit(path, text, match.start(), _digest(match.group().lower()))
        for match in RULE_PATTERN.finditer(text)
    )
    return hits


def test_digest_matching_uses_word_boundaries():
    digest = _digest("alpha beta")
    rules = {digest: frozenset(("space", "-"))}
    for text in ("x alpha-beta y", "ALPHA beta", "alpha  beta", "alpha\n   beta"):
        assert [found for _, found in _digest_matches(text, rules)] == [digest]
    for text in (
        "alphabeta",
        "alpha_beta",
        "alpha'beta",
        "alpha,beta",
        "alpha.beta",
        "alpha--beta",
        "alpha - beta",
        "alpha gamma beta",
    ):
        assert not list(_digest_matches(text, rules))


def test_digest_matching_preserves_phrase_shapes():
    pair_digest = _digest("alpha beta")
    space_only = {pair_digest: frozenset(("space",))}
    assert list(_digest_matches("alpha beta", space_only)) == [(0, pair_digest)]
    assert not list(_digest_matches("alpha-beta", space_only))

    shaped = {".": frozenset(((_digest("alpha_beta"), _digest("py")),))}
    expected = _digest("alpha_beta.py")
    assert list(_digest_matches("alpha_beta.py", {}, frozenset(), shaped)) == [
        (0, expected)
    ]
    for text in ("alpha_beta/py", "alpha_beta..py", "alpha_beta,py"):
        assert not list(_digest_matches(text, {}, frozenset(), shaped))


def test_identifier_digest_matching_preserves_component_boundaries():
    components = tuple(_digest(part) for part in ("alpha", "beta"))
    widths = (5, 4)
    for text, start in (("prefix_alpha_beta_suffix", 7), ("(!alpha_beta123?!", 2), ("xAlpha_Betay", 1)):
        assert list(_identifier_sequence_matches(text, components, widths)) == [(start, _digest("alpha_beta"))]
    for text in ("prefix_alpha__beta_suffix", "prefix_alpha-beta_suffix", "alph_abeta"):
        assert not list(_identifier_sequence_matches(text, components, widths))


def test_old_repository_intelligence_module_is_absent():
    paths = [ROOT / "pyproject.toml"]
    paths.extend(path for directory in ("src", "tests", "docs") for path in _source_files(directory))
    hits = []
    for path in paths:
        text = _text(path)
        hits.extend(
            _hit(path, text, offset, digest)
            for offset, digest in _identifier_sequence_matches(text, _OLD_MODULE_COMPONENT_DIGESTS)
        )
        relative = path.relative_to(ROOT).as_posix()
        hits.extend(
            _hit(path, relative, offset, digest)
            for offset, digest in _identifier_sequence_matches(relative, _OLD_MODULE_COMPONENT_DIGESTS)
        )
    assert not hits, "Old repository intelligence module references:\n" + "\n".join(hits)


def test_private_product_and_rule_references_are_absent():
    hits = [
        hit
        for directory in ("src", "docs")
        for path in _source_files(directory)
        for hit in _private_hits(path)
    ]
    assert not hits, "Private references:\n" + "\n".join(hits)
