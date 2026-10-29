"""Tree-sitter grammars from individual, hash-pinned PyPI packages (no grammar bundle, no downloads).

The engine's repo intel asks ``tree_sitter_language_pack`` for parsers. That pack bundles ~160 grammars with
mostly unaudited licenses and its 1.x line downloads grammars at runtime, so the product installs one MIT
package per language instead and exposes the small API the engine uses (``get_parser``, ``get_language``,
``available_languages``) under that module name. Grammars are only ever imported from installed local
packages; a language with no installed package raises ``LookupError`` and the engine falls back to its
text-scan repo map.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types

# Engine language name -> (installed module, function returning the language pointer).
GRAMMARS: dict[str, tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "csharp": ("tree_sitter_c_sharp", "language"),
    "html": ("tree_sitter_html", "language"),
    "css": ("tree_sitter_css", "language"),
    "json": ("tree_sitter_json", "language"),
    "toml": ("tree_sitter_toml", "language"),
    "yaml": ("tree_sitter_yaml", "language"),
}
ALIASES = {"c_sharp": "csharp"}
MODULE_NAME = "tree_sitter_language_pack"
LEGACY_MODULE = "tree_sitter_languages"


def get_language(name: str):
    import tree_sitter

    key = str(name).casefold()
    try:
        module_name, function = GRAMMARS[ALIASES.get(key, key)]
    except KeyError:
        raise LookupError(f"no tree-sitter grammar package for language {name!r}") from None
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise LookupError(f"grammar package {module_name} is not installed") from None
    return tree_sitter.Language(getattr(module, function)())


def get_parser(name: str):
    import tree_sitter

    return tree_sitter.Parser(get_language(name))


def available_languages() -> list[str]:
    names = []
    for name, (module_name, _) in sorted(GRAMMARS.items()):
        if importlib.util.find_spec(module_name) is not None:
            names.append(name)
    return names


def install_as_language_pack() -> bool:
    """Expose this module as ``tree_sitter_language_pack``. Returns True if set.

    It is authoritative: it wins even over an installed or already-loaded language pack (the 1.x pack fetches
    grammars from the network on a cache miss, and bundles grammars whose licenses are not individually
    audited), and it makes the legacy ``tree_sitter_languages`` bundle unimportable, so the engine's fallback
    to it can never load. Only the audited individual grammar packages can be used.
    """
    if getattr(sys.modules.get(MODULE_NAME), "__stateframe_grammar_adapter__", False):
        return False  # already installed
    try:
        importlib.import_module("tree_sitter")
    except ImportError:
        # No audited grammars available: block both bundles too, so the engine uses its text-scan map.
        sys.modules[MODULE_NAME] = None
        sys.modules[LEGACY_MODULE] = None
        return False
    shim = types.ModuleType(MODULE_NAME)
    shim.__doc__ = __doc__
    shim.get_parser, shim.get_language, shim.available_languages = get_parser, get_language, available_languages
    shim.__stateframe_grammar_adapter__ = True
    sys.modules[MODULE_NAME] = shim
    sys.modules[LEGACY_MODULE] = None  # `import tree_sitter_languages` now raises ImportError
    return True
