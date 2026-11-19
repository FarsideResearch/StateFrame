"""Filename rules shared by work copies and mutation input admission."""
from __future__ import annotations

from fnmatch import fnmatchcase
from os.path import splitext


SECRET_DIRS = frozenset({".env", ".secrets"})
SECRET_GLOBS = (
    ".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    "*.p12", "*.pfx", ".netrc", ".npmrc", ".pypirc", "credentials", "credentials.json",
    ".git-credentials", "secrets.json", "secrets.yaml", "secrets.yml", "secrets.toml",
    "*.secret", "*.secrets",
)
_SOURCE_SUFFIXES = frozenset({
    ".py", ".pyi", ".pyw", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
    ".go", ".rs", ".java", ".kt", ".kts", ".scala", ".rb", ".php", ".cs", ".fs", ".fsx",
    ".c", ".h", ".cc", ".hh", ".cpp", ".hpp", ".cxx", ".hxx", ".m", ".mm", ".swift",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".pl", ".pm", ".lua",
    ".ex", ".exs", ".erl", ".hrl", ".hs", ".ml", ".mli", ".clj", ".cljs", ".cljc",
    ".dart", ".r", ".jl", ".sql", ".vue", ".svelte", ".html", ".css", ".scss",
    ".zig", ".nim", ".groovy", ".vb", ".asm", ".s", ".f", ".f90", ".pas", ".sol",
    ".tcl", ".nix", ".elm", ".lisp", ".el",
})


def is_secret_file(name: str) -> bool:
    """Match a basename against the secret-file rules."""
    name = name.casefold()
    # Prefix rules for dotenv variants and SSH keys can also match source names.
    # Keep those modules in the copy so dependent imports still resolve.
    if splitext(name)[1] in _SOURCE_SUFFIXES:
        return False
    return any(fnmatchcase(name, pattern) for pattern in SECRET_GLOBS)
