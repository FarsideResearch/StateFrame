"""Host-resolved executable fixtures shared by tests."""

import shutil
from pathlib import Path


_TRUE = shutil.which("true")
assert _TRUE is not None, "the host must provide a true executable"
# Resolved: a package-manager link (e.g. Homebrew) is not a regular file to the hook check.
TRUE = str(Path(_TRUE).resolve())

_FALSE = shutil.which("false")
assert _FALSE is not None, "the host must provide a false executable"
FALSE = str(Path(_FALSE).resolve())
