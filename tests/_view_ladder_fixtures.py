"""Deterministic fixtures for the view ladder tests (built in memory, no files)."""
from __future__ import annotations

import hashlib
import random

CALC_PY = (
    '"""Calc module."""\n'
    "import json\n"
    "from typing import Sequence\n"
    "\n"
    "DEFAULT_SCALE = 2\n"
    "\n"
    "\n"
    "class Calc:\n"
    '    """Accumulator."""\n'
    "\n"
    "    def __init__(self, scale: int = DEFAULT_SCALE) -> None:\n"
    "        self.total = 0\n"
    "        self.scale = scale\n"
    "\n"
    "    def add(self, value: int) -> int:\n"
    "        self.total += value * self.scale\n"
    "        return self.total\n"
    "\n"
    "    def divide(self, value: int) -> float:\n"
    "        return self.total / value\n"
    "\n"
    "\n"
    "def mean(values: Sequence[int]) -> float:\n"
    "    return sum(values) / len(values)\n"
    "\n"
    "\n"
    "def dump(calc: Calc) -> str:\n"
    '    return json.dumps({"total": calc.total})\n'
    "\n"
)
assert len(CALC_PY.splitlines()) == 29

NESTED_PY = (
    "import os\n"
    "from dataclasses import (\n"
    "    dataclass,\n"
    "    field,\n"
    ")\n"
    "\n"
    "LIMIT = 3\n"
    "\n"
    "\n"
    "@dataclass\n"
    "class Outer:\n"
    "    value: int = 0\n"
    "\n"
    "    class Inner:\n"
    "        def run(\n"
    "            self,\n"
    "            count: int,\n"
    "        ) -> int:\n"
    "            import sys\n"
    "            if count > LIMIT:\n"
    "                raise ValueError(count)\n"
    "            return os.getpid() + count\n"
    "\n"
    "    @staticmethod\n"
    "    def make() -> 'Outer':\n"
    "        def helper():\n"
    "            return 1\n"
    "        return Outer(helper())\n"
    "\n"
    "\n"
    "async def main() -> None:\n"
    "    print(Outer.make())\n"
)

JS = (
    "import fs from 'fs';\n"
    "const path = require('path');\n"
    "\n"
    "export function load(name) {\n"
    "  return fs.readFileSync(path.join('x', name));\n"
    "}\n"
    "\n"
    "const handler = async (req) => {\n"
    "  return load(req.url);\n"
    "};\n"
    "\n"
    "export default class Server {\n"
    "  start(port) {\n"
    "    console.log(port);\n"
    "  }\n"
    "}\n"
)
TS = (
    "import { X } from './x';\n"
    "export interface Config {\n"
    "  port: number;\n"
    "}\n"
    "export type Handler = (c: Config) => void;\n"
    "export const DEFAULT: Config = { port: 1 };\n"
    "export async function serve(c: Config): Promise<void> {\n"
    "  return;\n"
    "}\n"
)
GO = (
    "package main\n\n"
    "import \"fmt\"\n\n"
    "type Server struct {\n"
    "\tPort int\n"
    "}\n\n"
    "func (s *Server) Start() error {\n"
    "\tfmt.Println(s.Port)\n"
    "\treturn nil\n"
    "}\n\n"
    "func main() {\n"
    "\t(&Server{Port: 1}).Start()\n"
    "}\n"
)
RUST = (
    "use std::fmt;\n\n"
    "pub struct Point {\n"
    "    x: i32,\n"
    "}\n\n"
    "impl fmt::Display for Point {\n"
    "    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {\n"
    "        write!(f, \"{}\", self.x)\n"
    "    }\n"
    "}\n\n"
    "pub(crate) fn origin() -> Point {\n"
    "    Point { x: 0 }\n"
    "}\n"
)
JSON_TEXT = (
    '{\n  "name": "demo",\n  "version": 3,\n  "tags": ["a", "b"],\n'
    '  "nested": {"deep": {"x": 1}, "flag": true, "none": null},\n'
    '  "long": "' + "z" * 40 + '"\n}\n'
)
TOML_TEXT = (
    "[project]\nname = \"demo\"\nversion = \"0.1\"\n\n[tool.pytest]\naddopts = \"-q\"\n\n[dependencies]\nx = 1\ny = 2\n"
)
YAML_TEXT = (
    "name: demo\nservices:\n  web:\n    image: x\n  db:\n    image: y\nports:\n  - 80\n  - 443\n# comment\nempty:\n"
)
MD_TEXT = "# Title\n\nIntro text.\n\n## Section A\n\nBody.\n\n### Sub\n\nMore.\n\nSetext\n======\n\n## Section B\n"


def log_text(lines: int = 3000) -> str:
    rng = random.Random(7)
    out = []
    for i in range(1, lines + 1):
        if i % 97 == 0:
            out.append(f"2026-01-01 00:00:{i % 60:02d} ERROR worker failed: code {rng.randint(1, 3)}\n")
        elif i % 500 == 0:
            out.append("Traceback (most recent call last):\n")
        else:
            out.append(f"2026-01-01 00:00:{i % 60:02d} INFO tick {i}\n")
    return "".join(out)


def revision_of(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def decode_like_delivery(raw: bytes) -> tuple[str, bool]:
    """Mirror `deliver_source`: strict utf-8, then errors=replace."""
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), True


FIXTURES: dict[str, tuple[str, bytes]] = {
    "calc.py": ("pkg/calc.py", CALC_PY.encode()),
    "nested.py": ("pkg/nested.py", NESTED_PY.encode()),
    "indented_slice.py": ("pkg/indented.py", "    def inner(self):\n        return 1\n\n    def other(self):\n        return 2\n".encode()),
    "syntax_error.py": ("pkg/broken.py", "def broken(:\n    pass\nclass Ok:\n    pass\n".encode()),
    "app.js": ("web/app.js", JS.encode()),
    "types.ts": ("web/types.ts", TS.encode()),
    "main.go": ("cmd/main.go", GO.encode()),
    "lib.rs": ("src/lib.rs", RUST.encode()),
    "package.json": ("package.json", JSON_TEXT.encode()),
    "pyproject.toml": ("pyproject.toml", TOML_TEXT.encode()),
    "compose.yaml": ("compose.yaml", YAML_TEXT.encode()),
    "README.md": ("README.md", MD_TEXT.encode()),
    "worker.log": ("logs/worker.log", log_text().encode()),
    "one_line.py": ("one.py", b"x = 1\n"),
    "one_line_no_newline.txt": ("one.txt", b"only line"),
    "empty.py": ("empty.py", b""),
    "binary.bin": ("blob.bin", bytes(range(256)) * 4),
    "crlf.py": ("pkg/crlf.py", CALC_PY.replace("\n", "\r\n").encode()),
    "mixed_endings.py": ("pkg/mixed.py", b"import os\r\n\r\ndef a():\n    return 1\r\n\ndef b():\n    return 2\n"),
    "bom.py": ("pkg/bom.py", b"\xef\xbb\xbf" + CALC_PY.encode()),
    "invalid_utf8.py": ("pkg/bad.py", b"def f():\n    return '\xff\xfe\xc3'\n\ndef g():\n    return 2\n"),
    "no_trailing_newline.py": ("pkg/tail.py", b"def f():\n    return 1\n\ndef g():\n    return 2"),
    "form_feed.py": ("pkg/ff.py", b"def f():\n    return 1\n\x0c\ndef g():\n    return 2\n"),
    "unknown.ext": ("data/thing.xyz", b"alpha\nbeta\ngamma\n"),
    "shebang": ("bin/tool", b"#!/usr/bin/env python\nimport sys\n\ndef main():\n    return sys.argv\n"),
}
assert len(FIXTURES) == 25


def artifact_for(name: str, start: int | None = None, end: int | None = None):
    """Build an Artifact exactly as a delivered page would describe it."""
    from kernel_view_ladder import Artifact

    display, raw = FIXTURES[name]
    text, _ = decode_like_delivery(raw)
    lines = text.splitlines(keepends=True)
    total = len(lines)
    start = 1 if start is None else start
    end = total if end is None else end
    held = "".join(lines[start - 1:end])
    return Artifact(display=display, revision=revision_of(raw), held_start=start, held_end=end,
                    total_lines=total, held_text=held), text
