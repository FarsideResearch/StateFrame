#!/usr/bin/env python3
"""Check one release wheel and optionally stage its license files."""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path


REQUIRED_LICENSES = frozenset({
    "LICENSE", "NOTICE", "COMMERCIAL-LICENSE.md", "THIRD_PARTY_LICENSES.md",
})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument("--python-image", required=True)
    parser.add_argument("--app-version", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--licenses-dir", type=Path)
    args = parser.parse_args()

    base_name = args.python_image.rsplit("/", 1)[-1]
    if not re.fullmatch(r"python:3\.13-slim-bookworm@sha256:[0-9a-f]{64}", base_name):
        raise SystemExit("PYTHON_IMAGE must be Python 3.13 slim Bookworm pinned by sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        raise SystemExit("WHEEL_SHA256 must be 64 lowercase hex digits")
    wheels = list(args.wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, found {len(wheels)}")
    wheel = wheels[0]
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != args.sha256:
        raise SystemExit("wheel SHA-256 does not match WHEEL_SHA256")

    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist()
                          if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise SystemExit("wheel must contain exactly one dist-info/METADATA")
        fields = {}
        for line in archive.read(metadata_names[0]).decode("utf-8").splitlines():
            if line.startswith(("Name: ", "Version: ")):
                key, value = line.split(": ", 1)
                fields[key] = value
        if fields.get("Name", "").lower() != "stateframe" or fields.get("Version") != args.app_version:
            raise SystemExit("wheel name or version does not match the requested release")

        if args.licenses_dir is not None:
            selected = {}
            for name in archive.namelist():
                base = name.rsplit("/", 1)[-1]
                if base not in REQUIRED_LICENSES:
                    continue
                if not (".dist-info/licenses/" in name
                        or name == "stateframe/THIRD_PARTY_LICENSES.md"):
                    continue
                if base in selected:
                    raise SystemExit(f"wheel has duplicate license file {base}")
                selected[base] = archive.read(name)
            if set(selected) != REQUIRED_LICENSES:
                raise SystemExit("wheel is missing one or more required license files")
            args.licenses_dir.mkdir(parents=True, exist_ok=True)
            for name, content in selected.items():
                (args.licenses_dir / name).write_bytes(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
