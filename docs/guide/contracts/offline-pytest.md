# Pinned pytest for online and offline checks

The harness pins pytest to 9.0.3 for its repository checks. This is the fixed release for [CVE-2025-71176](https://github.com/advisories/GHSA-6w46-j5rx-g56g); see the [pytest 9.0.3 release](https://github.com/pytest-dev/pytest/releases/tag/9.0.3).

## Install

The ordinary installer downloads exact dependency versions from PyPI and verifies the hashes in `requirements.txt`. Its default repo-map install also verifies the separate `requirements-repomap.txt` lock:

```sh
./install.sh
```

For an offline install, the installer verifies `wheelhouse/SHA256SUMS` and uses only the wheels shipped with the source tree:

```sh
./install.sh --offline
```

The source archive includes the same wheelhouse, manifest, and installer, so the offline command also works from an extracted source archive. The build backend uses only the Python standard library and does not fetch build requirements.

To install with a specific Python interpreter, set `PYTHON` for the installer. To run a check with a specific interpreter, use that interpreter directly or pass it as `--test-python` to `stateframe run`:

```sh
PYTHON=/path/to/python ./install.sh --offline
/path/to/python -m pytest --version
```

## Supported combinations

- The harness requires Python 3.11 or newer. The bundled dependency wheelhouse covers CPython 3.11–3.14 on Linux x86_64 or aarch64 and macOS x86_64 or arm64 where compatible wheels are supplied.
- The pytest wheel is `py3-none-any`; its artifact hash is `2c5efc453d45394fdd706ade797c0a81091eccd1d6e4bccfcd476e2b8e0ab5d9`.
- The installer itself supports Linux x86_64 and macOS arm64, matching the available command-isolation backends.
- The wheelhouse contains pytest and its declared dependencies. Optional pytest plugins such as `pytest-xdist` are not included.

Run pytest through the interpreter that owns the environment to avoid accidentally using a different installation:

```sh
python -m pytest -q tests/test_example.py
```

## Limits

The online requirements lock includes hashes for the PyPI wheel and source archive. The offline bundle contains the universal wheel and records its SHA-256 in `wheelhouse/SHA256SUMS`. The wheelhouse is intended for the supported interpreter and platform combinations above; it is not a general package mirror.
