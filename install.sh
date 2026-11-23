#!/bin/bash
# StateFrame installer. Requires Python 3.11+ and a platform sandbox backend.
#   install.sh               dependencies from PyPI, pinned and hash-checked (requirements.txt)
#   install.sh --offline     dependencies from the bundled wheelhouse only (no network)
#   install.sh --no-repomap  skip the syntax-aware repo map (tree-sitter + one MIT grammar package per language)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -P)"
CALLER_DIR="$(pwd -P)"
SF_HOME="${SF_HOME:-$HOME/.stateframe}"
PYTHON_INPUT="${PYTHON:-python3}"

die() {
  printf 'stateframe install: %s\n' "$1" >&2
  exit 1
}

OFFLINE=0
# On by default: individual, license-checked grammar packages (no grammar bundle; see requirements-repomap.txt).
REPOMAP=1
for arg in "$@"; do
  case "$arg" in
    --offline) OFFLINE=1 ;;
    --repomap) REPOMAP=1 ;;
    --no-repomap) REPOMAP=0 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) die "unknown option: $arg (see --help)" ;;
  esac
done

if [[ "$SF_HOME" != /* ]]; then
  SF_HOME="$CALLER_DIR/$SF_HOME"
fi
while [[ "$SF_HOME" != "/" && "$SF_HOME" == */ ]]; do
  SF_HOME="${SF_HOME%/}"
done

case "$PYTHON_INPUT" in
  */*)
    if [[ "$PYTHON_INPUT" == /* ]]; then
      PYTHON_PATH="$PYTHON_INPUT"
    else
      PYTHON_PATH="$CALLER_DIR/$PYTHON_INPUT"
    fi
    ;;
  *)
    PYTHON_PATH="$(command -v "$PYTHON_INPUT" 2>/dev/null || true)"
    if [[ -n "$PYTHON_PATH" && "$PYTHON_PATH" != /* ]]; then
      PYTHON_PATH="$CALLER_DIR/$PYTHON_PATH"
    fi
    ;;
esac

if [[ -z "${PYTHON_PATH:-}" || ! -f "$PYTHON_PATH" || ! -x "$PYTHON_PATH" ]]; then
  die "Python executable not found or not executable: $PYTHON_INPUT (set PYTHON to an executable path)"
fi

if ! PYTHON_VERSION="$("$PYTHON_PATH" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)"; then
  die "could not query Python version from $PYTHON_PATH"
fi
if [[ ! "$PYTHON_VERSION" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
  die "could not parse Python version from $PYTHON_PATH: ${PYTHON_VERSION:-no output}"
fi
PYTHON_MAJOR="${BASH_REMATCH[1]}"
PYTHON_MINOR="${BASH_REMATCH[2]}"
if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 11) )); then
  die "Python 3.11 or newer is required; found Python $PYTHON_VERSION"
fi
PYTHON_FULL="$("$PYTHON_PATH" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || true)"
if [[ "$PYTHON_FULL" == "3.14.1" ]]; then
  die "Python 3.14.1 is not supported: networkx 3.6.1, a required dependency, excludes this release; use another 3.14.x or Python 3.11-3.13"
fi

if ! "$PYTHON_PATH" -c 'import ensurepip, venv' >/dev/null 2>&1 \
  || ! "$PYTHON_PATH" -m venv --help >/dev/null 2>&1; then
  die "Python $PYTHON_VERSION must include venv and ensurepip to create an isolated install"
fi

PLATFORM="$(uname -s)"
case "$PLATFORM" in
  Darwin)
    command -v sandbox-exec >/dev/null 2>&1 \
      || die "sandbox-exec is required on macOS for StateFrame's command isolation"
    ;;
  Linux)
    command -v bwrap >/dev/null 2>&1 \
      || die "bwrap is required on Linux for StateFrame's command isolation (install it: sudo apt install bubblewrap)"
    ;;
  *)
    die "unsupported platform for StateFrame's command isolation: $PLATFORM"
    ;;
esac
if ! { [[ "$PLATFORM" == "Darwin" && "$(uname -m)" == "arm64" ]] || [[ "$PLATFORM" == "Linux" && "$(uname -m)" == "x86_64" ]]; }; then
  printf 'warning: this preview is tested on macOS on Apple silicon and Linux x86_64; %s %s is untested\n' \
    "$PLATFORM" "$(uname -m)" >&2
fi

WHEELHOUSE="$HERE/wheelhouse"
WHEELHOUSE_BUILDER="$HERE/scripts/build_wheelhouse.py"
REQUIREMENTS="$HERE/requirements.txt"
REPOMAP_REQUIREMENTS="$HERE/requirements-repomap.txt"
if (( OFFLINE )); then
  if [[ ! -f "$WHEELHOUSE_BUILDER" ]]; then
    die "offline wheelhouse verifier is missing: $WHEELHOUSE_BUILDER"
  fi
  WHEELHOUSE_ARGS=(--print-requirements --output "$WHEELHOUSE")
  if (( ! REPOMAP )); then
    WHEELHOUSE_ARGS+=(--no-default-extras)
  fi
  if ! OFFLINE_REQUIREMENTS="$("$PYTHON_PATH" "$WHEELHOUSE_BUILDER" "${WHEELHOUSE_ARGS[@]}")"; then
    die "offline wheelhouse is missing or incomplete; build it with scripts/build_wheelhouse.py"
  fi
else
  [[ -f "$REQUIREMENTS" ]] || die "pinned requirements are missing: $REQUIREMENTS"
  if (( REPOMAP )); then
    [[ -f "$REPOMAP_REQUIREMENTS" ]] || die "pinned repo map requirements are missing: $REPOMAP_REQUIREMENTS"
  fi
fi

if [[ "$SF_HOME" == "/" ]]; then
  die "SF_HOME must name a private install directory below the filesystem root"
fi
if [[ -L "$SF_HOME" ]]; then
  die "SF_HOME must be a real directory, not a symlink: $SF_HOME"
fi
if [[ -e "$SF_HOME" && ! -d "$SF_HOME" ]]; then
  die "SF_HOME must be a directory: $SF_HOME"
fi
if ! mkdir -p -m 700 "$SF_HOME"; then
  die "could not create the private install directory: $SF_HOME"
fi
if ! "$PYTHON_PATH" - "$SF_HOME" <<'PY'
import os
import stat
import sys

directory = sys.argv[1]
flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = -1
try:
    descriptor = os.open(directory, flags)
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise PermissionError("install directory is not owned by this user")
    os.fchmod(descriptor, 0o700)
except Exception as error:
    print(f"install directory must be a real directory owned by this user: {directory}: {error}",
          file=sys.stderr)
    sys.exit(1)
finally:
    if descriptor >= 0:
        os.close(descriptor)
PY
then
  die "install directory must be a real directory owned by this user: $SF_HOME"
fi
if ! SF_HOME="$(cd "$SF_HOME" && pwd -P)"; then
  die "could not resolve the private install directory: $SF_HOME"
fi

STABLE_VENV="$SF_HOME/venv"
if [[ -e "$STABLE_VENV" || -L "$STABLE_VENV" ]]; then
  if [[ -d "$STABLE_VENV" && ! -L "$STABLE_VENV" \
    && -f "$STABLE_VENV/pyvenv.cfg" \
    && ! -e "$STABLE_VENV/bin/stateframe" && ! -L "$STABLE_VENV/bin/stateframe" ]]; then
    if ! INCOMPLETE_DIR="$(mktemp -d "$SF_HOME/.venv-incomplete.XXXXXXXX")"; then
      die "could not reserve a preservation path for incomplete install target $STABLE_VENV; original contents were left in place"
    fi
    PRESERVED_VENV="$INCOMPLETE_DIR/venv"
    if ! "$PYTHON_PATH" - "$STABLE_VENV" "$PRESERVED_VENV" <<'PY'
import os
import sys

try:
    os.rename(sys.argv[1], sys.argv[2])
except OSError as error:
    print(f"could not rename incomplete environment: {error}", file=sys.stderr)
    sys.exit(1)
PY
    then
      die "could not preserve incomplete install target $STABLE_VENV at $PRESERVED_VENV; original contents were left in place"
    fi
    printf 'preserved incomplete environment at %s\n' "$PRESERVED_VENV"
  else
    die "install target already exists: $STABLE_VENV (choose a distinct SF_HOME to preserve the existing environment and incomplete material)"
  fi
fi

if ! VENV="$(mktemp -d "$SF_HOME/.venv-attempt.XXXXXXXX")"; then
  die "could not create an isolated install attempt under $SF_HOME; existing contents were preserved"
fi
if ! "$PYTHON_PATH" -m venv "$VENV"; then
  die "could not create isolated virtual environment at $VENV; attempt contents were retained"
fi
VENV_PYTHON="$VENV/bin/python"
if [[ ! -x "$VENV_PYTHON" ]] || ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
  die "the new virtual environment has no usable pip: $VENV; attempt contents were retained"
fi

if (( OFFLINE )); then
  PINNED_REQUIREMENTS=()
  while IFS= read -r requirement || [[ -n "$requirement" ]]; do
    [[ -z "$requirement" || "$requirement" == \#* ]] && continue
    PINNED_REQUIREMENTS+=("$requirement")
  done <<< "$OFFLINE_REQUIREMENTS"
  if (( ${#PINNED_REQUIREMENTS[@]} == 0 )); then
    die "offline wheelhouse has no pinned distributions; attempt contents were retained at $VENV"
  fi
  if ! "$VENV_PYTHON" -m pip install --disable-pip-version-check --no-index \
    --find-links "$WHEELHOUSE" "${PINNED_REQUIREMENTS[@]}"; then
    die "could not install pinned wheels from $WHEELHOUSE; attempt contents were retained at $VENV"
  fi
else
  # Downloads only; every file is checked against the hash pinned in the requirements file.
  REQUIREMENT_ARGS=(-r "$REQUIREMENTS")
  if (( REPOMAP )); then
    REQUIREMENT_ARGS+=(-r "$REPOMAP_REQUIREMENTS")
  fi
  if ! "$VENV_PYTHON" -m pip install --disable-pip-version-check --require-hashes --only-binary :all: \
    "${REQUIREMENT_ARGS[@]}"; then
    die "could not install pinned dependencies from PyPI (offline machine? use --offline); attempt contents were retained at $VENV"
  fi
fi
if ! "$VENV_PYTHON" -m pip install --disable-pip-version-check --no-index \
  --no-build-isolation --no-deps "$HERE"; then
  die "could not install StateFrame from $HERE; attempt contents were retained at $VENV"
fi

LICENSE_DOC="$HERE/src/stateframe/THIRD_PARTY_LICENSES.md"
if [[ ! -f "$LICENSE_DOC" ]]; then
  die "third-party license inventory is missing from the source package: $LICENSE_DOC; attempt contents were retained at $VENV"
fi
if ! PACKAGE_DIR="$("$VENV_PYTHON" -c 'import pathlib, stateframe; print(pathlib.Path(stateframe.__file__).resolve().parent)')"; then
  die "could not locate the installed StateFrame package; attempt contents were retained at $VENV"
fi
VENV_REAL="$(cd "$VENV" && pwd -P)"
case "$PACKAGE_DIR/" in
  "$VENV_REAL/"*) ;;
  *) die "StateFrame package was installed outside its isolated venv: $PACKAGE_DIR; attempt contents were retained at $VENV" ;;
esac
if [[ ! -d "$PACKAGE_DIR" ]] || ! cp "$LICENSE_DOC" "$PACKAGE_DIR/THIRD_PARTY_LICENSES.md"; then
  die "could not install the third-party license inventory into $PACKAGE_DIR; attempt contents were retained at $VENV"
fi

if ! "$VENV_PYTHON" - <<'PY'
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import sys

features = (
    ("schema validation", "jsonschema", "jsonschema"),
    ("YAML parsing", "yaml", "PyYAML"),
)
for label, module_name, distribution_name in features:
    try:
        installed_version = version(distribution_name)
    except PackageNotFoundError:
        print(f"{label}: OFF ({distribution_name} not installed)")
        continue
    try:
        __import__(module_name)
    except Exception as error:
        print(f"{label}: OFF ({distribution_name} installed but import failed: {type(error).__name__})")
        sys.exit(1)
    print(f"{label}: on ({distribution_name} {installed_version})")

repo_intel = []
for module_name, distribution_name, allow_non_import_error in (
    ("jedi", "jedi", True),
    ("networkx", "networkx", False),
):
    try:
        installed_version = version(distribution_name)
    except PackageNotFoundError:
        repo_intel.append(f"{module_name} OFF (not installed)")
        continue
    try:
        import_module(module_name)
    except ImportError as error:
        repo_intel.append(f"{module_name} OFF (import failed: {type(error).__name__})")
    except Exception as error:
        if not allow_non_import_error:
            print(f"repo intel: {module_name} import failed: {type(error).__name__}", file=sys.stderr)
            sys.exit(1)
        repo_intel.append(f"{module_name} OFF (import failed: {type(error).__name__})")
    else:
        repo_intel.append(f"{module_name} on ({installed_version})")
print("repo intel: " + ", ".join(repo_intel))

# Only the audited individual grammar packages count; grammar bundles are never used.
from stateframe import ts_grammars
working = []
if ts_grammars.install_as_language_pack():
    for name in ts_grammars.available_languages():
        try:
            ts_grammars.get_parser(name).parse(b"x")
            working.append(name)
        except Exception:
            pass
print(f"repo map: on ({len(working)} languages: {', '.join(working)})" if working
      else "repo map: OFF (tree-sitter not installed; text-scan fallback)")
PY
then
  die "an installed optional engine dependency could not be imported; attempt contents were retained at $VENV"
fi

ATTEMPT_CLI="$VENV/bin/stateframe"
if [[ ! -x "$ATTEMPT_CLI" ]]; then
  die "StateFrame CLI was not created at $ATTEMPT_CLI; attempt contents were retained at $VENV"
fi
if ! "$ATTEMPT_CLI" --version; then
  die "the installed CLI failed its version check at $ATTEMPT_CLI; attempt contents were retained at $VENV"
fi

# os.symlink is an exclusive atomic create: concurrent installers cannot replace one another's result.
if "$PYTHON_PATH" - "$VENV" "$STABLE_VENV" <<'PY'
import os
import sys

try:
    os.symlink(sys.argv[1], sys.argv[2])
except FileExistsError:
    raise SystemExit(3)
except OSError as error:
    print(f"could not publish the verified environment at {sys.argv[2]}: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
then
  :
else
  publish_status=$?
  if (( publish_status == 3 )); then
    die "install target already exists: $STABLE_VENV (choose a distinct SF_HOME to preserve it); attempt contents were retained at $VENV"
  fi
  die "could not publish the verified environment at $STABLE_VENV; attempt contents were retained at $VENV"
fi

CLI="$STABLE_VENV/bin/stateframe"

LOCAL_BIN="$HOME/.local/bin"
LINK="$LOCAL_BIN/stateframe"
if ! mkdir -p "$LOCAL_BIN"; then
  printf 'note: could not create optional command directory: %s\n' "$LOCAL_BIN" >&2
elif [[ -L "$LINK" && "$(readlink "$LINK")" == "$CLI" ]]; then
  :
elif [[ -e "$LINK" || -L "$LINK" ]]; then
  printf 'note: leaving existing command path unchanged: %s\n' "$LINK" >&2
else
  ln -s "$CLI" "$LINK" || printf 'note: could not create optional command link: %s\n' "$LINK" >&2
fi

printf 'installed: %s\n' "$CLI"
printf 'Run directly: %q --version\n' "$CLI"
printf 'PATH hint: export PATH=%q:"$PATH"\n' "$LOCAL_BIN"
