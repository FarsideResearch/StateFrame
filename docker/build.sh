#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  printf 'usage: PYTHON_IMAGE=IMAGE@sha256:DIGEST %s CLEAN_GIT_REF\n' "$0" >&2
  exit 2
fi

python_image=${PYTHON_IMAGE:-}
base_name=${python_image##*/}
if [[ ! "$base_name" =~ ^python:3\.13-slim-bookworm@sha256:[0-9a-f]{64}$ ]]; then
  printf 'PYTHON_IMAGE is required and must be Python 3.13 slim Bookworm pinned by a sha256 digest\n' >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ref=$1
command -v git >/dev/null 2>&1 || { printf 'git is required\n' >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { printf 'python3 is required\n' >&2; exit 1; }
builder_spec=${BUILDER:-docker buildx}
read -r -a builder_cmd <<<"$builder_spec"
if [[ -z "${builder_cmd[0]:-}" ]]; then
  printf 'BUILDER must name a BuildKit-capable command, such as "docker buildx"\n' >&2
  exit 2
fi
if [[ "${builder_cmd[0]##*/}" == docker && "${builder_cmd[1]:-}" != buildx ]]; then
  printf 'BUILDER must invoke docker buildx, not the legacy docker build command\n' >&2
  exit 2
fi
command -v "${builder_cmd[0]}" >/dev/null 2>&1 || {
  printf 'build command not found: %s\n' "${builder_cmd[0]}" >&2
  exit 1
}
if ! "${builder_cmd[@]}" version >/dev/null 2>&1; then
  printf 'BuildKit is required; install Docker Buildx or set BUILDER to a compatible buildx command\n' >&2
  exit 1
fi
git -C "$root" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null || {
  printf 'not a resolvable git commit ref: %s\n' "$ref" >&2
  exit 2
}

build_limits=()
if [[ -n ${BUILD_CPUS:-} ]]; then
  if [[ ! "$BUILD_CPUS" =~ ^[1-9][0-9]{0,2}$ ]] || (( BUILD_CPUS > 256 )); then
    printf 'BUILD_CPUS must be a whole number from 1 through 256\n' >&2
    exit 2
  fi
  build_limits+=(--cpu-period 100000 --cpu-quota "$((BUILD_CPUS * 100000))")
fi
if [[ -n ${BUILD_MEMORY:-} ]]; then
  if [[ ! "$BUILD_MEMORY" =~ ^[1-9][0-9]*([kKmMgGtT]([iI]?[bB])?)?$ ]]; then
    printf 'BUILD_MEMORY must be a positive byte value, such as 2g or 512m\n' >&2
    exit 2
  fi
  build_limits+=(--memory "$BUILD_MEMORY")
fi

# mktemp honors a caller's TMPDIR and otherwise uses the system default.
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/sf-eks-build.XXXXXXXX")
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$build_dir/tmp"
export TMPDIR="$build_dir/tmp"
source_dir="$build_dir/source"
context="$build_dir/context"
mkdir -p "$source_dir" "$context/docker" "$context/dist"
git -C "$root" archive --format=tar "$ref" | tar -xf - -C "$source_dir"

version=$(python3 -c 'import pathlib, sys, tomllib; print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text())["project"]["version"])' \
  "$source_dir/pyproject.toml")
if [[ ! "$version" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
  printf 'project version is not a valid image tag: %s\n' "$version" >&2
  exit 2
fi
python3 -m pip wheel --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$context/dist" "$source_dir"
mapfile -t wheels < <(find "$context/dist" -maxdepth 1 -type f -name '*.whl' -print)
if (( ${#wheels[@]} != 1 )); then
  printf 'expected one wheel from the in-tree backend, found %s\n' "${#wheels[@]}" >&2
  exit 1
fi
wheel_sha=$(sha256sum "${wheels[0]}" | awk '{print $1}')
cp "$source_dir/.dockerignore" "$context/.dockerignore"
cp "$source_dir/docker/Dockerfile" "$context/docker/Dockerfile"
cp "$source_dir/docker/verify_wheel.py" "$context/docker/verify_wheel.py"
cp "$source_dir/requirements.txt" "$context/requirements.txt"

"${builder_cmd[@]}" build \
  "${build_limits[@]}" \
  --load \
  --build-arg "PYTHON_IMAGE=$python_image" \
  --build-arg "APP_VERSION=$version" \
  --build-arg "WHEEL_SHA256=$wheel_sha" \
  --tag "sf-eks-stateframe:$version" \
  --file "$context/docker/Dockerfile" \
  "$context"
printf 'built sf-eks-stateframe:%s from %s\nwheel sha256: %s\n' "$version" "$ref" "$wheel_sha"
