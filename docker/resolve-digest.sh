#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  printf 'usage: %s IMAGE[:TAG]\n' "$0" >&2
  exit 2
fi

image=$1
if [[ "$image" =~ @sha256:[0-9a-f]{64}$ ]]; then
  printf '%s\n' "$image"
  exit 0
fi
if [[ "$image" == *@* ]]; then
  printf 'image digest must use sha256 followed by 64 lowercase hex digits\n' >&2
  exit 2
fi
command -v docker >/dev/null 2>&1 || {
  printf 'docker with buildx is required to resolve an image digest\n' >&2
  exit 1
}

inspection=$(docker buildx imagetools inspect "$image")
digest=$(awk '$1 == "Digest:" && $2 ~ /^sha256:[0-9a-f]{64}$/ { print $2; exit }' <<<"$inspection")
if [[ -z "$digest" ]]; then
  printf 'could not read a sha256 digest for %s\n' "$image" >&2
  exit 1
fi
printf '%s@%s\n' "$image" "$digest"
