"""Codec for the durable panel manifest and its encoding-specific fragments."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import NamedTuple

from kernel_turn_contracts import Artifact, Basis, Corrupt, Dep, Head, Key, Publication, Ref


MANIFEST = "turn/slot-manifest/1"
LAYOUT = "turn/layout/1"
FACT_PAGE = "turn/fact-page/1"
FRAGMENTS = {
    "json": "turn/panel-json/1",
    "stateframe-message": "turn/panel-stateframe-message/1",
}
MAX_SLOTS = 256
MAX_FACT_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
SlotName = tuple[str, str]


class Slot(NamedTuple):
    ref: Ref | None
    deps: tuple[Dep, ...]
    input_digest: str


class Manifest(NamedTuple):
    basis: Basis
    attempt: str
    last_event: str | None
    catalog_deps: tuple[Dep, ...]
    slots: dict[SlotName, Slot]


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_text(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def validate_ref(ref: Ref, codec: str, limit: int) -> None:
    if (ref.codec != codec or not digest_text(ref.sha256)
            or type(ref.size) is not int or not 0 <= ref.size <= limit):
        raise Corrupt(f"invalid {codec} reference")


def load_metadata(publication: Publication, ref: Ref, codec: str, limit: int) -> object:
    validate_ref(ref, codec, limit)
    data = publication.load(ref)
    try:
        value = json.loads(data)
        if canonical(value) != data:
            raise ValueError("metadata is not canonical")
        return value
    except (ValueError, RecursionError) as exc:
        raise Corrupt(f"invalid {codec} metadata") from exc


def slot_order(value: object, *, unique: bool) -> tuple[SlotName, ...]:
    if not isinstance(value, list) or len(value) > MAX_SLOTS:
        raise Corrupt("invalid panel slot list")
    slots = []
    for pair in value:
        if (not isinstance(pair, list) or len(pair) != 2
                or not all(isinstance(item, str) and item for item in pair)):
            raise Corrupt("invalid panel slot or variant")
        slots.append((pair[0], pair[1]))
    if unique and len(set(slots)) != len(slots):
        raise Corrupt("repeated slot in panel catalog")
    return tuple(slots)


def _encode_deps(deps: tuple[Dep, ...]) -> list[list[str]]:
    return [[dep.key.family, dep.key.selector.hex(), dep.revision] for dep in deps]


def _decode_deps(value: object) -> tuple[Dep, ...]:
    if not isinstance(value, list):
        raise ValueError("invalid panel dependency list")
    deps = []
    for row in value:
        if (not isinstance(row, list) or len(row) != 3
                or not all(isinstance(item, str) and item for item in row)):
            raise ValueError("invalid panel dependency")
        family, selector, revision = row
        key = Key(family, bytes.fromhex(selector))
        deps.append(Dep(key, revision))
    result = tuple(deps)
    if _encode_deps(result) != value or len({dep.key for dep in result}) != len(result):
        raise ValueError("noncanonical or repeated panel dependency")
    return result


def artifact_deps(manifest: Manifest) -> tuple[Dep, ...]:
    # Keep distinct revisions for the same key visible so disagreement between
    # two fact pages cannot be mistaken for a valid projection.
    return tuple(dict.fromkeys(dep for deps in (
        manifest.catalog_deps, *(slot.deps for slot in manifest.slots.values()),
    ) for dep in deps))


def write_manifest(publication: Publication, namespace: str, manifest: Manifest) -> Artifact:
    data = canonical({
        "scope": namespace,
        "producer": 1,
        "basis": manifest.basis,
        "attempt": manifest.attempt,
        "consumed": manifest.basis.journal,
        "last_event": manifest.last_event,
        "catalog_deps": _encode_deps(manifest.catalog_deps),
        "slots": [[name, slot.ref, _encode_deps(slot.deps)]
                  for name, slot in manifest.slots.items()],
        "inputs": [slot.input_digest for slot in manifest.slots.values()],
    })
    if len(data) > MAX_MANIFEST_BYTES:
        raise Corrupt("panel manifest exceeds the metadata limit")
    return Artifact(publication.stage(MANIFEST, data), artifact_deps(manifest))


def _read_head(value: object) -> Head:
    if (not isinstance(value, list) or len(value) != 2 or type(value[0]) is not int
            or value[0] < 0 or not isinstance(value[1], str) or not value[1]):
        raise ValueError("invalid panel journal head")
    return Head(*value)


def _read_basis(value: object) -> Basis:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError("invalid panel basis")
    parent, live, epoch, journal, step, authority = value
    if (not all(isinstance(item, str) for item in (parent, live, authority))
            or type(epoch) is not int or epoch < 0 or type(step) is not int or step < 0):
        raise ValueError("invalid panel basis fields")
    return Basis(parent, live, epoch, _read_head(journal), step, authority)


def _fragment_ref(value: object) -> Ref | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3 or value[0] not in FRAGMENTS.values():
        raise ValueError("unknown panel fragment codec")
    ref = Ref(*value)
    # Nested escaping can expand a bounded UTF-8 or ASCII fact page.
    validate_ref(ref, ref.codec, 8 * MAX_FACT_BYTES)
    return ref


def decode_manifest(publication: Publication, namespace: str, ref: Ref) -> Manifest:
    value = load_metadata(publication, ref, MANIFEST, MAX_MANIFEST_BYTES)
    try:
        if not isinstance(value, Mapping) or value["scope"] != namespace or value["producer"] != 1:
            raise ValueError("unknown producer or scope")
        basis = _read_basis(value["basis"])
        if _read_head(value["consumed"]) != basis.journal:
            raise ValueError("consumed head disagrees with the basis")
        attempt, last_event = value["attempt"], value["last_event"]
        if not isinstance(attempt, str) or (last_event is not None and not digest_text(last_event)):
            raise ValueError("invalid panel cursor metadata")
        rows, inputs = value["slots"], value["inputs"]
        if (not isinstance(rows, list) or not isinstance(inputs, list)
                or len(rows) != len(inputs) or len(rows) > MAX_SLOTS
                or any(not isinstance(row, list) or len(row) != 3 for row in rows)
                or not all(digest_text(item) for item in inputs)):
            raise ValueError("invalid panel manifest slots")
        order = slot_order([row[0] for row in rows], unique=True)
        slots = {name: Slot(_fragment_ref(row[1]), _decode_deps(row[2]), identity)
                 for name, row, identity in zip(order, rows, inputs, strict=True)}
        manifest = Manifest(basis, attempt, last_event, _decode_deps(value["catalog_deps"]), slots)
        deps = artifact_deps(manifest)
        if len({dep.key for dep in deps}) != len(deps):
            raise ValueError("persisted panel dependency revisions disagree")
        return manifest
    except (KeyError, TypeError, ValueError) as exc:
        raise Corrupt("invalid panel manifest") from exc



def read_manifest(publication: Publication, namespace: str, artifact: Artifact) -> Manifest:
    manifest = decode_manifest(publication, namespace, artifact.value)
    if artifact_deps(manifest) != artifact.deps:
        raise Corrupt("panel artifact dependencies disagree with the manifest")
    return manifest
