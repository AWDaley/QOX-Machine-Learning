"""Versioned, untrusted-input-safe QOX world saves and exact continuation.

A .qoxworld file stores world state, *not* neural weights. Model SHA-256
fingerprints bind saved learned worlds to the exact policies used at save time.
No Python pickle is used; file members and sizes are explicitly validated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Mapping
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from .config import SimulationConfig
from .entities import Actor, CallSignal, Carcass, Event, Settlement, Species, Tile
from .festivals import Camp, Festival, FestivalKind, FestivalResult
from .grid import Grid
from .world import World

SAVE_FORMAT = "qox.world"
SAVE_VERSION = 1
WORLD_SUFFIX = ".qoxworld"
WORLD_MEMBER = "world.json"
MANIFEST_MEMBER = "manifest.json"
MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_GRID_CELLS = 250_000
MAX_ACTORS = 50_000


@dataclass(frozen=True, slots=True)
class WorldSaveInfo:
    """Verified metadata; does not contain executable network weights."""

    tick: int
    seed: int
    mode: str
    policy_hashes: dict[str, str]
    options: dict[str, bool]
    state_hash: str
    path: Path


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_fingerprints(files: Mapping[str, str | Path] | None) -> dict[str, str]:
    """Hash the complete policy bytes, not just an editable checkpoint filename."""
    return {str(species): _fingerprint(path) for species, path in sorted((files or {}).items())}


def _as_tuples(value):
    if isinstance(value, list):
        return tuple(_as_tuples(element) for element in value)
    return value


def _as_serializable_world(world: World) -> dict:
    """Serialize complete authoritative state, including fields omitted by trace hashes."""
    return {
        "config": world.config.as_dict(),
        "tick": world.tick,
        "rng_state": world.rng.getstate(),
        "policy_rng_state": world.policy_rng.getstate(),
        "next_id": world.next_id,
        "next_carcass_id": world.next_carcass_id,
        "next_festival_id": world.next_festival_id,
        "birth_totals": {species.value: world.birth_totals[species] for species in Species},
        "tiles": [asdict(tile) for tile in world.grid.tiles],
        "actors": [asdict(actor) for _, actor in sorted(world.actors.items())],
        "settlements": [asdict(s) for s in world.settlements],
        "camps": [asdict(c) for c in world.camps],
        "carcasses": [asdict(c) for _, c in sorted(world.carcasses.items())],
        "calls": [asdict(call) for call in world.calls],
        "herd_goal": world.herd_goal,
        "active_festivals": [asdict(f) for _, f in sorted(world.active_festivals.items())],
        "festival_results": [asdict(f) for f in world.festival_results],
        "events": [asdict(e) for e in world.events],
        "last_events": [asdict(e) for e in world.last_events],
    }


def _validate_options(options: object) -> dict[str, bool]:
    if not isinstance(options, dict) or set(options) != {"sample_actions", "festival_assist"}:
        raise ValueError("Invalid saved controller options")
    if any(type(v) is not bool for v in options.values()):
        raise ValueError("Saved controller options must be boolean")
    return dict(options)


def _validate_manifest(manifest: object) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("Invalid QOX world manifest")
    if manifest.get("format") != SAVE_FORMAT or manifest.get("version") != SAVE_VERSION:
        raise ValueError("Unsupported QOX world format or version")
    if manifest.get("mode") not in ("scripted", "random", "idle", "multi-learned"):
        raise ValueError("Unknown saved world policy mode")
    hashes = manifest.get("policy_hashes")
    if not isinstance(hashes, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
            for key, value in hashes.items()):
        raise ValueError("Invalid policy fingerprint metadata")
    if manifest["mode"] == "multi-learned" and set(hashes) != {s.value for s in Species}:
        raise ValueError("Learned world requires all five policy fingerprints")
    if manifest["mode"] != "multi-learned" and hashes:
        raise ValueError("Scripted/random world must not claim neural policy fingerprints")
    _validate_options(manifest.get("options"))
    for key in ("world_sha256", "state_hash"):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
                ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"Invalid {key}")
    if type(manifest.get("tick")) is not int or manifest["tick"] < 0:
        raise ValueError("Invalid world tick")
    if type(manifest.get("seed")) is not int:
        raise ValueError("Invalid world seed")
    return manifest


def _parse_json(blob: bytes):
    def no_nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(blob.decode("utf-8"), parse_constant=no_nonfinite)


def _make_world(data: dict) -> World:
    if not isinstance(data, dict):
        raise ValueError("World payload must be a JSON object")
    config = SimulationConfig(**data["config"])
    if config.width * config.height > MAX_GRID_CELLS:
        raise ValueError("World dimensions exceed safe load limit")
    if len(data["actors"]) > MAX_ACTORS:
        raise ValueError("Actor count exceeds safe load limit")
    # Avoid World.__init__, which would generate new terrain and consume RNG.
    world = World.__new__(World)
    world.config = config
    world.rng = random.Random()
    world.policy_rng = random.Random()
    world.rng.setstate(_as_tuples(data["rng_state"]))
    world.policy_rng.setstate(_as_tuples(data["policy_rng_state"]))
    world.tick = data["tick"]
    world.events = [Event(**item) for item in data["events"]]
    world.last_events = [Event(**item) for item in data["last_events"]]
    world.next_id = data["next_id"]
    world.next_carcass_id = data["next_carcass_id"]
    world.next_festival_id = data["next_festival_id"]
    world.birth_totals = {species: int(data["birth_totals"][species.value])
                          for species in Species}
    world.grid = Grid(config.width, config.height,
                      [Tile(**item) for item in data["tiles"]])
    actors = [Actor(**{**item, "species": Species(item["species"])})
              for item in data["actors"]]
    if len({actor.id for actor in actors}) != len(actors):
        raise ValueError("Duplicate actor IDs")
    world.actors = {actor.id: actor for actor in actors}
    world.settlements = [Settlement(**item) for item in data["settlements"]]
    world.camps = [Camp(**item) for item in data["camps"]]
    carcasses = [Carcass(**{**item, "origin_species": Species(item["origin_species"])})
                 for item in data["carcasses"]]
    if len({c.id for c in carcasses}) != len(carcasses):
        raise ValueError("Duplicate carcass IDs")
    world.carcasses = {carcass.id: carcass for carcass in carcasses}
    world.calls = [CallSignal(**item) for item in data["calls"]]
    goal = data["herd_goal"]
    if goal is not None and (not isinstance(goal, list) or len(goal) != 2):
        raise ValueError("Invalid herd goal")
    world.herd_goal = tuple(goal) if goal is not None else None
    festivals = [Festival(**{**item, "kind": FestivalKind(item["kind"])})
                 for item in data["active_festivals"]]
    if len({f.id for f in festivals}) != len(festivals):
        raise ValueError("Duplicate festival IDs")
    world.active_festivals = {f.id: f for f in festivals}
    world.festival_results = [FestivalResult(**{**item,
                                               "kind": FestivalKind(item["kind"]),
                                               "participants": tuple(item["participants"])})
                              for item in data["festival_results"]]
    if (world.next_id <= max(world.actors, default=0)
            or world.next_carcass_id <= max(world.carcasses, default=0)
            or world.next_festival_id <= max([*world.active_festivals,
                                             *(f.id for f in world.festival_results)], default=0)):
        raise ValueError("Saved entity identity counters are invalid")
    try:
        world.check_invariants()
    except (AssertionError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("Saved world fails simulation invariants") from exc
    return world


def save_world(world: World, destination: str | Path, *, mode: str = "scripted",
               policy_files: Mapping[str, str | Path] | None = None,
               options: Mapping[str, bool] | None = None) -> Path:
    """Atomically save exact world/PRNG state; no neural-network weights included."""
    destination = Path(destination)
    if destination.suffix.lower() != WORLD_SUFFIX:
        destination = destination.with_suffix(WORLD_SUFFIX)
    if mode not in ("scripted", "random", "idle", "multi-learned"):
        raise ValueError("Unsupported mode for resumable QOX world")
    files = dict(policy_files or {})
    if mode == "multi-learned" and set(files) != {s.value for s in Species}:
        raise ValueError("Saving a learned world requires all five policy checkpoint files")
    if mode != "multi-learned" and files:
        raise ValueError("Non-neural world does not accept policy files")
    hashes = policy_fingerprints(files)
    controller_options = _validate_options(dict(options or {"sample_actions": True,
                                                             "festival_assist": False}))
    world.check_invariants()
    world_data = _canonical_bytes(_as_serializable_world(world))
    if len(world_data) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("World too large for a safe save file")
    manifest = {
        "format": SAVE_FORMAT, "version": SAVE_VERSION,
        "mode": mode, "seed": world.config.seed, "tick": world.tick,
        "state_hash": world.state_hash(),
        "world_sha256": hashlib.sha256(world_data).hexdigest(),
        "policy_hashes": hashes, "options": controller_options,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".qoxworld-", suffix=".tmp",
                                     dir=destination.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        with ZipFile(staged, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(MANIFEST_MEMBER, _canonical_bytes(manifest))
            archive.writestr(WORLD_MEMBER, world_data)
        if staged.stat().st_size > MAX_COMPRESSED_BYTES:
            raise ValueError("Compressed world save exceeds allowed limit")
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    return destination


def load_world(source: str | Path, *,
               policy_files: Mapping[str, str | Path] | None = None) -> tuple[World, WorldSaveInfo]:
    """Validate a world archive before constructing a mutable world.

    A learned save cannot resume under different, missing, or modified weights.
    """
    source = Path(source)
    if source.stat().st_size > MAX_COMPRESSED_BYTES:
        raise ValueError("World save exceeds compressed size limit")
    try:
        with ZipFile(source) as archive:
            members = archive.infolist()
            if (len(members) != 2 or {member.filename for member in members} !=
                    {WORLD_MEMBER, MANIFEST_MEMBER} or
                    any(member.file_size > MAX_UNCOMPRESSED_BYTES for member in members)):
                raise ValueError("Invalid or oversized QOX world save archive")
            # Inspect declared uncompressed sizes before reading. Also bound the
            # actual decompressed bytes, rather than trusting ZIP headers alone.
            with archive.open(MANIFEST_MEMBER) as stream:
                manifest_data = stream.read(1024 * 1024 + 1)
            if len(manifest_data) > 1024 * 1024:
                raise ValueError("World manifest exceeds limit")
            manifest = _validate_manifest(_parse_json(manifest_data))
            if manifest["mode"] == "multi-learned":
                if manifest["policy_hashes"] != policy_fingerprints(policy_files):
                    raise ValueError("Saved world requires its original five neural policies. "
                                     "Load the matching .qox model set first.")
            with archive.open(WORLD_MEMBER) as stream:
                payload = stream.read(MAX_UNCOMPRESSED_BYTES + 1)
            if len(payload) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("World state exceeds decompression limit")
    except BadZipFile as exc:
        raise ValueError("Not a valid .qoxworld ZIP archive") from exc
    if hashlib.sha256(payload).hexdigest() != manifest["world_sha256"]:
        raise ValueError("World save integrity check failed")
    try:
        world = _make_world(_parse_json(payload))
    except (KeyError, IndexError, AssertionError, TypeError, OverflowError) as exc:
        raise ValueError("Malformed saved world state") from exc
    if (world.tick != manifest["tick"] or world.config.seed != manifest["seed"]
            or world.state_hash() != manifest["state_hash"]):
        raise ValueError("World save does not match its verified manifest")
    info = WorldSaveInfo(world.tick, world.config.seed, manifest["mode"],
                         dict(manifest["policy_hashes"]),
                         dict(manifest["options"]), manifest["state_hash"], source)
    return world, info
