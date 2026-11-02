"""Checkpoint-backed five-policy desktop sessions, safe bundles and inference.

The launcher stores neural weights, not live world state. Training only happens
when a user explicitly starts a PPO curriculum, never as a side effect of viewing.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .actions import Kind
from .entities import Species
from .perception import (ACTIONS, COW_CALL_OBSERVATION_VERSION, FARMER_OBSERVATION_VERSION,
                         NOMAD_RELATIONAL_OBSERVATION_VERSION, RAVAGER_OBSERVATION_VERSION,
                         SCAVENGER_OBSERVATION_VERSION, features_for, observe)

ROLES = tuple(Species)  # five playable actor species
BUNDLE_FORMAT = "qox-neural-ensemble"
BUNDLE_VERSION = 1
MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 80 * 1024 * 1024

# Each role's class and checkpoint loader are already used by the established QOX CLI.
MODEL_TYPES = {
    Species.COW: ("cow_ppo", "CowPolicy", "load_cow"),
    Species.FARMER: ("farmer_ppo", "FarmerPolicy", "load_farmer"),
    Species.NOMAD: ("nomad_ppo", "NomadPolicy", "load_nomad"),
    Species.SCAVENGER: ("scavenger_ppo", "ScavengerPolicy", "load_scavenger"),
    Species.RAVAGER: ("ravager_ppo", "RavagerPolicy", "load_ravager"),
}


def _implementation(species: Species):
    return importlib.import_module(f".learning.{MODEL_TYPES[species][0]}", package="qox")


def load_policy(species: Species, checkpoint: Path):
    module = _implementation(species)
    return getattr(module, MODEL_TYPES[species][2])(checkpoint)


def checkpoint_information(species: Species, checkpoint: Path) -> dict:
    """Validate content AND tensor shapes with the role's existing strict loader."""
    import torch

    if checkpoint.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError(f"{species.value} checkpoint exceeds size limit")
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"{species.value} checkpoint must be a dictionary")
    load_policy(species, checkpoint)
    return {"steps": int(data.get("completed_steps", 0)),
            "hidden": int(data.get("model_hidden", 64)),
            "observation_version": int(data["obs_version"])}


def create_fresh_checkpoint(species: Species, destination: Path, *, seed: int) -> dict:
    """Create a valid *untrained* native checkpoint compatible with QOX loaders."""
    import torch

    module = _implementation(species)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model_cls = getattr(module, MODEL_TYPES[species][1])
        model = (model_cls(64, cow_calls=True) if species == Species.COW
                 else model_cls(64, nomad_v2=True) if species == Species.NOMAD
                 else model_cls(64))
    if species == Species.COW:
        names = tuple(a.to_dict()["kind"] + (f"_{a.dx}_{a.dy}"
                       if a.kind == Kind.MOVE else "") for a in ACTIONS[species])
        version = COW_CALL_OBSERVATION_VERSION
        observations = features_for(species, cow_calls=True)
    elif species == Species.NOMAD:
        names = module.action_names()
        version = NOMAD_RELATIONAL_OBSERVATION_VERSION
        observations = features_for(species, nomad_v2=True)
    elif species == Species.FARMER:
        names = module._action_names()
        version = FARMER_OBSERVATION_VERSION
        observations = features_for(species)
    else:
        names = module.action_names()
        version = (SCAVENGER_OBSERVATION_VERSION if species == Species.SCAVENGER
                   else RAVAGER_OBSERVATION_VERSION)
        observations = features_for(species)
    data = {"policy_state": model.state_dict(), "model_hidden": 64,
            "species": species.value, "obs_version": version,
            "observation_features": observations, "actions": names,
            "completed_steps": 0, "origin": "random-untrained", "initialization_seed": seed}
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, destination)
    return checkpoint_information(species, destination)


@dataclass(frozen=True)
class PolicyStatus:
    origin: str
    steps: int
    observation_version: int


class ModelSession:
    """A launcher session with independently replaceable native .pt weights."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.status: dict[Species, PolicyStatus] = {}
        self.saved_path: Path | None = None
        self.dirty = True

    def path(self, species: Species) -> Path:
        return self.root / f"{species.value}.pt"

    @classmethod
    def fresh(cls, root: Path, *, seed: int = 42) -> ModelSession:
        session = cls(root)
        for index, species in enumerate(ROLES):
            info = create_fresh_checkpoint(species, session.path(species),
                                           seed=seed + 101 * index)
            session.status[species] = PolicyStatus("fresh", info["steps"],
                                                    info["observation_version"])
        return session

    def replace(self, species: Species, source: Path, *, origin: str = "imported") -> None:
        source = Path(source)
        info = checkpoint_information(species, source)
        target = self.path(species)
        if source.resolve() != target.resolve():
            staged = target.with_suffix(".staged")
            try:
                shutil.copyfile(source, staged)
                os.replace(staged, target)
            finally:
                staged.unlink(missing_ok=True)
        self.status[species] = PolicyStatus(origin, info["steps"], info["observation_version"])
        self.dirty = True

    def save(self, destination: Path) -> Path:
        """Atomic ZIP of five checkpoints; user can keep it anywhere."""
        destination = Path(destination)
        if destination.suffix.lower() != ".qox":
            destination = destination.with_suffix(".qox")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if any(species not in self.status for species in ROLES):
            raise ValueError("All five neural networks are required")
        manifest = {
            "format": BUNDLE_FORMAT, "version": BUNDLE_VERSION,
            "roles": {s.value: {"file": f"policies/{s.value}.pt",
                                 "origin": self.status[s].origin,
                                 "steps": self.status[s].steps,
                                 "observation_version": self.status[s].observation_version}
                      for s in ROLES},
        }
        # Revalidate the actual data before committing a saved session.
        for role in ROLES:
            checkpoint_information(role, self.path(role))
        with tempfile.NamedTemporaryFile(prefix=".qox-", suffix=".zip",
                                         dir=destination.parent, delete=False) as stream:
            staged = Path(stream.name)
        try:
            with ZipFile(staged, "w", ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, indent=2))
                for species in ROLES:
                    archive.write(self.path(species), f"policies/{species.value}.pt")
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)
        self.saved_path = destination
        self.dirty = False
        return destination

    @classmethod
    def load(cls, bundle: Path, root: Path) -> ModelSession:
        """Reject malformed bundles before exposing a partially loaded session."""
        bundle = Path(bundle)
        if bundle.stat().st_size > MAX_BUNDLE_BYTES:
            raise ValueError("Model bundle exceeds size limit")
        with ZipFile(bundle) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != BUNDLE_FORMAT or manifest.get("version") != BUNDLE_VERSION:
                raise ValueError("Not a supported QOX model bundle")
            roles = manifest.get("roles", {})
            if set(roles) != {s.value for s in ROLES}:
                raise ValueError("Model bundle must contain exactly the five species")
            expected = {"manifest.json"} | {f"policies/{s.value}.pt" for s in ROLES}
            members = archive.infolist()
            if (len(members) != len(expected) or
                    {member.filename for member in members} != expected):
                raise ValueError("Model bundle has unexpected or duplicate files")
            if any(member.file_size > MAX_CHECKPOINT_BYTES for member in members):
                raise ValueError("Model bundle contains oversized data")
            # Extract explicitly named files only; never trust zip-provided file paths.
            with tempfile.TemporaryDirectory(prefix="qox-validate-") as temporary:
                folder = Path(temporary)
                result = cls(root)
                validated: dict[Species, Path] = {}
                for species in ROLES:
                    item = roles[species.value]
                    expected_file = f"policies/{species.value}.pt"
                    if item.get("file") != expected_file:
                        raise ValueError(f"Invalid checkpoint path for {species.value}")
                    candidate = folder / f"{species.value}.pt"
                    with archive.open(expected_file) as data, candidate.open("wb") as output:
                        shutil.copyfileobj(data, output)
                    info = checkpoint_information(species, candidate)
                    if int(item.get("observation_version", -1)) != info["observation_version"]:
                        raise ValueError("Bundle metadata does not match checkpoint")
                    validated[species] = candidate
                # All files have been fully validated; now make them active.
                for species, candidate in validated.items():
                    target = result.path(species)
                    shutil.copyfile(candidate, target)
                    info = checkpoint_information(species, target)
                    result.status[species] = PolicyStatus(
                        str(roles[species.value].get("origin", "loaded")),
                        info["steps"], info["observation_version"])
        result.saved_path = bundle
        result.dirty = False
        return result


def train_role(species: Species, start: Path, destination: Path, *,
               steps: int = 4096, seed: int = 42) -> dict[str, Any]:
    """Continue PPO from the active checkpoint using an established short curriculum.

    Training runs in its own dedicated simulated scenarios, *not* live in the
    visualized world; the world viewer always uses frozen inference weights.
    """
    from .learning.cow_ppo import TrainOptions, train_cow
    from .learning.farmer_ppo import FarmerTrainOptions, train_farmer
    from .learning.nomad_ppo import NomadTrainOptions, train_nomad
    from .learning.scavenger_ppo import ScavengerTrainOptions, train_scavenger
    from .learning.ravager_ppo import RavagerTrainOptions, train_ravager

    if steps < 1:
        raise ValueError("Training steps must be positive")
    starting = checkpoint_information(species, start)
    hidden = starting["hidden"]
    roll = 256
    if species == Species.COW:
        cow_calls = starting["observation_version"] == 2
        scenario = "cow_signal" if cow_calls else "forage"
        report = train_cow(TrainOptions(steps=steps, seed=seed, worlds=2,
                                        rollout=roll, max_steps=32 if cow_calls else 64,
                                        scenario=scenario, cow_calls=cow_calls,
                                        hidden=hidden), destination,
                           init_checkpoint=start)
    elif species == Species.FARMER:
        report = train_farmer(FarmerTrainOptions(steps=steps, seed=seed, worlds=2,
                                                rollout=roll, max_steps=96,
                                                scenario="farmer_mixed", hidden=hidden),
                              destination, init_checkpoint=start)
    elif species == Species.NOMAD:
        v2 = starting["observation_version"] == 2
        report = train_nomad(NomadTrainOptions(steps=steps, seed=seed, worlds=2,
                                               rollout=roll, max_steps=8,
                                               scenario="nomad_signal" if v2 else "nomad_herd",
                                               nomad_v2=v2, hidden=hidden),
                             destination, init_checkpoint=start)
    elif species == Species.SCAVENGER:
        report = train_scavenger(ScavengerTrainOptions(
            steps=steps, seed=seed, worlds=2, rollout=roll, max_steps=48,
            scenario="scavenger_mixed", hidden=hidden), destination, init_checkpoint=start)
    else:
        report = train_ravager(RavagerTrainOptions(
            steps=steps, seed=seed, worlds=2, rollout=roll, max_steps=64,
            scenario="ravager_mixed", hidden=hidden), destination, init_checkpoint=start)
    # PPO routines report the current run length; desktop sessions should
    # display cumulative decisions across repeated train/save/load cycles.
    import torch

    trained = torch.load(destination, map_location="cpu", weights_only=True)
    trained["completed_steps"] = starting["steps"] + int(trained["completed_steps"])
    torch.save(trained, destination)
    checkpoint_information(species, destination)
    return report


class MultiPolicyController:
    """Apply all five saved/fresh neural networks simultaneously in a real world."""

    def __init__(self, session: ModelSession, *, sample_actions: bool = True,
                 festival_assist: bool = False):
        self.models = {species: load_policy(species, session.path(species)) for species in ROLES}
        self.sample_actions = sample_actions
        self.festival_assist = festival_assist

    def probabilities(self, world, actor_id: int) -> list[float]:
        import torch

        actor = world.actors[actor_id]
        model = self.models[actor.species]
        observed = observe(world, actor,
                           cow_calls=bool(getattr(model, "cow_calls", False)),
                           nomad_v2=bool(getattr(model, "nomad_v2", False)))
        with torch.no_grad():
            distribution, _ = model(torch.as_tensor(observed["observation"]).unsqueeze(0),
                                    torch.as_tensor(observed["action_mask"]).unsqueeze(0))
        return distribution.probs.squeeze(0).tolist()

    def actions(self, world) -> dict:
        # Optional *explicitly scripted* celebration actions until a unified
        # farmer/nomad festival policy can be trained. No forced movement.
        scripted = None
        if self.festival_assist and world.config.festival_enabled:
            from .agents import scripted_actions
            scripted = scripted_actions(world)
        festival_kinds = (Kind.CONTRIBUTE_FESTIVAL, Kind.START_FARMER_FESTIVAL,
                          Kind.START_NOMAD_FESTIVAL, Kind.START_COMBINED_FESTIVAL)
        result = {}
        for actor_id, actor in sorted(world.actors.items()):
            if (scripted is not None and actor.species in (Species.FARMER, Species.NOMAD)
                    and scripted[actor_id].kind in festival_kinds):
                result[actor_id] = scripted[actor_id]
                continue
            probs = self.probabilities(world, actor_id)
            if self.sample_actions:
                value = world.policy_rng.random()
                running = 0.0
                choice = len(probs) - 1
                for index, prob in enumerate(probs):
                    running += prob
                    if value <= running:
                        choice = index
                        break
            else:
                choice = max(range(len(probs)), key=probs.__getitem__)
            result[actor_id] = ACTIONS[actor.species][choice]
        return result
