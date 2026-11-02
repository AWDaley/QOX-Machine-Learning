"""Use a frozen learned cow policy alongside transparent scripted other actors."""

from __future__ import annotations

from pathlib import Path

import torch

from ..agents import scripted_actions
from ..entities import Species
from ..perception import ACTIONS, observe
from ..world import World
from .cow_ppo import load_cow


class CowController:
    def __init__(self, checkpoint: str | Path):
        self.model = load_cow(checkpoint)

    def probabilities(self, world: World, actor_id: int) -> list[float]:
        actor = world.actors[actor_id]
        if actor.species != Species.COW:
            raise ValueError("CowController only controls cows")
        obs = observe(world, actor, cow_calls=self.model.cow_calls)
        with torch.no_grad():
            state = torch.as_tensor(obs["observation"]).unsqueeze(0)
            mask = torch.as_tensor(obs["action_mask"]).unsqueeze(0)
            distribution, _ = self.model(state, mask)
            return distribution.probs.squeeze(0).tolist()

    def actions(self, world: World) -> dict:
        """Only replace cow actions; other species remain scripted baselines."""
        orders = scripted_actions(world)
        for actor in world.actors.values():
            if actor.species == Species.COW:
                probabilities = self.probabilities(world, actor.id)
                choice = max(range(len(probabilities)), key=lambda index: probabilities[index])
                orders[actor.id] = ACTIONS[Species.COW][choice]
        return orders


class FarmerController:
    """Frozen learned farmer policy; cows/nomads/others remain scripted."""

    def __init__(self, checkpoint: str | Path):
        from .farmer_ppo import load_farmer
        self.model = load_farmer(checkpoint)

    def probabilities(self, world: World, actor_id: int) -> list[float]:
        actor = world.actors[actor_id]
        if actor.species != Species.FARMER:
            raise ValueError("FarmerController only controls farmers")
        obs = observe(world, actor)
        with torch.no_grad():
            distribution, _ = self.model(
                torch.as_tensor(obs["observation"]).unsqueeze(0),
                torch.as_tensor(obs["action_mask"]).unsqueeze(0))
            return distribution.probs.squeeze(0).tolist()

    def actions(self, world: World) -> dict:
        orders = scripted_actions(world)
        for actor in world.actors.values():
            if actor.species == Species.FARMER:
                probs = self.probabilities(world, actor.id)
                choice = max(range(len(probs)), key=lambda i: probs[i])
                orders[actor.id] = ACTIONS[Species.FARMER][choice]
        return orders


class NomadController:
    """Frozen nomad policy with deterministic argmax; all neighbors scripted."""

    def __init__(self, checkpoint: str | Path):
        from .nomad_ppo import load_nomad
        self.model = load_nomad(checkpoint)

    def probabilities(self, world: World, actor_id: int) -> list[float]:
        actor = world.actors[actor_id]
        if actor.species != Species.NOMAD:
            raise ValueError("NomadController only controls nomads")
        obs = observe(world, actor, nomad_v2=self.model.nomad_v2)
        with torch.no_grad():
            distribution, _ = self.model(
                torch.as_tensor(obs["observation"]).unsqueeze(0),
                torch.as_tensor(obs["action_mask"]).unsqueeze(0))
            return distribution.probs.squeeze(0).tolist()

    def actions(self, world: World) -> dict:
        orders = scripted_actions(world)
        for actor in world.actors.values():
            if actor.species == Species.NOMAD:
                probs = self.probabilities(world, actor.id)
                # The stochastic PPO policy is sampled with a seeded RNG,
                # separate from physics. Traces record all sampled actions.
                roll = world.policy_rng.random()
                total, choice = 0.0, len(probs) - 1
                for i, probability in enumerate(probs):
                    total += probability
                    if roll <= total:
                        choice = i
                        break
                orders[actor.id] = ACTIONS[Species.NOMAD][choice]
        return orders


class HerdController:
    """Run the frozen cow-v2 and nomad policies in one real world."""

    def __init__(self, nomad_checkpoint: str | Path, cow_checkpoint: str | Path):
        self.nomad = NomadController(nomad_checkpoint)
        self.cow = CowController(cow_checkpoint)
        if not self.cow.model.cow_calls:
            raise ValueError("Combined learned herd requires a v2 call-aware cow checkpoint")

    def probabilities(self, world: World, actor_id: int) -> list[float]:
        species = world.actors[actor_id].species
        if species == Species.COW:
            return self.cow.probabilities(world, actor_id)
        if species == Species.NOMAD:
            return self.nomad.probabilities(world, actor_id)
        raise ValueError("HerdController only controls cows and nomads")

    def actions(self, world: World) -> dict:
        orders = self.cow.actions(world)
        for actor in world.actors.values():
            if actor.species == Species.NOMAD:
                probs = self.nomad.probabilities(world, actor.id)
                roll = world.policy_rng.random()
                total, choice = 0.0, len(probs) - 1
                for i, probability in enumerate(probs):
                    total += probability
                    if roll <= total:
                        choice = i
                        break
                orders[actor.id] = ACTIONS[Species.NOMAD][choice]
        return orders
