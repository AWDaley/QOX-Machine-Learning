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


class RavagerController:
    """Frozen M6 learned ravager; other populations remain scripted."""

    def __init__(self, checkpoint: str | Path):
        from .ravager_ppo import load_ravager
        self.model = load_ravager(checkpoint)

    def probabilities(self, world: World, actor_id: int) -> list[float]:
        actor = world.actors[actor_id]
        if actor.species != Species.RAVAGER:
            raise ValueError("RavagerController only controls ravagers")
        state = observe(world, actor)
        with torch.no_grad():
            distribution, _ = self.model(
                torch.as_tensor(state["observation"]).unsqueeze(0),
                torch.as_tensor(state["action_mask"]).unsqueeze(0))
            return distribution.probs.squeeze(0).tolist()

    def actions(self, world: World) -> dict:
        orders = scripted_actions(world)
        for actor in world.actors.values():
            if actor.species == Species.RAVAGER:
                probabilities = self.probabilities(world, actor.id)
                choice = max(range(len(probabilities)), key=lambda index: probabilities[index])
                orders[actor.id] = ACTIONS[Species.RAVAGER][choice]
        return orders


class FarmerController:
    """Frozen learned farmer policy; cows/nomads/others remain scripted."""

    def __init__(self, checkpoint: str | Path):
        from .farmer_ppo import load_farmer
        self.model = load_farmer(checkpoint)
        self._world_identity: int | None = None
        self._memory: dict[int, object] = {}
        self._last_tick: dict[int, int] = {}

    def _actor_history(self, world: World, actor_id: int):
        """Consume only this actor's last completed tick, at most once."""
        from .farmer_task_memory import FarmerTaskHistory

        identity = id(world)
        if identity != self._world_identity:
            self._memory.clear()
            self._last_tick.clear()
            self._world_identity = identity
        if actor_id not in self._memory or world.tick < self._last_tick[actor_id]:
            self._memory[actor_id] = FarmerTaskHistory()
            self._last_tick[actor_id] = world.tick
        last = self._last_tick[actor_id]
        if world.tick > last:
            self._memory[actor_id].advance(world.last_events, actor_id,
                                           steps=world.tick - last)
            self._last_tick[actor_id] = world.tick
        return self._memory[actor_id]

    def probabilities(self, world: World, actor_id: int) -> list[float]:
        actor = world.actors[actor_id]
        if actor.species != Species.FARMER:
            raise ValueError("FarmerController only controls farmers")
        obs = observe(world, actor)
        from .farmer_task_memory import farmer_policy_observation
        history = (self._actor_history(world, actor_id)
                   if self.model.history_mode != "none" else None)
        inputs = farmer_policy_observation(obs["observation"],
                                           self.model.history_mode, history)
        with torch.no_grad():
            distribution, _ = self.model(
                torch.as_tensor(inputs).unsqueeze(0),
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


class ScavengerController:
    """Run a frozen scavenger PPO policy; keep other roles scripted."""

    def __init__(self, checkpoint: str | Path):
        from .scavenger_ppo import load_scavenger
        self.model = load_scavenger(checkpoint)

    def probabilities(self, world: World, actor_id: int) -> list[float]:
        actor = world.actors[actor_id]
        if actor.species != Species.SCAVENGER:
            raise ValueError("ScavengerController only controls scavengers")
        state = observe(world, actor)
        with torch.no_grad():
            dist, _ = self.model(torch.as_tensor(state["observation"]).unsqueeze(0),
                                 torch.as_tensor(state["action_mask"]).unsqueeze(0))
            return dist.probs.squeeze(0).tolist()

    def actions(self, world: World) -> dict:
        orders = scripted_actions(world)
        for actor in world.actors.values():
            if actor.species == Species.SCAVENGER:
                probs = self.probabilities(world, actor.id)
                choice = max(range(len(probs)), key=lambda index: probs[index])
                orders[actor.id] = ACTIONS[Species.SCAVENGER][choice]
        return orders


class FestivalController:
    """Use a frozen M8 farmer/nomad festival policy in a normal world.

    The full world must have festival_enabled=True. Other roles, including
    ravagers, retain their scripted baselines so festival consequences are real.
    """

    def __init__(self, checkpoint: str | Path):
        from .festival_ppo import load_festival, species_for
        self.model, self.scenario = load_festival(checkpoint)
        self.species = species_for(self.scenario)

    def probabilities(self, world: World, actor_id: int) -> list[float]:
        if not world.config.festival_enabled:
            raise ValueError("Festival policy requires festival_enabled=True")
        actor = world.actors[actor_id]
        if actor.species != self.species:
            raise ValueError("FestivalController only controls its trained species")
        state = observe(world, actor, festivals=True)
        with torch.no_grad():
            dist, _ = self.model(
                torch.as_tensor(state["observation"]).unsqueeze(0),
                torch.as_tensor(state["action_mask"]).unsqueeze(0))
            return dist.probs.squeeze(0).tolist()

    def actions(self, world: World) -> dict:
        from ..perception import action_catalog
        orders = scripted_actions(world)
        for actor in world.actors.values():
            if actor.species == self.species:
                probs = self.probabilities(world, actor.id)
                choice = max(range(len(probs)), key=lambda i: probs[i])
                orders[actor.id] = action_catalog(self.species, festivals=True)[choice]
        return orders
