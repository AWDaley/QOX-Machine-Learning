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
        obs = observe(world, actor)
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
