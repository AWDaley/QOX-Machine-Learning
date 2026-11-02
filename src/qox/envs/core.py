"""Dependency-light ParallelEnv-compatible execution and learning scenarios.

The same transition/reward rules are used by the PettingZoo adapter and the
local PyTorch trainer. No Gymnasium or PettingZoo import is required here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import random

from ..actions import Action
from ..agents import scripted_actions
from ..config import SimulationConfig
from ..entities import Species
from ..perception import ACTIONS, get_agent_name, observe
from ..world import World


class QoxCoreEnv:
    """Simultaneous multi-agent interface with stable names for initial actors.

    M9 will add bounded possible-agent slots for births. Until then, actors can
    die but no new actors are created during an episode.
    """

    def __init__(self, config: SimulationConfig | None = None, max_steps: int = 128,
                 scenario: str = "natural", scripted_others: bool = False):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if scenario not in ("natural", "forage", "threat"):
            raise ValueError(f"Unknown scenario: {scenario}")
        if config is None:
            if scenario == "forage":
                config = SimulationConfig(width=12, height=12, farmers=0, cows=1,
                                          nomads=0, scavengers=0, ravagers=0,
                                          cow_metabolism=0.7)
            elif scenario == "threat":
                config = SimulationConfig(width=12, height=12, farmers=0, cows=2,
                                          nomads=0, scavengers=0, ravagers=1,
                                          cow_metabolism=0.7)
            else:
                config = SimulationConfig()
        self.config = config
        self.scenario = scenario
        self.scripted_others = scripted_others
        self.max_steps = max_steps
        self.world: World
        self.possible_agents: list[str] = []
        self.agents: list[str] = []
        self.name_to_id: dict[str, int] = {}
        self.reset(seed=config.seed)

    def reset(self, seed: int | None = None,
              options: dict | None = None) -> tuple[dict, dict]:
        # PettingZoo may pass nonempty options for API testing. No QOX reset
        # options are currently interpreted, so safely ignore extra keys.
        if options is not None and not isinstance(options, dict):
            raise TypeError("reset options must be a dictionary or None")
        seed = self.config.seed if seed is None else int(seed)
        self.world = World(replace(self.config, seed=seed))
        if self.scenario != "natural":
            self._scenario_setup(seed)
        self.possible_agents = [get_agent_name(a) for a in self.world.actors.values()]
        self.name_to_id = {get_agent_name(a): aid for aid, a in self.world.actors.items()}
        self.agents = list(self.possible_agents)
        return (self._observations(), {name: {} for name in self.agents})

    def _scenario_setup(self, seed: int) -> None:
        """Patchy grazing creates a testable forage-navigation task.

        Local RNG does not consume physics RNG. All vegetation and cow-adjacent
        starting patches are in world state, not hidden knowledge to a policy.
        """
        rng = random.Random(seed ^ 0x45AB90)
        for tile in self.world.grid.tiles:
            tile.crop = 0.0
            tile.planted = False
            tile.grass = rng.uniform(2.0, 5.0) if rng.random() < 0.22 else 0.0
            tile.fertility = 0.75 if tile.grass > 0 else 0.05
        for cow in (a for a in self.world.actors.values() if a.species == Species.COW):
            candidate = [(cow.x + dx, cow.y + dy)
                         for dx, dy in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0),
                                        (2, 0), (-2, 0), (0, 2), (0, -2))
                         if self.world.grid.contains(cow.x + dx, cow.y + dy)]
            x, y = rng.choice(candidate)
            tile = self.world.grid.get(x, y)
            tile.grass = max(tile.grass, 3.0)
            tile.fertility = 0.75
        if self.scenario == "threat":
            # Move the scripted predator into local sensing range of a cow.
            predator = next((a for a in self.world.actors.values()
                             if a.species == Species.RAVAGER), None)
            cow = next((a for a in self.world.actors.values()
                        if a.species == Species.COW), None)
            if predator and cow:
                occupied = {a.position for a in self.world.actors.values() if a.id != predator.id}
                entrances = {(s.x, s.y) for s in self.world.settlements}
                places = [(x, y) for x, y in self.world.grid.neighbors(cow.x, cow.y)
                          if (x, y) not in occupied and (x, y) not in entrances]
                if places:
                    predator.x, predator.y = rng.choice(places)
        self.world.check_invariants()

    def _observations(self) -> dict:
        return {name: observe(self.world, self.world.actors[self.name_to_id[name]])
                for name in self.agents}

    def step(self, actions: dict[str, int]):
        if not self.agents:
            raise RuntimeError("step() called after episode finished; call reset()")
        if unknown := set(actions) - set(self.agents):
            raise ValueError(f"Actions submitted for inactive agents: {sorted(unknown)}")
        previous = list(self.agents)
        old_health = {name: self.world.actors[self.name_to_id[name]].health
                      for name in previous}
        old_energy = {name: self.world.actors[self.name_to_id[name]].energy
                      for name in previous}
        orders: dict[int, Action] = {}
        invalid: dict[str, bool] = {}
        scripted = scripted_actions(self.world) if self.scripted_others else {}
        for name in previous:
            actor = self.world.actors[self.name_to_id[name]]
            if name in actions:
                choice = int(actions[name])
                available = observe(self.world, actor)["action_mask"]
                if not 0 <= choice < len(ACTIONS[actor.species]):
                    raise ValueError(f"Action index outside action space: {name}={choice}")
                invalid[name] = not bool(available[choice])
                orders[actor.id] = (Action() if invalid[name] else
                                    ACTIONS[actor.species][choice])
            elif self.scripted_others:
                orders[actor.id] = scripted[actor.id]
            else:
                orders[actor.id] = Action()
        self.world.step(orders)

        # Rewards are computed from actually executed physical outcomes, not
        # from intentions, and are attributed to actors using event actor_id.
        rewards = {name: -0.003 for name in previous}
        components: dict[str, dict[str, float]] = {name: defaultdict(float)
                                                        for name in previous}
        id_to_name = {aid: name for name, aid in self.name_to_id.items()}
        for event in self.world.last_events:
            name = id_to_name.get(event.actor_id)
            if name not in rewards:
                continue
            if event.kind == "graze":
                # Reward calories actually absorbed, not wasteful eating at
                # full energy. One biomass unit supplies 2.4 energy units.
                useful = min(event.amount, max(0.0, 100.0 - old_energy[name]) / 2.4)
                value = 0.45 * useful
            elif event.kind in ("harvest", "eat_waste", "eat_carcass", "milk"):
                value = 0.15 * event.amount
            elif event.kind == "deposit":
                value = 0.05 * event.amount
            else:
                value = 0.0
            rewards[name] += value
            components[name][event.kind] += value
        terminations = {}
        truncations = {}
        for name in previous:
            actor = self.world.actors.get(self.name_to_id[name])
            if actor is None:
                terminations[name] = True
                rewards[name] -= 4.0
                components[name]["death"] = -4.0
            else:
                terminations[name] = False
                health_lost = max(0.0, old_health[name] - actor.health)
                if health_lost:
                    rewards[name] -= 0.025 * health_lost
                    components[name]["injury"] = -0.025 * health_lost
                # No reward for merely existing or cyclically increasing an
                # already full energy bar; finite intake is directly rewarded.
            truncations[name] = self.world.tick >= self.max_steps
        self.agents = [name for name in previous if not (terminations[name]
                                                         or truncations[name])]
        infos = {name: {"invalid_action": invalid.get(name, False),
                        "reward_components": dict(components[name]),
                        "world_tick": self.world.tick}
                 for name in previous}
        return self._observations(), rewards, terminations, truncations, infos

    def close(self) -> None:
        """Core environment owns no graphics or external process resources."""

