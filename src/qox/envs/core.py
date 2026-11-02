"""Dependency-light ParallelEnv-compatible execution and learning scenarios.

The same transition/reward rules are used by the PettingZoo adapter and the
local PyTorch trainer. No Gymnasium or PettingZoo import is required here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import random

import numpy as np

from ..actions import Action, Kind
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
                 scenario: str = "natural", scripted_others: bool = False,
                 cow_calls: bool = False, allow_nomad_calls: bool = True,
                 nomad_v2: bool = False):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if scenario not in ("natural", "forage", "threat", "agriculture", "refuge", "manure", "farmer_mixed", "nomad_herd", "nomad_defend", "nomad_mixed", "cow_signal", "nomad_signal", "ravager_ambush", "ravager_carcass", "ravager_hibernate", "ravager_mixed"):
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
            elif scenario in ("agriculture", "manure", "farmer_mixed"):
                config = SimulationConfig(width=12, height=12, farmers=1, cows=1,
                                          nomads=0, scavengers=0, ravagers=0,
                                          crop_growth=0.42, grass_growth=0.32)
            elif scenario in ("nomad_herd", "nomad_defend", "nomad_mixed"):
                config = SimulationConfig(width=14, height=12, farmers=0, cows=3,
                                          nomads=1, scavengers=0,
                                          ravagers=1 if scenario == "nomad_defend" else 0,
                                          grass_growth=0.12, cow_metabolism=0.16)
            elif scenario == "nomad_signal":
                config = SimulationConfig(width=20, height=20, farmers=0, cows=1,
                                          nomads=1, scavengers=0, ravagers=0,
                                          cow_metabolism=0.12, grass_growth=0.0)
            elif scenario == "cow_signal":
                config = SimulationConfig(width=20, height=20, farmers=0, cows=1,
                                          nomads=1, scavengers=0, ravagers=0,
                                          cow_metabolism=0.12, grass_growth=0.0)
            elif scenario == "refuge":
                config = SimulationConfig(width=12, height=12, farmers=1, cows=0,
                                          nomads=0, scavengers=0, ravagers=1)
            elif scenario in ("ravager_ambush", "ravager_carcass", "ravager_hibernate",
                              "ravager_mixed"):
                config = SimulationConfig(width=20, height=14, farmers=0,
                                          cows=1 if scenario in ("ravager_ambush", "ravager_mixed")
                                          else 0,
                                          nomads=2 if scenario == "ravager_carcass" else 0,
                                          scavengers=0, ravagers=1,
                                          grass_growth=0.10)
            else:
                config = SimulationConfig()
        if scenario in ("nomad_herd", "nomad_defend", "nomad_mixed"):
            if config.cows != 3 or config.nomads != 1 or config.farmers or config.scavengers:
                raise ValueError("Nomad curricula require exactly 3 cows, 1 nomad, "
                                 "no farmers and no scavengers")
            expected_ravagers = int(scenario == "nomad_defend")
            if scenario != "nomad_mixed" and config.ravagers != expected_ravagers:
                raise ValueError("Nomad curriculum predator count mismatch")
        self.config = config
        self.scenario = scenario
        self.scripted_others = scripted_others
        self.cow_calls = cow_calls
        self.nomad_v2 = nomad_v2
        self.allow_nomad_calls = allow_nomad_calls
        self.reached_goal_ids: set[int] = set()
        self.signal_kind: Kind | None = None
        self.flip_signal = False  # Counterfactual: different call, same world target.
        self.signal_committed = False  # M5b single-call curriculum, not world physics.
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
        config = self.config
        if self.scenario == "nomad_mixed":
            config = replace(config, ravagers=1 if seed % 2 else 0)
        if self.scenario == "farmer_mixed":
            # Training-only curriculum: instantiate the correct finite
            # population for each task rather than leave extra predators
            # loose in the agriculture task.
            refuge_episode = seed % 3 == 1
            config = replace(config, cows=0 if refuge_episode else 1,
                             ravagers=1 if refuge_episode else 0)
        if self.scenario == "ravager_mixed":
            config = replace(config, cows=int(seed % 3 == 0), nomads=2 * int(seed % 3 == 1))
        self.world = World(replace(config, seed=seed))
        if self.scenario != "natural":
            self._scenario_setup(seed)
        self.reached_goal_ids = set()
        self.signal_committed = False
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
        if self.scenario in ("nomad_herd", "nomad_defend", "nomad_mixed"):
            self._nomad_scenario(rng, seed)
            return
        if self.scenario.startswith("ravager_"):
            self._ravager_scenario(rng, seed)
            return
        if self.scenario == "cow_signal":
            self._cow_signal_scenario(rng)
            return
        if self.scenario == "nomad_signal":
            self._nomad_signal_scenario(rng)
            return
        if self.scenario in ("agriculture", "refuge", "manure", "farmer_mixed", "nomad_herd", "nomad_defend", "nomad_mixed"):
            self._farmer_scenario(rng, seed)
            return
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

    def _ravager_scenario(self, rng: random.Random, seed: int) -> None:
        """Seed-controlled hunting, drag-to-cover and hibernation curricula.

        The den is just a high-cover region; it is not a hidden waypoint or a
        teleport destination. Reward comes from consumed food and staying safe.
        """
        from ..entities import Carcass

        mode = ("ravager_ambush", "ravager_carcass", "ravager_hibernate")[seed % 3] \
            if self.scenario == "ravager_mixed" else self.scenario
        self.ravager_episode_mode = mode
        ravager = next(a for a in self.world.actors.values()
                       if a.species == Species.RAVAGER)
        ravager.x, ravager.y = 7, 7
        ravager.energy, ravager.stamina = 50.0, 100.0
        for tile in self.world.grid.tiles:
            tile.planted, tile.crop, tile.waste = False, 0.0, 0.0
            tile.grass, tile.fertility, tile.cover = 0.04, 0.4, 0.0
        # Create cover patches large enough to detect in local perception.
        for x in range(3, 6):
            for y in range(6, 9):
                self.world.grid.get(x, y).cover = 0.85
        if mode == "ravager_ambush":
            cow = next(a for a in self.world.actors.values()
                       if a.species == Species.COW)
            cow.x, cow.y = 11, 7 + (seed % 3 - 1)
            cow.health = 55
            cow.energy = 60
            self.world.grid.get(*cow.position).grass = 8.0
            self.world.grid.get(*cow.position).fertility = 0.85
            # Starting cover allows sneaking; open land near the cow remains visible.
            self.world.grid.get(*ravager.position).cover = 0.8
        elif mode == "ravager_carcass":
            cid = self.world.next_carcass_id
            self.world.carcasses[cid] = Carcass(cid, 9, 7, 12.0, Species.COW)
            self.world.next_carcass_id += 1
            defenders = [a for a in self.world.actors.values()
                         if a.species == Species.NOMAD]
            for defender, (x, y) in zip(defenders, ((10, 7), (9, 8)), strict=True):
                defender.x, defender.y = x, y
                defender.stamina = 100.0
        else:
            ravager.energy, ravager.satiety = 92, 0.85
            ravager.x, ravager.y = 4, 7
        self.world.check_invariants()

    def _nomad_scenario(self, rng: random.Random, seed: int) -> None:
        """Herd migration or cow-defense test with observable local interactions.

        The herd's target pasture is known to the nomad (as a navigation
        objective), but not automatically to each cow or ravager.
        """
        nomad = next(a for a in self.world.actors.values()
                     if a.species == Species.NOMAD)
        cows = sorted((a for a in self.world.actors.values()
                       if a.species == Species.COW), key=lambda a: a.id)
        threat = self.scenario == "nomad_defend" or (
            self.scenario == "nomad_mixed" and seed % 2 == 1)
        # Four migration arrangements: left/right pasture and nomad on
        # the leading/trailing edge. A fixed "always scary" policy is not
        # adequate for all four. Local terrain remains seeded and replayable.
        flip = bool(seed % 2)
        ahead = bool(seed % 4 >= 2)
        offset = (seed // 4) % 3 - 1
        herd_x = 9 if flip and not threat else 4
        center_y = 5 + offset
        coords = [(herd_x, center_y - 1), (herd_x, center_y),
                  (herd_x, center_y + 1)]
        for cow, (x, y) in zip(cows, coords, strict=True):
            cow.x, cow.y = x, y
            cow.energy = 75
            cow.health = 55.0 if threat else 100.0
        if threat:
            nomad.x, nomad.y = (5, 3)
        else:
            # The leading nomad must keep moving to pull the herd along.
            nomad.x = (7 if flip else 6) if ahead else (11 if flip else 2)
            nomad.y = center_y
        nomad.stamina, nomad.energy = 100.0, 80.0
        for tile in self.world.grid.tiles:
            tile.crop = tile.waste = 0.0
            tile.planted = False
            tile.cover = 0.0
            tile.grass = 0.04
            tile.fertility = 0.4
        goal = (1 if flip else 12, center_y)
        self.world.herd_goal = None if threat else goal
        range_x = range(0, 4) if flip else range(10, 14)
        for x in range_x:
            for y in range(center_y - 2, center_y + 3):
                tile = self.world.grid.get(x, y)
                tile.grass = 6.0 + rng.random()
                tile.fertility = 0.9
        if threat:
            predator = next(a for a in self.world.actors.values()
                            if a.species == Species.RAVAGER)
            predator.x, predator.y = 7, 5
            predator.stamina = 100.0
            # A wounded predator can be beaten by a coordinated defense, but
            # is individually much more threatening than a nomad.
            predator.health = 64.0
        self.world.check_invariants()

    def _nomad_signal_scenario(self, rng: random.Random) -> None:
        """M5b: stationary caller chooses push/pull for a visible one-cow goal.

        The training nomad knows the assigned destination, not a secret answer
        flag. Cow movement remains selected by the ordinary *scripted* cow
        controller from locally audible sound. The nomad must choose which
        signal changes the cow's behavior toward the requested waypoint.
        """
        cow = next(a for a in self.world.actors.values() if a.species == Species.COW)
        nomad = next(a for a in self.world.actors.values()
                     if a.species == Species.NOMAD)
        dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        pull = bool(rng.getrandbits(1))
        cow.x, cow.y, cow.energy = 10, 10, 80.0
        nomad.x, nomad.y, nomad.stamina = 10 + 4 * dx, 10 + 4 * dy, 100.0
        self.signal_kind = Kind.FRIENDLY_CALL if pull else Kind.SCARY_CALL
        direction = 1 if pull else -1
        self.world.herd_goal = (10 + 2 * dx * direction,
                                10 + 2 * dy * direction)
        for tile in self.world.grid.tiles:
            tile.cover = 0.0
            tile.grass = 0.01
            tile.crop = tile.waste = 0.0
            tile.planted = False
            tile.fertility = 0.2
        self.world.check_invariants()

    def _cow_signal_scenario(self, rng: random.Random) -> None:
        """Balanced two-signal conditioning, independent of fixed map direction.

        The training goal is hidden from the cow: only a correct interpretation
        of the nomad's locally audible call identifies the direction. Friendly
        means approach the caller; scary means move away from it. Calling never
        applies forces, and the destination is a *curriculum* waypoint rather
        than a target supplied to the network. This is not a full herd task.
        """
        cow = next(a for a in self.world.actors.values() if a.species == Species.COW)
        nomad = next(a for a in self.world.actors.values() if a.species == Species.NOMAD)
        dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        self.signal_kind = rng.choice((Kind.FRIENDLY_CALL, Kind.SCARY_CALL))
        cow.x, cow.y, cow.energy = 10, 10, 75
        # Put caller one tile to the side, never directly in the way.
        nomad.x, nomad.y = (10 + 2 * dx - dy, 10 + 2 * dy + dx)
        nomad.stamina = 100
        side = 1 if self.signal_kind == Kind.FRIENDLY_CALL else -1
        self.world.herd_goal = (10 + 5 * dx * side, 10 + 5 * dy * side)
        for tile in self.world.grid.tiles:
            tile.cover = 0.0
            tile.grass = 0.01  # no visible pasture clue or alternative nutrition
            tile.crop = tile.waste = 0.0
            tile.planted = False
            tile.fertility = 0.2
        self.world.check_invariants()

    def _farmer_scenario(self, rng: random.Random, seed: int) -> None:
        """Small, concrete curriculum tasks with independent physics RNG.

        The modes differ in the world they create, not the farmer's action set.
        A mixed episode alternates task types using the seed; the observation
        never provides a hidden scenario label.
        """
        mode = ("agriculture", "refuge", "manure")[seed % 3] if self.scenario == "farmer_mixed" else self.scenario
        home = self.world.settlements[0]
        farmer = next(a for a in self.world.actors.values()
                      if a.species == Species.FARMER)
        farmer.x, farmer.y = home.x + 1, home.y
        farmer.carried_food = 1.2
        farmer.energy = 70.0
        home.stored_food = 2.0
        for tile in self.world.grid.tiles:
            tile.crop, tile.waste, tile.planted = 0.0, 0.0, False
            tile.fertility = 0.45
            tile.grass = 0.2
        fields = [(home.x + 1, home.y), (home.x + 2, home.y),
                  (home.x + 1, home.y + 1), (home.x + 1, home.y - 1),
                  (home.x - 1, home.y), (home.x - 1, home.y + 1)]
        for i, (x, y) in enumerate(fields):
            tile = self.world.grid.get(x, y)
            tile.fertility = 0.65 + 0.23 * rng.random()
            if i not in (1, 5):
                tile.planted = True
                tile.crop = 1.3 + 1.8 * rng.random()
        if mode == "manure":
            tile = self.world.grid.get(*farmer.position)
            tile.planted, tile.crop, tile.fertility, tile.waste = False, 0, 0.19, 1.8
            self.world.grid.get(home.x + 2, home.y).waste = 2
            home.stored_food = 1.0
        if mode == "agriculture":
            self.world.grid.get(home.x + 2, home.y).waste = 0.8
        if mode == "refuge":
            ravager = next(a for a in self.world.actors.values()
                           if a.species == Species.RAVAGER)
            ravager.x, ravager.y = home.x + 3, home.y
            home.stored_food = 4.0
        else:
            cow = next((a for a in self.world.actors.values()
                        if a.species == Species.COW), None)
            if cow:
                cow.x, cow.y = home.x + 3, home.y + 1
                self.world.grid.get(*cow.position).grass = 4.5
        self.world.check_invariants()

    def _legal_mask(self, actor) -> "np.ndarray":
        """Scenario curriculum restrictions in addition to physics legality.

        Signal training is explicitly stationary; movements become available
        again, with the same action indices and policy weights, in nomad_herd.
        """
        from ..perception import legal_action_mask
        mask = legal_action_mask(self.world, actor)
        if actor.species == Species.NOMAD:
            if not self.allow_nomad_calls:
                for kind in (Kind.FRIENDLY_CALL, Kind.SCARY_CALL):
                    mask[ACTIONS[Species.NOMAD].index(Action(kind))] = 0
            if self.scenario == "nomad_signal":
                allowed = {Kind.WAIT, Kind.REST, Kind.FRIENDLY_CALL, Kind.SCARY_CALL}
                mask = mask.copy()
                for index, action in enumerate(ACTIONS[Species.NOMAD]):
                    if action.kind not in allowed or (self.signal_committed and action.kind in
                                                      (Kind.FRIENDLY_CALL, Kind.SCARY_CALL)):
                        mask[index] = 0
        return mask

    def _observations(self) -> dict:
        observations = {}
        for name in self.agents:
            actor = self.world.actors[self.name_to_id[name]]
            state = observe(self.world, actor, cow_calls=self.cow_calls,
                            nomad_v2=self.nomad_v2)
            state["action_mask"] = self._legal_mask(actor)
            observations[name] = state
        return observations

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
        old_cows = {aid: (a.position, a.health)
                    for aid, a in self.world.actors.items() if a.species == Species.COW}
        # Optional, very small potential-based pursuit shaping for the ravager
        # curriculum. Targets must be locally perceptible, not omniscient.
        ravager_target_distances: dict[int, tuple[tuple[int, int], str, int]] = {}
        if self.scenario.startswith("ravager_"):
            from ..perception import _visible_actor
            for hunter in self.world.actors.values():
                if hunter.species != Species.RAVAGER:
                    continue
                prey = sorted((other for other in self.world.actors.values()
                               if other.species != Species.RAVAGER and not other.sheltered
                               and _visible_actor(self.world, hunter, other)),
                              key=lambda o: (abs(o.x - hunter.x) + abs(o.y - hunter.y), o.id))
                near_carcass = sorted((carcass for carcass in self.world.carcasses.values()
                                      if carcass.carrier_id is None and
                                      abs(carcass.x - hunter.x) +
                                      abs(carcass.y - hunter.y) <= 7),
                                     key=lambda c: (abs(c.x - hunter.x) +
                                                     abs(c.y - hunter.y), c.id))
                target = prey[0] if prey else near_carcass[0] if near_carcass else None
                if target is not None:
                    kind = "actor" if prey else "carcass"
                    ravager_target_distances[hunter.id] = (hunter.position, kind, target.id)
        goal = self.world.herd_goal
        old_goal_distances = {aid: abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
                              for aid, (pos, _) in old_cows.items()} if goal else {}
        orders: dict[int, Action] = {}
        invalid: dict[str, bool] = {}
        scripted = scripted_actions(self.world) if self.scripted_others else {}
        for name in previous:
            actor = self.world.actors[self.name_to_id[name]]
            if name in actions:
                choice = int(actions[name])
                available = self._legal_mask(actor)
                if not 0 <= choice < len(ACTIONS[actor.species]):
                    raise ValueError(f"Action index outside action space: {name}={choice}")
                invalid[name] = not bool(available[choice])
                orders[actor.id] = (Action() if invalid[name] else
                                    ACTIONS[actor.species][choice])
            elif self.scripted_others:
                chosen = scripted[actor.id]
                if (self.scenario.startswith("ravager_")
                        and getattr(self, "ravager_episode_mode", None) ==
                        "ravager_carcass" and actor.species == Species.NOMAD):
                    # Scenario-only sentry: contests food within melee reach,
                    # but doesn't chase a retreating carcass carrier. This
                    # supplies a reason to relocate a kill without commanding
                    # the ravager to drag it.
                    close = any(other.species == Species.RAVAGER and
                                abs(other.x - actor.x) + abs(other.y - actor.y) <= 1
                                for other in self.world.actors.values())
                    chosen = Action(Kind.ATTACK if close else Kind.WAIT)
                if self.scenario == "cow_signal" and actor.species == Species.NOMAD:
                    kind = (Kind.SCARY_CALL if self.signal_kind == Kind.FRIENDLY_CALL
                            else Kind.FRIENDLY_CALL) if self.flip_signal else self.signal_kind
                    cooldown = (actor.friendly_cooldown if kind == Kind.FRIENDLY_CALL
                                else actor.scary_cooldown)
                    chosen = Action(kind) if cooldown == 0 else Action(Kind.REST)
                if (not self.allow_nomad_calls and actor.species == Species.NOMAD
                        and chosen.kind in (Kind.FRIENDLY_CALL, Kind.SCARY_CALL)):
                    chosen = Action()
                orders[actor.id] = chosen
            else:
                orders[actor.id] = Action()
        self.world.step(orders)
        if self.scenario == "nomad_signal":
            self.signal_committed |= any(e.kind in (Kind.FRIENDLY_CALL.value,
                                                    Kind.SCARY_CALL.value)
                                         for e in self.world.last_events)

        # Rewards are computed from actually executed physical outcomes, not
        # from intentions, and are attributed to actors using event actor_id.
        rewards = {name: -0.003 for name in previous}
        components: dict[str, dict[str, float]] = {name: defaultdict(float)
                                                        for name in previous}
        id_to_name = {aid: name for name, aid in self.name_to_id.items()}
        for aid, (old_pos, kind, target_id) in ravager_target_distances.items():
            actor = self.world.actors.get(aid)
            target = (self.world.actors.get(target_id) if kind == "actor"
                      else self.world.carcasses.get(target_id))
            name = id_to_name.get(aid)
            if actor is None or target is None or name not in rewards:
                continue
            # Signed distance improvement: backtracking costs what approach
            # gains; the action itself receives no arbitrary bonus.
            old_distance = abs(old_pos[0] - target.x) + abs(old_pos[1] - target.y)
            new_distance = abs(actor.x - target.x) + abs(actor.y - target.y)
            difference = 0.08 * (old_distance - new_distance)
            rewards[name] += difference
            components[name]["foraging_progress"] += difference
        for event in self.world.last_events:
            name = id_to_name.get(event.actor_id)
            if name not in rewards:
                continue
            if event.kind == "graze":
                # Reward calories actually absorbed, not wasteful eating at
                # full energy. One biomass unit supplies 2.4 energy units.
                useful = min(event.amount, max(0.0, 100.0 - old_energy[name]) / 2.4)
                value = 0.45 * useful
            elif event.kind == "harvest":
                value = 0.12 * event.amount
            elif event.kind == "fertilize":
                # Credit only real finite manure applied to depleted soil.
                # This is weak shaping; harvest/food-storage outcomes dominate.
                value = 0.025 * event.amount
            elif event.kind in ("eat_waste", "eat_carcass", "milk"):
                value = (0.50 if name.startswith("ravager_") and
                         event.kind == "eat_carcass" else 0.15) * event.amount
            elif event.kind == "charge_hit" and name.startswith("ravager_"):
                value = 0.012 * event.amount
            elif event.kind == "deposit":
                value = 0.55 * event.amount
            else:
                value = 0.0
            rewards[name] += value
            components[name][event.kind] += value
        if self.scenario in ("nomad_herd", "nomad_defend", "nomad_mixed",
                             "cow_signal", "nomad_signal"):
            actual_cows = {aid: a for aid, a in self.world.actors.items()
                           if a.species == Species.COW}
            gained = set()
            for name in previous:
                if not name.startswith("nomad_"):
                    continue
                if goal is not None:
                    # Progress depends exclusively on actual cattle movement,
                    # not call use. Once-per-cow pasture arrival prevents loops.
                    progress = sum(old_dist - (abs(actual_cows[aid].x - goal[0])
                                                + abs(actual_cows[aid].y - goal[1]))
                                   for aid, old_dist in old_goal_distances.items()
                                   if aid in actual_cows)
                    delta = 0.4 * progress / max(1, len(old_cows))
                    rewards[name] += delta
                    components[name]["herd_progress"] += delta
                    newly_reached = {aid for aid, cow in actual_cows.items()
                                     if aid not in self.reached_goal_ids
                                     and abs(cow.x - goal[0]) + abs(cow.y - goal[1]) <=
                                     (0 if self.scenario == "nomad_signal" else 2)}
                    gained.update(newly_reached)
                    for _ in newly_reached:
                        rewards[name] += 2.0
                        components[name]["new_cow_in_pasture"] += 2.0
                # Penalize harm to cattle, rather than paying a nomad for
                # arbitrary attack actions or staying near the herd.
                damage = sum(max(0, health - actual_cows[aid].health)
                             for aid, (_, health) in old_cows.items() if aid in actual_cows)
                deaths = sum(aid not in actual_cows for aid in old_cows)
                value = -0.04 * damage - 2.5 * deaths
                rewards[name] += value
                components[name]["herd_harm"] += value
            # A cow learns only from actual displacement and nourishment,
            # not from a magic instruction to obey the nomad's call.
            for name in previous:
                if not name.startswith("cow_"):
                    continue
                aid = self.name_to_id[name]
                if goal is not None and aid in old_goal_distances and aid in actual_cows:
                    cow = actual_cows[aid]
                    change = old_goal_distances[aid] - (
                        abs(cow.x - goal[0]) + abs(cow.y - goal[1]))
                    rewards[name] += 0.4 * change
                    components[name]["pasture_progress"] += 0.4 * change
                    if aid in gained:
                        rewards[name] += 2.0
                        components[name]["first_pasture_arrival"] += 2.0
            self.reached_goal_ids.update(gained)
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
                    injury_weight = 0.07 if name.startswith("farmer_") else 0.025
                    rewards[name] -= injury_weight * health_lost
                    components[name]["injury"] = -injury_weight * health_lost
                # No reward for merely existing or cyclically increasing an
                # already full energy bar; finite intake is directly rewarded.
            truncations[name] = self.world.tick >= self.max_steps
            if truncations[name] and name.startswith("nomad_") and self.scenario in (
                    "nomad_herd", "nomad_defend", "nomad_mixed"):
                # End-of-episode cow health/retention matters for nomad defense.
                retained = [a for a in self.world.actors.values()
                            if a.species == Species.COW]
                value = 1.5 * (sum(a.health / 100 for a in retained) /
                               max(1, len(old_cows)))
                rewards[name] += value
                components[name]["herd_condition_at_end"] += value
            if (truncations[name] and name.startswith("ravager_")
                    and self.scenario.startswith("ravager_") and actor is not None):
                # In the satiated episode, conservation of *actual* energy is
                # the outcome. No reward is paid for simply pressing hibernate.
                if self.ravager_episode_mode == "ravager_hibernate":
                    value = (actor.energy - 75.0) / 10.0
                    rewards[name] += value
                    components[name]["final_energy"] += value
                else:
                    value = 0.025 * (actor.energy - 50.0)
                    rewards[name] += value
                    components[name]["final_energy"] += value
        self.agents = [name for name in previous if not (terminations[name]
                                                         or truncations[name])]
        infos = {name: {"invalid_action": invalid.get(name, False),
                        "reward_components": dict(components[name]),
                        "world_tick": self.world.tick}
                 for name in previous}
        return self._observations(), rewards, terminations, truncations, infos

    def close(self) -> None:
        """Core environment owns no graphics or external process resources."""

