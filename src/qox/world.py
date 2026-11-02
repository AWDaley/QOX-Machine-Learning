"""Headless, seeded world physics for QOX M1. Policies live outside this module."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import random

from .actions import Action, Kind
from .config import SimulationConfig
from .entities import Actor, CallSignal, Carcass, Event, Settlement, Species, Tile
from .grid import Grid
from .population import (LIFESPAN, birth_step, mature_age,
                         population_caps, is_juvenile)
from .festivals import (Camp, Festival, FestivalKind, FestivalResult,
                        advance as advance_festivals, can_start as can_start_festival,
                        contribution_target, site_for as festival_site,
                        start as start_festival)


class World:
    """A fixed-step, reproducible world; one actor occupies each tile."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.rng = random.Random(self.config.seed)
        # Policy sampling must never advance the physics RNG: traces replay only actions.
        self.policy_rng = random.Random(self.config.seed ^ 0x5DEECE66D)
        self.tick = 0
        self.events: list[Event] = []
        self.last_events: list[Event] = []
        self.next_id = 1
        self.next_carcass_id = 1
        self.birth_totals = {species: 0 for species in Species}
        tiles = []
        for _ in range(self.config.width * self.config.height):
            fertility = self.rng.uniform(0.3, 0.95)
            tiles.append(Tile(fertility=fertility, grass=self.rng.uniform(2, 7) * fertility,
                              cover=self.rng.uniform(0, 0.65)))
        self.grid = Grid(self.config.width, self.config.height, tiles)
        w, h = self.config.width, self.config.height
        self.settlements = [Settlement(1, w // 4, h // 2), Settlement(2, 3 * w // 4, h // 2)]
        self.actors: dict[int, Actor] = {}
        self.carcasses: dict[int, Carcass] = {}
        self.calls: list[CallSignal] = []
        self.herd_goal: tuple[int, int] | None = None
        self._populate()
        # Camps are fixed home markers, not shelters or teleport destinations.
        self.camps: list[Camp] = []
        if self.config.festival_enabled:
            for band in sorted({a.band_id for a in self.actors.values()
                                if a.species == Species.NOMAD and a.band_id is not None}):
                leader = min((a for a in self.actors.values() if a.band_id == band),
                             key=lambda a: a.id)
                self.camps.append(Camp(band, leader.x, leader.y))
        self.active_festivals: dict[int, Festival] = {}
        self.festival_results: list[FestivalResult] = []
        self.next_festival_id = 1
        self.check_invariants()

    def _populate(self) -> None:
        available = {(x, y) for y in range(self.grid.height) for x in range(self.grid.width)}
        for settlement in self.settlements:
            available.remove((settlement.x, settlement.y))
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    x, y = settlement.x + dx, settlement.y + dy
                    if self.grid.contains(x, y) and self.rng.random() < 0.36:
                        tile = self.grid.get(x, y)
                        tile.planted = True
                        tile.crop = self.rng.uniform(1.0, 3.0)
        specs = ((Species.FARMER, self.config.farmers), (Species.COW, self.config.cows),
                 (Species.NOMAD, self.config.nomads), (Species.SCAVENGER, self.config.scavengers),
                 (Species.RAVAGER, self.config.ravagers))
        for species, count in specs:
            for i in range(count):
                home = self.settlements[i % len(self.settlements)]
                if species == Species.FARMER:
                    near = [p for p in available if abs(p[0] - home.x) + abs(p[1] - home.y) <= 6]
                elif species == Species.NOMAD:
                    cows = [a for a in self.actors.values() if a.species == Species.COW]
                    cow = cows[i % len(cows)] if cows else None
                    near = [p for p in available if cow and abs(p[0] - cow.x) + abs(p[1] - cow.y) <= 5]
                else:
                    near = []
                candidates = sorted(near or available)
                x, y = candidates[self.rng.randrange(len(candidates))]
                available.remove((x, y))
                actor = Actor(id=self.next_id, species=species, x=x, y=y,
                              home_id=home.id if species == Species.FARMER else None,
                              band_id=1 + i % 2 if species == Species.NOMAD else None,
                              carried_food=2.0 if species == Species.FARMER else 0.0)
                if (self.config.population_dynamics_enabled or
                        self.config.reproduction_enabled):
                    # Starting residents are adults; later births enter as juveniles.
                    actor.age = mature_age(species)
                self.actors[actor.id] = actor
                self.next_id += 1

    def emit(self, kind: str, actor_id: int | None = None, amount: float = 0.0,
             detail: str = "") -> None:
        actor = self.actors.get(actor_id) if actor_id is not None else None
        event = Event(self.tick, kind, actor_id, round(amount, 6), detail,
                      actor.x if actor else None, actor.y if actor else None)
        self.events.append(event)
        self.last_events.append(event)

    def emit_at(self, kind: str, x: int, y: int, actor_id: int | None = None,
                amount: float = 0.0, detail: str = "") -> None:
        event = Event(self.tick, kind, actor_id, round(amount, 6), detail, x, y)
        self.events.append(event)
        self.last_events.append(event)

    def _movement(self, orders: dict[int, Action]) -> None:
        """Resolve destinations together; permit swaps/chains but never co-occupation."""
        # Sheltered farmers occupy interior refuge slots, not outside grid cells.
        occupied = {a.position: a.id for a in self.actors.values() if not a.sheltered}
        requests: dict[tuple[int, int], list[int]] = defaultdict(list)
        for aid in sorted(self.actors):
            actor = self.actors[aid]
            action = orders.get(aid, Action())
            if is_juvenile(self, actor) and action.kind == Kind.DASH:
                continue
            if action.kind not in (Kind.MOVE, Kind.DASH, Kind.SNEAK):
                continue
            is_dash = action.kind == Kind.DASH
            is_sneak = action.kind == Kind.SNEAK
            if (actor.hibernating or actor.charge_windup or actor.charge_recovery
                    or actor.drag_cooldown and actor.dragged_carcass_id is not None
                    or is_sneak and (actor.species != Species.RAVAGER
                                     or actor.sneak_cooldown)):
                continue
            cost = (self.config.nomad_dash_stamina if is_dash else
                    self.config.ravager_sneak_stamina if is_sneak else 0.5)
            if actor.dragged_carcass_id is not None:
                cost += self.config.ravager_drag_stamina
            if (actor.stamina < cost or
                    (is_dash and actor.species != Species.NOMAD)):
                continue
            distance = 2 if is_dash else 1
            dest = (actor.x + distance * action.dx, actor.y + distance * action.dy)
            midpoint = (actor.x + action.dx, actor.y + action.dy)
            if is_dash and (not self.grid.contains(*midpoint)
                            or not self.grid.get(*midpoint).passable
                            or midpoint in occupied
                            or any(s.x == midpoint[0] and s.y == midpoint[1]
                                   for s in self.settlements)):
                continue
            if (not self.grid.contains(*dest) or not self.grid.get(*dest).passable
                    or any((s.x, s.y) == dest for s in self.settlements)):
                continue
            requests[dest].append(aid)
        winners: dict[int, tuple[int, int]] = {}
        for dest in sorted(requests):
            candidates = sorted(requests[dest])
            winner = candidates[self.rng.randrange(len(candidates))]
            winners[winner] = dest
        changed = True
        while changed:
            changed = False
            for aid, dest in list(winners.items()):
                blocker = occupied.get(dest)
                if blocker is not None and blocker not in winners:
                    del winners[aid]
                    changed = True
        for aid, (x, y) in sorted(winners.items()):
            actor = self.actors[aid]
            was_sheltered = actor.sheltered
            actor.x, actor.y = x, y
            actor.sheltered = False
            if was_sheltered:
                self.emit("exit_refuge", aid, detail=str(actor.home_id))
            is_dash = orders[aid].kind == Kind.DASH
            is_sneak = orders[aid].kind == Kind.SNEAK
            move_cost = (self.config.nomad_dash_stamina if is_dash else
                         self.config.ravager_sneak_stamina if is_sneak else 0.5)
            actor.stamina = max(0.0, actor.stamina - move_cost)
            actor.energy = max(0.0, actor.energy - (
                0.35 if is_dash else self.config.ravager_sneak_energy if is_sneak else 0.07))
            if is_sneak:
                actor.sneaking = True
                actor.sneak_cooldown = 2  # Applied before end-of-tick decrement.
            if actor.dragged_carcass_id is not None:
                carcass = self.carcasses.get(actor.dragged_carcass_id)
                if carcass is not None:
                    carcass.x, carcass.y = x, y
                    actor.drag_cooldown = 2  # Half-speed while carrying.
                    actor.stamina -= self.config.ravager_drag_stamina
                    actor.energy = max(0, actor.energy - 0.15)
                    self.emit("drag_carcass", aid, detail=str(carcass.id))
            self.emit("dash" if is_dash else "sneak" if is_sneak else "move",
                      aid, detail=f"{x},{y}")

    def _resolve_charges(self, pending: list[int]) -> None:
        """Resolve last tick's committed charge AFTER other actors move to dodge.

        The charge travels in a straight line. It stops at the first actor or
        blocking tile, causing one impact at most, never overlapping an actor.
        """
        for aid in pending:
            attacker = self.actors.get(aid)
            if attacker is None or attacker.charge_windup == 0:
                continue
            dx, dy = attacker.charge_dx, attacker.charge_dy
            attacker.charge_dx = attacker.charge_dy = attacker.charge_windup = 0
            attacker.charge_recovery = self.config.ravager_charge_recovery + 1
            attacker.sneaking = False
            result = "miss"
            for _ in range(self.config.ravager_charge_distance):
                x, y = attacker.x + dx, attacker.y + dy
                if (not self.grid.contains(x, y) or not self.grid.get(x, y).passable
                        or any((s.x, s.y) == (x, y) for s in self.settlements)):
                    attacker.health = max(0.0, attacker.health - 3.0)
                    result = "obstacle"
                    break
                victim = next((a for a in self.actors.values()
                               if a.id != aid and not a.sheltered and (a.x, a.y) == (x, y)),
                              None)
                if victim is not None:
                    if victim.species != Species.RAVAGER:
                        damage = self.config.ravager_charge_damage
                        victim.health = max(0.0, victim.health - damage)
                        self.emit("charge_hit", aid, damage, f"target:{victim.id}")
                        result = "hit"
                    else:
                        result = "blocked"
                    break
                attacker.x, attacker.y = x, y
            if attacker.dragged_carcass_id is not None:
                # A charge cannot be initiated while carrying, but this is a
                # defensive consistency guard for traces and future mechanics.
                self._drop_carcass(attacker)
            self.emit("charge_resolved", aid, detail=result)

    def _drop_carcass(self, actor: Actor) -> None:
        if actor.dragged_carcass_id is None:
            return
        carcass = self.carcasses.get(actor.dragged_carcass_id)
        if carcass is not None:
            carcass.carrier_id = None
            carcass.x, carcass.y = actor.position
            self.emit("drop_carcass", actor.id, detail=str(carcass.id))
        actor.dragged_carcass_id = None

    def home(self, actor: Actor) -> Settlement | None:
        return next((s for s in self.settlements if s.id == actor.home_id), None)

    def refuge_occupants(self, settlement_id: int) -> int:
        return sum(a.sheltered and a.home_id == settlement_id for a in self.actors.values())

    def _enter_refuges(self, orders: dict[int, Action]) -> None:
        """Admit only eligible adjacent home farmers into finite interior capacity.

        Simultaneous entry requests are resolved by a seeded lottery per shelter.
        Nobody is teleported from outside the entrance's immediate neighborhood.
        """
        for home in self.settlements:
            candidates = sorted(a.id for a in self.actors.values()
                                if a.species == Species.FARMER and a.home_id == home.id
                                and not a.sheltered and a.stamina >= 1
                                and orders.get(a.id, Action()).kind == Kind.ENTER_REFUGE
                                and abs(a.x - home.x) + abs(a.y - home.y) <= 1)
            vacancies = max(0, home.refuge_capacity - self.refuge_occupants(home.id))
            if not vacancies or not candidates:
                continue
            self.rng.shuffle(candidates)
            for aid in candidates[:vacancies]:
                actor = self.actors[aid]
                actor.sheltered = True
                actor.x, actor.y = home.x, home.y
                actor.stamina = max(0.0, actor.stamina - 1)
                self.emit("enter_refuge", aid, detail=str(home.id))

    def _nearby_settlement(self, actor: Actor) -> Settlement | None:
        return next((s for s in self.settlements
                     if (actor.home_id is None or actor.home_id == s.id)
                     and abs(actor.x - s.x) + abs(actor.y - s.y) <= 1), None)

    def _festival_proposals(self, orders: dict[int, Action]) -> None:
        """Validate simultaneous proposals BEFORE any same-tick contributions.

        A seeded lottery breaks ties at a physical site instead of favoring
        low actor IDs. Validation considers positions after simultaneous moves.
        """
        if not self.config.festival_enabled:
            return
        modes = {Kind.START_FARMER_FESTIVAL: FestivalKind.FARMER,
                 Kind.START_NOMAD_FESTIVAL: FestivalKind.NOMAD,
                 Kind.START_COMBINED_FESTIVAL: FestivalKind.COMBINED}
        requests: dict[tuple[str, int], list[tuple[int, FestivalKind]]] = defaultdict(list)
        for aid in sorted(orders):
            actor = self.actors[aid]
            if is_juvenile(self, actor):
                continue
            action = orders[aid]
            mode = modes.get(action.kind)
            if mode is None or actor.hibernating or actor.charge_windup or actor.charge_recovery:
                continue
            if not can_start_festival(self, actor, mode):
                continue
            site = festival_site(self, actor, mode)
            assert site is not None
            key = ("camp" if mode == FestivalKind.NOMAD else "settlement", site.id)
            requests[key].append((aid, mode))
        for key in sorted(requests):
            candidates = sorted(requests[key], key=lambda item: (item[0], item[1].value))
            aid, mode = candidates[self.rng.randrange(len(candidates))]
            start_festival(self, self.actors[aid], mode)

    def _interact(self, actor: Actor, action: Action) -> None:
        tile = self.grid.get(actor.x, actor.y)
        kind = action.kind
        if kind in (Kind.WAIT, Kind.MOVE, Kind.DASH, Kind.SNEAK, Kind.ENTER_REFUGE):
            return
        if actor.hibernating:
            if kind == Kind.WAKE:
                actor.hibernating = False
                actor.energy = max(0.0, actor.energy - 0.3)
                self.emit("wake", actor.id)
            return
        if actor.charge_windup or actor.charge_recovery:
            return
        # Juveniles can eat, graze, travel and hide, but have not developed
        # milking/attack/charge/call skills. Keep old action schemas stable.
        if is_juvenile(self, actor) and kind in (
                Kind.MILK, Kind.ATTACK, Kind.CHARGE, Kind.FRIENDLY_CALL,
                Kind.SCARY_CALL, Kind.START_FARMER_FESTIVAL,
                Kind.START_NOMAD_FESTIVAL, Kind.START_COMBINED_FESTIVAL,
                Kind.CONTRIBUTE_FESTIVAL):
            return
        # Interior shelter excludes all field work, festival participation and attacks.
        if actor.sheltered and kind not in (Kind.REST, Kind.EAT, Kind.DEPOSIT):
            return
        if kind == Kind.CONTRIBUTE_FESTIVAL:
            site = contribution_target(self, actor)
            if site is not None and actor.carried_food > 0:
                amount = min(2.0, actor.carried_food)
                actor.carried_food -= amount
                if actor.species == Species.FARMER:
                    site.festival_food += amount
                elif actor.species == Species.NOMAD:
                    site.festival_milk += amount
                self.emit("festival_contribution", actor.id, amount,
                          f"{actor.species.value}:{site.id}")
            return
        festival_actions = {
            Kind.START_FARMER_FESTIVAL: FestivalKind.FARMER,
            Kind.START_NOMAD_FESTIVAL: FestivalKind.NOMAD,
            Kind.START_COMBINED_FESTIVAL: FestivalKind.COMBINED,
        }
        if kind in festival_actions:
            # All start proposals resolve in a separate simultaneous stage;
            # no participant can spend food contributed later this same tick.
            return
        if actor.species == Species.RAVAGER:
            if kind == Kind.CHARGE:
                if (actor.charge_cooldown == 0 and actor.dragged_carcass_id is None
                        and actor.stamina >= self.config.ravager_charge_stamina
                        and actor.energy >= self.config.ravager_charge_energy):
                    actor.stamina -= self.config.ravager_charge_stamina
                    actor.energy -= self.config.ravager_charge_energy
                    actor.charge_cooldown = self.config.ravager_charge_cooldown + 1
                    actor.charge_windup = 1
                    actor.charge_dx, actor.charge_dy = action.dx, action.dy
                    actor.sneaking = False
                    self.emit("charge_windup", actor.id, detail=f"{action.dx},{action.dy}")
                return
            if kind == Kind.GRAB_CARCASS and actor.dragged_carcass_id is None:
                carcasses = sorted((c for c in self.carcasses.values()
                                    if c.carrier_id is None
                                    and abs(c.x - actor.x) + abs(c.y - actor.y) <= 1),
                                   key=lambda c: (abs(c.x - actor.x) + abs(c.y - actor.y), c.id))
                if carcasses:
                    carcass = carcasses[0]
                    actor.dragged_carcass_id = carcass.id
                    carcass.carrier_id = actor.id
                    carcass.x, carcass.y = actor.position
                    self.emit("grab_carcass", actor.id, detail=str(carcass.id))
                return
            if kind == Kind.DROP_CARCASS:
                self._drop_carcass(actor)
                return
            if kind == Kind.HIBERNATE:
                if actor.satiety >= 0.4 and actor.energy >= 35:
                    actor.hibernating = True
                    self.emit("hibernate", actor.id)
                return
        if kind == Kind.REST:
            actor.stamina = min(100.0, actor.stamina + 3.0)
        elif kind == Kind.GRAZE and actor.species == Species.COW:
            # Cows can graze agricultural crops and thus harm farmers.
            source = "grass" if tile.grass > 0.01 else "crop"
            amount = min(2.0, tile.grass if source == "grass" else tile.crop)
            if amount > 0:
                if source == "grass":
                    tile.grass -= amount
                else:
                    tile.crop -= amount
                    if tile.crop <= 1e-8:
                        tile.crop = 0.0
                        tile.planted = False
                actor.energy = min(100.0, actor.energy + amount * 2.4)
                actor.dung_pending += amount * 0.3
                # Juvenile cattle eat and grow but do not yet produce milk.
                if not is_juvenile(self, actor):
                    actor.milk = min(10.0, actor.milk + amount * 0.16)
                self.emit("graze", actor.id, amount, source)
        elif kind == Kind.PLANT and actor.species == Species.FARMER:
            if not tile.planted and tile.fertility >= 0.2 and actor.carried_food >= 0.3:
                actor.carried_food -= 0.3  # seed grain is not free
                tile.planted, tile.crop = True, 0.15
                self.emit("plant", actor.id, 0.3, "seed_cost")
        elif kind == Kind.HARVEST and actor.species == Species.FARMER:
            if tile.planted and tile.crop >= 1:
                amount = min(2.5, tile.crop)
                tile.crop -= amount
                if tile.crop <= 1e-8:
                    tile.crop, tile.planted = 0.0, False
                actor.carried_food += amount
                self.emit("harvest", actor.id, amount)
        elif kind == Kind.COLLECT_WASTE and actor.species == Species.FARMER:
            amount = min(1.5, tile.waste)
            tile.waste -= amount
            actor.carried_waste += amount
            if amount:
                self.emit("collect_waste", actor.id, amount)
        elif kind == Kind.FERTILIZE and actor.species == Species.FARMER:
            amount = min(1.0, actor.carried_waste)
            useful = min(amount, (1.0 - tile.fertility) / 0.18)
            if useful > 0:
                actor.carried_waste -= useful
                tile.fertility = min(1.0, tile.fertility + useful * 0.18)
                self.emit("fertilize", actor.id, useful)
        elif kind in (Kind.FRIENDLY_CALL, Kind.SCARY_CALL) and actor.species == Species.NOMAD:
            friendly = kind == Kind.FRIENDLY_CALL
            cooldown = actor.friendly_cooldown if friendly else actor.scary_cooldown
            cost = (self.config.friendly_call_stamina if friendly
                    else self.config.scary_call_stamina)
            if cooldown == 0 and actor.stamina >= cost:
                self.calls.append(CallSignal(actor.id, kind.value, actor.x, actor.y,
                                             self.tick, self.config.call_radius))
                if friendly:
                    actor.friendly_cooldown = self.config.call_cooldown
                else:
                    actor.scary_cooldown = self.config.call_cooldown
                actor.stamina -= cost
                actor.energy = max(0.0, actor.energy - (0.25 if friendly else 0.4))
                self.emit(kind.value, actor.id, detail=f"radius:{self.config.call_radius}")
        elif kind == Kind.ATTACK and actor.species == Species.NOMAD:
            if actor.stamina >= self.config.nomad_attack_stamina:
                targets = sorted((other for other in self.actors.values()
                                  if other.species == Species.RAVAGER
                                  and abs(other.x - actor.x) + abs(other.y - actor.y) <= 1),
                                 key=lambda enemy: (enemy.health, enemy.id))
                if targets:
                    target = targets[0]
                    target.health = max(0.0, target.health - self.config.nomad_attack_damage)
                    actor.stamina -= self.config.nomad_attack_stamina
                    actor.energy = max(0.0, actor.energy - 0.4)
                    self.emit("nomad_attack", actor.id, self.config.nomad_attack_damage,
                              f"target:{target.id}")
        elif kind == Kind.MILK and actor.species == Species.NOMAD:
            cows = sorted((other for other in self.actors.values()
                           if other.species == Species.COW
                           and abs(actor.x - other.x) + abs(actor.y - other.y) <= 1
                           and other.milk > 0), key=lambda c: c.id)
            if cows:
                amount = min(1.0, cows[0].milk)
                cows[0].milk -= amount
                actor.carried_food += amount
                self.emit("milk", actor.id, amount, f"cow:{cows[0].id}")
        elif kind == Kind.EAT:
            if actor.species in (Species.FARMER, Species.NOMAD):
                store = self._nearby_settlement(actor) if actor.species == Species.FARMER else None
                food = actor.carried_food
                amount = min(1.0, food) if food > 0 else min(1.0, store.stored_food) if store else 0
                if amount > 0:
                    if food > 0:
                        actor.carried_food -= amount
                    elif store:
                        store.stored_food -= amount
                    actor.energy = min(100.0, actor.energy + amount * 6)
                    self.emit("eat", actor.id, amount, "carried" if food > 0 else "store")
            elif actor.species == Species.SCAVENGER:
                amount = min(1.0, tile.waste)
                if amount > 0:
                    tile.waste -= amount
                    actor.energy = min(100.0, actor.energy + amount * 2.2)
                    # Digestion recycles a *fraction* of consumed organic matter.
                    # The residue is not edible until a later ecology tick.
                    actor.dung_pending += amount * self.config.scavenger_residue_fraction
                    self.emit("eat_waste", actor.id, amount)
        elif kind == Kind.EAT_CARCASS and actor.species in (Species.RAVAGER, Species.SCAVENGER):
            carcasses = sorted((c for c in self.carcasses.values()
                                if c.x == actor.x and c.y == actor.y
                                and c.carrier_id in (None, actor.id)), key=lambda c: c.id)
            if carcasses:
                carcass = carcasses[0]
                amount = min(2.0, carcass.meat)
                carcass.meat -= amount
                actor.energy = min(100.0, actor.energy + amount * 4)
                if actor.species == Species.RAVAGER:
                    actor.satiety = min(1.0, actor.satiety + amount / 8.0)
                else:
                    actor.dung_pending += amount * self.config.scavenger_residue_fraction
                self.emit("eat_carcass", actor.id, amount, str(carcass.id))
                if actor.species == Species.SCAVENGER:
                    self.emit("scavenger_carrion", actor.id, amount, str(carcass.id))
                if carcass.meat <= 1e-8:
                    if carcass.carrier_id is not None:
                        self.actors[carcass.carrier_id].dragged_carcass_id = None
                    del self.carcasses[carcass.id]
        elif kind == Kind.ATTACK and actor.species == Species.RAVAGER:
            if actor.stamina >= 4.0:
                victims = sorted((a for a in self.actors.values()
                                  if a.id != actor.id and a.species != Species.RAVAGER
                                  and not a.sheltered
                                  and abs(a.x - actor.x) + abs(a.y - actor.y) <= 1),
                                 key=lambda a: (a.health, a.id))
                if victims:
                    target = victims[0]
                    damage = 12.0
                    target.health = max(0.0, target.health - damage)
                    actor.stamina = max(0.0, actor.stamina - 4.0)
                    actor.energy = max(0.0, actor.energy - 0.5)
                    self.emit("attack", actor.id, damage, f"target:{target.id}")
        elif kind == Kind.DEPOSIT and actor.species == Species.FARMER:
            store = self._nearby_settlement(actor)
            if store is not None:
                amount = min(3.0, actor.carried_food)
                if amount:
                    actor.carried_food -= amount
                    store.stored_food += amount
                    self.emit("deposit", actor.id, amount, str(store.id))

    def _ecology(self) -> None:
        cfg = self.config
        for tile in self.grid.tiles:
            tile.grass = min(cfg.max_grass, tile.grass + cfg.grass_growth * tile.fertility)
            if tile.planted:
                growth = min(cfg.max_crop - tile.crop, cfg.crop_growth * tile.fertility)
                if growth > 0:
                    tile.crop += growth
                    tile.fertility = max(0.05, tile.fertility - growth * 0.014)
            decayed = min(tile.waste, cfg.waste_decay)
            tile.waste -= decayed
            tile.fertility = min(1.0, tile.fertility + decayed * 0.015)
        for actor in self.actors.values():
            # Scavenger residue is real digested material, NOT fresh food.
            if (actor.species == Species.COW and actor.dung_pending >= 0.25
                    or actor.species == Species.SCAVENGER and actor.dung_pending >= 0.02):
                tile = self.grid.get(actor.x, actor.y)
                amount = min(actor.dung_pending, max(0.0, cfg.max_waste - tile.waste))
                tile.waste += amount
                actor.dung_pending -= amount
                if amount:
                    self.emit("dung" if actor.species == Species.COW
                              else "scavenger_residue", actor.id, amount)
        for carcass in list(self.carcasses.values()):
            decay = min(carcass.meat, 0.04)
            carcass.meat -= decay
            # Decayed carrion returns nutrients locally; it is not new biomass.
            # Any fertility above capacity is lost, never duplicated as waste.
            tile = self.grid.get(carcass.x, carcass.y)
            tile.fertility = min(1.0, tile.fertility + decay * cfg.carcass_fertility_return)
            if decay:
                self.emit("carcass_decay", amount=decay,
                          detail=f"carcass:{carcass.id}")
            if carcass.meat <= 1e-8:
                if carcass.carrier_id is not None and carcass.carrier_id in self.actors:
                    self.actors[carcass.carrier_id].dragged_carcass_id = None
                del self.carcasses[carcass.id]

    def _metabolism(self) -> None:
        # Occupied shelters consume REAL stored food, which is distributed
        # fairly among residents. No ration appears from thin air, and actors
        # cannot gain energy beyond their maximum from shelter food.
        for home in self.settlements:
            residents = [a for a in self.actors.values()
                         if a.sheltered and a.home_id == home.id]
            if residents and home.stored_food > 0:
                quota = min(self.config.refuge_ration,
                            home.stored_food / len(residents))
                portions = [(a, min(quota, (100.0 - a.energy) / 6.0))
                            for a in residents]
                for actor, amount in portions:
                    if amount > 0:
                        actor.energy = min(100.0, actor.energy + amount * 6.0)
                        home.stored_food -= amount
                        self.emit("refuge_ration", actor.id, amount, str(home.id))
                home.stored_food = max(0.0, home.stored_food)
        costs = {Species.FARMER: 0.035, Species.COW: self.config.cow_metabolism, Species.NOMAD: 0.05,
                 Species.SCAVENGER: 0.035, Species.RAVAGER: 0.07}
        for aid, actor in list(sorted(self.actors.items())):
            actor.friendly_cooldown = max(0, actor.friendly_cooldown - 1)
            actor.scary_cooldown = max(0, actor.scary_cooldown - 1)
            actor.sneak_cooldown = max(0, actor.sneak_cooldown - 1)
            actor.drag_cooldown = max(0, actor.drag_cooldown - 1)
            actor.charge_cooldown = max(0, actor.charge_cooldown - 1)
            actor.charge_recovery = max(0, actor.charge_recovery - 1)
            actor.satiety = max(0.0, actor.satiety - 0.005)
            actor.age += 1
            if ((self.config.population_dynamics_enabled or self.config.reproduction_enabled)
                    and actor.age > LIFESPAN[actor.species]):
                actor.health = max(0.0, actor.health - 0.65)
            metabolic = (self.config.ravager_hibernate_metabolism
                         if actor.hibernating and actor.species == Species.RAVAGER
                         else costs[actor.species])
            actor.energy = max(0.0, actor.energy - metabolic)
            actor.stamina = min(100.0, actor.stamina + 0.3)
            if actor.energy <= 0:
                actor.health = max(0.0, actor.health - 0.7)
            if actor.health <= 0:
                self._drop_carcass(actor)
                meat = {Species.COW: 12.0, Species.RAVAGER: 10.0,
                        Species.NOMAD: 7.0, Species.FARMER: 6.0,
                        Species.SCAVENGER: 3.0}[actor.species]
                carcass = Carcass(self.next_carcass_id, actor.x, actor.y, meat, actor.species)
                self.carcasses[carcass.id] = carcass
                self.next_carcass_id += 1
                self.emit("death", aid, detail=actor.species.value)
                if self.config.population_dynamics_enabled or self.config.reproduction_enabled:
                    attacked = any(e.kind in ("attack", "charge_hit") and
                                   e.detail == f"target:{aid}" for e in self.last_events)
                    cause = "predation" if attacked else (
                        "starvation" if actor.energy <= 0 else
                        "age" if actor.age > LIFESPAN[actor.species] else "other")
                    self.emit("death_cause", aid, detail=cause)
                del self.actors[aid]

    def step(self, orders: dict[int, Action] | None = None) -> None:
        """Advance exactly one tick from a simultaneous action snapshot."""
        orders = orders or {}
        if unknown := set(orders) - set(self.actors):
            raise ValueError(f"Actions for nonexistent actors: {sorted(unknown)}")
        self.tick += 1
        self.last_events = []
        # Existing signals were visible to policies at the preceding tick.
        self.calls = [signal for signal in self.calls
                      if self.tick - signal.tick <= self.config.call_duration]
        pending = [a.id for a in self.actors.values() if a.charge_windup]
        for actor in self.actors.values():
            # A stationary/running ravager is no longer visually sneaking.
            actor.sneaking = actor.sneaking and orders.get(actor.id, Action()).kind in (
                Kind.SNEAK, Kind.WAIT)
        self._movement(orders)
        self._enter_refuges(orders)
        self._resolve_charges(pending)
        self._festival_proposals(orders)
        # Stable ID order matters for deterministic shared-resource resolution.
        for aid in sorted(self.actors):
            if aid in pending:
                continue  # Charge impact consumes this tick's action.
            self._interact(self.actors[aid], orders.get(aid, Action()))
        if self.tick % self.config.ecology_interval == 0:
            self._ecology()
        self._metabolism()
        if self.config.festival_enabled:
            advance_festivals(self)
        if self.config.reproduction_enabled and self.tick % self.config.reproduction_interval == 0:
            birth_step(self)
        self.check_invariants()

    def check_invariants(self) -> None:
        cfg = self.config
        outside = [a.position for a in self.actors.values() if not a.sheltered]
        assert len(set(outside)) == len(outside)
        if cfg.reproduction_enabled:
            caps = population_caps(cfg)
            counts = Counter(a.species for a in self.actors.values())
            assert all(counts[species] <= caps[species] for species in Species)
            assert all(0 <= self.birth_totals[species] <= cfg.births_per_species_limit
                       for species in Species)
            assert self.next_id > max(self.actors, default=0)
        for actor in self.actors.values():
            assert actor.age >= 0 and actor.last_birth_tick <= self.tick
            if actor.parent_id is not None:
                assert actor.parent_id < actor.id
            if actor.sheltered:
                home = self.home(actor)
                assert actor.species == Species.FARMER and home is not None
                assert actor.position == (home.x, home.y)
            else:
                assert all(actor.position != (s.x, s.y) for s in self.settlements)
        assert all(self.refuge_occupants(s.id) <= s.refuge_capacity
                   for s in self.settlements)
        for a in self.actors.values():
            assert self.grid.contains(a.x, a.y)
            assert 0 <= a.health <= 100 and 0 <= a.energy <= 100
            assert 0 <= a.stamina <= 100 and 0 <= a.milk <= 10.000001
            assert a.carried_food >= -1e-8 and a.carried_waste >= -1e-8
            assert a.dung_pending >= -1e-8
            assert a.friendly_cooldown >= 0 and a.scary_cooldown >= 0
            assert min(a.sneak_cooldown, a.drag_cooldown, a.charge_cooldown,
                       a.charge_recovery, a.charge_windup) >= 0
            assert 0 <= a.satiety <= 1
            if a.dragged_carcass_id is not None:
                carcass = self.carcasses[a.dragged_carcass_id]
                assert carcass.carrier_id == a.id and carcass.x == a.x and carcass.y == a.y
            if a.hibernating:
                assert a.species == Species.RAVAGER
        for tile in self.grid.tiles:
            assert 0 <= tile.fertility <= 1 and 0 <= tile.grass <= cfg.max_grass
            assert 0 <= tile.crop <= cfg.max_crop and 0 <= tile.waste <= cfg.max_waste + 1e-8
            assert tile.planted or tile.crop == 0
        for carcass in self.carcasses.values():
            assert carcass.meat > 0 and self.grid.contains(carcass.x, carcass.y)
            if carcass.carrier_id is not None:
                holder = self.actors[carcass.carrier_id]
                assert holder.dragged_carcass_id == carcass.id
        assert all(s.stored_food >= 0 and s.festival_food >= -1e-8 and
                   s.festival_milk >= -1e-8 and s.cooldown_until >= 0
                   for s in self.settlements)
        assert all(c.festival_milk >= -1e-8 and c.cooldown_until >= 0
                   for c in self.camps)
        assert all(a.band_id is not None for a in self.actors.values()
                   if a.species == Species.NOMAD)
        for festival in self.active_festivals.values():
            assert festival.end_tick > festival.start_tick
            assert festival.kind in FestivalKind
            assert self.grid.contains(festival.x, festival.y)
            assert festival.grain_spent >= 0 and festival.milk_spent >= 0
        assert all(sig.kind in (Kind.FRIENDLY_CALL.value, Kind.SCARY_CALL.value)
                   and sig.radius >= 1 and sig.tick <= self.tick for sig in self.calls)

    def snapshot(self) -> dict:
        # Legacy snapshots deliberately retain their M8 shape when opt-in
        # reproduction is disabled, so previously saved action traces replay.
        config = self.config.as_dict()
        actors = [asdict(self.actors[aid]) for aid in sorted(self.actors)]
        if not (self.config.population_dynamics_enabled or self.config.reproduction_enabled):
            from .population import REPRO_CONFIG_FIELDS
            for field in REPRO_CONFIG_FIELDS:
                config.pop(field)
            for actor in actors:
                actor.pop("parent_id")
                actor.pop("last_birth_tick")
        result = {
            "tick": self.tick,
            "config": config,
            "tiles": [asdict(t) for t in self.grid.tiles],
            "actors": actors,
            "settlements": [asdict(s) for s in self.settlements],
            "carcasses": [asdict(self.carcasses[cid]) for cid in sorted(self.carcasses)],
            "calls": [asdict(signal) for signal in self.calls],
            "camps": [asdict(camp) for camp in self.camps],
            "active_festivals": [asdict(self.active_festivals[fid])
                                 for fid in sorted(self.active_festivals)],
            "festival_results": [asdict(result) for result in self.festival_results],
            "next_festival_id": self.next_festival_id,
            "herd_goal": self.herd_goal,
        }
        if self.config.population_dynamics_enabled or self.config.reproduction_enabled:
            result["next_id"] = self.next_id
            result["birth_totals"] = {species.value: self.birth_totals[species]
                                      for species in Species}
        return result

    def state_hash(self) -> str:
        payload = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        counts = Counter(a.species.value for a in self.actors.values())
        kinds = Counter(e.kind for e in self.events)
        return {
            "tick": self.tick,
            "seed": self.config.seed,
            "state_hash": self.state_hash(),
            "population": {s.value: counts[s.value] for s in Species},
            "births": {s.value: self.birth_totals[s] for s in Species},
            "juveniles": {s.value: sum(a.species == s and is_juvenile(self, a)
                                       for a in self.actors.values()) for s in Species},
            "deaths": {s.value: sum(e.kind == "death" and e.detail == s.value
                                    for e in self.events) for s in Species},
            "grass": round(sum(t.grass for t in self.grid.tiles), 3),
            "crops": round(sum(t.crop for t in self.grid.tiles), 3),
            "waste": round(sum(t.waste for t in self.grid.tiles), 3),
            "stored_food": round(sum(s.stored_food for s in self.settlements), 3),
            "festival_grain_escrow": round(sum(s.festival_food for s in self.settlements), 3),
            "festival_milk_escrow": round(sum(s.festival_milk for s in self.settlements) +
                                          sum(c.festival_milk for c in self.camps), 3),
            "active_festivals": len(self.active_festivals),
            "festival_successes": sum(result.succeeded for result in self.festival_results),
            "festival_failures": sum(not result.succeeded for result in self.festival_results),
            "sheltered_farmers": sum(a.sheltered for a in self.actors.values()),
            "carcass_meat": round(sum(c.meat for c in self.carcasses.values()), 3),
            "events": dict(sorted(kinds.items())),
            "active_calls": len(self.calls),
            "hibernating_ravagers": sum(a.hibernating for a in self.actors.values()),
            "dragged_carcasses": sum(c.carrier_id is not None for c in self.carcasses.values()),
            "scavenger_waste_eaten": round(sum(e.amount for e in self.events
                                                 if e.kind == "eat_waste"), 3),
            "scavenger_carrion_eaten": round(sum(e.amount for e in self.events
                                                   if e.kind == "scavenger_carrion"), 3),
            "manure_collected": round(sum(e.amount for e in self.events
                                           if e.kind == "collect_waste"), 3),
            "scavenger_residue": round(sum(e.amount for e in self.events
                                            if e.kind == "scavenger_residue"), 3),
            "carcass_mass_decayed": round(sum(e.amount for e in self.events
                                               if e.kind == "carcass_decay"), 3),
        }
