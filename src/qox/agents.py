"""Simple transparent baseline actors used before neural-network training."""

from __future__ import annotations

from .actions import Action, DIRECTIONS, Kind
from .entities import Actor, Species
from .world import World


def random_actions(world: World) -> dict[int, Action]:
    """Seeded, intentionally weak legal-ish random baseline."""
    possible = {
        Species.FARMER: (Kind.PLANT, Kind.HARVEST, Kind.COLLECT_WASTE,
                         Kind.FERTILIZE, Kind.DEPOSIT, Kind.EAT),
        Species.COW: (Kind.GRAZE,),
        Species.NOMAD: (Kind.MILK, Kind.EAT),
        Species.SCAVENGER: (Kind.EAT, Kind.EAT_CARCASS),
        Species.RAVAGER: (Kind.EAT_CARCASS,),
    }
    orders = {}
    for aid in sorted(world.actors):
        actor = world.actors[aid]
        choices = (Kind.WAIT, Kind.REST, Kind.MOVE, Kind.MOVE, Kind.MOVE,
                   *possible[actor.species])
        kind = world.policy_rng.choice(choices)
        orders[aid] = Action(kind, *world.policy_rng.choice(DIRECTIONS)) if kind == Kind.MOVE else Action(kind)
    return orders


def _toward(world: World, actor: Actor, target: tuple[int, int]) -> Action:
    occupied = {a.position for a in world.actors.values()}
    possibilities = []
    for dx, dy in DIRECTIONS:
        x, y = actor.x + dx, actor.y + dy
        if (world.grid.contains(x, y) and (x, y) not in occupied
                and all((s.x, s.y) != (x, y) for s in world.settlements)):
            possibilities.append((abs(x - target[0]) + abs(y - target[1]), dx, dy))
    if not possibilities:
        return Action(Kind.REST)
    best = min(p[0] for p in possibilities)
    options = sorted((dx, dy) for dist, dx, dy in possibilities if dist == best)
    return Action(Kind.MOVE, *world.policy_rng.choice(options))


def _nearest(world: World, actor: Actor, predicate, radius: int = 5):
    """Script uses local tiles only; the actual learned-observation API comes in M2."""
    choices = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = actor.x + dx, actor.y + dy
            if world.grid.contains(x, y) and abs(dx) + abs(dy) <= radius:
                tile = world.grid.get(x, y)
                if predicate(tile):
                    choices.append((abs(dx) + abs(dy), x, y))
    return min(choices)[1:] if choices else None


def scripted_actions(world: World) -> dict[int, Action]:
    """Non-learning, mostly reactive actors for ecological sanity checks."""
    orders: dict[int, Action] = {}
    for aid in sorted(world.actors):
        actor = world.actors[aid]
        tile = world.grid.get(actor.x, actor.y)
        if actor.species == Species.COW:
            if tile.grass > 0.6 or tile.crop > 0.6:
                action = Action(Kind.GRAZE)
            else:
                target = _nearest(world, actor, lambda t: t.grass > 1 or t.crop > 1)
                action = _toward(world, actor, target) if target else Action(Kind.REST)
        elif actor.species == Species.FARMER:
            home = next(s for s in world.settlements if s.id == actor.home_id)
            if actor.energy < 45 and (actor.carried_food >= 1 or
                                       world._nearby_settlement(actor)):
                action = Action(Kind.EAT)
            elif actor.carried_food > 4 and world._nearby_settlement(actor):
                action = Action(Kind.DEPOSIT)
            elif actor.carried_food > 4:
                action = _toward(world, actor, (home.x, home.y))
            elif tile.planted and tile.crop >= 1.5:
                action = Action(Kind.HARVEST)
            elif actor.carried_waste >= 1 and tile.fertility < 0.75:
                action = Action(Kind.FERTILIZE)
            elif tile.waste >= 0.6 and actor.carried_waste < 2.5:
                action = Action(Kind.COLLECT_WASTE)
            elif not tile.planted and actor.carried_food >= 0.5:
                action = Action(Kind.PLANT)
            else:
                target = _nearest(world, actor, lambda t: t.crop >= 1.5 or t.waste >= 0.6)
                action = _toward(world, actor, target) if target else Action(Kind.REST)
        elif actor.species == Species.NOMAD:
            near_milk = any(c.species == Species.COW and c.milk > 0.4 and
                            abs(c.x - actor.x) + abs(c.y - actor.y) <= 1
                            for c in world.actors.values())
            if actor.energy < 60 and actor.carried_food >= 0.7:
                action = Action(Kind.EAT)
            elif near_milk:
                action = Action(Kind.MILK)
            else:
                cows = sorted((c for c in world.actors.values() if c.species == Species.COW),
                              key=lambda c: (abs(c.x - actor.x) + abs(c.y - actor.y), c.id))
                action = _toward(world, actor, cows[0].position) if cows else Action(Kind.REST)
        elif actor.species == Species.SCAVENGER:
            has_carcass = any(c.x == actor.x and c.y == actor.y
                              for c in world.carcasses.values())
            if has_carcass:
                action = Action(Kind.EAT_CARCASS)
            elif tile.waste >= 0.4:
                action = Action(Kind.EAT)
            else:
                target = _nearest(world, actor, lambda t: t.waste >= 0.4)
                action = _toward(world, actor, target) if target else Action(Kind.REST)
        else:
            # Minimal contact-bite predator for M3 threat scenarios. The full
            # sneak/charge/drag/hibernate behavior is reserved for M6.
            prey = sorted((a for a in world.actors.values()
                           if a.id != actor.id and a.species != Species.RAVAGER
                           and abs(a.x - actor.x) + abs(a.y - actor.y) <= 6),
                          key=lambda a: (abs(a.x - actor.x) + abs(a.y - actor.y), a.id))
            nearby = [a for a in prey if abs(a.x - actor.x) + abs(a.y - actor.y) <= 1]
            has_carcass = any(c.x == actor.x and c.y == actor.y
                              for c in world.carcasses.values())
            if has_carcass:
                action = Action(Kind.EAT_CARCASS)
            elif nearby and actor.stamina >= 4.0:
                action = Action(Kind.ATTACK)
            elif prey:
                action = _toward(world, actor, prey[0].position)
            else:
                carcasses = sorted(world.carcasses.values(),
                                   key=lambda c: (abs(c.x - actor.x) + abs(c.y - actor.y), c.id))
                action = _toward(world, actor, (carcasses[0].x, carcasses[0].y)) if carcasses else Action(Kind.REST)
        orders[aid] = action
    return orders


def actions_for(world: World, mode: str) -> dict[int, Action]:
    if mode == "scripted":
        return scripted_actions(world)
    if mode == "random":
        return random_actions(world)
    if mode == "idle":
        return {aid: Action(Kind.WAIT) for aid in world.actors}
    raise ValueError(f"Unknown actor mode: {mode}")
