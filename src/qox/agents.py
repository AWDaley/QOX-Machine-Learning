"""Simple transparent baseline actors used before neural-network training."""

from __future__ import annotations

from .actions import Action, DIRECTIONS, Kind
from .entities import Actor, Species
from .world import World


def random_actions(world: World) -> dict[int, Action]:
    """Seeded, intentionally weak legal-ish random baseline."""
    possible = {
        Species.FARMER: (Kind.PLANT, Kind.HARVEST, Kind.COLLECT_WASTE,
                         Kind.FERTILIZE, Kind.DEPOSIT, Kind.EAT, Kind.ENTER_REFUGE),
        Species.COW: (Kind.GRAZE,),
        Species.NOMAD: (Kind.MILK, Kind.EAT, Kind.ATTACK,
                         Kind.FRIENDLY_CALL, Kind.SCARY_CALL),
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
    occupied = {a.position for a in world.actors.values() if not a.sheltered}
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


def _away(world: World, actor: Actor, point: tuple[float, float]) -> Action:
    """Simple transparent cow-response baseline, not a neural decision."""
    occupied = {a.position for a in world.actors.values() if a.id != actor.id
                and not a.sheltered}
    options = []
    for dx, dy in DIRECTIONS:
        x, y = actor.x + dx, actor.y + dy
        if (not world.grid.contains(x, y) or not world.grid.get(x, y).passable
                or (x, y) in occupied or
                any(s.x == x and s.y == y for s in world.settlements)):
            continue
        before_x, before_y = actor.x - point[0], actor.y - point[1]
        # Prefer movement away on the axis with the largest displacement.
        alignment = dx * before_x + dy * before_y
        distance = abs(x - point[0]) + abs(y - point[1])
        options.append((distance, alignment, dx, dy))
    if not options:
        return Action(Kind.REST)
    best = max((distance, alignment) for distance, alignment, _, _ in options)
    return Action(Kind.MOVE, *world.policy_rng.choice(sorted(
        (dx, dy) for d, a, dx, dy in options if (d, a) == best)))


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
            # M5 scripted *comparison* policy: perceives local calls, then
            # chooses to react. Calls NEVER move cattle through the world rules.
            from .perception import call_features
            fx, fy, fstrength, sx, sy, sstrength = call_features(world, actor)
            if sstrength > 0.08:
                signal = (actor.x + sx * world.config.call_radius,
                          actor.y + sy * world.config.call_radius)
                action = _away(world, actor, signal)
            elif fstrength > 0.08:
                signal = (actor.x + fx * world.config.call_radius,
                          actor.y + fy * world.config.call_radius)
                action = _toward(world, actor, (round(signal[0]), round(signal[1])))
            elif tile.grass > 0.6 or tile.crop > 0.6:
                action = Action(Kind.GRAZE)
            else:
                target = _nearest(world, actor, lambda t: t.grass > 1 or t.crop > 1)
                action = _toward(world, actor, target) if target else Action(Kind.REST)
        elif actor.species == Species.FARMER:
            home = next(s for s in world.settlements if s.id == actor.home_id)
            threats = [r for r in world.actors.values()
                       if r.species == Species.RAVAGER and not r.sheltered
                       and abs(r.x - actor.x) + abs(r.y - actor.y) <= 4]
            if actor.sheltered:
                if actor.energy < 40 and world._nearby_settlement(actor):
                    action = Action(Kind.EAT)
                elif threats:
                    action = Action(Kind.REST)
                else:
                    target = _nearest(world, actor, lambda t: t.crop >= 1.5)
                    action = _toward(world, actor, target or (home.x + 2, home.y))
            elif threats and world._nearby_settlement(actor) and actor.home_id == world._nearby_settlement(actor).id:
                action = Action(Kind.ENTER_REFUGE)
            elif threats:
                action = _toward(world, actor, (home.x, home.y))
            elif actor.energy < 45 and (actor.carried_food >= 1 or
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
            visible_cows = sorted((c for c in world.actors.values()
                                   if c.species == Species.COW
                                   and abs(c.x - actor.x) + abs(c.y - actor.y) <= 5),
                                  key=lambda c: c.id)
            nearby_ravagers = sorted((r for r in world.actors.values()
                                      if r.species == Species.RAVAGER
                                      and abs(r.x - actor.x) + abs(r.y - actor.y) <= 5),
                                     key=lambda r: (abs(r.x - actor.x) + abs(r.y - actor.y), r.id))
            if nearby_ravagers and any(abs(c.x - nearby_ravagers[0].x)
                                       + abs(c.y - nearby_ravagers[0].y) <= 3
                                       for c in visible_cows):
                target = nearby_ravagers[0]
                distance = abs(target.x - actor.x) + abs(target.y - actor.y)
                if distance <= 1 and actor.stamina >= world.config.nomad_attack_stamina:
                    action = Action(Kind.ATTACK)
                else:
                    action = _toward(world, actor, target.position)
            elif world.herd_goal is not None and visible_cows:
                # Greedy hand-built reference: push cattle away from a nomad
                # behind the herd, or lure them from ahead; it sees only local cows.
                mean_x = sum(c.x for c in visible_cows) / len(visible_cows)
                mean_y = sum(c.y for c in visible_cows) / len(visible_cows)
                gx, gy = world.herd_goal
                toward_x = 1 if gx > mean_x else -1 if gx < mean_x else 0
                toward_y = 1 if gy > mean_y else -1 if gy < mean_y else 0
                relative = (actor.x - mean_x) * toward_x + (actor.y - mean_y) * toward_y
                if relative < -0.6 and actor.scary_cooldown == 0:
                    action = Action(Kind.SCARY_CALL)
                elif relative > 0.6 and actor.friendly_cooldown == 0:
                    action = Action(Kind.FRIENDLY_CALL)
                else:
                    behind = (round(mean_x - 2 * toward_x),
                              round(mean_y - 2 * toward_y))
                    action = _toward(world, actor, behind)
            elif actor.energy < 60 and actor.carried_food >= 0.7:
                action = Action(Kind.EAT)
            elif any(c.species == Species.COW and c.milk > 0.4 and
                     abs(c.x - actor.x) + abs(c.y - actor.y) <= 1
                     for c in world.actors.values()):
                action = Action(Kind.MILK)
            else:
                cows = sorted((c for c in world.actors.values()
                               if c.species == Species.COW
                               and abs(c.x - actor.x) + abs(c.y - actor.y) <= 5),
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
                           and not a.sheltered
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
