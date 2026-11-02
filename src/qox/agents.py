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
        Species.RAVAGER: (Kind.EAT_CARCASS, Kind.GRAB_CARCASS,
                          Kind.DROP_CARCASS, Kind.HIBERNATE, Kind.WAKE),
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
        # Transparent FESTIVAL BASELINE only: policies are free not to follow
        # this heuristic. Testing its success is not evidence of learned society.
        if world.config.festival_enabled and actor.species in (
                Species.FARMER, Species.NOMAD):
            from .festivals import (FestivalKind, can_start, contribution_target,
                                    distance)
            active_here = next((f for f in world.active_festivals.values()
                                if distance(actor, f.x, f.y) <=
                                world.config.festival_attendance_radius), None)
            if active_here is not None:
                ravagers = sorted((r for r in world.actors.values()
                                   if r.species == Species.RAVAGER and
                                   distance(actor, r.x, r.y) <= 1), key=lambda r: r.id)
                if actor.species == Species.NOMAD and ravagers:
                    orders[aid] = Action(Kind.ATTACK)
                else:
                    orders[aid] = Action(Kind.REST)
                continue
            if can_start(world, actor, FestivalKind.COMBINED):
                orders[aid] = Action(Kind.START_COMBINED_FESTIVAL)
                continue
            own_type = (FestivalKind.FARMER if actor.species == Species.FARMER
                        else FestivalKind.NOMAD)
            if can_start(world, actor, own_type):
                orders[aid] = Action(Kind.START_FARMER_FESTIVAL if
                                     own_type == FestivalKind.FARMER else
                                     Kind.START_NOMAD_FESTIVAL)
                continue
            site = contribution_target(world, actor)
            if site is not None and actor.carried_food >= 1.0:
                grain_need = (world.config.festival_combined_grain
                              if actor.species == Species.FARMER
                              else world.config.festival_combined_milk)
                escrow = (site.festival_food if actor.species == Species.FARMER
                          else site.festival_milk)
                if escrow < grain_need:
                    orders[aid] = Action(Kind.CONTRIBUTE_FESTIVAL)
                    continue
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
            from .perception import _visible_actor
            home = next(s for s in world.settlements if s.id == actor.home_id)
            threats = [r for r in world.actors.values()
                       if r.species == Species.RAVAGER and not r.sheltered
                       and _visible_actor(world, actor, r)]
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
            from .perception import _visible_actor
            visible_cows = sorted((c for c in world.actors.values()
                                   if c.species == Species.COW
                                   and abs(c.x - actor.x) + abs(c.y - actor.y) <= 5),
                                  key=lambda c: c.id)
            nearby_ravagers = sorted((r for r in world.actors.values()
                                      if r.species == Species.RAVAGER
                                      and _visible_actor(world, actor, r)),
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
            # A transparent reference policy: local smell/vision and real food,
            # no knowledge of resources beyond detection range.
            from .perception import _visible_actor
            predators = [r for r in world.actors.values()
                         if r.species == Species.RAVAGER and
                         _visible_actor(world, actor, r)]
            imminent = [r for r in predators if abs(r.x - actor.x) +
                        abs(r.y - actor.y) <= 2]
            edible = [c for c in world.carcasses.values()
                      if c.carrier_id is None and c.x == actor.x and c.y == actor.y]
            if imminent:
                action = _away(world, actor, imminent[0].position)
            elif edible and actor.energy < 99:
                action = Action(Kind.EAT_CARCASS)
            elif tile.waste >= 0.1 and actor.energy < 99:
                action = Action(Kind.EAT)
            else:
                opportunities = []
                for c in world.carcasses.values():
                    d = abs(c.x - actor.x) + abs(c.y - actor.y)
                    if c.carrier_id is None and d <= 7:
                        opportunities.append((-c.meat / (d + 1), d, c.y, c.x))
                for y in range(max(0, actor.y - 4), min(world.grid.height, actor.y + 5)):
                    for x in range(max(0, actor.x - 4), min(world.grid.width, actor.x + 5)):
                        d = abs(x - actor.x) + abs(y - actor.y)
                        waste = world.grid.get(x, y).waste
                        if d <= 4 and waste > 0.1:
                            opportunities.append((-waste / (d + 1), d, y, x))
                if opportunities and actor.energy < 98:
                    _, _, y, x = min(opportunities)
                    action = _toward(world, actor, (x, y))
                else:
                    action = Action(Kind.REST)
        else:
            # Interpretable M6 predator reference. This is a heuristic baseline,
            # not a hidden behavior imposed on neural-network-controlled actors.
            prey = sorted((a for a in world.actors.values()
                           if a.id != actor.id and a.species != Species.RAVAGER
                           and not a.sheltered
                           and abs(a.x - actor.x) + abs(a.y - actor.y) <= 6),
                          key=lambda a: (abs(a.x - actor.x) + abs(a.y - actor.y), a.id))
            nearby = [a for a in prey if abs(a.x - actor.x) + abs(a.y - actor.y) <= 1]
            has_carcass = any(c.x == actor.x and c.y == actor.y and
                              c.carrier_id in (None, actor.id)
                              for c in world.carcasses.values())
            local_carcass = sorted((c for c in world.carcasses.values()
                                    if c.carrier_id is None),
                                   key=lambda c: (abs(c.x - actor.x) + abs(c.y - actor.y), c.id))
            if actor.hibernating:
                action = (Action(Kind.WAKE) if actor.energy < 40 or actor.satiety < 0.1
                          else Action(Kind.WAIT))
            elif actor.charge_recovery or actor.charge_windup:
                action = Action(Kind.WAIT)
            elif (actor.dragged_carcass_id is not None and tile.cover < 0.5):
                protected = _nearest(world, actor, lambda t: t.cover >= 0.5, radius=5)
                if actor.drag_cooldown:
                    action = Action(Kind.WAIT)
                else:
                    action = (_toward(world, actor, protected) if protected
                              else Action(Kind.DROP_CARCASS))
            elif actor.dragged_carcass_id is not None and tile.cover >= 0.5:
                action = (Action(Kind.EAT_CARCASS) if has_carcass
                          else Action(Kind.DROP_CARCASS))
            elif has_carcass and tile.cover >= 0.5:
                action = Action(Kind.EAT_CARCASS)
            elif has_carcass and actor.satiety < 0.35 and actor.energy < 25:
                action = Action(Kind.EAT_CARCASS)
            elif local_carcass and (abs(local_carcass[0].x - actor.x) +
                                    abs(local_carcass[0].y - actor.y) <= 1):
                action = Action(Kind.GRAB_CARCASS)
            elif local_carcass and (abs(local_carcass[0].x - actor.x) +
                                    abs(local_carcass[0].y - actor.y) <= 3):
                action = _toward(world, actor,
                                 (local_carcass[0].x, local_carcass[0].y))
            elif nearby and actor.stamina >= 4.0:
                action = Action(Kind.ATTACK)
            elif (prey and abs(prey[0].x - actor.x) + abs(prey[0].y - actor.y)
                  in (2, 3) and (prey[0].x == actor.x or prey[0].y == actor.y)
                  and actor.charge_cooldown == 0
                  and actor.stamina >= world.config.ravager_charge_stamina):
                target = prey[0]
                dx = 0 if target.x == actor.x else 1 if target.x > actor.x else -1
                dy = 0 if target.y == actor.y else 1 if target.y > actor.y else -1
                action = Action(Kind.CHARGE, dx, dy)
            elif actor.satiety >= 0.4 and actor.energy >= 65 and not prey:
                action = Action(Kind.HIBERNATE)
            elif prey:
                step = _toward(world, actor, prey[0].position)
                action = (Action(Kind.SNEAK, step.dx, step.dy)
                          if step.kind == Kind.MOVE and actor.sneak_cooldown == 0
                          and actor.stamina >= world.config.ravager_sneak_stamina
                          and tile.cover >= 0.5 else step)
            else:
                from .festivals import festival_cue
                fx, fy, strength = festival_cue(world, actor)
                if strength > 0.03:
                    # Follow a LOCAL festival noise/scent gradient, not a
                    # global map target or a scripted teleport/spawn.
                    radius = world.config.festival_signal_radius
                    target = (actor.x + round(fx * radius),
                              actor.y + round(fy * radius))
                    action = _toward(world, actor, target)
                else:
                    action = (_toward(world, actor, (local_carcass[0].x,
                                                    local_carcass[0].y)) if local_carcass
                              else Action(Kind.REST))
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
