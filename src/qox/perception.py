"""Local, normalized observations and legal-action masks for QOX agents.

No global state is included in an actor's observation. The ordered feature
schema and per-species action vocabulary are versioned training contracts.
"""

from __future__ import annotations


import numpy as np

from .actions import Action, DIRECTIONS, Kind
from .entities import Actor, Species
from .world import World

OBSERVATION_VERSION = 1
VISION_RADIUS = 4

FEATURES = (
    "health", "energy", "stamina", "age", "carried_food", "carried_waste", "milk",
    "here_grass", "here_crop", "here_waste", "here_fertility", "here_cover",
    "grass_north", "grass_east", "grass_south", "grass_west",
    "crop_north", "crop_east", "crop_south", "crop_west",
    "passable_north", "passable_east", "passable_south", "passable_west",
    "food_seen", "food_dx", "food_dy", "food_amount",
    "ravager_seen", "ravager_dx", "ravager_dy", "ravager_distance",
    "cow_seen", "cow_dx", "cow_dy", "cow_distance",
    "nomad_seen", "nomad_dx", "nomad_dy", "nomad_distance",
    "farmer_seen", "farmer_dx", "farmer_dy", "farmer_distance",
    "carcass_seen", "carcass_dx", "carcass_dy", "carcass_meat",
    "home_known", "home_dx", "home_dy", "home_distance",
    "visible_cow_count", "visible_ravager_count",
    "waste_scent_dx", "waste_scent_dy", "waste_scent_strength",
    "recent_noise_dx", "recent_noise_dy", "recent_noise_strength",
)
OBSERVATION_SIZE = len(FEATURES)  # Legacy cow observation contract remains v1/60.
FARMER_FEATURES = (
    "sheltered", "home_entrance_near", "home_food_known", "home_food_stock",
    "home_refuge_fill", "here_planted", "here_crop_ready", "ripe_seen",
    "ripe_dx", "ripe_dy", "ripe_amount", "waste_seen", "waste_dx",
    "waste_dy", "waste_amount", "home_global_distance",
)
FARMER_OBSERVATION_VERSION = 1
NOMAD_OBSERVATION_VERSION = 1
# Opt-in v2 cow features preserve existing M3 v1 checkpoints in the default runner.
COW_CALL_OBSERVATION_VERSION = 2
COW_CALL_FEATURES = (
    "friendly_dx", "friendly_dy", "friendly_strength",
    "scary_dx", "scary_dy", "scary_strength",
)
NOMAD_FEATURES = (
    "visible_herd_fraction", "visible_herd_dx", "visible_herd_dy",
    "herd_goal_known", "herd_goal_dx", "herd_goal_dy",
    "nearby_cows_at_risk", "nearby_nomads", "friendly_ready", "scary_ready",
    "nearby_milk_available", "nearby_ravager_threat",
)


def features_for(species: Species, *, cow_calls: bool = False) -> tuple[str, ...]:
    if species == Species.FARMER:
        return FEATURES + FARMER_FEATURES
    if species == Species.NOMAD:
        return FEATURES + NOMAD_FEATURES
    if species == Species.COW and cow_calls:
        return FEATURES + COW_CALL_FEATURES
    return FEATURES


def observation_size(species: Species, *, cow_calls: bool = False) -> int:
    return len(features_for(species, cow_calls=cow_calls))


def _catalog(*kinds: Kind) -> tuple[Action, ...]:
    return (Action(Kind.WAIT), *(Action(Kind.MOVE, *delta) for delta in DIRECTIONS),
            Action(Kind.REST), *(Action(kind) for kind in kinds))


ACTIONS: dict[Species, tuple[Action, ...]] = {
    Species.FARMER: _catalog(Kind.PLANT, Kind.HARVEST, Kind.COLLECT_WASTE,
                             Kind.FERTILIZE, Kind.EAT, Kind.DEPOSIT, Kind.ENTER_REFUGE),
    Species.COW: _catalog(Kind.GRAZE),
    Species.NOMAD: (_catalog(Kind.MILK, Kind.EAT, Kind.FRIENDLY_CALL,
                             Kind.SCARY_CALL, Kind.ATTACK) +
                    tuple(Action(Kind.DASH, *delta) for delta in DIRECTIONS)),
    Species.SCAVENGER: _catalog(Kind.EAT, Kind.EAT_CARCASS),
    Species.RAVAGER: _catalog(Kind.EAT_CARCASS, Kind.ATTACK),
}


def _los(world: World, x0: int, y0: int, x1: int, y1: int) -> bool:
    """Bresenham visibility: the target is visible even if in opaque cover.

    Opaque *intermediate* cover/impassable tiles block vision; normal low
    vegetation affects neither line-of-sight nor an actor's detectability yet.
    """
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        twice = 2 * err
        if twice > -dy:
            err -= dy
            x += sx
        if twice < dx:
            err += dx
            y += sy
        if (x, y) != (x1, y1):
            tile = world.grid.get(x, y)
            if not tile.passable or tile.cover >= 0.9:
                return False
    return True


def _visible(world: World, observer: Actor, x: int, y: int) -> bool:
    return (abs(observer.x - x) + abs(observer.y - y) <= VISION_RADIUS
            and _los(world, observer.x, observer.y, x, y))


def _direction(observer: Actor, x: int, y: int) -> tuple[float, float, float]:
    dx, dy = x - observer.x, y - observer.y
    return (max(-1.0, min(1.0, dx / VISION_RADIUS)),
            max(-1.0, min(1.0, dy / VISION_RADIUS)),
            min(1.0, (abs(dx) + abs(dy)) / VISION_RADIUS))


def _nearest_actor(world: World, observer: Actor, species: Species):
    matches = [other for other in world.actors.values()
               if other.id != observer.id and other.species == species
               and _visible(world, observer, other.x, other.y)]
    return min(matches, key=lambda a: (abs(a.x - observer.x) + abs(a.y - observer.y),
                                       a.id), default=None)


def call_features(world: World, actor: Actor) -> tuple[float, ...]:
    """Loudest audible call of each type; no direct force or hidden global info.

    Other calls may interfere acoustically, but are not automatically interpreted
    as orders. Distances and strengths are local and normalized to [-1, 1].
    """
    values: list[float] = []
    for kind in (Kind.FRIENDLY_CALL, Kind.SCARY_CALL):
        audible = []
        for signal in world.calls:
            if signal.kind != kind.value or signal.source_id == actor.id:
                continue
            distance = abs(signal.x - actor.x) + abs(signal.y - actor.y)
            if distance > signal.radius:
                continue
            age = world.tick - signal.tick
            if not 0 <= age <= world.config.call_duration:
                continue
            strength = signal.intensity * (1 - distance / (signal.radius + 1)) * (
                1 - age / (world.config.call_duration + 1))
            if strength > 0:
                audible.append((-strength, signal.source_id, signal))
        if audible:
            neg_strength, _, source = min(audible)
            values.extend((max(-1.0, min(1.0, (source.x - actor.x) /
                                           world.config.call_radius)),
                           max(-1.0, min(1.0, (source.y - actor.y) /
                                           world.config.call_radius)),
                           min(1.0, -neg_strength)))
        else:
            values.extend((0.0, 0.0, 0.0))
    return tuple(values)


def legal_action_mask(world: World, actor: Actor) -> np.ndarray:
    """Mask only physically impossible actions; collision-risky moves stay legal."""
    tile = world.grid.get(*actor.position)
    mask = []
    for action in ACTIONS[actor.species]:
        kind = action.kind
        valid = True
        if actor.sheltered and kind not in (Kind.WAIT, Kind.MOVE, Kind.REST, Kind.EAT, Kind.DEPOSIT):
            mask.append(0)
            continue
        if kind in (Kind.MOVE, Kind.DASH):
            is_dash = kind == Kind.DASH
            distance = 2 if is_dash else 1
            cost = world.config.nomad_dash_stamina if is_dash else 0.5
            x, y = actor.x + distance * action.dx, actor.y + distance * action.dy
            valid = ((not is_dash or actor.species == Species.NOMAD)
                     and actor.stamina >= cost and world.grid.contains(x, y)
                     and world.grid.get(x, y).passable
                     and not any((s.x, s.y) == (x, y) for s in world.settlements))
            if valid and is_dash:
                mx, my = actor.x + action.dx, actor.y + action.dy
                valid = (world.grid.contains(mx, my)
                         and world.grid.get(mx, my).passable
                         and not any((s.x, s.y) == (mx, my) for s in world.settlements)
                         and not any(a.id != actor.id and not a.sheltered
                                     and a.position == (mx, my) for a in world.actors.values()))
        elif kind == Kind.GRAZE:
            valid = tile.grass > 0.01 or tile.crop > 0.01
        elif kind == Kind.ENTER_REFUGE:
            home = world.home(actor)
            valid = (actor.species == Species.FARMER and not actor.sheltered
                     and actor.stamina >= 1 and home is not None
                     and abs(actor.x - home.x) + abs(actor.y - home.y) <= 1
                     and world.refuge_occupants(home.id) < home.refuge_capacity)
        elif kind == Kind.PLANT:
            valid = not tile.planted and tile.fertility >= 0.2 and actor.carried_food >= 0.3
        elif kind == Kind.HARVEST:
            valid = tile.planted and tile.crop >= 1
        elif kind == Kind.COLLECT_WASTE:
            valid = tile.waste > 0
        elif kind == Kind.FERTILIZE:
            valid = actor.carried_waste > 0 and tile.fertility < 1 - 1e-8
        elif kind == Kind.MILK:
            valid = any(a.species == Species.COW and a.milk > 0
                        and abs(a.x - actor.x) + abs(a.y - actor.y) <= 1
                        for a in world.actors.values())
        elif kind == Kind.EAT:
            if actor.species in (Species.FARMER, Species.NOMAD):
                store = world._nearby_settlement(actor) if actor.species == Species.FARMER else None
                valid = actor.carried_food > 0 or (store is not None and store.stored_food > 0)
            else:
                valid = tile.waste > 0
        elif kind == Kind.EAT_CARCASS:
            valid = any(c.x == actor.x and c.y == actor.y for c in world.carcasses.values())
        elif kind == Kind.DEPOSIT:
            valid = actor.carried_food > 0 and world._nearby_settlement(actor) is not None
        elif kind in (Kind.FRIENDLY_CALL, Kind.SCARY_CALL):
            friendly = kind == Kind.FRIENDLY_CALL
            valid = (actor.species == Species.NOMAD
                     and (actor.friendly_cooldown if friendly else actor.scary_cooldown) == 0
                     and actor.stamina >= (world.config.friendly_call_stamina if friendly
                                           else world.config.scary_call_stamina))
        elif kind == Kind.ATTACK:
            if actor.species == Species.NOMAD:
                valid = (actor.stamina >= world.config.nomad_attack_stamina
                         and any(other.species == Species.RAVAGER
                                 and abs(other.x - actor.x) + abs(other.y - actor.y) <= 1
                                 for other in world.actors.values()))
            else:
                valid = actor.stamina >= 4 and any(
                    a.id != actor.id and a.species != Species.RAVAGER
                    and not a.sheltered
                    and abs(a.x - actor.x) + abs(a.y - actor.y) <= 1
                    for a in world.actors.values())
        mask.append(int(valid))
    return np.asarray(mask, dtype=np.int8)


def observe(world: World, actor: Actor, *, cow_calls: bool = False) -> dict[str, np.ndarray]:
    """Return bounded numeric observations and physical action masks."""
    tile = world.grid.get(*actor.position)
    values = [
        actor.health / 100, actor.energy / 100, actor.stamina / 100,
        min(actor.age / 1000, 1), min(actor.carried_food / 10, 1),
        min(actor.carried_waste / 10, 1), min(actor.milk / 10, 1),
        tile.grass / world.config.max_grass, tile.crop / world.config.max_crop,
        tile.waste / world.config.max_waste, tile.fertility, tile.cover,
    ]
    for resource in ("grass", "crop"):
        limit = world.config.max_grass if resource == "grass" else world.config.max_crop
        for dx, dy in DIRECTIONS:
            x, y = actor.x + dx, actor.y + dy
            values.append((getattr(world.grid.get(x, y), resource) / limit
                           if world.grid.contains(x, y) and _visible(world, actor, x, y)
                           else 0.0))
    for dx, dy in DIRECTIONS:
        x, y = actor.x + dx, actor.y + dy
        values.append(float(world.grid.contains(x, y)
                            and world.grid.get(x, y).passable
                            and not any((s.x, s.y) == (x, y) for s in world.settlements)))

    foods = []
    for y in range(max(0, actor.y - VISION_RADIUS),
                   min(world.grid.height, actor.y + VISION_RADIUS + 1)):
        for x in range(max(0, actor.x - VISION_RADIUS),
                       min(world.grid.width, actor.x + VISION_RADIUS + 1)):
            t = world.grid.get(x, y)
            if (t.grass + t.crop > 0.01 and _visible(world, actor, x, y)):
                foods.append((abs(x - actor.x) + abs(y - actor.y), y, x, t))
    if foods:
        _, y, x, source = min(foods)
        dx, dy, _ = _direction(actor, x, y)
        values.extend([1, dx, dy, min(1, (source.grass + source.crop) / 10)])
    else:
        values.extend([0, 0, 0, 0])

    for species in (Species.RAVAGER, Species.COW, Species.NOMAD, Species.FARMER):
        target = _nearest_actor(world, actor, species)
        if target is None:
            values.extend([0, 0, 0, 0])
        else:
            dx, dy, distance = _direction(actor, target.x, target.y)
            values.extend([1, dx, dy, distance])

    carcasses = [c for c in world.carcasses.values()
                 if _visible(world, actor, c.x, c.y)]
    nearest = min(carcasses,
                  key=lambda c: (abs(c.x - actor.x) + abs(c.y - actor.y), c.id),
                  default=None)
    if nearest:
        dx, dy, _ = _direction(actor, nearest.x, nearest.y)
        values.extend([1, dx, dy, min(nearest.meat / 12, 1)])
    else:
        values.extend([0, 0, 0, 0])

    home = next((s for s in world.settlements if s.id == actor.home_id), None)
    if home is not None:
        dx, dy, distance = _direction(actor, home.x, home.y)
        values.extend([1, dx, dy, distance])
    else:
        values.extend([0, 0, 0, 0])

    for species in (Species.COW, Species.RAVAGER):
        visible = sum(a.id != actor.id and a.species == species
                      and _visible(world, actor, a.x, a.y)
                      for a in world.actors.values())
        values.append(min(1, visible / 8))

    # Limited-radius scent gradient; not a perfect position or a remote map.
    scent_x = scent_y = scent_power = 0.0
    for y in range(max(0, actor.y - VISION_RADIUS),
                   min(world.grid.height, actor.y + VISION_RADIUS + 1)):
        for x in range(max(0, actor.x - VISION_RADIUS),
                       min(world.grid.width, actor.x + VISION_RADIUS + 1)):
            dist = abs(x - actor.x) + abs(y - actor.y)
            if dist <= VISION_RADIUS:
                strength = world.grid.get(x, y).waste / (1 + dist) ** 2
                scent_power += strength
                scent_x += strength * (x - actor.x)
                scent_y += strength * (y - actor.y)
    if scent_power > 1e-8:
        values.extend([max(-1, min(1, scent_x / scent_power / VISION_RADIUS)),
                       max(-1, min(1, scent_y / scent_power / VISION_RADIUS)),
                       min(1, scent_power / world.config.max_waste)])
    else:
        values.extend([0, 0, 0])

    # Recent physically generated noise, based on last-tick events only.
    noises = []
    for event in world.last_events:
        if event.kind not in ("move", "attack", "death"):
            continue
        if event.x is not None and event.y is not None and event.actor_id != actor.id:
            distance = abs(event.x - actor.x) + abs(event.y - actor.y)
            if distance <= VISION_RADIUS:
                noises.append((distance, event.actor_id, event.x, event.y))
    if noises:
        dist, _, x, y = min(noises)
        dx, dy, _ = _direction(actor, x, y)
        values.extend([dx, dy, max(0, 1 - dist / (VISION_RADIUS + 1))])
    else:
        values.extend([0, 0, 0])

    if actor.species == Species.FARMER:
        home = world.home(actor)
        adjacent = (home is not None and
                    abs(home.x - actor.x) + abs(home.y - actor.y) <= 1)
        can_inspect_store = bool(adjacent or actor.sheltered)
        crops = []
        waste = []
        for y in range(max(0, actor.y - VISION_RADIUS),
                       min(world.grid.height, actor.y + VISION_RADIUS + 1)):
            for x in range(max(0, actor.x - VISION_RADIUS),
                           min(world.grid.width, actor.x + VISION_RADIUS + 1)):
                if not _visible(world, actor, x, y):
                    continue
                seen_tile = world.grid.get(x, y)
                dist = abs(x - actor.x) + abs(y - actor.y)
                if seen_tile.planted and seen_tile.crop >= 1:
                    crops.append((dist, y, x, seen_tile.crop))
                if seen_tile.waste > 0.02:
                    waste.append((dist, y, x, seen_tile.waste))
        def target_features(choices: list[tuple[int, int, int, float]], scale: float):
            if not choices:
                return [0, 0, 0, 0]
            _, y, x, amt = min(choices)
            dx, dy, _ = _direction(actor, x, y)
            return [1, dx, dy, min(1, amt / scale)]

        values.extend([
            float(actor.sheltered), float(adjacent), float(can_inspect_store),
            min(1, home.stored_food / 25) if can_inspect_store and home else 0,
            (world.refuge_occupants(home.id) / max(1, home.refuge_capacity))
            if can_inspect_store and home else 0,
            float(tile.planted) if not actor.sheltered else 0,
            min(1, tile.crop / world.config.max_crop) if not actor.sheltered else 0,
            *target_features(crops, world.config.max_crop),
            *target_features(waste, world.config.max_waste),
            (min(1, (abs(home.x - actor.x) + abs(home.y - actor.y)) /
                 (world.grid.width + world.grid.height)) if home else 0),
        ])
    if actor.species == Species.NOMAD:
        visible_cows = [other for other in world.actors.values()
                        if other.species == Species.COW
                        and _visible(world, actor, other.x, other.y)]
        if visible_cows:
            herd_x = sum(c.x for c in visible_cows) / len(visible_cows)
            herd_y = sum(c.y for c in visible_cows) / len(visible_cows)
            herd_dx = max(-1, min(1, (herd_x - actor.x) / VISION_RADIUS))
            herd_dy = max(-1, min(1, (herd_y - actor.y) / VISION_RADIUS))
        else:
            herd_dx, herd_dy = 0.0, 0.0
        goal = world.herd_goal
        if goal is not None:
            goal_x = max(-1, min(1, (goal[0] - actor.x) / world.grid.width))
            goal_y = max(-1, min(1, (goal[1] - actor.y) / world.grid.height))
        else:
            goal_x, goal_y = 0.0, 0.0
        ravagers = [enemy for enemy in world.actors.values()
                    if enemy.species == Species.RAVAGER
                    and _visible(world, actor, enemy.x, enemy.y)]
        threatened = sum(any(abs(c.x - r.x) + abs(c.y - r.y) <= 2
                             for r in ravagers) for c in visible_cows)
        nearby_nomads = sum(a.id != actor.id and a.species == Species.NOMAD
                            and _visible(world, actor, a.x, a.y)
                            for a in world.actors.values())
        milk_cows = sum(c.milk > 0 and abs(c.x - actor.x) + abs(c.y - actor.y) <= 1
                        for c in visible_cows)
        values.extend((min(1, len(visible_cows) / 8), herd_dx, herd_dy,
                       float(goal is not None), goal_x, goal_y,
                       min(1, threatened / 4), min(1, nearby_nomads / 4),
                       float(actor.friendly_cooldown == 0 and
                             actor.stamina >= world.config.friendly_call_stamina),
                       float(actor.scary_cooldown == 0 and
                             actor.stamina >= world.config.scary_call_stamina),
                       min(1, milk_cows / 3), min(1, len(ravagers) / 4)))
    elif actor.species == Species.COW and cow_calls:
        values.extend(call_features(world, actor))
    result = np.asarray(values, dtype=np.float32)
    expected = observation_size(actor.species, cow_calls=cow_calls)
    if result.shape != (expected,):
        raise AssertionError(f"Broken observation schema: {len(result)} != {expected}")
    return {"observation": result, "action_mask": legal_action_mask(world, actor)}


def get_agent_name(actor: Actor) -> str:
    return f"{actor.species.value}_{actor.id}"
