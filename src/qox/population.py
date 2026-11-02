"""M9 explicit, bounded demographic rules; no scripted tactical decisions.

A birth is an energy-costly, local biological event, not a training reward.
This first model is single-parent reproduction (pairing/genetics are future work).
"""

from __future__ import annotations

from collections import Counter

from .entities import Actor, Species

# Species-specific maturity and recovery periods in simulation ticks.
MATURITY = {
    Species.FARMER: 120,
    Species.COW: 70,
    Species.NOMAD: 120,
    Species.SCAVENGER: 60,
    Species.RAVAGER: 260,
}
# Aging is an attrition mechanism in the optional population experiment.
LIFESPAN = {
    Species.FARMER: 1600,
    Species.COW: 900,
    Species.NOMAD: 1500,
    Species.SCAVENGER: 750,
    Species.RAVAGER: 2200,
}
COOLDOWN = {
    Species.FARMER: 150,
    Species.COW: 90,
    Species.NOMAD: 150,
    Species.SCAVENGER: 80,
    Species.RAVAGER: 360,
}

REPRO_CONFIG_FIELDS = (
    "population_dynamics_enabled", "reproduction_enabled", "reproduction_interval", "births_per_species_limit",
    "max_farmers", "max_cows", "max_nomads", "max_scavengers", "max_ravagers",
    "reproduction_energy_threshold", "reproduction_energy_cost", "offspring_energy",
)


def mature_age(species: Species) -> int:
    return MATURITY[species]


def is_juvenile(world, actor: Actor) -> bool:
    return bool((world.config.population_dynamics_enabled or
                 world.config.reproduction_enabled) and actor.age < mature_age(actor.species))


def population_caps(config) -> dict[Species, int]:
    return {species: getattr(config, f"max_{species.value}s") for species in Species}


def birth_step(world) -> None:
    """Births consume energy, reserve unique IDs, and require a vacant adjacent cell.

    Eligible parents compete in seeded random order to avoid systematic ID
    advantage. An offspring cannot act or reproduce on its birth tick.
    """
    cfg = world.config
    if not cfg.reproduction_enabled:
        return
    caps = population_caps(cfg)
    counts = Counter(a.species for a in world.actors.values())
    parents = sorted(world.actors)
    world.rng.shuffle(parents)
    for aid in parents:
        parent = world.actors[aid]
        species = parent.species
        if (parent.sheltered or is_juvenile(world, parent) or
                parent.energy < cfg.reproduction_energy_threshold or
                world.tick - parent.last_birth_tick < COOLDOWN[species] or
                counts[species] >= caps[species] or
                world.birth_totals[species] >= cfg.births_per_species_limit):
            continue
        # A simultaneous step cannot create offspring on an occupied tile.
        nearby = [(parent.x + dx, parent.y + dy)
                  for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))]
        vacant = [pos for pos in nearby if world.grid.contains(*pos)
                  and world.grid.get(*pos).passable
                  and pos not in {(s.x, s.y) for s in world.settlements}
                  and all(a.sheltered or a.position != pos for a in world.actors.values())]
        if not vacant:
            continue
        x, y = vacant[world.rng.randrange(len(vacant))]
        parent.energy -= cfg.reproduction_energy_cost
        parent.last_birth_tick = world.tick
        child = Actor(id=world.next_id, species=species, x=x, y=y,
                      energy=cfg.offspring_energy, health=70.0, stamina=70.0,
                      age=0, parent_id=parent.id,
                      home_id=parent.home_id, band_id=parent.band_id)
        world.actors[child.id] = child
        world.next_id += 1
        world.birth_totals[species] += 1
        counts[species] += 1
        world.emit("birth", child.id, 1.0, f"parent:{parent.id}")
