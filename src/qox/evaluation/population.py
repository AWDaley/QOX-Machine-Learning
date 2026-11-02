"""Frozen-policy, multi-seed M9 ecological persistence experiments.

No training or automatic rescue occurs here; extinction and failed festivals
are observed outcomes. Hold out seeds when reporting learned-policy results.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev

from ..agents import actions_for, scripted_actions
from ..config import SimulationConfig
from ..entities import Species
from ..perception import ACTIONS, action_catalog
from ..world import World


class FrozenControllers:
    """One immutable policy per configured species; the rest remain scripted."""

    def __init__(self, checkpoints: dict[str, Path | None],
                 festival_checkpoint: Path | None = None):
        from ..learning.inference import (CowController, FarmerController,
                                          NomadController, ScavengerController,
                                          RavagerController, FestivalController)
        constructors = {Species.COW: CowController, Species.FARMER: FarmerController,
                        Species.NOMAD: NomadController, Species.SCAVENGER: ScavengerController,
                        Species.RAVAGER: RavagerController}
        self.models = {species: constructors[species](path)
                       for species in Species
                       if (path := checkpoints.get(species.value)) is not None}
        if festival_checkpoint is not None:
            controller = FestivalController(festival_checkpoint)
            if controller.species in self.models:
                raise ValueError("Use either a festival or ordinary checkpoint for the same species")
            self.models[controller.species] = controller
            self.festival_species = controller.species
        else:
            self.festival_species = None
        if not self.models:
            raise ValueError("Learned evaluation requires at least one checkpoint")

    def actions(self, world: World) -> dict:
        orders = scripted_actions(world)
        for actor in sorted(world.actors.values(), key=lambda a: a.id):
            controller = self.models.get(actor.species)
            if controller is None:
                continue
            probabilities = controller.probabilities(world, actor.id)
            choice = max(range(len(probabilities)), key=probabilities.__getitem__)
            catalog = (action_catalog(actor.species, festivals=True)
                       if actor.species == self.festival_species else ACTIONS[actor.species])
            orders[actor.id] = catalog[choice]
        return orders


def run_study(config: SimulationConfig, *, ticks: int, seeds: int,
              initial_seed: int, sample_every: int, mode: str = "scripted",
              checkpoints: dict[str, Path | None] | None = None,
              festival_checkpoint: Path | None = None) -> dict:
    # Reproduction-disabled control keeps the same initial adult ages,
    # aging, juvenile restrictions and environmental dynamics.
    config = replace(config, population_dynamics_enabled=True)
    if ticks < 1 or seeds < 1 or sample_every < 1:
        raise ValueError("Study ticks, seeds, and sample interval must be positive")
    if mode not in ("scripted", "random", "idle", "learned"):
        raise ValueError(f"Unsupported study policy mode: {mode}")
    if mode != "learned" and (festival_checkpoint or any((checkpoints or {}).values())):
        raise ValueError("Checkpoint arguments require --mode learned")
    policies = (FrozenControllers(checkpoints or {}, festival_checkpoint)
                if mode == "learned" else None)
    if policies and policies.festival_species is not None and not config.festival_enabled:
        raise ValueError("Festival checkpoint requires festival_enabled=True")
    outcomes = []
    for seed in range(initial_seed, initial_seed + seeds):
        world = World(replace(config, seed=seed))
        series = [{"tick": 0, "population": world.summary()["population"]}]
        initial_population = world.summary()["population"]
        extinction_tick = {species.value: None for species in Species}
        for _ in range(ticks):
            orders = policies.actions(world) if policies else actions_for(world, mode)
            world.step(orders)
            alive = world.summary()["population"] if (
                world.tick % sample_every == 0 or world.tick == ticks) else None
            if alive is not None:
                series.append({"tick": world.tick, "population": alive,
                               "births": world.summary()["births"],
                               "stored_food": world.summary()["stored_food"],
                               "festival_successes": world.summary()["festival_successes"]})
            for species in Species:
                if extinction_tick[species.value] is None and not any(
                        actor.species == species for actor in world.actors.values()):
                    extinction_tick[species.value] = world.tick
        final = world.summary()
        causes = {cause: sum(event.kind == "death_cause" and event.detail == cause
                             for event in world.events)
                  for cause in ("starvation", "predation", "age", "other")}
        outcomes.append({"seed": seed, "initial_population": initial_population,
                         "final_population": final["population"],
                         "births": final["births"], "deaths": final["deaths"],
                         "death_causes": causes, "juveniles": final["juveniles"],
                         "extinction_tick": extinction_tick,
                         "festival_successes": final["festival_successes"],
                         "festival_failures": final["festival_failures"],
                         "stored_food": final["stored_food"],
                         "final_state_hash": final["state_hash"],
                         "time_series": series})
    def avg(fn):
        values = [fn(outcome) for outcome in outcomes]
        return {"mean": round(mean(values), 4),
                "std": round(pstdev(values), 4), "min": min(values), "max": max(values)}

    return {
        "mode": mode, "ticks": ticks, "seeds": seeds,
        "initial_seed": initial_seed, "sample_every": sample_every,
        "config": config.as_dict(),
        "checkpoints": {key: str(path) for key, path in (checkpoints or {}).items()
                        if path is not None},
        "festival_checkpoint": str(festival_checkpoint) if festival_checkpoint else None,
        "mean_final_population": {s.value: avg(lambda o, species=s.value: o["final_population"][species])
                                  for s in Species},
        "mean_births": {s.value: avg(lambda o, species=s.value: o["births"][species]) for s in Species},
        "mean_deaths": {s.value: avg(lambda o, species=s.value: o["deaths"][species]) for s in Species},
        "mean_festival_successes": avg(lambda o: o["festival_successes"]),
        "outcomes": outcomes,
    }
