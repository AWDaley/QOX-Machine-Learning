"""M4 physical refuge, agriculture, and learned farmer regression tests."""

from dataclasses import replace

import numpy as np
import pytest

from qox.actions import Action, Kind
from qox.config import SimulationConfig
from qox.entities import Species
from qox.envs.core import QoxCoreEnv
from qox.perception import (ACTIONS, FEATURES, FARMER_FEATURES,
                            features_for, legal_action_mask, observe)
from qox.trace import replay, run
from qox.world import World


def cfg(**kw):
    base = SimulationConfig(width=12, height=12, farmers=1, cows=0,
                            nomads=0, scavengers=0, ravagers=0)
    return replace(base, **kw)


def farmer_and_home(world):
    farmer = next(a for a in world.actors.values() if a.species == Species.FARMER)
    return farmer, world.home(farmer)


def test_shelter_admission_and_exit_through_actual_entrance():
    world = World(cfg())
    farmer, home = farmer_and_home(world)
    farmer.x, farmer.y = home.x + 1, home.y
    enter = ACTIONS[Species.FARMER].index(Action(Kind.ENTER_REFUGE))
    assert legal_action_mask(world, farmer)[enter] == 1
    world.step({farmer.id: Action(Kind.ENTER_REFUGE)})
    assert farmer.sheltered
    assert farmer.position == (home.x, home.y)
    assert world.refuge_occupants(home.id) == 1
    assert any(e.kind == "enter_refuge" for e in world.last_events)
    assert legal_action_mask(world, farmer)[enter] == 0
    assert legal_action_mask(world, farmer)[ACTIONS[Species.FARMER].index(
        Action(Kind.HARVEST))] == 0
    # Planting directly inside a sheltered settlement is impossible.
    before = world.grid.get(*farmer.position).crop
    world.step({farmer.id: Action(Kind.PLANT)})
    assert world.grid.get(*farmer.position).crop == before
    world.step({farmer.id: Action(Kind.MOVE, 1, 0)})
    assert not farmer.sheltered and farmer.position == (home.x + 1, home.y)
    assert world.refuge_occupants(home.id) == 0
    world.check_invariants()


def test_refuge_capacity_bounds_simultaneous_entry():
    world = World(cfg(farmers=3))
    home = world.settlements[0]
    home.refuge_capacity = 2
    candidates = list(world.actors.values())[:3]
    # Put every farmer into the same home group and at three distinct entrances.
    for actor, (x, y) in zip(candidates, [(home.x + 1, home.y),
                                         (home.x, home.y + 1),
                                         (home.x, home.y - 1)], strict=True):
        actor.home_id, actor.x, actor.y = home.id, x, y
    world.check_invariants()
    world.step({a.id: Action(Kind.ENTER_REFUGE) for a in candidates})
    assert world.refuge_occupants(home.id) == 2
    assert sum(not a.sheltered for a in candidates) == 1
    world.check_invariants()


def test_shelter_blocks_predator_but_only_when_inside():
    world = World(cfg(ravagers=1))
    farmer, home = farmer_and_home(world)
    predator = next(a for a in world.actors.values() if a.species == Species.RAVAGER)
    farmer.x, farmer.y = home.x - 1, home.y
    predator.x, predator.y = home.x - 2, home.y
    world.step({predator.id: Action(Kind.ATTACK)})
    assert farmer.health == 88.0  # outside is dangerous
    world.step({farmer.id: Action(Kind.ENTER_REFUGE), predator.id: Action(Kind.ATTACK)})
    assert farmer.sheltered and farmer.health == 88.0
    assert legal_action_mask(world, predator)[ACTIONS[Species.RAVAGER].index(
        Action(Kind.ATTACK))] == 0
    for _ in range(5):
        world.step({predator.id: Action(Kind.ATTACK)})
    assert farmer.health == 88.0


def test_refuge_deducts_real_food_and_stops_when_store_empty():
    world = World(cfg(refuge_ration=0.05))
    farmer, home = farmer_and_home(world)
    farmer.x, farmer.y = home.x + 1, home.y
    farmer.energy = 40
    home.stored_food = 0.12
    world.step({farmer.id: Action(Kind.ENTER_REFUGE)})
    for _ in range(5):
        world.step({})
    paid = sum(event.amount for event in world.events if event.kind == "refuge_ration")
    assert paid == pytest.approx(0.12)
    assert home.stored_food == pytest.approx(0, abs=1e-8)
    world.check_invariants()


def test_cattle_crop_damage_manure_recovery_and_fertilizer_budget():
    world = World(cfg(farmers=1, cows=1, ecology_interval=1,
                      grass_growth=0, crop_growth=0, waste_decay=0))
    farmer, home = farmer_and_home(world)
    cow = next(a for a in world.actors.values() if a.species == Species.COW)
    farmer.x, farmer.y = home.x + 1, home.y
    cow.x, cow.y = home.x + 2, home.y
    patch = world.grid.get(*cow.position)
    patch.grass, patch.crop, patch.planted = 0.0, 3.0, True
    world.step({cow.id: Action(Kind.GRAZE)})
    assert patch.crop == pytest.approx(1.0)
    assert patch.waste > 0
    assert any(e.kind == "dung" for e in world.last_events)
    # Move cow away so the farmer can collect the deposited waste.
    world.step({cow.id: Action(Kind.MOVE, 1, 0),
                farmer.id: Action(Kind.MOVE, 1, 0)})
    assert farmer.position == (home.x + 2, home.y)
    world.step({farmer.id: Action(Kind.COLLECT_WASTE)})
    collected = farmer.carried_waste
    assert collected > 0
    patch.fertility = 0.2
    world.step({farmer.id: Action(Kind.FERTILIZE)})
    assert patch.fertility > 0.2
    assert 0 <= farmer.carried_waste < collected
    world.check_invariants()


def test_cow_checkpoint_observation_contract_unchanged():
    assert len(FEATURES) == 60
    assert len(FARMER_FEATURES) == 16
    assert features_for(Species.COW) == FEATURES
    assert features_for(Species.FARMER) == FEATURES + FARMER_FEATURES
    world = World(cfg())
    farmer, _ = farmer_and_home(world)
    assert observe(world, farmer)["observation"].shape == (76,)
    assert (np.abs(observe(world, farmer)["observation"]) <= 1).all()


def test_curriculum_seeds_and_refuge_are_reproducible():
    for scenario in ("agriculture", "manure", "refuge", "farmer_mixed"):
        a = QoxCoreEnv(scenario=scenario, max_steps=10, scripted_others=True)
        b = QoxCoreEnv(scenario=scenario, max_steps=10, scripted_others=True)
        for seed in (42, 43, 44):
            a.reset(seed=seed)
            b.reset(seed=seed)
            assert a.world.state_hash() == b.world.state_hash()
            for _ in range(5):
                a.step({})
                b.step({})
                assert a.world.state_hash() == b.world.state_hash()


def test_farmer_training_load_benchmark_and_trace_replay(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.farmer_ppo import (FarmerPolicy, FarmerTrainOptions,
                                          evaluate_farmer, load_farmer, train_farmer)
    path = tmp_path / "farmer.pt"
    report = train_farmer(FarmerTrainOptions(steps=128, worlds=2, rollout=64,
                                            epochs=1, max_steps=32, seed=26), path)
    assert report["farmer_steps"] == 128
    policy = load_farmer(path)
    assert policy.actor.out_features == len(ACTIONS[Species.FARMER])
    original = evaluate_farmer(path, episodes=2, seed=45000, max_steps=32)
    assert original == evaluate_farmer(path, episodes=2, seed=45000, max_steps=32)
    world, trace = run(cfg(farmers=2, cows=1), ticks=12,
                       mode="learned-farmer", checkpoint=path, record=True)
    assert replay(trace).state_hash() == world.state_hash()
    assert main(["benchmark-farmer", "--checkpoint", str(path), "--episodes", "2",
                 "--max-steps", "32", "--report", str(tmp_path / "report.json")]) == 0
    assert (tmp_path / "report.json").exists()
    data = torch.load(path, weights_only=True)
    data["observation_features"] = ("broken",)
    bad = tmp_path / "bad.pt"
    torch.save(data, bad)
    with pytest.raises(ValueError, match="schema"):
        load_farmer(bad)
    assert isinstance(FarmerPolicy(16), torch.nn.Module)
