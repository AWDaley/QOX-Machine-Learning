"""M7 finite matter, scavenger sensing, competition, training and replay."""

import numpy as np
import pytest

from qox.actions import Action, Kind
from qox.config import SimulationConfig
from qox.entities import Carcass, Species
from qox.envs.core import QoxCoreEnv
from qox.perception import (ACTIONS, SCAVENGER_FEATURES, SCAVENGER_OBSERVATION_VERSION,
                            features_for, legal_action_mask, observe)
from qox.trace import replay, run
from qox.world import World


def setup_world(**kwargs):
    return World(SimulationConfig(width=14, height=12, farmers=0, cows=0,
                                  nomads=0, scavengers=1, ravagers=0,
                                  grass_growth=0, waste_decay=0, **kwargs))


def scavenger(world):
    return next(a for a in world.actors.values() if a.species == Species.SCAVENGER)


def test_scavenger_digestion_recycles_fraction_and_not_duplicate():
    world = setup_world(ecology_interval=1)
    actor = scavenger(world)
    tile = world.grid.get(*actor.position)
    tile.waste = 3.0
    before = tile.waste + actor.dung_pending
    world.step({actor.id: Action(Kind.EAT)})
    # 1 unit eaten, 0.12 material returned; 0.88 removed from waste economy.
    assert tile.waste == pytest.approx(2.12)
    assert actor.dung_pending == pytest.approx(0.0)
    assert tile.waste + actor.dung_pending < before
    assert sum(e.amount for e in world.events if e.kind == "scavenger_residue") == pytest.approx(0.12)
    assert world.summary()["scavenger_waste_eaten"] == pytest.approx(1.0)


def test_scavenger_repeated_digesting_cannot_generate_more_mass():
    world = setup_world(ecology_interval=1)
    actor = scavenger(world)
    tile = world.grid.get(*actor.position)
    tile.waste = 1.0
    total = tile.waste + actor.dung_pending
    for _ in range(40):
        world.step({actor.id: Action(Kind.EAT)})
        remaining = tile.waste + actor.dung_pending
        assert remaining <= total + 1e-7
        total = remaining
    assert total < 0.15


def test_carcass_consumption_is_exclusive_and_remaining_mass_finite():
    world = setup_world(ecology_interval=100)
    actor = scavenger(world)
    world.carcasses[1] = Carcass(1, actor.x, actor.y, 3.0, Species.COW)
    world.step({actor.id: Action(Kind.EAT_CARCASS)})
    assert world.carcasses[1].meat == pytest.approx(1.0)
    assert actor.dung_pending == pytest.approx(0.24)
    assert world.summary()["scavenger_carrion_eaten"] == pytest.approx(2.0)
    assert sum(e.amount for e in world.events if e.kind == "eat_carcass") == pytest.approx(2.0)
    assert world.events[-1].detail == "1"  # does not rewrite carried-carcass semantics
    world.carcasses[1].carrier_id = 9000
    # A carried carcass is not legally available to another scavenger.
    assert not legal_action_mask(world, actor)[ACTIONS[Species.SCAVENGER].index(
        Action(Kind.EAT_CARCASS))]
    world.carcasses[1].carrier_id = None


def test_carcass_decay_returns_fertility_without_new_waste_or_meat():
    world = setup_world(ecology_interval=1)
    actor = scavenger(world)
    tile = world.grid.get(*actor.position)
    tile.fertility = 0.3
    world.carcasses[1] = Carcass(1, actor.x, actor.y, 1.0, Species.COW)
    world.step({actor.id: Action()})
    assert world.carcasses[1].meat == pytest.approx(0.96)
    assert tile.fertility == pytest.approx(0.3 + 0.04 * world.config.carcass_fertility_return)
    assert tile.waste == 0
    assert world.summary()["carcass_mass_decayed"] == pytest.approx(0.04)


def test_scavenger_scent_respects_range_and_carrier():
    world = setup_world()
    actor = scavenger(world)
    actor.x, actor.y = 5, 5
    world.carcasses[1] = Carcass(1, 8, 5, 10, Species.COW)
    info = dict(zip(features_for(Species.SCAVENGER),
                    observe(world, actor)["observation"], strict=True))
    assert SCAVENGER_OBSERVATION_VERSION == 2
    assert len(SCAVENGER_FEATURES) == 11
    assert info["carcass_scent_dx"] > 0
    assert info["carcass_scent_power"] > 0
    assert info["carcass_here"] == 0
    world.carcasses[1].carrier_id = 22
    info = dict(zip(features_for(Species.SCAVENGER),
                    observe(world, actor)["observation"], strict=True))
    assert info["carcass_scent_power"] == 0
    world.carcasses[1].carrier_id = None
    world.carcasses[1].x = 13
    info = dict(zip(features_for(Species.SCAVENGER),
                    observe(world, actor)["observation"], strict=True))
    assert info["carcass_scent_power"] == 0


def test_waste_scent_is_local_without_leaking_map_wide_content():
    world = setup_world()
    actor = scavenger(world)
    actor.x, actor.y = 5, 5
    world.grid.get(6, 5).waste = 4
    info = dict(zip(features_for(Species.SCAVENGER),
                    observe(world, actor)["observation"], strict=True))
    assert info["waste_patch_seen"] == 1
    assert info["waste_patch_dx"] > 0
    world.grid.get(6, 5).waste = 0
    world.grid.get(12, 5).waste = 10
    info = dict(zip(features_for(Species.SCAVENGER),
                    observe(world, actor)["observation"], strict=True))
    assert info["waste_patch_seen"] == 0
    assert info["waste_scent_strength"] == 0


@pytest.mark.parametrize("scenario", ("scavenger_waste", "scavenger_carrion",
                                       "scavenger_competition", "scavenger_mixed"))
def test_scenarios_have_finite_food_and_scripted_reference(scenario):
    env = QoxCoreEnv(scenario=scenario, max_steps=55, scripted_others=True)
    obs, _ = env.reset(seed=44)
    name = next(name for name in env.agents if name.startswith("scavenger_"))
    assert len(obs[name]["observation"]) == 71
    assert np.all(np.isfinite(obs[name]["observation"]))
    for _ in range(55):
        if not env.agents:
            break
        env.step({})
    assert (sum(e.amount for e in env.world.events
                if e.kind in ("eat_waste", "scavenger_carrion")) > 0)


def test_farmer_competes_for_real_manure_without_teleportation():
    env = QoxCoreEnv(scenario="scavenger_competition", max_steps=48,
                     scripted_others=True)
    env.reset(seed=43)
    for _ in range(48):
        if not env.agents:
            break
        env.step({})
    events = env.world.events
    assert sum(e.amount for e in events if e.kind == "collect_waste") > 0
    assert sum(e.amount for e in events if e.kind == "eat_waste") > 0
    assert sum(e.amount for e in events if e.kind == "fertilize") > 0


def test_scavenger_interactions_replay_deterministically():
    config = SimulationConfig(width=16, height=14, farmers=2, cows=2,
                              nomads=0, scavengers=2, ravagers=0, seed=59)
    world, trace = run(config, ticks=80, mode="scripted", record=True)
    assert replay(trace).state_hash() == world.state_hash()


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        setup_world(scavenger_residue_fraction=1.0)
    with pytest.raises(ValueError):
        setup_world(carcass_fertility_return=-0.5)


def test_scavenger_ppo_checkpoint_training_smoke(tmp_path):
    pytest.importorskip("torch")
    from qox.learning.scavenger_ppo import (ScavengerTrainOptions, evaluate_scavenger,
                                           load_scavenger, train_scavenger)
    output = tmp_path / "scavenger.pt"
    result = train_scavenger(ScavengerTrainOptions(steps=128, worlds=2,
                             rollout=64, epochs=1, max_steps=12), output)
    assert result["steps"] == 128
    assert load_scavenger(output).encoder[0].in_features == 71
    from qox.learning.inference import ScavengerController
    controller = ScavengerController(output)
    world, trace = run(SimulationConfig(seed=22), 9, mode="learned-scavenger",
                       record=True, checkpoint=output)
    assert replay(trace).state_hash() == world.state_hash()
    selected = next(a for a in world.actors.values()
                    if a.species == Species.SCAVENGER)
    assert len(controller.probabilities(world, selected.id)) == len(ACTIONS[Species.SCAVENGER])
    report = evaluate_scavenger(output, episodes=2, max_steps=12)
    assert report["episodes"] == 2
    assert all("waste_eaten" in x for x in report["results"])


def test_parallel_adapter_optional_scavenger_scenario():
    import importlib.util
    if (importlib.util.find_spec("pettingzoo") is None or
            importlib.util.find_spec("gymnasium") is None):
        pytest.skip("Optional PettingZoo/Gymnasium extras absent")
    from qox.envs.parallel_env import QoxParallelEnv
    env = QoxParallelEnv(scenario="scavenger_waste", max_steps=4)
    observations, _ = env.reset(seed=51)
    for name, state in observations.items():
        assert env.observation_space(name).contains(state)
    with pytest.raises(ValueError):
        QoxParallelEnv(scenario="scavenger_mixed")
