"""M2 observation, masking, and simultaneous-agent contract tests."""

import numpy as np
import pytest

from qox.actions import Action, Kind
from qox.config import SimulationConfig
from qox.entities import Species
from qox.envs.core import QoxCoreEnv
from qox.perception import ACTIONS, FEATURES, OBSERVATION_SIZE, observe, observation_size
from qox.world import World


def small(**kw):
    config = SimulationConfig(width=12, height=12, farmers=2, cows=3,
                              nomads=2, scavengers=2, ravagers=1)
    return SimulationConfig(**{**config.as_dict(), **kw})


def test_observation_schema_ranges_and_masks_for_all_species():
    world = World(small())
    assert len(FEATURES) == OBSERVATION_SIZE == len(set(FEATURES))
    for actor in world.actors.values():
        obs = observe(world, actor)
        assert obs["observation"].shape == (observation_size(actor.species),)
        assert obs["observation"].dtype == np.float32
        assert np.isfinite(obs["observation"]).all()
        assert (obs["observation"] >= -1).all()
        assert (obs["observation"] <= 1).all()
        assert obs["action_mask"].shape == (len(ACTIONS[actor.species]),)
        assert obs["action_mask"].dtype == np.int8
        assert obs["action_mask"][0] == 1  # WAIT always physically valid
        assert obs["action_mask"][5] == 1  # REST


def test_vision_range_and_occlusion_no_global_enemy_coordinates():
    world = World(small(farmers=0, cows=1, nomads=0, scavengers=0, ravagers=1))
    cow = next(a for a in world.actors.values() if a.species == Species.COW)
    ravager = next(a for a in world.actors.values() if a.species == Species.RAVAGER)
    cow.x, cow.y = 3, 3
    ravager.x, ravager.y = 3, 6
    hidden = world.grid.get(3, 4)
    hidden.cover = 0.99
    obs = observe(world, cow)["observation"]
    assert obs[FEATURES.index("ravager_seen")] == 0
    assert obs[FEATURES.index("ravager_dx")] == 0
    assert obs[FEATURES.index("ravager_dy")] == 0
    hidden.cover = 0
    obs = observe(world, cow)["observation"]
    assert obs[FEATURES.index("ravager_seen")] == 1
    assert obs[FEATURES.index("ravager_dy")] > 0
    ravager.y = 9
    obs = observe(world, cow)["observation"]
    assert obs[FEATURES.index("ravager_seen")] == 0


def test_mask_and_invalid_action_fallback():
    env = QoxCoreEnv(config=small(farmers=0, cows=1, nomads=0, scavengers=0,
                                  ravagers=0), max_steps=3)
    cow_name = env.possible_agents[0]
    cow = env.world.actors[env.name_to_id[cow_name]]
    cow.x, cow.y = 0, 0
    tile = env.world.grid.get(0, 0)
    tile.grass = tile.crop = 0
    obs = observe(env.world, cow)
    north_idx = ACTIONS[Species.COW].index(Action(Kind.MOVE, 0, -1))
    graze_idx = ACTIONS[Species.COW].index(Action(Kind.GRAZE))
    assert obs["action_mask"][north_idx] == 0
    assert obs["action_mask"][graze_idx] == 0
    _, rewards, term, trunc, info = env.step({cow_name: graze_idx})
    assert info[cow_name]["invalid_action"] is True
    assert not term[cow_name] and not trunc[cow_name]
    assert not any(e.kind == "graze" for e in env.world.last_events)
    assert rewards[cow_name] <= 0


def test_core_step_has_parallel_api_lifetime_semantics():
    env = QoxCoreEnv(scenario="forage", max_steps=2)
    observations, infos = env.reset(seed=99)
    assert set(observations) == set(env.agents) == set(infos)
    initial = env.possible_agents.copy()
    new_obs, rewards, term, trunc, infos = env.step({})
    assert set(rewards) == set(initial) == set(term) == set(trunc) == set(infos)
    assert all(not finished for finished in trunc.values())
    assert set(new_obs) == set(env.agents)
    new_obs, rewards, term, trunc, infos = env.step({})
    assert all(trunc.values())
    assert new_obs == {} and env.agents == []
    with pytest.raises(RuntimeError, match="episode finished"):
        env.step({})
    env.reset(seed=99, options={"options": 1})
    assert env.agents == initial


def test_ravager_bite_is_a_real_survival_threat():
    config = small(farmers=0, cows=1, nomads=0, scavengers=0, ravagers=1)
    world = World(config)
    cow = next(a for a in world.actors.values() if a.species == Species.COW)
    ravager = next(a for a in world.actors.values() if a.species == Species.RAVAGER)
    cow.x, cow.y = 4, 4
    ravager.x, ravager.y = 4, 5
    before = cow.health
    assert observe(world, ravager)["action_mask"][ACTIONS[Species.RAVAGER].index(
        Action(Kind.ATTACK))] == 1
    world.step({ravager.id: Action(Kind.ATTACK)})
    assert cow.health < before
    assert any(event.kind == "attack" and event.actor_id == ravager.id
               for event in world.last_events)
    world.check_invariants()


def test_core_reset_reproducible_with_scripted_threat():
    a = QoxCoreEnv(scenario="threat", max_steps=10, scripted_others=True)
    b = QoxCoreEnv(scenario="threat", max_steps=10, scripted_others=True)
    oa, _ = a.reset(seed=183)
    ob, _ = b.reset(seed=183)
    assert a.world.state_hash() == b.world.state_hash()
    for name in oa:
        np.testing.assert_array_equal(oa[name]["observation"], ob[name]["observation"])
    for _ in range(10):
        a.step({})
        b.step({})
        assert a.world.state_hash() == b.world.state_hash()


def test_pettingzoo_parallel_api_if_dependency_installed():
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test
    from qox.envs.parallel_env import QoxParallelEnv

    env = QoxParallelEnv(config=small(), max_steps=20)
    parallel_api_test(env, num_cycles=12)
    env.close()
