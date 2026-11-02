"""M9 population simulation, dynamic-agent contract and research reproducibility."""

from dataclasses import replace

import pytest

from qox.actions import Action, Kind
from qox.agents import scripted_actions
from qox.config import SimulationConfig
from qox.entities import Species
from qox.envs.core import QoxCoreEnv
from qox.evaluation.population import run_study
from qox.perception import ACTIONS, legal_action_mask
from qox.population import LIFESPAN, mature_age
from qox.trace import Trace, replay
from qox.world import World


def config(**kwargs):
    defaults = dict(width=12, height=12, seed=193, farmers=0, cows=1,
                    nomads=0, scavengers=0, ravagers=0,
                    reproduction_enabled=True, reproduction_interval=1,
                    max_cows=3, births_per_species_limit=3)
    return SimulationConfig(**(defaults | kwargs))


def test_birth_is_bounded_local_and_energetically_costly():
    world = World(config())
    parent = next(iter(world.actors.values()))
    assert parent.age == mature_age(Species.COW)
    parent.energy = 100
    old_energy = parent.energy
    world.step({})
    assert world.birth_totals[Species.COW] == 1
    child = next(a for a in world.actors.values() if a.id != parent.id)
    assert child.parent_id == parent.id and child.id > parent.id
    assert child.age == 0 and child.energy == world.config.offspring_energy
    assert abs(parent.x - child.x) + abs(parent.y - child.y) == 1
    assert parent.energy + child.energy < old_energy
    assert any(e.kind == 'birth' and e.actor_id == child.id for e in world.last_events)
    assert world.summary()['juveniles']['cow'] == 1
    world.check_invariants()


def test_birth_does_not_happen_if_parent_is_juvenile_or_energy_low():
    world = World(config())
    parent = next(iter(world.actors.values()))
    parent.energy = 100
    parent.age = mature_age(Species.COW) - 3
    world.step({})
    assert world.birth_totals[Species.COW] == 0
    parent.age = mature_age(Species.COW)
    parent.energy = 79
    world.step({})
    assert world.birth_totals[Species.COW] == 0
    parent.energy = 100
    world.step({})
    assert world.birth_totals[Species.COW] == 1


def test_cooldown_species_cap_and_lifetime_limit_prevent_runaway_growth():
    world = World(config(max_cows=2, births_per_species_limit=1))
    parent = next(iter(world.actors.values()))
    parent.energy = 100
    world.step()
    child = next(a for a in world.actors.values() if a.id != parent.id)
    for _ in range(4):
        parent.energy = 100
        parent.last_birth_tick = -100000
        world.step()
    assert world.birth_totals[Species.COW] == 1
    child.health = 0
    world.step()
    assert child.id not in world.actors
    parent.energy = 100
    parent.last_birth_tick = -100000
    world.step()
    assert world.birth_totals[Species.COW] == 1  # lifetime budget is not refunded


def test_no_birth_when_adjacent_tiles_occupied_or_invalid():
    world = World(config())
    parent = next(iter(world.actors.values()))
    parent.energy = 100
    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        x, y = parent.x + dx, parent.y + dy
        if world.grid.contains(x, y):
            world.grid.get(x, y).passable = False
    world.step()
    assert world.birth_totals[Species.COW] == 0


def test_juvenile_cannot_charge_or_fight_and_cow_does_not_give_milk():
    world = World(config(cows=1, nomads=1, ravagers=1, max_nomads=4,
                         max_ravagers=4))
    for a in world.actors.values():
        a.age = 0
        a.energy = 100
        mask = legal_action_mask(world, a)
        if a.species == Species.NOMAD:
            for kind in (Kind.ATTACK, Kind.FRIENDLY_CALL, Kind.SCARY_CALL):
                assert mask[ACTIONS[a.species].index(Action(kind))] == 0
        elif a.species == Species.RAVAGER:
            assert mask[ACTIONS[a.species].index(Action(Kind.ATTACK))] == 0
            assert all(not mask[i] for i, action in enumerate(ACTIONS[a.species])
                       if action.kind == Kind.CHARGE)
    cow = next(a for a in world.actors.values() if a.species == Species.COW)
    cow.x, cow.y = 2, 2
    world.grid.get(2, 2).grass = 8
    world.step({cow.id: Action(Kind.GRAZE)})
    assert cow.milk == 0


def test_adult_cattle_give_milk_and_juveniles_mature():
    world = World(config())
    cow = next(iter(world.actors.values()))
    cow.age = mature_age(Species.COW) - 1
    cow.x, cow.y = 2, 2
    world.grid.get(2, 2).grass = 8
    world.step({cow.id: Action(Kind.GRAZE)})
    assert cow.age == mature_age(Species.COW) and cow.milk == 0
    world.step({cow.id: Action(Kind.GRAZE)})
    assert cow.milk > 0


def test_death_cause_distinguishes_starvation_from_aging():
    world = World(config())
    cow = next(iter(world.actors.values()))
    cow.health = 0.4
    cow.age = LIFESPAN[Species.COW] + 1
    cow.energy = 99
    world.step()
    assert cow.id not in world.actors
    assert any(e.kind == 'death_cause' and e.detail == 'age' for e in world.last_events)


def test_dynamic_core_announces_newborn_once_and_never_recycles_name():
    env = QoxCoreEnv(config=config(max_cows=3, births_per_species_limit=2),
                     max_steps=10)
    original_name = env.agents[0]
    env.world.actors[env.name_to_id[original_name]].energy = 100
    obs, rewards, term, trunc, infos = env.step({})
    birth_name = 'cow_birth_0001'
    assert birth_name in env.possible_agents and birth_name in env.agents
    assert birth_name in obs and not term[birth_name] and not trunc[birth_name]
    assert rewards[birth_name] == 0 and infos[birth_name]['newborn']
    old_id = env.name_to_id[birth_name]
    env.world.actors[old_id].health = 0
    env.step({})
    assert birth_name not in env.agents
    parent = env.world.actors[env.name_to_id[original_name]]
    parent.energy = 100
    parent.last_birth_tick = -100000
    env.step({})
    assert 'cow_birth_0002' in env.agents
    assert env.name_to_id['cow_birth_0002'] != old_id
    assert env.name_to_id['cow_birth_0001'] == old_id


def test_newborn_inherits_home_or_band_and_is_not_teleported_to_refuge():
    world = World(config(farmers=1, cows=0, nomads=1, max_farmers=2,
                         max_nomads=2))
    for actor in world.actors.values():
        actor.energy = 100
    world.step()
    for child in (a for a in world.actors.values() if a.parent_id is not None):
        parent = next(a for a in world.actors.values() if a.id == child.parent_id)
        assert child.home_id == parent.home_id and child.band_id == parent.band_id
        assert not child.sheltered


def test_birth_trace_replay_is_exact_and_id_counter_is_monotonic():
    world = World(config())
    trace = Trace(world.config.as_dict())
    for _ in range(45):
        orders = scripted_actions(world)
        world.step(orders)
        trace.append(world, orders)
    assert replay(trace).state_hash() == world.state_hash()
    assert world.next_id > max(world.actors, default=0)


def test_old_configuration_can_still_be_simulated_without_births():
    c = replace(config(), reproduction_enabled=False)
    world = World(c)
    before = list(world.actors)
    for _ in range(200):
        world.step({})
    assert not any(e.kind == 'birth' for e in world.events)
    assert set(world.actors) <= set(before)
    assert 'next_id' not in world.snapshot()
    assert 'parent_id' not in world.snapshot()['actors'][0]
    assert 'reproduction_enabled' not in world.snapshot()['config']


def test_study_is_deterministic_and_aggregates_multiple_seeds():
    c = config(max_cows=4, births_per_species_limit=2)
    params = dict(ticks=12, seeds=2, initial_seed=901, sample_every=4)
    a = run_study(c, mode='scripted', **params)
    b = run_study(c, mode='scripted', **params)
    assert a == b
    assert len(a['outcomes']) == 2
    assert [r['seed'] for r in a['outcomes']] == [901, 902]
    assert a['outcomes'][0]['time_series'][-1]['tick'] == 12
    assert len(a['outcomes'][0]['final_state_hash']) == 64
    idle = run_study(replace(c, reproduction_enabled=False), mode='idle', **params)
    assert all(not any(out['births'].values()) for out in idle['outcomes'])


def test_population_study_rejects_invalid_or_unmatched_checkpoint_modes():
    with pytest.raises(ValueError, match='positive'):
        run_study(config(), ticks=0, seeds=2, initial_seed=1, sample_every=1)
    with pytest.raises(ValueError, match='require --mode learned'):
        run_study(config(), ticks=3, seeds=1, initial_seed=1, sample_every=1,
                  checkpoints={'cow': 'missing.pt'})


def test_config_rejects_unfunded_birth_and_initial_cap_overflow():
    with pytest.raises(ValueError, match='cost more energy'):
        config(offspring_energy=50, reproduction_energy_cost=30)
    with pytest.raises(ValueError, match='exceeds'):
        config(max_cows=0)


def test_pettingzoo_dynamic_birth_space_if_available():
    pytest.importorskip('pettingzoo')
    from qox.envs.parallel_env import QoxParallelEnv
    env = QoxParallelEnv(config=config(births_per_species_limit=2), max_steps=8)
    observations, _ = env.reset(seed=12)
    assert len(env.possible_agents) == len(observations) + 2 * len(Species)
    name = env.agents[0]
    env.core.world.actors[env.core.name_to_id[name]].energy = 100
    observations, _, _, _, _ = env.step({})
    newborn = 'cow_birth_0001'
    assert newborn in observations and env.action_space(newborn).n > 1
    assert env.observation_space(newborn).contains(observations[newborn])
    env.close()


def test_cli_population_study_creates_report_and_fair_ablation(tmp_path):
    import json
    from qox.cli import main

    report = tmp_path / 'm9.json'
    assert main(['population-study', '--ticks', '12', '--seeds', '2',
                 '--seed', '331', '--sample-every', '4', '--mode', 'scripted',
                 '--without-births', '--report', str(report)]) == 0
    result = json.loads(report.read_text())
    assert result['config']['population_dynamics_enabled'] is True
    assert result['config']['reproduction_enabled'] is False
    assert len(result['outcomes']) == 2


def test_pettingzoo_parallel_dynamic_contract_if_available():
    pytest.importorskip('pettingzoo')
    from pettingzoo.test import parallel_api_test
    from qox.envs.parallel_env import QoxParallelEnv
    env = QoxParallelEnv(config=config(births_per_species_limit=2), max_steps=12)
    parallel_api_test(env, num_cycles=10)
    env.close()
