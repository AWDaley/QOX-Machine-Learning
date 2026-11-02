"""M8 festivals: resource conservation, quorum, danger, signals and learning contracts."""

from dataclasses import replace

import pytest

from qox.actions import Action, Kind
from qox.agents import scripted_actions
from qox.config import SimulationConfig
from qox.entities import Species
from qox.envs.core import QoxCoreEnv
from qox.festivals import FestivalKind, can_start, festival_cue
from qox.perception import (ACTIONS, action_catalog, features_for, legal_action_mask,
                            observe, observation_size)
from qox.trace import Trace, replay
from qox.world import World


def festival_env(scenario: str) -> QoxCoreEnv:
    return QoxCoreEnv(scenario=scenario, max_steps=24, scripted_others=False)


def actor_of(world, species):
    return next(a for a in world.actors.values() if a.species == species)


def prepare(world: World, mode: FestivalKind):
    farmer = next((a for a in world.actors.values()
                   if a.species == Species.FARMER), None)
    nomad = next((a for a in world.actors.values()
                  if a.species == Species.NOMAD), None)
    if farmer and mode != FestivalKind.NOMAD:
        world.step({farmer.id: Action(Kind.CONTRIBUTE_FESTIVAL)})
    if nomad and mode != FestivalKind.FARMER:
        world.step({nomad.id: Action(Kind.CONTRIBUTE_FESTIVAL)})
    return farmer if mode != FestivalKind.NOMAD else nomad


def start(world: World, mode: FestivalKind):
    owner = prepare(world, mode)
    assert owner is not None
    assert can_start(world, owner, mode)
    kind = {FestivalKind.FARMER: Kind.START_FARMER_FESTIVAL,
            FestivalKind.NOMAD: Kind.START_NOMAD_FESTIVAL,
            FestivalKind.COMBINED: Kind.START_COMBINED_FESTIVAL}[mode]
    world.step({owner.id: Action(kind)})
    assert len(world.active_festivals) == 1
    return owner


@pytest.mark.parametrize('scenario,mode', [
    ('festival_farmer', FestivalKind.FARMER),
    ('festival_nomad', FestivalKind.NOMAD),
    ('festival_combined', FestivalKind.COMBINED),
])
def test_festival_contributions_and_exact_resource_consumption(scenario, mode):
    world = festival_env(scenario).world
    assert world.config.festival_enabled
    owner = prepare(world, mode)
    assert owner is not None
    site = (world.camps[0] if mode == FestivalKind.NOMAD else world.settlements[0])
    grain, milk = (world.config.festival_farmer_grain, 0) if mode == FestivalKind.FARMER else (
        (0, world.config.festival_nomad_milk) if mode == FestivalKind.NOMAD else
        (world.config.festival_combined_grain, world.config.festival_combined_milk))
    before_food = getattr(site, 'festival_food', 0.0)
    before_milk = site.festival_milk
    survival_before = world.settlements[0].stored_food
    action = {FestivalKind.FARMER: Kind.START_FARMER_FESTIVAL,
              FestivalKind.NOMAD: Kind.START_NOMAD_FESTIVAL,
              FestivalKind.COMBINED: Kind.START_COMBINED_FESTIVAL}[mode]
    world.step({owner.id: Action(action)})
    assert site.festival_milk == pytest.approx(before_milk - milk)
    assert getattr(site, 'festival_food', 0) == pytest.approx(before_food - grain)
    assert world.settlements[0].stored_food == pytest.approx(survival_before)
    assert len(world.active_festivals) == 1
    committed = next(iter(world.active_festivals.values()))
    assert committed.grain_spent == grain and committed.milk_spent == milk
    world.step({owner.id: Action(action)})
    assert len(world.active_festivals) == 1  # No double spending or double start.
    assert site.festival_milk == pytest.approx(before_milk - milk)


@pytest.mark.parametrize('scenario,mode', [
    ('festival_farmer', FestivalKind.FARMER),
    ('festival_nomad', FestivalKind.NOMAD),
    ('festival_combined', FestivalKind.COMBINED),
])
def test_all_three_festivals_complete_after_duration(scenario, mode):
    world = festival_env(scenario).world
    start(world, mode)
    for _ in range(world.config.festival_duration):
        world.step()
    assert world.summary()['festival_successes'] == 1
    assert world.summary()['festival_failures'] == 0
    assert world.summary()['active_festivals'] == 0
    assert world.festival_results[-1].kind == mode
    assert world.festival_results[-1].reason == 'completed'
    assert sum(e.kind == 'festival_success' for e in world.events) == 1


def test_cow_required_for_nomad_and_combined_but_not_farmer():
    for scenario, mode in [('festival_nomad', FestivalKind.NOMAD),
                           ('festival_combined', FestivalKind.COMBINED)]:
        world = festival_env(scenario).world
        starter = prepare(world, mode)
        cow = actor_of(world, Species.COW)
        cow.x, cow.y = world.grid.width - 1, world.grid.height - 1
        for other in world.actors.values():
            if other.species == Species.COW:
                other.x, other.y = world.grid.width - 1, world.grid.height - other.id % 3 - 2
        assert not can_start(world, starter, mode)
    farmer_world = festival_env('festival_farmer').world
    farmer = prepare(farmer_world, FestivalKind.FARMER)
    assert can_start(farmer_world, farmer, FestivalKind.FARMER)


def test_attendance_loss_causes_failure_without_refund():
    world = festival_env('festival_farmer').world
    owner = start(world, FestivalKind.FARMER)
    home = world.home(owner)
    assert home is not None
    for farmer in (a for a in world.actors.values() if a.species == Species.FARMER):
        farmer.x, farmer.y = world.grid.width - 1, world.grid.height - 1 - farmer.id
    world.step()
    assert world.summary()['festival_failures'] == 1
    assert not world.active_festivals
    assert home.festival_food == pytest.approx(0.5)
    assert world.festival_results[-1].reason == 'attendance_lost'


def test_ravager_at_festival_site_causes_failure():
    world = festival_env('festival_ravager').world
    start(world, FestivalKind.COMBINED)
    predator = actor_of(world, Species.RAVAGER)
    home = world.settlements[0]
    predator.x, predator.y = home.x, home.y + 1
    # Move the previous resident to a free place to preserve occupancy.
    for other in world.actors.values():
        if other.id != predator.id and other.position == predator.position:
            other.x, other.y = home.x - 1, home.y - 1
    world.step()
    assert world.festival_results[-1].reason == 'predator_at_site'
    assert world.summary()['festival_successes'] == 0


def test_no_global_cue_outside_radius_and_no_predator_spawns():
    world = festival_env('festival_ravager').world
    predator = actor_of(world, Species.RAVAGER)
    before_ids = set(world.actors)
    assert festival_cue(world, predator) == (0, 0, 0)
    start(world, FestivalKind.COMBINED)
    cue = festival_cue(world, predator)
    assert cue[2] > 0 and cue[0] < 0
    assert set(world.actors) == before_ids
    predator.x, predator.y = world.grid.width - 1, world.grid.height - 1
    assert festival_cue(world, predator) == (0, 0, 0)


def test_festival_grain_cannot_be_eaten_as_survival_stock():
    world = festival_env('festival_farmer').world
    farmer = actor_of(world, Species.FARMER)
    home = world.home(farmer)
    assert home is not None
    world.step({farmer.id: Action(Kind.CONTRIBUTE_FESTIVAL)})
    home.stored_food = 0
    farmer.carried_food = 0
    farmer.energy = 10
    world.step({farmer.id: Action(Kind.EAT)})
    assert farmer.energy < 11
    assert home.festival_food > 0


def test_cooldown_prevents_repeated_festivals_even_if_resources_return():
    world = festival_env('festival_farmer').world
    farmer = start(world, FestivalKind.FARMER)
    home = world.home(farmer)
    assert home is not None
    home.festival_food = 5.0
    assert not can_start(world, farmer, FestivalKind.FARMER)
    for _ in range(world.config.festival_duration + 1):
        world.step()
    assert not can_start(world, farmer, FestivalKind.FARMER)
    assert world.summary()['festival_successes'] == 1


def test_legacy_observation_and_action_shapes_are_unchanged():
    world = World(SimulationConfig(festival_enabled=True))
    for species in Species:
        actor = actor_of(world, species)
        legacy = observe(world, actor)
        assert len(legacy['observation']) == observation_size(species)
        assert len(legacy['action_mask']) == len(ACTIONS[species])
        assert len(action_catalog(species)) == len(ACTIONS[species])
        festive = observe(world, actor, festivals=True)
        assert len(festive['observation']) == len(features_for(species, festivals=True))
        assert len(festive['action_mask']) == len(action_catalog(species, festivals=True))
        assert len(legal_action_mask(world, actor, festivals=True)) == len(festive['action_mask'])


def test_action_masks_block_start_before_contribution_then_allow():
    env = festival_env('festival_combined')
    farmer = actor_of(env.world, Species.FARMER)
    cat = action_catalog(Species.FARMER, festivals=True)
    idx = cat.index(Action(Kind.START_COMBINED_FESTIVAL))
    assert not observe(env.world, farmer, festivals=True)['action_mask'][idx]
    prepare(env.world, FestivalKind.COMBINED)
    assert observe(env.world, farmer, festivals=True)['action_mask'][idx]


@pytest.mark.parametrize('scenario', ['festival_farmer', 'festival_nomad', 'festival_combined'])
def test_scripted_reference_can_complete_each_type(scenario):
    env = festival_env(scenario)
    for _ in range(13):
        env.world.step(scripted_actions(env.world))
    assert env.world.summary()['festival_successes'] >= 1


def test_deterministic_festival_action_trace_replays():
    env = festival_env('festival_combined')
    # Replay starts from natural World, not scenario-modified state. Replay
    # therefore uses a seeded natural world with actor scripts instead.
    natural = World(replace(env.world.config, seed=7654))
    trace = Trace(natural.config.as_dict())
    for _ in range(90):
        actions = scripted_actions(natural)
        natural.step(actions)
        trace.append(natural, actions)
    loaded = replay(trace)
    assert loaded.state_hash() == natural.state_hash()


def test_festival_scenario_env_rewards_actual_completion():
    env = QoxCoreEnv(scenario='festival_farmer', max_steps=16, scripted_others=True)
    obs, _ = env.reset(seed=23)
    farmer = next(name for name in obs if name.startswith('farmer_'))
    catalog = action_catalog(Species.FARMER, festivals=True)
    contribution = catalog.index(Action(Kind.CONTRIBUTE_FESTIVAL))
    started = catalog.index(Action(Kind.START_FARMER_FESTIVAL))
    scores = []
    for action in [contribution, started] + [0] * 10:
        obs, rewards, _, _, infos = env.step({farmer: action})
        scores.append(rewards[farmer])
        assert 'reward_components' in infos[farmer]
    assert env.world.summary()['festival_successes'] == 1
    assert max(scores) >= 7.9


def test_parallel_api_if_installed():
    pytest.importorskip('pettingzoo')
    from qox.envs.parallel_env import QoxParallelEnv
    env = QoxParallelEnv(scenario='festival_farmer', max_steps=9)
    obs, _ = env.reset(seed=12)
    for name in env.agents:
        assert env.observation_space(name).contains(obs[name])
        assert env.action_space(name).n == len(obs[name]['action_mask'])
    env.close()


def test_scripted_predator_investigates_live_festival_noise_not_distant_preknowledge():
    world = festival_env('festival_ravager').world
    predator = actor_of(world, Species.RAVAGER)
    assert scripted_actions(world)[predator.id].kind == Kind.REST
    start(world, FestivalKind.COMBINED)
    first_move = scripted_actions(world)[predator.id]
    assert first_move == Action(Kind.MOVE, -1, 0)
    original_distance = abs(predator.x - world.settlements[0].x)
    world.step({predator.id: first_move})
    assert abs(predator.x - world.settlements[0].x) < original_distance


def test_festival_cue_is_opt_in_for_old_ravager_neural_models():
    world = festival_env('festival_ravager').world
    predator = actor_of(world, Species.RAVAGER)
    start(world, FestivalKind.COMBINED)
    old = dict(zip(features_for(Species.RAVAGER),
                   observe(world, predator)['observation'], strict=True))
    new = dict(zip(features_for(Species.RAVAGER, festivals=True),
                   observe(world, predator, festivals=True)['observation'], strict=True))
    assert 'festival_noise_strength' not in old
    assert new['festival_noise_strength'] > 0
    assert new['festival_noise_dx'] < 0


def test_festival_checkpoint_smoke(tmp_path):
    pytest.importorskip('torch')
    from qox.learning.festival_ppo import (FestivalTrainOptions, evaluate_festival,
                                           load_festival, train_festival)
    path = tmp_path / 'festival.pt'
    train_festival(FestivalTrainOptions(scenario='festival_farmer',
                                       steps=64, worlds=2, rollout=32,
                                       max_steps=8, epochs=1), path)
    model, scenario = load_festival(path)
    assert scenario == 'festival_farmer'
    assert model.species == Species.FARMER
    result = evaluate_festival(path, scenario='festival_farmer',
                               episodes=2, max_steps=8, mode='policy_greedy')
    assert result['episodes'] == 2
    with pytest.raises(ValueError, match='curriculum'):
        evaluate_festival(path, scenario='festival_nomad', episodes=1)


def test_learned_festival_replay(tmp_path):
    pytest.importorskip('torch')
    from qox.learning.festival_ppo import FestivalTrainOptions, train_festival
    path = tmp_path / 'festival.pt'
    train_festival(FestivalTrainOptions(scenario='festival_farmer', steps=64,
                                       worlds=2, rollout=32, max_steps=8,
                                       epochs=1), path)
    config = SimulationConfig(seed=19, festival_enabled=True)
    from qox.trace import run
    world, trace = run(config, 6, mode='learned-festival', checkpoint=path, record=True)
    assert trace is not None
    assert replay(trace).state_hash() == world.state_hash()


def test_festival_cli_entrypoints(tmp_path, capsys):
    from qox.cli import main
    assert main(['festival-demo', '--scenario', 'festival_farmer',
                 '--ticks', '12', '--report', str(tmp_path / 'result.json')]) == 0
    assert (tmp_path / 'result.json').is_file()
    output = capsys.readouterr().out
    assert 'festival_successes' in output


def test_same_tick_contribution_cannot_fund_simultaneous_start():
    world = festival_env('festival_farmer').world
    first, second = sorted((a for a in world.actors.values()
                            if a.species == Species.FARMER), key=lambda a: a.id)
    world.step({first.id: Action(Kind.CONTRIBUTE_FESTIVAL),
                second.id: Action(Kind.START_FARMER_FESTIVAL)})
    assert len(world.active_festivals) == 0
    assert can_start(world, second, FestivalKind.FARMER)
    world.step({second.id: Action(Kind.START_FARMER_FESTIVAL)})
    assert len(world.active_festivals) == 1


def test_simultaneous_start_proposals_choose_only_one_event_per_site():
    world = festival_env('festival_farmer').world
    first, second = sorted((a for a in world.actors.values()
                            if a.species == Species.FARMER), key=lambda a: a.id)
    world.step({first.id: Action(Kind.CONTRIBUTE_FESTIVAL)})
    world.step({first.id: Action(Kind.START_FARMER_FESTIVAL),
                second.id: Action(Kind.START_FARMER_FESTIVAL)})
    assert len(world.active_festivals) == 1
    assert sum(e.kind == 'festival_started' for e in world.events) == 1
    assert world.settlements[0].festival_food == pytest.approx(0.5)
