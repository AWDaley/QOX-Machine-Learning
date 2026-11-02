"""M5 herd communication, defense, and neural-policy regression tests."""

from dataclasses import replace

import numpy as np
import pytest

from qox.actions import Action, Kind
from qox.config import SimulationConfig
from qox.entities import Species
from qox.envs.core import QoxCoreEnv
from qox.perception import (ACTIONS, COW_CALL_FEATURES, FEATURES, NOMAD_FEATURES,
                            call_features, features_for, observe, legal_action_mask)
from qox.trace import replay, run
from qox.world import World


def config(**kw):
    base = SimulationConfig(width=14, height=12, farmers=0, cows=1, nomads=1,
                            scavengers=0, ravagers=0)
    return replace(base, **kw)


def get(world, species):
    return next(a for a in world.actors.values() if a.species == species)


def test_nomad_actions_and_backward_compatible_cow_schemas():
    world = World(config())
    cow, nomad = get(world, Species.COW), get(world, Species.NOMAD)
    assert features_for(Species.COW) == FEATURES and len(FEATURES) == 60
    assert features_for(Species.COW, cow_calls=True) == FEATURES + COW_CALL_FEATURES
    assert len(observe(world, cow)["observation"]) == 60
    assert len(observe(world, cow, cow_calls=True)["observation"]) == 66
    assert len(observe(world, nomad)["observation"]) == 60 + len(NOMAD_FEATURES)
    assert Action(Kind.FRIENDLY_CALL) in ACTIONS[Species.NOMAD]
    assert Action(Kind.SCARY_CALL) in ACTIONS[Species.NOMAD]
    assert Action(Kind.ATTACK) in ACTIONS[Species.NOMAD]
    assert Action(Kind.DASH, 1, 0) in ACTIONS[Species.NOMAD]
    assert np.isfinite(observe(world, nomad)["observation"]).all()


def test_signal_is_a_perception_not_forced_cattle_movement():
    world = World(config())
    cow, nomad = get(world, Species.COW), get(world, Species.NOMAD)
    cow.x, cow.y = 5, 5
    nomad.x, nomad.y = 3, 5
    cow_position = cow.position
    world.step({nomad.id: Action(Kind.FRIENDLY_CALL)})
    assert cow.position == cow_position
    fdx, fdy, strength, _, _, scary = call_features(world, cow)
    assert fdx < 0 and fdy == 0 and 0 < strength <= 1 and scary == 0
    assert observe(world, cow)["observation"].shape == (60,)
    new_obs = observe(world, cow, cow_calls=True)["observation"]
    assert new_obs[60:66] == pytest.approx([fdx, fdy, strength, 0, 0, 0])
    world.step({nomad.id: Action(Kind.SCARY_CALL)})
    assert cow.position == cow_position
    assert call_features(world, cow)[-1] > 0
    for _ in range(world.config.call_duration + 1):
        world.step({})
    assert call_features(world, cow) == (0,) * 6
    world.check_invariants()


def test_call_cost_cooldowns_and_masks_are_enforced_in_physics():
    world = World(config())
    nomad = get(world, Species.NOMAD)
    idx = ACTIONS[Species.NOMAD].index(Action(Kind.FRIENDLY_CALL))
    before_energy = nomad.energy
    world.step({nomad.id: Action(Kind.FRIENDLY_CALL)})
    assert nomad.energy < before_energy
    assert nomad.friendly_cooldown > 0
    assert legal_action_mask(world, nomad)[idx] == 0
    world.step({nomad.id: Action(Kind.FRIENDLY_CALL)})  # bypass mask attempt
    assert sum(e.kind == "friendly_call" for e in world.events) == 1
    for _ in range(world.config.call_cooldown + 1):
        world.step({})
    assert legal_action_mask(world, nomad)[idx] == 1
    nomad.stamina = 0
    assert legal_action_mask(world, nomad)[idx] == 0
    world.step({nomad.id: Action(Kind.FRIENDLY_CALL)})
    assert sum(e.kind == "friendly_call" for e in world.events) == 1


def test_nomad_dash_requires_open_two_tile_path_and_spends_stamina():
    world = World(config())
    nomad, cow = get(world, Species.NOMAD), get(world, Species.COW)
    nomad.x, nomad.y = 5, 5
    cow.x, cow.y = 6, 5
    dash = Action(Kind.DASH, 1, 0)
    idx = ACTIONS[Species.NOMAD].index(dash)
    assert legal_action_mask(world, nomad)[idx] == 0
    world.step({nomad.id: dash})
    assert nomad.position == (5, 5)  # Cannot jump over cattle.
    cow.x, cow.y = 9, 5
    stamina = nomad.stamina
    assert legal_action_mask(world, nomad)[idx] == 1
    world.step({nomad.id: dash})
    assert nomad.position == (7, 5)
    assert nomad.stamina < stamina
    assert any(e.kind == "dash" for e in world.last_events)
    with pytest.raises(ValueError):
        Action(Kind.DASH, 2, 0)


def test_nomad_melee_targets_ravager_not_cows_and_predator_remains_dangerous():
    world = World(config(ravagers=1))
    nomad = get(world, Species.NOMAD)
    cow = get(world, Species.COW)
    ravager = get(world, Species.RAVAGER)
    nomad.x, nomad.y = 6, 4
    ravager.x, ravager.y = 7, 4
    cow.x, cow.y = 6, 5
    old_health = ravager.health
    world.step({nomad.id: Action(Kind.ATTACK), ravager.id: Action(Kind.ATTACK)})
    assert ravager.health == pytest.approx(old_health - world.config.nomad_attack_damage)
    assert cow.health == 100  # Ravager chooses nearest lowest-health target: nomad.
    assert nomad.health < 100
    assert any(e.kind == "nomad_attack" for e in world.last_events)


def test_migration_seed_variation_and_calls_are_necessary_for_script_baseline():
    from qox.learning.nomad_ppo import evaluate_nomad
    targets = set()
    for seed in range(8):
        env = QoxCoreEnv(scenario="nomad_herd", scripted_others=True, cow_calls=True)
        env.reset(seed=seed)
        targets.add(env.world.herd_goal)
        for state in env._observations().values():
            assert np.isfinite(state["observation"]).all()
            assert (state["observation"] <= 1).all()
            assert (state["observation"] >= -1).all()
    assert len(targets) > 2
    baseline = evaluate_nomad(mode="scripted", episodes=8, seed=150000,
                              max_steps=96)
    no_calls = evaluate_nomad(mode="scripted_nocalls", episodes=8, seed=150000,
                              max_steps=96)
    assert baseline["mean_cows_in_pasture"] > no_calls["mean_cows_in_pasture"]
    assert no_calls["mean_calls"] == 0
    assert baseline["mean_calls"] > 0


def test_call_sensitive_training_env_deterministic_and_ablated_masks():
    left = QoxCoreEnv(scenario="nomad_herd", scripted_others=True, cow_calls=True)
    right = QoxCoreEnv(scenario="nomad_herd", scripted_others=True, cow_calls=True)
    disabled = QoxCoreEnv(scenario="nomad_herd", cow_calls=True, allow_nomad_calls=False)
    a, _ = left.reset(seed=733)
    b, _ = right.reset(seed=733)
    c, _ = disabled.reset(seed=733)
    assert left.world.state_hash() == right.world.state_hash() == disabled.world.state_hash()
    name = next(n for n in a if n.startswith("nomad_"))
    for kind in (Kind.FRIENDLY_CALL, Kind.SCARY_CALL):
        index = ACTIONS[Species.NOMAD].index(Action(kind))
        assert c[name]["action_mask"][index] == 0
    for _ in range(12):
        left.step({name: 0})
        right.step({name: 0})
        assert left.world.state_hash() == right.world.state_hash()


def test_nomad_training_checkpoint_cli_and_trace(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.nomad_ppo import (NomadPolicy, NomadTrainOptions, train_nomad,
                                        evaluate_nomad, load_nomad)
    checkpoint = tmp_path / "nomad.pt"
    report = train_nomad(NomadTrainOptions(steps=128, worlds=2, rollout=64,
                                           epochs=1, max_steps=24, seed=104), checkpoint)
    assert report["nomad_steps"] == 128
    model = load_nomad(checkpoint)
    assert model.actor.out_features == len(ACTIONS[Species.NOMAD])
    expected = evaluate_nomad(checkpoint, episodes=2, seed=26001, max_steps=24)
    assert expected == evaluate_nomad(checkpoint, episodes=2, seed=26001, max_steps=24)
    world, trace = run(config(cows=2, nomads=1), ticks=15, mode="learned-nomad",
                       checkpoint=checkpoint, record=True)
    assert replay(trace).state_hash() == world.state_hash()
    assert main(["benchmark-nomad", "--checkpoint", str(checkpoint),
                 "--episodes", "2", "--max-steps", "24",
                 "--report", str(tmp_path / "benchmark.json")]) == 0
    assert (tmp_path / "benchmark.json").exists()
    data = torch.load(checkpoint, weights_only=True)
    data["actions"] = ("incompatible",)
    broken = tmp_path / "broken.pt"
    torch.save(data, broken)
    with pytest.raises(ValueError, match="schema"):
        load_nomad(broken)
    assert isinstance(NomadPolicy(16), torch.nn.Module)


def test_call_aware_cow_train_and_legacy_checkpoint_contract(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.learning.cow_ppo import CowPolicy, TrainOptions, train_cow, load_cow, evaluate_cow
    legacy = tmp_path / "legacy.pt"
    call_aware = tmp_path / "calls.pt"
    train_cow(TrainOptions(steps=64, worlds=2, rollout=64, epochs=1,
                          max_steps=16, seed=117), legacy)
    old_model = load_cow(legacy)
    assert old_model.cow_calls is False
    assert old_model.encoder[0].in_features == 60
    assert old_model.actor.out_features == len(ACTIONS[Species.COW])
    train_cow(TrainOptions(steps=64, worlds=2, rollout=64, epochs=1,
                          max_steps=16, seed=118, scenario="nomad_herd",
                          cow_calls=True), call_aware)
    new_model = load_cow(call_aware)
    assert new_model.cow_calls is True
    assert new_model.encoder[0].in_features == 66
    assert evaluate_cow(call_aware, episodes=2, seed=160021,
                        max_steps=24, scenario="nomad_herd") == evaluate_cow(
        call_aware, episodes=2, seed=160021, max_steps=24, scenario="nomad_herd")
    results = evaluate_cow(call_aware, episodes=2, seed=160021,
                           max_steps=24, scenario="nomad_herd", mode="policy_nocalls")
    assert results["mean_calls"] == 0
    data = torch.load(call_aware, weights_only=True)
    data["observation_features"] = ("incorrect",)
    invalid = tmp_path / "invalid-cow.pt"
    torch.save(data, invalid)
    with pytest.raises(ValueError, match="schema"):
        load_cow(invalid)
    with pytest.raises(ValueError, match="requires cow_calls"):
        TrainOptions(scenario="nomad_herd", cow_calls=False)
    assert isinstance(CowPolicy(16, cow_calls=True), torch.nn.Module)


def test_combined_herd_replay_with_two_frozen_networks(tmp_path):
    pytest.importorskip("torch")
    from qox.learning.cow_ppo import TrainOptions, train_cow
    from qox.learning.nomad_ppo import NomadTrainOptions, train_nomad
    nomad_file, cow_file = tmp_path / "nomad.pt", tmp_path / "cow.pt"
    train_nomad(NomadTrainOptions(steps=64, worlds=2, rollout=64,
                                  epochs=1, max_steps=16), nomad_file)
    train_cow(TrainOptions(steps=64, worlds=2, rollout=64, epochs=1,
                          max_steps=16, scenario="nomad_herd", cow_calls=True), cow_file)
    cfg = config(cows=3, nomads=2, ravagers=1)
    world, trace = run(cfg, 14, mode="learned-herd", record=True,
                       checkpoint=nomad_file, cow_checkpoint=cow_file)
    assert replay(trace).state_hash() == world.state_hash()
    with pytest.raises(ValueError, match="--cow-checkpoint"):
        run(cfg, 1, mode="learned-herd", checkpoint=nomad_file)


def test_nomad_pettingzoo_v2_contract_if_installed():
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test
    from qox.envs.parallel_env import QoxParallelEnv
    env = QoxParallelEnv(max_steps=20, scenario="nomad_herd", cow_calls=True)
    assert env.observation_space("cow_1")["observation"].shape == (66,)
    assert env.observation_space("nomad_4")["observation"].shape == (72,)
    parallel_api_test(env, num_cycles=12)
    env.close()
