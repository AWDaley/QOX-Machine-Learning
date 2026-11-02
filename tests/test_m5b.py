"""M5b: controlled push/pull nomad curriculum and versioned checkpoints."""

from __future__ import annotations

import numpy as np
import pytest

from qox.actions import Action, Kind
from qox.entities import Species
from qox.envs.core import QoxCoreEnv
from qox.perception import (ACTIONS, NOMAD_FEATURES, NOMAD_RELATIONAL_FEATURES,
                            features_for, observe)


def _nomad(env):
    return next(name for name in env.possible_agents if name.startswith("nomad_"))


def test_stationary_signal_has_two_distinct_possible_targets_and_no_direct_forces():
    seen = set()
    for seed in range(32):
        env = QoxCoreEnv(scenario="nomad_signal", scripted_others=True,
                         cow_calls=True, nomad_v2=True, max_steps=8)
        obs, _ = env.reset(seed=seed)
        name = _nomad(env)
        cow = next(a for a in env.world.actors.values() if a.species == Species.COW)
        nomad = env.world.actors[env.name_to_id[name]]
        old_pos = cow.position
        seen.add((env.signal_kind, nomad.x - cow.x, nomad.y - cow.y))
        assert abs(cow.x - env.world.herd_goal[0]) + abs(cow.y - env.world.herd_goal[1]) == 2
        assert obs[name]["observation"].shape == (len(features_for(Species.NOMAD,
                                                                  nomad_v2=True)),)
        assert obs[name]["action_mask"][ACTIONS[Species.NOMAD].index(
            Action(Kind.MOVE, 1, 0))] == 0
        expected_idx = ACTIONS[Species.NOMAD].index(Action(env.signal_kind))
        _, _, _, _, _ = env.step({name: expected_idx})
        assert cow.position == old_pos  # Call must not move cattle directly.
        nomad_mask = env._observations()[name]["action_mask"]
        assert nomad_mask[ACTIONS[Species.NOMAD].index(Action(Kind.FRIENDLY_CALL))] == 0
        assert nomad_mask[ACTIONS[Species.NOMAD].index(Action(Kind.SCARY_CALL))] == 0
        for _ in range(3):
            env.step({name: 0})
        assert cow.id in env.reached_goal_ids
        env.world.check_invariants()
    assert len(seen) == 8  # 2 signal types x 4 relative directions


def test_m5b_counterfactual_signal_evaluation():
    pytest.importorskip("torch")
    from qox.learning.nomad_ppo import evaluate_nomad

    for mode, expected in (("scripted", 1.0), ("scripted_nocalls", 0.0),
                           ("scripted_flipped", 0.0)):
        report = evaluate_nomad(scenario="nomad_signal", episodes=12,
                                seed=81000, max_steps=8, mode=mode)
        assert report["mean_cows_in_pasture"] == expected
        assert report["friendly_target_rate"] in (0.0, 1.0)
        assert report["scary_target_rate"] in (0.0, 1.0)


def test_v2_herd_goal_geometry_does_not_reveal_hidden_cattle():
    env = QoxCoreEnv(scenario="nomad_signal", nomad_v2=True)
    env.reset(seed=142)
    name = _nomad(env)
    nomad = env.world.actors[env.name_to_id[name]]
    cow = next(a for a in env.world.actors.values() if a.species == Species.COW)
    state = observe(env.world, nomad, nomad_v2=True)["observation"]
    assert len(state) == 60 + len(NOMAD_FEATURES) + len(NOMAD_RELATIONAL_FEATURES)
    assert np.isclose(abs(state[-1]), 1.0)
    assert state[-1] == (1.0 if env.signal_kind == Kind.FRIENDLY_CALL else -1.0)
    cow.x, cow.y = 1, 1  # Beyond the nomad's vision.
    assert observe(env.world, nomad, nomad_v2=True)["observation"][-3:].tolist() == [0, 0, 0]
    assert observe(env.world, nomad)["observation"].shape == (72,)


def test_policy_schema_and_warmstart_transfer(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.learning.nomad_ppo import (NomadPolicy, NomadTrainOptions, action_names,
                                        evaluate_nomad, load_nomad, train_nomad)
    from qox.perception import NOMAD_OBSERVATION_VERSION

    legacy = tmp_path / "legacy.pt"
    torch.save({"species": "nomad", "obs_version": NOMAD_OBSERVATION_VERSION,
                "observation_features": features_for(Species.NOMAD),
                "actions": action_names(), "model_hidden": 16,
                "policy_state": NomadPolicy(16).state_dict()}, legacy)
    assert not load_nomad(legacy).nomad_v2

    stage1 = tmp_path / "stage1.pt"
    settings = NomadTrainOptions(steps=64, worlds=2, rollout=64, epochs=1,
                                 hidden=16, max_steps=8, scenario="nomad_signal",
                                 nomad_v2=True)
    train_nomad(settings, stage1)
    assert load_nomad(stage1).nomad_v2
    with pytest.raises(ValueError, match="version mismatch"):
        train_nomad(settings, tmp_path / "invalid.pt", init_checkpoint=legacy)

    stage2 = tmp_path / "stage2.pt"
    train_nomad(NomadTrainOptions(steps=64, worlds=2, rollout=64, epochs=1,
                                  hidden=16, max_steps=12, scenario="nomad_herd",
                                  nomad_v2=True), stage2, init_checkpoint=stage1)
    assert load_nomad(stage2).nomad_v2
    first = evaluate_nomad(stage1, episodes=3, seed=80000, scenario="nomad_signal",
                           max_steps=8, mode="policy_greedy")
    assert first == evaluate_nomad(stage1, episodes=3, seed=80000,
                                   scenario="nomad_signal", max_steps=8,
                                   mode="policy_greedy")


def test_frozen_cow_ensemble_evaluation_and_schema_rejection(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.learning.cow_ppo import CowPolicy
    from qox.learning.nomad_ppo import evaluate_nomad
    from qox.perception import COW_CALL_OBSERVATION_VERSION, OBSERVATION_VERSION

    def save(path, v2):
        model = CowPolicy(16, cow_calls=v2)
        torch.save({"obs_version": COW_CALL_OBSERVATION_VERSION if v2 else OBSERVATION_VERSION,
                    "observation_features": features_for(Species.COW, cow_calls=v2),
                    "actions": tuple(a.to_dict()["kind"] + (f"_{a.dx}_{a.dy}"
                                        if a.kind == Kind.MOVE else "")
                                     for a in ACTIONS[Species.COW]),
                    "model_hidden": 16, "policy_state": model.state_dict()}, path)

    v1 = tmp_path / "cow_v1.pt"
    v2 = tmp_path / "cow_v2.pt"
    save(v1, False)
    save(v2, True)
    with pytest.raises(ValueError, match="call-aware"):
        evaluate_nomad(mode="scripted", scenario="nomad_herd", episodes=1,
                       max_steps=6, cow_checkpoint=v1)
    report = evaluate_nomad(mode="scripted", scenario="nomad_herd", episodes=2,
                            max_steps=6, cow_checkpoint=v2)
    assert report["cow_controller"] == "learned_greedy"
    assert 0 <= report["mean_cows_in_pasture"] <= 3


def test_pettingzoo_v2_shape():
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test
    from qox.envs.parallel_env import parallel_env

    env = parallel_env(scenario="nomad_signal", nomad_v2=True, cow_calls=True,
                       max_steps=6)
    parallel_api_test(env, num_cycles=7)
