"""M5 extension: tests that signal information, not map geometry, carries the goal."""
from __future__ import annotations

import numpy as np
import pytest

from qox.envs.core import QoxCoreEnv
from qox.entities import Species
from qox.perception import FEATURES, COW_CALL_FEATURES, observe


def _env(seed: int) -> QoxCoreEnv:
    env = QoxCoreEnv(scenario="cow_signal", scripted_others=True,
                     cow_calls=True, max_steps=32)
    env.reset(seed=seed)
    return env


def test_signal_task_balances_direction_and_sign_and_hides_goal():
    cases = set()
    for seed in range(64):
        env = _env(seed)
        cow = next(a for a in env.world.actors.values() if a.species == Species.COW)
        goal = env.world.herd_goal
        cases.add((goal[0] - cow.x, goal[1] - cow.y, env.signal_kind))
        assert len(cases) <= 8
        assert all(np.isfinite(o["observation"]).all() for o in env._observations().values())
        assert all(t.grass == 0.01 for t in env.world.grid.tiles)
        assert tuple(observe(env.world, cow, cow_calls=True)["observation"][-6:]) == (0,) * 6
    assert len(cases) == 8
    assert len(FEATURES + COW_CALL_FEATURES) == 66


def test_call_is_perception_not_forced_action_and_flip_is_counterfactual():
    regular = _env(147)
    flipped = _env(147)
    flipped.flip_signal = True
    no_calls = _env(147)
    no_calls.allow_nomad_calls = False
    cow_name = next(n for n in regular.possible_agents if n.startswith("cow_"))
    before = regular.world.actors[regular.name_to_id[cow_name]].position
    observations = []
    for env in (regular, flipped, no_calls):
        states, _, _, _, _ = env.step({cow_name: 0})
        assert env.world.actors[env.name_to_id[cow_name]].position == before
        assert env.world.herd_goal == regular.world.herd_goal
        observations.append(states[cow_name]["observation"][-6:])
    assert (observations[0][2] > 0) != (observations[1][2] > 0)
    assert (observations[0][5] > 0) != (observations[1][5] > 0)
    assert list(observations[2]) == [0] * 6
    assert regular.world.calls and flipped.world.calls and not no_calls.world.calls


def test_cow_signal_training_and_checkpoint_evaluation(tmp_path):
    pytest.importorskip("torch")
    from qox.learning.cow_ppo import TrainOptions, train_cow, evaluate_cow, load_cow

    with pytest.raises(ValueError, match="cow_calls"):
        TrainOptions(scenario="cow_signal", cow_calls=False)
    checkpoint = tmp_path / "cow.pt"
    train_cow(TrainOptions(steps=64, worlds=2, rollout=64, max_steps=16,
                          epochs=1, scenario="cow_signal", cow_calls=True), checkpoint)
    assert load_cow(checkpoint).cow_calls
    result = evaluate_cow(checkpoint, scenario="cow_signal", episodes=2,
                          max_steps=16, mode="policy_flipped")
    assert 0 <= result["pasture_arrival_rate"] <= 1
