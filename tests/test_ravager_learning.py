"""Ravager learning pipeline and saved network contracts."""

import importlib.util

import pytest

from qox.actions import Action, Kind
from qox.entities import Species
from qox.envs.core import QoxCoreEnv
from qox.perception import ACTIONS


@pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                    reason="Optional machine-learning dependency missing")
def test_ravager_ppo_checkpoint_eval_and_inference(tmp_path):
    from qox.config import SimulationConfig
    from qox.learning.inference import RavagerController
    from qox.learning.ravager_ppo import (RavagerTrainOptions, evaluate_ravager,
                                          load_ravager, train_ravager)
    from qox.trace import run, replay

    checkpoint = tmp_path / "ravager.pt"
    result = train_ravager(RavagerTrainOptions(steps=128, worlds=2,
                                                rollout=64, epochs=1,
                                                max_steps=16), checkpoint)
    assert result["steps"] == 128 and checkpoint.exists()
    model = load_ravager(checkpoint)
    assert model.encoder[0].in_features == 82
    controller = RavagerController(checkpoint)
    world, trace = run(SimulationConfig(seed=2), ticks=8, mode="learned-ravager",
                       record=True, checkpoint=checkpoint)
    assert trace is not None and world.state_hash() == replay(trace).state_hash()
    predator = next(a for a in world.actors.values() if a.species == Species.RAVAGER)
    assert len(controller.probabilities(world, predator.id)) == len(ACTIONS[Species.RAVAGER])
    report = evaluate_ravager(checkpoint, episodes=2, max_steps=16,
                              scenario="ravager_ambush", mode="policy_greedy")
    assert len(report["results"]) == 2
    assert 0 <= report["mean_charge_hits"] <= 16


def test_ravager_train_validation():
    pytest.importorskip("torch", reason="Optional machine-learning dependency missing")
    from qox.learning.ravager_ppo import RavagerTrainOptions
    with pytest.raises(ValueError):
        RavagerTrainOptions(steps=-1)
    with pytest.raises(ValueError):
        RavagerTrainOptions(scenario="wrong")


def test_recovery_masks_only_stationary_actions():
    env = QoxCoreEnv(scenario="ravager_ambush", max_steps=12)
    obs, _ = env.reset(seed=41)
    name = next(name for name in obs if name.startswith("ravager_"))
    index = ACTIONS[Species.RAVAGER].index(Action(Kind.CHARGE, 1, 0))
    assert obs[name]["action_mask"][index]
    obs, _, _, _, _ = env.step({name: index})
    assert obs[name]["action_mask"].sum() == 1
    obs, _, _, _, _ = env.step({name: 0})
    assert obs[name]["action_mask"].sum() <= 2


def test_ravager_fixed_scenario_parallel_adapter():
    if (importlib.util.find_spec("pettingzoo") is None
            or importlib.util.find_spec("gymnasium") is None):
        pytest.skip("Optional PettingZoo dependency missing")
    from qox.envs.parallel_env import QoxParallelEnv
    env = QoxParallelEnv(scenario="ravager_ambush", max_steps=4)
    obs, _ = env.reset(seed=4)
    for agent in env.agents:
        assert env.observation_space(agent).contains(obs[agent])
    with pytest.raises(ValueError):
        QoxParallelEnv(scenario="ravager_mixed")
