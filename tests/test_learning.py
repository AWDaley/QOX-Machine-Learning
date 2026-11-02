"""M3 PPO training, checkpoint, and evaluation smoke tests."""

import pytest

from qox.cli import main


torch = pytest.importorskip("torch")


def test_masked_cow_policy_respects_legality():
    from qox.learning.cow_ppo import CowPolicy
    from qox.perception import ACTIONS, OBSERVATION_SIZE
    from qox.entities import Species

    net = CowPolicy(hidden=16)
    x = torch.zeros((3, OBSERVATION_SIZE))
    masks = torch.zeros((3, len(ACTIONS[Species.COW])))
    masks[:, 0] = 1
    dist, values = net(x, masks)
    assert values.shape == (3,)
    assert dist.sample().tolist() == [0, 0, 0]
    assert torch.isfinite(values).all()


def test_ppo_checkpoint_evaluation_determinism_and_cli(tmp_path, capsys):
    from qox.learning.cow_ppo import TrainOptions, train_cow, evaluate_cow, load_cow

    path = tmp_path / "cow.pt"
    report = train_cow(TrainOptions(steps=128, rollout=64, worlds=2, epochs=1,
                                    max_steps=32, seed=15), path)
    assert report["cow_steps"] == 128 and path.exists()
    assert path.with_suffix(".metrics.json").exists()
    assert load_cow(path).actor.out_features == 7
    first = evaluate_cow(path, episodes=3, seed=50000, max_steps=32)
    second = evaluate_cow(path, episodes=3, seed=50000, max_steps=32)
    assert first == second
    assert first["mean_grazed"] >= 0
    from qox.config import SimulationConfig
    from qox.trace import replay, run
    config = SimulationConfig(width=12, height=12, farmers=1, cows=2,
                              nomads=1, scavengers=1, ravagers=1)
    world, trace = run(config, ticks=12, mode="learned-cow", checkpoint=path, record=True)
    assert replay(trace).state_hash() == world.state_hash()
    assert main(["benchmark-cow", "--checkpoint", str(path), "--episodes", "2",
                 "--seed", "50010", "--max-steps", "32", "--report",
                 str(tmp_path / "benchmark.json")]) == 0
    assert "scripted" in capsys.readouterr().out
    assert (tmp_path / "benchmark.json").exists()


def test_reject_checkpoint_schema_mismatch(tmp_path):
    from qox.learning.cow_ppo import CowPolicy, load_cow

    path = tmp_path / "invalid.pt"
    torch.save({"obs_version": -1, "observation_features": (),
                "actions": (), "model_hidden": 8,
                "policy_state": CowPolicy(8).state_dict()}, path)
    with pytest.raises(ValueError, match="schema"):
        load_cow(path)
