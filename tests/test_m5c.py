"""M5c frozen-partner training, checkpoint compatibility and paired evaluations."""

from pathlib import Path

import pytest

from qox.cli import main
from qox.entities import Species
from qox.envs.core import QoxCoreEnv


torch = pytest.importorskip("torch")


def _make_initial_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    """Use M5 pre-trained checkpoint schema, not training assets committed to git."""
    from qox.learning.joint_ppo import JointTrainOptions, _save_model
    from qox.learning.cow_ppo import CowPolicy
    from qox.learning.nomad_ppo import NomadPolicy
    options = JointTrainOptions(cow_steps=96, nomad_steps=96, worlds=2,
                                rollout=96, max_steps=24, epochs=1)
    cow, nomad = CowPolicy(16, cow_calls=True), NomadPolicy(16, nomad_v2=True)
    cp, np = tmp_path / "cow_init.pt", tmp_path / "nomad_init.pt"
    _save_model(cow, torch.optim.Adam(cow.parameters()), Species.COW, cp, options, 0, 0)
    _save_model(nomad, torch.optim.Adam(nomad.parameters()), Species.NOMAD, np,
                options, 0, 0)
    return cp, np


def test_herd_training_options_validate_sample_shape():
    from qox.learning.joint_ppo import JointTrainOptions
    with pytest.raises(ValueError, match="divisible"):
        JointTrainOptions(worlds=4, rollout=128)
    with pytest.raises(ValueError, match="positive"):
        JointTrainOptions(cow_steps=0)
    with pytest.raises(ValueError, match="PPO"):
        JointTrainOptions(lr=0)


def test_one_shared_policy_controls_three_cows(tmp_path):
    from qox.learning.joint_ppo import JointTrainOptions, train_phase
    from qox.learning.cow_ppo import load_cow
    from qox.learning.nomad_ppo import load_nomad
    cow_checkpoint, nomad_checkpoint = _make_initial_checkpoints(tmp_path)
    cow, nomad = load_cow(cow_checkpoint), load_nomad(nomad_checkpoint)
    options = JointTrainOptions(cow_steps=96, nomad_steps=96, worlds=2,
                                rollout=96, max_steps=24, epochs=1)
    first = tmp_path / "trained_cow.pt"
    summary = train_phase(Species.COW, cow, nomad, options, first)
    assert summary["completed_steps"] == 96
    assert load_cow(first).cow_calls
    assert torch.load(first, map_location="cpu", weights_only=True)["round"] == 1
    second = tmp_path / "trained_nomad.pt"
    summary2 = train_phase(Species.NOMAD, cow, nomad, options, second)
    assert summary2["completed_steps"] == 96
    assert load_nomad(second).nomad_v2


def test_train_and_benchmark_joint_cli_and_paired_replay(tmp_path, capsys):
    from qox.learning.joint_ppo import benchmark_joint
    cow_checkpoint, nomad_checkpoint = _make_initial_checkpoints(tmp_path)
    directory = tmp_path / "joint"
    assert main(["train-herd", "--cow-checkpoint", str(cow_checkpoint),
                 "--nomad-checkpoint", str(nomad_checkpoint),
                 "--output-dir", str(directory), "--cow-steps", "96",
                 "--nomad-steps", "96", "--worlds", "2", "--rollout", "96",
                 "--max-steps", "24", "--epochs", "1"]) == 0
    assert (directory / "joint_training.json").exists()
    report_a = benchmark_joint(directory / "cow_round1.pt", directory / "nomad_round1.pt",
                               episodes=2, seed=350000, max_steps=24)
    report_b = benchmark_joint(directory / "cow_round1.pt", directory / "nomad_round1.pt",
                               episodes=2, seed=350000, max_steps=24)
    assert report_a == report_b
    assert len(report_a["paired_results"]) == 2
    assert {"learned", "no_calls", "reversed_calls", "scripted_nomad", "random_nomad",
            "scripted_cows"} <= set(report_a["summary"])
    outfile = tmp_path / "bench.json"
    assert main(["benchmark-herd", "--cow-checkpoint", str(directory / "cow_round1.pt"),
                 "--nomad-checkpoint", str(directory / "nomad_round1.pt"),
                 "--episodes", "2", "--seed", "350000", "--max-steps", "24",
                 "--report", str(outfile)]) == 0
    assert outfile.exists()
    assert "mean_gain_over_no_calls" in capsys.readouterr().out


def test_nomad_herd_observations_do_not_expose_cow_goal():
    from qox.perception import features_for
    env = QoxCoreEnv(scenario="nomad_herd", cow_calls=True, nomad_v2=True)
    obs, _ = env.reset(seed=426)
    assert len([n for n in obs if n.startswith("cow_")]) == 3
    assert "herd_goal_dx" not in features_for(Species.COW, cow_calls=True)
    assert "herd_goal_dx" in features_for(Species.NOMAD, nomad_v2=True)
    assert env.world.herd_goal is not None


def test_cow_only_preserves_frozen_nomad_checkpoint(tmp_path):
    from qox.learning.joint_ppo import JointTrainOptions, train_joint
    cow, nomad = _make_initial_checkpoints(tmp_path)
    nomad_bytes = nomad.read_bytes()
    out = tmp_path / "cow_only"
    opts = JointTrainOptions(cow_steps=96, nomad_steps=96, worlds=2,
                             rollout=96, epochs=1, max_steps=24)
    result = train_joint(opts, cow, nomad, out, phase="cow")
    assert len(result["rounds"]) == 1
    assert result["final_nomad_checkpoint"] == str(nomad)
    assert nomad.read_bytes() == nomad_bytes
    assert (out / "cow_round1.pt").exists()
    assert not (out / "nomad_round1.pt").exists()
    with pytest.raises(ValueError, match="phase"):
        train_joint(opts, cow, nomad, out, phase="invalid")
