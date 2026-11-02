"""M16a: partner uncertainty, predator resets and paired deployment guardrails."""

from pathlib import Path

import pytest

from qox.entities import Species
from qox.perception import features_for


torch = pytest.importorskip("torch")


def _fresh_pair(tmp_path: Path) -> tuple[Path, Path]:
    from qox.learning.cow_ppo import CowPolicy
    from qox.learning.nomad_ppo import NomadPolicy
    from qox.learning.joint_ppo import JointTrainOptions, _save_model

    options = JointTrainOptions(worlds=2, rollout=96, cow_steps=96,
                                nomad_steps=96, epochs=1, max_steps=24)
    cow = CowPolicy(16, cow_calls=True)
    nomad = NomadPolicy(16, nomad_v2=True)
    cow_path = tmp_path / "cow.pt"
    nomad_path = tmp_path / "nomad.pt"
    _save_model(cow, torch.optim.Adam(cow.parameters()), Species.COW,
                cow_path, options, 0, 0)
    _save_model(nomad, torch.optim.Adam(nomad.parameters()), Species.NOMAD,
                nomad_path, options, 0, 0)
    return cow_path, nomad_path


def test_m16_training_options_validate_curriculum_and_partner_mode():
    from qox.learning.joint_ppo import JointTrainOptions

    assert JointTrainOptions().scenario == "nomad_herd"
    assert JointTrainOptions(scenario="nomad_mixed", partner_mode="sample").partner_mode == "sample"
    with pytest.raises(ValueError, match="Joint scenario"):
        JointTrainOptions(scenario="invalid")
    with pytest.raises(ValueError, match="Partner mode"):
        JointTrainOptions(partner_mode="magical")


def test_mixed_curriculum_can_terminate_early_and_keep_checkpoint_schemas(tmp_path):
    from qox.learning.cow_ppo import load_cow
    from qox.learning.joint_ppo import JointTrainOptions, train_joint
    from qox.learning.nomad_ppo import load_nomad

    cow, nomad = _fresh_pair(tmp_path)
    original_cow_bytes, original_nomad_bytes = cow.read_bytes(), nomad.read_bytes()
    opts = JointTrainOptions(cow_steps=192, nomad_steps=96, worlds=2, rollout=96,
                             max_steps=48, epochs=1, scenario="nomad_mixed",
                             partner_mode="sample", seed=7)
    summary = train_joint(opts, cow, nomad, tmp_path / "trained", phase="alternate")
    assert len(summary["rounds"]) == 2
    assert summary["rounds"][0]["scenario"] == "nomad_mixed"
    assert summary["rounds"][1]["partner_mode"] == "sample"
    assert summary["rounds"][0]["completed_steps"] == 192
    assert load_cow(summary["final_cow_checkpoint"]).cow_calls
    assert load_nomad(summary["final_nomad_checkpoint"]).nomad_v2
    assert cow.read_bytes() == original_cow_bytes
    assert nomad.read_bytes() == original_nomad_bytes


def test_joint_benchmark_extends_to_predator_defense_without_magic_goal(tmp_path):
    from qox.learning.joint_ppo import benchmark_joint
    from qox.envs.core import QoxCoreEnv

    cow, nomad = _fresh_pair(tmp_path)
    env = QoxCoreEnv(scenario="nomad_defend", cow_calls=True, nomad_v2=True)
    obs, _ = env.reset(seed=31)
    assert env.world.herd_goal is None
    assert "herd_goal_dx" not in features_for(Species.COW, cow_calls=True)
    assert len([name for name in obs if name.startswith("cow_")]) == 3
    report = benchmark_joint(cow, nomad, episodes=2, seed=300000,
                             max_steps=24, scenario="nomad_defend")
    assert report["scenario"] == "nomad_defend"
    assert "mean_cow_health" in report["summary"]["learned"]
    assert "nomad_survival_rate" in report["summary"]["learned"]
    assert len(report["paired_results"]) == 2
    with pytest.raises(ValueError, match="Unsupported"):
        benchmark_joint(cow, nomad, episodes=1, scenario="unknown")


def test_identical_candidate_is_not_promoted_and_comparison_is_reproducible(tmp_path):
    from qox.learning.joint_ppo import compare_joint

    cow, nomad = _fresh_pair(tmp_path)
    result = compare_joint(cow, nomad, cow, nomad,
                           episodes=2, validation_seed=640000, max_steps=20)
    other = compare_joint(cow, nomad, cow, nomad,
                          episodes=2, validation_seed=640000, max_steps=20)
    assert result == other
    assert not result["candidate_eligible_for_manual_review"]
    assert result["migration"]["arrival_gain_per_world"] == 0
    assert result["defense"]["cow_health_gain"] == 0
    assert result["migration"]["arrival_gain_bootstrap_95pct"] == (0.0, 0.0)
    assert len(result["paired_migration"]) == 2
    assert cow.exists() and nomad.exists()


def test_m16_cli_train_and_compare_do_not_modify_existing_models(tmp_path, capsys):
    from qox.cli import main

    cow, nomad = _fresh_pair(tmp_path)
    original = (cow.read_bytes(), nomad.read_bytes())
    folder = tmp_path / "m16"
    assert main(["train-herd", "--scenario", "nomad_mixed",
                 "--partner-mode", "sample", "--phase", "cow",
                 "--cow-checkpoint", str(cow), "--nomad-checkpoint", str(nomad),
                 "--output-dir", str(folder), "--cow-steps", "96", "--nomad-steps", "96",
                 "--worlds", "2", "--rollout", "96", "--epochs", "1",
                 "--max-steps", "24"]) == 0
    report = folder / "comparison.json"
    assert main(["compare-herd", "--baseline-cow", str(cow),
                 "--baseline-nomad", str(nomad),
                 "--candidate-cow", str(folder / "cow_round1.pt"),
                 "--candidate-nomad", str(nomad), "--episodes", "2",
                 "--max-steps", "24", "--seed", "550000", "--report", str(report)]) == 0
    assert report.exists()
    assert "candidate_eligible_for_manual_review" in capsys.readouterr().out
    assert cow.read_bytes() == original[0]
    assert nomad.read_bytes() == original[1]
