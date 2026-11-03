"""M16b: tactical nomad defense without scripted attack selection."""

import pytest

from qox.entities import Species
from qox.envs.core import QoxCoreEnv
from qox.perception import features_for


def test_guard_and_intercept_rotations_preserve_locality_and_valid_physics():
    for scenario in ("nomad_guard", "nomad_intercept", "nomad_defense_mix"):
        threats = set()
        for seed in range(16):
            env = QoxCoreEnv(scenario=scenario, cow_calls=True, nomad_v2=True,
                             scripted_others=True, max_steps=24)
            obs, _ = env.reset(seed=seed)
            assert len([n for n in obs if n.startswith("cow_")]) == 3
            cows = [a for a in env.world.actors.values() if a.species == Species.COW]
            nomad = next(a for a in env.world.actors.values() if a.species == Species.NOMAD)
            ravager = next(a for a in env.world.actors.values()
                           if a.species == Species.RAVAGER)
            assert len({a.position for a in env.world.actors.values()}) == 5
            assert env.world.herd_goal is None
            assert nomad.health == 100
            assert len(features_for(Species.COW, cow_calls=True)) == 66
            threats.add((ravager.x - cows[1].x, ravager.y - cows[1].y))
            for _ in range(3):
                env.step({})
                env.world.check_invariants()
        assert len(threats) >= 4


def test_balanced_guard_migration_curriculum_does_not_fix_pasture_direction():
    seen = {"guard": 0, "herd": 0}
    goal_directions = set()
    for seed in range(32):
        env = QoxCoreEnv(scenario="nomad_guard_mixed", cow_calls=True,
                         nomad_v2=True, max_steps=12)
        env.reset(seed=seed)
        guard = (seed // 4) % 2
        seen["guard" if guard else "herd"] += 1
        assert int(env.world.herd_goal is None) == guard
        assert sum(a.species == Species.RAVAGER for a in env.world.actors.values()) == guard
        if not guard:
            goal_directions.add(env.world.herd_goal[0])
    assert seen == {"guard": 16, "herd": 16}
    assert goal_directions == {1, 12}
    assert "herd_goal_dx" not in features_for(Species.COW, cow_calls=True)


def test_m16b_anchors_require_checkpoint_and_validate_coefficients(tmp_path):
    pytest.importorskip("torch")
    from qox.learning.nomad_ppo import NomadTrainOptions, train_nomad
    for invalid in (-0.01, -10):
        with pytest.raises(ValueError, match="PPO"):
            NomadTrainOptions(anchor_kl=invalid)
    opts = NomadTrainOptions(steps=64, scenario="nomad_guard", max_steps=12,
                             worlds=2, rollout=64, epochs=1, hidden=16,
                             anchor_kl=0.03)
    with pytest.raises(ValueError, match="requires an initial checkpoint"):
        train_nomad(opts, tmp_path / "missing_anchor.pt")


def test_anchored_training_does_not_overwrite_old_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.learning.nomad_ppo import NomadPolicy, NomadTrainOptions, load_nomad, train_nomad
    from qox.learning.joint_ppo import JointTrainOptions, _save_model

    initial = tmp_path / "initial.pt"
    model = NomadPolicy(16, nomad_v2=True)
    _save_model(model, torch.optim.Adam(model.parameters()), Species.NOMAD,
                initial, JointTrainOptions(worlds=2, rollout=96,
                                             cow_steps=96, nomad_steps=96), 0, 0)
    old_bytes = initial.read_bytes()
    candidate = tmp_path / "candidate.pt"
    opts = NomadTrainOptions(steps=128, scenario="nomad_guard_mixed", max_steps=16,
                             worlds=2, rollout=64, epochs=1, minibatch=32,
                             hidden=16, nomad_v2=True, anchor_kl=.02, seed=29)
    result = train_nomad(opts, candidate, init_checkpoint=initial)
    assert result["nomad_steps"] == 128
    assert initial.read_bytes() == old_bytes
    assert load_nomad(candidate).nomad_v2
    saved = torch.load(candidate, map_location="cpu", weights_only=True)
    assert saved["options"]["anchor_kl"] == .02


def test_noattack_ablation_and_paired_defender_evaluation(tmp_path):
    pytest.importorskip("torch")
    from qox.learning.defender_eval import compare_defenders
    from qox.learning.nomad_ppo import evaluate_nomad
    from qox.learning.cow_ppo import CowPolicy
    from qox.learning.nomad_ppo import NomadPolicy
    from qox.learning.joint_ppo import JointTrainOptions, _save_model
    import torch
    cow = tmp_path / "cow.pt"
    nomad = tmp_path / "nomad.pt"
    options = JointTrainOptions(worlds=2, rollout=96, cow_steps=96, nomad_steps=96)
    c = CowPolicy(hidden=16, cow_calls=True)
    n = NomadPolicy(hidden=16, nomad_v2=True)
    _save_model(c, torch.optim.Adam(c.parameters()), Species.COW,
                cow, options, 0, 0)
    _save_model(n, torch.optim.Adam(n.parameters()), Species.NOMAD,
                nomad, options, 0, 0)
    standard = evaluate_nomad(nomad, cow_checkpoint=cow, episodes=2, seed=993000,
                               max_steps=16, scenario="nomad_guard", mode="policy_greedy")
    disabled = evaluate_nomad(nomad, cow_checkpoint=cow, episodes=2, seed=993000,
                               max_steps=16, scenario="nomad_guard",
                               mode="policy_greedy_noattack")
    assert disabled["mean_nomad_attacks"] == 0
    assert len(standard["results"]) == 2
    report = compare_defenders(nomad, nomad, cow, episodes=2, seed=993000,
                               max_steps=16)
    again = compare_defenders(nomad, nomad, cow, episodes=2, seed=993000,
                              max_steps=16)
    assert report == again
    assert len(report["scenarios"]) == 4
    assert report["scenarios"]["nomad_guard"]["candidate_minus_baseline"]["cow_health"] == 0
    assert not report["candidate_eligible_for_manual_review"]
    assert report["checkpoint_hashes"]["baseline"] == report["checkpoint_hashes"]["candidate"]


def test_compare_defenders_cli_can_export_without_writing_models(tmp_path, capsys):
    pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.joint_ppo import JointTrainOptions, _save_model
    from qox.learning.cow_ppo import CowPolicy
    from qox.learning.nomad_ppo import NomadPolicy
    import torch
    opts = JointTrainOptions(worlds=2, rollout=96, cow_steps=96, nomad_steps=96)
    cow, nomad = tmp_path / "cow.pt", tmp_path / "nomad.pt"
    c, n = CowPolicy(16, cow_calls=True), NomadPolicy(16, nomad_v2=True)
    _save_model(c, torch.optim.Adam(c.parameters()), Species.COW, cow, opts, 0, 0)
    _save_model(n, torch.optim.Adam(n.parameters()), Species.NOMAD, nomad, opts, 0, 0)
    before = nomad.read_bytes()
    out = tmp_path / "result.json"
    assert main(["compare-defenders", "--baseline-nomad", str(nomad),
                 "--candidate-nomad", str(nomad), "--cow-checkpoint", str(cow),
                 "--episodes", "2", "--max-steps", "16", "--seed", "950000",
                 "--report", str(out)]) == 0
    assert out.exists()
    assert "candidate_eligible_for_manual_review" in capsys.readouterr().out
    assert nomad.read_bytes() == before
