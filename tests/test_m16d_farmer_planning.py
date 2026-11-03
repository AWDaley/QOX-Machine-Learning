"""M16d long-horizon farmer curriculum and non-promoting paired evaluation."""

from __future__ import annotations

import json

import pytest

from qox.entities import Species
from qox.envs.core import QoxCoreEnv
from qox.training_lab import scenarios_for, scenario_steps


def test_farmstead_has_real_crops_finite_manure_and_cattle():
    for seed in range(8):
        env = QoxCoreEnv(scenario="farmstead", scripted_others=True)
        observations, _ = env.reset(seed=seed)
        assert "farmer_1" in observations
        world = env.world
        assert {a.species for a in world.actors.values()} == {
            Species.FARMER, Species.COW}
        assert sum(t.waste for t in world.grid.tiles) == pytest.approx(3.3)
        assert 0 < sum(t.crop for t in world.grid.tiles) < 10
        for _ in range(20):
            env.step({})
        world.check_invariants()


def test_farmstead_world_hashes_repeat_with_same_seed_and_actions():
    for scenario in ("farmstead", "farmstead_threat", "farmstead_mixed"):
        for seed in (5, 8, 19):
            a = QoxCoreEnv(scenario=scenario, scripted_others=True, max_steps=42)
            b = QoxCoreEnv(scenario=scenario, scripted_others=True, max_steps=42)
            a.reset(seed=seed)
            b.reset(seed=seed)
            predators = sum(x.species == Species.RAVAGER
                            for x in a.world.actors.values())
            assert predators == int(scenario == "farmstead_threat" or
                                    (scenario == "farmstead_mixed" and seed % 2 == 1))
            for _ in range(20):
                assert a.world.state_hash() == b.world.state_hash()
                a.step({})
                b.step({})
            assert a.world.state_hash() == b.world.state_hash()


def test_farmer_schema_and_training_lab_are_compatible():
    from qox.perception import observation_size
    assert observation_size(Species.FARMER) == 76
    assert "farmstead_mixed" in scenarios_for(Species.FARMER, 1)
    assert scenario_steps(Species.FARMER, "farmstead_mixed") == 192
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")
    from qox.envs.parallel_env import QoxParallelEnv
    with pytest.raises(ValueError, match="Mixed curricula"):
        QoxParallelEnv(scenario="farmstead_mixed")


def test_all_training_scenarios_can_be_evaluated_without_a_checkpoint():
    pytest.importorskip("torch")
    from qox.learning.farmer_ppo import FARMER_SCENARIOS, FarmerTrainOptions, evaluate_farmer

    assert set(FARMER_SCENARIOS) == set(scenarios_for(Species.FARMER, 1))
    for scenario in FARMER_SCENARIOS:
        FarmerTrainOptions(scenario=scenario)
        result = evaluate_farmer(episodes=1, seed=19, scenario=scenario,
                                 max_steps=2, mode="scripted")
        assert result["scenario"] == scenario
        assert len(result["results"]) == 1


def test_pairwise_comparison_is_reproducible_without_model_changes(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.learning.farmer_ppo import FarmerPolicy, FarmerTrainOptions, _action_names
    from qox.learning.farmer_eval import compare_farmers, SCENARIOS
    from qox.perception import FARMER_OBSERVATION_VERSION, features_for

    model = FarmerPolicy(16)
    checkpoint = tmp_path / "farmer.pt"
    torch.save({"species": "farmer", "model_hidden": 16,
                "policy_state": model.state_dict(),
                "obs_version": FARMER_OBSERVATION_VERSION,
                "observation_features": features_for(Species.FARMER),
                "actions": _action_names()}, checkpoint)
    original = checkpoint.read_bytes()
    first = compare_farmers(checkpoint, checkpoint, episodes=2,
                            seed=1_600_000, max_steps=24)
    second = compare_farmers(checkpoint, checkpoint, episodes=2,
                             seed=1_600_000, max_steps=24)
    assert first == second
    assert not first["eligible_for_manual_review"]
    assert set(first["cases"]) == set(SCENARIOS)
    for entry in first["cases"].values():
        assert all(d["mean"] == 0 for d in entry["differences"].values())
    assert checkpoint.read_bytes() == original
    assert FarmerTrainOptions(scenario="farmstead_mixed").max_steps == 96


def test_cli_train_and_compare_without_promotion(tmp_path):
    pytest.importorskip("torch")
    from qox.cli import main
    model = tmp_path / "farmer.pt"
    report = tmp_path / "comparison.json"
    assert main(["train-farmer", "--scenario", "farmstead_mixed",
                 "--steps", "64", "--worlds", "2", "--rollout", "64",
                 "--max-steps", "24", "--output", str(model)]) == 0
    saved = model.read_bytes()
    assert main(["compare-farmers", "--baseline", str(model),
                 "--candidate", str(model), "--episodes", "2",
                 "--max-steps", "18", "--report", str(report)]) == 0
    assert model.read_bytes() == saved
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["promotion_checks"]["enough_heldout_worlds"] is False
    assert "farmstead_threat" in data["cases"]
