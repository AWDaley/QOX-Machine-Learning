"""M16j: selective frozen-M4 KL on fresh PPO food-task states."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _baseline(path: Path) -> None:
    torch = pytest.importorskip("torch")
    from qox.entities import Species
    from qox.learning.farmer_ppo import FarmerPolicy, _action_names
    from qox.perception import FARMER_OBSERVATION_VERSION, features_for

    torch.manual_seed(103)
    model = FarmerPolicy(16)
    torch.save({"species": "farmer", "model_hidden": 16,
                "policy_state": model.state_dict(),
                "obs_version": FARMER_OBSERVATION_VERSION,
                "observation_features": features_for(Species.FARMER),
                "actions": _action_names()}, path)


def test_anchor_covers_teacher_food_actions_and_loaded_food_navigation_only():
    torch = pytest.importorskip("torch")
    from torch.distributions import Categorical
    from torch.distributions.kl import kl_divergence
    from qox.actions import Kind
    from qox.entities import Species
    from qox.learning.farmer_ppo import _on_policy_food_anchor
    from qox.perception import ACTIONS, features_for

    actions = ACTIONS[Species.FARMER]
    deposit = next(i for i, a in enumerate(actions) if a.kind == Kind.DEPOSIT)
    move = next(i for i, a in enumerate(actions) if a.kind == Kind.MOVE)
    rest = next(i for i, a in enumerate(actions) if a.kind == Kind.REST)
    n = len(actions)
    teacher_logits = torch.zeros((4, n), requires_grad=True)
    with torch.no_grad():
        teacher_logits[0, deposit] = 8
        teacher_logits[1, move] = 8
        teacher_logits[2, move] = 8
        teacher_logits[3, rest] = 8

    class FrozenTeacher:
        def __call__(self, observations, masks):
            return Categorical(logits=teacher_logits), None

    features = features_for(Species.FARMER)
    states = torch.zeros((4, len(features)))
    states[1, features.index("carried_food")] = 0.30
    states[2, features.index("carried_food")] = 0.12
    masks = torch.ones((4, n))
    student_logits = torch.zeros((4, n), requires_grad=True)
    student = Categorical(logits=student_logits)
    loss, fraction = _on_policy_food_anchor(FrozenTeacher(), student, states, masks)
    expected = kl_divergence(Categorical(logits=teacher_logits.detach()), student)
    assert float(fraction) == pytest.approx(0.5)
    assert float(loss.detach()) == pytest.approx(float(expected[:2].sum().detach() / 4))
    loss.backward()
    assert student_logits.grad[:2].abs().sum() > 0
    assert student_logits.grad[2:].abs().sum() == 0
    assert teacher_logits.grad is None


def test_anchor_options_require_finite_strength_and_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from qox.learning.farmer_ppo import FarmerTrainOptions, train_farmer
    from qox.learning.farmer_staged import StagedFarmerOptions
    from qox.learning.farmer_sweep import FarmerSweepOptions, _slug

    for invalid in (-1, 10.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Food anchor"):
            FarmerTrainOptions(food_anchor_kl=invalid)
        with pytest.raises(ValueError, match="Food anchor"):
            StagedFarmerOptions(food_anchor_kl=invalid)
        with pytest.raises(ValueError, match="Food anchor"):
            FarmerSweepOptions(food_anchor_kls=(invalid,))
    with pytest.raises(ValueError, match="distinct"):
        FarmerSweepOptions(food_anchor_kls=(0.5, 0.5))
    with pytest.raises(ValueError, match="requires.*checkpoint"):
        train_farmer(FarmerTrainOptions(steps=64, rollout=64, worlds=2,
                                       food_anchor_kl=0.5), tmp_path / "must_not_exist.pt")
    assert not (tmp_path / "must_not_exist.pt").exists()
    assert _slug(3, 4096, 0, 4, 0.3, 0).endswith("_manure_0p29999999999999999")
    assert _slug(3, 4096, 0, 4, 0.3, 0.5).endswith("_foodkl_0p5")


def test_matched_three_arm_budget_and_noncreating_dry_run(tmp_path, capsys):
    pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.farmer_sweep import FarmerSweepOptions

    opts = FarmerSweepOptions(survival_weights=(3,), stage_budgets=(4096,),
                              interleave_cycles=(4,), manure_credits=(0.3,),
                              food_anchor_kls=(0, 0.5, 2), epochs=2)
    estimate = opts.estimate()
    assert estimate["configurations"] == 3
    assert estimate["training_runs"] == 3 * 3 * 4 * 4
    assert estimate["approximate_ppo_decisions"] == 3 * 3 * 4 * 4096
    output = tmp_path / "not_created"
    assert main(["sweep-farmers", "--baseline", str(tmp_path / "missing.pt"),
                 "--output-dir", str(output), "--weights", "3",
                 "--stage-budgets", "4096", "--interleave-cycles", "4",
                 "--manure-credits", "0.3", "--food-anchor-kls", "0", "0.5", "2",
                 "--epochs", "2", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["approximate_ppo_decisions"] == 147456
    assert not output.exists()


def test_real_cli_sweep_and_direct_train_keep_reference_immutable(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    output = tmp_path / "sweep"
    command = ["sweep-farmers", "--baseline", str(baseline),
               "--output-dir", str(output), "--weights", "3", "--stage-budgets", "256",
               "--interleave-cycles", "4", "--manure-credits", "0.3",
               "--food-anchor-kls", "0", "0.5", "--training-seeds", "17",
               "--worlds", "2", "--rollout", "64", "--epochs", "1",
               "--episode-cap", "18", "--development-episodes", "2",
               "--audit-episodes", "2"]
    assert main(command) == 0
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [c["food_anchor_kl"] for c in result["configurations"]] == [0, 0.5]
    assert len(list(output.rglob("*.pt"))) == 32
    assert result["audit"] is None and not result["eligible_for_manual_review"]
    for config in result["configurations"]:
        stages = json.loads((Path(config["location"]) / "summary.json").read_text())[
            "replicas"][0]["stages"]
        assert len(stages) == 16
        assert all(0 <= stage["food_anchor_fraction"] <= 1 for stage in stages)
        model = torch.load(stages[-1]["checkpoint"], map_location="cpu", weights_only=True)
        assert model["options"]["food_anchor_kl"] == config["food_anchor_kl"]
    direct = tmp_path / "direct.pt"
    assert main(["train-farmer", "--output", str(direct), "--steps", "64",
                 "--worlds", "2", "--rollout", "64", "--max-steps", "18",
                 "--food-anchor-kl", "0.5", "--food-anchor-checkpoint", str(baseline)]) == 0
    assert torch.load(direct, map_location="cpu", weights_only=True)["options"][
        "food_anchor_kl"] == 0.5
    assert baseline.read_bytes() == original


def test_anchor_selection_keeps_one_blind_audit(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from qox.learning import farmer_sweep
    from qox.learning.farmer_sweep import FarmerSweepOptions, run_farmer_sweep

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    audited = []

    def fake_staged(source, location, options, *, audit_enabled):
        assert not audit_enabled
        return {"replicas": [{"training_seed": seed,
                             "candidate": str(location / f"seed_{seed}.pt"),
                             "food_gain": 3 + options.food_anchor_kl,
                             "development_passed": True,
                             "development_checks": {"pass": True},
                             "development_report": str(location / "development.json")}
                            for seed in options.training_seeds]}

    def fake_compare(source, candidate, *, episodes, seed, max_steps):
        audited.append(seed)
        return {"promotion_checks": {"pass": True},
                "cases": {scenario: {"metrics": {
                    "baseline": {"mean_final_health": 80, "mean_crop_eaten_by_cows": 1,
                                 "mean_deposited": 2},
                    "candidate": {"mean_final_health": 80, "mean_crop_eaten_by_cows": 1,
                                  "mean_deposited": 3}}}
                    for scenario in ("farmstead", "farmstead_threat")}}

    monkeypatch.setattr(farmer_sweep, "train_farmer_staged", fake_staged)
    monkeypatch.setattr(farmer_sweep, "compare_farmers", fake_compare)
    monkeypatch.setattr(farmer_sweep, "evaluate_farmer", lambda **kwargs: {})
    opts = FarmerSweepOptions(survival_weights=(3,), stage_budgets=(4096,),
                              interleave_cycles=(4,), manure_credits=(0.3,),
                              food_anchor_kls=(0, 0.5, 2),
                              development_seed=2_400_000, audit_seed=3_400_000)
    result = run_farmer_sweep(baseline, tmp_path / "sweep", opts)
    assert audited == [3_400_000]
    assert result["selected_before_audit"]["food_anchor_kl"] == 2
    assert result["selected_before_audit"]["training_seed"] == 17
    assert result["eligible_for_manual_review"]
