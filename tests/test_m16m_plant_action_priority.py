"""M16m: legal, M4-selected planting preference without action overrides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _baseline(path: Path) -> None:
    from test_m16l_crop_cycle_credit import baseline

    baseline(path)


def test_plant_priority_margin_follows_frozen_teacher_only_on_legal_plant():
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.distributions import Categorical

    from qox.actions import Kind
    from qox.entities import Species
    from qox.learning.farmer_ppo import _plant_action_priority_loss
    from qox.perception import ACTIONS, observation_size

    plant = next(i for i, action in enumerate(ACTIONS[Species.FARMER])
                 if action.kind == Kind.PLANT)
    deposit = next(i for i, action in enumerate(ACTIONS[Species.FARMER])
                   if action.kind == Kind.DEPOSIT)
    count = len(ACTIONS[Species.FARMER])

    class FrozenTeacher(nn.Module):
        def __init__(self, preference: int | tuple[int, ...]):
            super().__init__()
            self.preference = preference

        def forward(self, observations, masks):
            assert observations.shape[1] == observation_size(Species.FARMER)
            logits = torch.full((observations.shape[0], count), -8.0)
            preferences = ((self.preference,) * len(masks)
                           if isinstance(self.preference, int) else self.preference)
            for row, preferred in enumerate(preferences):
                logits[row, preferred] = 8.0
            logits = logits.masked_fill(masks <= 0, -1e9)
            return Categorical(logits=logits), torch.zeros(len(masks))

    masks = torch.ones((2, count))
    logits = torch.zeros((2, count), requires_grad=True)
    with torch.no_grad():
        logits[0, plant] = -1.5
        logits[0, deposit] = 2.0
    student = Categorical(logits=logits)
    loss, fraction = _plant_action_priority_loss(
        FrozenTeacher((plant, deposit)), student,
        torch.zeros((2, observation_size(Species.FARMER) + 14)), masks)
    assert fraction.item() == 0.5
    assert loss.item() == pytest.approx(3.75)
    loss.backward()
    assert logits.grad[:, plant].tolist() == pytest.approx([-1.0, 0.0])
    assert logits.grad[:, deposit].tolist() == pytest.approx([1.0, 0.0])

    no_plant_loss, no_plant_fraction = _plant_action_priority_loss(
        FrozenTeacher(deposit), student, torch.zeros((2, observation_size(Species.FARMER))),
        masks)
    assert no_plant_loss.item() == 0 and no_plant_fraction.item() == 0
    illegal = masks.clone()
    illegal[:, plant] = 0
    illegal_loss, illegal_fraction = _plant_action_priority_loss(
        FrozenTeacher(plant), Categorical(logits=logits.detach().masked_fill(
            illegal <= 0, -1e9)), torch.zeros((2, observation_size(Species.FARMER))), illegal)
    assert illegal_loss.item() == 0 and illegal_fraction.item() == 0


def test_invalid_priority_rejected_at_every_layer_and_dry_run_does_not_write(tmp_path, capsys):
    pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.farmer_ppo import FarmerTrainOptions
    from qox.learning.farmer_staged import StagedFarmerOptions
    from qox.learning.farmer_sweep import FarmerSweepOptions, _slug

    for invalid in (-0.1, 10.01):
        with pytest.raises(ValueError, match="Plant priority"):
            FarmerTrainOptions(plant_priority_weight=invalid)
        with pytest.raises(ValueError, match="Plant priority"):
            StagedFarmerOptions(plant_priority_weight=invalid)
        with pytest.raises(ValueError, match="Plant priority"):
            FarmerSweepOptions(plant_priorities=(invalid,))
    with pytest.raises(ValueError, match="distinct"):
        FarmerSweepOptions(plant_priorities=(0, 0))
    opts = FarmerSweepOptions(survival_weights=(3,), stage_budgets=(4096,),
                              interleave_cycles=(4,), manure_credits=(0.3,),
                              food_anchor_kls=(2,), crop_cycle_credits=(0,),
                              history_modes=("none",), plant_priorities=(0, 1), epochs=2)
    assert opts.estimate()["configurations"] == 2
    assert opts.estimate()["approximate_ppo_decisions"] == 98_304
    assert _slug(3, 4096, 0, 4, 0.3, 2, "none", 0, 1).endswith("_plantpriority_1")
    untouched = tmp_path / "not-created"
    assert main(["sweep-farmers", "--baseline", str(tmp_path / "missing.pt"),
                 "--output-dir", str(untouched), "--weights", "3",
                 "--stage-budgets", "4096", "--interleave-cycles", "4",
                 "--manure-credits", "0.3", "--food-anchor-kls", "2",
                 "--plant-priorities", "0", "1", "--epochs", "2",
                 "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["approximate_ppo_decisions"] == 98_304
    assert not untouched.exists()


def test_priority_training_requires_immutable_teacher_and_reports_coverage(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.farmer_ppo import (FarmerTrainOptions, evaluate_farmer,
                                         train_farmer)

    baseline = tmp_path / "m4.pt"
    _baseline(baseline)
    baseline_bytes = baseline.read_bytes()
    with pytest.raises(ValueError, match="immutable M4 teacher"):
        train_farmer(FarmerTrainOptions(steps=64, rollout=64, worlds=2,
                                       plant_priority_weight=1), tmp_path / "missing.pt")
    candidate = tmp_path / "candidate.pt"
    assert main(["train-farmer", "--output", str(candidate), "--steps", "128",
                 "--worlds", "2", "--rollout", "64", "--max-steps", "18",
                 "--scenario", "agriculture", "--plant-priority", "1",
                 "--food-anchor-checkpoint", str(baseline)]) == 0
    report = json.loads(candidate.with_suffix(".metrics.json").read_text())
    assert "final_plant_priority_loss" in report
    assert 0 <= report["final_plant_priority_fraction"] <= 1
    assert torch.load(candidate, weights_only=True)["options"]["plant_priority_weight"] == 1
    diagnostics = evaluate_farmer(candidate, episodes=2, seed=1_850_000,
                                  scenario="farmstead", max_steps=18,
                                  chain_diagnostics=True)
    assert sum(diagnostics["plant_competitors"].values()) == sum(
        row["chain"]["plant_legal_steps"] - row["chain"]["plant_chosen_steps"]
        for row in diagnostics["results"])
    assert baseline.read_bytes() == baseline_bytes


def test_matched_two_arm_small_sweep_uses_new_version_and_no_blind_audit(tmp_path):
    pytest.importorskip("torch")
    from qox.cli import main

    baseline = tmp_path / "m4.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    dest = tmp_path / "study"
    assert main(["sweep-farmers", "--baseline", str(baseline),
                 "--output-dir", str(dest), "--weights", "3",
                 "--stage-budgets", "256", "--interleave-cycles", "4",
                 "--manure-credits", "0.3", "--food-anchor-kls", "2",
                 "--plant-priorities", "0", "1", "--history-modes", "none",
                 "--training-seeds", "17", "--worlds", "2", "--rollout", "64",
                 "--epochs", "1", "--episode-cap", "18",
                 "--development-episodes", "2", "--audit-episodes", "2"]) == 0
    result = json.loads((dest / "summary.json").read_text())
    assert result["version"] == 8
    assert [arm["plant_priority_weight"] for arm in result["configurations"]] == [0, 1]
    assert result["audit"] is None and result["development_qualifiers"] == 0
    assert not (dest / "audit.json").exists()
    for arm in result["configurations"]:
        path = Path(arm["location"]) / "seed_17"
        assert len(list(path.glob("*.pt"))) == 16
        stage = json.loads((Path(arm["location"]) / "summary.json").read_text())
        assert stage["replicas"][0]["stages"][0]["plant_priority_weight"] in (0, 1)
    assert baseline.read_bytes() == original
