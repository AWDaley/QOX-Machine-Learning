"""M16l verified crop-cycle reward, counterfactual, and audit-integrity tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qox.entities import Event, Species
from qox.learning.farmer_crop_cycle import (CropCycleDelta,
                                            FarmerCropCycleLedger)


def event(tick: int, kind: str, farmer: int, amount: float,
          x: int | None = None, y: int | None = None,
          detail: str = "") -> Event:
    return Event(tick, kind, farmer, amount, detail, x, y)


def baseline(path: Path) -> None:
    torch = pytest.importorskip("torch")
    from qox.learning.farmer_ppo import FarmerPolicy, _action_names
    from qox.perception import FARMER_OBSERVATION_VERSION, features_for

    torch.manual_seed(163)
    model = FarmerPolicy(16)
    torch.save({"species": "farmer", "model_hidden": 16,
                "policy_state": model.state_dict(),
                "obs_version": FARMER_OBSERVATION_VERSION,
                "observation_features": features_for(Species.FARMER),
                "actions": _action_names()}, path)


def test_crop_cycle_requires_actor_owned_verified_plot_and_delivered_harvest():
    ledger = FarmerCropCycleLedger()
    assert ledger.observe([event(1, "plant", 4, 0.3, 2, 3)], 5, 1) == CropCycleDelta()
    assert ledger.observe([event(1, "plant", 5, 0),
                           event(1, "plant", 5, 0.3)], 5, 1) == CropCycleDelta()
    plant = ledger.observe([event(2, "plant", 5, 0.3, 2, 3)], 5, 1)
    assert plant.planted == 1 and plant.training_bonus == pytest.approx(0.6)
    absent = ledger.observe([event(3, "harvest", 5, 2, 7, 8),
                             event(3, "harvest", 4, 2, 2, 3),
                             event(3, "deposit", 5, 0.9)], 5, 1)
    assert absent == CropCycleDelta()
    harvested = ledger.observe([event(4, "harvest", 5, 2.5, 2, 3)], 5, 1)
    assert harvested.harvested_from_own_plant == 2.5
    assert harvested.training_bonus == pytest.approx(1.0)
    assert ledger.own_harvest_on_hand == pytest.approx(2.5)
    ledger.observe([event(5, "eat", 5, 0.5, detail="carried")], 5, 1)
    deposited = ledger.observe([event(6, "deposit", 5, 3)], 5, 1)
    assert deposited.deposited_from_own_harvest == 2.0
    assert deposited.training_bonus == pytest.approx(1.2)
    assert ledger.own_harvest_on_hand == 0
    assert ledger.observe([event(7, "deposit", 5, 3)], 5, 1) == CropCycleDelta()


def test_crop_cycle_credit_cap_seed_cost_and_equal_zero_control():
    events = [event(1, "plant", 5, 0.3, 1, 1),
              event(2, "plant", 5, 0.3, 2, 2),
              event(3, "plant", 5, 0.3, 3, 3),
              event(4, "harvest", 5, 2, 3, 3),
              event(5, "plant", 5, 0.3, 4, 4),
              event(6, "deposit", 5, 2)]
    control, treated = FarmerCropCycleLedger(), FarmerCropCycleLedger()
    control_values = [control.observe([item], 5, 0) for item in events]
    treated_values = [treated.observe([item], 5, 1) for item in events]
    assert [x.planted for x in control_values] == [x.planted for x in treated_values]
    assert all(x.training_bonus == 0 for x in control_values)
    assert [round(x.training_bonus, 2) for x in treated_values] == [
        0.6, 0.6, 0.0, 0.8, 0.0, 1.02]
    assert control.own_harvest_on_hand == treated.own_harvest_on_hand == 0
    assert treated.observe([event(7, "harvest", 5, 3, 4, 4)], 5, 2).training_bonus == pytest.approx(2.4)
    for invalid in (-0.1, 2.1):
        with pytest.raises(ValueError, match="strength"):
            treated.observe([], 5, invalid)


def test_training_options_sweep_budget_and_nonwriting_dry_run(tmp_path, capsys):
    pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.farmer_ppo import FarmerTrainOptions
    from qox.learning.farmer_staged import StagedFarmerOptions
    from qox.learning.farmer_sweep import FarmerSweepOptions, _slug

    for invalid in (-1, 2.01):
        with pytest.raises(ValueError, match="Crop-cycle credit"):
            FarmerTrainOptions(crop_cycle_credit=invalid)
        with pytest.raises(ValueError, match="Crop-cycle credit"):
            StagedFarmerOptions(crop_cycle_credit=invalid)
        with pytest.raises(ValueError, match="Crop-cycle credit"):
            FarmerSweepOptions(crop_cycle_credits=(invalid,))
    with pytest.raises(ValueError, match="distinct"):
        FarmerSweepOptions(crop_cycle_credits=(1, 1))
    opts = FarmerSweepOptions(survival_weights=(3,), stage_budgets=(4096,),
                              interleave_cycles=(4,), manure_credits=(0.3,),
                              food_anchor_kls=(2,), history_modes=("none",),
                              crop_cycle_credits=(0, 1), epochs=2)
    assert opts.estimate()["configurations"] == 2
    assert opts.estimate()["approximate_ppo_decisions"] == 98_304
    assert _slug(3, 4096, 0, 4, 0.3, 2, "none", 0).endswith("_foodkl_2")
    assert _slug(3, 4096, 0, 4, 0.3, 2, "none", 1).endswith("_cropcycle_1")
    untouched = tmp_path / "never_created"
    assert main(["sweep-farmers", "--baseline", str(tmp_path / "missing.pt"),
                 "--output-dir", str(untouched), "--weights", "3",
                 "--stage-budgets", "4096", "--interleave-cycles", "4",
                 "--manure-credits", "0.3", "--food-anchor-kls", "2",
                 "--crop-cycle-credits", "0", "1", "--epochs", "2",
                 "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["approximate_ppo_decisions"] == 98_304
    assert not untouched.exists()


def test_real_training_reports_credit_and_preserves_original_m4(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.farmer_ppo import load_farmer

    m4 = tmp_path / "baseline.pt"
    baseline(m4)
    original_bytes = m4.read_bytes()
    path = tmp_path / "candidate.pt"
    assert main(["train-farmer", "--output", str(path), "--steps", "128",
                 "--worlds", "2", "--rollout", "64", "--max-steps", "18",
                 "--scenario", "agriculture", "--crop-cycle-credit", "1",
                 "--food-anchor-kl", "2", "--food-anchor-checkpoint", str(m4)]) == 0
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert data["options"]["crop_cycle_credit"] == 1
    assert load_farmer(path).history_mode == "none"
    report = json.loads(path.with_suffix(".metrics.json").read_text())
    assert report["crop_cycle_credit_awarded"] >= 0
    assert report["own_crop_food_deposited"] <= report["own_crops_harvested"]
    assert m4.read_bytes() == original_bytes


def test_two_arm_smoke_sweep_uses_no_audit_without_qualifying_candidate(tmp_path, capsys):
    pytest.importorskip("torch")
    from qox.cli import main

    m4 = tmp_path / "baseline.pt"
    baseline(m4)
    original_bytes = m4.read_bytes()
    output = tmp_path / "crop_sweep"
    assert main(["sweep-farmers", "--baseline", str(m4),
                 "--output-dir", str(output), "--weights", "3",
                 "--stage-budgets", "256", "--interleave-cycles", "4",
                 "--manure-credits", "0.3", "--food-anchor-kls", "2",
                 "--history-modes", "none", "--crop-cycle-credits", "0", "1",
                 "--training-seeds", "17", "--worlds", "2", "--rollout", "64",
                 "--epochs", "1", "--episode-cap", "18",
                 "--development-episodes", "2", "--audit-episodes", "2"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["version"] == 7
    assert [arm["crop_cycle_credit"] for arm in summary["configurations"]] == [0, 1]
    assert summary["audit"] is None and summary["development_qualifiers"] == 0
    assert summary["recommended_checkpoint"] is None
    assert not (output / "audit.json").exists()
    for arm in summary["configurations"]:
        stage_dir = Path(arm["location"]) / "seed_17"
        assert len(list(stage_dir.glob("*.pt"))) == 16
        assert json.loads((stage_dir / "development.json").read_text())
    assert m4.read_bytes() == original_bytes
