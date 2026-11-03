"""M16e multi-seed PPO pipeline, selection and non-promoting policy gates."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

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


def test_staged_options_require_distinct_training_and_evaluation_seeds():
    pytest.importorskip("torch")
    from qox.learning.farmer_staged import StagedFarmerOptions, _stage_seed

    for kwargs in ({"training_seeds": (1, 1)}, {"training_seeds": ()},
                   {"training_seeds": (-1,)}, {"rollout": 3, "worlds": 2},
                   {"development_seed": 2_600_000},
                   {"audit_seed": 1_600_000}, {"audit_seed": 999_999_999},
                   {"survival_weight": 0.5}, {"survival_weight": 21}):
        with pytest.raises(ValueError):
            StagedFarmerOptions(**kwargs)
    stage_seeds = {_stage_seed(replica, stage, seed)
                   for replica, seed in enumerate((17, 37, 59)) for stage in range(4)}
    assert len(stage_seeds) == 12
    assert min(stage_seeds) > 1_000_000_000
    assert max(stage_seeds) < 2**32


def test_staged_cli_keeps_baseline_and_rejects_unreviewable_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    folder = tmp_path / "experiment"
    command = ["train-farmer-staged", "--baseline", str(baseline),
               "--output-dir", str(folder), "--training-seeds", "17",
               "--stage-steps", "64", "--worlds", "2", "--rollout", "64",
               "--epochs", "1", "--episode-cap", "12",
               "--development-episodes", "2", "--audit-episodes", "2"]
    assert main(command) == 0
    data = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    stages = data["replicas"][0]["stages"]
    assert [stage["scenario"] for stage in stages] == [
        "agriculture", "manure", "refuge", "farmstead_mixed"]
    assert [stage["survival_weight"] for stage in stages] == [1.0, 1.0, 3.0, 3.0]
    assert all(Path(stage["checkpoint"]).is_file() for stage in stages)
    final_checkpoint = torch.load(stages[-1]["checkpoint"], map_location="cpu",
                                  weights_only=True)
    assert final_checkpoint["options"]["survival_weight"] == 3.0
    assert data["replicas"][0]["development_checks"]["enough_heldout_worlds"] is False
    assert not data["eligible_for_manual_review"]
    assert data["recommended_checkpoint"] is None
    assert baseline.read_bytes() == original
    with pytest.raises(FileExistsError, match="already exists"):
        main(command)
    assert baseline.read_bytes() == original


@pytest.mark.parametrize("audit_pass", [False, True])
def test_staged_gate_audits_only_one_development_winner(monkeypatch, tmp_path,
                                                        audit_pass):
    pytest.importorskip("torch")
    from qox.learning import farmer_staged
    from qox.learning.farmer_staged import StagedFarmerOptions, train_farmer_staged

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    calls = []

    def fake_train(options, output, init_checkpoint):
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(init_checkpoint, output)
        return {"farmer_steps": options.steps, "completed_episodes": 1}

    def fake_compare(source, candidate, *, episodes, seed, max_steps):
        calls.append((str(candidate), seed))
        candidate_text = str(candidate)
        food_gain = next(score for name, score in (("seed_17", 1), ("seed_37", 3),
                                                   ("seed_59", 2)) if name in candidate_text)
        return {"promotion_checks": {"existing_health_and_survival_gate":
                                      audit_pass if seed == 2_600_000 else True},
                "cases": {scenario: {"metrics": {
                    "baseline": {"mean_deposited": 2.0, "mean_final_health": 80.0,
                                 "mean_crop_eaten_by_cows": 1.0},
                    "candidate": {"mean_deposited": 2.0 + food_gain,
                                  "mean_final_health": 80.0,
                                  "mean_crop_eaten_by_cows": 1.0}}}
                    for scenario in ("farmstead", "farmstead_threat")}}

    monkeypatch.setattr(farmer_staged, "train_farmer", fake_train)
    monkeypatch.setattr(farmer_staged, "compare_farmers", fake_compare)
    data = train_farmer_staged(baseline, tmp_path / "experiment", StagedFarmerOptions())
    assert len(calls) == 4  # Three development checks, only one untouched audit range.
    assert [seed for _, seed in calls] == [1_600_000] * 3 + [2_600_000]
    assert "seed_37" in calls[-1][0]
    assert data["audit"]["passed"] is audit_pass
    assert data["eligible_for_manual_review"] is audit_pass
    assert bool(data["recommended_checkpoint"]) is audit_pass
    assert baseline.read_bytes() == original


def test_staged_gate_rejects_productive_but_unhealthy_farmer():
    pytest.importorskip("torch")
    from qox.learning.farmer_staged import _gate

    report = {"promotion_checks": {"food_storage_improves": True}, "cases": {
        "farmstead": {"metrics": {
            "baseline": {"mean_final_health": 100, "mean_crop_eaten_by_cows": 1},
            "candidate": {"mean_final_health": 92, "mean_crop_eaten_by_cows": 2}}},
        "farmstead_threat": {"metrics": {
            "baseline": {"mean_deposited": 3, "mean_crop_eaten_by_cows": 0},
            "candidate": {"mean_deposited": 1, "mean_crop_eaten_by_cows": 2}}}}}
    checks = _gate(report)
    assert not checks["safe_health_not_worse"]
    assert not checks["crop_damage_not_worse"]
    assert not checks["threat_crop_damage_not_worse"]
    assert not checks["threat_food_retained"]


def test_training_safety_weight_uses_only_actual_injury_and_death_penalties():
    pytest.importorskip("torch")
    from qox.learning.farmer_ppo import (FarmerTrainOptions,
                                         _safety_adjusted_reward)

    reported = {"harvest": 1.2, "injury": -0.7, "death": -4.0}
    assert _safety_adjusted_reward(2.0, reported, 1.0) == 2.0
    assert _safety_adjusted_reward(2.0, reported, 3.0) == pytest.approx(-7.4)
    assert _safety_adjusted_reward(2.0, {"harvest": 1.2}, 3.0) == 2.0
    with pytest.raises(ValueError, match="survival weight"):
        FarmerTrainOptions(survival_weight=21.0)
