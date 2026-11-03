"""M16h interleaved on-policy farming, matched budgets and blind audit."""

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


def test_interleaved_options_enforce_full_horizons_and_exact_decisions():
    pytest.importorskip("torch")
    from qox.learning.farmer_staged import (StagedFarmerOptions, _interleave_seed,
                                            _stage_seed)

    good = StagedFarmerOptions(interleave_cycles=4, steps_per_stage=4096)
    assert good.steps_per_stage // good.interleave_cycles // good.worlds == 256
    for changes in ({"interleave_cycles": 0}, {"interleave_cycles": 9},
                    {"interleave_cycles": 4, "steps_per_stage": 1024},
                    {"interleave_cycles": 4, "steps_per_stage": 3000},
                    {"interleave_cycles": 4, "steps_per_stage": 4096,
                     "max_steps_override": 257}):
        with pytest.raises(ValueError, match="Interleave|Interleaved|interleaved"):
            StagedFarmerOptions(**changes)
    fresh = {_interleave_seed(replica, cycle, stage, seed)
             for replica, seed in enumerate((17, 37, 59))
             for cycle in range(8) for stage in range(4)}
    assert len(fresh) == 3 * 8 * 4
    assert all(3_000_000_000 <= value < 2**32 for value in fresh)
    assert fresh.isdisjoint({_stage_seed(replica, stage, seed)
                             for replica, seed in enumerate((17, 37, 59))
                             for stage in range(4)})


def test_interleaved_staged_cli_preserves_baseline_and_updates_every_task(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    output = tmp_path / "interleaved"
    command = ["train-farmer-staged", "--baseline", str(baseline),
               "--output-dir", str(output), "--training-seeds", "17",
               "--stage-steps", "256", "--interleave-cycles", "4",
               "--worlds", "2", "--rollout", "64", "--epochs", "1",
               "--episode-cap", "18", "--development-episodes", "2",
               "--audit-episodes", "2"]
    assert main(command) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    stages = report["replicas"][0]["stages"]
    expected = ["agriculture", "manure", "refuge", "farmstead_mixed"] * 4
    assert report["training_sequence"] == expected
    assert len(stages) == 16
    assert [stage["cycle"] for stage in stages] == [n for n in range(1, 5) for _ in range(4)]
    assert [stage["scenario"] for stage in stages] == expected
    assert all(stage["decisions"] == 64 for stage in stages)
    assert all(Path(stage["checkpoint"]).is_file() for stage in stages)
    assert len({stage["seed"] for stage in stages}) == 16
    assert sum(stage["decisions"] for stage in stages) == 4 * 256
    assert [sum(x["decisions"] for x in stages if x["scenario"] == scenario)
            for scenario in set(expected)] == [256] * 4
    assert [stage["survival_weight"] for stage in stages[:4]] == [1, 1, 3, 3]
    final = torch.load(stages[-1]["checkpoint"], map_location="cpu", weights_only=True)
    assert final["options"]["scenario"] == "farmstead_mixed"
    assert final["options"]["steps"] == 64
    assert not report["eligible_for_manual_review"]
    assert baseline.read_bytes() == original
    with pytest.raises(FileExistsError):
        main(command)


def test_interleaved_sweep_matches_budget_and_keeps_one_global_audit(tmp_path,
                                                                      monkeypatch):
    pytest.importorskip("torch")
    from qox.learning import farmer_sweep
    from qox.learning.farmer_sweep import FarmerSweepOptions, run_farmer_sweep

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    opts = FarmerSweepOptions(survival_weights=(3,), stage_budgets=(256,),
                              interleave_cycles=(1, 4), training_seeds=(17, 37, 59),
                              worlds=2, rollout=64, max_steps_override=18)
    estimate = opts.estimate()
    assert estimate["configurations"] == 2
    assert estimate["training_runs"] == (1 + 4) * 4 * 3
    assert estimate["approximate_ppo_decisions"] == 2 * 4 * 256 * 3

    training, audits = [], []

    def fake_staged(source, location, options, *, audit_enabled):
        training.append((options.interleave_cycles, options.steps_per_stage,
                         audit_enabled))
        return {"replicas": [{"training_seed": seed,
                             "candidate": str(location / f"seed_{seed}.pt"),
                             "food_gain": (3.0 if options.interleave_cycles == 4 and
                                           seed == 37 else 1.0),
                             "development_passed": True,
                             "development_checks": {"passed": True},
                             "development_report": str(location / f"seed_{seed}.json")}
                            for seed in options.training_seeds]}

    def fake_compare(source, candidate, *, episodes, seed, max_steps):
        audits.append((candidate, seed))
        return {"promotion_checks": {"safe": True},
                "cases": {scenario: {"metrics": {
                    "baseline": {"mean_final_health": 80, "mean_crop_eaten_by_cows": 1,
                                 "mean_deposited": 2},
                    "candidate": {"mean_final_health": 80, "mean_crop_eaten_by_cows": 1,
                                  "mean_deposited": 3}}}
                    for scenario in ("farmstead", "farmstead_threat")}}

    monkeypatch.setattr(farmer_sweep, "train_farmer_staged", fake_staged)
    monkeypatch.setattr(farmer_sweep, "compare_farmers", fake_compare)
    monkeypatch.setattr(farmer_sweep, "evaluate_farmer", lambda **kwargs: {})
    result = run_farmer_sweep(baseline, tmp_path / "sweep", opts)
    assert training == [(1, 256, False), (4, 256, False)]
    assert len(audits) == 1 and audits[0][1] == 2_600_000
    assert result["selected_before_audit"]["interleave_cycles"] == 4
    assert result["selected_before_audit"]["training_seed"] == 37
    assert result["eligible_for_manual_review"]
    assert baseline.read_bytes() == original


def test_interleaved_cli_dry_run_never_creates_or_loads(tmp_path, capsys):
    pytest.importorskip("torch")
    from qox.cli import main

    dest = tmp_path / "not_created"
    assert main(["sweep-farmers", "--baseline", "missing.pt", "--output-dir",
                 str(dest), "--weights", "3", "--stage-budgets", "256",
                 "--interleave-cycles", "1", "4", "--training-seeds", "17",
                 "--worlds", "2", "--rollout", "64", "--episode-cap", "18",
                 "--dry-run"]) == 0
    values = json.loads(capsys.readouterr().out)
    assert values["training_runs"] == 20
    assert values["approximate_ppo_decisions"] == 2048
    assert not dest.exists()


def test_real_sweep_control_and_four_cycles_remain_non_promoting(tmp_path):
    pytest.importorskip("torch")
    from qox.cli import main

    checkpoint = tmp_path / "baseline.pt"
    _baseline(checkpoint)
    original = checkpoint.read_bytes()
    output = tmp_path / "sweep"
    command = ["sweep-farmers", "--baseline", str(checkpoint),
               "--output-dir", str(output), "--weights", "3",
               "--stage-budgets", "256", "--interleave-cycles", "1", "4",
               "--training-seeds", "17", "--worlds", "2", "--rollout", "64",
               "--epochs", "1", "--episode-cap", "18",
               "--development-episodes", "2", "--audit-episodes", "2"]
    assert main(command) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [config["interleave_cycles"] for config in report["configurations"]] == [1, 4]
    assert len(list(output.rglob("*.pt"))) == 4 + 16
    assert report["selected_before_audit"] is None
    assert report["audit"] is None
    assert not report["eligible_for_manual_review"]
    assert checkpoint.read_bytes() == original
