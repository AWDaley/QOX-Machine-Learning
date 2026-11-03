"""Global farmer sweep selection, independent audit and CPU smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _baseline(path: Path) -> None:
    torch = pytest.importorskip("torch")
    from qox.entities import Species
    from qox.learning.farmer_ppo import FarmerPolicy, _action_names
    from qox.perception import FARMER_OBSERVATION_VERSION, features_for

    torch.manual_seed(999)
    model = FarmerPolicy(16)
    torch.save({"species": "farmer", "model_hidden": 16,
                "policy_state": model.state_dict(),
                "obs_version": FARMER_OBSERVATION_VERSION,
                "observation_features": features_for(Species.FARMER),
                "actions": _action_names()}, path)


def test_sweep_budget_estimate_and_seed_validation():
    pytest.importorskip("torch")
    from qox.learning.farmer_sweep import FarmerSweepOptions, _slug

    estimate = FarmerSweepOptions().estimate()
    assert estimate["configurations"] == 6
    assert estimate["training_runs"] == 72
    assert estimate["approximate_ppo_decisions"] == 184_320
    assert estimate["development_world_episodes"] == 6 * 3 * 2 * 3 * 48
    assert _slug(1.000001, 64) != _slug(1.000002, 64)
    for kwargs in ({"survival_weights": ()}, {"stage_budgets": ()},
                   {"survival_weights": (1.0, 1.0)},
                   {"stage_budgets": (64, 64)},
                   {"survival_weights": (0.0,)},
                   {"stage_budgets": (-1,)},
                   {"development_seed": 2_600_000},
                   {"survival_weights": tuple(range(1, 14))}):
        with pytest.raises(ValueError):
            FarmerSweepOptions(**kwargs)


def test_sweep_cli_dry_run_never_loads_baseline_or_writes(tmp_path, capsys):
    pytest.importorskip("torch")
    from qox.cli import main

    output = tmp_path / "never_written"
    assert main(["sweep-farmers", "--baseline", str(tmp_path / "missing.pt"),
                 "--output-dir", str(output), "--weights", "1", "3",
                 "--stage-budgets", "64", "--training-seeds", "17",
                 "--rollout", "64", "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["training_runs"] == 8
    assert not output.exists()


def test_sweep_cli_runs_two_real_configs_without_premature_audit(tmp_path):
    pytest.importorskip("torch")
    from qox.cli import main

    baseline = tmp_path / "original.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    location = tmp_path / "sweep"
    command = ["sweep-farmers", "--baseline", str(baseline),
               "--output-dir", str(location), "--weights", "1", "3",
               "--stage-budgets", "64", "--training-seeds", "17",
               "--worlds", "2", "--rollout", "64", "--epochs", "1",
               "--episode-cap", "12", "--development-episodes", "2",
               "--audit-episodes", "2"]
    assert main(command) == 0
    report = json.loads((location / "summary.json").read_text(encoding="utf-8"))
    assert len(report["configurations"]) == 2
    assert report["selected_before_audit"] is None
    assert report["audit"] is None
    assert report["eligible_for_manual_review"] is False
    assert all(len(c["replicas"]) == 1 for c in report["configurations"])
    for config in report["configurations"]:
        sub = json.loads((Path(config["location"]) / "summary.json").read_text())
        assert sub["audit_deferred"] is True
    assert baseline.read_bytes() == original
    with pytest.raises(FileExistsError):
        main(command)


def test_sweep_audits_exactly_one_global_development_winner(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from qox.learning import farmer_sweep
    from qox.learning.farmer_sweep import FarmerSweepOptions, run_farmer_sweep

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    training_calls = []
    audit_calls = []
    reference_calls = []

    def fake_staged(source, location, options, *, audit_enabled):
        training_calls.append((options.survival_weight, options.steps_per_stage,
                               audit_enabled))
        return {"replicas": [{"training_seed": seed,
                             "candidate": str(location / f"seed_{seed}.pt"),
                             "food_gain": (4.0 if options.survival_weight == 3.0 and
                                           options.steps_per_stage == 128 and seed == 37
                                           else 1.0),
                             "development_passed": True,
                             "development_checks": {"survival": True},
                             "development_report": str(location / f"seed_{seed}.json")}
                            for seed in options.training_seeds]}

    def fake_compare(source, candidate, *, episodes, seed, max_steps):
        audit_calls.append((str(candidate), episodes, seed))
        return {"promotion_checks": {"survival": True},
                "cases": {scenario: {"metrics": {
                    "baseline": {"mean_final_health": 80, "mean_crop_eaten_by_cows": 1,
                                 "mean_deposited": 2},
                    "candidate": {"mean_final_health": 80, "mean_crop_eaten_by_cows": 1,
                                  "mean_deposited": 3}}}
                    for scenario in ("farmstead", "farmstead_threat")}}

    def fake_reference(*, episodes, seed, scenario, max_steps, mode):
        reference_calls.append((scenario, mode, seed))
        return {"scenario": scenario, "mode": mode, "episodes": episodes}

    monkeypatch.setattr(farmer_sweep, "train_farmer_staged", fake_staged)
    monkeypatch.setattr(farmer_sweep, "compare_farmers", fake_compare)
    monkeypatch.setattr(farmer_sweep, "evaluate_farmer", fake_reference)
    options = FarmerSweepOptions(survival_weights=(1.0, 3.0),
                                 stage_budgets=(64, 128))
    report = run_farmer_sweep(baseline, tmp_path / "experiment", options)
    assert len(training_calls) == 4
    assert all(not enabled for _, _, enabled in training_calls)
    assert len(audit_calls) == 1
    assert audit_calls[0][1:] == (48, 2_600_000)
    assert "seed_37.pt" in audit_calls[0][0]
    assert report["selected_before_audit"]["survival_weight"] == 3.0
    assert report["selected_before_audit"]["stage_budget"] == 128
    assert report["eligible_for_manual_review"]
    assert len(reference_calls) == 4
    assert baseline.read_bytes() == original


def test_no_qualified_candidate_never_reads_audit_seed(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from qox.learning import farmer_sweep
    from qox.learning.farmer_sweep import FarmerSweepOptions, run_farmer_sweep

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)

    def fake_staged(source, location, options, *, audit_enabled):
        return {"replicas": [{"training_seed": seed,
                             "candidate": str(location / f"seed_{seed}.pt"),
                             "food_gain": 5.0, "development_passed": False,
                             "development_checks": {"survival": False},
                             "development_report": str(location / f"seed_{seed}.json")}
                            for seed in options.training_seeds]}

    def forbidden(*args, **kwargs):
        raise AssertionError("Audit must remain untouched when no development candidate qualifies")

    monkeypatch.setattr(farmer_sweep, "train_farmer_staged", fake_staged)
    monkeypatch.setattr(farmer_sweep, "compare_farmers", forbidden)
    monkeypatch.setattr(farmer_sweep, "evaluate_farmer", forbidden)
    report = run_farmer_sweep(baseline, tmp_path / "experiment",
                              FarmerSweepOptions(survival_weights=(1.0,),
                                                 stage_budgets=(64,)))
    assert report["development_qualifiers"] == 0
    assert report["audit"] is None
