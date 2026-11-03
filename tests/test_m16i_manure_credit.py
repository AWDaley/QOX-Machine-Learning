"""M16i: physically verified, training-only manure credit and fair sweep."""

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


def test_credit_uses_only_positive_verified_manure_events_for_this_farmer():
    pytest.importorskip("torch")
    from qox.entities import Event
    from qox.learning.farmer_ppo import _manure_event_credit

    events = [Event(2, "collect_waste", 11, 1.5),
              Event(3, "fertilize", 11, 1.0),
              Event(3, "fertilize", 12, 1.0),
              Event(3, "harvest", 11, 5.0),
              Event(3, "fertilize", 11, 0.0),
              Event(3, "collect_waste", 11, -3.0)]
    assert _manure_event_credit(events, 11, 0.0) == (1.5, 1.0, 0.0)
    assert _manure_event_credit(events, 11, 0.5) == (1.5, 1.0, 0.65)
    assert _manure_event_credit(events, 22, 1.0) == (0.0, 0.0, 0.0)
    assert _manure_event_credit([], 11, 2.0) == (0.0, 0.0, 0.0)


def test_real_world_collect_and_apply_conserve_manure_and_evaluation_reward():
    pytest.importorskip("torch")
    from qox.actions import Kind
    from qox.entities import Species
    from qox.envs.core import QoxCoreEnv
    from qox.learning.farmer_ppo import _manure_event_credit
    from qox.perception import ACTIONS

    env = QoxCoreEnv(scenario="manure", max_steps=20, scripted_others=False)
    env.reset(seed=145)
    name = next(name for name in env.possible_agents if name.startswith("farmer_"))
    aid = env.name_to_id[name]
    farmer = env.world.actors[aid]
    tile = env.world.grid.get(*farmer.position)
    tile.waste = 1.5
    tile.fertility = 0.2
    farmer.carried_waste = 0.0
    collect_index = next(i for i, a in enumerate(ACTIONS[Species.FARMER])
                         if a.kind == Kind.COLLECT_WASTE)
    fertilize_index = next(i for i, a in enumerate(ACTIONS[Species.FARMER])
                           if a.kind == Kind.FERTILIZE)

    _, earned, _, _, _ = env.step({name: collect_index})
    assert tile.waste == 0
    assert farmer.carried_waste == 1.5
    assert _manure_event_credit(env.world.last_events, aid, 1) == pytest.approx(
        (1.5, 0, 0.3))
    assert earned[name] < 0.3  # Shaping is deliberately absent from environment rewards.

    _, earned, _, _, _ = env.step({name: fertilize_index})
    assert farmer.carried_waste == pytest.approx(0.5)
    assert tile.fertility == pytest.approx(0.38)
    assert _manure_event_credit(env.world.last_events, aid, 1) == (0, 1, 1)
    assert earned[name] < 1  # Original weak 0.025 fertilizer reward remains intact.

    _, _, _, _, _ = env.step({name: collect_index})
    assert _manure_event_credit(env.world.last_events, aid, 1) == (0, 0, 0)
    tile.fertility = 1.0
    _, _, _, _, _ = env.step({name: fertilize_index})
    assert _manure_event_credit(env.world.last_events, aid, 1) == (0, 0, 0)


def test_strength_validations_and_matched_sweep_costs():
    pytest.importorskip("torch")
    from qox.learning.farmer_ppo import FarmerTrainOptions
    from qox.learning.farmer_staged import StagedFarmerOptions
    from qox.learning.farmer_sweep import FarmerSweepOptions, _slug

    for strength in (-1, 2.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Manure"):
            FarmerTrainOptions(manure_credit=strength)
        with pytest.raises(ValueError, match="Manure"):
            StagedFarmerOptions(manure_credit=strength)
        with pytest.raises(ValueError, match="Manure"):
            FarmerSweepOptions(manure_credits=(strength,))
    opts = FarmerSweepOptions(survival_weights=(3,), stage_budgets=(4096,),
                              interleave_cycles=(4,), manure_credits=(0, 0.3, 1),
                              epochs=2)
    estimate = opts.estimate()
    assert estimate["configurations"] == 3
    assert estimate["approximate_ppo_decisions"] == 3 * 3 * 4 * 4096
    assert estimate["training_runs"] == 3 * 3 * 4 * 4
    assert _slug(3, 4096, 0, 4, 0) == "safety_3_steps_4096_cycles_4"
    assert _slug(3, 4096, 0, 4, 0.3) == "safety_3_steps_4096_cycles_4_manure_0p29999999999999999"
    with pytest.raises(ValueError, match="distinct"):
        FarmerSweepOptions(manure_credits=(0.3, 0.3))


def test_cli_smoke_control_and_bonus_leave_baseline_and_models_untouched(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    output = tmp_path / "experiment"
    command = ["sweep-farmers", "--baseline", str(baseline),
               "--output-dir", str(output), "--weights", "3", "--stage-budgets", "256",
               "--interleave-cycles", "4", "--manure-credits", "0", "0.5",
               "--training-seeds", "17", "--worlds", "2", "--rollout", "64",
               "--epochs", "1", "--episode-cap", "18",
               "--development-episodes", "2", "--audit-episodes", "2"]
    assert main([*command, "--dry-run"]) == 0
    assert not output.exists()
    assert main(command) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [c["manure_credit"] for c in report["configurations"]] == [0, 0.5]
    assert report["audit"] is None and report["recommended_checkpoint"] is None
    assert not report["eligible_for_manual_review"]
    assert len(list(output.rglob("*.pt"))) == 32
    for config in report["configurations"]:
        stages = json.loads((Path(config["location"]) / "summary.json").read_text())[
            "replicas"][0]["stages"]
        assert len(stages) == 16
        assert all(stage["manure_credit_awarded"] >= 0 for stage in stages)
        model = torch.load(stages[-1]["checkpoint"], weights_only=True)
        assert model["options"]["manure_credit"] == config["manure_credit"]
    assert baseline.read_bytes() == original


def test_manure_grid_never_selects_multiple_audits(tmp_path, monkeypatch):
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
                             "food_gain": 2 + options.manure_credit,
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
                              interleave_cycles=(4,), manure_credits=(0, 0.3, 1),
                              development_seed=2_200_000, audit_seed=3_200_000)
    report = run_farmer_sweep(baseline, tmp_path / "run", opts)
    assert len(audited) == 1 and audited[0] == 3_200_000
    assert report["selected_before_audit"]["manure_credit"] == 1
    assert report["selected_before_audit"]["training_seed"] == 17
    assert report["eligible_for_manual_review"]
