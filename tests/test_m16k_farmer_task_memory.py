"""M16k: locally verified farm-event memory, matched ablation and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qox.entities import Event, Species
from qox.learning.farmer_task_memory import (FARMER_TASK_HISTORY_FEATURES,
                                            FarmerTaskHistory, farmer_policy_observation)
from qox.perception import ACTIONS, features_for


def _baseline(path: Path) -> None:
    torch = pytest.importorskip("torch")
    from qox.learning.farmer_ppo import FarmerPolicy, _action_names
    from qox.perception import FARMER_OBSERVATION_VERSION

    torch.manual_seed(103)
    model = FarmerPolicy(16)
    torch.save({"species": "farmer", "model_hidden": 16,
                "policy_state": model.state_dict(),
                "obs_version": FARMER_OBSERVATION_VERSION,
                "observation_features": features_for(Species.FARMER),
                "actions": _action_names()}, path)


def test_verified_memory_ignores_other_actors_failed_actions_and_expires():
    memory = FarmerTaskHistory()
    assert len(memory.features()) == len(FARMER_TASK_HISTORY_FEATURES) == 14
    assert not memory.features()[:5].any()
    memory.advance([Event(1, "harvest", 5, 2.5),
                    Event(1, "plant", 8, 0.3),
                    Event(1, "fertilize", 5, 0),
                    Event(1, "move", 5, 1)], 5)
    values = dict(zip(FARMER_TASK_HISTORY_FEATURES, memory.features(), strict=True))
    assert values["recent_harvest"] == pytest.approx(0.25)
    assert values["last_harvest"] == 1
    assert values["recent_plant"] == values["last_plant"] == 0
    assert values["since_harvest"] == values["since_last_success"] == 0
    memory.advance([], 5, steps=16)
    values = dict(zip(FARMER_TASK_HISTORY_FEATURES, memory.features(), strict=True))
    assert values["since_harvest"] == pytest.approx(0.5)
    memory.advance([], 5, steps=16)
    assert memory.features()[0:5].sum() == 0
    assert memory.features()[5] == 1
    with pytest.raises(ValueError, match="at least one"):
        memory.advance([], 5, steps=0)


def test_m4_upgrade_is_neutral_and_v2_checkpoint_strict(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.learning.farmer_ppo import _expand_m4_farmer, load_farmer

    baseline = tmp_path / "m4.pt"
    _baseline(baseline)
    original = load_farmer(baseline)
    observation = np.arange(len(features_for(Species.FARMER)), dtype=np.float32) / 100
    mask = torch.ones((1, len(ACTIONS[Species.FARMER])), dtype=torch.float32)
    with torch.no_grad():
        old_distribution, old_value = original(torch.tensor(observation)[None], mask)
    for history_mode in ("masked", "events"):
        expanded = _expand_m4_farmer(original, history_mode)
        payload = farmer_policy_observation(observation, history_mode,
                                            FarmerTaskHistory())
        with torch.no_grad():
            new_distribution, new_value = expanded(torch.tensor(payload)[None], mask)
        assert torch.allclose(new_distribution.logits, old_distribution.logits, atol=1e-6)
        assert torch.allclose(new_value, old_value, atol=1e-6)
        assert expanded.encoder[0].in_features == len(observation) + 14
        saved = torch.load(baseline, map_location="cpu", weights_only=True)
        saved["policy_state"] = expanded.state_dict()
        saved["obs_version"] = 2
        saved["task_history_mode"] = history_mode
        saved["observation_features"] = (features_for(Species.FARMER) +
                                         FARMER_TASK_HISTORY_FEATURES)
        target = tmp_path / f"{history_mode}.pt"
        torch.save(saved, target)
        assert load_farmer(target).history_mode == history_mode
        saved["observation_features"] = features_for(Species.FARMER)
        torch.save(saved, target)
        with pytest.raises(ValueError, match="schema mismatch"):
            load_farmer(target)


def test_sweep_two_modes_budget_and_cli_noncreating_dry_run(tmp_path, capsys):
    pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.farmer_sweep import FarmerSweepOptions, _slug

    opts = FarmerSweepOptions(survival_weights=(3,), stage_budgets=(4096,),
                              interleave_cycles=(4,), manure_credits=(0.3,),
                              food_anchor_kls=(2,), history_modes=("masked", "events"),
                              epochs=2)
    assert opts.estimate()["configurations"] == 2
    assert opts.estimate()["approximate_ppo_decisions"] == 98_304
    assert _slug(3, 4096, 0, 4, 0.3, 2, "events").endswith("_history_events")
    output = tmp_path / "not_created"
    assert main(["sweep-farmers", "--baseline", str(tmp_path / "missing.pt"),
                 "--output-dir", str(output), "--weights", "3",
                 "--stage-budgets", "4096", "--interleave-cycles", "4",
                 "--manure-credits", "0.3", "--food-anchor-kls", "2",
                 "--history-modes", "masked", "events", "--epochs", "2",
                 "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["approximate_ppo_decisions"] == 98_304
    assert not output.exists()
    with pytest.raises(ValueError, match="distinct"):
        FarmerSweepOptions(history_modes=("masked", "masked"))
    with pytest.raises(ValueError, match="history"):
        FarmerSweepOptions(history_modes=("invented",))


def test_real_event_memory_training_evaluation_controller_and_diagnostics(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main
    from qox.envs.core import QoxCoreEnv
    from qox.learning.farmer_ppo import evaluate_farmer, load_farmer
    from qox.learning.inference import FarmerController

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    v2 = tmp_path / "events.pt"
    assert main(["train-farmer", "--output", str(v2),
                 "--steps", "128", "--worlds", "2", "--rollout", "64",
                 "--max-steps", "18", "--history-mode", "events",
                 "--food-anchor-kl", "2", "--food-anchor-checkpoint", str(baseline)]) == 0
    assert load_farmer(v2).history_mode == "events"
    data = torch.load(v2, map_location="cpu", weights_only=True)
    assert data["obs_version"] == 2
    assert data["task_history_mode"] == "events"
    assert data["observation_features"][-14:] == FARMER_TASK_HISTORY_FEATURES
    report = evaluate_farmer(v2, episodes=2, max_steps=32,
                             scenario="farmstead", chain_diagnostics=True)
    assert len(report["results"]) == 2
    for row in report["results"]:
        chain = row["chain"]
        assert chain["harvest_chosen_steps"] <= chain["harvest_legal_steps"]
        assert (chain["harvest_legal_steps"] == chain["harvest_missed_steps"] +
                chain["harvest_chosen_steps"])
        assert chain["deposit_chosen_steps"] <= chain["deposit_legal_steps"]

    env = QoxCoreEnv(scenario="farmstead", max_steps=32)
    env.reset(seed=50)
    controller = FarmerController(v2)
    actor = next(a for a in env.world.actors.values() if a.species == Species.FARMER)
    controller.probabilities(env.world, actor.id)
    assert controller._actor_history(env.world, actor.id).tick == 0
    env.world.tick += 1
    env.world.last_events = [Event(env.world.tick, "plant", actor.id, 0.3)]
    controller.probabilities(env.world, actor.id)
    controller.probabilities(env.world, actor.id)
    history = controller._actor_history(env.world, actor.id)
    assert history.tick == 1
    assert history.last_kind == "plant"
    assert baseline.read_bytes() == original


def test_real_matched_two_mode_sweep_never_opens_audit_without_pass(tmp_path):
    pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.farmer_ppo import load_farmer

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    output = tmp_path / "sweep"
    assert main(["sweep-farmers", "--baseline", str(baseline),
                 "--output-dir", str(output), "--weights", "3",
                 "--stage-budgets", "256", "--interleave-cycles", "4",
                 "--manure-credits", "0.3", "--food-anchor-kls", "2",
                 "--history-modes", "masked", "events", "--training-seeds", "17",
                 "--worlds", "2", "--rollout", "64", "--epochs", "1",
                 "--episode-cap", "18", "--development-episodes", "2",
                 "--audit-episodes", "2"]) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [arm["history_mode"] for arm in summary["configurations"]] == [
        "masked", "events"]
    assert summary["audit"] is None and summary["development_qualifiers"] == 0
    assert not (output / "audit.json").exists()
    for arm in summary["configurations"]:
        seed_dir = Path(arm["location"]) / "seed_17"
        checkpoints = list(seed_dir.glob("*.pt"))
        assert len(checkpoints) == 16
        assert load_farmer(checkpoints[-1]).history_mode == arm["history_mode"]
    assert baseline.read_bytes() == original
