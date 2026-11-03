"""M16g fixed M4 farmer demonstrations and development-only KL sweep."""

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


def test_frozen_teacher_replay_is_deterministic_and_masked(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.learning.farmer_rehearsal import collect_farmer_rehearsal

    checkpoint = tmp_path / "frozen.pt"
    _baseline(checkpoint)
    original = checkpoint.read_bytes()
    a = collect_farmer_rehearsal(checkpoint, seed=1_200_000,
                                 episodes=2, max_steps=12)
    b = collect_farmer_rehearsal(checkpoint, seed=1_200_000,
                                 episodes=2, max_steps=12)
    assert a.metadata["total_states"] == 72
    assert a.metadata["sample_sha256"] == b.metadata["sample_sha256"]
    assert a.metadata["baseline_sha256"] == b.metadata["baseline_sha256"]
    assert a.metadata["scenarios"].keys() == {"agriculture", "manure", "farmstead"}
    assert torch.allclose(a.teacher_probs.sum(dim=1), torch.ones(72), atol=1e-6)
    assert torch.all(a.teacher_probs[a.masks == 0] < 1e-6)
    assert set(a.sampling_weights.tolist()) <= {1.0, 2.0, 8.0}
    assert torch.equal(a.observations, b.observations)
    assert checkpoint.read_bytes() == original


def test_replay_kl_uses_teacher_distribution_with_descending_gradient(tmp_path):
    torch = pytest.importorskip("torch")
    from torch.distributions import Categorical
    from torch.distributions.kl import kl_divergence
    from qox.learning.farmer_ppo import load_farmer
    from qox.learning.farmer_rehearsal import collect_farmer_rehearsal

    checkpoint = tmp_path / "frozen.pt"
    _baseline(checkpoint)
    replay = collect_farmer_rehearsal(checkpoint, episodes=1, max_steps=12)
    teacher = Categorical(probs=replay.teacher_probs)
    baseline = load_farmer(checkpoint)
    reference, _ = baseline(replay.observations, replay.masks)
    assert kl_divergence(teacher, reference).mean().item() < 1e-6

    student = load_farmer(checkpoint)
    with torch.no_grad():
        student.actor.bias[0] += 2.0
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    first = None
    for _ in range(12):
        distribution, _ = student(replay.observations, replay.masks)
        loss = kl_divergence(teacher, distribution).mean()
        if first is None:
            first = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert first is not None and first > 0.01
    distribution, _ = student(replay.observations, replay.masks)
    assert float(kl_divergence(teacher, distribution).mean().detach()) < first


def test_replay_kl_requires_training_demonstrations(tmp_path):
    pytest.importorskip("torch")
    from qox.learning.farmer_ppo import FarmerTrainOptions, train_farmer

    output = tmp_path / "not_created.pt"
    with pytest.raises(ValueError, match="demonstrations"):
        train_farmer(FarmerTrainOptions(rehearsal_kl=1.0), output)
    assert not output.exists()
    with pytest.raises(ValueError, match="rehearsal KL"):
        FarmerTrainOptions(rehearsal_kl=-0.5)


def test_sweep_rehearsal_validation_and_dry_run(tmp_path, capsys):
    pytest.importorskip("torch")
    from qox.cli import main
    from qox.learning.farmer_sweep import FarmerSweepOptions

    for changes in ({"rehearsal_kls": ()}, {"rehearsal_kls": (0, 0)},
                    {"rehearsal_kls": (-0.1,)}, {"rehearsal_kls": (11,)},
                    {"rehearsal_kls": (0.5,), "rehearsal_seed": 1_500_000},
                    {"rehearsal_kls": (0.5,), "rehearsal_episodes": 0}):
        with pytest.raises(ValueError):
            FarmerSweepOptions(**changes)
    assert FarmerSweepOptions(survival_weights=(3,), stage_budgets=(64,),
                              rehearsal_kls=(0, 0.5)).estimate()["rehearsal_world_episodes"] == 12
    location = tmp_path / "not_created"
    assert main(["sweep-farmers", "--baseline", "missing.pt", "--output-dir",
                 str(location), "--weights", "3", "--stage-budgets", "64",
                 "--rehearsal-kls", "0", "0.5", "--training-seeds", "17",
                 "--rollout", "64", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["training_runs"] == 8
    assert not location.exists()


def test_rehearsal_sweep_runs_control_and_anchor_without_auto_promotion(tmp_path):
    torch = pytest.importorskip("torch")
    from qox.cli import main

    baseline = tmp_path / "original.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    output = tmp_path / "experiment"
    command = ["sweep-farmers", "--baseline", str(baseline), "--output-dir",
               str(output), "--weights", "3", "--stage-budgets", "64",
               "--rehearsal-kls", "0", "0.5", "--training-seeds", "17",
               "--worlds", "2", "--rollout", "64", "--epochs", "1",
               "--episode-cap", "12", "--rehearsal-episodes", "1",
               "--rehearsal-max-steps", "12", "--development-episodes", "2",
               "--audit-episodes", "2"]
    assert main(command) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [config["rehearsal_kl"] for config in summary["configurations"]] == [0, 0.5]
    assert summary["selected_before_audit"] is None
    assert summary["audit"] is None
    assert not summary["eligible_for_manual_review"]
    control, anchored = summary["configurations"]
    assert control["rehearsal"] is None
    assert anchored["rehearsal"]["total_states"] == 36
    assert anchored["rehearsal"]["baseline_sha256"] == summary["baseline_sha256"]
    anchor_checkpoint = Path(anchored["replicas"][0]["candidate"])
    weights = torch.load(anchor_checkpoint, map_location="cpu", weights_only=True)
    assert weights["options"]["rehearsal_kl"] == 0.5
    assert baseline.read_bytes() == original
    with pytest.raises(FileExistsError):
        main(command)


def test_global_rehearsal_selection_has_one_blind_audit(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from qox.learning import farmer_sweep
    from qox.learning.farmer_sweep import FarmerSweepOptions, run_farmer_sweep

    baseline = tmp_path / "baseline.pt"
    _baseline(baseline)
    original = baseline.read_bytes()
    calls = []
    audit_calls = []

    def fake_staged(source, location, options, *, audit_enabled):
        calls.append((options.rehearsal_kl, audit_enabled))
        return {"replicas": [{"training_seed": seed,
                             "candidate": str(location / f"seed_{seed}.pt"),
                             "food_gain": (4.0 if options.rehearsal_kl == 0.5 and seed == 37
                                           else 1.0),
                             "development_passed": True,
                             "development_checks": {"survival": True},
                             "development_report": str(location / f"seed_{seed}.json")}
                            for seed in options.training_seeds]}

    def fake_compare(source, candidate, *, episodes, seed, max_steps):
        audit_calls.append((candidate, seed))
        return {"promotion_checks": {"passed": True},
                "cases": {scenario: {"metrics": {
                    "baseline": {"mean_final_health": 80, "mean_crop_eaten_by_cows": 1,
                                 "mean_deposited": 2},
                    "candidate": {"mean_final_health": 80, "mean_crop_eaten_by_cows": 1,
                                  "mean_deposited": 3}}}
                    for scenario in ("farmstead", "farmstead_threat")}}

    monkeypatch.setattr(farmer_sweep, "train_farmer_staged", fake_staged)
    monkeypatch.setattr(farmer_sweep, "compare_farmers", fake_compare)
    monkeypatch.setattr(farmer_sweep, "evaluate_farmer", lambda **kwargs: {})
    result = run_farmer_sweep(baseline, tmp_path / "research", FarmerSweepOptions(
        survival_weights=(3,), stage_budgets=(64,), rehearsal_kls=(0, 0.5)))
    assert calls == [(0, False), (0.5, False)]
    assert len(audit_calls) == 1
    assert audit_calls[0][1] == 2_600_000
    assert result["selected_before_audit"]["rehearsal_kl"] == 0.5
    assert result["selected_before_audit"]["training_seed"] == 37
    assert result["eligible_for_manual_review"]
    assert baseline.read_bytes() == original
