"""M14 training laboratory: paired outcomes, cancellation and rollback semantics."""
from __future__ import annotations

import pytest

from qox.entities import Species
from qox.training_lab import (CheckpointHistory, TrainingCancelled, choose_default,
                              compare_checkpoints, evaluation_seed, scenarios_for,
                              validate_scenario)

pytest.importorskip("torch")

from qox.desktop_models import ModelSession, checkpoint_information, train_role  # noqa: E402


@pytest.fixture
def session(tmp_path):
    return ModelSession.fresh(tmp_path / "models", seed=200)


def test_scenario_contracts_filter_legacy_networks():
    assert "cow_signal" in scenarios_for(Species.COW, 2)
    assert "cow_signal" not in scenarios_for(Species.COW, 1)
    assert "nomad_signal" not in scenarios_for(Species.NOMAD, 1)
    assert choose_default(Species.COW, 1) == "forage"
    with pytest.raises(ValueError, match="incompatible"):
        validate_scenario(Species.NOMAD, 1, "nomad_signal")
    with pytest.raises(ValueError, match="incompatible"):
        validate_scenario(Species.RAVAGER, 2, "farmer_mixed")
    assert evaluation_seed(44) >= 10_000_000


@pytest.mark.parametrize("species", list(Species))
def test_progress_report_and_scenarios_for_all_species(species, session, tmp_path):
    before = session.path(species).read_bytes()
    target = tmp_path / f"candidate-{species.value}.pt"
    scenario = choose_default(species, session.status[species].observation_version)
    updates = []
    report = train_role(species, session.path(species), target,
                        steps=256, seed=18, scenario=scenario,
                        progress_callback=updates.append)
    assert target.exists() and len(updates) == 1
    assert updates[0]["total"] == 256 and updates[0]["steps"] >= 256
    assert "completed_episodes" in updates[0]
    assert "loss" in updates[0]
    assert report["scenario"] == scenario
    assert session.path(species).read_bytes() == before  # Never automatic install
    assert checkpoint_information(species, target)["steps"] >= 256


def test_cancel_at_boundary_never_creates_candidate(session, tmp_path):
    before = session.path(Species.COW).read_bytes()
    target = tmp_path / "candidate-cancelled.pt"
    with pytest.raises(TrainingCancelled):
        train_role(Species.COW, session.path(Species.COW), target,
                   scenario="cow_signal", steps=512, seed=51,
                   cancel_requested=lambda: True)
    assert not target.exists() and session.path(Species.COW).read_bytes() == before


def test_cancel_after_one_complete_update_does_not_replace_active(session, tmp_path):
    target = tmp_path / "candidate-cancelled.pt"
    updates = []
    with pytest.raises(TrainingCancelled):
        train_role(Species.COW, session.path(Species.COW), target,
                   scenario="cow_signal", steps=768, seed=51,
                   progress_callback=updates.append,
                   cancel_requested=lambda: len(updates) >= 1)
    assert len(updates) == 1 and not target.exists()


def test_strict_paired_comparison_does_not_mutate_active(session, tmp_path):
    role = Species.COW
    before = session.path(role).read_bytes()
    target = tmp_path / "candidate.pt"
    train_role(role, session.path(role), target, steps=256, seed=39,
               scenario="cow_signal")
    comparison = compare_checkpoints(role, session.path(role), target,
                                     seed=39, scenario="cow_signal", episodes=4)
    assert comparison["metric"] == "pasture_arrival_rate"
    assert comparison["episodes"] == 4
    assert len(comparison["paired_deltas"]) == 4
    assert comparison["seed_start"] == evaluation_seed(39)
    assert comparison["active_report"]["seed_start"] == comparison["seed_start"]
    assert comparison["candidate_report"]["seed_start"] == comparison["seed_start"]
    assert comparison["scripted_report"]["seed_start"] == comparison["seed_start"]
    assert session.path(role).read_bytes() == before


def test_accepted_model_can_be_rolled_back_without_undoing_other_roles(session, tmp_path):
    role = Species.FARMER
    other = session.path(Species.COW).read_bytes()
    original = session.path(role).read_bytes()
    candidate = tmp_path / "farmer-candidate.pt"
    train_role(role, session.path(role), candidate, steps=256, seed=21,
               scenario="farmer_mixed")
    undo = CheckpointHistory(tmp_path / "undo", max_versions=2)
    assert undo.rollback(session, role) is False
    undo.remember(session, role)
    session.replace(role, candidate, origin="trained")
    assert session.path(role).read_bytes() == candidate.read_bytes()
    assert undo.rollback(session, role) is True
    assert session.path(role).read_bytes() == original
    assert session.path(Species.COW).read_bytes() == other
    assert undo.rollback(session, role) is False
    undo.clear()


def test_checkpoint_optimizer_continuity(session, tmp_path):
    import torch

    role = Species.SCAVENGER
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    train_role(role, session.path(role), first, steps=256, seed=7,
               scenario="scavenger_waste")
    train_role(role, first, second, steps=256, seed=8,
               scenario="scavenger_carrion")
    a = torch.load(first, map_location="cpu", weights_only=True)
    b = torch.load(second, map_location="cpu", weights_only=True)
    assert b["completed_steps"] > a["completed_steps"]
    # Adam update counter has continued, not been reset to a new optimizer.
    key = next(iter(a["optimizer_state"]["state"]))
    bkey = next(iter(b["optimizer_state"]["state"]))
    assert b["optimizer_state"]["state"][bkey]["step"].item() > (
        a["optimizer_state"]["state"][key]["step"].item())
