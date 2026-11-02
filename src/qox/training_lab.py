"""Validated training curricula, held-out comparisons and reversible model promotion.

This module owns no GUI and never mutates a running QOX world.  A training run
creates a candidate; adopting it remains a separate user decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from statistics import mean, pstdev
from typing import Any

from .entities import Species


class TrainingCancelled(Exception):
    """User requested cancellation at a safe PPO rollout boundary."""


SCENARIOS: dict[Species, tuple[str, ...]] = {
    Species.COW: ("forage", "threat", "nomad_herd", "cow_signal"),
    Species.FARMER: ("agriculture", "refuge", "manure", "farmer_mixed"),
    Species.NOMAD: ("nomad_signal", "nomad_herd", "nomad_defend", "nomad_mixed"),
    Species.SCAVENGER: ("scavenger_waste", "scavenger_carrion",
                        "scavenger_competition", "scavenger_mixed"),
    Species.RAVAGER: ("ravager_ambush", "ravager_carcass", "ravager_hibernate",
                      "ravager_mixed"),
}

DEFAULT_SCENARIOS = {
    Species.COW: "cow_signal",
    Species.FARMER: "farmer_mixed",
    Species.NOMAD: "nomad_signal",
    Species.SCAVENGER: "scavenger_mixed",
    Species.RAVAGER: "ravager_mixed",
}

# Primary test metrics are observable task outcomes, not training reward.
METRICS: dict[Species, dict[str, tuple[str, str, str]]] = {
    Species.COW: {
        "forage": ("mean_grazed", "grazed", "Food grazed"),
        "threat": ("survival_rate", "alive", "Survival"),
        "nomad_herd": ("pasture_arrival_rate", "in_pasture", "Pasture arrivals"),
        "cow_signal": ("pasture_arrival_rate", "in_pasture", "Target arrivals"),
    },
    Species.FARMER: {
        "agriculture": ("mean_deposited", "deposited", "Food deposited"),
        "refuge": ("mean_final_health", "final_health", "Final health"),
        "manure": ("mean_fertilized", "fertilized", "Manure applied"),
        "farmer_mixed": ("mean_return", "return", "Episode reward (mixed)"),
    },
    Species.NOMAD: {
        "nomad_signal": ("mean_cows_in_pasture", "cows_in_pasture", "Call target arrivals"),
        "nomad_herd": ("mean_cows_in_pasture", "cows_in_pasture", "Cows to pasture"),
        "nomad_defend": ("mean_cows_alive", "cows_alive", "Surviving cows"),
        "nomad_mixed": ("mean_return", "return", "Episode reward (mixed)"),
    },
    Species.SCAVENGER: {
        "scavenger_waste": ("mean_waste_eaten", "waste_eaten", "Waste consumed"),
        "scavenger_carrion": ("mean_carrion_eaten", "carrion_eaten", "Meat consumed"),
        "scavenger_competition": ("mean_waste_eaten", "waste_eaten", "Waste consumed"),
        "scavenger_mixed": ("mean_return", "return", "Episode reward (mixed)"),
    },
    Species.RAVAGER: {
        "ravager_ambush": ("mean_charge_hits", "charge_hits", "Charge hits"),
        "ravager_carcass": ("mean_meat_eaten", "meat_eaten", "Meat consumed"),
        "ravager_hibernate": ("mean_final_energy", "final_energy", "Final energy"),
        "ravager_mixed": ("mean_return", "return", "Episode reward (mixed)"),
    },
}


def scenarios_for(species: Species, observation_version: int) -> tuple[str, ...]:
    """Only show curricula compatible with this checkpoint's fixed inputs."""
    available = SCENARIOS[species]
    if species == Species.COW and observation_version == 1:
        return ("forage", "threat")
    if species == Species.NOMAD and observation_version == 1:
        return tuple(s for s in available if s != "nomad_signal")
    return available


def choose_default(species: Species, observation_version: int) -> str:
    options = scenarios_for(species, observation_version)
    preferred = DEFAULT_SCENARIOS[species]
    return preferred if preferred in options else options[0]


def validate_scenario(species: Species, observation_version: int, scenario: str) -> None:
    if scenario not in scenarios_for(species, observation_version):
        raise ValueError(f"{scenario!r} is incompatible with the {species.value} "
                         f"observation schema v{observation_version}")


def scenario_steps(species: Species, scenario: str) -> int:
    if scenario == "nomad_signal":
        return 8
    if scenario == "cow_signal":
        return 24
    return {Species.COW: 64, Species.FARMER: 96, Species.NOMAD: 96,
            Species.SCAVENGER: 48, Species.RAVAGER: 64}[species]


def evaluation_seed(training_seed: int) -> int:
    """Hold evaluation seeds well outside the current short PPO curricula."""
    return training_seed + 10_000_000


def evaluate_checkpoint(species: Species, checkpoint: Path, *, scenario: str,
                        episodes: int, seed: int, mode: str = "policy") -> dict:
    """Evaluate a frozen checkpoint; never update policy weights."""
    from .desktop_models import checkpoint_information

    if episodes < 1 or episodes > 200:
        raise ValueError("Evaluation episodes must be between 1 and 200")
    info = checkpoint_information(species, Path(checkpoint))
    validate_scenario(species, info["observation_version"], scenario)
    max_steps = scenario_steps(species, scenario)
    if species == Species.COW:
        from .learning.cow_ppo import evaluate_cow
        return evaluate_cow(checkpoint if mode == "policy" else None,
                            episodes=episodes, seed=seed, scenario=scenario,
                            max_steps=max_steps, mode=mode)
    if species == Species.FARMER:
        from .learning.farmer_ppo import evaluate_farmer
        return evaluate_farmer(checkpoint if mode == "policy" else None,
                               episodes=episodes, seed=seed, scenario=scenario,
                               max_steps=max_steps, mode=mode)
    if species == Species.NOMAD:
        from .learning.nomad_ppo import evaluate_nomad
        return evaluate_nomad(checkpoint if mode == "policy" else None,
                              episodes=episodes, seed=seed, scenario=scenario,
                              max_steps=max_steps,
                              mode="policy_greedy" if mode == "policy" else mode)
    if species == Species.SCAVENGER:
        from .learning.scavenger_ppo import evaluate_scavenger
        return evaluate_scavenger(checkpoint if mode == "policy" else None,
                                  episodes=episodes, seed=seed, scenario=scenario,
                                  max_steps=max_steps,
                                  mode="policy_greedy" if mode == "policy" else mode)
    from .learning.ravager_ppo import evaluate_ravager
    return evaluate_ravager(checkpoint if mode == "policy" else None,
                            episodes=episodes, seed=seed, scenario=scenario,
                            max_steps=max_steps,
                            mode="policy_greedy" if mode == "policy" else mode)


def compare_checkpoints(species: Species, active: Path, candidate: Path, *,
                        scenario: str, seed: int, episodes: int = 12) -> dict[str, Any]:
    """Strict paired comparison. Candidate is not installed or promoted here."""
    from .desktop_models import checkpoint_information

    old = checkpoint_information(species, Path(active))
    new = checkpoint_information(species, Path(candidate))
    if old["observation_version"] != new["observation_version"]:
        raise ValueError("Candidate must use the same observation schema")
    validate_scenario(species, new["observation_version"], scenario)
    heldout = evaluation_seed(seed)
    existing = evaluate_checkpoint(species, active, scenario=scenario,
                                   episodes=episodes, seed=heldout)
    proposed = evaluate_checkpoint(species, candidate, scenario=scenario,
                                   episodes=episodes, seed=heldout)
    reference = evaluate_checkpoint(species, active, scenario=scenario,
                                    episodes=episodes, seed=heldout, mode="scripted")
    key, row_key, label = METRICS[species][scenario]
    old_rows = [float(r[row_key]) for r in existing["results"]]
    new_rows = [float(r[row_key]) for r in proposed["results"]]
    paired = [b - a for a, b in zip(old_rows, new_rows, strict=True)]
    return {"species": species.value, "scenario": scenario, "metric": key,
            "metric_label": label, "seed_start": heldout, "episodes": episodes,
            "active": float(existing[key]), "candidate": float(proposed[key]),
            "delta": round(mean(paired), 4),
            "paired_std": round(pstdev(paired), 4),
            "paired_deltas": [round(x, 4) for x in paired],
            "scripted": float(reference[key]),
            "active_report": existing, "candidate_report": proposed,
            "scripted_report": reference,
            "assessment": ("improved" if mean(paired) > 0 else
                           "declined" if mean(paired) < 0 else "tied"),
            "note": "Matched held-out evaluation, not proof of general performance; "
                    "the active model remains unchanged until accepted."}


@dataclass
class CheckpointHistory:
    """Bounded local undo history for a single desktop session."""

    root: Path
    max_versions: int = 5

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._items: dict[Species, list[tuple[Path, str]]] = {}

    def remember(self, session, species: Species) -> None:
        from .desktop_models import checkpoint_information

        stack = self._items.setdefault(species, [])
        target = self.root / f"{species.value}-{len(stack)}-{len(list(self.root.iterdir()))}.pt"
        shutil.copyfile(session.path(species), target)
        try:
            checkpoint_information(species, target)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        stack.append((target, session.status[species].origin))
        while len(stack) > self.max_versions:
            old, _ = stack.pop(0)
            old.unlink(missing_ok=True)

    def rollback(self, session, species: Species) -> bool:
        stack = self._items.get(species, [])
        if not stack:
            return False
        target, origin = stack[-1]
        session.replace(species, target, origin=origin)
        stack.pop()
        target.unlink(missing_ok=True)
        return True

    def clear(self) -> None:
        for stack in self._items.values():
            for path, _ in stack:
                path.unlink(missing_ok=True)
        self._items.clear()
