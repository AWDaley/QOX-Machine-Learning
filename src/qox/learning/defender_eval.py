"""M16b: paired cattle-defense evaluation with no-attack and migration controls.

Evaluation loads frozen learned checkpoints and uses identical seeds and
conditions for each policy. It never mutates or promotes a checkpoint.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .nomad_ppo import evaluate_nomad


SCENARIOS = ("nomad_guard", "nomad_intercept", "nomad_defend", "nomad_herd")
METRICS = ("mean_cows_alive", "mean_cow_health", "mean_nomad_attacks",
           "mean_nomad_health", "nomad_survival_rate", "ravager_survival_rate",
           "mean_cows_in_pasture", "mean_calls")


def _paired_ci(differences: list[float], seed: int) -> list[float]:
    values = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(seed)
    picks = generator.integers(0, len(values), size=(600, len(values)))
    return [round(float(x), 4) for x in np.percentile(values[picks].mean(1), [2.5, 97.5])]


def compare_defenders(baseline_nomad: str | Path, candidate_nomad: str | Path,
                      cow_checkpoint: str | Path, *, episodes: int = 48,
                      seed: int = 910_000, max_steps: int = 48) -> dict:
    """Compare on identical, held-out guard/intercept/old-defense/migration worlds.

    The success gate is deliberately conservative: preventing cattle losses
    matters most, but the candidate must not forget herding or use attacks
    that have no causal consequence. Guard encounters are easy, intercept is
    harder; the original M5 defense setup is a transfer control.
    """
    if episodes < 1 or max_steps < 1 or seed < 0:
        raise ValueError("episodes and max_steps must be positive and seed nonnegative")
    paths = {"baseline": Path(baseline_nomad), "candidate": Path(candidate_nomad)}
    cow_path = Path(cow_checkpoint)
    torch.set_num_threads(1)
    details: dict[str, dict] = {}
    for index, scenario in enumerate(SCENARIOS):
        start_seed = seed + 200_000 * index
        runs = {}
        for role, path in paths.items():
            runs[role] = evaluate_nomad(
                path, cow_checkpoint=cow_path, mode="policy_greedy",
                scenario=scenario, episodes=episodes, seed=start_seed,
                max_steps=max_steps)
        if scenario in ("nomad_guard", "nomad_intercept"):
            runs["candidate_noattack"] = evaluate_nomad(
                paths["candidate"], cow_checkpoint=cow_path,
                mode="policy_greedy_noattack", scenario=scenario,
                episodes=episodes, seed=start_seed, max_steps=max_steps)
            runs["scripted"] = evaluate_nomad(
                paths["baseline"], cow_checkpoint=cow_path,
                mode="scripted", scenario=scenario, episodes=episodes,
                seed=start_seed, max_steps=max_steps)
        if scenario == "nomad_herd":
            runs["baseline_nocalls"] = evaluate_nomad(
                paths["baseline"], cow_checkpoint=cow_path,
                mode="policy_greedy_nocalls", scenario=scenario,
                episodes=episodes, seed=start_seed, max_steps=max_steps)
            runs["candidate_nocalls"] = evaluate_nomad(
                paths["candidate"], cow_checkpoint=cow_path,
                mode="policy_greedy_nocalls", scenario=scenario,
                episodes=episodes, seed=start_seed, max_steps=max_steps)
        seed_rows = []
        for i in range(episodes):
            row = {"seed": start_seed + i}
            for role, output in runs.items():
                row[role] = {k: output["results"][i][k] for k in
                             ("cows_alive", "mean_cow_health", "nomad_attacks",
                              "nomad_alive", "nomad_health", "ravager_alive",
                              "cows_in_pasture")}
            seed_rows.append(row)
        health_delta = [r["candidate"]["mean_cow_health"] -
                        r["baseline"]["mean_cow_health"] for r in seed_rows]
        survivors_delta = [r["candidate"]["cows_alive"] -
                           r["baseline"]["cows_alive"] for r in seed_rows]
        arrival_delta = [r["candidate"]["cows_in_pasture"] -
                         r["baseline"]["cows_in_pasture"] for r in seed_rows]
        details[scenario] = {
            "seed_start": start_seed,
            "metrics": {role: {k: out[k] for k in METRICS} for role, out in runs.items()},
            "candidate_minus_baseline": {
                "cow_health": round(float(np.mean(health_delta)), 4),
                "cow_health_ci95": _paired_ci(health_delta, start_seed + 7),
                "surviving_cows": round(float(np.mean(survivors_delta)), 4),
                "surviving_cows_ci95": _paired_ci(survivors_delta, start_seed + 11),
                "pasture_arrivals": round(float(np.mean(arrival_delta)), 4),
                "pasture_arrivals_ci95": _paired_ci(arrival_delta, start_seed + 13),
            },
            "per_seed": seed_rows,
        }
    guard = details["nomad_guard"]["metrics"]
    intercept = details["nomad_intercept"]["metrics"]
    migration = details["nomad_herd"]["metrics"]
    old_defense = details["nomad_defend"]["metrics"]
    migration_call_gain = (migration["candidate"]["mean_cows_in_pasture"] -
                           migration["candidate_nocalls"]["mean_cows_in_pasture"])
    checks = {
        "sufficient_validation_worlds": episodes >= 24,
        "guard_more_cattle_survive": (guard["candidate"]["mean_cows_alive"] >
                                      guard["baseline"]["mean_cows_alive"]),
        "intercept_health_improves": (
            details["nomad_intercept"]["candidate_minus_baseline"]["cow_health_ci95"][0]
            > 0),
        "attack_causes_predator_elimination": (
            intercept["candidate"]["ravager_survival_rate"] + 0.1 <=
            intercept["candidate_noattack"]["ravager_survival_rate"]),
        "attacks_improve_defender_survival": (
            guard["candidate"]["nomad_survival_rate"] >=
            guard["candidate_noattack"]["nomad_survival_rate"] + 0.1),
        "attacks_measurably_help_cattle": (
            max(guard["candidate"]["mean_cow_health"] -
                guard["candidate_noattack"]["mean_cow_health"],
                intercept["candidate"]["mean_cow_health"] -
                intercept["candidate_noattack"]["mean_cow_health"]) >= 2.0),
        "existing_defense_not_worse": (
            old_defense["candidate"]["mean_cows_alive"] >=
            old_defense["baseline"]["mean_cows_alive"] and
            old_defense["candidate"]["mean_cow_health"] >=
            old_defense["baseline"]["mean_cow_health"] - 1.0),
        "migration_preserved": (
            migration["candidate"]["mean_cows_in_pasture"] >=
            migration["baseline"]["mean_cows_in_pasture"] - 0.1),
        "calls_still_help_migration": migration_call_gain >= 0.1,
    }
    return {
        "version": 1, "episodes_per_scenario": episodes, "seed_start": seed,
        "max_steps": max_steps, "cow_controller": "frozen_learned_greedy",
        "checkpoint_hashes": {**{k: sha256(v.read_bytes()).hexdigest()
                                  for k, v in paths.items()},
                              "cow": sha256(cow_path.read_bytes()).hexdigest()},
        "scenarios": details, "candidate_call_lift": round(migration_call_gain, 4),
        "promotion_checks": checks,
        "candidate_eligible_for_manual_review": all(checks.values()),
        "notes": ("No checkpoint promoted or overwritten. Short defender encounters "
                  "are controlled tasks, not proof of long-run herd protection. "
                  "Confirmation on fresh blind seeds is required before promotion."),
    }
