"""M16c: frozen-pair convoy and defense evaluation with explicit counterfactuals.

Scores and promotion checks depend on world outcomes, not training loss or
chosen combat actions. This module NEVER writes to a model checkpoint.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .nomad_ppo import evaluate_nomad

SCENARIOS = ("nomad_convoy", "nomad_guard", "nomad_intercept", "nomad_herd")
METRICS = ("mean_cows_alive", "mean_cow_health", "mean_cows_in_pasture",
           "mean_nomad_health", "nomad_survival_rate", "ravager_survival_rate",
           "mean_calls", "mean_nomad_attacks")


def _paired_interval(values: list[float], seed: int) -> list[float]:
    samples = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    picks = generator.integers(len(samples), size=(600, len(samples)))
    return [round(float(v), 4) for v in np.percentile(samples[picks].mean(1), [2.5, 97.5])]


def compare_protectors(baseline_cow: str | Path, baseline_nomad: str | Path,
                       candidate_cow: str | Path, candidate_nomad: str | Path,
                       *, episodes: int = 48, seed: int = 1_160_000,
                       max_steps: int = 80) -> dict:
    """Compare intact learned pairs on held-out convoy, guard and migration worlds.

    Each world is repeated with matching initial seed and clock duration. No-call
    and no-attack ablations are counterfactuals, not additional policy training.
    A gate is informative only for >=48 episodes; passing it is permission for
    manual review, *never* an automatic promotion.
    """
    if episodes < 1 or max_steps < 1 or seed < 0:
        raise ValueError("episodes/max_steps must be positive and seed nonnegative")
    paths = {"baseline_cow": Path(baseline_cow),
             "baseline_nomad": Path(baseline_nomad),
             "candidate_cow": Path(candidate_cow),
             "candidate_nomad": Path(candidate_nomad)}
    for role, path in paths.items():
        if not path.is_file():
            raise ValueError(f"Missing {role} checkpoint: {path}")
    torch.set_num_threads(1)
    cases = {}
    for index, scenario in enumerate(SCENARIOS):
        start = seed + 100_000 * index

        def evaluate(nomad: str, cow: str, mode: str,
                     scenario: str = scenario, start: int = start):
            return evaluate_nomad(paths[nomad], cow_checkpoint=paths[cow],
                                  scenario=scenario, mode=mode, episodes=episodes,
                                  max_steps=max_steps, seed=start)

        runs = {
            "baseline": evaluate("baseline_nomad", "baseline_cow", "policy_greedy"),
            "candidate": evaluate("candidate_nomad", "candidate_cow", "policy_greedy"),
        }
        if scenario in ("nomad_convoy", "nomad_herd"):
            runs["candidate_no_calls"] = evaluate(
                "candidate_nomad", "candidate_cow", "policy_greedy_nocalls")
            runs["baseline_no_calls"] = evaluate(
                "baseline_nomad", "baseline_cow", "policy_greedy_nocalls")
        if scenario in ("nomad_convoy", "nomad_guard", "nomad_intercept"):
            runs["candidate_no_attack"] = evaluate(
                "candidate_nomad", "candidate_cow", "policy_greedy_noattack")
        differences = {}
        for key in ("cows_alive", "mean_cow_health", "cows_in_pasture",
                    "nomad_health", "nomad_alive"):
            delta = [float(c[key]) - float(b[key])
                     for c, b in zip(runs["candidate"]["results"],
                                     runs["baseline"]["results"], strict=True)]
            differences[key] = {
                "mean": round(float(np.mean(delta)), 4),
                "paired_ci95": _paired_interval(delta, start + len(key)),
            }
        cases[scenario] = {
            "seed_start": start,
            "policy_metrics": {name: {key: result[key] for key in METRICS}
                               for name, result in runs.items()},
            "difference": differences,
            "per_seed": [{
                "seed": start + i,
                **{name: {k: out["results"][i][k] for k in
                           ("cows_alive", "mean_cow_health", "cows_in_pasture",
                            "nomad_attacks", "nomad_alive", "ravager_alive")}
                   for name, out in runs.items()}}
                for i in range(episodes)],
        }
    migration = cases["nomad_herd"]["policy_metrics"]
    convoy = cases["nomad_convoy"]["policy_metrics"]
    guard = cases["nomad_guard"]["policy_metrics"]
    intercept = cases["nomad_intercept"]["policy_metrics"]
    migration_lift = (migration["candidate"]["mean_cows_in_pasture"] -
                      migration["candidate_no_calls"]["mean_cows_in_pasture"])
    convoy_lift = (convoy["candidate"]["mean_cows_in_pasture"] -
                   convoy["candidate_no_calls"]["mean_cows_in_pasture"])
    attack_cow_lift = max(
        convoy["candidate"]["mean_cow_health"] -
        convoy["candidate_no_attack"]["mean_cow_health"],
        guard["candidate"]["mean_cow_health"] -
        guard["candidate_no_attack"]["mean_cow_health"],
        intercept["candidate"]["mean_cow_health"] -
        intercept["candidate_no_attack"]["mean_cow_health"],
    )
    checks = {
        "enough_heldout_worlds": episodes >= 48,
        "migration_retained": (migration["candidate"]["mean_cows_in_pasture"] >=
                                migration["baseline"]["mean_cows_in_pasture"] - 0.10),
        "calls_help_migration": migration_lift >= 0.10,
        "convoy_arrivals_retained": (convoy["candidate"]["mean_cows_in_pasture"] >=
                                     convoy["baseline"]["mean_cows_in_pasture"] - 0.10),
        "convoy_cow_survival_not_worse": (convoy["candidate"]["mean_cows_alive"] >=
                                          convoy["baseline"]["mean_cows_alive"]),
        "convoy_health_improves": (convoy["candidate"]["mean_cow_health"] >=
                                   convoy["baseline"]["mean_cow_health"] + 2.0),
        "guard_and_intercept_not_worse": all(
            group["candidate"]["mean_cows_alive"] >=
            group["baseline"]["mean_cows_alive"] for group in (guard, intercept)),
        "attacks_causally_help_cattle": attack_cow_lift >= 2.0,
        "nomad_survival_retained": convoy["candidate"]["nomad_survival_rate"] >=
                                    convoy["baseline"]["nomad_survival_rate"] - 0.05,
    }
    return {
        "version": 1, "episodes_per_scenario": episodes,
        "seed_start": seed, "max_steps": max_steps,
        "checkpoint_sha256": {key: sha256(path.read_bytes()).hexdigest()
                              for key, path in paths.items()},
        "cow_controller": "frozen learned greedy", "nomad_controller": "frozen learned greedy",
        "predator_controller": "three seeded scripted archetypes in convoy",
        "cases": cases,
        "candidate_call_lift_migration": round(migration_lift, 4),
        "candidate_call_lift_convoy": round(convoy_lift, 4),
        "candidate_attack_cow_health_lift": round(attack_cow_lift, 4),
        "promotion_checks": checks,
        "eligible_for_manual_review": all(checks.values()),
        "notes": ("No models are modified or promoted. Scripted predator variants "
                  "are test opponents, not learned adversaries. Passing screening "
                  "checks does not prove generalization; confirm on blind seeds."),
    }
