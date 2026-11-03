"""Conservative M16d farmer comparison; no checkpoint writes or promotion."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from .farmer_ppo import evaluate_farmer, load_farmer

SCENARIOS = ("farmstead", "farmstead_threat")
FIELDS = ("deposited", "harvested", "fertilized", "waste_collected",
          "alive", "final_health", "final_home_food", "sheltered_steps",
          "crop_eaten_by_cows", "net_store_change")


def _paired_interval(deltas: list[float], seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.array(deltas, dtype=np.float64)
    picks = rng.integers(len(values), size=(500, len(values)))
    return [round(float(x), 4) for x in np.percentile(values[picks].mean(axis=1),
                                                      (2.5, 97.5))]


def compare_farmers(baseline: str | Path, candidate: str | Path, *,
                    episodes: int = 48, seed: int = 1_600_000,
                    max_steps: int = 192) -> dict:
    """Compare two frozen farmer policies on identical long-horizon worlds."""
    if episodes < 1 or max_steps < 1 or seed < 0:
        raise ValueError("episodes and max_steps must be positive; seed nonnegative")
    sources = {"baseline": Path(baseline), "candidate": Path(candidate)}
    for name, path in sources.items():
        if not path.is_file():
            raise ValueError(f"Missing {name} farmer checkpoint: {path}")
        load_farmer(path)
    cases = {}
    for offset, scenario in enumerate(SCENARIOS):
        first_seed = seed + offset * 100_000
        runs = {
            name: evaluate_farmer(path, episodes=episodes, seed=first_seed,
                                  scenario=scenario, max_steps=max_steps, mode="policy")
            for name, path in sources.items()
        }
        ablation = ("policy_no_refuge" if scenario == "farmstead_threat"
                    else "policy_no_fertilize")
        runs["candidate_ablated"] = evaluate_farmer(
            candidate, episodes=episodes, seed=first_seed,
            scenario=scenario, max_steps=max_steps, mode=ablation)
        differences = {}
        for metric in FIELDS:
            vals = [float(c[metric]) - float(b[metric]) for c, b in
                    zip(runs["candidate"]["results"], runs["baseline"]["results"],
                        strict=True)]
            differences[metric] = {"mean": round(float(np.mean(vals)), 4),
                                   "paired_ci95": _paired_interval(vals, first_seed + len(metric))}
        cases[scenario] = {
            "seed_start": first_seed,
            "ablated_action": ("enter_refuge" if scenario == "farmstead_threat"
                               else "fertilize"),
            "metrics": {name: {key: value for key, value in run.items()
                               if key != "results"} for name, run in runs.items()},
            "differences": differences,
            "per_seed": [{"seed": first_seed + i,
                          **{name: {field: data["results"][i][field] for field in FIELDS}
                             for name, data in runs.items()}}
                         for i in range(episodes)],
        }
    safe = cases["farmstead"]["metrics"]
    threat = cases["farmstead_threat"]["metrics"]
    checks = {
        "enough_heldout_worlds": episodes >= 48,
        "food_storage_improves": safe["candidate"]["mean_deposited"] >=
                                  safe["baseline"]["mean_deposited"] + 0.5,
        "manure_use_not_worse": safe["candidate"]["mean_fertilized"] >=
                                safe["baseline"]["mean_fertilized"] - 0.10,
        "safe_survival_retained": safe["candidate"]["survival_rate"] >=
                                  safe["baseline"]["survival_rate"],
        "threat_survival_retained": threat["candidate"]["survival_rate"] >=
                                    threat["baseline"]["survival_rate"],
        "threat_health_not_worse": threat["candidate"]["mean_final_health"] >=
                                   threat["baseline"]["mean_final_health"] - 1.0,
    }
    return {
        "version": 1, "episodes_per_scenario": episodes, "max_steps": max_steps,
        "seed_start": seed,
        "checkpoint_sha256": {name: sha256(path.read_bytes()).hexdigest()
                              for name, path in sources.items()},
        "cases": cases,
        "promotion_checks": checks,
        "eligible_for_manual_review": all(checks.values()),
        "notes": ("No checkpoints are modified or promoted. No-fertilizer and "
                  "no-refuge ablations disable one action while keeping seeds "
                  "and policies fixed. A larger blind evaluation is required."),
    }
